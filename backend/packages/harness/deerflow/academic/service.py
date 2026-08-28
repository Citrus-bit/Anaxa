from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from deerflow.config.paths import VIRTUAL_PATH_PREFIX, get_paths
from deerflow.runtime.utils import now_iso

from .adapters import AcademicSourceAdapter, build_default_adapters
from .formatters import (
    DEFAULT_REFERENCE_STYLE,
    format_bibtex_entry,
    format_reference,
    normalize_reference_style,
    reference_style_label,
)
from .quality import (
    canonical_reason,
    hydrate_quality_metadata,
    preprint_ratio,
    provider_breakdown,
    venue_breakdown,
    version_priority_score,
)
from .queries import build_query_expansions
from .ranking import score_papers, select_core_papers
from .repository import AcademicRepository
from .types import (
    AcademicGraph,
    EvidenceCardRecord,
    IngestResult,
    OutlineNodeRecord,
    PaperEdgeRecord,
    PaperRecord,
    ReferenceEntry,
    ReportExportRecord,
    ResearchProject,
    SearchQueryRecord,
    SynthesisResult,
)
from .utils import (
    detect_domain,
    merge_unique,
    normalize_doi,
    normalize_title_key,
    slugify,
    summarize_abstract,
    today_stamp,
    topic_terms,
)

logger = logging.getLogger(__name__)

_ACADEMIC_FETCH_CONCURRENCY = 6
_REVIEW_DELIVERABLE_TYPES = {
    "experiment_design",
    "experiment_report",
    "literature_review",
    "manuscript",
    "paper",
    "paper_draft",
    "review",
    "review_article",
    "survey",
    "systematic_review",
}
_MANUSCRIPT_EVIDENCE_TYPES = [
    "dataset",
    "benchmark",
    "metric",
    "baseline",
    "ablation",
    "external validation",
]
_REVIEW_DELIVERABLE_HINTS = {
    "experiment design",
    "experiment report",
    "literature review",
    "manuscript",
    "paper draft",
    "review article",
    "survey",
    "systematic review",
    "论文",
    "综述",
    "成稿",
}
_QUANTITATIVE_EVIDENCE_TERMS = {
    "ablation",
    "accuracy",
    "auc",
    "baseline",
    "benchmark",
    "case study",
    "confidence interval",
    "dataset",
    "empirical",
    "experiment",
    "f1",
    "metric",
    "p-value",
    "result",
    "statistical",
}


class AcademicResearchService:
    def __init__(
        self,
        repository: AcademicRepository,
        *,
        adapters: list[AcademicSourceAdapter] | None = None,
    ) -> None:
        self._repository = repository
        self._adapters = adapters or []

    async def create_project(
        self,
        *,
        thread_id: str,
        topic: str,
        scope: str | None = None,
        metadata: dict[str, Any] | None = None,
        force_new: bool = False,
    ) -> ResearchProject:
        topic_key = normalize_title_key(topic)
        domain = detect_domain(topic, scope)
        if not force_new:
            existing = await self._repository.find_project_by_thread_topic(thread_id, topic_key)
            if existing is not None:
                return existing

        if force_new:
            topic_key = f"{topic_key}:{uuid.uuid4().hex[:8]}"

        project = await self._repository.create_project(
            project_id=str(uuid.uuid4()),
            thread_id=thread_id,
            topic=topic.strip(),
            topic_key=topic_key,
            scope=scope.strip() if scope else None,
            status="created",
            domain=domain,
            metadata=metadata or {},
        )
        return project

    async def ingest_project(
        self,
        project_id: str,
        *,
        max_candidates: int = 120,
        core_paper_limit: int = 24,
        use_local_uploads: bool = True,
        source_profile: str | None = None,
        quality_mode: str | None = None,
        preprint_policy: str | None = None,
        deliverable_type: str | None = None,
        min_reference_count: int | None = None,
        target_reference_count: int | None = None,
        required_topics: list[str] | None = None,
        required_evidence_types: list[str] | None = None,
    ) -> IngestResult:
        project = await self._require_project(project_id)
        project = await self._apply_coverage_metadata(
            project,
            deliverable_type=deliverable_type,
            min_reference_count=min_reference_count,
            target_reference_count=target_reference_count,
            required_topics=required_topics,
            required_evidence_types=required_evidence_types,
        )
        queries = build_query_expansions(project)
        all_queries = list(queries)
        await self._repository.replace_search_queries(project.project_id, all_queries)

        resolved_source_profile = source_profile or ("cs_ai_premium" if project.domain == "cs_ai" else "default")
        resolved_quality_mode = quality_mode or ("strict" if project.domain == "cs_ai" else "balanced")
        resolved_preprint_policy = preprint_policy or "prefer_final"
        coverage_targets = self._coverage_targets(project)
        if coverage_targets["review_quality_profile"]:
            max_candidates = max(max_candidates, coverage_targets["target_reference_count"] * 3)
            core_paper_limit = max(core_paper_limit, coverage_targets["min_core_paper_count"])
        ingest_request_signature = self._ingest_request_signature(
            project,
            core_paper_limit=core_paper_limit,
            source_profile=resolved_source_profile,
            quality_mode=resolved_quality_mode,
            preprint_policy=resolved_preprint_policy,
        )

        configured_adapters = self._adapters or list(
            build_default_adapters(
                project.domain,
                topic=project.topic,
                scope=project.scope,
                source_profile=resolved_source_profile,
            )
        )
        terms = topic_terms(project.topic, project.scope)

        local_papers = await self._load_local_upload_papers(project) if use_local_uploads else []
        provider_limit = max(4, min(25, max_candidates // max(len(queries), 1)))
        raw_papers: list[PaperRecord] = list(local_papers)
        if len(raw_papers) < max_candidates:
            fetched_papers = await self._fetch_candidates_for_queries(
                project,
                configured_adapters,
                queries,
                max_candidates=max_candidates - len(raw_papers),
                provider_limit=provider_limit,
                failure_context="academic search",
            )
            raw_papers.extend(fetched_papers)

        normalized_papers = [hydrate_quality_metadata(paper.model_copy(deep=True)) for paper in raw_papers]
        merged = self._dedupe_candidates(
            normalized_papers,
            quality_mode=resolved_quality_mode,
            preprint_policy=resolved_preprint_policy,
        )
        scored = score_papers(list(merged.values()), terms=terms, quality_mode=resolved_quality_mode)
        stored = await self._repository.upsert_papers(project.project_id, scored)
        selected = select_core_papers(stored, limit=max(20, min(core_paper_limit, 80)))
        references = self._build_references(stored, style=self._resolve_reference_style(project))
        coverage_audit = self._reference_coverage_audit(project, stored, references)
        repair_query_count = 0
        repair_candidate_count = 0
        if coverage_targets["review_quality_profile"] and coverage_audit["status"] != "pass":
            repair_queries = self._coverage_repair_query_records(project, coverage_audit, existing_queries=all_queries)
            if repair_queries:
                all_queries.extend(repair_queries)
                await self._repository.replace_search_queries(project.project_id, all_queries)
                repair_raw_papers = await self._fetch_candidates_for_queries(
                    project,
                    configured_adapters,
                    repair_queries,
                    max_candidates=max(coverage_targets["target_reference_count"], len(repair_queries) * provider_limit),
                    provider_limit=provider_limit,
                )
                if repair_raw_papers:
                    raw_papers.extend(repair_raw_papers)
                    normalized_papers = [hydrate_quality_metadata(paper.model_copy(deep=True)) for paper in raw_papers]
                    merged = self._dedupe_candidates(
                        normalized_papers,
                        quality_mode=resolved_quality_mode,
                        preprint_policy=resolved_preprint_policy,
                    )
                    scored = score_papers(list(merged.values()), terms=terms, quality_mode=resolved_quality_mode)
                    stored = await self._repository.upsert_papers(project.project_id, scored)
                    selected = select_core_papers(stored, limit=max(20, min(core_paper_limit, 80)))
                    references = self._build_references(stored, style=self._resolve_reference_style(project))
                    coverage_audit = self._reference_coverage_audit(project, stored, references)
                repair_query_count = len(repair_queries)
                repair_candidate_count = len(repair_raw_papers)
        coverage_audit = {
            **coverage_audit,
            "auto_repair_attempted": repair_query_count > 0,
            "auto_repair_query_count": repair_query_count,
            "auto_repair_candidate_count": repair_candidate_count,
        }

        await self._repository.update_project_status(
            project.project_id,
            status="ingested",
            metadata={
                "raw_candidate_count": len(raw_papers),
                "paper_count": len(stored),
                "query_count": len(all_queries),
                "core_paper_limit": max(20, min(core_paper_limit, 80)),
                "source_profile": resolved_source_profile,
                "quality_mode": resolved_quality_mode,
                "preprint_policy": resolved_preprint_policy,
                "ingest_request_signature": ingest_request_signature,
                "provider_breakdown": provider_breakdown(stored),
                "venue_breakdown": venue_breakdown(stored),
                "preprint_ratio": preprint_ratio(stored),
                "canonical_reference_count": len([entry for entry in references if entry.included_in_final]),
                "reference_coverage_audit": coverage_audit,
                "last_ingested_at": now_iso(),
            },
            domain=project.domain,
        )
        updated_project = await self._require_project(project.project_id)
        return IngestResult(
            project=updated_project,
            queries=all_queries,
            raw_candidate_count=len(raw_papers),
            paper_count=len(stored),
            selected_papers=selected,
            provider_breakdown=provider_breakdown(stored),
            venue_breakdown=venue_breakdown(stored),
            preprint_ratio=preprint_ratio(stored),
            canonical_reference_count=len([entry for entry in references if entry.included_in_final]),
        )

    async def _fetch_candidates_for_queries(
        self,
        project: ResearchProject,
        adapters: list[AcademicSourceAdapter],
        queries: list[SearchQueryRecord],
        *,
        max_candidates: int,
        provider_limit: int,
        failure_context: str = "coverage repair",
    ) -> list[PaperRecord]:
        if max_candidates <= 0 or not adapters or not queries:
            return []

        jobs: list[tuple[int, AcademicSourceAdapter, SearchQueryRecord]] = []
        for adapter in adapters:
            query_window = self._query_window_for_adapter(adapter.name)
            adapter_queries = queries[:query_window]
            if adapter.name == "arxiv":
                adapter_queries = adapter_queries[:1]
            for query in adapter_queries:
                jobs.append((len(jobs), adapter, query))

        semaphore = asyncio.Semaphore(_ACADEMIC_FETCH_CONCURRENCY)

        async def fetch(
            job_index: int,
            adapter: AcademicSourceAdapter,
            query: SearchQueryRecord,
        ) -> tuple[int, list[PaperRecord]]:
            async with semaphore:
                try:
                    results = await adapter.search(
                        query.query_text,
                        project_id=project.project_id,
                        limit=min(provider_limit, max_candidates),
                    )
                except Exception:
                    logger.warning("Academic adapter failed during %s: %s", failure_context, adapter.name, exc_info=True)
                    return job_index, []
                return job_index, results

        fetched = await asyncio.gather(*(fetch(job_index, adapter, query) for job_index, adapter, query in jobs))
        raw_papers: list[PaperRecord] = []
        for _, results in sorted(fetched, key=lambda item: item[0]):
            if len(raw_papers) >= max_candidates:
                break
            remaining = max_candidates - len(raw_papers)
            raw_papers.extend(results[:remaining])
        return raw_papers

    async def synthesize_project(
        self,
        project_id: str,
        *,
        output_dir: Path | None = None,
        include_graph: bool = False,
        reference_style: str | None = None,
    ) -> SynthesisResult:
        project = await self._require_project(project_id)
        papers = await self._repository.list_project_papers(project.project_id)
        if not papers:
            raise ValueError(f"Project {project_id} has no papers. Run ingest first.")

        resolved_reference_style = self._resolve_reference_style(project, reference_style)
        core_papers = self._core_papers_for_synthesis(project, papers)
        outline = self._build_outline(project, core_papers)
        evidence_cards = self._build_evidence_cards(project, core_papers, outline)
        edges = self._build_graph_edges(project, core_papers, evidence_cards, outline)
        references = self._build_references(papers, style=resolved_reference_style)
        graph = await self._persist_graph_bundle(project, outline, evidence_cards, edges)

        export_paths: list[Path] = []
        if output_dir is not None:
            export_paths = await self._write_exports(
                project=project,
                output_dir=output_dir,
                all_papers=papers,
                core_papers=core_papers,
                outline=outline,
                evidence_cards=evidence_cards,
                references=references,
                reference_style=resolved_reference_style,
                graph=graph if include_graph else None,
            )
            virtual_exports = [self._to_virtual_output(project.thread_id, path) for path in export_paths]
            await self._repository.replace_report_exports(
                project.project_id,
                [
                    ReportExportRecord(
                        export_id=str(uuid.uuid4()),
                        project_id=project.project_id,
                        export_kind=path.suffix.lstrip(".") or path.name,
                        path=virtual_path,
                        summary={
                            "project_id": project.project_id,
                            "reference_count": len([entry for entry in references if entry.included_in_final]),
                            "reference_style": resolved_reference_style,
                            "evidence_count": len(evidence_cards),
                            "reference_mix": provider_breakdown(papers),
                            "preprint_ratio": preprint_ratio(papers),
                            "top_venues": venue_breakdown(papers),
                        },
                        created_at=now_iso(),
                    )
                    for path, virtual_path in zip(export_paths, virtual_exports, strict=True)
                ],
            )
        await self._repository.update_project_status(
            project.project_id,
            status="synthesized",
            metadata={
                "outline_count": len(outline),
                "evidence_count": len(evidence_cards),
                "reference_count": len([entry for entry in references if entry.included_in_final]),
                "reference_style": resolved_reference_style,
                "last_synthesized_at": now_iso(),
            },
        )
        updated_project = await self._require_project(project.project_id)
        return SynthesisResult(
            project=updated_project,
            outline=outline,
            evidence_cards=evidence_cards,
            references=references,
            graph=graph,
            export_files=[str(path) for path in export_paths],
        )

    async def run_research(
        self,
        *,
        thread_id: str,
        topic: str,
        scope: str | None = None,
        output_dir: Path | None = None,
        include_graph: bool = False,
        max_candidates: int = 120,
        core_paper_limit: int = 24,
        source_profile: str | None = None,
        quality_mode: str | None = None,
        preprint_policy: str | None = None,
        reference_style: str | None = None,
        deliverable_type: str | None = None,
        min_reference_count: int | None = None,
        target_reference_count: int | None = None,
        required_topics: list[str] | None = None,
        required_evidence_types: list[str] | None = None,
    ) -> SynthesisResult:
        normalized_reference_style = normalize_reference_style(reference_style)
        project = await self.create_project(
            thread_id=thread_id,
            topic=topic,
            scope=scope,
            metadata={
                "reference_style": normalized_reference_style,
                **self._coverage_metadata(
                    deliverable_type=deliverable_type,
                    min_reference_count=min_reference_count,
                    target_reference_count=target_reference_count,
                    required_topics=required_topics,
                    required_evidence_types=required_evidence_types,
                ),
            },
        )
        project = await self._apply_coverage_metadata(
            project,
            deliverable_type=deliverable_type,
            min_reference_count=min_reference_count,
            target_reference_count=target_reference_count,
            required_topics=required_topics,
            required_evidence_types=required_evidence_types,
        )
        await self._repository.update_project_status(
            project.project_id,
            status=project.status,
            metadata={"reference_style": normalized_reference_style},
            domain=project.domain,
        )
        project = await self._require_project(project.project_id)
        ingest_request_signature = self._ingest_request_signature(
            project,
            core_paper_limit=core_paper_limit,
            source_profile=source_profile,
            quality_mode=quality_mode,
            preprint_policy=preprint_policy,
        )
        if not await self._can_reuse_ingested_project(project, ingest_request_signature):
            await self.ingest_project(
                project.project_id,
                max_candidates=max_candidates,
                core_paper_limit=core_paper_limit,
                source_profile=source_profile,
                quality_mode=quality_mode,
                preprint_policy=preprint_policy,
                deliverable_type=deliverable_type,
                min_reference_count=min_reference_count,
                target_reference_count=target_reference_count,
                required_topics=required_topics,
                required_evidence_types=required_evidence_types,
            )
        return await self.synthesize_project(
            project.project_id,
            output_dir=output_dir,
            include_graph=include_graph,
            reference_style=normalized_reference_style,
        )

    async def get_project_summary(self, project_id: str) -> dict[str, Any]:
        project = await self._require_project(project_id)
        papers = await self._repository.list_project_papers(project.project_id)
        outline = await self._repository.list_outline_nodes(project.project_id)
        evidence = await self._repository.list_evidence_cards(project.project_id)
        exports = await self._repository.list_report_exports(project.project_id)
        reference_style = self._resolve_reference_style(project)
        references = self._build_references(papers, style=reference_style)
        return {
            "project": project.model_dump(),
            "paper_count": len(papers),
            "reference_count": len([entry for entry in references if entry.included_in_final]),
            "evidence_count": len(evidence),
            "outline_count": len(outline),
            "export_files": [item.path for item in exports],
            "provider_breakdown": provider_breakdown(papers),
            "venue_breakdown": venue_breakdown(papers),
            "preprint_ratio": preprint_ratio(papers),
            "canonical_reference_count": len([entry for entry in references if entry.included_in_final]),
            "reference_style": reference_style,
            "reference_style_label": reference_style_label(reference_style),
            "reference_coverage_audit": (project.metadata or {}).get("reference_coverage_audit"),
        }

    async def get_references(self, project_id: str, *, style: str | None = None) -> list[ReferenceEntry]:
        project = await self._require_project(project_id)
        papers = await self._repository.list_project_papers(project.project_id)
        return self._build_references(papers, style=self._resolve_reference_style(project, style))

    async def get_graph(self, project_id: str) -> AcademicGraph:
        await self._require_project(project_id)
        return await self._repository.get_graph(project_id)

    def thread_outputs_dir(self, thread_id: str) -> Path:
        return get_paths().sandbox_outputs_dir(thread_id)

    async def _persist_graph_bundle(
        self,
        project: ResearchProject,
        outline: list[OutlineNodeRecord],
        evidence_cards: list[EvidenceCardRecord],
        edges: list[PaperEdgeRecord],
    ) -> AcademicGraph:
        await self._repository.replace_outline_nodes(project.project_id, outline)
        await self._repository.replace_evidence_cards(project.project_id, evidence_cards)
        await self._repository.replace_paper_edges(project.project_id, edges)
        return await self._repository.get_graph(project.project_id)

    def _dedupe_candidates(
        self,
        papers: list[PaperRecord],
        *,
        quality_mode: str,
        preprint_policy: str,
    ) -> dict[str, PaperRecord]:
        deduped: dict[str, PaperRecord] = {}
        for paper in papers:
            existing = deduped.get(paper.canonical_id)
            if existing is None:
                deduped[paper.canonical_id] = paper
                continue

            preferred, alternate = self._preferred_and_alternate_version(
                existing,
                paper,
                quality_mode=quality_mode,
                preprint_policy=preprint_policy,
            )
            merged = preferred.model_copy(deep=True)
            merged.provider = preferred.provider
            merged.provider_id = preferred.provider_id
            merged.source_url = preferred.source_url or alternate.source_url
            merged.oa_url = preferred.oa_url or alternate.oa_url
            merged.canonical_source = preferred.canonical_source
            merged.quality_signals = {**alternate.quality_signals, **preferred.quality_signals}
            merged.abstract = merged.abstract or paper.abstract
            merged.venue = merged.venue or paper.venue
            merged.year = merged.year or paper.year
            merged.doi = merged.doi or paper.doi
            merged.pmid = merged.pmid or paper.pmid
            merged.pmcid = merged.pmcid or paper.pmcid
            merged.arxiv_id = merged.arxiv_id or paper.arxiv_id
            merged.cited_by_count = max(existing.cited_by_count or 0, paper.cited_by_count or 0) or None
            merged.authors = existing.authors or paper.authors
            merged.keywords = merge_unique(existing.keywords + paper.keywords)
            merged.methods = merge_unique(existing.methods + paper.methods)
            merged.populations = merge_unique(existing.populations + paper.populations)
            merged.conflict_flags = merge_unique(existing.conflict_flags + paper.conflict_flags)
            merged.raw_source = {**alternate.raw_source, **preferred.raw_source}
            merged.metadata_only = existing.metadata_only and paper.metadata_only
            if paper.provider == "local-upload":
                merged.provider = paper.provider
                merged.canonical_source = paper.canonical_source
            merged = hydrate_quality_metadata(merged)
            deduped[paper.canonical_id] = merged
        return deduped

    @staticmethod
    def _preferred_and_alternate_version(
        existing: PaperRecord,
        incoming: PaperRecord,
        *,
        quality_mode: str,
        preprint_policy: str,
    ) -> tuple[PaperRecord, PaperRecord]:
        existing_score = version_priority_score(existing, quality_mode=quality_mode, preprint_policy=preprint_policy)
        incoming_score = version_priority_score(incoming, quality_mode=quality_mode, preprint_policy=preprint_policy)
        if incoming_score > existing_score:
            return incoming, existing
        return existing, incoming

    async def _load_local_upload_papers(self, project: ResearchProject) -> list[PaperRecord]:
        uploads_dir = get_paths().sandbox_uploads_dir(project.thread_id)
        if not uploads_dir.exists():
            return []

        papers: list[PaperRecord] = []
        markdown_files = sorted([path for path in uploads_dir.iterdir() if path.is_file() and path.suffix.lower() == ".md"])
        for path in markdown_files:
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue
            title = self._extract_local_title(path, content)
            doi = self._extract_doi(content)
            year = self._extract_year(content)
            summary = summarize_abstract(content, limit=800)
            papers.append(
                PaperRecord(
                    paper_id=f"{project.project_id}:local-upload:{slugify(path.stem, fallback='upload')}",
                    project_id=project.project_id,
                    canonical_id=f"local:{normalize_doi(doi) or normalize_title_key(title)}",
                    title=title,
                    authors=[],
                    year=year,
                    venue="User uploaded document",
                    abstract=summary,
                    doi=normalize_doi(doi),
                    provider="local-upload",
                    provider_id=path.name,
                    source_url=f"/api/threads/{project.thread_id}/artifacts/mnt/user-data/uploads/{path.name}",
                    oa_url=None,
                    metadata_only=False,
                    keywords=[],
                    methods=[],
                    populations=[],
                    conflict_flags=[],
                    raw_source={
                        "path": str(path),
                        "uploaded_at": path.stat().st_mtime,
                    },
                    relevance_score=1.0,
                    recency_score=0.6,
                    completeness_score=0.6,
                    rank_score=0.9,
                    created_at=now_iso(),
                    updated_at=now_iso(),
                )
            )
        return papers

    @staticmethod
    def _query_window_for_adapter(adapter_name: str) -> int:
        if adapter_name == "arxiv":
            return 1
        if adapter_name in {"crossref", "pubmed", "dblp", "openreview", "acl-anthology", "semantic-scholar"}:
            return 3
        return 5

    def _core_papers_for_synthesis(self, project: ResearchProject, papers: list[PaperRecord]) -> list[PaperRecord]:
        metadata = project.metadata or {}
        limit = metadata.get("core_paper_limit") or 24
        upper_bound = 80 if self._coverage_targets(project)["review_quality_profile"] else 40
        bounded_limit = max(20, min(int(limit), upper_bound))
        return select_core_papers(papers, limit=bounded_limit)

    async def _apply_coverage_metadata(
        self,
        project: ResearchProject,
        *,
        deliverable_type: str | None,
        min_reference_count: int | None,
        target_reference_count: int | None,
        required_topics: list[str] | None,
        required_evidence_types: list[str] | None,
    ) -> ResearchProject:
        metadata = self._coverage_metadata(
            deliverable_type=deliverable_type,
            min_reference_count=min_reference_count,
            target_reference_count=target_reference_count,
            required_topics=required_topics,
            required_evidence_types=required_evidence_types,
        )
        if not metadata:
            return project
        await self._repository.update_project_status(
            project.project_id,
            status=project.status,
            metadata=metadata,
            domain=project.domain,
        )
        return await self._require_project(project.project_id)

    def _coverage_metadata(
        self,
        *,
        deliverable_type: str | None,
        min_reference_count: int | None,
        target_reference_count: int | None,
        required_topics: list[str] | None,
        required_evidence_types: list[str] | None,
    ) -> dict[str, Any]:
        normalized_deliverable_type = self._normalize_deliverable_type(deliverable_type)
        review_quality_profile = normalized_deliverable_type in _REVIEW_DELIVERABLE_TYPES
        if min_reference_count is not None or target_reference_count is not None:
            review_quality_profile = True

        metadata: dict[str, Any] = {}
        if normalized_deliverable_type:
            metadata["deliverable_type"] = normalized_deliverable_type
        if required_topics:
            metadata["required_topics"] = self._normalize_text_list(required_topics)
        if required_evidence_types:
            metadata["required_evidence_types"] = self._normalize_text_list(required_evidence_types)
        if review_quality_profile:
            minimum = min_reference_count if min_reference_count is not None else 50
            target = target_reference_count if target_reference_count is not None else max(80, minimum)
            metadata["review_quality_profile"] = True
            metadata["min_reference_count"] = max(1, int(minimum))
            metadata["target_reference_count"] = max(int(target), metadata["min_reference_count"])
            metadata["min_core_paper_count"] = min(80, max(30, metadata["min_reference_count"] // 2))
            current_evidence = metadata.get("required_evidence_types", [])
            metadata["required_evidence_types"] = self._merge_text_lists(current_evidence, _MANUSCRIPT_EVIDENCE_TYPES)
        elif min_reference_count is not None:
            metadata["min_reference_count"] = max(1, int(min_reference_count))
        elif target_reference_count is not None:
            metadata["target_reference_count"] = max(1, int(target_reference_count))
        return metadata

    def _coverage_targets(self, project: ResearchProject) -> dict[str, Any]:
        metadata = project.metadata or {}
        deliverable_type = self._normalize_deliverable_type(metadata.get("deliverable_type"))
        review_quality_profile = bool(metadata.get("review_quality_profile")) or deliverable_type in _REVIEW_DELIVERABLE_TYPES
        min_reference_count = int(metadata.get("min_reference_count") or (50 if review_quality_profile else 0))
        target_reference_count = int(metadata.get("target_reference_count") or (80 if review_quality_profile else max(min_reference_count, 24)))
        min_core_paper_count = int(metadata.get("min_core_paper_count") or (30 if review_quality_profile else 24))
        return {
            "review_quality_profile": review_quality_profile,
            "min_reference_count": min_reference_count,
            "target_reference_count": max(target_reference_count, min_reference_count),
            "min_core_paper_count": min(80, max(1, min_core_paper_count)),
            "required_topics": self._normalize_text_list(metadata.get("required_topics")),
            "required_evidence_types": self._normalize_text_list(metadata.get("required_evidence_types")),
        }

    def _ingest_request_signature(
        self,
        project: ResearchProject,
        *,
        core_paper_limit: int,
        source_profile: str | None,
        quality_mode: str | None,
        preprint_policy: str | None,
    ) -> dict[str, Any]:
        targets = self._coverage_targets(project)
        effective_core_paper_limit = core_paper_limit
        if targets["review_quality_profile"]:
            effective_core_paper_limit = max(effective_core_paper_limit, targets["min_core_paper_count"])
        return {
            "source_profile": source_profile or ("cs_ai_premium" if project.domain == "cs_ai" else "default"),
            "quality_mode": quality_mode or ("strict" if project.domain == "cs_ai" else "balanced"),
            "preprint_policy": preprint_policy or "prefer_final",
            "core_paper_limit": max(20, min(int(effective_core_paper_limit), 80)),
            "coverage_targets": {
                "review_quality_profile": bool(targets["review_quality_profile"]),
                "min_reference_count": int(targets["min_reference_count"]),
                "target_reference_count": int(targets["target_reference_count"]),
                "min_core_paper_count": int(targets["min_core_paper_count"]),
                "required_topics": sorted(targets["required_topics"], key=str.lower),
                "required_evidence_types": sorted(targets["required_evidence_types"], key=str.lower),
            },
        }

    async def _can_reuse_ingested_project(
        self,
        project: ResearchProject,
        expected_ingest_signature: dict[str, Any] | None = None,
    ) -> bool:
        if project.status not in {"ingested", "synthesized"}:
            return False

        metadata = project.metadata or {}
        if expected_ingest_signature is not None and metadata.get("ingest_request_signature") != expected_ingest_signature:
            return False

        audit = metadata.get("reference_coverage_audit")
        if not isinstance(audit, dict) or audit.get("status") != "pass":
            return False

        papers = await self._repository.list_project_papers(project.project_id)
        if not papers:
            return False

        targets = self._coverage_targets(project)
        included_count = int(audit.get("included_reference_count") or 0)
        if included_count < targets["min_reference_count"]:
            return False
        if targets["review_quality_profile"] and int(audit.get("quantitative_evidence_count") or 0) <= 0:
            return False

        audited_topics = set(self._normalize_text_list(audit.get("required_topics")))
        audited_evidence = set(self._normalize_text_list(audit.get("required_evidence_types")))
        if not set(targets["required_topics"]).issubset(audited_topics):
            return False
        if not set(targets["required_evidence_types"]).issubset(audited_evidence):
            return False

        return True

    def _reference_coverage_audit(
        self,
        project: ResearchProject,
        papers: list[PaperRecord],
        references: list[ReferenceEntry],
    ) -> dict[str, Any]:
        targets = self._coverage_targets(project)
        included_reference_ids = {entry.paper_id for entry in references if entry.included_in_final}
        included_papers = [paper for paper in papers if paper.paper_id in included_reference_ids]
        included_count = len(included_papers)
        topic_tokens = topic_terms(project.topic, project.scope)
        required_topics = targets["required_topics"]
        required_evidence_types = targets["required_evidence_types"]

        off_topic = [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "provider": paper.provider,
                "topic_fit": round(self._paper_topic_fit(paper, topic_tokens, required_topics), 4),
            }
            for paper in included_papers
            if self._paper_topic_fit(paper, topic_tokens, required_topics) < 0.08
        ]
        quantitative_evidence = [paper for paper in included_papers if self._paper_has_any_term(paper, _QUANTITATIVE_EVIDENCE_TERMS)]
        missing_required_topics = [topic for topic in required_topics if not any(topic.lower() in self._paper_text(paper).lower() for paper in included_papers)]
        missing_evidence_types = [evidence_type for evidence_type in required_evidence_types if not any(evidence_type.lower() in self._paper_text(paper).lower() for paper in included_papers)]
        recommended_queries: list[str] = []
        if included_count < targets["min_reference_count"]:
            recommended_queries.extend(
                [
                    f"{project.topic} review benchmark",
                    f"{project.topic} systematic review",
                    f"{project.topic} empirical results",
                ]
            )
        recommended_queries.extend(f"{project.topic} {topic}" for topic in missing_required_topics)
        recommended_queries.extend(f"{project.topic} {evidence_type}" for evidence_type in missing_evidence_types)
        if not quantitative_evidence and targets["review_quality_profile"]:
            recommended_queries.extend(
                [
                    f"{project.topic} benchmark dataset",
                    f"{project.topic} case study empirical",
                    f"{project.topic} ablation metric",
                ]
            )

        status = "pass"
        findings: list[str] = []
        if included_count < targets["min_reference_count"]:
            status = "block"
            findings.append(f"Usable references below target: {included_count}/{targets['min_reference_count']}.")
        if off_topic:
            status = "revise" if status == "pass" else status
            findings.append(f"{len(off_topic)} reference(s) have weak topic fit.")
        if targets["review_quality_profile"] and not quantitative_evidence:
            status = "block"
            findings.append("No quantitative, benchmark, dataset, metric, or case-study reference was detected.")
        if missing_required_topics or missing_evidence_types:
            status = "revise" if status == "pass" else status
            findings.append("Coverage hints are missing from the final reference set.")

        return {
            "status": status,
            "review_quality_profile": targets["review_quality_profile"],
            "included_reference_count": included_count,
            "min_reference_count": targets["min_reference_count"],
            "target_reference_count": targets["target_reference_count"],
            "min_core_paper_count": targets["min_core_paper_count"],
            "off_topic_reference_count": len(off_topic),
            "off_topic_references": off_topic[:12],
            "quantitative_evidence_count": len(quantitative_evidence),
            "quantitative_evidence_titles": [paper.title for paper in quantitative_evidence[:12]],
            "required_topics": required_topics,
            "missing_required_topics": missing_required_topics,
            "required_evidence_types": required_evidence_types,
            "missing_evidence_types": missing_evidence_types,
            "recommended_queries": sorted(set(recommended_queries)),
            "findings": findings,
        }

    def _coverage_repair_query_records(
        self,
        project: ResearchProject,
        coverage_audit: dict[str, Any],
        *,
        existing_queries: list[SearchQueryRecord],
    ) -> list[SearchQueryRecord]:
        existing = {query.query_text.lower() for query in existing_queries}
        raw_queries = [str(item).strip() for item in coverage_audit.get("recommended_queries", []) if str(item).strip()]
        records: list[SearchQueryRecord] = []
        timestamp = now_iso()
        for query_text in raw_queries:
            normalized = " ".join(query_text.split())
            if not normalized or normalized.lower() in existing:
                continue
            records.append(
                SearchQueryRecord(
                    query_id=str(uuid.uuid4()),
                    project_id=project.project_id,
                    query_text=normalized,
                    rationale="Auto-repair query generated by reference coverage audit.",
                    query_type="coverage_repair",
                    source="quality-audit",
                    created_at=timestamp,
                )
            )
            existing.add(normalized.lower())
            if len(records) >= 6:
                break
        return records

    def _collection_audit(self, papers: list[PaperRecord], references: list[ReferenceEntry]) -> dict[str, Any]:
        included_papers = [paper for paper in papers if self._include_reference_in_export(paper)]
        excluded_reasons: list[dict[str, Any]] = []
        for paper in papers:
            if self._include_reference_in_export(paper):
                continue
            reasons: list[str] = []
            if not paper.title:
                reasons.append("missing-title")
            if not (paper.doi or paper.source_url or paper.oa_url):
                reasons.append("missing-resolvable-source")
            if not paper.provider:
                reasons.append("missing-provider")
            excluded_reasons.append(
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "provider": paper.provider,
                    "reasons": reasons or ["failed-reference-eligibility"],
                }
            )

        canonical_versions = [
            {
                "paper_id": paper.paper_id,
                "canonical_id": paper.canonical_id,
                "canonical_source": paper.canonical_source,
                "reason": canonical_reason(paper),
            }
            for paper in included_papers
        ]
        return {
            "provider_breakdown": provider_breakdown(papers),
            "venue_breakdown": venue_breakdown(papers),
            "preprint_ratio": preprint_ratio(included_papers),
            "canonical_reference_count": len([entry for entry in references if entry.included_in_final]),
            "canonical_versions": canonical_versions,
            "excluded_from_final_references": excluded_reasons,
        }

    @staticmethod
    def _extract_local_title(path: Path, content: str) -> str:
        for line in content.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if len(stripped) >= 6:
                return stripped
        return path.stem.replace("_", " ").replace("-", " ").strip() or "Uploaded paper"

    @staticmethod
    def _extract_doi(content: str) -> str | None:
        match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", content, flags=re.IGNORECASE)
        return match.group(0) if match else None

    @staticmethod
    def _extract_year(content: str) -> int | None:
        match = re.search(r"\b(19|20)\d{2}\b", content[:5000])
        if not match:
            return None
        return int(match.group(0))

    @classmethod
    def _normalize_deliverable_type(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text:
            return None
        if any(hint in text for hint in _REVIEW_DELIVERABLE_HINTS):
            return "literature_review" if "review" in text or "综述" in text else "manuscript"
        normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        return normalized or None

    @staticmethod
    def _normalize_text_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = re.split(r"[,;\n]+", value)
        elif isinstance(value, list | tuple | set):
            raw_items = [str(item) for item in value]
        else:
            raw_items = [str(value)]
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            text = " ".join(str(item).split()).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            cleaned.append(text)
            seen.add(key)
        return cleaned

    @classmethod
    def _merge_text_lists(cls, *values: Any) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for value in values:
            for item in cls._normalize_text_list(value):
                key = item.lower()
                if key in seen:
                    continue
                merged.append(item)
                seen.add(key)
        return merged

    @staticmethod
    def _paper_text(paper: PaperRecord) -> str:
        return " ".join(
            [
                paper.title,
                paper.abstract or "",
                paper.venue or "",
                " ".join(paper.keywords),
                " ".join(paper.methods),
                " ".join(paper.populations),
            ]
        )

    @classmethod
    def _paper_has_any_term(cls, paper: PaperRecord, terms: set[str]) -> bool:
        text = cls._paper_text(paper).lower()
        return any(term in text for term in terms)

    @classmethod
    def _paper_topic_fit(cls, paper: PaperRecord, topic_tokens: set[str], required_topics: list[str]) -> float:
        text = cls._paper_text(paper).lower()
        if required_topics:
            return sum(1 for topic in required_topics if topic.lower() in text) / len(required_topics)
        if not topic_tokens:
            return 1.0
        paper_tokens = {token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text)}
        return len(topic_tokens & paper_tokens) / len(topic_tokens)

    def _build_outline(self, project: ResearchProject, papers: list[PaperRecord]) -> list[OutlineNodeRecord]:
        now = now_iso()
        sections = [
            (
                "landscape",
                "Research Landscape and Problem Framing",
                "Define the research landscape, core problem, and why the topic matters now.",
            ),
            (
                "methods",
                "Methods and Technical Routes",
                "Compare the dominant methods, study designs, and technical routes in the literature.",
            ),
            (
                "evidence",
                "Empirical Evidence and Application Scenarios",
                "Synthesize the most concrete findings, applications, and scenario-specific results.",
            ),
            (
                "limitations",
                "Limitations and Conflicting Findings",
                "Surface limitations, contradictions, and unresolved debates across the evidence pool.",
            ),
            (
                "innovation",
                "Candidate Innovation Directions",
                "Condense evidence-backed research gaps into candidate innovation directions.",
            ),
        ]
        return [
            OutlineNodeRecord(
                node_id=f"outline:{project.project_id}:{key}",
                project_id=project.project_id,
                title=title,
                purpose=purpose,
                section_order=index + 1,
                claim_spans=[],
                metadata={"section_key": key},
                created_at=now,
                updated_at=now,
            )
            for index, (key, title, purpose) in enumerate(sections)
        ]

    def _build_evidence_cards(
        self,
        project: ResearchProject,
        papers: list[PaperRecord],
        outline: list[OutlineNodeRecord],
    ) -> list[EvidenceCardRecord]:
        outline_by_key = {item.metadata.get("section_key"): item for item in outline}
        cards: list[EvidenceCardRecord] = []
        for paper in papers:
            section_keys = self._section_keys_for_paper(paper)
            outline_ids = [outline_by_key[key].node_id for key in section_keys if key in outline_by_key]
            cards.append(
                EvidenceCardRecord(
                    evidence_id=str(uuid.uuid4()),
                    project_id=project.project_id,
                    paper_id=paper.paper_id,
                    summary=summarize_abstract(paper.abstract or paper.title),
                    claim_spans=[
                        f"Title: {paper.title}",
                        f"Venue/Year: {(paper.venue or 'Unknown venue')} ({paper.year or 'n.d.'})",
                    ],
                    outline_node_ids=outline_ids,
                    relevance_score=paper.relevance_score,
                    recency_score=paper.recency_score,
                    evidence_level="summary+metadata" if paper.abstract else "metadata-only",
                    method_tags=paper.methods,
                    novelty_tags=self._novelty_tags_for_paper(paper),
                    source_snapshot={
                        "provider": paper.provider,
                        "source_url": paper.source_url,
                        "doi": paper.doi,
                    },
                    created_at=now_iso(),
                    updated_at=now_iso(),
                )
            )

        self._ensure_minimum_section_coverage(cards, outline, papers)
        return cards

    def _ensure_minimum_section_coverage(
        self,
        cards: list[EvidenceCardRecord],
        outline: list[OutlineNodeRecord],
        papers: list[PaperRecord],
    ) -> None:
        outline_ids = [node.node_id for node in outline]
        card_by_paper = {card.paper_id: card for card in cards}
        ranked_papers = sorted(papers, key=lambda item: item.rank_score, reverse=True)
        for outline_id in outline_ids:
            attached = [card for card in cards if outline_id in card.outline_node_ids]
            if len(attached) >= 2:
                continue
            for paper in ranked_papers:
                card = card_by_paper[paper.paper_id]
                if outline_id in card.outline_node_ids:
                    continue
                card.outline_node_ids.append(outline_id)
                attached.append(card)
                if len(attached) >= 2:
                    break

    def _build_graph_edges(
        self,
        project: ResearchProject,
        papers: list[PaperRecord],
        evidence_cards: list[EvidenceCardRecord],
        outline: list[OutlineNodeRecord],
    ) -> list[PaperEdgeRecord]:
        outline_lookup = {node.node_id: node for node in outline}
        edges: list[PaperEdgeRecord] = []
        now = now_iso()

        by_canonical = {paper.canonical_id: paper for paper in papers}
        for paper in papers:
            referenced = paper.raw_source.get("referenced_works", [])
            if isinstance(referenced, list):
                for ref in referenced[:12]:
                    target_paper = next(
                        (candidate for candidate in by_canonical.values() if candidate.provider_id == ref or candidate.source_url == ref),
                        None,
                    )
                    edges.append(
                        PaperEdgeRecord(
                            edge_id=str(uuid.uuid4()),
                            project_id=project.project_id,
                            source_paper_id=paper.paper_id,
                            relation_type="cites",
                            target_kind="paper",
                            target_ref=ref,
                            target_paper_id=target_paper.paper_id if target_paper else None,
                            created_at=now,
                        )
                    )

        for card in evidence_cards:
            for outline_node_id in card.outline_node_ids:
                outline_node = outline_lookup.get(outline_node_id)
                edges.append(
                    PaperEdgeRecord(
                        edge_id=str(uuid.uuid4()),
                        project_id=project.project_id,
                        source_paper_id=card.paper_id,
                        relation_type="supports",
                        target_kind="claim",
                        target_ref=outline_node.title if outline_node else outline_node_id,
                        target_outline_node_id=outline_node_id,
                        metadata={"evidence_id": card.evidence_id},
                        created_at=now,
                    )
                )
        return edges

    def _build_references(self, papers: list[PaperRecord], *, style: str | None = None) -> list[ReferenceEntry]:
        resolved_style = normalize_reference_style(style)
        references = [format_reference(paper, resolved_style) for paper in papers]
        seen: set[str] = set()
        deduped: list[ReferenceEntry] = []
        for entry in references:
            key = normalize_title_key(entry.formatted_text)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
        return deduped

    async def _write_exports(
        self,
        *,
        project: ResearchProject,
        output_dir: Path,
        all_papers: list[PaperRecord],
        core_papers: list[PaperRecord],
        outline: list[OutlineNodeRecord],
        evidence_cards: list[EvidenceCardRecord],
        references: list[ReferenceEntry],
        reference_style: str,
        graph: AcademicGraph | None,
    ) -> list[Path]:
        export_dir = output_dir / "academic-research" / f"{slugify(project.topic)}-{project.project_id[:8]}"
        export_dir.mkdir(parents=True, exist_ok=True)

        report_path = export_dir / "report.md"
        references_path = export_dir / "references.md"
        bibtex_path = export_dir / "references.bib"
        evidence_map_path = export_dir / "evidence_map.json"
        audit_path = export_dir / "retrieval_audit.json"
        graph_path = export_dir / "graph.json"

        evidence_map = self._build_evidence_map(project, outline, evidence_cards, core_papers)
        retrieval_audit = self._collection_audit(all_papers, references)
        report_path.write_text(
            self._render_report(project, outline, evidence_cards, core_papers, references, reference_style),
            encoding="utf-8",
        )
        references_path.write_text(
            self._render_references(project, references, reference_style),
            encoding="utf-8",
        )
        bibtex_path.write_text(
            "\n\n".join(format_bibtex_entry(paper) for paper in all_papers if self._include_reference_in_export(paper)),
            encoding="utf-8",
        )
        evidence_map_path.write_text(
            json.dumps(evidence_map, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        audit_path.write_text(
            json.dumps(retrieval_audit, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if graph is not None:
            graph_path.write_text(graph.model_dump_json(indent=2), encoding="utf-8")

        exports = [report_path, references_path, bibtex_path, evidence_map_path, audit_path]
        if graph is not None:
            exports.append(graph_path)
        return exports

    def _render_report(
        self,
        project: ResearchProject,
        outline: list[OutlineNodeRecord],
        evidence_cards: list[EvidenceCardRecord],
        papers: list[PaperRecord],
        references: list[ReferenceEntry],
        reference_style: str,
    ) -> str:
        paper_lookup = {paper.paper_id: paper for paper in papers}
        cards_by_outline: dict[str, list[EvidenceCardRecord]] = {}
        for card in evidence_cards:
            for outline_node_id in card.outline_node_ids:
                cards_by_outline.setdefault(outline_node_id, []).append(card)

        lines = [
            f"# {project.topic}",
            "",
            f"- Project ID: `{project.project_id}`",
            f"- Topic Scope: {project.scope or 'Not specified'}",
            f"- Domain: {project.domain}",
            f"- Generated: {today_stamp()}",
            f"- Core Evidence Set: {len(papers)} papers used for synthesis",
            f"- Reference Style: {reference_style_label(reference_style)}",
            "",
            "## Executive Summary",
            "",
            (
                f"This report synthesizes {len(papers)} core papers for the topic **{project.topic}**. "
                "The literature was normalized across multiple scholarly metadata providers and ranked by "
                "topic relevance, metadata completeness, recency, and citation strength."
            ),
            "",
        ]

        for node in outline:
            lines.append(f"## {node.title}")
            lines.append("")
            lines.append(node.purpose)
            lines.append("")
            node_cards = sorted(
                cards_by_outline.get(node.node_id, []),
                key=lambda item: (item.relevance_score, item.recency_score),
                reverse=True,
            )[:4]
            for card in node_cards:
                paper = paper_lookup.get(card.paper_id)
                if paper is None:
                    continue
                lines.append(f"- {paper.title} {_author_year_citation(paper)}: {card.summary}")
            if node.metadata.get("section_key") == "innovation":
                lines.append("")
                lines.append("### Candidate Innovation Directions")
                lines.extend(self._render_innovation_directions(papers))
            lines.append("")

        lines.append("## References")
        lines.append("")
        for entry in references:
            if entry.included_in_final:
                lines.append(f"- {entry.formatted_text}")
        lines.append("")
        lines.append("Full export bundle: `report.md`, `references.md`, `references.bib`, `evidence_map.json`, and `retrieval_audit.json`.")
        return "\n".join(lines).strip() + "\n"

    def _render_references(
        self,
        project: ResearchProject,
        references: list[ReferenceEntry],
        reference_style: str,
    ) -> str:
        lines = [f"# References for {project.topic}", "", f"Style: {reference_style_label(reference_style)}", ""]
        for entry in references:
            if entry.included_in_final:
                lines.append(f"- {entry.formatted_text}")
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _resolve_reference_style(project: ResearchProject, requested_style: str | None = None) -> str:
        if requested_style:
            return normalize_reference_style(requested_style)
        metadata_style = (project.metadata or {}).get("reference_style")
        if isinstance(metadata_style, str) and metadata_style:
            return normalize_reference_style(metadata_style)
        return DEFAULT_REFERENCE_STYLE

    def _build_evidence_map(
        self,
        project: ResearchProject,
        outline: list[OutlineNodeRecord],
        evidence_cards: list[EvidenceCardRecord],
        papers: list[PaperRecord],
    ) -> dict[str, Any]:
        paper_lookup = {paper.paper_id: paper for paper in papers}
        return {
            "project_id": project.project_id,
            "topic": project.topic,
            "evidence_classes": {
                "literature": [
                    {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "year": paper.year,
                        "provider": paper.provider,
                        "source_url": paper.source_url,
                        "doi": paper.doi,
                    }
                    for paper in papers
                ],
                "dataset_benchmark": self._dataset_benchmark_evidence(project, papers),
                "experiment": [],
            },
            "claim_support_policy": ("Manuscript experimental claims require dataset/benchmark or experiment artifact support. Literature-only support should be labeled as background or hypothesis."),
            "outline": [
                {
                    "node_id": node.node_id,
                    "title": node.title,
                    "purpose": node.purpose,
                    "evidence": [
                        {
                            "evidence_id": card.evidence_id,
                            "paper_id": card.paper_id,
                            "paper_title": paper_lookup.get(card.paper_id).title if paper_lookup.get(card.paper_id) else None,
                            "summary": card.summary,
                            "novelty_tags": card.novelty_tags,
                        }
                        for card in evidence_cards
                        if node.node_id in card.outline_node_ids
                    ],
                }
                for node in outline
            ],
        }

    def _dataset_benchmark_evidence(self, project: ResearchProject, papers: list[PaperRecord]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for paper in papers:
            text = self._paper_text(paper).lower()
            if not any(term in text for term in ("dataset", "benchmark", "leaderboard", "baseline", "metric", "ablation")):
                continue
            evidence.append(
                {
                    "paper_id": paper.paper_id,
                    "paper_title": paper.title,
                    "evidence_terms": [term for term in ("dataset", "benchmark", "leaderboard", "baseline", "metric", "ablation") if term in text],
                    "source_url": paper.source_url,
                    "support_status": "literature_indicates_benchmark_or_dataset",
                }
            )
        if (project.metadata or {}).get("review_quality_profile") and not evidence:
            evidence.append(
                {
                    "support_status": "missing",
                    "note": "No dataset or benchmark evidence was detected in the retained literature set.",
                }
            )
        return evidence

    def _render_innovation_directions(self, papers: list[PaperRecord]) -> list[str]:
        bullets = []
        population_gap = next((paper for paper in papers if paper.populations), None)
        method_gap = next((paper for paper in papers if len(paper.methods) >= 2), None)
        conflict_gap = next((paper for paper in papers if paper.conflict_flags), None)
        design_gap = next(
            (paper for paper in papers if "benchmark" in " ".join(paper.methods + paper.keywords).lower() or "dataset" in (paper.abstract or "").lower()),
            None,
        )

        if population_gap:
            bullets.append(f"- Research-object gap: extend the topic toward under-described populations or scenarios highlighted by **{population_gap.title}** {_author_year_citation(population_gap)}.")
        if method_gap:
            bullets.append(f"- Method-combination gap: combine complementary methods observed in **{method_gap.title}** {_author_year_citation(method_gap)} to test whether mixed pipelines outperform single-route studies.")
        if conflict_gap:
            bullets.append(f"- Conflicting-findings gap: resolve contradictory evidence surfaced in **{conflict_gap.title}** {_author_year_citation(conflict_gap)} with better-controlled replication or stratified analysis.")
        if design_gap:
            bullets.append(f"- Dataset/design limitation: strengthen benchmarking, sampling, or external validation beyond the setup described in **{design_gap.title}** {_author_year_citation(design_gap)}.")
        if not bullets:
            bullets.append("- Research-object gap: extend the current evidence base toward less-studied populations, contexts, or datasets.")
            bullets.append("- Method-combination gap: compare hybrid approaches instead of single isolated methods.")
            bullets.append("- Conflicting-findings gap: design targeted replication studies to reconcile inconsistent conclusions.")
            bullets.append("- Dataset/design limitation: improve dataset diversity, benchmark coverage, or experimental controls.")
        return bullets

    @staticmethod
    def _section_keys_for_paper(paper: PaperRecord) -> list[str]:
        text = f"{paper.title} {paper.abstract or ''}".lower()
        keys = ["landscape"]
        if paper.methods or any(token in text for token in ("method", "framework", "model", "trial", "dataset")):
            keys.append("methods")
        if any(token in text for token in ("application", "result", "case", "empirical", "patient", "deployment", "evaluation")):
            keys.append("evidence")
        if paper.conflict_flags or any(token in text for token in ("limitation", "challenge", "risk", "bias", "controvers")):
            keys.append("limitations")
        if any(token in text for token in ("future", "gap", "opportun", "open question")) or paper.methods or paper.populations:
            keys.append("innovation")
        return merge_unique(keys)

    @staticmethod
    def _novelty_tags_for_paper(paper: PaperRecord) -> list[str]:
        tags: list[str] = []
        if paper.populations:
            tags.append("research-object-gap")
        if len(paper.methods) >= 2:
            tags.append("method-combination-gap")
        if paper.conflict_flags:
            tags.append("conflicting-findings")
        if "dataset" in (paper.abstract or "").lower() or "benchmark" in (paper.abstract or "").lower():
            tags.append("dataset-or-design-limitation")
        return merge_unique(tags)

    @staticmethod
    def _include_reference_in_export(paper: PaperRecord) -> bool:
        return bool(paper.title and paper.provider and paper.canonical_source and (paper.doi or paper.source_url or paper.oa_url))

    @staticmethod
    def _to_virtual_output(thread_id: str, path: Path) -> str:
        outputs_root = get_paths().sandbox_outputs_dir(thread_id).resolve()
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(outputs_root)
        except ValueError:
            return str(resolved)
        return f"{VIRTUAL_PATH_PREFIX}/outputs/{relative.as_posix()}"

    async def _require_project(self, project_id: str) -> ResearchProject:
        project = await self._repository.get_project(project_id)
        if project is None:
            raise ValueError(f"Project {project_id} not found")
        return project


def _author_year_citation(paper: PaperRecord) -> str:
    if paper.authors:
        lead = paper.authors[0].family_name or paper.authors[0].display_name.split(" ")[-1]
        if len(paper.authors) == 1:
            label = lead
        else:
            label = f"{lead} et al."
    else:
        label = paper.title[:32].rstrip()
    year = paper.year if paper.year is not None else "n.d."
    return f"({label}, {year})"
