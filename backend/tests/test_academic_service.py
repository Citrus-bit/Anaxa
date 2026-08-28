import asyncio
from pathlib import Path

from medrix_flow.academic import AcademicRepository, AcademicResearchService
from medrix_flow.academic.types import PaperAuthor, PaperRecord
from medrix_flow.config import paths as paths_module
from medrix_flow.runtime.db import SQLiteRuntimeDB
from medrix_flow.runtime.utils import now_iso


class FakeAdapter:
    def __init__(self, name: str, papers: list[PaperRecord]) -> None:
        self.name = name
        self._papers = papers

    async def search(self, query: str, *, project_id: str, limit: int) -> list[PaperRecord]:
        results: list[PaperRecord] = []
        for paper in self._papers[:limit]:
            results.append(paper.model_copy(update={"project_id": project_id, "paper_id": f"{project_id}:{paper.paper_id}"}))
        return results


def _paper(
    *,
    provider: str,
    provider_id: str,
    title: str,
    year: int,
    venue: str,
    doi: str | None,
    abstract: str,
    referenced_works: list[str] | None = None,
) -> PaperRecord:
    return PaperRecord(
        paper_id=f"{provider}:{provider_id}",
        project_id="project-x",
        canonical_id=f"doi:{doi.lower()}" if doi else f"{provider}:{provider_id}",
        title=title,
        authors=[PaperAuthor(display_name="Alice Smith", given_name="Alice", family_name="Smith", ordinal=0)],
        year=year,
        venue=venue,
        abstract=abstract,
        doi=doi,
        provider=provider,
        provider_id=provider_id,
        source_url=f"https://example.org/{provider_id}",
        oa_url=None,
        metadata_only=False,
        keywords=[],
        methods=["benchmarking"] if "benchmark" in abstract.lower() else ["review"],
        populations=["patients"] if "patient" in abstract.lower() else [],
        conflict_flags=["challenge"] if "challenge" in abstract.lower() else [],
        raw_source={"referenced_works": referenced_works or []},
        created_at=now_iso(),
        updated_at=now_iso(),
    )


def test_academic_service_reuses_project_and_prefers_local_uploads(tmp_path, monkeypatch):
    async def scenario() -> None:
        monkeypatch.setenv("MEDRIX_FLOW_HOME", str(tmp_path))
        paths_module._paths = None

        uploads_dir = paths_module.get_paths().sandbox_uploads_dir("thread-1")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        (uploads_dir / "uploaded-paper.md").write_text(
            "# Uploaded Evidence Paper\n\nDOI: 10.9999/uploaded\n\n2023\n\nA patient benchmark challenge with practical limitations.",
            encoding="utf-8",
        )

        db = SQLiteRuntimeDB(":memory:")
        await db.connect()
        repo = AcademicRepository(db)
        await repo.setup()
        service = AcademicResearchService(repo, adapters=[FakeAdapter("openalex", [])])

        first = await service.create_project(thread_id="thread-1", topic="clinical reasoning foundation models")
        second = await service.create_project(thread_id="thread-1", topic="clinical reasoning foundation models")
        assert first.project_id == second.project_id

        ingested = await service.ingest_project(first.project_id, max_candidates=20, core_paper_limit=20)
        assert ingested.paper_count >= 1
        assert any(paper.provider == "local-upload" for paper in ingested.selected_papers)

        await db.close()

    asyncio.run(scenario())


def test_academic_service_synthesizes_exports_and_graph(tmp_path, monkeypatch):
    async def scenario() -> None:
        monkeypatch.setenv("MEDRIX_FLOW_HOME", str(tmp_path))
        paths_module._paths = None

        db = SQLiteRuntimeDB(":memory:")
        await db.connect()
        repo = AcademicRepository(db)
        await repo.setup()
        base_papers = [
            _paper(
                provider="openalex",
                provider_id="w1",
                title="Foundation models for clinical reasoning",
                year=2025,
                venue="Journal of Clinical AI",
                doi="10.1000/w1",
                abstract="A review of clinical reasoning methods and benchmark design.",
                referenced_works=["https://openalex.org/w2"],
            ),
            _paper(
                provider="openalex",
                provider_id="w2",
                title="Benchmark datasets for patient-facing decision support",
                year=2024,
                venue="Medical Data Systems",
                doi="10.1000/w2",
                abstract="Benchmark dataset construction for patient decision support with empirical evaluation.",
            ),
            _paper(
                provider="crossref",
                provider_id="w3",
                title="Challenges in multimodal evidence synthesis",
                year=2023,
                venue="Computational Medicine Review",
                doi="10.1000/w3",
                abstract="A review describing challenge patterns, limitations, and controversy in evidence synthesis.",
            ),
            _paper(
                provider="arxiv",
                provider_id="2501.12345",
                title="Hybrid retrieval pipelines for experiment reports",
                year=2025,
                venue="arXiv",
                doi=None,
                abstract="A benchmark study on hybrid retrieval pipelines for experiment reports and open questions.",
            ),
        ]
        service = AcademicResearchService(repo, adapters=[FakeAdapter("openalex", base_papers)])

        project = await service.create_project(thread_id="thread-2", topic="experiment report retrieval")
        ingested = await service.ingest_project(project.project_id, max_candidates=20, core_paper_limit=20)
        assert ingested.paper_count >= 4

        output_dir = tmp_path / "thread-2-outputs"
        result = await service.synthesize_project(project.project_id, output_dir=output_dir, include_graph=True)

        export_files = {Path(path).name for path in result.export_files}
        assert {
            "report.md",
            "references.md",
            "references.bib",
            "evidence_map.json",
            "graph.json",
            "retrieval_audit.json",
        } <= export_files
        assert len([entry for entry in result.references if entry.included_in_final]) >= 3
        assert len(result.graph.nodes) >= 4
        assert len(result.graph.edges) >= 4
        assert any("Candidate Innovation Directions" in Path(path).read_text(encoding="utf-8") for path in result.export_files if path.endswith("report.md"))

        await db.close()

    asyncio.run(scenario())


def test_manuscript_research_defaults_to_experimental_evidence_requirements(tmp_path, monkeypatch):
    async def scenario() -> None:
        monkeypatch.setenv("MEDRIX_FLOW_HOME", str(tmp_path))
        paths_module._paths = None

        db = SQLiteRuntimeDB(":memory:")
        await db.connect()
        repo = AcademicRepository(db)
        await repo.setup()
        service = AcademicResearchService(repo, adapters=[FakeAdapter("openalex", [])])

        project = await service.create_project(thread_id="thread-manuscript", topic="medical AI benchmark paper")
        updated = await service._apply_coverage_metadata(
            project,
            deliverable_type="manuscript",
            min_reference_count=None,
            target_reference_count=None,
            required_topics=None,
            required_evidence_types=None,
        )

        evidence_types = updated.metadata["required_evidence_types"]
        assert {"dataset", "benchmark", "metric", "baseline", "ablation", "external validation"} <= set(evidence_types)
        assert updated.metadata["review_quality_profile"] is True

        await db.close()

    asyncio.run(scenario())


def test_academic_service_prefers_published_versions_and_exports_all_verified_references(tmp_path, monkeypatch):
    async def scenario() -> None:
        monkeypatch.setenv("MEDRIX_FLOW_HOME", str(tmp_path))
        paths_module._paths = None

        db = SQLiteRuntimeDB(":memory:")
        await db.connect()
        repo = AcademicRepository(db)
        await repo.setup()

        matched_doi = "10.5555/final-version"
        papers = [
            _paper(
                provider="arxiv",
                provider_id="2501.11111",
                title="Retrieval-augmented evaluation for large language models",
                year=2025,
                venue="arXiv",
                doi=matched_doi,
                abstract="A benchmark study on large language model evaluation with retrieval augmentation.",
            ),
            _paper(
                provider="dblp",
                provider_id="conf/neurips/final-version",
                title="Retrieval-augmented evaluation for large language models",
                year=2025,
                venue="NeurIPS",
                doi=matched_doi,
                abstract="A benchmark study on large language model evaluation with retrieval augmentation.",
            ),
        ]
        papers.extend(
            [
                _paper(
                    provider="crossref",
                    provider_id=f"paper-{idx}",
                    title=f"Verified study {idx}",
                    year=2024,
                    venue="Journal of Evaluation Science",
                    doi=f"10.6000/{idx}",
                    abstract="A verified study with sufficient metadata for APA export.",
                )
                for idx in range(30)
            ]
        )

        class RollingAdapter:
            name = "mixed"

            def __init__(self, rows: list[PaperRecord]) -> None:
                self._rows = rows
                self._offset = 0

            async def search(self, query: str, *, project_id: str, limit: int) -> list[PaperRecord]:
                start = self._offset
                end = min(start + limit, len(self._rows))
                self._offset = end
                batch = self._rows[start:end]
                return [
                    paper.model_copy(
                        update={
                            "project_id": project_id,
                            "paper_id": f"{project_id}:{paper.paper_id}:{idx}",
                        }
                    )
                    for idx, paper in enumerate(batch, start=start)
                ]

        service = AcademicResearchService(repo, adapters=[RollingAdapter(papers)])
        project = await service.create_project(thread_id="thread-refs", topic="retrieval augmented generation evaluation")
        ingested = await service.ingest_project(project.project_id, max_candidates=120, core_paper_limit=20)

        stored = await repo.list_project_papers(project.project_id)
        canonical = next(paper for paper in stored if paper.canonical_id == f"doi:{matched_doi}")
        assert canonical.provider == "dblp"
        assert canonical.is_preprint is False

        result = await service.synthesize_project(project.project_id, output_dir=tmp_path / "outputs", include_graph=False)
        included = [entry for entry in result.references if entry.included_in_final]

        assert len(ingested.selected_papers) <= 20
        assert len(included) >= 30
        assert all("arxiv.org" not in entry.formatted_text.lower() for entry in included if "Retrieval-augmented evaluation" in entry.formatted_text)

        await db.close()

    asyncio.run(scenario())


def test_academic_service_writes_requested_reference_style(tmp_path, monkeypatch):
    async def scenario() -> None:
        monkeypatch.setenv("MEDRIX_FLOW_HOME", str(tmp_path))
        paths_module._paths = None

        db = SQLiteRuntimeDB(":memory:")
        await db.connect()
        repo = AcademicRepository(db)
        await repo.setup()
        service = AcademicResearchService(
            repo,
            adapters=[
                FakeAdapter(
                    "openalex",
                    [
                        _paper(
                            provider="openalex",
                            provider_id="w-style",
                            title="Reference style selection for academic agents",
                            year=2026,
                            venue="Journal of Research Tooling",
                            doi="10.7000/style",
                            abstract="A benchmark study on reference style selection.",
                        )
                    ],
                )
            ],
        )

        result = await service.run_research(
            thread_id="thread-style",
            topic="reference style selection",
            output_dir=tmp_path / "outputs",
            max_candidates=20,
            core_paper_limit=20,
            reference_style="mla",
        )
        references_path = next(Path(path) for path in result.export_files if path.endswith("references.md"))
        references_text = references_path.read_text(encoding="utf-8")
        summary = await service.get_project_summary(result.project.project_id)

        assert result.references
        assert {entry.style for entry in result.references} == {"mla9"}
        assert "Style: MLA 9" in references_text
        assert summary["reference_style"] == "mla9"

        await db.close()

    asyncio.run(scenario())


def test_academic_service_reuses_passed_coverage_bundle(tmp_path, monkeypatch):
    async def scenario() -> None:
        monkeypatch.setenv("MEDRIX_FLOW_HOME", str(tmp_path))
        paths_module._paths = None

        db = SQLiteRuntimeDB(":memory:")
        await db.connect()
        repo = AcademicRepository(db)
        await repo.setup()

        class CountingAdapter:
            name = "openalex"

            def __init__(self) -> None:
                self.calls = 0

            async def search(self, query: str, *, project_id: str, limit: int) -> list[PaperRecord]:
                self.calls += 1
                return [
                    _paper(
                        provider="openalex",
                        provider_id="reuse-1",
                        title="Reusable benchmark dataset with ablation evidence",
                        year=2025,
                        venue="Journal of Reusable Evidence",
                        doi="10.8300/reuse-1",
                        abstract=("A benchmark dataset paper with metric reporting, baseline comparison, ablation analysis, and external validation."),
                    ).model_copy(update={"project_id": project_id, "paper_id": f"{project_id}:reuse-1"})
                ]

        adapter = CountingAdapter()
        service = AcademicResearchService(repo, adapters=[adapter])

        first = await service.run_research(
            thread_id="thread-reuse",
            topic="reusable benchmark evidence",
            output_dir=tmp_path / "outputs-1",
            min_reference_count=1,
            target_reference_count=1,
        )
        calls_after_first = adapter.calls
        second = await service.run_research(
            thread_id="thread-reuse",
            topic="reusable benchmark evidence",
            output_dir=tmp_path / "outputs-2",
            min_reference_count=1,
            target_reference_count=1,
        )

        assert first.project.project_id == second.project.project_id
        assert calls_after_first > 0
        assert adapter.calls == calls_after_first
        assert (second.project.metadata or {})["reference_coverage_audit"]["status"] == "pass"
        assert any(Path(path).name == "references.bib" for path in second.export_files)

        await db.close()

    asyncio.run(scenario())


def test_academic_service_reingests_when_reuse_signature_changes(tmp_path, monkeypatch):
    async def scenario() -> None:
        monkeypatch.setenv("MEDRIX_FLOW_HOME", str(tmp_path))
        paths_module._paths = None

        db = SQLiteRuntimeDB(":memory:")
        await db.connect()
        repo = AcademicRepository(db)
        await repo.setup()

        class CountingAdapter:
            name = "openalex"

            def __init__(self) -> None:
                self.calls = 0

            async def search(self, query: str, *, project_id: str, limit: int) -> list[PaperRecord]:
                self.calls += 1
                return [
                    _paper(
                        provider="openalex",
                        provider_id="signature-1",
                        title="Signature-sensitive benchmark dataset with ablation evidence",
                        year=2025,
                        venue="Journal of Reusable Evidence",
                        doi="10.8300/signature-1",
                        abstract=("A benchmark dataset paper with metric reporting, baseline comparison, ablation analysis, and external validation."),
                    ).model_copy(update={"project_id": project_id, "paper_id": f"{project_id}:signature-1"})
                ]

        adapter = CountingAdapter()
        service = AcademicResearchService(repo, adapters=[adapter])

        first = await service.run_research(
            thread_id="thread-signature",
            topic="signature sensitive benchmark evidence",
            output_dir=tmp_path / "outputs-1",
            min_reference_count=1,
            target_reference_count=1,
            source_profile="default",
        )
        calls_after_first = adapter.calls
        second = await service.run_research(
            thread_id="thread-signature",
            topic="signature sensitive benchmark evidence",
            output_dir=tmp_path / "outputs-2",
            min_reference_count=1,
            target_reference_count=1,
            source_profile="alternate-profile",
        )
        calls_after_source_change = adapter.calls
        third = await service.run_research(
            thread_id="thread-signature",
            topic="signature sensitive benchmark evidence",
            output_dir=tmp_path / "outputs-3",
            min_reference_count=1,
            target_reference_count=1,
            source_profile="alternate-profile",
            quality_mode="strict",
        )
        calls_after_quality_change = adapter.calls
        fourth = await service.run_research(
            thread_id="thread-signature",
            topic="signature sensitive benchmark evidence",
            output_dir=tmp_path / "outputs-4",
            min_reference_count=1,
            target_reference_count=1,
            source_profile="alternate-profile",
            quality_mode="strict",
            preprint_policy="include_preprints",
        )
        calls_after_preprint_change = adapter.calls
        fifth = await service.run_research(
            thread_id="thread-signature",
            topic="signature sensitive benchmark evidence",
            output_dir=tmp_path / "outputs-5",
            min_reference_count=1,
            target_reference_count=1,
            source_profile="alternate-profile",
            quality_mode="strict",
            preprint_policy="include_preprints",
            core_paper_limit=40,
        )

        assert first.project.project_id == second.project.project_id
        assert first.project.project_id == third.project.project_id
        assert first.project.project_id == fourth.project.project_id
        assert first.project.project_id == fifth.project.project_id
        assert calls_after_first > 0
        assert adapter.calls > calls_after_first
        assert calls_after_source_change > calls_after_first
        assert calls_after_quality_change > calls_after_source_change
        assert calls_after_preprint_change > calls_after_quality_change
        assert adapter.calls > calls_after_preprint_change
        signature = (fifth.project.metadata or {})["ingest_request_signature"]
        assert signature["source_profile"] == "alternate-profile"
        assert signature["quality_mode"] == "strict"
        assert signature["preprint_policy"] == "include_preprints"
        assert signature["core_paper_limit"] == 40

        await db.close()

    asyncio.run(scenario())


def test_academic_service_review_deliverable_applies_coverage_targets(tmp_path, monkeypatch):
    async def scenario() -> None:
        monkeypatch.setenv("MEDRIX_FLOW_HOME", str(tmp_path))
        paths_module._paths = None

        db = SQLiteRuntimeDB(":memory:")
        await db.connect()
        repo = AcademicRepository(db)
        await repo.setup()
        papers = [
            _paper(
                provider="openalex",
                provider_id=f"review-{idx}",
                title=f"Core review benchmark evidence {idx}",
                year=2024,
                venue="Journal of Evaluation",
                doi=f"10.8000/review-{idx}",
                abstract="A benchmark dataset and empirical result for review-quality evidence.",
            )
            for idx in range(12)
        ]
        service = AcademicResearchService(repo, adapters=[FakeAdapter("openalex", papers)])
        project = await service.create_project(thread_id="thread-review-quality", topic="agent evaluation frameworks")

        await service.ingest_project(
            project.project_id,
            max_candidates=20,
            core_paper_limit=20,
            deliverable_type="literature_review",
            min_reference_count=50,
            required_topics=["agent evaluation"],
            required_evidence_types=["benchmark", "dataset"],
        )
        summary = await service.get_project_summary(project.project_id)
        audit = summary["reference_coverage_audit"]

        assert summary["project"]["metadata"]["review_quality_profile"] is True
        assert summary["project"]["metadata"]["target_reference_count"] == 80
        assert summary["project"]["metadata"]["core_paper_limit"] >= 30
        assert audit["status"] == "block"
        assert audit["included_reference_count"] == 12
        assert audit["min_reference_count"] == 50
        assert audit["quantitative_evidence_count"] == 12
        assert audit["auto_repair_attempted"] is True
        assert audit["auto_repair_query_count"] >= 1
        assert summary["project"]["metadata"]["query_count"] > 10
        assert any("systematic review" in query for query in audit["recommended_queries"])

        await db.close()

    asyncio.run(scenario())


def test_academic_service_coverage_auto_repair_can_fill_review_reference_gap(tmp_path, monkeypatch):
    async def scenario() -> None:
        monkeypatch.setenv("MEDRIX_FLOW_HOME", str(tmp_path))
        paths_module._paths = None

        class RepairAwareAdapter:
            name = "openalex"

            async def search(self, query: str, *, project_id: str, limit: int) -> list[PaperRecord]:
                if "systematic review" in query.lower() or "empirical results" in query.lower():
                    start = 10 if "systematic review" in query.lower() else 30
                    count = min(limit, 20)
                else:
                    start = 0
                    count = min(limit, 10)
                return [
                    _paper(
                        provider="openalex",
                        provider_id=f"repair-{start + idx}",
                        title=f"Agent evaluation frameworks repair benchmark evidence {start + idx}",
                        year=2024,
                        venue="Journal of Repair Evidence",
                        doi=f"10.8200/repair-{start + idx}",
                        abstract=("A benchmark dataset and empirical result for agent evaluation frameworks with metric reporting, baseline comparison, ablation, and external validation."),
                    ).model_copy(update={"project_id": project_id, "paper_id": f"{project_id}:repair-{start + idx}"})
                    for idx in range(count)
                ]

        db = SQLiteRuntimeDB(":memory:")
        await db.connect()
        repo = AcademicRepository(db)
        await repo.setup()
        service = AcademicResearchService(repo, adapters=[RepairAwareAdapter()])
        project = await service.create_project(thread_id="thread-review-repair", topic="agent evaluation frameworks")

        await service.ingest_project(
            project.project_id,
            max_candidates=20,
            core_paper_limit=20,
            deliverable_type="literature_review",
            min_reference_count=30,
            target_reference_count=40,
        )
        summary = await service.get_project_summary(project.project_id)
        audit = summary["reference_coverage_audit"]

        assert audit["auto_repair_attempted"] is True
        assert audit["auto_repair_candidate_count"] > 0
        assert audit["included_reference_count"] >= 30
        assert audit["status"] == "pass"

        await db.close()

    asyncio.run(scenario())


def test_academic_service_coverage_audit_reports_missing_required_topic(tmp_path, monkeypatch):
    async def scenario() -> None:
        monkeypatch.setenv("MEDRIX_FLOW_HOME", str(tmp_path))
        paths_module._paths = None

        db = SQLiteRuntimeDB(":memory:")
        await db.connect()
        repo = AcademicRepository(db)
        await repo.setup()
        service = AcademicResearchService(
            repo,
            adapters=[
                FakeAdapter(
                    "openalex",
                    [
                        _paper(
                            provider="openalex",
                            provider_id="topic-gap",
                            title="Benchmark evidence for generic evaluation",
                            year=2024,
                            venue="Journal of Evaluation",
                            doi="10.8100/topic-gap",
                            abstract="A benchmark dataset with empirical results for evaluation.",
                        )
                    ],
                )
            ],
        )
        project = await service.create_project(thread_id="thread-topic-gap", topic="agent evaluation frameworks")

        await service.ingest_project(
            project.project_id,
            max_candidates=20,
            core_paper_limit=20,
            deliverable_type="short_report",
            min_reference_count=1,
            required_topics=["virtual cell foundation model"],
        )
        audit = (await service.get_project_summary(project.project_id))["reference_coverage_audit"]

        assert audit["status"] == "revise"
        assert audit["missing_required_topics"] == ["virtual cell foundation model"]
        assert "agent evaluation frameworks virtual cell foundation model" in audit["recommended_queries"]

        await db.close()

    asyncio.run(scenario())
