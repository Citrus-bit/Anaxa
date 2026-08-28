from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import httpx

from deerflow.academic.utils import slugify
from deerflow.runtime.utils import now_iso

from .types import DatasetBenchmarkEntry, DatasetBenchmarkMap

_DEFAULT_SOURCES = [
    "papers-with-code",
    "huggingface",
    "openml",
    "zenodo",
    "github",
    "kaggle",
]
_SOURCE_ALIASES = {
    "paperswithcode": "papers-with-code",
    "papers-with-code": "papers-with-code",
    "pwc": "papers-with-code",
    "hf": "huggingface",
    "huggingface": "huggingface",
    "hugging-face": "huggingface",
    "openml": "openml",
    "uci": "uci",
    "zenodo": "zenodo",
    "figshare": "figshare",
    "osf": "osf",
    "github": "github",
    "kaggle": "kaggle",
    "official": "official-leaderboard",
    "leaderboard": "official-leaderboard",
}
_METRIC_TERMS = {
    "accuracy",
    "auroc",
    "auc",
    "auprc",
    "bleu",
    "cer",
    "dice",
    "f1",
    "fid",
    "mae",
    "map",
    "mcc",
    "mse",
    "ndcg",
    "precision",
    "psnr",
    "r2",
    "recall",
    "rmse",
    "rouge",
    "wer",
}


class DatasetBenchmarkDiscoveryService:
    """Discover dataset and benchmark candidates without bypassing access gates."""

    def __init__(self, *, timeout: float = 15.0) -> None:
        self._timeout = timeout

    async def discover(
        self,
        *,
        topic: str,
        scope: str | None = None,
        output_dir: Path,
        max_results: int = 30,
        sources: list[str] | None = None,
    ) -> tuple[DatasetBenchmarkMap, Path]:
        normalized_sources = self._normalize_sources(sources)
        per_source_limit = max(2, min(10, max_results // max(1, len(normalized_sources)) + 1))
        entries: list[DatasetBenchmarkEntry] = []
        source_errors: dict[str, str] = {}

        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={"User-Agent": "MedrixFlowDatasetBenchmarkDiscovery/0.1", "Accept": "application/json"},
        ) as client:
            tasks = [self._search_source(client, source, topic, scope, per_source_limit) for source in normalized_sources]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for source, result in zip(normalized_sources, results, strict=True):
            if isinstance(result, Exception):
                source_errors[source] = str(result)
                continue
            entries.extend(result)

        deduped = self._dedupe(entries)[: max(1, max_results)]
        benchmark_map = DatasetBenchmarkMap(
            topic=topic,
            scope=scope,
            generated_at=now_iso(),
            sources_requested=normalized_sources,
            entries=deduped,
            source_errors=source_errors,
            notes=[
                "Restricted, login-gated, or license-gated datasets are marked instead of downloaded.",
                "Verify leaderboard dates and dataset terms before using results as manuscript evidence.",
            ],
        )
        export_dir = output_dir / "dataset-benchmark-discovery" / slugify(topic, fallback="benchmark-discovery")
        export_dir.mkdir(parents=True, exist_ok=True)
        output_path = export_dir / "dataset_benchmark_map.json"
        output_path.write_text(benchmark_map.model_dump_json(indent=2), encoding="utf-8")
        return benchmark_map, output_path

    async def _search_source(
        self,
        client: httpx.AsyncClient,
        source: str,
        topic: str,
        scope: str | None,
        limit: int,
    ) -> list[DatasetBenchmarkEntry]:
        if source == "huggingface":
            return await self._search_huggingface(client, topic, scope, limit)
        if source == "openml":
            return await self._search_openml(client, topic, scope, limit)
        if source == "zenodo":
            return await self._search_zenodo(client, topic, scope, limit)
        if source == "papers-with-code":
            return await self._search_papers_with_code(client, topic, scope, limit)
        if source == "github":
            return await self._search_github(client, topic, scope, limit)
        return self._reference_only_entries(source, topic, scope)

    async def _search_huggingface(
        self,
        client: httpx.AsyncClient,
        topic: str,
        scope: str | None,
        limit: int,
    ) -> list[DatasetBenchmarkEntry]:
        params = {"search": self._query(topic, scope), "limit": str(limit), "full": "true"}
        response = await client.get(f"https://huggingface.co/api/datasets?{urlencode(params)}")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return []

        entries: list[DatasetBenchmarkEntry] = []
        for item in payload[:limit]:
            dataset_id = str(item.get("id") or item.get("datasetId") or "").strip()
            if not dataset_id:
                continue
            tags = [str(tag) for tag in item.get("tags", []) if isinstance(tag, str)]
            entries.append(
                DatasetBenchmarkEntry(
                    entry_id=f"huggingface:{dataset_id}",
                    source="huggingface",
                    name=dataset_id,
                    task=self._first_tag(tags, prefixes=("task_categories:", "task_ids:")),
                    modality=self._first_tag(tags, prefixes=("modality:",)) or self._infer_modality(" ".join(tags + [dataset_id])),
                    version_or_date=str(item.get("lastModified") or item.get("createdAt") or "") or None,
                    license=self._first_tag(tags, prefixes=("license:",)),
                    access_url=f"https://huggingface.co/datasets/{dataset_id}",
                    access_type="open-or-gated",
                    standard_splits=[],
                    metrics=self._extract_metrics(" ".join(tags + [dataset_id])),
                    baselines_or_sota=[],
                    citation=None,
                    download_feasibility="check_dataset_card_and_gating",
                    risks=self._risks_for_license(self._first_tag(tags, prefixes=("license:",))),
                    source_record=self._compact_record(item),
                )
            )
        return entries

    async def _search_openml(
        self,
        client: httpx.AsyncClient,
        topic: str,
        scope: str | None,
        limit: int,
    ) -> list[DatasetBenchmarkEntry]:
        response = await client.get("https://www.openml.org/api/v1/json/data/list/limit/100")
        response.raise_for_status()
        payload = response.json()
        datasets = ((payload.get("data") or {}).get("dataset") or []) if isinstance(payload, dict) else []
        query_terms = self._tokens(self._query(topic, scope))
        scored: list[tuple[int, dict[str, Any]]] = []
        for item in datasets:
            text = " ".join(str(item.get(key) or "") for key in ("name", "description", "format", "tag"))
            score = sum(1 for token in query_terms if token in text.lower())
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        entries: list[DatasetBenchmarkEntry] = []
        for _, item in scored[:limit]:
            dataset_id = str(item.get("did") or item.get("id") or item.get("name") or "").strip()
            name = str(item.get("name") or dataset_id).strip()
            if not name:
                continue
            entries.append(
                DatasetBenchmarkEntry(
                    entry_id=f"openml:{dataset_id}",
                    source="openml",
                    name=name,
                    task=str(item.get("task_type") or "") or None,
                    modality=self._infer_modality(name),
                    version_or_date=str(item.get("version") or item.get("upload_date") or "") or None,
                    license=str(item.get("licence") or item.get("license") or "") or None,
                    access_url=f"https://www.openml.org/d/{dataset_id}" if dataset_id else "https://www.openml.org/search?type=data",
                    access_type="open-metadata",
                    standard_splits=[],
                    metrics=[],
                    baselines_or_sota=[],
                    citation=str(item.get("citation") or "") or None,
                    download_feasibility="metadata_available_check_terms_before_download",
                    risks=self._risks_for_license(str(item.get("licence") or item.get("license") or "") or None),
                    source_record=self._compact_record(item),
                )
            )
        return entries

    async def _search_zenodo(
        self,
        client: httpx.AsyncClient,
        topic: str,
        scope: str | None,
        limit: int,
    ) -> list[DatasetBenchmarkEntry]:
        params = {"q": self._query(topic, scope), "size": str(limit), "type": "dataset", "sort": "mostrecent"}
        response = await client.get(f"https://zenodo.org/api/records?{urlencode(params)}")
        response.raise_for_status()
        payload = response.json()
        hits = ((payload.get("hits") or {}).get("hits") or []) if isinstance(payload, dict) else []

        entries: list[DatasetBenchmarkEntry] = []
        for item in hits[:limit]:
            metadata = item.get("metadata") or {}
            title = str(metadata.get("title") or item.get("title") or "").strip()
            if not title:
                continue
            license_info = metadata.get("license") or {}
            license_name = license_info.get("id") if isinstance(license_info, dict) else str(license_info or "")
            entries.append(
                DatasetBenchmarkEntry(
                    entry_id=f"zenodo:{item.get('id') or title}",
                    source="zenodo",
                    name=title,
                    task=None,
                    modality=self._infer_modality(" ".join([title, str(metadata.get("description") or "")])),
                    version_or_date=str(metadata.get("publication_date") or item.get("updated") or "") or None,
                    license=str(license_name or "") or None,
                    access_url=item.get("links", {}).get("html") if isinstance(item.get("links"), dict) else None,
                    access_type="repository-record",
                    standard_splits=[],
                    metrics=self._extract_metrics(" ".join([title, str(metadata.get("description") or "")])),
                    baselines_or_sota=[],
                    citation=str(metadata.get("doi") or item.get("doi") or "") or None,
                    download_feasibility="repository_files_available_check_license",
                    risks=self._risks_for_license(str(license_name or "") or None),
                    source_record=self._compact_record(item),
                )
            )
        return entries

    async def _search_papers_with_code(
        self,
        client: httpx.AsyncClient,
        topic: str,
        scope: str | None,
        limit: int,
    ) -> list[DatasetBenchmarkEntry]:
        query = self._query(topic, scope)
        entries: list[DatasetBenchmarkEntry] = []
        for endpoint, kind in (("datasets", "dataset"), ("tasks", "task")):
            params = {"q": query, "page_size": str(limit)}
            try:
                response = await client.get(f"https://paperswithcode.com/api/v1/{endpoint}/?{urlencode(params)}")
                response.raise_for_status()
            except httpx.HTTPStatusError:
                continue
            payload = response.json()
            results = payload.get("results") if isinstance(payload, dict) else []
            if not isinstance(results, list):
                continue
            for item in results[:limit]:
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                url = item.get("url") or item.get("homepage")
                entries.append(
                    DatasetBenchmarkEntry(
                        entry_id=f"papers-with-code:{kind}:{item.get('id') or name}",
                        source="papers-with-code",
                        name=name,
                        task=str(item.get("task") or name if kind == "task" else item.get("task") or "") or None,
                        modality=self._infer_modality(" ".join([name, str(item.get("description") or "")])),
                        version_or_date=str(item.get("introduced_date") or item.get("created") or "") or None,
                        license=str(item.get("license") or "") or None,
                        access_url=url if isinstance(url, str) else None,
                        access_type="benchmark-index",
                        standard_splits=[],
                        metrics=self._extract_metrics(" ".join([name, str(item.get("description") or "")])),
                        baselines_or_sota=[],
                        citation=str(item.get("paper") or "") or None,
                        download_feasibility="index_record_verify_dataset_page",
                        risks=self._risks_for_license(str(item.get("license") or "") or None),
                        source_record=self._compact_record(item),
                    )
                )
        return entries

    async def _search_github(
        self,
        client: httpx.AsyncClient,
        topic: str,
        scope: str | None,
        limit: int,
    ) -> list[DatasetBenchmarkEntry]:
        query = f"{self._query(topic, scope)} benchmark dataset"
        params = {"q": query, "per_page": str(min(limit, 10)), "sort": "updated"}
        response = await client.get(f"https://api.github.com/search/repositories?{urlencode(params)}")
        response.raise_for_status()
        payload = response.json()
        items = payload.get("items") if isinstance(payload, dict) else []
        if not isinstance(items, list):
            return []

        entries: list[DatasetBenchmarkEntry] = []
        for item in items[:limit]:
            full_name = str(item.get("full_name") or item.get("name") or "").strip()
            if not full_name:
                continue
            text = " ".join([full_name, str(item.get("description") or ""), " ".join(item.get("topics") or [])])
            entries.append(
                DatasetBenchmarkEntry(
                    entry_id=f"github:{full_name}",
                    source="github",
                    name=full_name,
                    task=None,
                    modality=self._infer_modality(text),
                    version_or_date=str(item.get("updated_at") or item.get("pushed_at") or "") or None,
                    license=(item.get("license") or {}).get("spdx_id") if isinstance(item.get("license"), dict) else None,
                    access_url=str(item.get("html_url") or "") or None,
                    access_type="code-or-benchmark-repository",
                    standard_splits=[],
                    metrics=self._extract_metrics(text),
                    baselines_or_sota=[],
                    citation=None,
                    download_feasibility="repository_available_verify_data_license",
                    risks=["Repository may provide code only; verify dataset hosting and benchmark protocol."],
                    source_record=self._compact_record(item),
                )
            )
        return entries

    def _reference_only_entries(self, source: str, topic: str, scope: str | None) -> list[DatasetBenchmarkEntry]:
        query = quote(self._query(topic, scope))
        source_urls = {
            "kaggle": f"https://www.kaggle.com/datasets?search={query}",
            "uci": "https://archive.ics.uci.edu/",
            "figshare": f"https://figshare.com/search?q={query}",
            "osf": f"https://osf.io/search/?q={query}",
            "official-leaderboard": f"https://www.google.com/search?q={query}%20official%20leaderboard%20benchmark",
        }
        name = f"{source} search for {topic}"
        risks = ["Manual verification required; this source may require login, API credentials, or domain-specific access terms."]
        return [
            DatasetBenchmarkEntry(
                entry_id=f"{source}:{slugify(topic, fallback='topic')}",
                source=source,
                name=name,
                task=topic,
                modality=self._infer_modality(topic),
                access_url=source_urls.get(source),
                access_type="manual-verification",
                download_feasibility="manual_check_required",
                risks=risks,
                source_record={"query": self._query(topic, scope)},
            )
        ]

    @staticmethod
    def _normalize_sources(sources: list[str] | None) -> list[str]:
        if not sources:
            return list(_DEFAULT_SOURCES)
        normalized: list[str] = []
        seen: set[str] = set()
        for item in sources:
            key = re.sub(r"[^a-z0-9]+", "-", str(item).strip().lower()).strip("-")
            source = _SOURCE_ALIASES.get(key, key)
            if source and source not in seen:
                normalized.append(source)
                seen.add(source)
        return normalized or list(_DEFAULT_SOURCES)

    @staticmethod
    def _query(topic: str, scope: str | None) -> str:
        return " ".join(part for part in [topic.strip(), (scope or "").strip()] if part)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text.lower())}

    @staticmethod
    def _first_tag(tags: Iterable[str], *, prefixes: tuple[str, ...]) -> str | None:
        for tag in tags:
            lowered = tag.lower()
            for prefix in prefixes:
                if lowered.startswith(prefix):
                    return tag.split(":", 1)[1] if ":" in tag else tag
        return None

    @staticmethod
    def _extract_metrics(text: str) -> list[str]:
        lowered = text.lower()
        return sorted({term for term in _METRIC_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered)})

    @staticmethod
    def _infer_modality(text: str) -> str | None:
        lowered = text.lower()
        if any(term in lowered for term in ("image", "vision", "segmentation", "detection", "video")):
            return "vision"
        if any(term in lowered for term in ("text", "nlp", "language", "translation", "qa", "question answering")):
            return "text"
        if any(term in lowered for term in ("audio", "speech", "voice")):
            return "audio"
        if any(term in lowered for term in ("tabular", "classification", "regression", "openml")):
            return "tabular"
        if any(term in lowered for term in ("protein", "gene", "genomic", "rna", "single-cell", "single cell")):
            return "bioinformatics"
        return None

    @staticmethod
    def _risks_for_license(license_name: str | None) -> list[str]:
        if not license_name:
            return ["License not detected; verify terms before downloading or redistributing."]
        lowered = license_name.lower()
        if any(term in lowered for term in ("unknown", "other", "custom", "non-commercial", "nc")):
            return ["License may restrict reuse; verify paper and dataset terms before experiments."]
        return []

    @staticmethod
    def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
        raw = json.loads(json.dumps(record, default=str))
        compact: dict[str, Any] = {}
        for key, value in raw.items():
            if key in {"description", "readme", "cardData"} and isinstance(value, str) and len(value) > 600:
                compact[key] = value[:600] + "..."
            elif key in {"files", "siblings"} and isinstance(value, list):
                compact[key] = value[:20]
            elif key not in {"downloads", "likes"} or isinstance(value, int | float | str | bool | type(None)):
                compact[key] = value
        return compact

    @staticmethod
    def _dedupe(entries: list[DatasetBenchmarkEntry]) -> list[DatasetBenchmarkEntry]:
        seen: set[str] = set()
        deduped: list[DatasetBenchmarkEntry] = []
        for entry in entries:
            key = re.sub(r"[^a-z0-9]+", "", f"{entry.source}:{entry.name}".lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
        return deduped
