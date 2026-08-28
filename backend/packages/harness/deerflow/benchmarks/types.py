from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetBenchmarkEntry(BaseModel):
    entry_id: str
    source: str
    name: str
    task: str | None = None
    modality: str | None = None
    version_or_date: str | None = None
    license: str | None = None
    access_url: str | None = None
    access_type: str = "unknown"
    standard_splits: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    baselines_or_sota: list[dict[str, Any]] = Field(default_factory=list)
    citation: str | None = None
    download_feasibility: str = "unknown"
    risks: list[str] = Field(default_factory=list)
    source_record: dict[str, Any] = Field(default_factory=dict)


class DatasetBenchmarkMap(BaseModel):
    topic: str
    scope: str | None = None
    generated_at: str
    sources_requested: list[str] = Field(default_factory=list)
    entries: list[DatasetBenchmarkEntry] = Field(default_factory=list)
    source_errors: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
