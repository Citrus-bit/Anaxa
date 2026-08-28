from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResearchProject(BaseModel):
    project_id: str
    thread_id: str
    topic: str
    topic_key: str
    scope: str | None = None
    status: str = "created"
    domain: str = "general"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class SearchQueryRecord(BaseModel):
    query_id: str
    project_id: str
    query_text: str
    rationale: str | None = None
    query_type: str = "general"
    source: str = "planner"
    created_at: str


class PaperAuthor(BaseModel):
    display_name: str
    given_name: str | None = None
    family_name: str | None = None
    orcid: str | None = None
    ordinal: int = 0


class PaperRecord(BaseModel):
    paper_id: str
    project_id: str
    canonical_id: str
    title: str
    authors: list[PaperAuthor] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    arxiv_id: str | None = None
    cited_by_count: int | None = None
    provider: str
    provider_id: str | None = None
    source_url: str | None = None
    oa_url: str | None = None
    metadata_only: bool = False
    source_kind: str = "metadata"
    venue_type: str = "unknown"
    venue_tier: str = "unknown"
    publication_status: str = "unknown"
    is_preprint: bool = False
    canonical_source: str | None = None
    quality_signals: dict[str, Any] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    populations: list[str] = Field(default_factory=list)
    conflict_flags: list[str] = Field(default_factory=list)
    raw_source: dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = 0.0
    recency_score: float = 0.0
    completeness_score: float = 0.0
    rank_score: float = 0.0
    created_at: str
    updated_at: str


class PaperEdgeRecord(BaseModel):
    edge_id: str
    project_id: str
    source_paper_id: str | None = None
    relation_type: str
    target_kind: str
    target_ref: str
    target_paper_id: str | None = None
    target_outline_node_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class EvidenceCardRecord(BaseModel):
    evidence_id: str
    project_id: str
    paper_id: str
    summary: str
    claim_spans: list[str] = Field(default_factory=list)
    outline_node_ids: list[str] = Field(default_factory=list)
    relevance_score: float = 0.0
    recency_score: float = 0.0
    evidence_level: str = "metadata-only"
    method_tags: list[str] = Field(default_factory=list)
    novelty_tags: list[str] = Field(default_factory=list)
    source_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class OutlineNodeRecord(BaseModel):
    node_id: str
    project_id: str
    title: str
    purpose: str
    section_order: int
    claim_spans: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class ReportExportRecord(BaseModel):
    export_id: str
    project_id: str
    export_kind: str
    path: str
    summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ReferenceEntry(BaseModel):
    paper_id: str
    style: str = "apa7"
    formatted_text: str
    doi_url: str | None = None
    completeness: str = "complete"
    included_in_final: bool = True


class AcademicGraph(BaseModel):
    project_id: str
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)


class IngestResult(BaseModel):
    project: ResearchProject
    queries: list[SearchQueryRecord]
    raw_candidate_count: int
    paper_count: int
    selected_papers: list[PaperRecord]
    provider_breakdown: dict[str, int] = Field(default_factory=dict)
    venue_breakdown: dict[str, int] = Field(default_factory=dict)
    preprint_ratio: float = 0.0
    canonical_reference_count: int = 0


class SynthesisResult(BaseModel):
    project: ResearchProject
    outline: list[OutlineNodeRecord]
    evidence_cards: list[EvidenceCardRecord]
    references: list[ReferenceEntry]
    graph: AcademicGraph
    export_files: list[str]
