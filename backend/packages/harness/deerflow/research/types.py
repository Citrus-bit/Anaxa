from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ResearchStage = Literal[
    "intake",
    "literature",
    "novelty_check",
    "evidence_verified",
    "experiment_planned",
    "experiment_running",
    "results_synthesized",
    "manuscript_draft",
    "review",
    "revision",
    "final_bundle",
]

ResearchQuestStatus = Literal["active", "blocked", "completed", "error", "cancelled"]
PipelineStatus = Literal["completed", "blocked_on_gate", "stopped_at_max_stages", "draft_ready", "error", "cancelled"]
GateStatus = Literal["pending", "approved", "rejected"]
SupportStatus = Literal["supported", "unsupported", "contradicted", "uncertain"]
OverlapRisk = Literal["low", "medium", "high"]
BranchStatus = Literal["planned", "running", "completed", "failed", "skipped"]
ReviewerVerdict = Literal["pass", "revise", "block"]
QualityAuditStatus = Literal["pass", "revise", "block"]

RESEARCH_STAGES: tuple[ResearchStage, ...] = (
    "intake",
    "literature",
    "novelty_check",
    "evidence_verified",
    "experiment_planned",
    "experiment_running",
    "results_synthesized",
    "manuscript_draft",
    "review",
    "revision",
    "final_bundle",
)

GATED_TRANSITIONS: dict[ResearchStage, str] = {
    "experiment_running": "experiment_execution",
    "review": "pre_review",
    "final_bundle": "final_release",
}


class PipelineStageEvent(BaseModel):
    stage: ResearchStage
    entered_at: str
    completed_at: str | None = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    error: str | None = None


class PipelineRunResult(BaseModel):
    quest_id: str
    status: PipelineStatus
    stages_executed: list[PipelineStageEvent] = Field(default_factory=list)
    final_stage: ResearchStage
    blocked_gate: str | None = None
    error: str | None = None
    message: str


class ResearchQuest(BaseModel):
    quest_id: str
    thread_id: str
    title: str
    topic: str
    scope: str | None = None
    objective: str | None = None
    domain: str = "general"
    stage: ResearchStage = "intake"
    status: ResearchQuestStatus = "active"
    academic_project_id: str | None = None
    experiment_project_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class ResearchLedgerEntry(BaseModel):
    entry_id: str
    quest_id: str
    stage: ResearchStage
    event_type: str
    summary: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    tool_name: str | None = None
    model_name: str | None = None
    error: str | None = None
    gate_decision: str | None = None
    created_at: str


class ResearchGate(BaseModel):
    gate_id: str
    quest_id: str
    stage: ResearchStage
    gate_type: str
    status: GateStatus = "pending"
    decision: str | None = None
    reason: str | None = None
    required: bool = True
    created_at: str
    decided_at: str | None = None


class ClaimEvidenceRecord(BaseModel):
    claim_id: str
    quest_id: str
    claim: str
    paper_id: str | None = None
    source_title: str | None = None
    locator: str | None = None
    snippet: str | None = None
    quote: str | None = None
    support_status: SupportStatus = "uncertain"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    artifact_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class NoveltyCheckRecord(BaseModel):
    check_id: str
    quest_id: str
    idea: str
    overlap_risk: OverlapRisk
    closest_papers: list[dict[str, Any]] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    minimum_experiment: str | None = None
    decision: str = "proceed"
    created_at: str


class ExperimentBranchRecord(BaseModel):
    branch_id: str
    quest_id: str
    experiment_project_id: str | None = None
    parent_branch_id: str | None = None
    name: str
    branch_type: str = "baseline"
    status: BranchStatus = "planned"
    priority: float = 0.0
    seed: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: list[str] = Field(default_factory=list)
    failure_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class ManuscriptSectionRecord(BaseModel):
    section_id: str
    quest_id: str
    section_key: str
    title: str
    content: str = ""
    claim_ids: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    status: str = "draft"
    created_at: str
    updated_at: str


class ReviewerReportRecord(BaseModel):
    report_id: str
    quest_id: str
    stage: ResearchStage
    reviewer_profile: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    verdict: ReviewerVerdict = "revise"
    findings: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    created_at: str


class ResearchQualityAuditRecord(BaseModel):
    audit_id: str
    quest_id: str
    stage: ResearchStage
    status: QualityAuditStatus = "revise"
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    findings: list[str] = Field(default_factory=list)
    repair_actions: list[str] = Field(default_factory=list)
    recommended_queries: list[str] = Field(default_factory=list)
    created_at: str


class ResearchQuestSnapshot(BaseModel):
    quest: ResearchQuest
    ledger: list[ResearchLedgerEntry] = Field(default_factory=list)
    gates: list[ResearchGate] = Field(default_factory=list)
    evidence: list[ClaimEvidenceRecord] = Field(default_factory=list)
    novelty_checks: list[NoveltyCheckRecord] = Field(default_factory=list)
    experiment_branches: list[ExperimentBranchRecord] = Field(default_factory=list)
    manuscript_sections: list[ManuscriptSectionRecord] = Field(default_factory=list)
    reviewer_reports: list[ReviewerReportRecord] = Field(default_factory=list)
    quality_audits: list[ResearchQualityAuditRecord] = Field(default_factory=list)


class ResearchAdvanceResult(BaseModel):
    quest: ResearchQuest
    advanced: bool
    blocked: bool = False
    required_gate: ResearchGate | None = None
    ledger_entry: ResearchLedgerEntry | None = None
    generated: dict[str, Any] = Field(default_factory=dict)
