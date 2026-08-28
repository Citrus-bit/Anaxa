from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExperimentProject(BaseModel):
    project_id: str
    thread_id: str
    agent_name: str
    domain: str
    topic: str
    dataset_ids: list[str] = Field(default_factory=list)
    linked_academic_project_id: str | None = None
    status: str = "created"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class ExperimentRun(BaseModel):
    run_id: str
    project_id: str
    stage: str
    status: str
    method_key: str
    metrics_json: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
    created_at: str
    updated_at: str


class ExperimentFigureSpec(BaseModel):
    figure_id: str
    run_id: str
    intent: str
    chart_type: str
    grade: str = "paper"
    source_tables: list[str] = Field(default_factory=list)
    output_files: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentArtifact(BaseModel):
    artifact_id: str
    project_id: str
    run_id: str
    filepath: str
    artifact_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ExperimentBundle(BaseModel):
    project_id: str
    run_id: str
    export_files: list[str] = Field(default_factory=list)
    figure_count: int = 0
    table_count: int = 0
    linked_academic_project_id: str | None = None
    artifacts: list[ExperimentArtifact] = Field(default_factory=list)


class ExperimentExecutionResult(BaseModel):
    project: ExperimentProject
    run: ExperimentRun
    figures: list[ExperimentFigureSpec] = Field(default_factory=list)
    bundle: ExperimentBundle
    summary: dict[str, Any] = Field(default_factory=dict)


class ExperimentProjectSummary(BaseModel):
    project: ExperimentProject
    latest_run: ExperimentRun | None = None
    artifacts: list[ExperimentArtifact] = Field(default_factory=list)
    figure_count: int = 0
    run_count: int = 0
