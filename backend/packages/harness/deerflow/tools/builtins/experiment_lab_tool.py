from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState
from deerflow.config.paths import get_paths
from deerflow.experiments import ExperimentRepository, ExperimentService
from deerflow.runtime.db import SQLiteRuntimeDB


@tool("experiment_lab", parse_docstring=True)
async def experiment_lab_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    topic: str,
    dataset_paths: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
    domain: str | None = None,
    analysis_type: str | None = None,
    target_column: str | None = None,
    metadata_path: str | None = None,
    sample_id_column: str | None = None,
    group_column: str | None = None,
    linked_academic_project_id: str | None = None,
    publication_grade: str = "paper",
    synthetic_data_mode: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Command:
    """Run the experiment laboratory pipeline and export reproducible result bundles.

    Use this tool when the user wants CS/AI experiments, bioinformatics analyses,
    diagnostic plots, manuscript-ready figures, or experiment bundles that can feed
    a technical or academic report.

    For iterative or autonomous experiment requests, use this tool as the
    structured execution/export step while keeping the surrounding loop grounded
    in a fixed baseline, fixed primary metric, and explicit keep/discard/crash
    trial log.

    Args:
        topic: The experiment objective or analysis question.
        dataset_paths: Dataset paths under `/mnt/user-data/uploads` or `/mnt/user-data/workspace`.
        domain: Optional explicit domain (`cs_ai` or `bioinformatics`).
        analysis_type: Optional explicit analysis type such as `classification`,
            `regression`, `clustering`, `bulk_expression`, or `single_cell`.
        target_column: Optional target column for supervised CS/AI tasks.
        metadata_path: Optional metadata table path for bulk bioinformatics tasks.
        sample_id_column: Optional sample/cell identifier column in metadata.
        group_column: Optional group/condition column in metadata.
        linked_academic_project_id: Optional academic project to export paper-ready result files for.
        publication_grade: Figure quality target. Defaults to `paper`.
        synthetic_data_mode: Allow assumption-driven simulated personal experiment data when real execution inputs are unavailable.
        metadata: Optional extra project metadata to retain in the experiment store.
    """
    thread_id = runtime.context.get("thread_id")
    if not thread_id:
        return Command(update={"messages": [ToolMessage("Error: thread_id is required for experiment_lab.", tool_call_id=tool_call_id)]})
    thread_data = runtime.state.get("thread_data") or {}
    outputs_path = thread_data.get("outputs_path")
    if not outputs_path:
        return Command(update={"messages": [ToolMessage("Error: thread outputs path is not available.", tool_call_id=tool_call_id)]})
    agent_name = runtime.context.get("agent_name") or (domain or "cs-ai-lab")
    project_metadata = dict(metadata or {})
    context_synthetic_mode = bool(runtime.context.get("synthetic_data_mode"))
    if synthetic_data_mode or context_synthetic_mode:
        project_metadata["synthetic_data_mode"] = True
        project_metadata.setdefault("simulation_trigger", "synthetic_data_mode")

    db = SQLiteRuntimeDB(get_paths().experiment_db_file)
    await db.connect()
    try:
        repository = ExperimentRepository(db)
        await repository.setup()
        service = ExperimentService(repository)
        result = await service.run_experiment(
            thread_id=thread_id,
            agent_name=str(agent_name),
            topic=topic,
            dataset_ids=dataset_paths,
            output_dir=Path(outputs_path),
            domain=domain,
            linked_academic_project_id=linked_academic_project_id,
            metadata=project_metadata,
            analysis_type=analysis_type,
            target_column=target_column,
            metadata_path=metadata_path,
            sample_id_column=sample_id_column,
            group_column=group_column,
            publication_grade=publication_grade,
        )
    except Exception as exc:
        await db.close()
        return Command(update={"messages": [ToolMessage(f"Error: {exc}", tool_call_id=tool_call_id)]})

    try:
        summary = f"Experiment project `{result.project.project_id}` completed. Run `{result.run.run_id}` produced {result.bundle.figure_count} figure(s) and {result.bundle.table_count} table(s)."
        return Command(
            update={
                "artifacts": result.bundle.export_files,
                "messages": [ToolMessage(summary, tool_call_id=tool_call_id)],
            }
        )
    finally:
        await db.close()
