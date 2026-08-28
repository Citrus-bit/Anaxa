from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.typing import ContextT

from deerflow.academic import AcademicRepository, AcademicResearchService, reference_style_label
from deerflow.agents.thread_state import ThreadState
from deerflow.config.paths import VIRTUAL_PATH_PREFIX, get_paths
from deerflow.runtime.db import SQLiteRuntimeDB


@tool("academic_research", parse_docstring=True)
async def academic_research_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    topic: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    scope: str | None = None,
    include_graph: bool = False,
    max_candidates: int = 120,
    core_paper_limit: int = 24,
    reference_style: str | None = None,
    deliverable_type: str | None = None,
    min_reference_count: int | None = None,
    target_reference_count: int | None = None,
    required_topics: list[str] | None = None,
    required_evidence_types: list[str] | None = None,
) -> Command:
    """Run the academic literature pipeline and produce report artifacts.

    Use this tool when the user asks for a literature review, experiment report,
    academic report, user-selected references, innovation-point mining, or a
    research evidence pack from a topic.

    The tool will:
    - create or reuse a thread-scoped academic project
    - retrieve and normalize scholarly metadata
    - rank and deduplicate papers
    - synthesize a report, references in the requested style, BibTeX, and evidence map
    - present the generated files back to the user as artifacts

    Args:
        topic: Research topic to investigate.
        scope: Optional scope, domain constraint, or audience qualifier.
        include_graph: Whether to also export graph.json.
        max_candidates: Maximum number of raw candidate records to fetch.
        core_paper_limit: Maximum number of core papers kept after ranking.
        reference_style: Optional reference style requested by the user. Supports
            apa7, mla9, chicago, gbt7714, plain, and bibtex; defaults to apa7
            only when the user does not specify a style.
        deliverable_type: Optional target deliverable such as literature_review,
            manuscript, paper, review_article, survey, or short_report.
        min_reference_count: Optional minimum usable final references. Review
            and manuscript deliverables default to 50.
        target_reference_count: Optional target candidate final references.
            Review and manuscript deliverables default to 80.
        required_topics: Optional topic/model/benchmark coverage hints that
            should appear in the final reference set.
        required_evidence_types: Optional evidence coverage hints such as
            benchmark, dataset, case study, empirical result, ablation, or metric.
    """
    thread_id = runtime.context.get("thread_id")
    if not thread_id:
        return Command(update={"messages": [ToolMessage("Error: thread_id is required for academic_research.", tool_call_id=tool_call_id)]})

    thread_data = runtime.state.get("thread_data") or {}
    outputs_path = thread_data.get("outputs_path")
    if not outputs_path:
        return Command(update={"messages": [ToolMessage("Error: thread outputs path is not available.", tool_call_id=tool_call_id)]})

    db = SQLiteRuntimeDB(get_paths().academic_db_file)
    await db.connect()
    try:
        repository = AcademicRepository(db)
        await repository.setup()
        service = AcademicResearchService(repository)
        result = await service.run_research(
            thread_id=thread_id,
            topic=topic,
            scope=scope,
            output_dir=Path(outputs_path),
            include_graph=include_graph,
            max_candidates=max_candidates,
            core_paper_limit=core_paper_limit,
            reference_style=reference_style,
            deliverable_type=deliverable_type,
            min_reference_count=min_reference_count,
            target_reference_count=target_reference_count,
            required_topics=required_topics,
            required_evidence_types=required_evidence_types,
        )
    except Exception as exc:
        await db.close()
        return Command(update={"messages": [ToolMessage(f"Error: {exc}", tool_call_id=tool_call_id)]})

    try:
        artifact_paths = []
        outputs_dir = Path(outputs_path).resolve()
        for filepath in result.export_files:
            relative = Path(filepath).resolve().relative_to(outputs_dir)
            artifact_paths.append(f"{VIRTUAL_PATH_PREFIX}/outputs/{relative.as_posix()}")
        included_refs = len([entry for entry in result.references if entry.included_in_final])
        style = result.references[0].style if result.references else reference_style
        coverage_audit = (result.project.metadata or {}).get("reference_coverage_audit") or {}
        summary = (
            f"Academic project `{result.project.project_id}` ready. "
            f"Core papers: {len(result.evidence_cards)} evidence cards, "
            f"References ({reference_style_label(style)}): {included_refs}. "
            f"Artifacts: {', '.join(Path(path).name for path in result.export_files)}"
        )
        if coverage_audit:
            summary += f" Coverage audit: {coverage_audit.get('status', 'unknown')} ({coverage_audit.get('included_reference_count', included_refs)}/{coverage_audit.get('min_reference_count', 'n/a')} minimum references)."
        return Command(
            update={
                "artifacts": artifact_paths,
                "messages": [ToolMessage(summary, tool_call_id=tool_call_id)],
            }
        )
    finally:
        await db.close()
