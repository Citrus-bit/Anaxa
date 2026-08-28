from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, cast

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.typing import ContextT

from medrix_flow.agents.thread_state import ThreadState
from medrix_flow.config import get_app_config
from medrix_flow.config.paths import VIRTUAL_PATH_PREFIX, get_paths
from medrix_flow.models import create_chat_model
from medrix_flow.research import (
    RESEARCH_STAGES,
    ResearchQuestOrchestrator,
    ResearchQuestService,
    ResearchQuestSnapshot,
    ResearchRepository,
    ResearchStage,
)
from medrix_flow.research.orchestrator import ContentGenerator
from medrix_flow.runtime.db import SQLiteRuntimeDB
from medrix_flow.runtime.events.store.sqlite import SQLiteRunEventStore
from medrix_flow.runtime.runs import RunStatus
from medrix_flow.runtime.runs.store.sqlite import SQLiteRunStore
from medrix_flow.runtime.utils import now_iso
from medrix_flow.tools.builtins.manuscript_export_tool import manuscript_export_tool


@dataclass(frozen=True)
class FinalBundleExportResult:
    artifacts: list[str]
    message: str
    status: str


def _as_stage(value: str | None) -> ResearchStage | None:
    if value is None:
        return None
    if value not in RESEARCH_STAGES:
        raise ValueError(f"Unknown research stage {value!r}. Expected one of: {', '.join(RESEARCH_STAGES)}")
    return cast(ResearchStage, value)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _manuscript_prompt_for(section_key: str, snapshot: ResearchQuestSnapshot) -> str:
    evidence = [
        {
            "claim": item.claim,
            "support_status": item.support_status,
            "source_title": item.source_title,
            "locator": item.locator,
        }
        for item in snapshot.evidence[:20]
    ]
    branches = [
        {
            "name": branch.name,
            "branch_type": branch.branch_type,
            "status": branch.status,
            "metrics": branch.metrics,
            "failure_summary": branch.failure_summary,
        }
        for branch in snapshot.experiment_branches[:10]
    ]
    return (
        "Draft one concise manuscript section for a research quest.\n"
        "Use LaTeX-friendly prose. Do not invent citations or unsupported claims.\n"
        f"Section key: {section_key}\n"
        f"Title: {snapshot.quest.title}\n"
        f"Topic: {snapshot.quest.topic}\n"
        f"Objective: {snapshot.quest.objective or 'not specified'}\n"
        f"Evidence records: {evidence}\n"
        f"Experiment branches: {branches}\n"
    )


def _build_content_generator(model_name: str | None) -> ContentGenerator:
    llm = create_chat_model(model_name, thinking_enabled=False)

    async def generate(section_key: str, snapshot: ResearchQuestSnapshot) -> str:
        response = await llm.ainvoke(_manuscript_prompt_for(section_key, snapshot))
        return _message_content_to_text(response.content).strip()

    return generate


def _resolve_thread_model_name(runtime: ToolRuntime[ContextT, ThreadState]) -> str | None:
    model_name = runtime.context.get("model_name")
    if isinstance(model_name, str) and model_name:
        return model_name
    runtime_config = getattr(runtime, "config", None)
    configurable = runtime_config.get("configurable") if isinstance(runtime_config, dict) else None
    if isinstance(configurable, dict):
        configured_model = configurable.get("model_name")
        if isinstance(configured_model, str) and configured_model:
            return configured_model
    return None


def _virtual_output_path(thread_id: str, path: Path) -> str:
    outputs_dir = get_paths().sandbox_outputs_dir(thread_id).resolve()
    return f"{VIRTUAL_PATH_PREFIX}/outputs/{path.resolve().relative_to(outputs_dir).as_posix()}"


def _section_to_latex(section: Any) -> str:
    title = str(getattr(section, "title", "") or "Section")
    content = str(getattr(section, "content", "") or "").strip()
    return f"\\section{{{title}}}\n\n{content or '% Draft content pending.'}\n"


def _snapshot_to_tex(snapshot: ResearchQuestSnapshot) -> str:
    sections = "\n\n".join(_section_to_latex(section) for section in snapshot.manuscript_sections)
    return "\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\begin{document}",
            f"\\title{{{snapshot.quest.title}}}",
            "\\maketitle",
            sections,
            "\\end{document}",
            "",
        ]
    )


def _snapshot_to_claim_map(snapshot: ResearchQuestSnapshot) -> list[dict[str, Any]]:
    return [
        {
            "claim": item.claim,
            "support_status": item.support_status,
            "source_title": item.source_title,
            "locator": item.locator,
            "paper_id": item.paper_id,
        }
        for item in snapshot.evidence
    ]


async def _write_fast_draft_artifact(thread_id: str, snapshot: ResearchQuestSnapshot) -> str | None:
    if not snapshot.manuscript_sections:
        return None
    outputs_dir = get_paths().sandbox_outputs_dir(thread_id)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"fast-draft-{snapshot.quest.quest_id}.tex"
    draft_path = outputs_dir / filename
    draft_path.write_text(_snapshot_to_tex(snapshot), encoding="utf-8")
    return _virtual_output_path(thread_id, draft_path)


def _extract_tool_message(update: dict[str, Any]) -> str:
    messages = update.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    last_message = messages[-1]
    content = getattr(last_message, "content", "")
    return str(content) if content else ""


async def _export_final_manuscript_bundle(thread_id: str, snapshot: ResearchQuestSnapshot) -> FinalBundleExportResult:
    outputs_dir = get_paths().sandbox_outputs_dir(thread_id)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    references_path = outputs_dir / "references.bib"
    bibtex_content = references_path.read_text(encoding="utf-8") if references_path.exists() else ""
    runtime = type(
        "ManuscriptExportRuntime",
        (),
        {
            "state": {
                "thread_data": {
                    "outputs_path": str(outputs_dir),
                    "workspace_path": str(get_paths().sandbox_work_dir(thread_id)),
                    "uploads_path": str(get_paths().sandbox_uploads_dir(thread_id)),
                }
            },
            "context": {"thread_id": thread_id},
        },
    )()
    try:
        command = manuscript_export_tool.func(
            runtime=runtime,
            tex_content=_snapshot_to_tex(snapshot),
            bibtex_content=bibtex_content,
            claim_map_json=json.dumps(_snapshot_to_claim_map(snapshot), ensure_ascii=False),
            filename_stem=f"manuscript-{snapshot.quest.quest_id}",
            quality_profile="paper",
            tool_call_id=f"research-final-export-{snapshot.quest.quest_id}",
        )
    except Exception as exc:
        return FinalBundleExportResult(
            artifacts=[],
            message=f"FAIL: final manuscript export errored: {exc}",
            status="error",
        )

    update = getattr(command, "update", {}) or {}
    raw_artifacts = update.get("artifacts") or []
    if not isinstance(raw_artifacts, list):
        raw_artifacts = []
    artifacts = [str(artifact) for artifact in raw_artifacts if isinstance(artifact, str)]
    message = _extract_tool_message(update)
    has_pdf = any(artifact.lower().endswith(".pdf") for artifact in artifacts)
    return FinalBundleExportResult(
        artifacts=artifacts,
        message=message,
        status="passed" if has_pdf else "blocked",
    )


async def _register_finalization_run(thread_id: str, quest_id: str) -> str:
    run_id = f"research-finalize-{uuid.uuid4().hex[:12]}"
    runtime_db = SQLiteRuntimeDB(get_paths().runtime_db_file)
    await runtime_db.connect()
    try:
        run_store = SQLiteRunStore(runtime_db)
        event_store = SQLiteRunEventStore(runtime_db)
        await run_store.setup()
        await event_store.setup()
        timestamp = now_iso()
        await run_store.put(
            run_id,
            thread_id=thread_id,
            assistant_id="research_assistant",
            status=RunStatus.running.value,
            multitask_strategy="reject",
            source="external",
            metadata={"quest_id": quest_id, "kind": "research_finalization"},
            kwargs={},
            pre_message_count=0,
            created_at=timestamp,
            updated_at=timestamp,
        )
        await event_store.put(
            thread_id=thread_id,
            run_id=run_id,
            event_type="research_finalization_started",
            caller="research_assistant",
            content={"quest_id": quest_id, "stage": "review"},
            created_at=timestamp,
        )
    finally:
        await runtime_db.close()
    return run_id


async def _complete_finalization_run(
    thread_id: str,
    run_id: str,
    result_status: str,
    final_stage: str,
    *,
    error: str | None = None,
) -> None:
    runtime_db = SQLiteRuntimeDB(get_paths().runtime_db_file)
    await runtime_db.connect()
    try:
        run_store = SQLiteRunStore(runtime_db)
        event_store = SQLiteRunEventStore(runtime_db)
        await run_store.setup()
        await event_store.setup()
        status = RunStatus.error.value if error or result_status == "error" else RunStatus.success.value
        await event_store.put(
            thread_id=thread_id,
            run_id=run_id,
            event_type="research_finalization_completed" if status == RunStatus.success.value else "research_finalization_failed",
            caller="research_assistant",
            content={"status": result_status, "final_stage": final_stage, "error": error},
        )
        await run_store.update_status(run_id, status, error=error)
    finally:
        await runtime_db.close()


async def _set_quest_metadata(quest_id: str, metadata: dict[str, Any]) -> None:
    db = SQLiteRuntimeDB(get_paths().research_db_file)
    await db.connect()
    try:
        repository = ResearchRepository(db)
        await repository.setup()
        quest = await repository.get_quest(quest_id)
        if quest is None:
            return
        quest.metadata = {**quest.metadata, **metadata}
        quest.updated_at = now_iso()
        await repository.update_quest(quest)
    finally:
        await db.close()


async def _attach_draft_artifact(quest_id: str, artifact: str) -> None:
    db = SQLiteRuntimeDB(get_paths().research_db_file)
    await db.connect()
    try:
        repository = ResearchRepository(db)
        await repository.setup()
        sections = await repository.list_manuscript_sections(quest_id)
        for section in sections:
            if artifact not in section.artifact_paths:
                section.artifact_paths = [*section.artifact_paths, artifact]
                section.updated_at = now_iso()
                await repository.upsert_manuscript_section(section)
    finally:
        await db.close()


async def _run_finalization_background(
    *,
    thread_id: str,
    quest_id: str,
    run_id: str,
    auto_gates: list[str],
    max_stages: int,
    quality_mode: str,
    repair_budget: int,
    model_name: str | None,
) -> None:
    db = SQLiteRuntimeDB(get_paths().research_db_file)
    await db.connect()
    try:
        repository = ResearchRepository(db)
        await repository.setup()
        service = ResearchQuestService(repository)
        orchestrator = ResearchQuestOrchestrator(service)
        result = await orchestrator.run_pipeline(
            quest_id,
            auto_gates=auto_gates,
            max_stages=max_stages,
            quality_mode=quality_mode,
            repair_budget=repair_budget,
            delivery_mode="final_only",
            content_generator=_build_content_generator(model_name),
        )
        export_result: FinalBundleExportResult | None = None
        if result.final_stage == "final_bundle":
            export_result = await _export_final_manuscript_bundle(thread_id, await service.get_snapshot(quest_id))
        await _complete_finalization_run(thread_id, run_id, result.status, result.final_stage, error=result.error)
        if export_result is not None:
            await _set_quest_metadata(
                quest_id,
                {
                    "final_bundle_artifacts": export_result.artifacts,
                    "final_bundle_export_status": export_result.status,
                    "final_bundle_export_message": export_result.message,
                    **({"final_bundle_ready_at": now_iso()} if export_result.status == "passed" else {}),
                },
            )
    except Exception as exc:
        await _complete_finalization_run(thread_id, run_id, "error", "manuscript_draft", error=str(exc))
    finally:
        await db.close()


@tool("research_assistant", parse_docstring=True)
async def research_assistant_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    action: str = "start",
    topic: str | None = None,
    quest_id: str | None = None,
    target_stage: str | None = None,
    inputs: dict[str, Any] | None = None,
    artifacts: list[str] | None = None,
    scope: str | None = None,
    objective: str | None = None,
    gate_stage: str | None = None,
    gate_type: str | None = None,
    gate_status: str = "approved",
    gate_reason: str | None = None,
    auto_gates: list[str] | None = None,
    max_stages: int | None = None,
    quality_mode: str | None = None,
    quality_repair_budget: int | None = None,
    delivery_mode: str | None = None,
) -> Command:
    """Start, inspect, or advance a staged research quest.

    Use this tool when the user wants an automatic research assistant,
    research lifecycle tracking, novelty checks, claim-level evidence mapping,
    experiment gates, reviewer loops, or a manuscript workspace. This tool
    coordinates the research quest ledger; use `academic_research` for heavy
    literature retrieval and `experiment_lab` for actual dataset execution.

    Args:
        action: One of `start`, `status`, `advance`, `gate`, or `run_pipeline`.
        topic: Research topic. Required when starting a quest.
        quest_id: Existing research quest id. If omitted for status/advance,
            the latest quest for the current thread is used.
        target_stage: Optional immediate next lifecycle stage for `advance`.
        inputs: Structured stage inputs such as claims, idea, branches, metrics,
            academic_project_id, experiment_project_id, or completed_actions.
        artifacts: Artifact paths to attach to the ledger entry.
        scope: Optional research scope for a new quest.
        objective: Optional concrete research objective for a new quest.
        gate_stage: Stage guarded by a human gate when action is `gate`.
        gate_type: Gate type such as `experiment_execution`, `pre_review`, or `final_release`.
        gate_status: Gate status for action `gate`; defaults to `approved`.
        gate_reason: Optional human reason or note for the gate decision.
        auto_gates: Optional gate types to auto-approve for `run_pipeline`; defaults to config.
        max_stages: Optional max lifecycle stages to advance for `run_pipeline`; defaults to config.
        quality_mode: Optional quality gate mode for `run_pipeline`: `auto_repair`, `audit_only`, or `strict_gate`.
        quality_repair_budget: Optional max automatic quality-repair approvals for `run_pipeline`.
        delivery_mode: Optional delivery mode for `run_pipeline`: `fast_draft_first` or `final_only`.
    """
    thread_id = runtime.context.get("thread_id")
    if not thread_id:
        return Command(update={"messages": [ToolMessage("Error: thread_id is required for research_assistant.", tool_call_id=tool_call_id)]})

    db = SQLiteRuntimeDB(get_paths().research_db_file)
    await db.connect()
    response_artifacts: list[str] = []
    try:
        repository = ResearchRepository(db)
        await repository.setup()
        service = ResearchQuestService(repository)
        orchestrator = ResearchQuestOrchestrator(service)

        resolved_quest_id = quest_id
        if not resolved_quest_id and action in {"status", "advance", "gate", "run_pipeline"}:
            quests = await service.list_quests(str(thread_id))
            resolved_quest_id = quests[0].quest_id if quests else None

        if action == "start":
            if not topic:
                raise ValueError("topic is required when starting a research quest")
            quest = await service.create_quest(
                thread_id=str(thread_id),
                topic=topic,
                scope=scope,
                objective=objective,
                metadata={"created_by": "research_assistant"},
            )
            message = f"Research quest `{quest.quest_id}` started at stage `{quest.stage}` for topic: {quest.topic}"
        elif action == "status":
            if not resolved_quest_id:
                raise ValueError("quest_id is required and no quest exists for this thread")
            snapshot = await service.get_snapshot(resolved_quest_id)
            message = (
                f"Research quest `{snapshot.quest.quest_id}` is `{snapshot.quest.status}` at stage `{snapshot.quest.stage}`. "
                f"Evidence claims: {len(snapshot.evidence)}, branches: {len(snapshot.experiment_branches)}, "
                f"review reports: {len(snapshot.reviewer_reports)}, gates: {len(snapshot.gates)}."
            )
        elif action == "advance":
            if not resolved_quest_id:
                if not topic:
                    raise ValueError("quest_id is required for advance when no topic is supplied")
                quest = await service.create_quest(thread_id=str(thread_id), topic=topic, scope=scope, objective=objective)
                resolved_quest_id = quest.quest_id
            result = await service.advance_quest(
                resolved_quest_id,
                target_stage=_as_stage(target_stage),
                inputs=inputs or {},
                artifacts=artifacts or [],
                tool_name="research_assistant",
            )
            if result.blocked and result.required_gate:
                message = f"Research quest `{result.quest.quest_id}` is blocked before `{result.required_gate.stage}`. Required gate: `{result.required_gate.gate_type}`."
            else:
                message = f"Research quest `{result.quest.quest_id}` advanced to `{result.quest.stage}`."
        elif action == "run_pipeline":
            if not resolved_quest_id:
                if not topic:
                    raise ValueError("quest_id is required for run_pipeline when no topic is supplied")
                quest = await service.create_quest(
                    thread_id=str(thread_id),
                    topic=topic,
                    scope=scope,
                    objective=objective,
                    metadata={"created_by": "research_assistant", "pipeline": "run_pipeline"},
                )
                resolved_quest_id = quest.quest_id
            config = get_app_config()
            await _set_quest_metadata(
                resolved_quest_id,
                {"manuscript_section_concurrency": config.research.manuscript_section_concurrency},
            )
            resolved_delivery_mode = delivery_mode or config.research.default_delivery_mode
            thread_model_name = _resolve_thread_model_name(runtime)
            draft_model_name = config.research.fast_draft_model or config.research.manuscript_model or thread_model_name
            finalization_model_name = config.research.finalization_model or config.research.manuscript_model or config.research.fast_draft_model or thread_model_name
            config_auto_gates = auto_gates if auto_gates is not None else config.research.default_auto_gates
            resolved_auto_gates = list(config_auto_gates)
            if resolved_delivery_mode == "fast_draft_first" and "experiment_execution" not in resolved_auto_gates:
                resolved_auto_gates.append("experiment_execution")
            if resolved_delivery_mode == "final_only":
                for gate_type in ("experiment_execution", "pre_review", "final_release"):
                    if gate_type not in resolved_auto_gates:
                        resolved_auto_gates.append(gate_type)
            finalization_auto_gates = list(resolved_auto_gates)
            for gate_type in ("pre_review", "final_release"):
                if gate_type not in finalization_auto_gates:
                    finalization_auto_gates.append(gate_type)
            resolved_max_stages = max_stages if max_stages is not None else config.research.default_max_stages
            if resolved_delivery_mode == "fast_draft_first":
                resolved_max_stages = max(resolved_max_stages, 8)
            elif max_stages is None:
                resolved_max_stages = max(resolved_max_stages, 11)
            resolved_quality_mode = quality_mode or config.research.default_quality_mode
            resolved_repair_budget = quality_repair_budget if quality_repair_budget is not None else config.research.default_quality_repair_budget
            pipeline_model_name = finalization_model_name if resolved_delivery_mode == "final_only" else draft_model_name
            result = await orchestrator.run_pipeline(
                resolved_quest_id,
                auto_gates=resolved_auto_gates,
                max_stages=resolved_max_stages,
                quality_mode=resolved_quality_mode,
                repair_budget=resolved_repair_budget,
                delivery_mode=resolved_delivery_mode,
                content_generator=_build_content_generator(pipeline_model_name),
            )
            message = f"Research pipeline `{result.quest_id}` returned `{result.status}` at stage `{result.final_stage}`. Stages executed: {len(result.stages_executed)}."
            if result.final_stage == "final_bundle":
                export_result = await _export_final_manuscript_bundle(str(thread_id), await service.get_snapshot(resolved_quest_id))
                response_artifacts.extend(export_result.artifacts)
                metadata = {
                    "delivery_mode": resolved_delivery_mode,
                    "final_bundle_artifacts": export_result.artifacts,
                    "final_bundle_export_status": export_result.status,
                    "final_bundle_export_message": export_result.message,
                }
                if export_result.status == "passed":
                    metadata["final_bundle_ready_at"] = now_iso()
                    message += f" Final PDF export passed. {export_result.message}"
                elif export_result.status == "blocked":
                    message += f" Final bundle export was blocked. {export_result.message}"
                else:
                    message += f" Final bundle export errored. {export_result.message}"
                await _set_quest_metadata(resolved_quest_id, metadata)
            if result.status == "draft_ready" and resolved_delivery_mode == "fast_draft_first":
                snapshot = await service.get_snapshot(resolved_quest_id)
                draft_artifact = await _write_fast_draft_artifact(str(thread_id), snapshot)
                finalization_run_id = await _register_finalization_run(str(thread_id), resolved_quest_id)
                metadata: dict[str, Any] = {
                    "delivery_mode": resolved_delivery_mode,
                    "finalization_run_id": finalization_run_id,
                    "draft_ready_at": now_iso(),
                }
                if draft_artifact:
                    metadata["draft_artifacts"] = [draft_artifact]
                    response_artifacts.append(draft_artifact)
                    await _attach_draft_artifact(resolved_quest_id, draft_artifact)
                await _set_quest_metadata(resolved_quest_id, metadata)
                asyncio.create_task(
                    _run_finalization_background(
                        thread_id=str(thread_id),
                        quest_id=resolved_quest_id,
                        run_id=finalization_run_id,
                        auto_gates=finalization_auto_gates,
                        max_stages=11,
                        quality_mode=resolved_quality_mode,
                        repair_budget=resolved_repair_budget,
                        model_name=finalization_model_name,
                    )
                )
                message += f" Fast draft is ready; finalization run `{finalization_run_id}` is continuing in the background."
                if draft_artifact:
                    message += f" Draft artifact: `{draft_artifact}`."
            if result.blocked_gate:
                message += f" Blocked gate: `{result.blocked_gate}`."
            if result.error:
                message += f" Error: {result.error}"
        elif action == "gate":
            if not resolved_quest_id:
                raise ValueError("quest_id is required and no quest exists for this thread")
            if not gate_stage or not gate_type:
                raise ValueError("gate_stage and gate_type are required for gate decisions")
            gate = await service.decide_gate(
                resolved_quest_id,
                stage=_as_stage(gate_stage) or "intake",
                gate_type=gate_type,
                status=gate_status,
                reason=gate_reason,
            )
            message = f"Research gate `{gate.gate_type}` for `{gate.stage}` is now `{gate.status}`."
        else:
            raise ValueError("action must be one of: start, status, advance, gate, run_pipeline")
    except Exception as exc:
        await db.close()
        return Command(update={"messages": [ToolMessage(f"Error: {exc}", tool_call_id=tool_call_id)]})

    await db.close()
    update: dict[str, Any] = {"messages": [ToolMessage(message, tool_call_id=tool_call_id)]}
    if response_artifacts:
        update["artifacts"] = response_artifacts
    return Command(update=update)
