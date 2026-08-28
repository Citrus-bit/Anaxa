from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from medrix_flow.runtime.utils import now_iso

from .service import ResearchQuestService
from .types import (
    RESEARCH_STAGES,
    PipelineRunResult,
    PipelineStageEvent,
    ResearchAdvanceResult,
    ResearchQuest,
    ResearchQuestSnapshot,
    ResearchStage,
)

ContentGenerator = Callable[[str, ResearchQuestSnapshot], Awaitable[str]]
ReviewerGenerator = Callable[[str, ResearchQuestSnapshot], Awaitable[dict[str, Any]]]


class ResearchQuestOrchestrator:
    """Thin orchestration facade for end-to-end research lifecycle actions.

    The service owns persistence and deterministic stage handlers. This facade
    gives agent tools and future schedulers a stable place to coordinate longer
    workflows without coupling them directly to router request models.
    """

    def __init__(self, service: ResearchQuestService) -> None:
        self._service = service

    async def start(
        self,
        *,
        thread_id: str,
        topic: str,
        title: str | None = None,
        scope: str | None = None,
        objective: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchQuest:
        return await self._service.create_quest(
            thread_id=thread_id,
            topic=topic,
            title=title,
            scope=scope,
            objective=objective,
            metadata=metadata,
        )

    async def advance(
        self,
        quest_id: str,
        *,
        target_stage: ResearchStage | None = None,
        inputs: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
        tool_name: str | None = None,
        model_name: str | None = None,
    ) -> ResearchAdvanceResult:
        return await self._service.advance_quest(
            quest_id,
            target_stage=target_stage,
            inputs=inputs,
            artifacts=artifacts,
            tool_name=tool_name,
            model_name=model_name,
        )

    async def run_pipeline(
        self,
        quest_id: str,
        *,
        auto_gates: list[str] | None = None,
        max_stages: int = 11,
        quality_mode: str = "auto_repair",
        repair_budget: int = 2,
        delivery_mode: str = "final_only",
        content_generator: ContentGenerator | None = None,
        reviewer_generator: ReviewerGenerator | None = None,
    ) -> PipelineRunResult:
        """Advance a quest until completion, a human gate, or the stage budget.

        The generator callbacks are accepted here so agent tools can use a
        stable orchestration signature. Stage handlers opt into them as their
        service contracts are upgraded.
        """
        if max_stages < 1:
            raise ValueError("max_stages must be at least 1.")
        if delivery_mode not in {"fast_draft_first", "final_only"}:
            raise ValueError("delivery_mode must be fast_draft_first or final_only.")

        allowed_gates = set(auto_gates or [])
        stages_executed: list[PipelineStageEvent] = []
        final_stage: ResearchStage = "intake"
        repairs_used = 0

        try:
            while len(stages_executed) < max_stages:
                snapshot = await self._service.get_snapshot(quest_id)
                quest = snapshot.quest
                final_stage = quest.stage
                if quest.status == "cancelled":
                    return PipelineRunResult(
                        quest_id=quest_id,
                        status="cancelled",
                        stages_executed=stages_executed,
                        final_stage=quest.stage,
                        message="Research pipeline stopped because the quest was cancelled.",
                    )
                if quest.stage == "final_bundle":
                    return PipelineRunResult(
                        quest_id=quest_id,
                        status="completed",
                        stages_executed=stages_executed,
                        final_stage=quest.stage,
                        message="Research pipeline is already at final_bundle.",
                    )

                next_stage = self._next_stage(quest.stage)
                if next_stage is None:
                    return PipelineRunResult(
                        quest_id=quest_id,
                        status="completed",
                        stages_executed=stages_executed,
                        final_stage=quest.stage,
                        message="Research pipeline has no remaining stages to run.",
                    )

                entered_at = now_iso()
                stage_inputs: dict[str, Any] = {}
                if next_stage == "manuscript_draft" and snapshot.quest.metadata.get("manuscript_section_concurrency"):
                    stage_inputs["section_concurrency"] = snapshot.quest.metadata["manuscript_section_concurrency"]
                result = await self._service.advance_quest(
                    quest_id,
                    inputs=stage_inputs,
                    content_generator=content_generator,
                    reviewer_generator=reviewer_generator,
                )
                if result.blocked and result.required_gate is not None:
                    gate = result.required_gate
                    if gate.gate_type == "final_quality_repair" and quality_mode == "auto_repair" and repairs_used < repair_budget:
                        repairs_used += 1
                        await self._service.attempt_quality_repair(quest_id)
                        continue
                    if gate.gate_type in allowed_gates:
                        await self._service.decide_gate(
                            quest_id,
                            stage=gate.stage,
                            gate_type=gate.gate_type,
                            status="approved",
                            reason="Auto-approved by run_pipeline.",
                        )
                        continue
                    final_stage = result.quest.stage
                    return PipelineRunResult(
                        quest_id=quest_id,
                        status="blocked_on_gate",
                        stages_executed=stages_executed,
                        final_stage=result.quest.stage,
                        blocked_gate=gate.gate_type,
                        message=f"Research pipeline is blocked on human gate {gate.gate_type}.",
                    )

                if not result.advanced:
                    final_stage = result.quest.stage
                    return PipelineRunResult(
                        quest_id=quest_id,
                        status="completed",
                        stages_executed=stages_executed,
                        final_stage=result.quest.stage,
                        message="Research pipeline has no remaining stages to run.",
                    )

                final_stage = result.quest.stage
                stages_executed.append(
                    PipelineStageEvent(
                        stage=result.quest.stage,
                        entered_at=entered_at,
                        completed_at=now_iso(),
                        outputs=result.generated,
                        artifacts=result.ledger_entry.artifacts if result.ledger_entry else [],
                    )
                )

                if delivery_mode == "fast_draft_first" and result.quest.stage == "manuscript_draft":
                    return PipelineRunResult(
                        quest_id=quest_id,
                        status="draft_ready",
                        stages_executed=stages_executed,
                        final_stage=result.quest.stage,
                        message="Research pipeline paused after fast manuscript draft.",
                    )

                if result.quest.stage == "final_bundle":
                    return PipelineRunResult(
                        quest_id=quest_id,
                        status="completed",
                        stages_executed=stages_executed,
                        final_stage=result.quest.stage,
                        message="Research pipeline completed through final_bundle.",
                    )
        except Exception as exc:
            return PipelineRunResult(
                quest_id=quest_id,
                status="error",
                stages_executed=stages_executed,
                final_stage=final_stage,
                error=str(exc),
                message=f"Research pipeline failed: {exc}",
            )

        return PipelineRunResult(
            quest_id=quest_id,
            status="stopped_at_max_stages",
            stages_executed=stages_executed,
            final_stage=final_stage,
            message=f"Research pipeline stopped after reaching max_stages={max_stages}.",
        )

    @staticmethod
    def _next_stage(stage: ResearchStage) -> ResearchStage | None:
        index = RESEARCH_STAGES.index(stage)
        if index >= len(RESEARCH_STAGES) - 1:
            return None
        return RESEARCH_STAGES[index + 1]
