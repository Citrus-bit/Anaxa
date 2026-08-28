from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, cast

from medrix_flow.academic.utils import detect_domain, slugify
from medrix_flow.runtime.utils import now_iso

from .quality import build_quality_audit
from .repository import ResearchRepository
from .types import (
    GATED_TRANSITIONS,
    RESEARCH_STAGES,
    ClaimEvidenceRecord,
    ExperimentBranchRecord,
    ManuscriptSectionRecord,
    NoveltyCheckRecord,
    OverlapRisk,
    ResearchAdvanceResult,
    ResearchGate,
    ResearchLedgerEntry,
    ResearchQualityAuditRecord,
    ResearchQuest,
    ResearchQuestSnapshot,
    ResearchStage,
    ReviewerReportRecord,
)

ContentGenerator = Callable[[str, ResearchQuestSnapshot], Awaitable[str]]
ReviewerGenerator = Callable[[str, ResearchQuestSnapshot], Awaitable[dict[str, Any]]]

_EMPIRICAL_METHOD_HINTS = {
    "difference-in-differences": "did",
    "staggered did": "staggered_did",
    "event study": "event_study",
    "event-study": "event_study",
    "instrumental variable": "iv",
    "regression discontinuity": "rdd",
    "synthetic control": "synthetic_control",
    "target trial": "target_trial",
    "causal forest": "causal_forest",
    "propensity score": "psm",
    "double machine learning": "dml",
    "survival": "survival",
    "tmle": "tmle",
    "did": "did",
    "iv": "iv",
    "rdd": "rdd",
    "psm": "psm",
    "ipw": "ipw",
    "dml": "dml",
}

_EMPIRICAL_DOMAIN_HINTS = {
    "applied economics",
    "econometric",
    "empirical",
    "finance",
    "management",
    "policy",
    "public health",
    "sociology",
}


class ResearchQuestService:
    def __init__(self, repository: ResearchRepository) -> None:
        self._repository = repository

    async def create_quest(
        self,
        *,
        thread_id: str,
        topic: str,
        title: str | None = None,
        scope: str | None = None,
        objective: str | None = None,
        domain: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchQuest:
        timestamp = now_iso()
        quest = ResearchQuest(
            quest_id=f"rq-{uuid.uuid4().hex[:12]}",
            thread_id=thread_id,
            title=(title or topic).strip(),
            topic=topic.strip(),
            scope=scope.strip() if scope else None,
            objective=objective.strip() if objective else None,
            domain=domain or self._detect_research_domain(topic, scope, objective, metadata),
            stage="intake",
            status="active",
            metadata=self._initial_metadata(topic=topic, scope=scope, objective=objective, metadata=metadata),
            created_at=timestamp,
            updated_at=timestamp,
        )
        created = await self._repository.create_quest(quest)
        await self._add_ledger(
            created,
            event_type="quest_created",
            summary="Research quest created and ready for staged execution.",
            inputs={"topic": topic, "scope": scope, "objective": objective},
            outputs={"stage": created.stage, "human_gates": list(GATED_TRANSITIONS.values())},
        )
        return created

    def _initial_metadata(
        self,
        *,
        topic: str,
        scope: str | None,
        objective: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(metadata or {})
        if self._is_empirical_research(topic, scope, objective, metadata):
            methods = self._infer_empirical_methods(topic, scope, objective, metadata)
            merged.setdefault("skill_guidance", []).append("empirical-research-methods")
            merged.setdefault("methodology_skill_path", "/mnt/skills/public/empirical-research-methods/SKILL.md")
            merged.setdefault("empirical_methods", methods)
            merged.setdefault(
                "experiment_metadata_contract",
                {
                    "required": [
                        "empirical_method",
                        "estimand",
                        "outcome",
                        "treatment_or_exposure",
                        "sample_restrictions",
                        "identification_assumptions",
                    ],
                    "conditional": [
                        "unit_id",
                        "time",
                        "treatment_time",
                        "instrument",
                        "running_variable",
                        "cutoff",
                        "covariates",
                        "fixed_effects",
                        "cluster",
                    ],
                },
            )
        return merged

    def _detect_research_domain(
        self,
        topic: str,
        scope: str | None,
        objective: str | None,
        metadata: dict[str, Any] | None,
    ) -> str:
        if self._is_empirical_research(topic, scope, objective, metadata):
            return "empirical_social_science"
        return detect_domain(topic, scope)

    def _is_empirical_research(
        self,
        topic: str,
        scope: str | None,
        objective: str | None,
        metadata: dict[str, Any] | None,
    ) -> bool:
        haystack = " ".join(
            [
                topic.lower(),
                (scope or "").lower(),
                (objective or "").lower(),
                " ".join(str(value).lower() for value in (metadata or {}).values()),
            ]
        )
        return any(hint in haystack for hint in _EMPIRICAL_DOMAIN_HINTS) or bool(self._infer_empirical_methods(topic, scope, objective, metadata))

    def _infer_empirical_methods(
        self,
        topic: str,
        scope: str | None,
        objective: str | None,
        metadata: dict[str, Any] | None,
    ) -> list[str]:
        haystack = " ".join(
            [
                topic.lower(),
                (scope or "").lower(),
                (objective or "").lower(),
                " ".join(str(value).lower() for value in (metadata or {}).values()),
            ]
        )
        return sorted({method for hint, method in _EMPIRICAL_METHOD_HINTS.items() if hint in haystack})

    async def list_quests(self, thread_id: str | None = None) -> list[ResearchQuest]:
        return await self._repository.list_quests(thread_id)

    async def get_snapshot(self, quest_id: str) -> ResearchQuestSnapshot:
        quest = await self._require_quest(quest_id)
        return ResearchQuestSnapshot(
            quest=quest,
            ledger=await self._repository.list_ledger(quest_id),
            gates=await self._repository.list_gates(quest_id),
            evidence=await self._repository.list_claim_evidence(quest_id),
            novelty_checks=await self._repository.list_novelty_checks(quest_id),
            experiment_branches=await self._repository.list_experiment_branches(quest_id),
            manuscript_sections=await self._repository.list_manuscript_sections(quest_id),
            reviewer_reports=await self._repository.list_reviewer_reports(quest_id),
            quality_audits=await self._repository.list_quality_audits(quest_id),
        )

    async def advance_quest(
        self,
        quest_id: str,
        *,
        target_stage: ResearchStage | None = None,
        inputs: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
        tool_name: str | None = None,
        model_name: str | None = None,
        content_generator: ContentGenerator | None = None,
        reviewer_generator: ReviewerGenerator | None = None,
    ) -> ResearchAdvanceResult:
        quest = await self._require_quest(quest_id)
        inputs = inputs or {}
        artifacts = artifacts or []
        next_stage = target_stage or self._next_stage(quest.stage)
        if next_stage is None:
            entry = await self._add_ledger(
                quest,
                event_type="already_complete",
                summary="Research quest is already at final_bundle.",
                inputs=inputs,
                outputs={"stage": quest.stage},
                artifacts=artifacts,
                tool_name=tool_name,
                model_name=model_name,
            )
            return ResearchAdvanceResult(quest=quest, advanced=False, ledger_entry=entry)

        self._validate_transition(quest.stage, next_stage)
        if next_stage == "experiment_planned" and not await self._novelty_allows_experiment(quest.quest_id):
            gate = await self._ensure_gate(quest, next_stage, "novelty_override")
            if gate.status != "approved":
                quest.status = "blocked"
                quest.updated_at = now_iso()
                quest = await self._repository.update_quest(quest)
                entry = await self._add_ledger(
                    quest,
                    stage=next_stage,
                    event_type="novelty_block",
                    summary="High overlap risk blocks experiment planning until a human override is recorded.",
                    inputs=inputs,
                    outputs={"gate_type": gate.gate_type, "gate_status": gate.status},
                    artifacts=artifacts,
                    tool_name=tool_name,
                    model_name=model_name,
                )
                return ResearchAdvanceResult(
                    quest=quest,
                    advanced=False,
                    blocked=True,
                    required_gate=gate,
                    ledger_entry=entry,
                    generated={"reason": "high_overlap_risk"},
                )
        gate_type = GATED_TRANSITIONS.get(next_stage)
        if next_stage == "final_bundle" and not await self._quality_allows_final_bundle(quest, inputs):
            gate = await self._ensure_gate(quest, next_stage, "final_quality_repair")
            if gate.status != "approved":
                quest.status = "blocked"
                quest.updated_at = now_iso()
                quest = await self._repository.update_quest(quest)
                latest_audit = (await self._repository.list_quality_audits(quest.quest_id))[-1]
                entry = await self._add_ledger(
                    quest,
                    stage=next_stage,
                    event_type="quality_repair_required",
                    summary="Final bundle is blocked because quality audit still requires repair.",
                    inputs=inputs,
                    outputs={
                        "gate_type": gate.gate_type,
                        "quality_audit_status": latest_audit.status,
                        "repair_actions": latest_audit.repair_actions,
                        "recommended_queries": latest_audit.recommended_queries,
                    },
                    artifacts=artifacts,
                    tool_name=tool_name,
                    model_name=model_name,
                )
                return ResearchAdvanceResult(
                    quest=quest,
                    advanced=False,
                    blocked=True,
                    required_gate=gate,
                    ledger_entry=entry,
                    generated={
                        "reason": "quality_repair_required",
                        "quality_audit_status": latest_audit.status,
                        "repair_actions": latest_audit.repair_actions,
                        "recommended_queries": latest_audit.recommended_queries,
                    },
                )
        if gate_type:
            gate = await self._ensure_gate(quest, next_stage, gate_type)
            if gate.status != "approved":
                quest.status = "blocked"
                quest.updated_at = now_iso()
                quest = await self._repository.update_quest(quest)
                entry = await self._add_ledger(
                    quest,
                    stage=next_stage,
                    event_type="gate_required",
                    summary=f"Human approval is required before entering {next_stage}.",
                    inputs=inputs,
                    outputs={"gate_type": gate.gate_type, "gate_status": gate.status},
                    artifacts=artifacts,
                    tool_name=tool_name,
                    model_name=model_name,
                )
                return ResearchAdvanceResult(
                    quest=quest,
                    advanced=False,
                    blocked=True,
                    required_gate=gate,
                    ledger_entry=entry,
                    generated={"reason": "human_gate_required"},
                )

        generated = await self._apply_stage(
            quest,
            next_stage,
            inputs,
            artifacts,
            content_generator=content_generator,
            reviewer_generator=reviewer_generator,
        )
        quest.stage = next_stage
        quest.status = "completed" if next_stage == "final_bundle" else "active"
        quest.updated_at = now_iso()
        quest = await self._repository.update_quest(quest)
        entry = await self._add_ledger(
            quest,
            stage=next_stage,
            event_type="stage_advanced",
            summary=f"Research quest advanced to {next_stage}.",
            inputs=inputs,
            outputs=generated,
            artifacts=artifacts,
            tool_name=tool_name,
            model_name=model_name,
        )
        return ResearchAdvanceResult(
            quest=quest,
            advanced=True,
            ledger_entry=entry,
            generated=generated,
        )

    async def decide_gate(
        self,
        quest_id: str,
        *,
        stage: ResearchStage,
        gate_type: str,
        status: str,
        decision: str | None = None,
        reason: str | None = None,
    ) -> ResearchGate:
        quest = await self._require_quest(quest_id)
        if status not in {"approved", "rejected", "pending"}:
            raise ValueError("Gate status must be approved, rejected, or pending.")
        existing = await self._repository.get_gate(quest_id, stage, gate_type)
        timestamp = now_iso()
        gate = ResearchGate(
            gate_id=existing.gate_id if existing else f"gate-{uuid.uuid4().hex[:12]}",
            quest_id=quest_id,
            stage=stage,
            gate_type=gate_type,
            status=cast(Any, status),
            decision=decision,
            reason=reason,
            required=True,
            created_at=existing.created_at if existing else timestamp,
            decided_at=timestamp if status in {"approved", "rejected"} else None,
        )
        stored = await self._repository.upsert_gate(gate)
        quest.status = "active" if status == "approved" else "blocked"
        quest.updated_at = timestamp
        await self._repository.update_quest(quest)
        await self._add_ledger(
            quest,
            stage=stage,
            event_type="gate_decision",
            summary=f"Human gate {gate_type} marked {status}.",
            inputs={"gate_type": gate_type, "decision": decision, "reason": reason},
            outputs={"status": status},
            gate_decision=status,
        )
        return stored

    async def list_evidence(self, quest_id: str) -> list[ClaimEvidenceRecord]:
        await self._require_quest(quest_id)
        return await self._repository.list_claim_evidence(quest_id)

    async def list_experiment_branches(self, quest_id: str) -> list[ExperimentBranchRecord]:
        await self._require_quest(quest_id)
        return await self._repository.list_experiment_branches(quest_id)

    async def list_manuscript_sections(self, quest_id: str) -> list[ManuscriptSectionRecord]:
        await self._require_quest(quest_id)
        return await self._repository.list_manuscript_sections(quest_id)

    async def list_quality_audits(self, quest_id: str) -> list[ResearchQualityAuditRecord]:
        await self._require_quest(quest_id)
        return await self._repository.list_quality_audits(quest_id)

    async def _apply_stage(
        self,
        quest: ResearchQuest,
        stage: ResearchStage,
        inputs: dict[str, Any],
        artifacts: list[str],
        *,
        content_generator: ContentGenerator | None = None,
        reviewer_generator: ReviewerGenerator | None = None,
    ) -> dict[str, Any]:
        if stage == "literature":
            return await self._stage_literature(quest, inputs)
        if stage == "novelty_check":
            return await self._stage_novelty_check(quest, inputs)
        if stage == "evidence_verified":
            return await self._stage_evidence_verified(quest, inputs)
        if stage == "experiment_planned":
            return await self._stage_experiment_planned(quest, inputs)
        if stage == "experiment_running":
            return await self._stage_experiment_running(quest, inputs, artifacts)
        if stage == "results_synthesized":
            return await self._stage_results_synthesized(quest, inputs, artifacts)
        if stage == "manuscript_draft":
            return await self._stage_manuscript_draft(quest, inputs, artifacts, content_generator=content_generator)
        if stage == "review":
            return await self._stage_review(quest, inputs)
        if stage == "revision":
            return await self._stage_revision(quest, inputs)
        if stage == "final_bundle":
            return await self._stage_final_bundle(quest, artifacts)
        return {"stage": stage}

    async def _stage_literature(self, quest: ResearchQuest, inputs: dict[str, Any]) -> dict[str, Any]:
        academic_project_id = inputs.get("academic_project_id")
        if isinstance(academic_project_id, str) and academic_project_id:
            quest.academic_project_id = academic_project_id
            quest.metadata = {**quest.metadata, "academic_project_id": academic_project_id}
            await self._repository.update_quest(quest)

        created_claims = 0
        for claim in self._normalize_claims(inputs.get("claims")):
            await self._repository.add_claim_evidence(
                ClaimEvidenceRecord(
                    claim_id=f"claim-{uuid.uuid4().hex[:12]}",
                    quest_id=quest.quest_id,
                    claim=claim["claim"],
                    paper_id=claim.get("paper_id"),
                    source_title=claim.get("source_title"),
                    locator=claim.get("locator"),
                    snippet=claim.get("snippet"),
                    quote=claim.get("quote"),
                    support_status=claim.get("support_status", "uncertain"),
                    confidence=float(claim.get("confidence", 0.4)),
                    artifact_path=claim.get("artifact_path"),
                    metadata=claim.get("metadata", {}),
                    created_at=now_iso(),
                )
            )
            created_claims += 1

        if created_claims == 0:
            await self._repository.add_claim_evidence(
                ClaimEvidenceRecord(
                    claim_id=f"claim-{uuid.uuid4().hex[:12]}",
                    quest_id=quest.quest_id,
                    claim=f"The research question focuses on {quest.topic}.",
                    support_status="uncertain",
                    confidence=0.2,
                    metadata={"source": "intake-placeholder", "needs_full_text_verification": True},
                    created_at=now_iso(),
                )
            )
            created_claims = 1

        return {
            "academic_project_id": quest.academic_project_id,
            "claim_count": created_claims,
            "next_required": "Run or attach academic_research outputs for paper-backed evidence.",
        }

    async def _stage_novelty_check(self, quest: ResearchQuest, inputs: dict[str, Any]) -> dict[str, Any]:
        idea = str(inputs.get("idea") or quest.objective or quest.topic)
        closest_papers = self._normalize_dict_list(inputs.get("closest_papers"))
        hypotheses = [str(item).strip() for item in inputs.get("hypotheses", []) if str(item).strip()]
        if not hypotheses:
            hypotheses = [f"Test whether the proposed approach improves a measurable outcome for {quest.topic}."]
        risk = self._estimate_overlap_risk(idea, closest_papers, inputs.get("overlap_risk"))
        decision = "revise" if risk == "high" else "proceed"
        record = NoveltyCheckRecord(
            check_id=f"novelty-{uuid.uuid4().hex[:12]}",
            quest_id=quest.quest_id,
            idea=idea,
            overlap_risk=risk,
            closest_papers=closest_papers,
            hypotheses=hypotheses,
            minimum_experiment=str(inputs.get("minimum_experiment") or "Define one baseline, one ablation, fixed seeds, and a primary metric before execution."),
            decision=decision,
            created_at=now_iso(),
        )
        await self._repository.add_novelty_check(record)
        return {
            "overlap_risk": risk,
            "decision": decision,
            "hypothesis_count": len(hypotheses),
            "closest_paper_count": len(closest_papers),
        }

    async def _stage_evidence_verified(self, quest: ResearchQuest, inputs: dict[str, Any]) -> dict[str, Any]:
        evidence = await self._repository.list_claim_evidence(quest.quest_id)
        if not evidence:
            await self._repository.add_claim_evidence(
                ClaimEvidenceRecord(
                    claim_id=f"claim-{uuid.uuid4().hex[:12]}",
                    quest_id=quest.quest_id,
                    claim=f"No source-backed claim has been attached for {quest.topic}.",
                    support_status="unsupported",
                    confidence=0.0,
                    metadata={"source": "evidence-integrity-gate"},
                    created_at=now_iso(),
                )
            )
            evidence = await self._repository.list_claim_evidence(quest.quest_id)
        counts: dict[str, int] = {}
        for item in evidence:
            counts[item.support_status] = counts.get(item.support_status, 0) + 1
        return {
            "claim_count": len(evidence),
            "support_status_counts": counts,
            "strict_rule": "Manuscript claims must remain linked to evidence or be marked unsupported.",
        }

    async def _stage_experiment_planned(self, quest: ResearchQuest, inputs: dict[str, Any]) -> dict[str, Any]:
        branch_payloads = self._normalize_dict_list(inputs.get("branches"))
        if not branch_payloads:
            metadata = {"budget_policy": "human approval required before execution"}
            if quest.domain == "empirical_social_science":
                metadata.update(
                    {
                        "skill_guidance": "empirical-research-methods",
                        "methodology_skill_path": "/mnt/skills/public/empirical-research-methods/SKILL.md",
                        "empirical_methods": quest.metadata.get("empirical_methods", []),
                        "identification_gate": "required before causal claims",
                        "experiment_lab_metadata_required": quest.metadata.get("experiment_metadata_contract", {}),
                    }
                )
            branch_payloads = [
                {
                    "name": "Baseline protocol",
                    "branch_type": "baseline",
                    "priority": 1.0,
                    "seed": 42,
                    "metadata": metadata,
                }
            ]
        created = 0
        for payload in branch_payloads:
            await self._repository.upsert_experiment_branch(
                ExperimentBranchRecord(
                    branch_id=str(payload.get("branch_id") or f"branch-{uuid.uuid4().hex[:12]}"),
                    quest_id=quest.quest_id,
                    experiment_project_id=payload.get("experiment_project_id"),
                    parent_branch_id=payload.get("parent_branch_id"),
                    name=str(payload.get("name") or "Experiment branch"),
                    branch_type=str(payload.get("branch_type") or "baseline"),
                    status=payload.get("status", "planned"),
                    priority=float(payload.get("priority", 0.0)),
                    seed=payload.get("seed"),
                    metrics=payload.get("metrics", {}),
                    artifact_paths=payload.get("artifact_paths", []),
                    failure_summary=payload.get("failure_summary"),
                    metadata=payload.get("metadata", {}),
                    created_at=now_iso(),
                    updated_at=now_iso(),
                )
            )
            created += 1
        await self._ensure_gate(quest, "experiment_running", "experiment_execution")
        return {"branch_count": created, "execution_gate": "experiment_execution"}

    async def _stage_experiment_running(
        self,
        quest: ResearchQuest,
        inputs: dict[str, Any],
        artifacts: list[str],
    ) -> dict[str, Any]:
        experiment_project_id = inputs.get("experiment_project_id")
        if isinstance(experiment_project_id, str) and experiment_project_id:
            ids = [*quest.experiment_project_ids]
            if experiment_project_id not in ids:
                ids.append(experiment_project_id)
            quest.experiment_project_ids = ids
            await self._repository.update_quest(quest)

        branches = await self._repository.list_experiment_branches(quest.quest_id)
        if branches:
            branch = branches[0]
            branch.status = "running" if not experiment_project_id else "completed"
            branch.experiment_project_id = experiment_project_id or branch.experiment_project_id
            branch.artifact_paths = sorted(set(branch.artifact_paths + artifacts))
            branch.updated_at = now_iso()
            await self._repository.upsert_experiment_branch(branch)
        return {
            "experiment_project_ids": quest.experiment_project_ids,
            "branch_status": "completed" if experiment_project_id else "running",
            "artifact_count": len(artifacts),
        }

    async def _stage_results_synthesized(
        self,
        quest: ResearchQuest,
        inputs: dict[str, Any],
        artifacts: list[str],
    ) -> dict[str, Any]:
        metrics = inputs.get("metrics") if isinstance(inputs.get("metrics"), dict) else {}
        branches = await self._repository.list_experiment_branches(quest.quest_id)
        for branch in branches:
            if metrics:
                branch.metrics = {**branch.metrics, **metrics}
            if artifacts:
                branch.artifact_paths = sorted(set(branch.artifact_paths + artifacts))
            if branch.status == "running":
                branch.status = "completed"
            branch.updated_at = now_iso()
            await self._repository.upsert_experiment_branch(branch)
        branches = await self._repository.list_experiment_branches(quest.quest_id)
        hypotheses = await self._collect_hypotheses(quest.quest_id, inputs)
        return {
            "branch_count": len(branches),
            "metric_keys": sorted(metrics),
            "artifact_count": len(artifacts),
            "hypothesis_outcomes": [self._evaluate_hypothesis(hypothesis, branches, metrics) for hypothesis in hypotheses],
            "significance_summary": self._summarize_significance(metrics),
        }

    async def _stage_manuscript_draft(
        self,
        quest: ResearchQuest,
        inputs: dict[str, Any],
        artifacts: list[str],
        content_generator: ContentGenerator | None = None,
    ) -> dict[str, Any]:
        sections = self._normalize_dict_list(inputs.get("sections"))
        evidence = await self._repository.list_claim_evidence(quest.quest_id)
        claim_ids = [item.claim_id for item in evidence]
        content_generation_errors: list[dict[str, str]] = []
        if not sections:
            sections = [
                {"section_key": "introduction", "title": "Introduction"},
                {"section_key": "related_work", "title": "Related Work"},
                {"section_key": "methods", "title": "Methods"},
                {"section_key": "results", "title": "Results"},
                {"section_key": "limitations", "title": "Limitations"},
            ]

        normalized_sections: list[dict[str, Any]] = []
        for payload in sections:
            normalized_sections.append(
                {
                    **payload,
                    "section_key": str(payload.get("section_key") or slugify(str(payload.get("title") or "section"))),
                    "title": str(payload.get("title") or "Section"),
                    "content": str(payload.get("content") or ""),
                }
            )

        concurrency = self._resolve_section_concurrency(quest, inputs)
        generated_contents: dict[str, str] = {}
        sections_to_generate = [payload for payload in normalized_sections if not payload["content"] and content_generator is not None]
        if sections_to_generate and content_generator is not None:
            snapshot = await self.get_snapshot(quest.quest_id)
            semaphore = asyncio.Semaphore(concurrency)

            async def generate_section(payload: dict[str, Any]) -> tuple[str, str | None, str | None]:
                section_key = str(payload["section_key"])
                try:
                    async with semaphore:
                        generated_content = (await content_generator(section_key, snapshot)).strip()
                    if not generated_content:
                        return section_key, None, "content_generator returned empty content"
                    return section_key, generated_content, None
                except Exception as exc:
                    return section_key, None, str(exc)

            generated = await asyncio.gather(*(generate_section(payload) for payload in sections_to_generate))
            for section_key, generated_content, error in generated:
                if generated_content:
                    generated_contents[section_key] = generated_content
                elif error:
                    content_generation_errors.append({"section_key": section_key, "error": error})

        now = now_iso()
        draft_status = str(inputs.get("draft_status") or "draft_ready")
        draft_artifacts = [artifact for artifact in artifacts if artifact]
        for payload in normalized_sections:
            section_key = str(payload["section_key"])
            content = str(payload["content"] or generated_contents.get(section_key) or "")
            await self._repository.upsert_manuscript_section(
                ManuscriptSectionRecord(
                    section_id=str(payload.get("section_id") or f"section-{uuid.uuid4().hex[:12]}"),
                    quest_id=quest.quest_id,
                    section_key=section_key,
                    title=str(payload["title"]),
                    content=content,
                    claim_ids=payload.get("claim_ids", claim_ids),
                    artifact_paths=payload.get("artifact_paths", draft_artifacts),
                    status=str(payload.get("status") or draft_status),
                    created_at=now,
                    updated_at=now,
                )
            )

        quest.metadata = {
            **quest.metadata,
            "draft_ready_at": now,
            "draft_status": draft_status,
            "manuscript_section_concurrency": concurrency,
        }
        await self._repository.update_quest(quest)
        await self._ensure_gate(quest, "review", "pre_review")
        return {
            "section_count": len(normalized_sections),
            "linked_claim_count": len(claim_ids),
            "review_gate": "pre_review",
            "draft_status": draft_status,
            "section_generation_mode": "parallel" if sections_to_generate else "provided",
            "section_generation_concurrency": concurrency,
            "section_generation_errors": content_generation_errors,
            "content_generation_errors": content_generation_errors,
            "draft_artifacts": draft_artifacts,
        }

    async def _stage_review(self, quest: ResearchQuest, inputs: dict[str, Any]) -> dict[str, Any]:
        evidence = await self._repository.list_claim_evidence(quest.quest_id)
        sections = await self._repository.list_manuscript_sections(quest.quest_id)
        unsupported = len([item for item in evidence if item.support_status == "unsupported"])
        audit = await self._create_quality_audit(quest, "review", inputs, evidence=evidence, sections=sections)
        profiles = [
            "literature-coverage",
            "evidence-quantitative",
            "citation-integrity",
            "argument-validity",
            "writing-style",
            "devils-advocate",
        ]
        created = 0
        for profile in profiles:
            score = self._review_score(profile, evidence, sections, unsupported, audit)
            verdict = "block" if score < 0.45 else "revise" if score < 0.75 else "pass"
            await self._repository.add_reviewer_report(
                ReviewerReportRecord(
                    report_id=f"review-{uuid.uuid4().hex[:12]}",
                    quest_id=quest.quest_id,
                    stage="review",
                    reviewer_profile=profile,
                    score=score,
                    verdict=verdict,
                    findings=self._review_findings(profile, unsupported, len(sections), audit),
                    required_actions=self._review_actions(profile, verdict, audit),
                    created_at=now_iso(),
                )
            )
            created += 1
        await self._ensure_gate(quest, "final_bundle", "final_release")
        return {
            "reviewer_count": created,
            "unsupported_claims": unsupported,
            "quality_audit_status": audit.status,
            "quality_score": audit.score,
            "repair_action_count": len(audit.repair_actions),
            "final_gate": "final_release",
        }

    async def _stage_revision(self, quest: ResearchQuest, inputs: dict[str, Any]) -> dict[str, Any]:
        reports = await self._repository.list_reviewer_reports(quest.quest_id)
        required_actions = [action for report in reports for action in report.required_actions]
        completed_actions = [str(item) for item in inputs.get("completed_actions", [])]
        return {
            "required_action_count": len(required_actions),
            "completed_action_count": len(completed_actions),
            "remaining_action_count": max(len(required_actions) - len(completed_actions), 0),
        }

    async def _stage_final_bundle(self, quest: ResearchQuest, artifacts: list[str]) -> dict[str, Any]:
        snapshot = await self.get_snapshot(quest.quest_id)
        latest_audit = snapshot.quality_audits[-1] if snapshot.quality_audits else None
        if latest_audit is None:
            latest_audit = await self._create_quality_audit(
                quest,
                "final_bundle",
                {},
                evidence=snapshot.evidence,
                sections=snapshot.manuscript_sections,
            )
        return {
            "artifact_count": len(artifacts),
            "claim_count": len(snapshot.evidence),
            "branch_count": len(snapshot.experiment_branches),
            "reviewer_count": len(snapshot.reviewer_reports),
            "quality_audit_status": latest_audit.status,
            "quality_score": latest_audit.score,
            "repair_actions": latest_audit.repair_actions,
            "bundle_policy": "Final output was released after human approval.",
        }

    async def _require_quest(self, quest_id: str) -> ResearchQuest:
        quest = await self._repository.get_quest(quest_id)
        if quest is None:
            raise ValueError(f"Research quest {quest_id} not found")
        return quest

    async def _add_ledger(
        self,
        quest: ResearchQuest,
        *,
        event_type: str,
        summary: str,
        stage: ResearchStage | None = None,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
        tool_name: str | None = None,
        model_name: str | None = None,
        error: str | None = None,
        gate_decision: str | None = None,
    ) -> ResearchLedgerEntry:
        entry = ResearchLedgerEntry(
            entry_id=f"ledger-{uuid.uuid4().hex[:12]}",
            quest_id=quest.quest_id,
            stage=stage or quest.stage,
            event_type=event_type,
            summary=summary,
            inputs=inputs or {},
            outputs=outputs or {},
            artifacts=artifacts or [],
            tool_name=tool_name,
            model_name=model_name,
            error=error,
            gate_decision=gate_decision,
            created_at=now_iso(),
        )
        return await self._repository.add_ledger_entry(entry)

    async def _ensure_gate(self, quest: ResearchQuest, stage: ResearchStage, gate_type: str) -> ResearchGate:
        existing = await self._repository.get_gate(quest.quest_id, stage, gate_type)
        if existing is not None:
            return existing
        return await self._repository.upsert_gate(
            ResearchGate(
                gate_id=f"gate-{uuid.uuid4().hex[:12]}",
                quest_id=quest.quest_id,
                stage=stage,
                gate_type=gate_type,
                status="pending",
                required=True,
                created_at=now_iso(),
            )
        )

    async def _create_quality_audit(
        self,
        quest: ResearchQuest,
        stage: ResearchStage,
        inputs: dict[str, Any],
        *,
        evidence: list[ClaimEvidenceRecord] | None = None,
        sections: list[ManuscriptSectionRecord] | None = None,
    ) -> ResearchQualityAuditRecord:
        evidence = evidence if evidence is not None else await self._repository.list_claim_evidence(quest.quest_id)
        sections = sections if sections is not None else await self._repository.list_manuscript_sections(quest.quest_id)
        quality = dict(quest.metadata.get("quality") or {})
        quality.update(inputs.get("quality") or {})
        min_reference_count = int(quality.get("min_reference_count") or inputs.get("min_reference_count") or 50)
        min_cited_reference_count = int(quality.get("min_cited_reference_count") or inputs.get("min_cited_reference_count") or 30)
        required_topics = inputs.get("required_topics") or quality.get("required_topics") or quest.metadata.get("required_topics") or []
        reference_count = inputs.get("reference_count") or quest.metadata.get("reference_count")
        cited_reference_count = inputs.get("cited_reference_count") or quest.metadata.get("cited_reference_count")
        audit = build_quality_audit(
            audit_id=f"qa-{uuid.uuid4().hex[:12]}",
            quest_id=quest.quest_id,
            stage=stage,
            topic=quest.topic,
            evidence=evidence,
            sections=sections,
            reference_count=int(reference_count) if reference_count is not None else None,
            cited_reference_count=int(cited_reference_count) if cited_reference_count is not None else None,
            min_reference_count=min_reference_count,
            min_cited_reference_count=min_cited_reference_count,
            required_topics=list(required_topics) if isinstance(required_topics, list) else [],
            created_at=now_iso(),
        )
        await self._repository.add_quality_audit(audit)
        await self._add_ledger(
            quest,
            stage=stage,
            event_type="quality_audit",
            summary=f"Research quality audit returned {audit.status}.",
            inputs={"quality": quality, "required_topics": required_topics},
            outputs={
                "status": audit.status,
                "score": audit.score,
                "metrics": audit.metrics,
                "repair_actions": audit.repair_actions,
                "recommended_queries": audit.recommended_queries,
            },
        )
        return audit

    async def _novelty_allows_experiment(self, quest_id: str) -> bool:
        checks = await self._repository.list_novelty_checks(quest_id)
        if not checks:
            return True
        latest = checks[-1]
        return latest.overlap_risk != "high" and latest.decision == "proceed"

    async def _quality_allows_final_bundle(self, quest: ResearchQuest, inputs: dict[str, Any]) -> bool:
        quality_mode = str(inputs.get("quality_mode") or quest.metadata.get("quality_mode") or "auto_repair")
        if quality_mode == "audit_only":
            return True
        audits = await self._repository.list_quality_audits(quest.quest_id)
        latest = audits[-1] if audits else await self._create_quality_audit(quest, "final_bundle", inputs)
        if latest.status == "pass":
            return True
        if quality_mode == "strict_gate":
            return False
        repair_attempts = len([entry for entry in await self._repository.list_ledger(quest.quest_id) if entry.event_type == "quality_repair_attempted"])
        max_repairs = int(quest.metadata.get("max_quality_repair_attempts", 2))
        return repair_attempts >= max_repairs

    async def attempt_quality_repair(self, quest_id: str) -> ResearchQualityAuditRecord:
        quest = await self._require_quest(quest_id)
        prior_audits = await self._repository.list_quality_audits(quest_id)
        latest = prior_audits[-1] if prior_audits else await self._create_quality_audit(quest, "final_bundle", {})
        sections = await self._repository.list_manuscript_sections(quest_id)
        evidence = await self._repository.list_claim_evidence(quest_id)
        timestamp = now_iso()
        repair_metadata = {
            "source_audit_id": latest.audit_id,
            "source_status": latest.status,
            "repair_actions": latest.repair_actions,
            "recommended_queries": latest.recommended_queries,
        }

        if latest.metrics.get("has_feasibility_section") is False and sections:
            await self._repository.upsert_manuscript_section(
                ManuscriptSectionRecord(
                    section_id=f"section-{uuid.uuid4().hex[:12]}",
                    quest_id=quest_id,
                    section_key="implementation_challenges_mitigations",
                    title="Implementation Challenges and Mitigations",
                    content=("Implementation challenges should be treated as first-class evidence targets. The next revision must bind each challenge to specific papers, benchmarks, datasets, or experiment artifacts before final release."),
                    claim_ids=[item.claim_id for item in evidence],
                    artifact_paths=[],
                    status="draft-repair",
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )
        if sections:
            for section in sections:
                repaired_content = self._repair_manuscript_style(section.content)
                if repaired_content != section.content:
                    section.content = repaired_content
                    section.status = "draft-repair"
                    section.updated_at = timestamp
                    await self._repository.upsert_manuscript_section(section)

        repaired_audit = await self._create_quality_audit(
            quest,
            "final_bundle",
            {},
            evidence=await self._repository.list_claim_evidence(quest_id),
            sections=await self._repository.list_manuscript_sections(quest_id),
        )
        await self._add_ledger(
            quest,
            stage="final_bundle",
            event_type="quality_repair_attempted",
            summary="Automatic quality repair attempted before final release.",
            inputs=repair_metadata,
            outputs={
                "new_audit_id": repaired_audit.audit_id,
                "new_status": repaired_audit.status,
                "remaining_repair_actions": repaired_audit.repair_actions,
            },
        )
        return repaired_audit

    @staticmethod
    def _repair_manuscript_style(content: str) -> str:
        repaired = content
        replacements = {
            "bibliography keys are synchronized": "",
            "citation keys are synchronized": "",
            "are not enough": "are insufficient in this setting",
            "is not enough": "is insufficient in this setting",
            "must support": "should support",
            "must provide": "should provide",
            "always": "often",
            "never": "rarely",
            "universally": "in many settings",
            "without exception": "in the evaluated settings",
        }
        for needle, replacement in replacements.items():
            repaired = re.sub(re.escape(needle), replacement, repaired, flags=re.IGNORECASE)
        return re.sub(r"[ \t]+\n", "\n", repaired).strip()

    @staticmethod
    def _next_stage(stage: ResearchStage) -> ResearchStage | None:
        index = RESEARCH_STAGES.index(stage)
        if index >= len(RESEARCH_STAGES) - 1:
            return None
        return RESEARCH_STAGES[index + 1]

    @staticmethod
    def _validate_transition(current: ResearchStage, target: ResearchStage) -> None:
        current_index = RESEARCH_STAGES.index(current)
        target_index = RESEARCH_STAGES.index(target)
        if target_index != current_index + 1:
            raise ValueError(f"Invalid research stage transition: {current} -> {target}")

    @staticmethod
    def _normalize_claims(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        claims: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                claims.append({"claim": item.strip()})
            elif isinstance(item, dict) and str(item.get("claim", "")).strip():
                claims.append({**item, "claim": str(item["claim"]).strip()})
        return claims

    @staticmethod
    def _normalize_dict_list(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    @staticmethod
    def _resolve_section_concurrency(quest: ResearchQuest, inputs: dict[str, Any]) -> int:
        value = inputs.get("section_concurrency") or quest.metadata.get("manuscript_section_concurrency") or 3
        try:
            return max(1, min(int(value), 12))
        except (TypeError, ValueError):
            return 3

    async def _collect_hypotheses(self, quest_id: str, inputs: dict[str, Any]) -> list[dict[str, Any]]:
        hypotheses: list[dict[str, Any]] = []
        for index, payload in enumerate(self._normalize_dict_list(inputs.get("hypotheses"))):
            statement = str(payload.get("statement") or payload.get("hypothesis") or "").strip()
            if statement:
                hypotheses.append(
                    {
                        "hypothesis_id": str(payload.get("hypothesis_id") or payload.get("id") or f"input:{index}"),
                        "statement": statement,
                        "primary_metric": payload.get("primary_metric"),
                        "baseline": payload.get("baseline"),
                        "direction": payload.get("direction"),
                    }
                )

        novelty_checks = await self._repository.list_novelty_checks(quest_id)
        for check in novelty_checks:
            for index, statement in enumerate(check.hypotheses):
                text = str(statement).strip()
                if text:
                    hypotheses.append(
                        {
                            "hypothesis_id": f"{check.check_id}:{index}",
                            "statement": text,
                            "primary_metric": None,
                            "baseline": None,
                            "direction": None,
                        }
                    )
        return hypotheses

    @classmethod
    def _evaluate_hypothesis(
        cls,
        hypothesis: dict[str, Any],
        branches: list[ExperimentBranchRecord],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        primary_metric = hypothesis.get("primary_metric")
        if not isinstance(primary_metric, str) or not primary_metric:
            return {
                "hypothesis_id": hypothesis["hypothesis_id"],
                "statement": hypothesis["statement"],
                "outcome": "inconclusive",
                "evidence_metric_keys": [],
            }

        evidence_values = cls._metric_values(primary_metric, branches, metrics)
        baseline = cls._coerce_float(hypothesis.get("baseline"))
        if baseline is None:
            baseline = cls._baseline_metric_value(primary_metric, branches, metrics)
        if not evidence_values or baseline is None:
            outcome = "inconclusive"
        else:
            best_value = max(evidence_values)
            worst_value = min(evidence_values)
            direction = str(hypothesis.get("direction") or "increase").lower()
            if direction in {"decrease", "lower", "less", "minimize", "<", "<="}:
                outcome = "supported" if worst_value <= baseline else "refuted"
            else:
                outcome = "supported" if best_value >= baseline else "refuted"

        return {
            "hypothesis_id": hypothesis["hypothesis_id"],
            "statement": hypothesis["statement"],
            "outcome": outcome,
            "evidence_metric_keys": [primary_metric],
        }

    @classmethod
    def _metric_values(
        cls,
        primary_metric: str,
        branches: list[ExperimentBranchRecord],
        metrics: dict[str, Any],
    ) -> list[float]:
        values: list[float] = []
        direct = cls._coerce_float(metrics.get(primary_metric))
        if direct is not None:
            values.append(direct)
        for branch in branches:
            value = cls._coerce_float(branch.metrics.get(primary_metric))
            if value is not None and branch.branch_type != "baseline":
                values.append(value)
        if values:
            return values
        for branch in branches:
            value = cls._coerce_float(branch.metrics.get(primary_metric))
            if value is not None:
                values.append(value)
        return values

    @classmethod
    def _baseline_metric_value(
        cls,
        primary_metric: str,
        branches: list[ExperimentBranchRecord],
        metrics: dict[str, Any],
    ) -> float | None:
        baseline_metrics = metrics.get("baseline")
        if isinstance(baseline_metrics, dict):
            value = cls._coerce_float(baseline_metrics.get(primary_metric))
            if value is not None:
                return value
        baseline_key = f"baseline_{primary_metric}"
        value = cls._coerce_float(metrics.get(baseline_key))
        if value is not None:
            return value
        for branch in branches:
            if branch.branch_type == "baseline":
                value = cls._coerce_float(branch.metrics.get(primary_metric))
                if value is not None:
                    return value
                value = cls._coerce_float(branch.metadata.get(primary_metric))
                if value is not None:
                    return value
        return None

    @staticmethod
    def _summarize_significance(metrics: dict[str, Any]) -> dict[str, Any]:
        alpha = 0.05
        p_value_keys: list[str] = []
        ci_metric_roots: set[str] = set()
        significant_keys: set[str] = set()
        for key, value in metrics.items():
            if key.endswith(("_p_value", "_p")):
                p_value = ResearchQuestService._coerce_float(value)
                p_value_keys.append(key)
                if p_value is not None and p_value < alpha:
                    significant_keys.add(key)
            elif key.endswith("_ci_lower"):
                ci_metric_roots.add(key.removesuffix("_ci_lower"))
            elif key.endswith("_ci_upper"):
                ci_metric_roots.add(key.removesuffix("_ci_upper"))

        for root in ci_metric_roots:
            lower = ResearchQuestService._coerce_float(metrics.get(f"{root}_ci_lower"))
            upper = ResearchQuestService._coerce_float(metrics.get(f"{root}_ci_upper"))
            if lower is not None and upper is not None and (lower > 0 or upper < 0):
                significant_keys.add(f"{root}_ci")

        return {
            "has_significance_data": bool(p_value_keys or ci_metric_roots),
            "significant_keys": sorted(significant_keys),
            "alpha": alpha,
        }

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            if value is None or isinstance(value, bool):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _estimate_overlap_risk(idea: str, closest_papers: list[dict[str, Any]], requested: Any) -> OverlapRisk:
        if requested in {"low", "medium", "high"}:
            return cast(OverlapRisk, requested)
        idea_terms = {term for term in re.findall(r"[a-zA-Z0-9]+", idea.lower()) if len(term) > 4}
        if not closest_papers:
            return "medium"
        max_overlap = 0.0
        for paper in closest_papers:
            title = str(paper.get("title") or "").lower()
            title_terms = {term for term in re.findall(r"[a-zA-Z0-9]+", title) if len(term) > 4}
            if idea_terms:
                max_overlap = max(max_overlap, len(idea_terms & title_terms) / len(idea_terms))
        if max_overlap >= 0.5:
            return "high"
        if max_overlap >= 0.2:
            return "medium"
        return "low"

    @staticmethod
    def _review_score(
        profile: str,
        evidence: list[ClaimEvidenceRecord],
        sections: list[ManuscriptSectionRecord],
        unsupported_count: int,
        audit: ResearchQualityAuditRecord | None = None,
    ) -> float:
        base = 0.7
        if not evidence:
            base -= 0.25
        if not sections:
            base -= 0.2
        if unsupported_count:
            base -= min(0.3, unsupported_count * 0.1)
        if profile == "citation-integrity" and unsupported_count:
            base -= 0.15
        if audit is not None:
            base = min(base, audit.score + 0.1)
            if profile == "literature-coverage" and audit.metrics.get("reference_count", 0) < audit.metrics.get("min_reference_count", 0):
                base -= 0.2
            if profile == "evidence-quantitative" and not audit.metrics.get("quantitative_evidence_count"):
                base -= 0.2
            if profile == "writing-style" and (audit.metrics.get("repeated_phrase_count", 0) or audit.metrics.get("absolute_phrase_count", 0) or audit.metrics.get("author_note_count", 0)):
                base -= 0.2
        if profile == "devils-advocate":
            base -= 0.05
        return max(0.0, min(1.0, base))

    @staticmethod
    def _review_findings(
        profile: str,
        unsupported_count: int,
        section_count: int,
        audit: ResearchQualityAuditRecord | None = None,
    ) -> list[str]:
        findings = [f"{profile} review completed with structured integrity checks."]
        if unsupported_count:
            findings.append(f"{unsupported_count} claim(s) are still marked unsupported.")
        if section_count == 0:
            findings.append("No manuscript sections are available for review.")
        if audit is not None:
            findings.extend(audit.findings[:4])
        return findings

    @staticmethod
    def _review_actions(profile: str, verdict: str, audit: ResearchQualityAuditRecord | None = None) -> list[str]:
        if verdict == "pass" and (audit is None or not audit.repair_actions):
            return []
        if audit is not None and audit.repair_actions:
            return audit.repair_actions
        if profile == "citation-integrity":
            return ["Resolve unsupported claims or explicitly mark them as limitations."]
        if profile == "methodology":
            return ["Document baseline, ablation, seed, metric, and failure handling."]
        if profile == "domain":
            return ["Confirm the stated contribution against closest related work."]
        return ["Add limitations and a concrete falsification path before final release."]
