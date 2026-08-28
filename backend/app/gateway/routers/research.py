from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.gateway.deps import get_research_service
from deerflow.utils.thread_id import ThreadId
from medrix_flow.research import ResearchStage

router = APIRouter(prefix="/api/research", tags=["research"])


class ResearchQuestCreateRequest(BaseModel):
    thread_id: str
    topic: str
    title: str | None = None
    scope: str | None = None
    objective: str | None = None
    domain: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchQuestAdvanceRequest(BaseModel):
    target_stage: ResearchStage | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    tool_name: str | None = None
    model_name: str | None = None


class ResearchGateDecisionRequest(BaseModel):
    stage: ResearchStage
    gate_type: str
    status: Literal["pending", "approved", "rejected"] = "approved"
    decision: str | None = None
    reason: str | None = None


def _raise_for_value_error(exc: ValueError) -> None:
    message = str(exc)
    if "not found" in message.lower():
        raise HTTPException(status_code=404, detail=message) from exc
    raise HTTPException(status_code=422, detail=message) from exc


@router.get("/quests")
async def list_quests(
    request: Request,
    # The default makes this query filter optional; keeping the canonical
    # ``ThreadId`` annotation ensures any supplied value is still validated.
    thread_id: ThreadId = Query(default=None),
) -> dict[str, Any]:
    service = get_research_service(request)
    quests = await service.list_quests(thread_id)
    return {"data": [quest.model_dump() for quest in quests]}


@router.post("/quests")
async def create_quest(body: ResearchQuestCreateRequest, request: Request) -> dict[str, Any]:
    service = get_research_service(request)
    quest = await service.create_quest(
        thread_id=body.thread_id,
        topic=body.topic,
        title=body.title,
        scope=body.scope,
        objective=body.objective,
        domain=body.domain,
        metadata=body.metadata,
    )
    return {"quest": quest.model_dump()}


@router.get("/quests/{quest_id}")
async def get_quest(quest_id: str, request: Request) -> dict[str, Any]:
    service = get_research_service(request)
    try:
        snapshot = await service.get_snapshot(quest_id)
    except ValueError as exc:
        _raise_for_value_error(exc)
    return snapshot.model_dump()


@router.post("/quests/{quest_id}/advance")
async def advance_quest(
    quest_id: str,
    body: ResearchQuestAdvanceRequest,
    request: Request,
) -> dict[str, Any]:
    service = get_research_service(request)
    try:
        result = await service.advance_quest(
            quest_id,
            target_stage=body.target_stage,
            inputs=body.inputs,
            artifacts=body.artifacts,
            tool_name=body.tool_name,
            model_name=body.model_name,
        )
    except ValueError as exc:
        _raise_for_value_error(exc)
    return result.model_dump()


@router.post("/quests/{quest_id}/gate")
async def decide_gate(
    quest_id: str,
    body: ResearchGateDecisionRequest,
    request: Request,
) -> dict[str, Any]:
    service = get_research_service(request)
    try:
        gate = await service.decide_gate(
            quest_id,
            stage=body.stage,
            gate_type=body.gate_type,
            status=body.status,
            decision=body.decision,
            reason=body.reason,
        )
    except ValueError as exc:
        _raise_for_value_error(exc)
    return {"gate": gate.model_dump()}


@router.get("/quests/{quest_id}/evidence")
async def get_evidence(quest_id: str, request: Request) -> dict[str, Any]:
    service = get_research_service(request)
    try:
        evidence = await service.list_evidence(quest_id)
    except ValueError as exc:
        _raise_for_value_error(exc)
    return {"data": [item.model_dump() for item in evidence]}


@router.get("/quests/{quest_id}/experiments")
async def get_experiments(quest_id: str, request: Request) -> dict[str, Any]:
    service = get_research_service(request)
    try:
        branches = await service.list_experiment_branches(quest_id)
    except ValueError as exc:
        _raise_for_value_error(exc)
    return {"data": [item.model_dump() for item in branches]}


@router.get("/quests/{quest_id}/manuscript")
async def get_manuscript(quest_id: str, request: Request) -> dict[str, Any]:
    service = get_research_service(request)
    try:
        sections = await service.list_manuscript_sections(quest_id)
    except ValueError as exc:
        _raise_for_value_error(exc)
    return {"data": [item.model_dump() for item in sections]}
