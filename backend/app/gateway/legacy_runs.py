from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.gateway.deps import get_run_service
from app.gateway.run_models import RunCreateRequest
from app.gateway.workflow import WorkflowSnapshot
from deerflow.utils.thread_id import ThreadId
from medrix_flow.runtime.runs import RunRecord, RunStatus

# This is a migration-only surface for clients written against Anaxa 1.x.
# Keep it out of the DeerFlow OpenAPI document; the canonical implementation
# is ``app.gateway.routers.thread_runs``.
router = APIRouter(prefix="/api/threads", tags=["runs"], include_in_schema=False)


class RunResponse(BaseModel):
    run_id: str
    thread_id: str
    assistant_id: str | None = None
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    multitask_strategy: str = "reject"
    created_at: str
    updated_at: str


class RunMessage(BaseModel):
    seq: int
    run_id: str
    thread_id: str
    event_type: str
    caller: str
    content: dict[str, Any]
    created_at: str


class RunMessagePage(BaseModel):
    data: list[RunMessage]
    has_more: bool


class FeedbackRequest(BaseModel):
    rating: Literal[-1, 1]
    comment: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    run_id: str
    thread_id: str
    rating: int
    comment: str | None = None
    created_at: str
    updated_at: str | None = None


class RunCompletionRequest(BaseModel):
    status: Literal["success", "interrupted", "error"] = "success"


class RunEventCreateRequest(BaseModel):
    event_type: str = Field(..., min_length=1, max_length=80)
    caller: str = Field(default="frontend", max_length=120)
    content: dict[str, Any] = Field(default_factory=dict)


def _record_to_response(record: RunRecord) -> RunResponse:
    return RunResponse(
        run_id=record.run_id,
        thread_id=record.thread_id,
        assistant_id=record.assistant_id,
        status=record.status.value,
        metadata=record.metadata,
        kwargs=record.kwargs,
        multitask_strategy=record.multitask_strategy,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.post("/{thread_id}/runs", response_model=RunResponse)
async def create_run(thread_id: ThreadId, body: RunCreateRequest, request: Request) -> RunResponse:
    service = get_run_service(request)
    record = await service.start_run(thread_id, body)
    return _record_to_response(record)


@router.get("/{thread_id}/runs", response_model=list[RunResponse])
async def list_runs(thread_id: ThreadId, request: Request) -> list[RunResponse]:
    service = get_run_service(request)
    records = await service.list_runs(thread_id)
    return [_record_to_response(record) for record in records]


@router.get("/{thread_id}/runs/{run_id}", response_model=RunResponse)
async def get_run(thread_id: ThreadId, run_id: str, request: Request) -> RunResponse:
    service = get_run_service(request)
    record = await service.require_run(thread_id, run_id)
    return _record_to_response(record)


@router.get("/{thread_id}/runs/{run_id}/workflow", response_model=WorkflowSnapshot)
async def get_run_workflow(
    thread_id: ThreadId,
    run_id: str,
    request: Request,
    limit: int = Query(default=200, ge=1, le=500),
    after_seq: int | None = Query(default=None),
) -> WorkflowSnapshot:
    service = get_run_service(request)
    payload = await service.build_workflow(thread_id, run_id, limit=limit, after_seq=after_seq)
    return WorkflowSnapshot(**payload)


@router.post("/{thread_id}/runs/{run_id}/events", response_model=RunMessage)
async def create_run_event(
    thread_id: ThreadId,
    run_id: str,
    body: RunEventCreateRequest,
    request: Request,
) -> RunMessage:
    service = get_run_service(request)
    row = await service.record_external_event(
        thread_id,
        run_id,
        event_type=body.event_type,
        caller=body.caller,
        content=body.content,
    )
    return RunMessage(**row)


@router.post("/{thread_id}/runs/stream")
async def stream_run(thread_id: ThreadId, body: RunCreateRequest, request: Request) -> StreamingResponse:
    if body.run_id:
        raise HTTPException(status_code=400, detail="run_id is not supported for /runs/stream")

    service = get_run_service(request)
    record = await service.start_run(thread_id, body)
    return StreamingResponse(
        service.sse_consumer(record, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Location": f"/api/threads/{thread_id}/runs/{record.run_id}",
        },
    )


@router.post("/{thread_id}/runs/wait")
async def wait_run(thread_id: ThreadId, body: RunCreateRequest, request: Request) -> dict[str, Any]:
    if body.run_id:
        raise HTTPException(status_code=400, detail="run_id is not supported for /runs/wait")

    service = get_run_service(request)
    record = await service.start_run(thread_id, body)
    if record.task is not None:
        try:
            await record.task
        except asyncio.CancelledError:
            pass
    updated = await service.require_run(thread_id, record.run_id)
    return {"run": _record_to_response(updated).model_dump()}


@router.post("/{thread_id}/runs/{run_id}/cancel")
async def cancel_run(
    thread_id: ThreadId,
    run_id: str,
    request: Request,
    action: Literal["interrupt", "rollback"] = Query(default="interrupt"),
    wait: bool = Query(default=False),
) -> Response:
    service = get_run_service(request)
    record = await service.cancel_run(thread_id, run_id, action=action)
    if wait and record.task is not None:
        try:
            await record.task
        except asyncio.CancelledError:
            pass
        return Response(status_code=204)
    return Response(status_code=202)


@router.get("/{thread_id}/runs/{run_id}/messages", response_model=RunMessagePage)
async def get_run_messages(
    thread_id: ThreadId,
    run_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    before_seq: int | None = Query(default=None),
    after_seq: int | None = Query(default=None),
) -> RunMessagePage:
    service = get_run_service(request)
    page = await service.list_run_messages(
        thread_id,
        run_id,
        limit=limit,
        before_seq=before_seq,
        after_seq=after_seq,
    )
    return RunMessagePage(**page)


@router.get("/{thread_id}/runs/{run_id}/feedback", response_model=FeedbackResponse | None)
async def get_feedback(thread_id: ThreadId, run_id: str, request: Request) -> FeedbackResponse | None:
    service = get_run_service(request)
    feedback = await service.get_feedback(thread_id, run_id)
    return FeedbackResponse(**feedback) if feedback else None


@router.put("/{thread_id}/runs/{run_id}/feedback", response_model=FeedbackResponse)
async def put_feedback(
    thread_id: ThreadId,
    run_id: str,
    body: FeedbackRequest,
    request: Request,
) -> FeedbackResponse:
    service = get_run_service(request)
    feedback = await service.upsert_feedback(thread_id, run_id, rating=body.rating, comment=body.comment)
    return FeedbackResponse(**feedback)


@router.delete("/{thread_id}/runs/{run_id}/feedback", status_code=204)
async def delete_feedback(thread_id: ThreadId, run_id: str, request: Request) -> Response:
    service = get_run_service(request)
    await service.delete_feedback(thread_id, run_id)
    return Response(status_code=204)


@router.post("/{thread_id}/runs/{run_id}/complete", include_in_schema=False, response_model=RunResponse)
async def complete_run(
    thread_id: ThreadId,
    run_id: str,
    body: RunCompletionRequest,
    request: Request,
) -> RunResponse:
    service = get_run_service(request)
    record = await service.complete_run(thread_id, run_id, status=RunStatus(body.status))
    return _record_to_response(record)
