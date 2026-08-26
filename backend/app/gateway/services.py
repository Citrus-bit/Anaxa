"""Gateway service layer for runtime-backed runs and feedback."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException, Request
from langchain_core.messages import HumanMessage
from langgraph.types import Checkpointer

from medrix_flow.agents.checkpointer.provider import get_checkpointer as get_sync_checkpointer
from medrix_flow.client import MedrixFlowClient
from medrix_flow.config.paths import get_paths
from medrix_flow.runtime.events.store.base import RunEventStore
from medrix_flow.runtime.feedback import SQLiteFeedbackRepo
from medrix_flow.runtime.runs import RunManager, RunRecord, RunStatus
from medrix_flow.runtime.runs.manager import ConflictError, UnsupportedStrategyError
from medrix_flow.runtime.runs.store.base import RunStore
from medrix_flow.runtime.serialization import extract_text, message_caller, message_event_type, serialize_message
from medrix_flow.runtime.stream_bridge import END_SENTINEL, MemoryStreamBridge

from .workflow import normalize_stream_event

logger = logging.getLogger(__name__)

_ITERATION_END = object()
STALE_EXTERNAL_RUN_AFTER = timedelta(minutes=30)


def format_sse(event: str, data: Any, *, event_id: str | None = None) -> str:
    payload = json.dumps(data, default=str, ensure_ascii=False)
    parts = [f"event: {event}", f"data: {payload}"]
    if event_id:
        parts.append(f"id: {event_id}")
    parts.extend(["", ""])
    return "\n".join(parts)


def _next_event(iterator: Iterable[Any]) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return _ITERATION_END


def _extract_last_human_text(raw_input: dict[str, Any] | None) -> str:
    if not raw_input:
        return ""
    messages = raw_input.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return extract_text(message.content)
        if isinstance(message, dict):
            role = message.get("role", message.get("type", ""))
            if role not in {"human", "user"}:
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                return extract_text(content)
            if isinstance(content, str):
                return content
    return ""


def _parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _is_stale_external_run(record: RunRecord, *, now: datetime | None = None) -> bool:
    if record.source != "external" or record.status not in {RunStatus.pending, RunStatus.running}:
        return False
    updated_at = _parse_iso_datetime(record.updated_at or record.created_at)
    if updated_at is None:
        return False
    current = now or datetime.now(UTC)
    return current - updated_at > STALE_EXTERNAL_RUN_AFTER


@dataclass
class MaterializationResult:
    persisted_count: int
    complete: bool


class GatewayRunService:
    def __init__(
        self,
        *,
        checkpointer: Checkpointer,
        stream_bridge: MemoryStreamBridge,
        run_manager: RunManager,
        run_store: RunStore,
        event_store: RunEventStore,
        feedback_repo: SQLiteFeedbackRepo,
    ) -> None:
        self._checkpointer = checkpointer
        self._stream_bridge = stream_bridge
        self._run_manager = run_manager
        self._run_store = run_store
        self._event_store = event_store
        self._feedback_repo = feedback_repo

    async def start_run(self, thread_id: str, body: Any) -> RunRecord:
        if getattr(body, "run_id", None):
            return await self.register_external_run(thread_id, body)

        pre_message_count = await self._safe_pre_message_count(thread_id)
        kwargs = {
            "input": getattr(body, "input", None),
            "config": getattr(body, "config", None),
            "context": getattr(body, "context", None),
        }
        try:
            record = await self._run_manager.create_or_reject(
                thread_id,
                getattr(body, "assistant_id", None),
                metadata=getattr(body, "metadata", None),
                kwargs=kwargs,
                multitask_strategy=getattr(body, "multitask_strategy", "reject"),
                source="gateway",
                pre_message_count=pre_message_count,
            )
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except UnsupportedStrategyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        task = asyncio.create_task(self._execute_gateway_run(record, body))
        await self._run_manager.attach_task(record.run_id, task)
        return record

    async def register_external_run(self, thread_id: str, body: Any) -> RunRecord:
        run_id = getattr(body, "run_id", None)
        if not run_id:
            raise HTTPException(status_code=400, detail="run_id is required for external run registration")

        pre_message_count = await self._safe_pre_message_count(thread_id)
        return await self._run_manager.register_external(
            run_id=run_id,
            thread_id=thread_id,
            assistant_id=getattr(body, "assistant_id", None),
            metadata=getattr(body, "metadata", None),
            kwargs={
                "input": getattr(body, "input", None),
                "config": getattr(body, "config", None),
                "context": getattr(body, "context", None),
            },
            pre_message_count=pre_message_count,
        )

    async def complete_run(
        self,
        thread_id: str,
        run_id: str,
        *,
        status: RunStatus = RunStatus.success,
    ) -> RunRecord:
        record = await self.require_run(thread_id, run_id)
        await self.materialize_run_messages(record, finalize=True, final_status=status)
        return await self.require_run(thread_id, run_id)

    async def cancel_run(self, thread_id: str, run_id: str, *, action: str = "interrupt") -> RunRecord:
        record = await self.require_run(thread_id, run_id)
        cancelled = await self._run_manager.cancel(run_id, action=action)
        if not cancelled:
            raise HTTPException(status_code=409, detail=f"Run {run_id} is not cancellable (status: {record.status.value})")
        await self.materialize_run_messages(record, finalize=True, final_status=RunStatus.interrupted)
        return await self.require_run(thread_id, run_id)

    async def require_run(self, thread_id: str, run_id: str) -> RunRecord:
        record = await self._run_manager.get(run_id)
        if record is None or record.thread_id != thread_id:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        record = await self._reconcile_stale_external_run(record)
        return record

    async def list_runs(self, thread_id: str) -> list[RunRecord]:
        records = await self._run_manager.list_by_thread(thread_id)
        return [await self._reconcile_stale_external_run(record) for record in records]

    async def get_feedback(self, thread_id: str, run_id: str) -> dict | None:
        await self.require_run(thread_id, run_id)
        return await self._feedback_repo.get_by_run(thread_id=thread_id, run_id=run_id)

    async def upsert_feedback(self, thread_id: str, run_id: str, *, rating: int, comment: str | None = None) -> dict:
        record = await self.require_run(thread_id, run_id)
        if not record.messages_complete:
            await self.materialize_run_messages(record, finalize=True, final_status=RunStatus.success)
        return await self._feedback_repo.upsert(thread_id=thread_id, run_id=run_id, rating=rating, comment=comment)

    async def delete_feedback(self, thread_id: str, run_id: str) -> bool:
        await self.require_run(thread_id, run_id)
        return await self._feedback_repo.delete_by_run(thread_id=thread_id, run_id=run_id)

    async def list_run_messages(
        self,
        thread_id: str,
        run_id: str,
        *,
        limit: int = 50,
        before_seq: int | None = None,
        after_seq: int | None = None,
    ) -> dict[str, Any]:
        record = await self.require_run(thread_id, run_id)
        if not record.messages_complete and record.status in {RunStatus.success, RunStatus.error, RunStatus.interrupted}:
            await self.materialize_run_messages(record, finalize=True, final_status=record.status)
        rows = await self._event_store.list_messages_by_run(
            thread_id,
            run_id,
            limit=limit + 1,
            before_seq=before_seq,
            after_seq=after_seq,
        )
        has_more = len(rows) > limit
        return {"data": rows[:limit] if has_more else rows, "has_more": has_more}

    async def build_workflow(
        self,
        thread_id: str,
        run_id: str,
        *,
        limit: int = 200,
        after_seq: int | None = None,
    ) -> dict[str, Any]:
        from .workflow import build_workflow_snapshot

        record = await self.require_run(thread_id, run_id)
        if not record.messages_complete and record.status in {RunStatus.success, RunStatus.error, RunStatus.interrupted}:
            await self.materialize_run_messages(record, finalize=True, final_status=record.status)
            record = await self.require_run(thread_id, run_id)
        rows = await self._event_store.list_messages_by_run(
            thread_id,
            run_id,
            limit=limit + 1,
            after_seq=after_seq,
        )
        has_more = len(rows) > limit
        snapshot = build_workflow_snapshot(record=record, event_rows=rows[:limit], has_more=has_more)
        return snapshot.model_dump()

    async def record_external_event(
        self,
        thread_id: str,
        run_id: str,
        *,
        event_type: str,
        caller: str,
        content: dict[str, Any],
    ) -> dict[str, Any]:
        record = await self.require_run(thread_id, run_id)
        row = await self._event_store.put(
            thread_id=thread_id,
            run_id=run_id,
            event_type=event_type,
            caller=caller,
            content=content,
        )
        if record.status in {RunStatus.pending, RunStatus.running}:
            await self._run_manager.set_status(run_id, RunStatus.running, error=record.error)
        return row

    async def materialize_run_messages(
        self,
        record: RunRecord,
        *,
        finalize: bool,
        final_status: RunStatus | None,
    ) -> MaterializationResult:
        if record.source == "gateway":
            if finalize or not record.messages_complete:
                await self._run_manager.mark_materialized(
                    record.run_id,
                    persisted_message_count=record.persisted_message_count,
                    complete=finalize,
                )
            if finalize and final_status is not None:
                await self._run_manager.set_status(record.run_id, final_status, error=record.error)
            return MaterializationResult(persisted_count=record.persisted_message_count, complete=finalize)

        checkpoint_messages = await self._get_checkpoint_messages(record.thread_id)
        persisted_start = record.pre_message_count + record.persisted_message_count
        if persisted_start < 0:
            persisted_start = 0
        new_messages = checkpoint_messages[persisted_start:]
        persisted_count = record.persisted_message_count

        if new_messages:
            try:
                await self._event_store.put_batch(
                    [
                        {
                            "thread_id": record.thread_id,
                            "run_id": record.run_id,
                            "event_type": message_event_type(message),
                            "content": serialize_message(message),
                            "caller": message_caller(message),
                        }
                        for message in new_messages
                    ]
                )
                persisted_count += len(new_messages)
            except Exception:
                logger.warning("Best-effort run message persistence failed for %s", record.run_id, exc_info=True)

        if finalize or persisted_count != record.persisted_message_count:
            await self._run_manager.mark_materialized(
                record.run_id,
                persisted_message_count=persisted_count,
                complete=finalize,
            )

        if finalize and final_status is not None:
            await self._run_manager.set_status(record.run_id, final_status, error=record.error)

        return MaterializationResult(persisted_count=persisted_count, complete=finalize)

    async def sse_consumer(self, record: RunRecord, request: Request) -> AsyncIterator[str]:
        async for event in self._stream_bridge.subscribe(record.run_id):
            if await request.is_disconnected():
                break
            name = event.get("event")
            if name == END_SENTINEL:
                break
            yield format_sse(name, event.get("data", {}), event_id=record.run_id)

    async def _reconcile_stale_external_run(self, record: RunRecord) -> RunRecord:
        if not _is_stale_external_run(record):
            return record
        await self._run_manager.set_status(
            record.run_id,
            RunStatus.interrupted,
            error="External run was marked interrupted after 30 minutes without updates.",
        )
        refreshed = await self._run_manager.get(record.run_id)
        return refreshed or record

    async def _execute_gateway_run(self, record: RunRecord, body: Any) -> None:
        started = False
        final_status = RunStatus.interrupted
        try:
            started = await self._run_manager.try_start(record.run_id)
            if not started:
                return
            get_paths().ensure_thread_dirs(record.thread_id)

            context = getattr(body, "context", None) or {}
            assistant_id = getattr(body, "assistant_id", None)
            agent_name = context.get("agent_name")
            if not agent_name and assistant_id not in {None, "lead_agent"}:
                agent_name = assistant_id

            client = MedrixFlowClient(
                checkpointer=get_sync_checkpointer(),
                model_name=context.get("model_name"),
                thinking_enabled=context.get("thinking_enabled", True),
                reasoning_effort=context.get("reasoning_effort"),
                subagent_enabled=context.get("subagent_enabled", False),
                plan_mode=context.get("is_plan_mode", False),
                agent_name=agent_name,
            )
            prompt = _extract_last_human_text(getattr(body, "input", None))
            if not prompt:
                raise ValueError("Gateway run requires a human input message")

            iterator = client.stream(prompt, thread_id=record.thread_id)
            while not record.abort_event.is_set():
                event = await asyncio.to_thread(_next_event, iterator)
                if event is _ITERATION_END:
                    break
                await self._stream_bridge.publish(
                    record.run_id,
                    {"event": event.type, "data": event.data},
                )
                await self._persist_stream_event(record, event.type, event.data)

            final_status = RunStatus.interrupted if record.abort_event.is_set() else RunStatus.success
        except asyncio.CancelledError:
            final_status = RunStatus.interrupted
        except Exception as exc:
            logger.exception("Gateway run failed: run_id=%s", record.run_id)
            record.error = str(exc)
            await self._stream_bridge.publish(
                record.run_id,
                {"event": "error", "data": {"message": str(exc)}},
            )
            final_status = RunStatus.error
        finally:
            if started:
                await self.materialize_run_messages(record, finalize=True, final_status=final_status)
            await self._stream_bridge.close(record.run_id)

    async def _persist_stream_event(self, record: RunRecord, event_type: str, data: dict[str, Any]) -> None:
        try:
            normalized_type, caller, content = normalize_stream_event(event_type, data)
            row = await self._event_store.put(
                thread_id=record.thread_id,
                run_id=record.run_id,
                event_type=normalized_type,
                caller=caller,
                content=content,
            )
            record.persisted_message_count += 1
            await self._run_manager.mark_materialized(
                record.run_id,
                persisted_message_count=record.persisted_message_count,
                complete=False,
            )
            if row.get("created_at"):
                await self._run_manager.set_status(record.run_id, record.status, error=record.error)
        except Exception:
            logger.warning("Best-effort stream event persistence failed for %s", record.run_id, exc_info=True)

    async def _get_checkpoint_message_count(self, thread_id: str) -> int:
        return len(await self._get_checkpoint_messages(thread_id))

    async def _safe_pre_message_count(self, thread_id: str) -> int:
        try:
            return await self._get_checkpoint_message_count(thread_id)
        except Exception:
            logger.warning("Best-effort pre-run message count capture failed for thread %s", thread_id, exc_info=True)
            return 0

    async def _get_checkpoint_messages(self, thread_id: str) -> list[Any]:
        try:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint_tuple = await self._checkpointer.aget_tuple(config)
        except Exception:
            logger.warning("Best-effort checkpoint read failed for thread %s", thread_id, exc_info=True)
            return []
        if checkpoint_tuple is None:
            return []
        checkpoint = getattr(checkpoint_tuple, "checkpoint", {}) or {}
        channel_values = checkpoint.get("channel_values", {}) or {}
        messages = channel_values.get("messages") or []
        return list(messages) if isinstance(messages, list) else []
