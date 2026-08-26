"""Run registry with best-effort persistence backing."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from medrix_flow.runtime.utils import now_iso

from .schemas import RunRecord, RunStatus
from .store.base import RunStore

logger = logging.getLogger(__name__)


class ConflictError(Exception):
    """Raised when multitask_strategy=reject and a thread already has an active run."""


class UnsupportedStrategyError(Exception):
    """Raised when a multitask strategy is unsupported."""


class RunManager:
    def __init__(self, store: RunStore | None = None) -> None:
        self._store = store
        self._runs: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def create_or_reject(
        self,
        thread_id: str,
        assistant_id: str | None = None,
        *,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        multitask_strategy: str = "reject",
        source: str = "gateway",
        pre_message_count: int = 0,
    ) -> RunRecord:
        supported = {"reject", "interrupt", "rollback"}
        if multitask_strategy not in supported:
            raise UnsupportedStrategyError(
                f"Unsupported multitask strategy '{multitask_strategy}'. Supported: {', '.join(sorted(supported))}"
            )

        existing = await self.get(run_id) if run_id is not None else None
        if existing is not None:
            return existing

        run_id = run_id or str(uuid.uuid4())
        now = now_iso()
        async with self._lock:
            inflight = [
                record
                for record in self._runs.values()
                if record.thread_id == thread_id and record.status in {RunStatus.pending, RunStatus.running}
            ]

            if multitask_strategy == "reject" and inflight:
                raise ConflictError(f"Thread {thread_id} already has an active run")

            if multitask_strategy in {"interrupt", "rollback"}:
                for record in inflight:
                    record.abort_action = multitask_strategy
                    record.abort_event.set()
                    if record.task is not None and not record.task.done():
                        record.task.cancel()
                    record.status = RunStatus.interrupted
                    record.updated_at = now

            record = RunRecord(
                run_id=run_id,
                thread_id=thread_id,
                assistant_id=assistant_id,
                status=RunStatus.pending,
                metadata=metadata or {},
                kwargs=kwargs or {},
                multitask_strategy=multitask_strategy,
                source=source,
                pre_message_count=pre_message_count,
                created_at=now,
                updated_at=now,
            )
            self._runs[run_id] = record

        await self._persist_new_record(record)
        return record

    async def register_external(
        self,
        *,
        run_id: str,
        thread_id: str,
        assistant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        pre_message_count: int = 0,
    ) -> RunRecord:
        existing = await self.get(run_id)
        if existing is not None:
            return existing

        return await self.create_or_reject(
            thread_id,
            assistant_id,
            run_id=run_id,
            metadata=metadata,
            kwargs=kwargs,
            multitask_strategy="reject",
            source="external",
            pre_message_count=pre_message_count,
        )

    async def attach_task(self, run_id: str, task: asyncio.Task) -> None:
        async with self._lock:
            record = self._runs.get(run_id)
            if record is not None:
                record.task = task

    async def try_start(self, run_id: str) -> bool:
        """Atomically admit a pending run unless cancellation already won."""
        async with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return False
            if record.abort_event.is_set() or record.status is not RunStatus.pending:
                return False
            record.status = RunStatus.running
            record.updated_at = now_iso()
            return True

    async def get(self, run_id: str | None) -> RunRecord | None:
        if not run_id:
            return None
        async with self._lock:
            record = self._runs.get(run_id)
        if record is not None:
            return record
        if self._store is None:
            return None
        row = await self._store.get(run_id)
        return self._hydrate(row) if row is not None else None

    async def list_by_thread(self, thread_id: str, *, limit: int = 100) -> list[RunRecord]:
        seen: dict[str, RunRecord] = {}
        if self._store is not None:
            for row in await self._store.list_by_thread(thread_id, limit=limit):
                seen[row["run_id"]] = self._hydrate(row)

        async with self._lock:
            for record in self._runs.values():
                if record.thread_id == thread_id:
                    seen[record.run_id] = record

        records = sorted(seen.values(), key=lambda item: item.created_at, reverse=True)
        return records[:limit]

    async def set_status(self, run_id: str, status: RunStatus, *, error: str | None = None) -> None:
        async with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                hydrated = None
            else:
                record.status = status
                record.error = error
                record.updated_at = now_iso()
                hydrated = record

        if self._store is not None:
            try:
                await self._store.update_status(
                    run_id,
                    status.value,
                    error=error,
                    updated_at=hydrated.updated_at if hydrated is not None else None,
                )
            except Exception:
                logger.warning("Failed to persist run status for %s", run_id, exc_info=True)

    async def mark_materialized(self, run_id: str, *, persisted_message_count: int, complete: bool) -> None:
        updated_at = now_iso()
        async with self._lock:
            record = self._runs.get(run_id)
            if record is not None:
                record.persisted_message_count = persisted_message_count
                record.messages_complete = complete
                record.updated_at = updated_at
        if self._store is not None:
            try:
                await self._store.update_materialization(
                    run_id,
                    persisted_message_count=persisted_message_count,
                    messages_complete=complete,
                    updated_at=updated_at,
                )
            except Exception:
                logger.warning("Failed to persist materialization state for %s", run_id, exc_info=True)

    async def cancel(self, run_id: str, *, action: str = "interrupt") -> bool:
        async with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return False
            if record.status not in {RunStatus.pending, RunStatus.running}:
                return False
            record.abort_action = action
            record.abort_event.set()
            if record.task is not None and not record.task.done():
                record.task.cancel()
            record.status = RunStatus.interrupted
            record.updated_at = now_iso()
        if self._store is not None:
            try:
                await self._store.update_status(run_id, RunStatus.interrupted.value, updated_at=record.updated_at)
            except Exception:
                logger.warning("Failed to persist interrupted status for %s", run_id, exc_info=True)
        return True

    async def _persist_new_record(self, record: RunRecord) -> None:
        if self._store is None:
            return
        try:
            await self._store.put(
                record.run_id,
                thread_id=record.thread_id,
                assistant_id=record.assistant_id,
                status=record.status.value,
                multitask_strategy=record.multitask_strategy,
                source=record.source,
                metadata=record.metadata,
                kwargs=record.kwargs,
                pre_message_count=record.pre_message_count,
                persisted_message_count=record.persisted_message_count,
                messages_complete=record.messages_complete,
                error=record.error,
                created_at=record.created_at,
                updated_at=record.updated_at,
            )
        except Exception:
            logger.warning("Failed to persist new run %s", record.run_id, exc_info=True)

    @staticmethod
    def _hydrate(row: dict[str, Any]) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            assistant_id=row.get("assistant_id"),
            status=RunStatus(row["status"]),
            metadata=row.get("metadata") or {},
            kwargs=row.get("kwargs") or {},
            multitask_strategy=row.get("multitask_strategy", "reject"),
            source=row.get("source", "gateway"),
            pre_message_count=int(row.get("pre_message_count", 0)),
            persisted_message_count=int(row.get("persisted_message_count", 0)),
            messages_complete=bool(row.get("messages_complete", False)),
            error=row.get("error"),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
        )
