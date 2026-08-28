from __future__ import annotations

import json
from typing import Any

from medrix_flow.runtime.db import SQLiteRuntimeDB
from medrix_flow.runtime.utils import now_iso

from .base import RunStore


class SQLiteRunStore(RunStore):
    def __init__(self, db: SQLiteRuntimeDB) -> None:
        self._db = db

    async def setup(self) -> None:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    assistant_id TEXT,
                    status TEXT NOT NULL,
                    multitask_strategy TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    kwargs_json TEXT NOT NULL,
                    pre_message_count INTEGER NOT NULL DEFAULT 0,
                    persisted_message_count INTEGER NOT NULL DEFAULT 0,
                    messages_complete INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            await self._db.conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_thread_id_created_at ON runs(thread_id, created_at DESC)")
            await self._db.conn.commit()

    async def put(
        self,
        run_id: str,
        *,
        thread_id: str,
        assistant_id: str | None,
        status: str,
        multitask_strategy: str,
        source: str,
        metadata: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        pre_message_count: int = 0,
        persisted_message_count: int = 0,
        messages_complete: bool = False,
        error: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        now = now_iso()
        async with self._db.lock:
            await self._db.conn.execute(
                """
                INSERT INTO runs (
                    run_id, thread_id, assistant_id, status, multitask_strategy, source,
                    metadata_json, kwargs_json, pre_message_count, persisted_message_count,
                    messages_complete, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    thread_id,
                    assistant_id,
                    status,
                    multitask_strategy,
                    source,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    json.dumps(kwargs or {}, ensure_ascii=False),
                    pre_message_count,
                    persisted_message_count,
                    int(messages_complete),
                    error,
                    created_at or now,
                    updated_at or created_at or now,
                ),
            )
            await self._db.conn.commit()

    async def get(self, run_id: str) -> dict[str, Any] | None:
        async with self._db.lock:
            cursor = await self._db.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = await cursor.fetchone()
        return self._row_to_dict(row) if row is not None else None

    async def list_by_thread(self, thread_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        async with self._db.lock:
            cursor = await self._db.conn.execute(
                "SELECT * FROM runs WHERE thread_id = ? ORDER BY created_at DESC LIMIT ?",
                (thread_id, limit),
            )
            rows = await cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    async def update_status(self, run_id: str, status: str, *, error: str | None = None, updated_at: str | None = None) -> None:
        async with self._db.lock:
            await self._db.conn.execute(
                "UPDATE runs SET status = ?, error = ?, updated_at = ? WHERE run_id = ?",
                (status, error, updated_at or now_iso(), run_id),
            )
            await self._db.conn.commit()

    async def update_materialization(
        self,
        run_id: str,
        *,
        persisted_message_count: int,
        messages_complete: bool,
        updated_at: str | None = None,
    ) -> None:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                UPDATE runs
                SET persisted_message_count = ?, messages_complete = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (persisted_message_count, int(messages_complete), updated_at or now_iso(), run_id),
            )
            await self._db.conn.commit()

    @staticmethod
    def _row_to_dict(row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "thread_id": row["thread_id"],
            "assistant_id": row["assistant_id"],
            "status": row["status"],
            "multitask_strategy": row["multitask_strategy"],
            "source": row["source"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "kwargs": json.loads(row["kwargs_json"] or "{}"),
            "pre_message_count": row["pre_message_count"],
            "persisted_message_count": row["persisted_message_count"],
            "messages_complete": bool(row["messages_complete"]),
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
