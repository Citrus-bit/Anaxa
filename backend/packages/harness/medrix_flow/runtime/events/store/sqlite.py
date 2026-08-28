from __future__ import annotations

import json
from typing import Any

from medrix_flow.runtime.db import SQLiteRuntimeDB
from medrix_flow.runtime.utils import now_iso

from .base import RunEventStore


class SQLiteRunEventStore(RunEventStore):
    def __init__(self, db: SQLiteRuntimeDB) -> None:
        self._db = db

    async def setup(self) -> None:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    caller TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            await self._db.conn.execute("CREATE INDEX IF NOT EXISTS idx_run_events_thread_seq ON run_events(thread_id, seq)")
            await self._db.conn.execute("CREATE INDEX IF NOT EXISTS idx_run_events_thread_run_seq ON run_events(thread_id, run_id, seq)")
            await self._db.conn.commit()

    async def put(
        self,
        *,
        thread_id: str,
        run_id: str,
        event_type: str,
        content: dict[str, Any],
        caller: str,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        now = created_at or now_iso()
        async with self._db.lock:
            cursor = await self._db.conn.execute(
                """
                INSERT INTO run_events (thread_id, run_id, event_type, caller, content_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (thread_id, run_id, event_type, caller, json.dumps(content, ensure_ascii=False), now),
            )
            await self._db.conn.commit()
            seq = cursor.lastrowid
        return {
            "seq": seq,
            "thread_id": thread_id,
            "run_id": run_id,
            "event_type": event_type,
            "caller": caller,
            "content": content,
            "created_at": now,
        }

    async def put_batch(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event in events:
            rows.append(await self.put(**event))
        return rows

    async def list_messages_by_run(
        self,
        thread_id: str,
        run_id: str,
        *,
        limit: int = 50,
        before_seq: int | None = None,
        after_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT seq, thread_id, run_id, event_type, caller, content_json, created_at
            FROM run_events
            WHERE thread_id = ? AND run_id = ?
        """
        params: list[Any] = [thread_id, run_id]
        if before_seq is not None:
            query += " AND seq < ?"
            params.append(before_seq)
        if after_seq is not None:
            query += " AND seq > ?"
            params.append(after_seq)
        if before_seq is not None:
            query += " ORDER BY seq DESC LIMIT ?"
        else:
            query += " ORDER BY seq ASC LIMIT ?"
        params.append(limit)

        async with self._db.lock:
            cursor = await self._db.conn.execute(query, tuple(params))
            rows = await cursor.fetchall()

        data = [self._row_to_dict(row) for row in rows]
        if before_seq is not None:
            data.reverse()
        return data

    @staticmethod
    def _row_to_dict(row) -> dict[str, Any]:
        return {
            "seq": row["seq"],
            "thread_id": row["thread_id"],
            "run_id": row["run_id"],
            "event_type": row["event_type"],
            "caller": row["caller"],
            "content": json.loads(row["content_json"] or "{}"),
            "created_at": row["created_at"],
        }
