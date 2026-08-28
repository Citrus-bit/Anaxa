from __future__ import annotations

import uuid

from medrix_flow.runtime.db import SQLiteRuntimeDB
from medrix_flow.runtime.utils import now_iso


class SQLiteFeedbackRepo:
    def __init__(self, db: SQLiteRuntimeDB) -> None:
        self._db = db

    async def setup(self) -> None:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK(rating IN (-1, 1)),
                    comment TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(thread_id, run_id)
                )
                """
            )
            await self._db.conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_thread_run ON feedback(thread_id, run_id)")
            await self._db.conn.commit()

    async def get_by_run(self, *, thread_id: str, run_id: str) -> dict | None:
        async with self._db.lock:
            cursor = await self._db.conn.execute(
                "SELECT * FROM feedback WHERE thread_id = ? AND run_id = ?",
                (thread_id, run_id),
            )
            row = await cursor.fetchone()
        return self._row_to_dict(row) if row is not None else None

    async def upsert(self, *, thread_id: str, run_id: str, rating: int, comment: str | None = None) -> dict:
        if rating not in (-1, 1):
            raise ValueError(f"rating must be +1 or -1, got {rating}")

        existing = await self.get_by_run(thread_id=thread_id, run_id=run_id)
        now = now_iso()
        async with self._db.lock:
            if existing is None:
                feedback_id = str(uuid.uuid4())
                await self._db.conn.execute(
                    """
                    INSERT INTO feedback (feedback_id, run_id, thread_id, rating, comment, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (feedback_id, run_id, thread_id, rating, comment, now, now),
                )
            else:
                feedback_id = existing["feedback_id"]
                await self._db.conn.execute(
                    """
                    UPDATE feedback
                    SET rating = ?, comment = ?, updated_at = ?
                    WHERE thread_id = ? AND run_id = ?
                    """,
                    (rating, comment, now, thread_id, run_id),
                )
            await self._db.conn.commit()

        return {
            "feedback_id": feedback_id,
            "run_id": run_id,
            "thread_id": thread_id,
            "rating": rating,
            "comment": comment,
            "created_at": existing["created_at"] if existing else now,
            "updated_at": now,
        }

    async def delete_by_run(self, *, thread_id: str, run_id: str) -> bool:
        async with self._db.lock:
            cursor = await self._db.conn.execute(
                "DELETE FROM feedback WHERE thread_id = ? AND run_id = ?",
                (thread_id, run_id),
            )
            await self._db.conn.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_dict(row) -> dict:
        return {
            "feedback_id": row["feedback_id"],
            "run_id": row["run_id"],
            "thread_id": row["thread_id"],
            "rating": row["rating"],
            "comment": row["comment"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
