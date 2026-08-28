from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite


class SQLiteRuntimeDB:
    """Shared aiosqlite connection for runtime repositories.

    A single connection keeps `:memory:` tests usable and avoids schema drift
    across repositories.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def path(self) -> str:
        return self._path

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteRuntimeDB is not connected")
        return self._conn

    async def connect(self) -> None:
        if self._conn is not None:
            return

        if self._path not in {":memory:"} and not self._path.startswith("file:"):
            Path(self._path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

        self._conn = await aiosqlite.connect(self._path, uri=self._path.startswith("file:"))
        self._conn.row_factory = aiosqlite.Row
        if self._path != ":memory:" and "mode=memory" not in self._path:
            await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is None:
            return
        await self._conn.close()
        self._conn = None
