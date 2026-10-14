"""M105: SQLite-backed JobStore (aiosqlite) — persists across restarts."""
from __future__ import annotations

import json
import os
import time
from typing import Any

import aiosqlite

from server.backends.base import AbstractJobStoreBackend, JobRecord

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'pending',
    result_json TEXT,
    error       TEXT,
    created_at  REAL NOT NULL
);
"""

_INSERT = """
INSERT INTO jobs (job_id, status, created_at)
VALUES (?, 'pending', ?);
"""

_SELECT = "SELECT job_id, status, result_json, error, created_at FROM jobs WHERE job_id = ?;"

_SET_RUNNING = "UPDATE jobs SET status = 'running' WHERE job_id = ?;"

_SET_DONE = "UPDATE jobs SET status = 'done', result_json = ? WHERE job_id = ?;"

_SET_FAILED = "UPDATE jobs SET status = 'failed', error = ? WHERE job_id = ?;"

# Inclusive (<=), as in the memory backend: ttl_seconds=0 expires every job.
_DELETE_EXPIRED = "DELETE FROM jobs WHERE created_at <= ?;"

_COUNT_CHANGES = "SELECT changes();"


class SqliteBackend(AbstractJobStoreBackend):
    """Persistent SQLite job store using aiosqlite.

    Parameters
    ----------
    path:
        File system path to the SQLite database file.
        Use ``":memory:"`` for an in-process SQLite (useful in tests).
    ttl_seconds:
        Jobs older than this are evicted on :meth:`get` and
        :meth:`delete_expired` calls.
    """

    def __init__(self, path: str, ttl_seconds: float = 3600.0) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self._db: aiosqlite.Connection | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open (or create) the SQLite file and create the ``jobs`` table."""
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(_CREATE_TABLE)
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def create(self) -> JobRecord:
        self._require_connected()
        record = JobRecord()
        await self._db.execute(_INSERT, (record.job_id, record.created_at))  # type: ignore[union-attr]
        await self._db.commit()  # type: ignore[union-attr]
        return record

    async def get(self, job_id: str) -> JobRecord | None:
        self._require_connected()
        # Opportunistic TTL eviction first
        await self.delete_expired(self.ttl_seconds)
        async with self._db.execute(_SELECT, (job_id,)) as cur:  # type: ignore[union-attr]
            row = await cur.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    async def set_running(self, job_id: str) -> None:
        self._require_connected()
        await self._db.execute(_SET_RUNNING, (job_id,))  # type: ignore[union-attr]
        await self._db.commit()  # type: ignore[union-attr]

    async def set_done(self, job_id: str, result: Any) -> None:
        self._require_connected()
        result_json = json.dumps(result, default=str)
        await self._db.execute(_SET_DONE, (result_json, job_id))  # type: ignore[union-attr]
        await self._db.commit()  # type: ignore[union-attr]

    async def set_failed(self, job_id: str, error: str) -> None:
        self._require_connected()
        await self._db.execute(_SET_FAILED, (error, job_id))  # type: ignore[union-attr]
        await self._db.commit()  # type: ignore[union-attr]

    async def delete_expired(self, ttl_seconds: float) -> int:
        self._require_connected()
        cutoff = time.time() - ttl_seconds
        await self._db.execute(_DELETE_EXPIRED, (cutoff,))  # type: ignore[union-attr]
        async with self._db.execute(_COUNT_CHANGES) as cur:  # type: ignore[union-attr]
            row = await cur.fetchone()
        await self._db.commit()  # type: ignore[union-attr]
        return int(row[0]) if row else 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _require_connected(self) -> None:
        if self._db is None:
            raise RuntimeError("SqliteBackend is not connected. Call await backend.connect() first.")

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> JobRecord:
        result_raw = row["result_json"]
        return JobRecord(
            job_id=row["job_id"],
            status=row["status"],
            result=json.loads(result_raw) if result_raw else None,
            error=row["error"],
            created_at=float(row["created_at"]),
        )
