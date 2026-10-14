"""M105: In-memory JobStore backend (default, zero dependencies)."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from server.backends.base import AbstractJobStoreBackend, JobRecord


class InMemoryBackend(AbstractJobStoreBackend):
    """Asyncio-safe in-memory store.

    Identical behaviour to the original ``JobStore`` class, extracted into the
    backend protocol so that ``SqliteBackend`` and ``RedisBackend`` can be
    swapped in transparently.

    Parameters
    ----------
    ttl_seconds:
        Jobs older than this are evicted on the next :meth:`get` call.
    """

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    # ── AbstractJobStoreBackend ───────────────────────────────────────────────

    async def create(self) -> JobRecord:
        record = JobRecord()
        async with self._lock:
            self._jobs[record.job_id] = record
        return record

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            await self._evict_locked()
            return self._jobs.get(job_id)

    async def set_running(self, job_id: str) -> None:
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "running"

    async def set_done(self, job_id: str, result: Any) -> None:
        async with self._lock:
            if job_id in self._jobs:
                r = self._jobs[job_id]
                r.status = "done"
                r.result = result

    async def set_failed(self, job_id: str, error: str) -> None:
        async with self._lock:
            if job_id in self._jobs:
                r = self._jobs[job_id]
                r.status = "failed"
                r.error = error

    async def delete_expired(self, ttl_seconds: float) -> int:
        async with self._lock:
            # Inclusive (>=): ttl_seconds=0 expires every job, even one created in the
            # same clock tick (Windows' time.time() can return the same value twice).
            now = time.time()
            expired = [
                jid for jid, rec in self._jobs.items()
                if now - rec.created_at >= ttl_seconds
            ]
            for jid in expired:
                del self._jobs[jid]
            return len(expired)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _evict_locked(self) -> None:
        """Must be called with ``self._lock`` held."""
        expired = [
            jid for jid, rec in self._jobs.items()
            if time.time() - rec.created_at > self.ttl_seconds
        ]
        for jid in expired:
            del self._jobs[jid]
