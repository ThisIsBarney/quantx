"""M103: In-memory async job store with TTL expiry.

Jobs are keyed by UUID. Status lifecycle:
  pending → running → done | failed

The store is a plain dict protected by an asyncio.Lock so concurrent
background tasks can update job records safely.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JobRecord:
    """Single job entry stored in ``JobStore``."""

    job_id: str
    status: str  # "pending" | "running" | "done" | "failed"
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)


class JobStore:
    """Thread-safe in-memory job store.

    Parameters
    ----------
    ttl_seconds : float
        Jobs older than this are silently evicted on the next ``get`` call.
        Default: 3600 s (1 hour).
    """

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._jobs: dict[str, JobRecord] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    async def create(self) -> JobRecord:
        """Create a new job in ``pending`` state and return its record."""
        job_id = str(uuid.uuid4())
        record = JobRecord(job_id=job_id, status="pending")
        async with self._lock:
            self._jobs[job_id] = record
        return record

    async def get(self, job_id: str) -> JobRecord | None:
        """Return the record for *job_id*, or ``None`` if not found / expired."""
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

    def _is_expired(self, record: JobRecord) -> bool:
        return time.monotonic() - record.created_at > self.ttl_seconds

    async def _evict_locked(self) -> None:
        """Remove expired jobs.  Must be called with ``self._lock`` held."""
        expired = [jid for jid, rec in self._jobs.items() if self._is_expired(rec)]
        for jid in expired:
            del self._jobs[jid]
