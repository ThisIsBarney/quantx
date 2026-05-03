"""M103/M105: JobStore — thin wrapper around an AbstractJobStoreBackend.

Backward-compatible API: code that imports ``JobStore`` and calls
``create/get/set_running/set_done/set_failed`` continues to work without
changes.  The default backend is ``InMemoryBackend`` (no external services
required).

To select a different backend at runtime use the ``QUANTX_JOB_STORE_URL``
environment variable and call ``create_job_store_backend()``::

    QUANTX_JOB_STORE_URL=sqlite:///var/quantx/jobs.db

See ``server.backends`` for supported URL schemes.
"""
from __future__ import annotations

import os
from typing import Any

from server.backends.base import JobRecord
from server.backends import AbstractJobStoreBackend, create_job_store_backend


class JobStore:
    """Async job store.  Delegates all operations to the configured backend.

    Parameters
    ----------
    backend:
        If supplied, use this backend directly.  If *None* (default), the
        backend is created from the ``QUANTX_JOB_STORE_URL`` env var, falling
        back to the in-memory backend.
    """

    def __init__(self, backend: AbstractJobStoreBackend | None = None) -> None:
        if backend is None:
            url = os.environ.get("QUANTX_JOB_STORE_URL", "memory://")
            backend = create_job_store_backend(url)
        self._backend = backend

    @property
    def backend(self) -> AbstractJobStoreBackend:
        """The underlying backend instance (useful for lifespan connect/close)."""
        return self._backend

    # ── Public API (unchanged from M103) ─────────────────────────────────────

    async def create(self) -> JobRecord:
        """Create a new job in ``pending`` state."""
        return await self._backend.create()

    async def get(self, job_id: str) -> JobRecord | None:
        """Return the job record, or ``None`` if absent / expired."""
        return await self._backend.get(job_id)

    async def set_running(self, job_id: str) -> None:
        await self._backend.set_running(job_id)

    async def set_done(self, job_id: str, result: Any) -> None:
        await self._backend.set_done(job_id, result)

    async def set_failed(self, job_id: str, error: str) -> None:
        await self._backend.set_failed(job_id, error)
