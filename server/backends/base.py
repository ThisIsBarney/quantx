"""M105: Abstract base class for JobStore backends."""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JobRecord:
    """Single job entry.  Shared across all backends.

    Attributes
    ----------
    job_id : str
        UUID-4 string identifier.
    status : str
        One of ``"pending"``, ``"running"``, ``"done"``, ``"failed"``.
    result : Any
        Populated when status is ``"done"``.
    error : str | None
        Populated when status is ``"failed"``.
    created_at : float
        ``time.time()`` wall-clock timestamp (not monotonic) so that SQLite /
        Redis can compare timestamps across process restarts.
    """

    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "pending"
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)


class AbstractJobStoreBackend(ABC):
    """Protocol that every JobStore backend must satisfy.

    Backends are **async-first**: all mutating methods are coroutines.  The
    ``connect`` / ``close`` pair brackets the server lifespan.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open connections / create tables.  Called once at server startup."""

    async def close(self) -> None:
        """Release connections.  Called once at server shutdown."""

    # ── CRUD ──────────────────────────────────────────────────────────────────

    @abstractmethod
    async def create(self) -> JobRecord:
        """Insert a new job in ``pending`` state and return its record."""

    @abstractmethod
    async def get(self, job_id: str) -> JobRecord | None:
        """Return the record, or ``None`` if absent or TTL-expired."""

    @abstractmethod
    async def set_running(self, job_id: str) -> None:
        """Transition job to ``running``."""

    @abstractmethod
    async def set_done(self, job_id: str, result: Any) -> None:
        """Transition job to ``done`` and store *result*."""

    @abstractmethod
    async def set_failed(self, job_id: str, error: str) -> None:
        """Transition job to ``failed`` and store *error*."""

    @abstractmethod
    async def delete_expired(self, ttl_seconds: float) -> int:
        """Remove all jobs at least *ttl_seconds* old (so 0 removes them all).  Return count deleted.

        A backend that expires jobs natively (Redis) may ignore *ttl_seconds* and return 0.
        """
