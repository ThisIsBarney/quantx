"""M105: Pluggable JobStore backend system.

Usage::

    from server.backends import create_job_store_backend

    # Default in-memory (no external services)
    store = create_job_store_backend("memory://")

    # SQLite — persists across restarts
    store = create_job_store_backend("sqlite:///var/quantx/jobs.db")

    # Redis — horizontal scaling
    store = create_job_store_backend("redis://localhost:6379/0")

The URL scheme is resolved from ``QUANTX_JOB_STORE_URL`` at server startup.
"""
from __future__ import annotations

import re
import sys
from urllib.parse import urlparse

from server.backends.base import AbstractJobStoreBackend, JobRecord
from server.backends.memory import InMemoryBackend
from server.backends.sqlite_backend import SqliteBackend


def create_job_store_backend(url: str | None = None) -> AbstractJobStoreBackend:
    """Instantiate the appropriate backend from a URL string.

    Parameters
    ----------
    url:
        ``memory://``                  — in-memory, default when *url* is None/empty.
        ``sqlite:///path/to/jobs.db``  — SQLite file (aiosqlite).
        ``redis://host:port/db``       — Redis (redis.asyncio, optional dep).

    Returns
    -------
    AbstractJobStoreBackend
        Unconnected backend instance.  Call ``await backend.connect()`` before
        use (the server lifespan handles this).
    """
    if not url or url.startswith("memory://"):
        return InMemoryBackend()

    if url.startswith("sqlite"):
        # Accept both "sqlite:///path" and "sqlite+aiosqlite:///path".
        # Use urlparse so that "sqlite:///var/db.db" yields path "/var/db.db"
        # rather than stripping the leading "/" via a naive split.
        path = urlparse(url).path
        # On Windows urlparse gives "/C:/path" for absolute drive paths;
        # strip the spurious leading slash so os.path operations work correctly.
        if sys.platform == "win32":
            path = re.sub(r"^/([A-Za-z]:)", r"\1", path)
        return SqliteBackend(path)

    if url.startswith("redis"):
        # Lazy import so redis is optional
        try:
            from server.backends.redis_backend import RedisBackend  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "redis.asyncio is required for the Redis backend. "
                "Install it with: pip install redis"
            ) from exc
        return RedisBackend(url)

    raise ValueError(
        f"Unknown job store URL scheme: {url!r}. "
        "Supported: memory://, sqlite:///path, redis://host:port/db"
    )


__all__ = [
    "AbstractJobStoreBackend",
    "JobRecord",
    "InMemoryBackend",
    "SqliteBackend",
    "create_job_store_backend",
]
