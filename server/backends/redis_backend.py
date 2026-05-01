"""M105: Redis-backed JobStore (redis.asyncio) — optional, enables horizontal scaling.

Install the extra dependency::

    pip install redis

"""
from __future__ import annotations

import json
import time
from typing import Any

from server.backends.base import AbstractJobStoreBackend, JobRecord

try:
    import redis.asyncio as aioredis  # type: ignore[import-untyped]
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


def _require_redis() -> None:
    if not _REDIS_AVAILABLE:
        raise ImportError(
            "redis.asyncio is required for RedisBackend. "
            "Install it with: pip install redis"
        )


_KEY_PREFIX = "quantx:job:"


class RedisBackend(AbstractJobStoreBackend):
    """Redis-backed job store using redis.asyncio.

    Each job is stored as a Redis hash at key ``quantx:job:{job_id}``.
    The Redis key TTL mirrors *ttl_seconds* so that expiry is handled
    natively by Redis without a separate eviction sweep.

    Parameters
    ----------
    url:
        Redis connection URL, e.g. ``redis://localhost:6379/0``.
    ttl_seconds:
        Lifetime of each job entry in Redis.
    """

    def __init__(self, url: str = "redis://localhost:6379/0", ttl_seconds: float = 3600.0) -> None:
        _require_redis()
        self.url = url
        self.ttl_seconds = ttl_seconds
        self._client: Any = None  # aioredis.Redis

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        _require_redis()
        self._client = aioredis.from_url(self.url, decode_responses=True)
        # Ping to verify connectivity at startup
        await self._client.ping()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def create(self) -> JobRecord:
        self._require_connected()
        record = JobRecord()
        key = _KEY_PREFIX + record.job_id
        mapping = {
            "job_id": record.job_id,
            "status": "pending",
            "result_json": "",
            "error": "",
            "created_at": str(record.created_at),
        }
        await self._client.hset(key, mapping=mapping)
        await self._client.expire(key, int(self.ttl_seconds))
        return record

    async def get(self, job_id: str) -> JobRecord | None:
        self._require_connected()
        key = _KEY_PREFIX + job_id
        data = await self._client.hgetall(key)
        if not data:
            return None
        return self._dict_to_record(data)

    async def set_running(self, job_id: str) -> None:
        self._require_connected()
        key = _KEY_PREFIX + job_id
        await self._client.hset(key, "status", "running")

    async def set_done(self, job_id: str, result: Any) -> None:
        self._require_connected()
        key = _KEY_PREFIX + job_id
        await self._client.hset(key, mapping={
            "status": "done",
            "result_json": json.dumps(result, default=str),
        })

    async def set_failed(self, job_id: str, error: str) -> None:
        self._require_connected()
        key = _KEY_PREFIX + job_id
        await self._client.hset(key, mapping={"status": "failed", "error": error})

    async def delete_expired(self, ttl_seconds: float) -> int:
        """Redis handles TTL natively; this is a no-op for the Redis backend."""
        return 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _require_connected(self) -> None:
        if self._client is None:
            raise RuntimeError("RedisBackend is not connected. Call await backend.connect() first.")

    @staticmethod
    def _dict_to_record(data: dict[str, str]) -> JobRecord:
        result_raw = data.get("result_json", "")
        return JobRecord(
            job_id=data["job_id"],
            status=data["status"],
            result=json.loads(result_raw) if result_raw else None,
            error=data.get("error") or None,
            created_at=float(data.get("created_at", time.time())),
        )
