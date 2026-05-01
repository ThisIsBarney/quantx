"""M103: SHA-256 keyed result cache with LRU eviction and TTL.

Usage::

    cache = ResultCache(CacheConfig(max_entries=500, ttl_seconds=1800))

    key = ResultCache.make_key("network/repeater-chain", {"n_hops": 2})
    cached = cache.get(key)
    if cached is None:
        result = run_solver(...)
        cache.set(key, result)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from typing import Any


@dataclasses.dataclass
class CacheConfig:
    """Tunable parameters for :class:`ResultCache`.

    Environment variable overrides (applied at construction time):

    ``QUANTX_CACHE_MAX_ENTRIES``  — integer, maximum cache size.
    ``QUANTX_CACHE_TTL_SECONDS``  — float, entry lifetime in seconds.
    """

    max_entries: int = 1000
    ttl_seconds: float = 3600.0

    def __post_init__(self) -> None:
        if env_max := os.environ.get("QUANTX_CACHE_MAX_ENTRIES"):
            self.max_entries = int(env_max)
        if env_ttl := os.environ.get("QUANTX_CACHE_TTL_SECONDS"):
            self.ttl_seconds = float(env_ttl)


class ResultCache:
    """In-memory result cache keyed by SHA-256 digest.

    Eviction policy: when the cache is full, the oldest entry by insertion
    time is removed (insertion-order LRU via dict ordering, Python 3.7+).

    This class is **not** thread-safe on its own; the ``CacheMiddleware``
    wraps accesses under a lock where needed.
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        # Maps cache_key → (value, inserted_at_monotonic)
        self._store: dict[str, tuple[Any, float]] = {}

    # ── Key generation ────────────────────────────────────────────────────────

    @staticmethod
    def make_key(endpoint: str, payload: Any) -> str:
        """Compute a deterministic SHA-256 cache key.

        Parameters
        ----------
        endpoint:
            Route path, e.g. ``"network/repeater-chain"``.
        payload:
            JSON-serialisable request body (dict, list, …).

        Returns
        -------
        str
            64-character lowercase hex digest.
        """
        raw = json.dumps(
            {"endpoint": endpoint, "payload": payload},
            sort_keys=True,
            default=str,
        ).encode()
        return hashlib.sha256(raw).hexdigest()

    # ── Cache operations ──────────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        """Return the cached value for *key*, or ``None`` if absent / expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.monotonic() - ts > self.config.ttl_seconds:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key*, evicting the oldest entry if at capacity."""
        if len(self._store) >= self.config.max_entries:
            # Python dicts preserve insertion order → first key is oldest
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[key] = (value, time.monotonic())

    def clear(self) -> None:
        """Remove all entries (useful in tests)."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
