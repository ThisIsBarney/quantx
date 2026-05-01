"""Tests for M103 result cache — ResultCache, CacheConfig, GET /cache/{key}.

Unit tests for ResultCache and integration tests for the /cache endpoint.
Uses httpx.AsyncClient + anyio for the HTTP tests.
"""
from __future__ import annotations

import sys
import os
import time

import pytest
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from server.result_cache import ResultCache, CacheConfig
from server.main import app
from server.routers.jobs import reset_state

pytestmark = pytest.mark.anyio


# ── ResultCache unit tests ────────────────────────────────────────────────────

class TestResultCache:
    def test_get_miss_returns_none(self):
        cache = ResultCache()
        assert cache.get("nonexistent") is None

    def test_set_and_get(self):
        cache = ResultCache()
        cache.set("k1", {"result": 42})
        assert cache.get("k1") == {"result": 42}

    def test_different_keys_independent(self):
        cache = ResultCache()
        cache.set("k1", "value1")
        cache.set("k2", "value2")
        assert cache.get("k1") == "value1"
        assert cache.get("k2") == "value2"

    def test_overwrite_key(self):
        cache = ResultCache()
        cache.set("k", "old")
        cache.set("k", "new")
        assert cache.get("k") == "new"

    def test_clear_empties_cache(self):
        cache = ResultCache()
        cache.set("k", "v")
        cache.clear()
        assert cache.get("k") is None
        assert len(cache) == 0

    def test_len_tracks_entries(self):
        cache = ResultCache()
        assert len(cache) == 0
        cache.set("a", 1)
        cache.set("b", 2)
        assert len(cache) == 2

    def test_ttl_expiry(self):
        cache = ResultCache(CacheConfig(ttl_seconds=0.05))
        cache.set("k", "v")
        assert cache.get("k") == "v"
        time.sleep(0.1)
        assert cache.get("k") is None

    def test_max_entries_evicts_oldest(self):
        cache = ResultCache(CacheConfig(max_entries=3))
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("d") == 4

    def test_max_entries_size_never_exceeds_limit(self):
        cache = ResultCache(CacheConfig(max_entries=5))
        for i in range(10):
            cache.set(f"k{i}", i)
        assert len(cache) <= 5


class TestMakeKey:
    def test_deterministic(self):
        k1 = ResultCache.make_key("ep", {"n": 1})
        k2 = ResultCache.make_key("ep", {"n": 1})
        assert k1 == k2

    def test_different_endpoint_different_key(self):
        k1 = ResultCache.make_key("ep/a", {"n": 1})
        k2 = ResultCache.make_key("ep/b", {"n": 1})
        assert k1 != k2

    def test_different_payload_different_key(self):
        k1 = ResultCache.make_key("ep", {"n": 1})
        k2 = ResultCache.make_key("ep", {"n": 2})
        assert k1 != k2

    def test_payload_key_order_invariant(self):
        """JSON is sorted, so key order in payload dict must not matter."""
        k1 = ResultCache.make_key("ep", {"a": 1, "b": 2})
        k2 = ResultCache.make_key("ep", {"b": 2, "a": 1})
        assert k1 == k2

    def test_returns_64_hex_chars(self):
        k = ResultCache.make_key("ep", {})
        assert len(k) == 64
        assert all(c in "0123456789abcdef" for c in k)


class TestCacheConfig:
    def test_defaults(self):
        cfg = CacheConfig()
        assert cfg.max_entries == 1000
        assert cfg.ttl_seconds == 3600.0

    def test_custom_values(self):
        cfg = CacheConfig(max_entries=50, ttl_seconds=60.0)
        assert cfg.max_entries == 50
        assert cfg.ttl_seconds == 60.0

    def test_env_var_max_entries(self, monkeypatch):
        monkeypatch.setenv("QUANTX_CACHE_MAX_ENTRIES", "42")
        cfg = CacheConfig()
        assert cfg.max_entries == 42

    def test_env_var_ttl(self, monkeypatch):
        monkeypatch.setenv("QUANTX_CACHE_TTL_SECONDS", "120.5")
        cfg = CacheConfig()
        assert cfg.ttl_seconds == 120.5


# ── GET /cache/{key} integration tests ───────────────────────────────────────

class TestCacheEndpoint:
    @pytest.fixture(autouse=True)
    def clean(self):
        reset_state()
        # Clear the global cache singleton used by the /cache endpoint in tests
        from server.main import _global_result_cache
        _global_result_cache.clear()
        yield
        from server.main import _global_result_cache
        _global_result_cache.clear()

    async def test_miss_returns_404(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/cache/nonexistentkey")
        assert r.status_code == 404

    async def test_hit_returns_200(self):
        from server.main import _global_result_cache
        key = ResultCache.make_key("network/repeater-chain", {"n_hops": 0})
        _global_result_cache.set(key, {"ok": True, "test_value": 99})

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get(f"/cache/{key}")
        assert r.status_code == 200
        data = r.json()
        assert data["cache_key"] == key
        assert data["result"]["test_value"] == 99

    async def test_cache_key_matches_make_key(self):
        """The key returned by make_key must be accepted by the /cache endpoint."""
        from server.main import _global_result_cache
        key = ResultCache.make_key("protocols/bb84", {"channel_error_rate": 0.01})
        _global_result_cache.set(key, {"stored": True})

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get(f"/cache/{key}")
        assert r.status_code == 200
        assert r.json()["result"]["stored"] is True
