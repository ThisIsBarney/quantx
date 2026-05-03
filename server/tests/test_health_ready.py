"""M108: Tests for the extended /health and new /ready endpoints."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── App fixture ───────────────────────────────────────────────────────────────

_mock_engine = MagicMock()
_mock_engine.version.return_value = "0.0.0-test"
_mock_engine.list_solvers.return_value = []

with patch("quantx_engine.Engine", return_value=_mock_engine):
    from server.main import app


def _make_client(backend=None):
    """Return a TestClient with ``app.state`` pre-configured."""
    app.state.engine = _mock_engine
    app.state.job_store_backend = backend
    return TestClient(app, raise_server_exceptions=False)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_ok_no_backend():
    """Health returns 200 with expected keys even when no backend in state."""
    app.state.job_store_backend = None
    client = _make_client(backend=None)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "version" in body
    assert "db_ok" in body
    assert "cache_ok" in body
    assert "redis_ok" in body


def test_health_db_ok_in_memory_backend():
    """InMemoryBackend has no _client; db_ok should be True (no ping needed)."""
    in_mem = MagicMock(spec=[])  # no _client attribute
    client = _make_client(backend=in_mem)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["db_ok"] is True
    assert body["redis_ok"] is None  # not configured


def test_health_redis_backend_ok():
    """When Redis backend _client.ping() succeeds, redis_ok=True."""
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    backend = MagicMock()
    backend._client = mock_redis

    client = _make_client(backend=backend)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["db_ok"] is True
    assert body["redis_ok"] is True


def test_health_redis_backend_down():
    """When Redis ping() raises, db_ok=False and redis_ok=False."""
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(side_effect=ConnectionError("connection refused"))
    backend = MagicMock()
    backend._client = mock_redis

    client = _make_client(backend=backend)
    r = client.get("/health")
    assert r.status_code == 200  # /health always returns 200 (report-only)
    body = r.json()
    assert body["db_ok"] is False
    assert body["redis_ok"] is False


# ── /ready ────────────────────────────────────────────────────────────────────

def test_ready_no_backend_returns_503():
    """/ready returns 503 when app.state.job_store_backend is not set."""
    client = _make_client(backend=None)
    r = client.get("/ready")
    assert r.status_code == 503
    # FastAPI wraps HTTPException detail in {"detail": ...}
    detail = r.json().get("detail", r.json())
    assert detail["ok"] is False


def test_ready_in_memory_backend_returns_200():
    """InMemoryBackend (no _client) passes readiness check."""
    in_mem = MagicMock(spec=[])  # no _client
    client = _make_client(backend=in_mem)
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_ready_redis_backend_ok():
    """Redis backend ping() succeeds → 200."""
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    backend = MagicMock()
    backend._client = mock_redis

    client = _make_client(backend=backend)
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_ready_redis_backend_down():
    """Redis backend ping() raises → 503."""
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(side_effect=ConnectionError("refused"))
    backend = MagicMock()
    backend._client = mock_redis

    client = _make_client(backend=backend)
    r = client.get("/ready")
    assert r.status_code == 503
    # FastAPI wraps HTTPException detail in {"detail": ...}
    detail = r.json().get("detail", r.json())
    assert detail["ok"] is False
    assert "backend unreachable" in detail["reason"]
