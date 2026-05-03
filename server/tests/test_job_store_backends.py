"""M105: Tests for pluggable JobStore backends.

Parametrized over InMemoryBackend and SqliteBackend (temp-file).
RedisBackend tests are skipped unless a Redis server is reachable.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time

import pytest

from server.backends import create_job_store_backend, AbstractJobStoreBackend
from server.backends.memory import InMemoryBackend
from server.backends.sqlite_backend import SqliteBackend
from server.job_store import JobStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_sqlite_backend(ttl: float = 60.0) -> tuple[SqliteBackend, str]:
    """Return (SqliteBackend, tmp_path_str) using a secure temp file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)  # let aiosqlite create it fresh
    return SqliteBackend(path, ttl_seconds=ttl), path


async def _connect(b: AbstractJobStoreBackend) -> AbstractJobStoreBackend:
    await b.connect()
    return b


# ── Core lifecycle tests (run for both backends) ──────────────────────────────

async def _run_lifecycle_tests(b: AbstractJobStoreBackend) -> None:
    """Exercise the full CRUD lifecycle on any backend."""
    # create
    r = await b.create()
    assert r.status == "pending"
    assert len(r.job_id) > 0

    # get pending
    fetched = await b.get(r.job_id)
    assert fetched is not None
    assert fetched.job_id == r.job_id
    assert fetched.status == "pending"

    # running
    await b.set_running(r.job_id)
    assert (await b.get(r.job_id)).status == "running"

    # done
    result_data = {"ok": True, "energy": -3.14, "items": [1, 2, 3]}
    await b.set_done(r.job_id, result_data)
    done = await b.get(r.job_id)
    assert done.status == "done"
    assert done.result == result_data

    # failed (new job)
    r2 = await b.create()
    await b.set_failed(r2.job_id, "boom")
    failed = await b.get(r2.job_id)
    assert failed.status == "failed"
    assert failed.error == "boom"
    assert failed.result is None

    # unknown job → None
    assert await b.get("no-such-id") is None

    # noop for missing job
    await b.set_done("no-such-id", {"x": 1})   # must not raise
    await b.set_failed("no-such-id", "err")    # must not raise


@pytest.mark.anyio
async def test_in_memory_full_lifecycle():
    b = InMemoryBackend(ttl_seconds=60.0)
    await b.connect()
    await _run_lifecycle_tests(b)
    await b.close()


@pytest.mark.anyio
async def test_sqlite_full_lifecycle():
    b, path = _make_sqlite_backend()
    await b.connect()
    try:
        await _run_lifecycle_tests(b)
    finally:
        await b.close()
        if os.path.exists(path):
            os.unlink(path)


@pytest.mark.anyio
async def test_unique_ids_memory():
    b = InMemoryBackend()
    await b.connect()
    ids = {(await b.create()).job_id for _ in range(10)}
    assert len(ids) == 10
    await b.close()


@pytest.mark.anyio
async def test_unique_ids_sqlite():
    b, path = _make_sqlite_backend()
    await b.connect()
    try:
        ids = {(await b.create()).job_id for _ in range(10)}
        assert len(ids) == 10
    finally:
        await b.close()
        if os.path.exists(path):
            os.unlink(path)


# ── TTL / eviction tests ──────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_delete_expired_memory():
    b = InMemoryBackend(ttl_seconds=60.0)
    await b.connect()
    r = await b.create()
    count = await b.delete_expired(ttl_seconds=0.0)  # everything is expired
    assert count >= 1
    assert await b.get(r.job_id) is None
    await b.close()


@pytest.mark.anyio
async def test_delete_expired_sqlite():
    b, path = _make_sqlite_backend()
    await b.connect()
    try:
        r = await b.create()
        count = await b.delete_expired(ttl_seconds=0.0)
        assert count >= 1
        assert await b.get(r.job_id) is None
    finally:
        await b.close()
        if os.path.exists(path):
            os.unlink(path)


@pytest.mark.anyio
async def test_ttl_eviction_on_get_memory():
    b = InMemoryBackend(ttl_seconds=0.01)
    await b.connect()
    r = await b.create()
    await asyncio.sleep(0.05)
    assert await b.get(r.job_id) is None
    await b.close()


@pytest.mark.anyio
async def test_ttl_eviction_on_get_sqlite():
    b, path = _make_sqlite_backend(ttl=0.01)
    await b.connect()
    try:
        r = await b.create()
        await asyncio.sleep(0.05)
        assert await b.get(r.job_id) is None
    finally:
        await b.close()
        if os.path.exists(path):
            os.unlink(path)


# ── SQLite-specific: persistence across connections ───────────────────────────

@pytest.mark.anyio
async def test_sqlite_survives_reconnect():
    b1, path = _make_sqlite_backend()
    await b1.connect()
    record = await b1.create()
    await b1.set_done(record.job_id, {"energy": -3.14})
    await b1.close()

    b2 = SqliteBackend(path)
    await b2.connect()
    try:
        fetched = await b2.get(record.job_id)
        assert fetched is not None
        assert fetched.status == "done"
        assert fetched.result["energy"] == pytest.approx(-3.14)
    finally:
        await b2.close()
        if os.path.exists(path):
            os.unlink(path)


@pytest.mark.anyio
async def test_sqlite_wal_mode():
    b, path = _make_sqlite_backend()
    await b.connect()
    try:
        import aiosqlite
        async with b._db.execute("PRAGMA journal_mode;") as cur:
            row = await cur.fetchone()
        assert row[0].lower() == "wal"
    finally:
        await b.close()
        if os.path.exists(path):
            os.unlink(path)


@pytest.mark.anyio
async def test_sqlite_requires_connect():
    b, path = _make_sqlite_backend()
    with pytest.raises(RuntimeError, match="not connected"):
        await b.create()
    if os.path.exists(path):
        os.unlink(path)


# ── create_job_store_backend factory ─────────────────────────────────────────

def test_factory_memory_url():
    b = create_job_store_backend("memory://")
    assert isinstance(b, InMemoryBackend)


def test_factory_none_url():
    b = create_job_store_backend(None)
    assert isinstance(b, InMemoryBackend)


def test_factory_empty_url():
    b = create_job_store_backend("")
    assert isinstance(b, InMemoryBackend)


def test_factory_sqlite_url():
    # Test that the absolute path is preserved: sqlite:///var/db.db → /var/db.db
    b = create_job_store_backend("sqlite:///var/quantx/jobs.db")
    assert isinstance(b, SqliteBackend)
    assert b.path == "/var/quantx/jobs.db"


def test_factory_sqlite_url_relative():
    b = create_job_store_backend("sqlite:///relative/jobs.db")
    assert isinstance(b, SqliteBackend)
    assert b.path == "/relative/jobs.db"


def test_factory_sqlite_aiosqlite_url():
    b = create_job_store_backend("sqlite+aiosqlite:///var/db.db")
    assert isinstance(b, SqliteBackend)
    assert b.path == "/var/db.db"


def test_factory_unknown_scheme_raises():
    with pytest.raises(ValueError, match="Unknown job store URL scheme"):
        create_job_store_backend("postgres://localhost/quantx")


# ── JobStore wrapper ─────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_job_store_wrapper_default_is_memory():
    store = JobStore()
    assert isinstance(store.backend, InMemoryBackend)
    await store.backend.connect()
    r = await store.create()
    assert r.status == "pending"
    await store.backend.close()


@pytest.mark.anyio
async def test_job_store_wrapper_with_explicit_backend():
    b = InMemoryBackend()
    await b.connect()
    store = JobStore(backend=b)
    r = await store.create()
    await store.set_running(r.job_id)
    r2 = await store.get(r.job_id)
    assert r2.status == "running"
    await b.close()


@pytest.mark.anyio
async def test_job_store_env_var_sqlite(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    # Build a cross-platform sqlite URL: use forward slashes and ensure a
    # leading "/" so urlparse strips exactly one slash.  On Windows, drive
    # paths like "C:/..." become "/C:/..." and the factory strips the extra
    # leading slash via a regex (sys.platform == "win32" guard).
    fwd = path.replace("\\", "/")
    if not fwd.startswith("/"):
        fwd = "/" + fwd
    monkeypatch.setenv("QUANTX_JOB_STORE_URL", f"sqlite://{fwd}")
    store = JobStore()
    assert isinstance(store.backend, SqliteBackend)
    await store.backend.connect()
    r = await store.create()
    assert r.job_id
    await store.backend.close()
    if os.path.exists(path):
        os.unlink(path)


# ── Redis backend: skip if unavailable ───────────────────────────────────────

@pytest.mark.anyio
async def test_redis_backend_skip_if_unavailable():
    pytest.importorskip("redis", reason="redis package not installed")
    from server.backends.redis_backend import RedisBackend
    b = RedisBackend("redis://localhost:6379/15")
    try:
        await b.connect()
    except Exception:
        pytest.skip("Redis server not reachable")
    try:
        record = await b.create()
        assert record.status == "pending"
        await b.set_done(record.job_id, {"ok": True})
        fetched = await b.get(record.job_id)
        assert fetched.status == "done"
    finally:
        await b.close()
