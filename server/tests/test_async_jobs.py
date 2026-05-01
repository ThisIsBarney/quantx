"""Tests for M103 async job queue — POST /jobs/submit + GET /jobs/{id}/status.

Uses httpx.AsyncClient + anyio so the asyncio event loop is live and
asyncio.create_task() can drive background job execution.
"""
from __future__ import annotations

import asyncio
import sys
import os

import pytest
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from server.main import app
from server.routers.jobs import reset_state

pytestmark = pytest.mark.anyio


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _wait_for_done(
    client: httpx.AsyncClient,
    job_id: str,
    timeout: float = 10.0,
    interval: float = 0.05,
) -> dict:
    """Poll /jobs/{job_id}/status until status is 'done' or 'failed'."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        r = await client.get(f"/jobs/{job_id}/status")
        assert r.status_code == 200, r.text
        data = r.json()
        if data["status"] in ("done", "failed"):
            return data
        await asyncio.sleep(interval)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")


@pytest.fixture(autouse=True)
def clean_state():
    """Reset module-level singletons between tests."""
    reset_state()
    yield
    reset_state()


# ── Submit endpoint ───────────────────────────────────────────────────────────

class TestJobSubmit:
    async def test_submit_returns_202(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/jobs/submit", json={
                "endpoint": "network/repeater-chain",
                "payload": {"n_hops": 0, "segment_length_km": 10.0},
            })
        assert r.status_code == 202

    async def test_submit_returns_job_id(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/jobs/submit", json={
                "endpoint": "network/repeater-chain",
                "payload": {},
            })
        data = r.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str) and len(data["job_id"]) == 36  # UUID

    async def test_submit_initial_status_pending_or_running(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/jobs/submit", json={
                "endpoint": "network/repeater-chain",
                "payload": {},
            })
        data = r.json()
        assert data["status"] in ("pending", "running")

    async def test_submit_unknown_endpoint_returns_400(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/jobs/submit", json={
                "endpoint": "nonexistent/solver",
                "payload": {},
            })
        assert r.status_code == 400

    async def test_submit_empty_payload_uses_defaults(self):
        """Submitting with no payload should still succeed using solver defaults."""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/jobs/submit", json={
                "endpoint": "network/repeater-chain",
            })
        assert r.status_code == 202


# ── Status polling ────────────────────────────────────────────────────────────

class TestJobStatus:
    async def test_status_404_for_unknown_job(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.get("/jobs/00000000-0000-0000-0000-000000000000/status")
        assert r.status_code == 404

    async def test_job_completes_with_done_status(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            sub = await client.post("/jobs/submit", json={
                "endpoint": "network/repeater-chain",
                "payload": {"n_hops": 0, "segment_length_km": 5.0},
            })
            job_id = sub.json()["job_id"]
            data = await _wait_for_done(client, job_id)
        assert data["status"] == "done"

    async def test_done_result_contains_fidelity(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            sub = await client.post("/jobs/submit", json={
                "endpoint": "network/repeater-chain",
                "payload": {"n_hops": 1, "segment_length_km": 5.0},
            })
            job_id = sub.json()["job_id"]
            data = await _wait_for_done(client, job_id)
        assert data["result"] is not None
        assert "end_to_end_fidelity" in data["result"]

    async def test_bb84_job_completes(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            sub = await client.post("/jobs/submit", json={
                "endpoint": "protocols/bb84",
                "payload": {"channel_error_rate": 0.01, "seed": 42},
            })
            job_id = sub.json()["job_id"]
            data = await _wait_for_done(client, job_id)
        assert data["status"] == "done"
        assert "secure_key_rate" in data["result"]

    async def test_otoc_job_completes(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            sub = await client.post("/jobs/submit", json={
                "endpoint": "many-body/otoc",
                "payload": {"n_qubits": 4, "t": 0.5},
            })
            job_id = sub.json()["job_id"]
            data = await _wait_for_done(client, job_id)
        assert data["status"] == "done"
        assert "otoc_value" in data["result"]
        assert "fotoc" in data["result"]

    async def test_multiple_jobs_independent(self):
        """Two concurrent jobs should each complete independently."""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r1 = await client.post("/jobs/submit", json={
                "endpoint": "network/repeater-chain",
                "payload": {"n_hops": 0},
            })
            r2 = await client.post("/jobs/submit", json={
                "endpoint": "protocols/bb84",
                "payload": {"channel_error_rate": 0.01, "seed": 1},
            })
            jid1 = r1.json()["job_id"]
            jid2 = r2.json()["job_id"]
            d1, d2 = await asyncio.gather(
                _wait_for_done(client, jid1),
                _wait_for_done(client, jid2),
            )
        assert d1["status"] == "done"
        assert d2["status"] == "done"
        assert jid1 != jid2

    async def test_second_identical_submission_uses_cache(self):
        """Submitting the same payload twice returns the cached result on the second call."""
        payload = {"endpoint": "network/repeater-chain", "payload": {"n_hops": 0, "segment_length_km": 7.0}}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r1 = await client.post("/jobs/submit", json=payload)
            jid1 = r1.json()["job_id"]
            d1 = await _wait_for_done(client, jid1)

            r2 = await client.post("/jobs/submit", json=payload)
            jid2 = r2.json()["job_id"]
            d2 = await _wait_for_done(client, jid2)

        assert d1["status"] == "done"
        assert d2["status"] == "done"
        assert d1["result"]["end_to_end_fidelity"] == d2["result"]["end_to_end_fidelity"]
