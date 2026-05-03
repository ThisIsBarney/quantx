"""M103: Async job submission and status-polling endpoints.

Routes
------
POST /jobs/submit             Submit a solver job; returns job_id immediately
GET  /jobs/{job_id}/status    Poll job status (pending | running | done | failed)

The job runner dispatches to registered Python-SDK solver callables in a
thread-pool executor so it never blocks the event loop.  Results are stored
in the module-level ``_store`` and also written to ``_cache`` so that
identical payloads are returned instantly on resubmission.
"""
from __future__ import annotations

import asyncio
import sys
import os
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Make the Python SDK importable when running from the server/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from server.job_store import JobStore
from server.result_cache import ResultCache, shared_result_cache
from server.observability.metrics import (  # M107
    track_job_status,
    track_job_duration,
    update_jobs_active,
)

router = APIRouter(prefix="/jobs", tags=["async-jobs"])

# ── Module-level singletons ───────────────────────────────────────────────────
# _cache uses the shared singleton from result_cache so that job results are
# visible to the GET /cache/{key} endpoint in main.py (P1 fix).

_store: JobStore = JobStore()
_cache: ResultCache = shared_result_cache


def _get_store() -> JobStore:
    return _store


def _get_cache() -> ResultCache:
    return _cache


def reset_state() -> None:
    """Re-initialise store and cache.  Called in tests between test cases."""
    global _store, _cache
    _store = JobStore()
    _cache = ResultCache()


# ── Solver dispatch table ─────────────────────────────────────────────────────

def _run_network_repeater(payload: dict) -> dict:
    from quantx_engine.quantum_network import simulate_repeater_chain
    r = simulate_repeater_chain(
        n_hops=payload.get("n_hops", 0),
        segment_length_km=payload.get("segment_length_km", 10.0),
        attenuation_db_km=payload.get("attenuation_db_km", 0.2),
        dephasing_per_segment=payload.get("dephasing_per_segment", 0.01),
        bsm_fidelity=payload.get("bsm_fidelity", 0.99),
        purification_rounds=payload.get("purification_rounds", 0),
        memory_coherence_time=payload.get("memory_coherence_time"),
        t_segment_s=payload.get("t_segment_s"),
    )
    return {
        "ok": True,
        "end_to_end_fidelity": r.end_to_end_fidelity,
        "hop_fidelities": r.hop_fidelities,
        "total_length_km": r.total_length_km,
        "n_hops": r.n_hops,
    }


def _run_protocols_bb84(payload: dict) -> dict:
    from quantx_engine.quantum_protocols import BB84Protocol
    proto = BB84Protocol(
        channel_error_rate=payload.get("channel_error_rate", 0.01),
        eve_intercept_fraction=payload.get("eve_intercept_fraction", 0.0),
        detection_threshold=payload.get("detection_threshold", 0.11),
        seed=payload.get("seed"),
    )
    r = proto.run()
    return {
        "ok": True,
        "secure_key_rate": r.secure_key_rate,
        "qber": r.qber,
        "n_raw_bits": r.n_raw_bits,
        "n_secure_bits": r.n_secure_bits,
        "eve_detected": r.eve_detected,
    }


def _run_many_body_otoc(payload: dict) -> dict:
    import numpy as np
    from quantx_engine.many_body import compute_otoc
    from quantx_engine.trotter import ising_trotter
    n = payload.get("n_qubits", 4)
    t = payload.get("t", 1.0)
    sim = ising_trotter(n=n, J=payload.get("J", 1.0), h=payload.get("h", 0.5))
    r = compute_otoc(
        sim.hamiltonian,
        op_w=payload.get("op_w", "X"),
        op_v=payload.get("op_v", "Z"),
        site_w=payload.get("site_w", 0),
        site_v=payload.get("site_v", n - 1),
        t=t,
        n_qubits=n,
    )
    return {
        "ok": True,
        "otoc_value": r.otoc_value,
        "fotoc": r.fotoc,
        "t": r.t,
        "n_qubits": r.n_qubits,
    }


# Registry: endpoint-string → synchronous callable(payload) → dict
_SOLVER_REGISTRY: dict[str, Any] = {
    "network/repeater-chain": _run_network_repeater,
    "protocols/bb84": _run_protocols_bb84,
    "many-body/otoc": _run_many_body_otoc,
}


# ── Pydantic models ───────────────────────────────────────────────────────────

class JobSubmitRequest(BaseModel):
    endpoint: str
    payload: dict[str, Any] = {}


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Any = None
    error: str | None = None


# ── Background task ───────────────────────────────────────────────────────────

async def _run_job(job_id: str, endpoint: str, payload: dict) -> None:
    """Execute a solver in a thread-pool executor and update the job record."""
    store = _get_store()
    cache = _get_cache()

    t0 = time.perf_counter()
    update_jobs_active(+1)
    try:
        await store.set_running(job_id)

        # Check result cache first
        cache_key = ResultCache.make_key(endpoint, payload)
        cached = cache.get(cache_key)
        if cached is not None:
            await store.set_done(job_id, cached)
            track_job_status("done")
            track_job_duration(endpoint, time.perf_counter() - t0)
            return

        solver = _SOLVER_REGISTRY.get(endpoint)
        if solver is None:
            await store.set_failed(job_id, f"Unknown endpoint: {endpoint!r}")
            track_job_status("failed")
            return

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, solver, payload)
            cache.set(cache_key, result)
            await store.set_done(job_id, result)
            track_job_status("done")
            track_job_duration(endpoint, time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            await store.set_failed(job_id, str(exc))
            track_job_status("failed")
            track_job_duration(endpoint, time.perf_counter() - t0)
    finally:
        update_jobs_active(-1)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/submit", response_model=JobSubmitResponse, status_code=202)
async def submit_job(req: JobSubmitRequest) -> JobSubmitResponse:
    """Submit a solver job asynchronously.

    The job is enqueued immediately and executed in the background.
    Use ``GET /jobs/{job_id}/status`` to poll for the result.

    Supported endpoints
    -------------------
    * ``network/repeater-chain``
    * ``protocols/bb84``
    * ``many-body/otoc``
    """
    if req.endpoint not in _SOLVER_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown endpoint {req.endpoint!r}. "
                   f"Available: {sorted(_SOLVER_REGISTRY)}",
        )

    record = await _get_store().create()
    asyncio.create_task(_run_job(record.job_id, req.endpoint, req.payload))
    return JobSubmitResponse(job_id=record.job_id, status=record.status)


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def job_status(job_id: str) -> JobStatusResponse:
    """Poll the status of a previously submitted job.

    Returns
    -------
    ``status``   one of ``pending``, ``running``, ``done``, ``failed``
    ``result``   present when status is ``done``
    ``error``    present when status is ``failed``
    """
    record = await _get_store().get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status,
        result=record.result,
        error=record.error,
    )
