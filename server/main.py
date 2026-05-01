"""QuantX REST API server — FastAPI wrapper around the QuantX engine SDK."""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Allow running from repo root or server/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from quantx_engine import Engine, CircuitError

from server.routers.network import router as network_router
from server.routers.protocols import router as protocols_router
from server.routers.pulse import router as pulse_router
from server.routers.advanced import router as advanced_router
from server.routers.jobs import router as jobs_router          # M103
from server.result_cache import ResultCache, shared_result_cache  # M103

# M103: re-export the shared singleton so the /cache endpoint and the jobs
# router always operate on the same cache object (P1 fix).
_global_result_cache: ResultCache = shared_result_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the native engine library at startup."""
    lib_path = os.environ.get("QUANTX_LIB_PATH")
    app.state.engine = Engine(lib_path)
    app.state.result_cache = _global_result_cache      # M103
    yield


app = FastAPI(
    title="QuantX Engine API",
    description=(
        "REST API for the QuantX quantum simulation engine. "
        "Exposes 37 C++ physics solvers (VQE, Hubbard, toric code, SYK, ...), "
        "a gate-model circuit simulator, quantum network simulation (M96), "
        "quantum communication protocols (M97), pulse-level simulation (M89), "
        "and advanced solvers (Floquet M92, TDMRG M94, noise-aware VQE M95, "
        "fault-tolerant qubits M91)."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# M98: new Python-SDK routers (no native library required)
app.include_router(network_router)
app.include_router(protocols_router)
app.include_router(pulse_router)
app.include_router(advanced_router)
# M103: async job queue
app.include_router(jobs_router)


def _engine(request: Request) -> Engine:
    return request.app.state.engine


# ── M103: result cache endpoint ───────────────────────────────────────────────

@app.get("/cache/{cache_key}", tags=["cache"])
async def get_cached_result(cache_key: str, request: Request) -> Any:
    """Return a previously cached solver result by its SHA-256 cache key.

    The cache key is the value returned by ``ResultCache.make_key(endpoint,
    payload)``.  Returns 404 if the key is absent or the entry has expired.
    """
    # Use app.state cache if lifespan ran, otherwise fall back to module singleton
    cache: ResultCache = getattr(request.app.state, "result_cache", _global_result_cache)
    value = cache.get(cache_key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Cache key {cache_key!r} not found")
    return {"cache_key": cache_key, "result": value}


@app.get("/health", tags=["meta"])
async def health(request: Request) -> dict:
    """Return server health and engine version."""
    try:
        version = _engine(request).version()
    except Exception:
        version = "unknown"
    return {"ok": True, "version": version}


@app.get("/solvers", tags=["solvers"])
async def list_solvers(request: Request) -> dict:
    """Return the list of all available physics solver names."""
    return {"solvers": _engine(request).list_solvers()}


@app.post("/solve/{solver}", tags=["solvers"])
async def run_solver(solver: str, request: Request) -> Any:
    """Run a named physics solver with the provided JSON options body."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        result = _engine(request).run_solver(solver, body if isinstance(body, dict) else {})
    except CircuitError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error_code": exc.code, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})
    # P1: engine wraps failures into {"ok": false, ...} without raising — translate to 4xx/5xx
    if isinstance(result, dict) and result.get("ok") is False:
        error_kind = result.get("error_kind", "")
        status = 400 if error_kind in ("circuit", "validation") else 422
        raise HTTPException(status_code=status, detail=result)
    return result


@app.post("/run", tags=["simulation"])
async def run_simulation(request: Request) -> Any:
    """Run a Hamiltonian simulation (legacy problem/options schema)."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail={"ok": False, "error": "request body must be a JSON object"})
    problem = body.get("problem", body)
    options = body.get("options", {})
    try:
        result = _engine(request).run(problem, options)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})
    return result


@app.post("/circuit", tags=["circuit"])
async def run_circuit(request: Request) -> Any:
    """Run a gate-model circuit simulation."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    circuit = body.get("circuit", body)
    options = body.get("options", {})
    try:
        result = _engine(request).run_circuit(circuit, options)
    except CircuitError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error_code": exc.code, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})
    return result


@app.get("/circuit/capabilities", tags=["circuit"])
async def circuit_capabilities(request: Request) -> Any:
    """Return the gate-model capability metadata."""
    try:
        return _engine(request).circuit_capabilities()
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})


@app.post("/circuit/validate", tags=["circuit"])
async def validate_circuit(request: Request) -> Any:
    """Validate a circuit JSON payload."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        return _engine(request).validate_circuit(body if isinstance(body, dict) else {})
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})


@app.post("/problem/validate", tags=["simulation"])
async def validate_problem(request: Request) -> Any:
    """Validate a problem JSON payload."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        return _engine(request).validate_problem(body if isinstance(body, dict) else {})
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
