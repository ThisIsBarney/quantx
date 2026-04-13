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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the native engine library at startup."""
    lib_path = os.environ.get("QUANTX_LIB_PATH")
    app.state.engine = Engine(lib_path)
    yield


app = FastAPI(
    title="QuantX Engine API",
    description=(
        "REST API for the QuantX quantum simulation engine. "
        "Exposes 37 physics solvers (VQE, Hubbard, toric code, SYK, ...) "
        "and a gate-model circuit simulator via HTTP."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _engine(request: Request) -> Engine:
    return request.app.state.engine


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
