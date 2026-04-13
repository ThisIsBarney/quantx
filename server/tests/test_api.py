"""Tests for the QuantX REST API server."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Patch Engine before importing app to avoid needing the native .so/.dll
_mock_engine = MagicMock()
_mock_engine.version.return_value = "0.0.0-test"
_mock_engine.list_solvers.return_value = ["vqe", "toric"]
_mock_engine.run_solver.return_value = {"ok": True, "E0": -1.0}
_mock_engine.run.return_value = {"ok": True}
_mock_engine.run_circuit.return_value = {"ok": True, "counts": {}}
_mock_engine.circuit_capabilities.return_value = {"backend": "test", "max_qubits": 32}
_mock_engine.validate_circuit.return_value = {"ok": True, "validated": "circuit"}
_mock_engine.validate_problem.return_value = {"ok": True, "validated": "problem"}

with patch("quantx_engine.Engine", return_value=_mock_engine):
    from server.main import app

app.state.engine = _mock_engine
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert "version" in r.json()


def test_list_solvers():
    r = client.get("/solvers")
    assert r.status_code == 200
    solvers = r.json()["solvers"]
    assert isinstance(solvers, list)
    assert "vqe" in solvers


def test_solve_known_solver():
    r = client.post("/solve/toric", json={"L": 2, "scenario": "spectrum"})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_solve_exception_returns_500():
    _mock_engine.run_solver.side_effect = Exception("unexpected internal error")
    r = client.post("/solve/bad_solver", json={})
    assert r.status_code == 500
    _mock_engine.run_solver.side_effect = None
    _mock_engine.run_solver.return_value = {"ok": True, "E0": -1.0}


def test_solve_engine_error_json_returns_4xx():
    """Engine ok=False result (e.g. unknown solver path) must not return 200."""
    _mock_engine.run_solver.return_value = {"ok": False, "error_kind": "runtime", "error": "unknown solver: bad"}
    r = client.post("/solve/bad_solver", json={})
    assert r.status_code in (400, 422)
    _mock_engine.run_solver.return_value = {"ok": True, "E0": -1.0}


def test_run_simulation_non_object_body_returns_400():
    r = client.post("/run", content=b"[]", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_run_simulation():
    r = client.post("/run", json={"problem": {"type": "ising"}, "options": {}})
    assert r.status_code == 200


def test_run_circuit():
    r = client.post("/circuit", json={"circuit": {"qubits": 2, "operations": []}})
    assert r.status_code == 200


def test_circuit_capabilities():
    r = client.get("/circuit/capabilities")
    assert r.status_code == 200
    assert "backend" in r.json()


def test_validate_circuit():
    r = client.post("/circuit/validate", json={"qubits": 2})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_validate_problem():
    r = client.post("/problem/validate", json={"type": "ising"})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_openapi_schema_includes_all_routes():
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/health" in paths
    assert "/solvers" in paths
    assert "/solve/{solver}" in paths
    assert "/circuit" in paths
    assert "/circuit/capabilities" in paths
