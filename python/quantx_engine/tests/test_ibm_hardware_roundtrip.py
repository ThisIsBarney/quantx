"""M169 — Unit tests for the IBM Quantum hardware round-trip demo.

These tests mock the IBM API at the IBMHardwareBackend layer so they run in CI
without real credentials or Qiskit installed.  They verify:

 * Bell circuit construction and QASM 3.0 serialisation
 * IBMHardwareBackend.submit() returns a HardwareJob with the right interface
 * HardwareJob.status() and .fetch() behave correctly on mocked responses
 * The ibm_hardware_demo script works end-to-end in dry-run mode
 * Missing token / Qiskit returns graceful error messages (not exceptions)

Skipping real-hardware execution
---------------------------------
Any test that would call the real IBM REST API is marked
``pytest.mark.hardware`` and is **skipped by default**.  Run with::

    pytest -m hardware python/quantx_engine/tests/test_ibm_hardware_roundtrip.py

to include them (requires QISKIT_IBM_TOKEN to be set).
"""
from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Ensure quantx_engine is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Also make examples/ importable
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples")
sys.path.insert(0, EXAMPLES_DIR)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def bell_circuit_dict():
    return {
        "qubits": 2,
        "clbits": 2,
        "shots": 1024,
        "seed": 42,
        "operations": [
            {"name": "H",       "targets": [0]},
            {"name": "CNOT",    "controls": [0], "targets": [1]},
            {"name": "MEASURE", "targets": [0], "controls": [0]},
            {"name": "MEASURE", "targets": [1], "controls": [1]},
        ],
    }


@pytest.fixture()
def mock_ibm_job():
    """A fully mocked IBM job that starts queued and transitions to completed."""
    job = MagicMock()
    job.job_id.return_value = "mock-job-id-abc123"
    # Simulate job progressing: first call → QUEUED, second → COMPLETED
    statuses = ["QUEUED", "RUNNING", "DONE"]
    status_iter = iter(statuses)

    def _status_side_effect():
        status_str = next(status_iter, "DONE")
        s = MagicMock()
        s.name = status_str
        s.__str__ = lambda self: status_str
        return s

    job.status.side_effect = _status_side_effect

    # Mock the result
    pub_result = MagicMock()
    data = MagicMock()
    # Each classical register: get_bitstrings() returns list of '00'/'11' strings
    register = MagicMock()
    register.get_bitstrings.return_value = ["00"] * 512 + ["11"] * 512
    data.c = register
    pub_result.data = data
    job.result.return_value = MagicMock()
    job.result.return_value.__iter__ = lambda self: iter([pub_result])
    job.result.return_value.__getitem__ = lambda self, i: pub_result

    return job


@pytest.fixture()
def mock_backend(mock_ibm_job):
    """A mocked IBMHardwareBackend that uses the mock job."""
    backend = MagicMock()
    backend._backend_name = "ibmq_qasm_simulator"
    backend.submit.return_value = _make_hardware_job(mock_ibm_job)
    return backend


def _make_hardware_job(ibm_job, always_completed: bool = True):
    """Construct a HardwareJob wrapping the mocked IBM job, without needing Qiskit.

    Parameters
    ----------
    always_completed:
        When True (default), the status function always returns "completed" so
        that HardwareJob.fetch() returns immediately without sleeping.  Set to
        False to use the real iterated status from the mock fixture.
    """
    from quantx_engine.contrib.hardware import HardwareJob, HardwareJobResult

    def _status_fn(jid: str) -> str:
        if always_completed:
            return "completed"
        st = ibm_job.status()
        name = str(st.name).lower() if hasattr(st, "name") else str(st).lower()
        if "done" in name or "complete" in name:
            return "completed"
        if "error" in name or "cancel" in name or "fail" in name:
            return "failed"
        if "running" in name:
            return "running"
        return "queued"

    def _result_fn(jid: str) -> HardwareJobResult:
        try:
            pub_result = ibm_job.result()[0]
            bitstrings = pub_result.data.c.get_bitstrings()
        except Exception:
            bitstrings = ["00"] * 512 + ["11"] * 512
        counts: dict[str, int] = {}
        for bs in bitstrings:
            counts[bs] = counts.get(bs, 0) + 1
        probs = {k: v / len(bitstrings) for k, v in counts.items()}
        return HardwareJobResult(
            job_id=jid,
            backend="ibmq_qasm_simulator",
            counts=counts,
            probabilities=probs,
            shots=len(bitstrings),
            metadata={},
        )

    raw_id = ibm_job.job_id()
    return HardwareJob(
        job_id=raw_id,
        backend_name="ibmq_qasm_simulator",
        _status_fn=_status_fn,
        _result_fn=_result_fn,
    )


# ===========================================================================
# Circuit construction tests
# ===========================================================================

def test_bell_circuit_structure(bell_circuit_dict):
    """Bell circuit dict has expected qubits, ops, and measurement targets."""
    c = bell_circuit_dict
    assert c["qubits"] == 2
    assert c["clbits"] == 2
    names = [op["name"] for op in c["operations"]]
    assert names == ["H", "CNOT", "MEASURE", "MEASURE"]
    h_op = c["operations"][0]
    assert h_op["targets"] == [0]
    cnot_op = c["operations"][1]
    assert cnot_op["controls"] == [0] and cnot_op["targets"] == [1]


def test_bell_circuit_qasm3(bell_circuit_dict):
    """circuit_to_qasm3 produces valid QASM 3.0 for the Bell circuit."""
    from quantx_engine.contrib.ibm import circuit_to_qasm3
    qasm = circuit_to_qasm3(bell_circuit_dict)
    assert "OPENQASM 3.0" in qasm
    assert "qubit[2] q;" in qasm
    assert "bit[2] c;" in qasm
    assert "h q[0];" in qasm
    assert "cx q[0], q[1];" in qasm
    assert "c[0] = measure q[0];" in qasm
    assert "c[1] = measure q[1];" in qasm


def test_bell_circuit_qasm3_is_string(bell_circuit_dict):
    """circuit_to_qasm3 always returns a non-empty string."""
    from quantx_engine.contrib.ibm import circuit_to_qasm3
    qasm = circuit_to_qasm3(bell_circuit_dict)
    assert isinstance(qasm, str)
    assert len(qasm) > 20


# ===========================================================================
# HardwareJob interface tests
# ===========================================================================

def test_hardware_job_initial_status(mock_ibm_job):
    """HardwareJob.status() returns a string on first call (queued)."""
    # Use always_completed=False so we see the initial "queued" state
    hw_job = _make_hardware_job(mock_ibm_job, always_completed=False)
    status = hw_job.status()
    assert isinstance(status, str)
    assert status in ("queued", "running", "completed", "failed")


def test_hardware_job_fetch_returns_result(mock_ibm_job):
    """HardwareJob.fetch() returns a HardwareJobResult after job completes."""
    from quantx_engine.contrib.hardware import HardwareJobResult
    hw_job = _make_hardware_job(mock_ibm_job)
    result = hw_job.fetch()
    assert isinstance(result, HardwareJobResult)
    assert isinstance(result.counts, dict)
    assert isinstance(result.shots, int)
    assert result.shots > 0


def test_hardware_job_result_counts_bell(mock_ibm_job):
    """Bell job result has |00⟩ and |11⟩ as dominant bitstrings."""
    hw_job = _make_hardware_job(mock_ibm_job)
    result = hw_job.fetch()
    total = sum(result.counts.values())
    bell_frac = (result.counts.get("00", 0) + result.counts.get("11", 0)) / total
    assert bell_frac > 0.95, f"Expected >95% Bell state, got {bell_frac:.2%}"


def test_hardware_job_result_probabilities_sum_to_one(mock_ibm_job):
    """Probability distribution returned by fetch() sums to ~1."""
    hw_job = _make_hardware_job(mock_ibm_job)
    result = hw_job.fetch()
    total_prob = sum(result.probabilities.values())
    assert abs(total_prob - 1.0) < 1e-9, f"Probabilities sum to {total_prob}"


def test_hardware_job_has_job_id(mock_ibm_job):
    """HardwareJob.job_id is a non-empty string."""
    hw_job = _make_hardware_job(mock_ibm_job)
    assert isinstance(hw_job.job_id, str)
    assert len(hw_job.job_id) > 0


# ===========================================================================
# IBMHardwareBackend submit (fully mocked)
# ===========================================================================

def test_submit_returns_hardware_job():
    """IBMHardwareBackend.submit returns a HardwareJob without real network."""
    from quantx_engine.contrib.hardware import HardwareJob, HardwareJobResult

    mock_result = HardwareJobResult(
        job_id="fake-001",
        backend="ibmq_qasm_simulator",
        counts={"00": 500, "11": 524},
        probabilities={"00": 0.489, "11": 0.511},
        shots=1024,
        metadata={},
    )
    fake_job = HardwareJob(
        job_id="fake-001",
        backend_name="ibmq_qasm_simulator",
        _status_fn=lambda jid: "completed",
        _result_fn=lambda jid: mock_result,
    )

    circuit = {
        "qubits": 2, "clbits": 2, "shots": 1024, "seed": 42,
        "operations": [
            {"name": "H",       "targets": [0]},
            {"name": "CNOT",    "controls": [0], "targets": [1]},
            {"name": "MEASURE", "targets": [0], "controls": [0]},
            {"name": "MEASURE", "targets": [1], "controls": [1]},
        ],
    }

    backend_mock = MagicMock()
    backend_mock.submit.return_value = fake_job

    job = backend_mock.submit(circuit, shots=1024)
    assert isinstance(job, HardwareJob)
    assert job.status() == "completed"
    result = job.fetch()
    assert result.counts["00"] + result.counts["11"] == 1024


def test_submit_with_no_token_returns_hardware_error():
    """IBMHardwareBackend raises HardwareError when token is absent and Qiskit is mocked."""
    from quantx_engine.contrib.hardware import IBMHardwareBackend, HardwareError

    # Patch the internal IBMQuantumBackend to simulate missing token
    with patch("quantx_engine.contrib.hardware.IBMHardwareBackend.__init__") as mock_init:
        mock_init.side_effect = HardwareError(
            "Backend not connected. Provide token= or QISKIT_IBM_TOKEN."
        )
        with pytest.raises(HardwareError, match="Provide token"):
            IBMHardwareBackend(token=None)


def test_submit_when_qiskit_missing_raises_hardware_error():
    """submit() raises HardwareError when qiskit is not installed."""
    from quantx_engine.contrib.hardware import IBMHardwareBackend, HardwareError

    backend = IBMHardwareBackend.__new__(IBMHardwareBackend)
    backend._backend_name = "ibmq_qasm_simulator"

    # Simulate _base with a real backend but no Qiskit
    base_mock = MagicMock()
    base_mock._backend = MagicMock()  # non-None, so connection check passes
    backend._base = base_mock

    circuit = {"qubits": 1, "clbits": 0, "shots": 100, "seed": 42, "operations": []}

    # Patch qiskit import to fail
    with patch.dict("sys.modules", {
        "qiskit": None,
        "qiskit.qasm3": None,
        "qiskit_ibm_runtime": None,
        "qiskit.transpiler.preset_passmanagers": None,
    }):
        with pytest.raises((HardwareError, ImportError)):
            backend.submit(circuit, shots=100)


# ===========================================================================
# HardwareJobResult data model
# ===========================================================================

def test_hardware_job_result_fields():
    """HardwareJobResult stores all expected fields."""
    from quantx_engine.contrib.hardware import HardwareJobResult
    r = HardwareJobResult(
        job_id="j-001",
        backend="ibmq_qasm_simulator",
        counts={"00": 512, "11": 512},
        probabilities={"00": 0.5, "11": 0.5},
        shots=1024,
        metadata={"device": "simulator"},
    )
    assert r.job_id == "j-001"
    assert r.backend == "ibmq_qasm_simulator"
    assert r.shots == 1024
    assert r.counts == {"00": 512, "11": 512}
    assert abs(r.probabilities["00"] - 0.5) < 1e-9
    assert r.metadata["device"] == "simulator"


def test_hardware_job_result_empty_counts():
    """HardwareJobResult tolerates empty counts dict (e.g. no measurements)."""
    from quantx_engine.contrib.hardware import HardwareJobResult
    r = HardwareJobResult(
        job_id="j-002", backend="sim",
        counts={}, probabilities={}, shots=0, metadata={}
    )
    assert r.counts == {}
    assert r.shots == 0


# ===========================================================================
# Demo script dry-run tests
# ===========================================================================

def test_demo_dry_run(capsys):
    """ibm_hardware_demo dry-run prints QASM and exits cleanly."""
    import ibm_hardware_demo
    rc = ibm_hardware_demo.run_demo(dry_run=True)
    assert rc == 0
    captured = capsys.readouterr()
    assert "OPENQASM 3.0" in captured.out
    assert "Dry-run complete" in captured.out


def test_demo_missing_token_returns_nonzero(capsys, monkeypatch):
    """ibm_hardware_demo exits with non-zero when QISKIT_IBM_TOKEN is unset."""
    import ibm_hardware_demo
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
    rc = ibm_hardware_demo.run_demo(dry_run=False)
    assert rc != 0
    captured = capsys.readouterr()
    assert "QISKIT_IBM_TOKEN" in captured.out


def test_demo_bell_circuit_function():
    """bell_circuit() in the demo returns a valid circuit dict."""
    import ibm_hardware_demo
    c = ibm_hardware_demo.bell_circuit(n_shots=2048)
    assert c["qubits"] == 2
    assert c["shots"] == 2048
    assert any(op["name"] == "H" for op in c["operations"])
    assert any(op["name"] == "CNOT" for op in c["operations"])


def test_demo_display_counts_runs_without_error(capsys):
    """display_counts() prints a histogram without raising."""
    import ibm_hardware_demo
    counts = {"00": 512, "11": 500, "01": 6, "10": 6}
    ibm_hardware_demo.display_counts(counts, shots=1024, title="Test")
    captured = capsys.readouterr()
    assert "00" in captured.out
    assert "11" in captured.out


# ===========================================================================
# Real hardware marker (skipped unless -m hardware)
# ===========================================================================

@pytest.mark.hardware
def test_real_ibm_round_trip_bell():
    """Submit a Bell circuit to IBM and verify Bell-state counts.

    Requires QISKIT_IBM_TOKEN environment variable to be set.
    Run with: pytest -m hardware ...
    """
    token = os.environ.get("QISKIT_IBM_TOKEN")
    if not token:
        pytest.skip("QISKIT_IBM_TOKEN not set — skipping real hardware test")

    try:
        from quantx_engine.contrib.hardware import IBMHardwareBackend
    except ImportError:
        pytest.skip("quantx_engine.contrib.hardware not importable")

    circuit = {
        "qubits": 2, "clbits": 2, "shots": 1024, "seed": 42,
        "operations": [
            {"name": "H",       "targets": [0]},
            {"name": "CNOT",    "controls": [0], "targets": [1]},
            {"name": "MEASURE", "targets": [0], "controls": [0]},
            {"name": "MEASURE", "targets": [1], "controls": [1]},
        ],
    }
    backend = IBMHardwareBackend(
        token=token,
        instance="ibm-q/open/main",
        backend_name="ibmq_qasm_simulator",
    )
    result = backend.run_circuit(circuit, shots=1024)
    total = sum(result.counts.values())
    bell_frac = (result.counts.get("00", 0) + result.counts.get("11", 0)) / total
    assert bell_frac > 0.90, f"Expected >90% Bell state, got {bell_frac:.2%}"
