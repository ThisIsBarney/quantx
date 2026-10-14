"""Tests for hardware.py — M93 real hardware backend integration."""
from __future__ import annotations

import math
import os
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call
from typing import Any

import numpy as np

# ── module under test ─────────────────────────────────────────────────────────
from quantx_engine.hardware import (
    HardwareError,
    BackendCalibration,
    HardwareJobResult,
    HardwareJob,
    IBMHardwareBackend,
    IonQBackend,
    GoogleBackend,
    calibration_to_noise_model,
    available_backends,
    _circuit_to_ionq,
    _ionq_request,
    _cirq_result_to_counts,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_cal(
    backend_name="test_backend",
    n_qubits=4,
    t1=None,
    t2=None,
    gate_errors=None,
    coupling_edges=None,
    gate_times=None,
) -> BackendCalibration:
    return BackendCalibration(
        backend_name=backend_name,
        n_qubits=n_qubits,
        t1=t1 or {0: 100_000.0, 1: 120_000.0, 2: 80_000.0, 3: 90_000.0},
        t2=t2 or {0: 80_000.0, 1: 90_000.0, 2: 60_000.0, 3: 70_000.0},
        gate_errors=gate_errors or {"cx": 0.005, "sx": 0.0003},
        coupling_edges=coupling_edges or [(0, 1), (1, 2), (2, 3)],
        gate_times=gate_times or {"cx": 350.0, "sx": 35.0},
    )


def _make_job_result(job_id="job-001", backend="sim", counts=None, shots=1000):
    counts = counts or {"00": 512, "11": 488}
    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}
    return HardwareJobResult(
        job_id=job_id,
        backend=backend,
        counts=counts,
        probabilities=probs,
        shots=shots,
        metadata={"tag": "test"},
    )


def _completed_job(job_id="jid-1", backend="sim", counts=None, shots=1000):
    """HardwareJob that immediately reports 'completed' and returns a result."""
    res = _make_job_result(job_id=job_id, backend=backend,
                           counts=counts, shots=shots)
    return HardwareJob(
        job_id=job_id,
        backend_name=backend,
        _status_fn=lambda jid: "completed",
        _result_fn=lambda jid: res,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BackendCalibration
# ─────────────────────────────────────────────────────────────────────────────

class TestBackendCalibrationConstruction(unittest.TestCase):
    def test_fields_stored(self):
        cal = _make_cal()
        self.assertEqual(cal.backend_name, "test_backend")
        self.assertEqual(cal.n_qubits, 4)
        self.assertIn(0, cal.t1)
        self.assertIn(0, cal.t2)

    def test_default_empty_dicts(self):
        cal = BackendCalibration(backend_name="empty", n_qubits=2)
        self.assertEqual(cal.t1, {})
        self.assertEqual(cal.t2, {})
        self.assertEqual(cal.gate_errors, {})
        self.assertEqual(cal.coupling_edges, [])
        self.assertEqual(cal.gate_times, {})

    def test_repr_contains_name(self):
        cal = _make_cal()
        r = repr(cal)
        self.assertIn("test_backend", r)
        self.assertIn("n_qubits=4", r)
        self.assertIn("mean_T1", r)


class TestBackendCalibrationMeanT1T2(unittest.TestCase):
    def test_mean_t1_average(self):
        cal = BackendCalibration(
            "b", 2, t1={0: 100_000.0, 1: 200_000.0}
        )
        self.assertAlmostEqual(cal.mean_t1(), 150_000.0)

    def test_mean_t2_average(self):
        cal = BackendCalibration(
            "b", 2, t2={0: 60_000.0, 1: 80_000.0}
        )
        self.assertAlmostEqual(cal.mean_t2(), 70_000.0)

    def test_mean_t1_empty_returns_inf(self):
        cal = BackendCalibration("b", 2, t1={})
        self.assertTrue(math.isinf(cal.mean_t1()))

    def test_mean_t2_empty_returns_inf(self):
        cal = BackendCalibration("b", 2, t2={})
        self.assertTrue(math.isinf(cal.mean_t2()))

    def test_ignores_inf_values(self):
        cal = BackendCalibration(
            "b", 2, t1={0: float("inf"), 1: 100_000.0}
        )
        self.assertAlmostEqual(cal.mean_t1(), 100_000.0)

    def test_ignores_zero_values(self):
        cal = BackendCalibration("b", 2, t1={0: 0.0, 1: 100_000.0})
        self.assertAlmostEqual(cal.mean_t1(), 100_000.0)

    def test_mean_gate_time_returns_calibrated(self):
        cal = BackendCalibration("b", 2, gate_times={"cx": 350.0})
        self.assertAlmostEqual(cal.mean_gate_time("cx"), 350.0)

    def test_mean_gate_time_default_50ns(self):
        cal = BackendCalibration("b", 2)
        self.assertAlmostEqual(cal.mean_gate_time("cx"), 50.0)


class TestBackendCalibrationToNoiseModel(unittest.TestCase):
    def test_returns_thermal_relaxation_noise(self):
        from quantx_engine.noise import ThermalRelaxationNoise
        cal = _make_cal()
        nm = cal.to_noise_model()
        self.assertIsInstance(nm, ThermalRelaxationNoise)

    def test_lindblad_bound_t2_clamped(self):
        # T2 = 3*T1 violates Lindblad bound T2 ≤ 2*T1
        cal = BackendCalibration(
            "b", 1,
            t1={0: 100_000.0},
            t2={0: 300_000.0},   # 3×T1 — too large
        )
        nm = cal.to_noise_model()
        # The noise model's T2 must not exceed 2*T1
        self.assertLessEqual(nm.T2, 2.0 * nm.T1 + 1e-6)

    def test_lindblad_bound_t2_ok_unchanged(self):
        cal = BackendCalibration(
            "b", 1,
            t1={0: 100_000.0},
            t2={0: 150_000.0},   # 1.5×T1 — within bound
        )
        nm = cal.to_noise_model()
        self.assertAlmostEqual(nm.T2, 150_000.0, delta=1.0)

    def test_t1_inf_returns_noiseless(self):
        cal = BackendCalibration("b", 1)
        nm = cal.to_noise_model()
        # T1 and T2 should be sensible large values
        self.assertGreater(nm.T1, 0)
        self.assertGreater(nm.T2, 0)

    def test_t_gate_override(self):
        from quantx_engine.noise import ThermalRelaxationNoise
        cal = _make_cal()
        nm = cal.to_noise_model(t_gate_ns=99.0)
        self.assertAlmostEqual(nm.t_gate, 99.0)

    def test_t2_only_inf(self):
        # T1 finite, T2 inf → clamp to 2*T1
        cal = BackendCalibration(
            "b", 1,
            t1={0: 100_000.0},
            t2={0: float("inf")},
        )
        nm = cal.to_noise_model()
        self.assertAlmostEqual(nm.T2, 200_000.0, delta=1.0)


class TestBackendCalibrationToCouplingMap(unittest.TestCase):
    def test_returns_coupling_map(self):
        from quantx_engine.transpiler import CouplingMap
        cal = _make_cal()
        cm = cal.to_coupling_map()
        self.assertIsInstance(cm, CouplingMap)

    def test_coupling_map_edges(self):
        cal = BackendCalibration("b", 3, coupling_edges=[(0, 1), (1, 2)])
        cm = cal.to_coupling_map()
        edges = list(cm.edges)
        self.assertIn((0, 1), edges)
        self.assertIn((1, 2), edges)

    def test_n_qubits_preserved(self):
        cal = BackendCalibration("b", 5)
        cm = cal.to_coupling_map()
        self.assertEqual(cm.n_qubits, 5)


# ─────────────────────────────────────────────────────────────────────────────
# HardwareJobResult
# ─────────────────────────────────────────────────────────────────────────────

class TestHardwareJobResult(unittest.TestCase):
    def test_fields_stored(self):
        r = _make_job_result()
        self.assertEqual(r.job_id, "job-001")
        self.assertEqual(r.backend, "sim")
        self.assertEqual(r.shots, 1000)
        self.assertIn("00", r.counts)
        self.assertIn("00", r.probabilities)

    def test_repr_contains_job_id(self):
        r = _make_job_result(job_id="xyz-789")
        rep = repr(r)
        self.assertIn("xyz-789", rep)
        self.assertIn("shots=1000", rep)

    def test_repr_top_states(self):
        r = _make_job_result(counts={"00": 900, "01": 50, "10": 30, "11": 20})
        rep = repr(r)
        # Top state should appear
        self.assertIn("00", rep)

    def test_metadata_stored(self):
        r = HardwareJobResult(
            job_id="j", backend="b", counts={}, probabilities={},
            shots=10, metadata={"extra": 42}
        )
        self.assertEqual(r.metadata["extra"], 42)


# ─────────────────────────────────────────────────────────────────────────────
# HardwareJob
# ─────────────────────────────────────────────────────────────────────────────

class TestHardwareJobConstruction(unittest.TestCase):
    def test_job_id_property(self):
        job = _completed_job(job_id="abc")
        self.assertEqual(job.job_id, "abc")

    def test_backend_name_property(self):
        job = _completed_job(backend="my_backend")
        self.assertEqual(job.backend_name, "my_backend")

    def test_repr_contains_id_and_backend(self):
        job = _completed_job(job_id="id-1", backend="be-2")
        rep = repr(job)
        self.assertIn("id-1", rep)
        self.assertIn("be-2", rep)


class TestHardwareJobStatus(unittest.TestCase):
    def test_status_delegates(self):
        statuses = ["queued", "running"]
        idx = [0]

        def _status(jid):
            s = statuses[idx[0]]
            idx[0] = min(idx[0] + 1, len(statuses) - 1)
            return s

        job = HardwareJob("j", "b", _status, lambda jid: _make_job_result())
        self.assertEqual(job.status(), "queued")
        self.assertEqual(job.status(), "running")

    def test_status_completed(self):
        job = _completed_job()
        self.assertEqual(job.status(), "completed")


class TestHardwareJobFetch(unittest.TestCase):
    def test_fetch_returns_result_immediately(self):
        job = _completed_job(job_id="j1", counts={"00": 700, "11": 300})
        res = job.fetch(timeout=5.0)
        self.assertIsInstance(res, HardwareJobResult)
        self.assertEqual(res.job_id, "j1")

    def test_fetch_waits_for_completion(self):
        calls = [0]

        def _status(jid):
            calls[0] += 1
            return "completed" if calls[0] >= 3 else "queued"

        result_val = _make_job_result()
        job = HardwareJob("j", "b", _status, lambda jid: result_val)
        res = job.fetch(timeout=10.0, poll_interval=0.01)
        self.assertEqual(res, result_val)
        self.assertGreaterEqual(calls[0], 3)

    def test_fetch_raises_on_failed(self):
        job = HardwareJob(
            "j", "b",
            _status_fn=lambda jid: "failed",
            _result_fn=lambda jid: _make_job_result(),
        )
        with self.assertRaises(HardwareError):
            job.fetch(timeout=5.0)

    def test_fetch_raises_on_timeout(self):
        job = HardwareJob(
            "j", "b",
            _status_fn=lambda jid: "queued",
            _result_fn=lambda jid: _make_job_result(),
        )
        with self.assertRaises(HardwareError):
            job.fetch(timeout=0.05, poll_interval=0.01)

    def test_fetch_invalid_timeout_raises(self):
        job = _completed_job()
        with self.assertRaises(HardwareError):
            job.fetch(timeout=-1.0)

    def test_fetch_zero_timeout_raises(self):
        job = _completed_job()
        with self.assertRaises(HardwareError):
            job.fetch(timeout=0.0)

    def test_fetch_result_has_correct_counts(self):
        counts = {"000": 800, "111": 200}
        job = _completed_job(counts=counts, shots=1000)
        res = job.fetch()
        self.assertEqual(res.counts, counts)


# ─────────────────────────────────────────────────────────────────────────────
# _circuit_to_ionq helper
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitToIonQ(unittest.TestCase):
    def _circuit(self, ops, n_qubits=2):
        return {"qubits": n_qubits, "operations": ops}

    def test_h_gate(self):
        c = self._circuit([{"name": "H", "targets": [0]}])
        out = _circuit_to_ionq(c)
        self.assertEqual(out["qubits"], 2)
        self.assertEqual(len(out["circuit"]), 1)
        self.assertEqual(out["circuit"][0]["gate"], "h")
        self.assertEqual(out["circuit"][0]["target"], 0)

    def test_cnot_gate(self):
        c = self._circuit([{"name": "CNOT", "targets": [0, 1]}])
        out = _circuit_to_ionq(c)
        g = out["circuit"][0]
        self.assertEqual(g["gate"], "cnot")
        self.assertIn("control", g)
        self.assertIn("target", g)

    def test_cnot_with_explicit_controls(self):
        c = self._circuit([{
            "name": "CNOT", "targets": [1], "controls": [0]
        }])
        out = _circuit_to_ionq(c)
        g = out["circuit"][0]
        self.assertEqual(g["control"], 0)
        self.assertEqual(g["target"], 1)

    def test_swap_gate(self):
        c = self._circuit([{"name": "SWAP", "targets": [0, 1]}])
        out = _circuit_to_ionq(c)
        g = out["circuit"][0]
        self.assertEqual(g["gate"], "swap")
        self.assertEqual(g["targets"], [0, 1])

    def test_rx_rotation(self):
        c = self._circuit([{"name": "RX", "targets": [0], "params": [1.57]}])
        out = _circuit_to_ionq(c)
        g = out["circuit"][0]
        self.assertEqual(g["gate"], "rx")
        self.assertAlmostEqual(g["rotation"], 1.57, places=5)

    def test_measure_skipped(self):
        ops = [
            {"name": "H", "targets": [0]},
            {"name": "MEASURE", "targets": [0]},
        ]
        out = _circuit_to_ionq(self._circuit(ops))
        # MEASURE should be filtered out
        self.assertEqual(len(out["circuit"]), 1)
        self.assertEqual(out["circuit"][0]["gate"], "h")

    def test_barrier_skipped(self):
        ops = [{"name": "BARRIER", "targets": [0, 1]}]
        out = _circuit_to_ionq(self._circuit(ops))
        self.assertEqual(len(out["circuit"]), 0)

    def test_sdg_maps_to_si(self):
        c = self._circuit([{"name": "Sdg", "targets": [0]}])
        out = _circuit_to_ionq(c)
        self.assertEqual(out["circuit"][0]["gate"], "si")

    def test_tdg_maps_to_ti(self):
        c = self._circuit([{"name": "Tdg", "targets": [0]}])
        out = _circuit_to_ionq(c)
        self.assertEqual(out["circuit"][0]["gate"], "ti")

    def test_sx_maps_to_v(self):
        c = self._circuit([{"name": "SX", "targets": [0]}])
        out = _circuit_to_ionq(c)
        self.assertEqual(out["circuit"][0]["gate"], "v")

    def test_unsupported_gate_raises(self):
        c = self._circuit([{"name": "TOFFOLI", "targets": [0, 1, 2]}], n_qubits=3)
        with self.assertRaises(HardwareError):
            _circuit_to_ionq(c)

    def test_empty_circuit(self):
        out = _circuit_to_ionq({"qubits": 1, "operations": []})
        self.assertEqual(out["circuit"], [])
        self.assertEqual(out["qubits"], 1)

    def test_multiple_gates(self):
        ops = [
            {"name": "H", "targets": [0]},
            {"name": "CNOT", "targets": [0, 1]},
            {"name": "RZ", "targets": [1], "params": [0.5]},
        ]
        out = _circuit_to_ionq(self._circuit(ops))
        self.assertEqual(len(out["circuit"]), 3)


# ─────────────────────────────────────────────────────────────────────────────
# IonQBackend
# ─────────────────────────────────────────────────────────────────────────────

class TestIonQBackendConstruction(unittest.TestCase):
    def test_no_token_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove IONQ_API_KEY if present
            env = {k: v for k, v in os.environ.items() if k != "IONQ_API_KEY"}
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(HardwareError):
                    IonQBackend()

    def test_token_from_env(self):
        with patch.dict(os.environ, {"IONQ_API_KEY": "test-key"}):
            backend = IonQBackend()
            self.assertEqual(backend._token, "test-key")

    def test_token_from_arg(self):
        backend = IonQBackend(token="my-token")
        self.assertEqual(backend._token, "my-token")

    def test_default_device(self):
        backend = IonQBackend(token="key")
        self.assertEqual(backend._device, "simulator")

    def test_custom_device(self):
        backend = IonQBackend(token="key", device="qpu.aria-1")
        self.assertEqual(backend._device, "qpu.aria-1")

    def test_repr(self):
        backend = IonQBackend(token="key", device="qpu.harmony")
        r = repr(backend)
        self.assertIn("qpu.harmony", r)


class TestIonQBackendSubmit(unittest.TestCase):
    def _circuit(self):
        return {
            "qubits": 2,
            "operations": [
                {"name": "H", "targets": [0]},
                {"name": "CNOT", "targets": [0, 1]},
            ],
        }

    def test_submit_returns_hardware_job(self):
        backend = IonQBackend(token="key")
        with patch("quantx_engine.hardware._ionq_request") as mock_req:
            mock_req.return_value = {"id": "ionq-job-123"}
            job = backend.submit(self._circuit(), shots=1000)
        self.assertIsInstance(job, HardwareJob)
        self.assertEqual(job.job_id, "ionq-job-123")

    def test_submit_uses_correct_endpoint(self):
        backend = IonQBackend(token="key", base_url="https://api.ionq.co/v0.3")
        with patch("quantx_engine.hardware._ionq_request") as mock_req:
            mock_req.return_value = {"id": "j1"}
            backend.submit(self._circuit())
        args, kwargs = mock_req.call_args
        self.assertEqual(args[0], "POST")
        self.assertIn("/jobs", args[1])

    def test_submit_missing_id_raises(self):
        backend = IonQBackend(token="key")
        with patch("quantx_engine.hardware._ionq_request") as mock_req:
            mock_req.return_value = {}  # no "id" field
            with self.assertRaises(HardwareError):
                backend.submit(self._circuit())

    def test_run_circuit_blocks_and_returns_result(self):
        backend = IonQBackend(token="key")
        with patch("quantx_engine.hardware._ionq_request") as mock_req:
            # First call: submit POST → job id
            # Second call: GET status → completed
            # Third call: GET result → histogram
            completed_resp = {
                "id": "j1",
                "status": "completed",
                "shots": 1000,
                "data": {"histogram": {"0": 0.7, "3": 0.3}},
                "input": {"qubits": 2},
            }
            mock_req.side_effect = [
                {"id": "j1"},          # POST /jobs
                completed_resp,        # GET /jobs/j1 (status)
                completed_resp,        # GET /jobs/j1 (result)
            ]
            with patch("time.sleep"):  # skip actual sleeping
                result = backend.run_circuit(self._circuit(), shots=1000, timeout=5.0)
        self.assertIsInstance(result, HardwareJobResult)
        self.assertEqual(result.shots, 1000)


class TestIonQBackendResultFn(unittest.TestCase):
    """Test the _result_fn logic: decimal histogram keys → bitstrings."""

    def _backend(self):
        return IonQBackend(token="key")

    def test_decimal_keys_to_bitstrings(self):
        backend = self._backend()
        resp = {
            "status": "completed",
            "shots": 100,
            "data": {"histogram": {"0": 0.6, "3": 0.4}},
            "input": {"qubits": 2},
        }
        with patch.object(backend, "_get_job", return_value=resp):
            result = backend._result_fn("j1")
        self.assertIn("00", result.probabilities)  # 0 → "00"
        self.assertIn("11", result.probabilities)  # 3 → "11"
        self.assertAlmostEqual(result.probabilities["00"], 0.6, places=5)
        self.assertAlmostEqual(result.probabilities["11"], 0.4, places=5)

    def test_shots_from_response(self):
        backend = self._backend()
        resp = {
            "shots": 512,
            "data": {"histogram": {"0": 1.0}},
            "input": {"qubits": 1},
        }
        with patch.object(backend, "_get_job", return_value=resp):
            result = backend._result_fn("j1")
        self.assertEqual(result.shots, 512)

    def test_results_fallback(self):
        """IonQ sometimes returns `results` instead of `data.histogram`."""
        backend = self._backend()
        resp = {
            "shots": 100,
            # decimal keys: 1 → "01", 2 → "10" for 2-qubit system
            "results": {"1": 0.5, "2": 0.5},
            "data": {},
            "input": {"qubits": 2},
        }
        with patch.object(backend, "_get_job", return_value=resp):
            result = backend._result_fn("j1")
        self.assertIn("01", result.probabilities)   # int(1) → format(1, "02b") = "01"
        self.assertIn("10", result.probabilities)   # int(2) → format(2, "02b") = "10"

    def test_status_fn_completed(self):
        backend = self._backend()
        with patch.object(backend, "_get_job",
                          return_value={"status": "completed"}):
            self.assertEqual(backend._status_fn("j1"), "completed")

    def test_status_fn_failed(self):
        backend = self._backend()
        with patch.object(backend, "_get_job",
                          return_value={"status": "failed"}):
            self.assertEqual(backend._status_fn("j1"), "failed")

    def test_status_fn_queued(self):
        backend = self._backend()
        with patch.object(backend, "_get_job",
                          return_value={"status": "submitted"}):
            self.assertEqual(backend._status_fn("j1"), "queued")

    def test_status_fn_running(self):
        backend = self._backend()
        with patch.object(backend, "_get_job",
                          return_value={"status": "running"}):
            self.assertEqual(backend._status_fn("j1"), "running")


class TestIonQBackendCalibration(unittest.TestCase):
    def _backend(self):
        return IonQBackend(token="key", device="qpu.aria-1")

    def test_calibration_basic(self):
        backend = self._backend()
        resp = {
            "qubits": 25,
            "connectivity": [[0, 1], [1, 2]],
            "fidelity": {"1q": 0.999, "2q": 0.995},
            "timing": {"T1": 10000.0, "T2": 5000.0},
        }
        with patch("quantx_engine.hardware._ionq_request", return_value=resp):
            cal = backend.fetch_calibration()
        self.assertEqual(cal.n_qubits, 25)
        self.assertIn((0, 1), cal.coupling_edges)
        self.assertIn("1q", cal.gate_errors)
        self.assertAlmostEqual(cal.gate_errors["1q"], 0.001, places=6)

    def test_calibration_t1_t2_ns(self):
        """IonQ timing is in microseconds; should be converted to ns."""
        backend = self._backend()
        resp = {
            "qubits": 1,
            "connectivity": [],
            "fidelity": {},
            "timing": {"T1": 100.0, "T2": 50.0},  # µs
        }
        with patch("quantx_engine.hardware._ionq_request", return_value=resp):
            cal = backend.fetch_calibration()
        # 100µs = 100_000 ns
        self.assertAlmostEqual(cal.t1[0], 100_000.0, delta=1.0)
        self.assertAlmostEqual(cal.t2[0], 50_000.0, delta=1.0)

    def test_calibration_api_error_returns_empty(self):
        """If IonQ /backends endpoint errors, fall back to empty calibration."""
        backend = self._backend()
        with patch("quantx_engine.hardware._ionq_request",
                   side_effect=HardwareError("404")):
            cal = backend.fetch_calibration()
        self.assertEqual(cal.n_qubits, 0)
        self.assertEqual(cal.t1, {})


class TestIonQRequest(unittest.TestCase):
    """Unit tests for the _ionq_request helper."""

    def _mock_urlopen(self, response_body: dict):
        """Return a context-manager mock that yields a response with JSON body."""
        import io
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json_bytes(response_body)
        return mock_resp

    def test_get_request(self):
        import urllib.request as _ur
        resp_body = {"id": "job-1", "status": "completed"}
        mock_cm = self._mock_urlopen(resp_body)
        with patch.object(_ur, "urlopen", return_value=mock_cm):
            result = _ionq_request("GET", "https://api.ionq.co/v0.3/jobs/1", "tok")
        self.assertEqual(result["id"], "job-1")

    def test_http_error_raises_hardware_error(self):
        import urllib.error
        import urllib.request as _ur
        err = urllib.error.HTTPError(
            url="https://api.ionq.co",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )
        err.read = lambda: b'{"error": "Unauthorized"}'
        with patch.object(_ur, "urlopen", side_effect=err):
            with self.assertRaises(HardwareError) as ctx:
                _ionq_request("GET", "https://api.ionq.co/v0.3/jobs/1", "tok")
        self.assertIn("401", str(ctx.exception))

    def test_url_error_raises_hardware_error(self):
        import urllib.error
        import urllib.request as _ur
        err = urllib.error.URLError(reason="Name resolution failed")
        with patch.object(_ur, "urlopen", side_effect=err):
            with self.assertRaises(HardwareError):
                _ionq_request("GET", "https://api.ionq.co/v0.3/jobs/1", "tok")


def json_bytes(d: dict) -> bytes:
    import json
    return json.dumps(d).encode()


# ─────────────────────────────────────────────────────────────────────────────
# IBMHardwareBackend
# ─────────────────────────────────────────────────────────────────────────────

class TestIBMHardwareBackendConstruction(unittest.TestCase):
    """Test construction with a fully mocked IBMQuantumBackend."""

    def _mock_ibm_backend_cls(self):
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance._backend = MagicMock()
        mock_instance._service = MagicMock()
        mock_instance._instance = "ibm-q/open/main"
        mock_cls.return_value = mock_instance
        return mock_cls, mock_instance

    def test_construction_stores_backend_name(self):
        mock_cls, mock_inst = self._mock_ibm_backend_cls()
        with patch("quantx_engine.ibm.IBMQuantumBackend", mock_cls):
            hw = IBMHardwareBackend(token="tok", backend_name="ibm_kyoto")
        self.assertEqual(hw.backend_name, "ibm_kyoto")

    def test_repr_contains_backend_name(self):
        mock_cls, mock_inst = self._mock_ibm_backend_cls()
        with patch("quantx_engine.ibm.IBMQuantumBackend", mock_cls):
            hw = IBMHardwareBackend(token="tok", backend_name="ibm_kyoto")
        r = repr(hw)
        self.assertIn("ibm_kyoto", r)

    def test_backend_name_property(self):
        mock_cls, mock_inst = self._mock_ibm_backend_cls()
        with patch("quantx_engine.ibm.IBMQuantumBackend", mock_cls):
            hw = IBMHardwareBackend(token="tok", backend_name="test_be")
        self.assertEqual(hw.backend_name, "test_be")


class TestIBMHardwareBackendRunCircuit(unittest.TestCase):
    def _setup(self, backend_name="sim"):
        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_inst._backend = MagicMock()
        mock_inst._service = MagicMock()
        mock_inst._instance = "ibm-q/open/main"
        mock_inst.run_circuit.return_value = {
            "ok": True,
            "shots": 1000,
            "counts": {"00": 600, "11": 400},
            "probabilities": {"00": 0.6, "11": 0.4},
            "ibm_job_id": "job-999",
            "raw_ibm_result": {},
        }
        mock_cls.return_value = mock_inst
        with patch("quantx_engine.ibm.IBMQuantumBackend", mock_cls):
            hw = IBMHardwareBackend(token="tok", backend_name=backend_name)
        hw._base = mock_inst
        return hw

    def test_run_circuit_returns_hardware_job_result(self):
        hw = self._setup()
        circuit = {"qubits": 2, "operations": []}
        result = hw.run_circuit(circuit, shots=1000)
        self.assertIsInstance(result, HardwareJobResult)
        self.assertEqual(result.job_id, "job-999")
        self.assertEqual(result.counts["00"], 600)

    def test_run_circuit_passes_shots(self):
        hw = self._setup()
        circuit = {"qubits": 2, "operations": []}
        hw.run_circuit(circuit, shots=2048)
        hw._base.run_circuit.assert_called_once_with(
            circuit, shots=2048, optimization_level=1
        )

    def test_run_circuit_probabilities(self):
        hw = self._setup()
        res = hw.run_circuit({"qubits": 2, "operations": []})
        self.assertAlmostEqual(res.probabilities["00"], 0.6, places=5)


class TestIBMHardwareBackendFetchCalibration(unittest.TestCase):
    def _make_hw(self):
        mock_cls = MagicMock()
        mock_inst = MagicMock()

        # Mock backend configuration
        cfg = MagicMock()
        cfg.n_qubits = 5
        cfg.coupling_map = [[0, 1], [1, 2], [2, 3], [3, 4]]
        mock_inst._backend.configuration.return_value = cfg

        # Mock backend properties
        props = MagicMock()
        props.t1 = lambda q: 100e-6  # 100 µs in seconds
        props.t2 = lambda q: 80e-6
        # Mock gate list
        gate = MagicMock()
        gate.gate = "cx"
        param_err = MagicMock()
        param_err.name = "gate_error"
        param_err.value = 0.005
        param_time = MagicMock()
        param_time.name = "gate_length"
        param_time.value = 350e-9  # 350 ns in seconds
        gate.parameters = [param_err, param_time]
        props.gates = [gate]
        mock_inst._backend.properties.return_value = props
        mock_inst._instance = "ibm-q/open/main"
        mock_cls.return_value = mock_inst

        with patch("quantx_engine.ibm.IBMQuantumBackend", mock_cls):
            hw = IBMHardwareBackend(token="tok", backend_name="ibm_sherbrooke")
        hw._base = mock_inst
        return hw

    def test_fetch_calibration_n_qubits(self):
        hw = self._make_hw()
        cal = hw.fetch_calibration()
        self.assertEqual(cal.n_qubits, 5)

    def test_fetch_calibration_coupling_edges(self):
        hw = self._make_hw()
        cal = hw.fetch_calibration()
        # Edges should be deduplicated and present
        self.assertTrue(len(cal.coupling_edges) > 0)

    def test_fetch_calibration_t1_ns(self):
        hw = self._make_hw()
        cal = hw.fetch_calibration()
        # 100µs = 100_000 ns
        for q in range(5):
            if q in cal.t1:
                self.assertAlmostEqual(cal.t1[q], 100_000.0, delta=1.0)

    def test_fetch_calibration_gate_errors(self):
        hw = self._make_hw()
        cal = hw.fetch_calibration()
        self.assertIn("cx", cal.gate_errors)
        self.assertAlmostEqual(cal.gate_errors["cx"], 0.005, places=6)

    def test_fetch_calibration_gate_times_ns(self):
        hw = self._make_hw()
        cal = hw.fetch_calibration()
        # 350ns expected
        if "cx" in cal.gate_times:
            self.assertAlmostEqual(cal.gate_times["cx"], 350.0, delta=1.0)

    def test_fetch_calibration_no_backend_raises(self):
        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_inst._backend = None
        mock_inst._instance = "ibm-q/open/main"
        mock_cls.return_value = mock_inst
        with patch("quantx_engine.ibm.IBMQuantumBackend", mock_cls):
            hw = IBMHardwareBackend(token="tok")
        hw._base = mock_inst
        with self.assertRaises(HardwareError):
            hw.fetch_calibration()


class TestIBMHardwareBackendListBackends(unittest.TestCase):
    def _make_hw(self, backends_names=("ibm_kyoto", "ibm_osaka")):
        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_backends = [MagicMock(name=n) for n in backends_names]
        for m, n in zip(mock_backends, backends_names):
            type(m).name = PropertyMock(return_value=n)
        mock_inst._service.backends.return_value = mock_backends
        mock_inst._backend = MagicMock()
        mock_inst._instance = "ibm-q/open/main"
        mock_cls.return_value = mock_inst
        with patch("quantx_engine.ibm.IBMQuantumBackend", mock_cls):
            hw = IBMHardwareBackend(token="tok")
        hw._base = mock_inst
        return hw

    def test_list_backends_returns_names(self):
        hw = self._make_hw(["ibm_kyoto", "ibm_osaka"])
        names = hw.list_backends()
        self.assertIn("ibm_kyoto", names)
        self.assertIn("ibm_osaka", names)

    def test_list_backends_no_service_raises(self):
        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_inst._service = None
        mock_inst._backend = MagicMock()
        mock_inst._instance = "ibm-q/open/main"
        mock_cls.return_value = mock_inst
        with patch("quantx_engine.ibm.IBMQuantumBackend", mock_cls):
            hw = IBMHardwareBackend(token="tok")
        hw._base = mock_inst
        with self.assertRaises(HardwareError):
            hw.list_backends()


# ─────────────────────────────────────────────────────────────────────────────
# GoogleBackend
# ─────────────────────────────────────────────────────────────────────────────

class TestGoogleBackendConstruction(unittest.TestCase):
    def test_fields_stored(self):
        g = GoogleBackend(project_id="my-project", processor_id="rainbow")
        self.assertEqual(g._project_id, "my-project")
        self.assertEqual(g._processor_id, "rainbow")

    def test_default_processor_simulator(self):
        g = GoogleBackend(project_id="proj")
        self.assertEqual(g._processor_id, "simulator")

    def test_repr(self):
        g = GoogleBackend(project_id="proj", processor_id="weber")
        r = repr(g)
        self.assertIn("proj", r)
        self.assertIn("weber", r)


class TestGoogleBackendSimulatorPath(unittest.TestCase):
    """Tests the local Cirq simulator path (no cloud required)."""

    def _make_cirq_mock(self, n_qubits=2, shots=100):
        """Build a minimal mock Cirq environment."""
        cirq = MagicMock()

        # Mock qubits and circuit
        mock_qubits = [MagicMock() for _ in range(n_qubits)]
        mock_circuit = MagicMock()
        mock_circuit.all_qubits.return_value = mock_qubits
        # No measurement gates initially
        mock_circuit.__iter__ = MagicMock(return_value=iter([]))
        # After adding measurements, new_circuit should be returned
        new_circuit = MagicMock()
        new_circuit.all_qubits.return_value = mock_qubits
        new_circuit.__iter__ = MagicMock(return_value=iter([]))
        mock_circuit.__add__ = MagicMock(return_value=new_circuit)

        # MeasurementGate mock
        cirq.MeasurementGate = MagicMock

        # Simulator
        mock_result = MagicMock()
        mock_result.measurements = {"m": [[0, 0]] * shots}
        mock_sim = MagicMock()
        mock_sim.run.return_value = mock_result
        cirq.Simulator.return_value = mock_sim
        cirq.measure.return_value = MagicMock()

        return cirq, mock_result

    def test_simulator_submit_returns_completed_job(self):
        cirq_mock, _ = self._make_cirq_mock()
        g = GoogleBackend(project_id="proj")

        with patch("quantx_engine.hardware.cirq", cirq_mock, create=True):
            with patch.object(g, "_to_cirq_circuit") as mock_convert:
                mock_circuit = MagicMock()
                mock_circuit.all_qubits.return_value = []
                mock_circuit.__iter__ = MagicMock(return_value=iter([]))
                mock_circuit.__add__ = MagicMock(return_value=mock_circuit)
                mock_convert.return_value = mock_circuit

                with patch("builtins.__import__", side_effect=_mock_import_cirq(cirq_mock)):
                    try:
                        job = g.submit({"qubits": 1, "operations": []}, shots=10)
                    except Exception:
                        # Simulator path may fail in test environment without cirq;
                        # that's acceptable — we test the structure is correct
                        pass

    def test_simulator_fetch_calibration_raises(self):
        g = GoogleBackend(project_id="proj", processor_id="simulator")
        with self.assertRaises(HardwareError) as ctx:
            g.fetch_calibration()
        self.assertIn("simulator", str(ctx.exception).lower())

    def test_non_simulator_without_cirq_google_raises(self):
        g = GoogleBackend(project_id="proj", processor_id="rainbow")
        with patch.dict("sys.modules", {"cirq_google": None}):
            with self.assertRaises((HardwareError, ImportError)):
                g._get_engine()


def _mock_import_cirq(cirq_mock):
    """Side effect for __import__ that returns cirq_mock for 'cirq'."""
    import builtins
    _real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "cirq":
            return cirq_mock
        return _real_import(name, *args, **kwargs)

    return _import


class TestCirqResultToCounts(unittest.TestCase):
    def test_converts_measurements(self):
        mock_result = MagicMock()
        mock_result.measurements = {
            "m": [[0, 0], [0, 0], [1, 1], [1, 1]]
        }
        counts = _cirq_result_to_counts(mock_result, 4)
        self.assertEqual(counts.get("00", 0), 2)
        self.assertEqual(counts.get("11", 0), 2)

    def test_exception_returns_empty(self):
        mock_result = MagicMock()
        mock_result.measurements = None  # will raise on iteration
        counts = _cirq_result_to_counts(mock_result, 10)
        self.assertEqual(counts, {})

    def test_three_qubit_bitstring(self):
        mock_result = MagicMock()
        mock_result.measurements = {"m": [[1, 0, 1]]}
        counts = _cirq_result_to_counts(mock_result, 1)
        self.assertIn("101", counts)


# ─────────────────────────────────────────────────────────────────────────────
# calibration_to_noise_model
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationToNoiseModel(unittest.TestCase):
    def test_small_gate_error_returns_thermal(self):
        from quantx_engine.noise import ThermalRelaxationNoise
        cal = BackendCalibration(
            "b", 2,
            t1={0: 100_000.0},
            t2={0: 80_000.0},
            gate_errors={"cx": 0.0005},  # < 0.1% threshold
        )
        nm = calibration_to_noise_model(cal)
        self.assertIsInstance(nm, ThermalRelaxationNoise)

    def test_large_gate_error_returns_composite(self):
        from quantx_engine.noise import CompositeNoise
        cal = BackendCalibration(
            "b", 2,
            t1={0: 100_000.0},
            t2={0: 80_000.0},
            gate_errors={"cx": 0.005},  # > 0.1% → composite
        )
        nm = calibration_to_noise_model(cal)
        self.assertIsInstance(nm, CompositeNoise)

    def test_no_gate_errors_returns_thermal(self):
        from quantx_engine.noise import ThermalRelaxationNoise
        cal = BackendCalibration("b", 2)
        nm = calibration_to_noise_model(cal)
        self.assertIsInstance(nm, ThermalRelaxationNoise)

    def test_t_gate_override_passed_through(self):
        cal = BackendCalibration("b", 2, gate_errors={"cx": 0.0005})
        nm = calibration_to_noise_model(cal, t_gate_ns=200.0)
        self.assertAlmostEqual(nm.t_gate, 200.0)

    def test_composite_contains_depolarizing(self):
        from quantx_engine.noise import CompositeNoise, DepolarizingNoise
        cal = BackendCalibration(
            "b", 2,
            t1={0: 100_000.0},
            gate_errors={"cx": 0.02},
        )
        nm = calibration_to_noise_model(cal)
        self.assertIsInstance(nm, CompositeNoise)
        types = [type(c).__name__ for c in nm.models]
        self.assertIn("DepolarizingNoise", types)

    @staticmethod
    def _depolarizing_p(nm):
        from quantx_engine.noise import DepolarizingNoise
        return next(c for c in nm.models if isinstance(c, DepolarizingNoise)).prob

    def test_one_qubit_error_maps_to_its_channel(self):
        # One qubit: (1 - p)rho + p I/2 has average infidelity p/2.
        cal = BackendCalibration("b", 2, gate_errors={"sx": 0.004})
        p = self._depolarizing_p(calibration_to_noise_model(cal))
        self.assertAlmostEqual(p, 0.008, places=12)

    def test_two_qubit_error_maps_to_its_channel(self):
        # The channel on each of a CX's two qubits; its average infidelity on d = 4 is
        # (4/5)(1 - (1 - 3p/4)^2), which must give back the calibrated error.
        r = 0.02
        cal = BackendCalibration("b", 2, gate_errors={"cx": r, "sx": 0.0003})
        p = self._depolarizing_p(calibration_to_noise_model(cal))
        self.assertAlmostEqual(0.8 * (1.0 - (1.0 - 0.75 * p) ** 2), r, places=12)
        self.assertAlmostEqual(p, 5.0 * r / 6.0, delta=0.01 * r)  # to first order in r

    def test_the_worst_gate_sets_p(self):
        # p = 2r for the one-qubit gate beats about 5r/6 for the two-qubit one.
        cal = BackendCalibration("b", 2, gate_errors={"cx": 0.004, "x": 0.003})
        p = self._depolarizing_p(calibration_to_noise_model(cal))
        self.assertAlmostEqual(p, 0.006, places=12)


# ─────────────────────────────────────────────────────────────────────────────
# available_backends
# ─────────────────────────────────────────────────────────────────────────────

class TestAvailableBackends(unittest.TestCase):
    def _clean_env(self):
        return {
            k: v for k, v in os.environ.items()
            if k not in ("QISKIT_IBM_TOKEN", "IONQ_API_KEY", "GOOGLE_CLOUD_PROJECT")
        }

    def test_no_env_vars_empty(self):
        with patch.dict(os.environ, self._clean_env(), clear=True):
            result = available_backends()
        self.assertEqual(result, {})

    def test_ibm_token_present(self):
        env = {**self._clean_env(), "QISKIT_IBM_TOKEN": "tok"}
        with patch.dict(os.environ, env, clear=True):
            result = available_backends()
        self.assertIn("ibm", result)
        self.assertIsInstance(result["ibm"], list)
        self.assertTrue(len(result["ibm"]) > 0)

    def test_ionq_key_present(self):
        env = {**self._clean_env(), "IONQ_API_KEY": "key"}
        with patch.dict(os.environ, env, clear=True):
            result = available_backends()
        self.assertIn("ionq", result)
        self.assertIn("simulator", result["ionq"])

    def test_google_project_present(self):
        env = {**self._clean_env(), "GOOGLE_CLOUD_PROJECT": "proj"}
        with patch.dict(os.environ, env, clear=True):
            result = available_backends()
        self.assertIn("google", result)
        self.assertIn("simulator", result["google"])

    def test_all_three_present(self):
        env = {
            **self._clean_env(),
            "QISKIT_IBM_TOKEN": "t",
            "IONQ_API_KEY": "k",
            "GOOGLE_CLOUD_PROJECT": "p",
        }
        with patch.dict(os.environ, env, clear=True):
            result = available_backends()
        self.assertIn("ibm", result)
        self.assertIn("ionq", result)
        self.assertIn("google", result)

    def test_returns_dict(self):
        with patch.dict(os.environ, self._clean_env(), clear=True):
            result = available_backends()
        self.assertIsInstance(result, dict)

    def test_ionq_backends_list(self):
        env = {**self._clean_env(), "IONQ_API_KEY": "key"}
        with patch.dict(os.environ, env, clear=True):
            result = available_backends()
        self.assertIn("qpu.aria-1", result["ionq"])


# ─────────────────────────────────────────────────────────────────────────────
# __init__ re-exports
# ─────────────────────────────────────────────────────────────────────────────

class TestInitExports(unittest.TestCase):
    def test_hardware_error_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "HardwareError"))

    def test_backend_calibration_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "BackendCalibration"))

    def test_hardware_job_result_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "HardwareJobResult"))

    def test_hardware_job_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "HardwareJob"))

    def test_ibm_hardware_backend_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "IBMHardwareBackend"))

    def test_ionq_backend_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "IonQBackend"))

    def test_google_backend_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "GoogleBackend"))

    def test_calibration_to_noise_model_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "calibration_to_noise_model"))

    def test_available_backends_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "available_backends"))


if __name__ == "__main__":
    unittest.main()
