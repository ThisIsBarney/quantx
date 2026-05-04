"""M109: Tests for GPU-accelerated MPS simulator (CuPy backend + CPU fallback).

All tests run on CPU and pass regardless of whether CuPy/CUDA is installed.
GPU-specific assertions are conditionally enabled via ``is_gpu_available()``.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import warnings

from quantx_engine.mps_simulator import (
    MPSSimulator,
    MPSSimulatorError,
    GpuBenchmarkResult,
    BenchmarkResult,
    get_backend_module,
    is_gpu_available,
    gpu_mps_benchmark,
    ghz_fidelity,
    compare_mps_sv,
)


# ── get_backend_module ────────────────────────────────────────────────────────

class TestGetBackendModule:
    def test_cpu_returns_numpy(self):
        xp = get_backend_module("cpu")
        assert xp is np

    def test_cuda_fallback_to_numpy_when_no_gpu(self):
        if is_gpu_available():
            pytest.skip("GPU present; fallback not triggered")
        with pytest.warns(UserWarning, match=r"CuPy|fallback"):
            xp = get_backend_module("cuda")
        assert xp is np

    def test_unknown_backend_raises(self):
        with pytest.raises(MPSSimulatorError, match="Unknown backend"):
            get_backend_module("tpu")


# ── MPSSimulator backend parameter ───────────────────────────────────────────

class TestMPSSimulatorBackend:
    def test_default_backend_is_cpu(self):
        sim = MPSSimulator(n_qubits=4)
        assert sim.backend == "cpu"

    def test_explicit_cpu_backend(self):
        sim = MPSSimulator(n_qubits=4, backend="cpu")
        assert sim.backend == "cpu"

    def test_cuda_backend_falls_back_gracefully(self):
        """MPSSimulator(backend='cuda') must not raise even without a GPU."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sim = MPSSimulator(n_qubits=4, backend="cuda")
        # After fallback, simulator must still work correctly
        sim.h(0)
        sim.cx(0, 1)
        assert sim.norm() == pytest.approx(1.0, abs=1e-10)

    def test_cuda_fallback_updates_backend_attribute(self):
        """P2: when CUDA falls back to CPU, .backend must report 'cpu', not 'cuda'."""
        if is_gpu_available():
            pytest.skip("Real GPU present; no fallback occurs")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sim = MPSSimulator(n_qubits=2, backend="cuda")
        # Backend must reflect the *resolved* compute substrate
        assert sim.backend == "cpu", (
            f"Expected backend='cpu' after CPU fallback, got '{sim.backend}'"
        )
        # to_cpu() on a CPU sim should return self (no redundant copy)
        assert sim.to_cpu() is sim

    def test_cuda_backend_produces_correct_ghz_fidelity(self):
        """Results from GPU (or CPU fallback) backend match CPU reference."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sim_gpu = MPSSimulator(n_qubits=6, max_bond=4, backend="cuda")
        sim_cpu = MPSSimulator(n_qubits=6, max_bond=4, backend="cpu")

        for sim in (sim_gpu, sim_cpu):
            sim.h(0)
            for q in range(5):
                sim.cx(q, q + 1)

        sv_cpu = sim_cpu.to_statevector()
        sv_gpu = sim_gpu.to_statevector()

        fidelity = float(abs(np.dot(sv_cpu.conj(), sv_gpu)) ** 2)
        assert fidelity == pytest.approx(1.0, abs=1e-8)

    def test_to_cpu_returns_cpu_sim(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sim = MPSSimulator(n_qubits=3, backend="cuda")
        sim.h(0)
        sim.cx(0, 1)
        cpu_sim = sim.to_cpu()
        assert cpu_sim.backend == "cpu"
        assert cpu_sim.norm() == pytest.approx(1.0, abs=1e-10)

    def test_to_cpu_on_cpu_returns_self(self):
        sim = MPSSimulator(n_qubits=3, backend="cpu")
        assert sim.to_cpu() is sim

    def test_to_statevector_returns_numpy_array(self):
        """to_statevector() must always return a plain numpy array."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sim = MPSSimulator(n_qubits=3, backend="cuda")
        sim.h(0)
        sv = sim.to_statevector()
        assert isinstance(sv, np.ndarray)

    def test_to_mps_returns_mps_with_numpy_tensors(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sim = MPSSimulator(n_qubits=3, backend="cuda")
        from quantx_engine.tensor_network import MPS
        mps = sim.to_mps()
        assert isinstance(mps, MPS)
        for t in mps.tensors:
            assert isinstance(t, np.ndarray)


# ── GPU backend parity with CPU ───────────────────────────────────────────────

class TestGpuCpuParity:
    """Verify GPU backend produces identical results to CPU across gate types."""

    @pytest.mark.parametrize("n_qubits,seed", [(4, 0), (6, 1), (8, 2)])
    def test_random_circuit_parity(self, n_qubits: int, seed: int):
        rng = np.random.default_rng(seed)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sim_gpu = MPSSimulator(n_qubits=n_qubits, max_bond=16, backend="cuda")
        sim_cpu = MPSSimulator(n_qubits=n_qubits, max_bond=16, backend="cpu")

        for _ in range(3):
            angles = rng.uniform(0, 2 * math.pi, n_qubits)
            for q in range(n_qubits):
                sim_cpu.ry(float(angles[q]), q)
                sim_gpu.ry(float(angles[q]), q)
            for parity in (0, 1):
                for q in range(parity, n_qubits - 1, 2):
                    sim_cpu.cz(q, q + 1)
                    sim_gpu.cz(q, q + 1)

        sv_cpu = sim_cpu.to_statevector()
        sv_gpu = sim_gpu.to_statevector()
        np.testing.assert_allclose(sv_cpu, sv_gpu, atol=1e-10)

    def test_all_named_gates_parity(self):
        """Single-qubit named gates produce identical results."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sim_gpu = MPSSimulator(n_qubits=4, backend="cuda")
        sim_cpu = MPSSimulator(n_qubits=4, backend="cpu")

        for sim in (sim_cpu, sim_gpu):
            sim.h(0)
            sim.x(1)
            sim.y(2)
            sim.z(3)
            sim.s(0)
            sim.t(1)
            sim.rx(0.5, 2)
            sim.ry(1.0, 3)
            sim.rz(1.5, 0)
            sim.cx(0, 1)
            sim.cz(2, 3)

        np.testing.assert_allclose(
            sim_cpu.to_statevector(), sim_gpu.to_statevector(), atol=1e-10
        )


# ── GpuBenchmarkResult ────────────────────────────────────────────────────────

class TestGpuBenchmarkResult:
    def test_returns_gpu_benchmark_result(self):
        result = gpu_mps_benchmark(n_qubits=6, circuit_depth=2, max_bond=4, seed=42)
        assert isinstance(result, GpuBenchmarkResult)

    def test_speedup_positive(self):
        result = gpu_mps_benchmark(n_qubits=6, circuit_depth=2, max_bond=4, seed=0)
        assert result.speedup > 0.0

    def test_gpu_available_field_is_bool(self):
        result = gpu_mps_benchmark(n_qubits=4, circuit_depth=1, max_bond=4)
        assert isinstance(result.gpu_available, bool)
        assert result.gpu_available == is_gpu_available()

    def test_truncation_errors_match_cpu_run(self):
        """cpu_truncation_error must match a standalone mps_benchmark call."""
        from quantx_engine.mps_simulator import mps_benchmark
        ref = mps_benchmark(n_qubits=6, circuit_depth=2, max_bond=4, seed=42)
        result = gpu_mps_benchmark(n_qubits=6, circuit_depth=2, max_bond=4, seed=42)
        assert result.cpu_truncation_error == pytest.approx(ref.truncation_error, rel=1e-6)

    def test_no_gpu_speedup_near_one(self):
        """Without a GPU, both runs are CPU → speedup ≈ 1 (within noise)."""
        if is_gpu_available():
            pytest.skip("Real GPU present; this test is for CPU-only environments")
        result = gpu_mps_benchmark(n_qubits=4, circuit_depth=1, max_bond=4)
        # Both runs are identical CPU code; speedup can differ slightly due to
        # cache/OS effects, but should be in a reasonable range
        assert 0.1 < result.speedup < 100.0

    def test_default_n30_runs_without_error(self):
        """The advertised default benchmark (n=30) completes successfully."""
        result = gpu_mps_benchmark(n_qubits=30, circuit_depth=3, max_bond=16, seed=0)
        assert result.n_qubits == 30
        assert result.cpu_seconds > 0.0
        assert result.gpu_seconds > 0.0

    def test_warmup_false_still_works(self):
        result = gpu_mps_benchmark(n_qubits=4, circuit_depth=1, max_bond=4, warmup=False)
        assert isinstance(result, GpuBenchmarkResult)


# ── is_gpu_available ──────────────────────────────────────────────────────────

def test_is_gpu_available_returns_bool():
    assert isinstance(is_gpu_available(), bool)


# ── Backward compatibility: existing CPU API unchanged ────────────────────────

def test_cpu_ghz_fidelity_unchanged():
    """Existing ghz_fidelity function still works after M109 changes."""
    fid = ghz_fidelity(n_qubits=10, max_bond=2)
    assert fid == pytest.approx(1.0, abs=1e-8)


def test_cpu_compare_mps_sv_unchanged():
    result = compare_mps_sv(n_qubits=6, circuit_depth=3, max_bond=16, seed=7)
    assert result.fidelity == pytest.approx(1.0, abs=1e-6)


def test_adaptive_truncation_duck_typing_with_numpy():
    """AdaptiveTruncation.truncate uses duck-typed .sum() — verify no regression."""
    from quantx_engine.mps_simulator import AdaptiveTruncation
    at = AdaptiveTruncation(max_bond=4, eps=0.1)
    S = np.array([3.0, 2.0, 0.1, 0.01])
    U = np.eye(4)
    Vh = np.eye(4)
    Ut, St, Vht, err = at.truncate(U, S, Vh)
    # eps=0.1 → threshold = 0.3; S[2]=0.1 < 0.3, S[3]=0.01 < 0.3 → keep=2 (capped at max_bond=4)
    assert len(St) == 2
    assert err == pytest.approx(0.1**2 + 0.01**2)
