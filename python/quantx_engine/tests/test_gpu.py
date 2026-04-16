"""Tests for the GPU backend module (all pass on CPU via numpy fallback)."""
from __future__ import annotations
import math
import pytest

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

pytestmark = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy required")

from quantx_engine.gpu import (
    GpuError, GpuBackend, GpuStatevector, GpuLanczos, GpuBatchedCircuit,
    gpu_info, to_gpu, to_cpu,
)


# ---------------------------------------------------------------------------
# GpuBackend tests
# ---------------------------------------------------------------------------

def test_backend_auto_selects_numpy_without_cupy():
    b = GpuBackend("auto")
    assert b.name in ("numpy", "cupy")


def test_backend_numpy_explicit():
    b = GpuBackend("numpy")
    assert b.name == "numpy"
    assert b.is_gpu() is False


def test_backend_cupy_raises_without_cupy():
    if _HAS_CUPY:
        pytest.skip("CuPy installed; skip 'no cupy' test")
    with pytest.raises(GpuError, match="CuPy"):
        GpuBackend("cupy")


def test_backend_invalid_raises():
    with pytest.raises(GpuError):
        GpuBackend("tpu")


def test_backend_to_dict():
    b = GpuBackend("numpy")
    d = b.to_dict()
    assert "backend" in d and "gpu_available" in d


# ---------------------------------------------------------------------------
# GpuStatevector tests (all use backend="numpy")
# ---------------------------------------------------------------------------

def test_sv_initial_state():
    sv = GpuStatevector(n_qubits=2, backend="numpy")
    state = sv.state()
    assert state.shape == (4,)
    assert abs(state[0] - 1.0) < 1e-10


def test_sv_norm_is_one():
    sv = GpuStatevector(n_qubits=3, backend="numpy")
    assert abs(sv.norm() - 1.0) < 1e-10


def test_sv_hadamard_superposition():
    sv = GpuStatevector(n_qubits=1, backend="numpy")
    sv.apply_gate("H", 0)
    probs = sv.probabilities()
    np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-10)


def test_sv_pauli_x_flips():
    sv = GpuStatevector(n_qubits=1, backend="numpy")
    sv.apply_gate("X", 0)
    state = sv.state()
    assert abs(state[1] - 1.0) < 1e-10
    assert abs(state[0]) < 1e-10


def test_sv_cnot_entangles():
    sv = GpuStatevector(n_qubits=2, backend="numpy")
    sv.apply_gate("H", 0)
    sv.apply_gate("CNOT", [0, 1])
    probs = sv.probabilities()
    # Bell state: |00⟩ and |11⟩ each with prob 0.5
    np.testing.assert_allclose(probs[[0, 3]], [0.5, 0.5], atol=1e-10)
    np.testing.assert_allclose(probs[[1, 2]], [0.0, 0.0], atol=1e-10)


def test_sv_rx_expectation():
    sv = GpuStatevector(n_qubits=1, backend="numpy")
    theta = 0.7
    sv.apply_gate("RX", 0, params=[theta])
    expval = sv.expectation_z(0)
    assert abs(expval - math.cos(theta)) < 1e-10


def test_sv_ry_expectation():
    sv = GpuStatevector(n_qubits=1, backend="numpy")
    theta = 1.2
    sv.apply_gate("RY", 0, params=[theta])
    expval = sv.expectation_z(0)
    assert abs(expval - math.cos(theta)) < 1e-10


def test_sv_reset():
    sv = GpuStatevector(n_qubits=2, backend="numpy")
    sv.apply_gate("X", 0)
    sv.reset()
    assert abs(sv.state()[0] - 1.0) < 1e-10


def test_sv_marginal_probs():
    sv = GpuStatevector(n_qubits=2, backend="numpy")
    sv.apply_gate("X", 0)  # |10⟩
    probs0 = sv.probabilities(qubits=[0])
    probs1 = sv.probabilities(qubits=[1])
    np.testing.assert_allclose(probs0, [0.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(probs1, [1.0, 0.0], atol=1e-10)


def test_sv_to_dict():
    sv = GpuStatevector(n_qubits=3, backend="numpy")
    d = sv.to_dict()
    assert d["n_qubits"] == 3
    assert d["hilbert_dim"] == 8
    assert "backend" in d


def test_sv_invalid_n_qubits():
    with pytest.raises(GpuError):
        GpuStatevector(n_qubits=0, backend="numpy")


def test_sv_three_qubit_ghz():
    sv = GpuStatevector(n_qubits=3, backend="numpy")
    sv.apply_gate("H", 0)
    sv.apply_gate("CNOT", [0, 1])
    sv.apply_gate("CNOT", [1, 2])
    probs = sv.probabilities()
    assert abs(probs[0] - 0.5) < 1e-10  # |000⟩
    assert abs(probs[7] - 0.5) < 1e-10  # |111⟩


# ---------------------------------------------------------------------------
# GpuLanczos tests
# ---------------------------------------------------------------------------

def test_gpu_lanczos_dense():
    H = np.diag([1.0, 2.0, 3.0, 4.0])
    solver = GpuLanczos(k=2, backend="numpy", seed=0)
    result = solver.solve(H)
    assert len(result.eigenvalues) == 2
    np.testing.assert_allclose(np.sort(result.eigenvalues), [1.0, 2.0], atol=1e-8)


def test_gpu_lanczos_sparse():
    from quantx_engine.lanczos import SparseHamiltonian
    H = SparseHamiltonian.ising_1d(n_sites=4, J=1.0, h=0.0)
    solver = GpuLanczos(k=2, backend="numpy", seed=1)
    result = solver.solve(H)
    assert result.eigenvalues[0] < 0  # ferromagnetic ground state is negative


def test_gpu_lanczos_result_type():
    from quantx_engine.lanczos import EigenResult
    H = np.eye(4)
    solver = GpuLanczos(k=1, backend="numpy", seed=2)
    result = solver.solve(H)
    assert isinstance(result, EigenResult)


def test_gpu_lanczos_to_dict():
    solver = GpuLanczos(k=4, backend="numpy")
    d = solver.to_dict()
    assert "k" in d and "backend" in d


# ---------------------------------------------------------------------------
# GpuBatchedCircuit tests
# ---------------------------------------------------------------------------

def test_batched_initial_state():
    bc = GpuBatchedCircuit(n_qubits=2, n_circuits=4, backend="numpy")
    probs = bc.probabilities()
    assert probs.shape == (4, 4)
    np.testing.assert_allclose(probs[:, 0], np.ones(4), atol=1e-10)


def test_batched_pauli_x():
    bc = GpuBatchedCircuit(n_qubits=1, n_circuits=3, backend="numpy")
    bc.apply_gate("X", 0)
    probs = bc.probabilities()
    np.testing.assert_allclose(probs[:, 1], np.ones(3), atol=1e-10)


def test_batched_hadamard():
    bc = GpuBatchedCircuit(n_qubits=1, n_circuits=2, backend="numpy")
    bc.apply_gate("H", 0)
    probs = bc.probabilities()
    np.testing.assert_allclose(probs, [[0.5, 0.5], [0.5, 0.5]], atol=1e-10)


def test_batched_expectation_z():
    bc = GpuBatchedCircuit(n_qubits=1, n_circuits=2, backend="numpy")
    expvals = bc.expectation_z(0)
    np.testing.assert_allclose(expvals, [1.0, 1.0], atol=1e-10)


def test_batched_cnot():
    bc = GpuBatchedCircuit(n_qubits=2, n_circuits=2, backend="numpy")
    bc.apply_gate("X", 0)   # |10⟩
    bc.apply_cnot(0, 1)     # → |11⟩
    probs = bc.probabilities()
    np.testing.assert_allclose(probs[:, 3], [1.0, 1.0], atol=1e-10)


def test_batched_reset():
    bc = GpuBatchedCircuit(n_qubits=1, n_circuits=2, backend="numpy")
    bc.apply_gate("X", 0)
    bc.reset()
    probs = bc.probabilities()
    np.testing.assert_allclose(probs[:, 0], [1.0, 1.0], atol=1e-10)


def test_batched_to_dict():
    bc = GpuBatchedCircuit(n_qubits=2, n_circuits=4, backend="numpy")
    d = bc.to_dict()
    assert d["n_qubits"] == 2
    assert d["n_circuits"] == 4
    assert "backend" in d


def test_batched_rx_per_circuit():
    """Different RX angle per circuit."""
    bc = GpuBatchedCircuit(n_qubits=1, n_circuits=3, backend="numpy")
    thetas = [0.0, math.pi / 2, math.pi]
    bc.apply_gate("RX", 0, params=thetas)
    expvals = bc.expectation_z(0)
    expected = [math.cos(t) for t in thetas]
    np.testing.assert_allclose(expvals, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

def test_gpu_info_returns_dict():
    info = gpu_info()
    assert isinstance(info, dict)
    assert "cupy_available" in info


def test_to_gpu_returns_array():
    arr = np.array([1.0, 2.0, 3.0])
    result = to_gpu(arr)
    assert hasattr(result, '__array__')


def test_to_cpu_returns_numpy():
    arr = np.array([1.0, 2.0, 3.0])
    result = to_cpu(arr)
    assert isinstance(result, np.ndarray)


# ── Codex fix tests ───────────────────────────────────────────────────────────

def test_batched_scalar_rx_param():
    """P1: scalar RX param should apply same angle to all circuits."""
    bc = GpuBatchedCircuit(n_qubits=1, n_circuits=3, backend="numpy")
    bc.apply_gate("RX", 0, params=math.pi)   # scalar param, not a list
    expvals = bc.expectation_z(0)
    np.testing.assert_allclose(expvals, [-1.0, -1.0, -1.0], atol=1e-8)


def test_batched_single_element_list_rx():
    """P1: single-element list should broadcast to all circuits."""
    bc = GpuBatchedCircuit(n_qubits=1, n_circuits=3, backend="numpy")
    bc.apply_gate("RX", 0, params=[math.pi])  # singleton list
    expvals = bc.expectation_z(0)
    np.testing.assert_allclose(expvals, [-1.0, -1.0, -1.0], atol=1e-8)


def test_gpu_lanczos_callable_raises_helpful_error():
    """P2: bare callable raises GpuError with helpful message."""
    solver = GpuLanczos(k=2, backend="numpy", seed=0)
    with pytest.raises(GpuError, match="solve_matvec"):
        solver.solve(lambda v: v)


def test_gpu_lanczos_solve_matvec():
    """P2: solve_matvec with identity gives eigenvalues ~1.0."""
    diag = np.array([1.0, 2.0, 3.0, 4.0])
    solver = GpuLanczos(k=4, backend="numpy", seed=0)
    result = solver.solve_matvec(lambda v: diag * v, n=4)
    np.testing.assert_allclose(np.sort(result.eigenvalues), diag, atol=1e-8)
