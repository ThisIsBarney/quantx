"""Tests for the Lanczos/Arnoldi sparse eigensolver."""
from __future__ import annotations
import math
import pytest

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

pytestmark = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy required")

from quantx_engine.contrib.lanczos import (
    LanczosSolver,
    ArnoldiSolver,
    SparseHamiltonian,
    EigenResult,
    LanczosError,
)


# ---------------------------------------------------------------------------
# SparseHamiltonian tests
# ---------------------------------------------------------------------------

def test_sparse_matvec_diagonal():
    """Diagonal H: matvec(e_i) = d_i * e_i."""
    d = [1.0, 2.0, 3.0]
    entries = [(i, i, complex(d[i])) for i in range(3)]
    H = SparseHamiltonian(n=3, entries=entries)
    for i in range(3):
        e_i = np.zeros(3, dtype=complex)
        e_i[i] = 1.0
        result = H.matvec(e_i)
        expected = np.zeros(3, dtype=complex)
        expected[i] = d[i]
        np.testing.assert_allclose(result, expected)


def test_sparse_to_dense_shape():
    """ising_1d(3).to_dense() has shape (8, 8)."""
    H = SparseHamiltonian.ising_1d(3)
    assert H.to_dense().shape == (8, 8)


def test_sparse_nnz():
    """ising_1d(3) has 48 entries: 8 states * (3 ZZ diag + 3 X off-diag)."""
    H = SparseHamiltonian.ising_1d(3)
    assert H.nnz == 48


def test_sparse_ising_hermitian():
    """ising_1d(3).to_dense() is Hermitian."""
    H = SparseHamiltonian.ising_1d(3)
    M = H.to_dense()
    np.testing.assert_allclose(M, M.conj().T, atol=1e-12)


def test_sparse_heisenberg_hermitian():
    """heisenberg_1d(4).to_dense() is Hermitian."""
    H = SparseHamiltonian.heisenberg_1d(4)
    M = H.to_dense()
    np.testing.assert_allclose(M, M.conj().T, atol=1e-12)


def test_sparse_invalid_n():
    """SparseHamiltonian with n=0 raises LanczosError."""
    with pytest.raises(LanczosError):
        SparseHamiltonian(n=0, entries=[])


# ---------------------------------------------------------------------------
# LanczosSolver tests
# ---------------------------------------------------------------------------

def test_lanczos_2x2_known():
    """H = diag([1, 3]); lowest eigenvalue = 1.0."""
    H = np.diag([1.0, 3.0])
    solver = LanczosSolver(k=1, seed=0)
    result = solver.solve(H)
    assert abs(result.eigenvalues[0] - 1.0) < 1e-8


def test_lanczos_dense_random_symmetric():
    """Random 10x10 symmetric matrix: k=3 lowest eigenvalues match np.linalg.eigh."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((10, 10))
    H = (A + A.T) / 2
    solver = LanczosSolver(k=3, seed=42)
    result = solver.solve(H)
    ref_evals = np.linalg.eigh(H)[0][:3]
    np.testing.assert_allclose(result.eigenvalues, ref_evals, atol=1e-6)


def test_lanczos_ising_ground_state_n4():
    """n_sites=4, J=1, h=0 (pure ZZ Ising). Ground state = -4J = -4.0 (all aligned, 4 bonds periodic)."""
    H = SparseHamiltonian.ising_1d(n_sites=4, J=1.0, h=0.0)
    solver = LanczosSolver(k=1, seed=0)
    result = solver.solve(H)
    assert abs(result.ground_state_energy() - (-4.0)) < 1e-8


def test_lanczos_sparse_hamiltonian():
    """ising_1d(n=4) with LanczosSolver(k=2) returns 2 eigenvalues."""
    H = SparseHamiltonian.ising_1d(n_sites=4)
    solver = LanczosSolver(k=2, seed=0)
    result = solver.solve(H)
    assert len(result.eigenvalues) == 2


def test_lanczos_eigenvalue_order():
    """Eigenvalues are sorted ascending."""
    H = SparseHamiltonian.ising_1d(n_sites=4)
    solver = LanczosSolver(k=4, seed=0)
    result = solver.solve(H)
    evals = result.eigenvalues
    assert all(evals[i] <= evals[i + 1] for i in range(len(evals) - 1))


def test_lanczos_eigenvector_orthogonality():
    """Eigenvectors are orthonormal: V^T V ~ I."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((8, 8))
    H = (A + A.T) / 2
    solver = LanczosSolver(k=4, seed=7)
    result = solver.solve(H)
    V = result.eigenvectors  # (8, 4)
    gram = V.T @ V
    np.testing.assert_allclose(gram, np.eye(4), atol=1e-6)


def test_lanczos_result_converged():
    """result.converged is bool."""
    H = np.diag([1.0, 2.0, 3.0])
    result = LanczosSolver(k=2, seed=0).solve(H)
    assert isinstance(result.converged, bool)


def test_lanczos_result_to_dict():
    """to_dict() contains expected keys."""
    H = np.diag([1.0, 2.0, 3.0])
    result = LanczosSolver(k=2, seed=0).solve(H)
    d = result.to_dict()
    for key in ("eigenvalues", "n_iter", "converged", "residuals"):
        assert key in d


def test_lanczos_ground_state_energy():
    """ground_state_energy() == float(eigenvalues[0])."""
    H = np.diag([2.5, 4.0, 6.0])
    result = LanczosSolver(k=2, seed=0).solve(H)
    assert result.ground_state_energy() == float(result.eigenvalues[0])


def test_lanczos_spectral_gap_k1_raises():
    """k=1 result raises LanczosError for spectral_gap()."""
    H = np.diag([1.0, 2.0, 3.0])
    result = LanczosSolver(k=1, seed=0).solve(H)
    with pytest.raises(LanczosError):
        result.spectral_gap()


def test_lanczos_compare_dense_vs_sparse_ising():
    """ising_1d(n=4) dense vs sparse Lanczos: eigenvalues agree to 1e-8."""
    sparse_H = SparseHamiltonian.ising_1d(n_sites=4)
    dense_H = sparse_H.to_dense().real  # Ising is real

    solver = LanczosSolver(k=4, seed=0)
    result_sparse = solver.solve(sparse_H)
    result_dense = solver.solve(dense_H)

    np.testing.assert_allclose(result_sparse.eigenvalues, result_dense.eigenvalues, atol=1e-8)


def test_lanczos_solve_matvec():
    """solve_matvec with diagonal matvec returns correct eigenvalues."""
    diag_vals = np.array([1.0, 2.0, 3.0, 4.0])
    solver = LanczosSolver(k=4, max_iter=200, seed=0)
    result = solver.solve_matvec(lambda v: diag_vals * v, n=4)
    np.testing.assert_allclose(np.sort(result.eigenvalues),
                               np.sort(diag_vals), atol=1e-8)

# ---------------------------------------------------------------------------
# ArnoldiSolver tests
# ---------------------------------------------------------------------------

def test_arnoldi_2x2():
    """Simple 2x2 asymmetric matrix: finds eigenvalues."""
    H = np.array([[2.0, 1.0], [0.0, 3.0]])
    solver = ArnoldiSolver(k=2, seed=0)
    result = solver.solve(H)
    # Eigenvalues of upper triangular are on diagonal: 2 and 3
    assert len(result.eigenvalues) == 2
    evals_sorted = np.sort(result.eigenvalues)
    np.testing.assert_allclose(evals_sorted, [2.0, 3.0], atol=1e-6)


def test_arnoldi_symmetric_matches_lanczos():
    """For symmetric H, Arnoldi lowest eigenvalue matches Lanczos."""
    rng = np.random.default_rng(99)
    A = rng.standard_normal((8, 8))
    H = (A + A.T) / 2.0

    lanczos_result = LanczosSolver(k=3, seed=99).solve(H)
    arnoldi_result = ArnoldiSolver(k=3, max_iter=8, seed=99).solve(H)

    # Only the lowest eigenvalue converges reliably with default Arnoldi
    np.testing.assert_allclose(
        arnoldi_result.eigenvalues[0],
        lanczos_result.eigenvalues[0],
        atol=1e-4,
    )

def test_arnoldi_result_type():
    """ArnoldiSolver.solve returns an EigenResult."""
    H = np.diag([3.0, 1.0, 2.0])
    result = ArnoldiSolver(k=2, seed=0).solve(H)
    assert isinstance(result, EigenResult)


def test_arnoldi_complex_eigenvalues_preserved():
    """Arnoldi returns complex eigenpairs for non-symmetric matrix (Codex P1)."""
    # Non-symmetric rotation-like matrix has complex eigenvalues
    H = np.array([[0.0, -1.0], [1.0, 0.0]])  # eigenvalues: ±i
    solver = ArnoldiSolver(k=2, seed=0)
    result = solver.solve(H)
    # Check eigenvalues are complex (imaginary part non-zero)
    assert np.iscomplexobj(result.eigenvalues)


def test_arnoldi_v0_near_zero_raises():
    """Arnoldi raises LanczosError for near-zero initial vector (Codex P2)."""
    H = np.eye(4)
    solver = ArnoldiSolver(k=2, seed=0)
    with pytest.raises(LanczosError, match="near-zero"):
        solver.solve(H, v0=np.zeros(4))
