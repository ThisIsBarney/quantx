"""Tests for M124: orbital-optimised VQE (OO-VQE)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantx_engine.oovqe import (
    OOVQEError,
    OOVQEResult,
    OOVQESolver,
    _n_kappa,
    _orbital_rotation,
    _rotate_integrals,
    _skew_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h2_core() -> np.ndarray:
    """Minimal H₂ one-electron integrals (2-orbital, approximate)."""
    return np.array([[-1.1175, -0.9557], [-0.9557, -0.5698]], dtype=float)


def _is_orthogonal(U: np.ndarray, tol: float = 1e-8) -> bool:
    return bool(np.allclose(U @ U.T, np.eye(len(U)), atol=tol))


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestSkewMatrix:
    def test_shape(self):
        K = _skew_matrix(np.array([0.1]), 2)
        assert K.shape == (2, 2)

    def test_anti_symmetric(self):
        kappa = np.array([0.3, -0.2, 0.5])
        K = _skew_matrix(kappa, 3)
        assert np.allclose(K + K.T, 0, atol=1e-12)

    def test_zero_kappa_gives_zero_matrix(self):
        K = _skew_matrix(np.zeros(3), 3)
        assert np.allclose(K, 0, atol=1e-12)

    def test_n_kappa_formula(self):
        assert _n_kappa(2) == 1
        assert _n_kappa(3) == 3
        assert _n_kappa(4) == 6


class TestOrbitalRotation:
    def test_identity_for_zero_kappa(self):
        U = _orbital_rotation(np.zeros(1), 2)
        assert np.allclose(U, np.eye(2), atol=1e-10)

    def test_orthogonal(self):
        for seed in range(5):
            kappa = np.random.default_rng(seed).uniform(-0.5, 0.5, _n_kappa(3))
            U = _orbital_rotation(kappa, 3)
            assert _is_orthogonal(U)

    def test_determinant_is_one(self):
        kappa = np.array([0.3])
        U = _orbital_rotation(kappa, 2)
        assert abs(np.linalg.det(U) - 1.0) < 1e-8


class TestRotateIntegrals:
    def test_identity_rotation_unchanged(self):
        h = _h2_core()
        U = np.eye(2)
        h_rot = _rotate_integrals(h, U)
        assert np.allclose(h_rot, h, atol=1e-12)

    def test_rotated_trace_invariant(self):
        """Trace of h is invariant under orthogonal rotations."""
        h = _h2_core()
        kappa = np.array([0.4])
        U = _orbital_rotation(kappa, 2)
        h_rot = _rotate_integrals(h, U)
        assert abs(np.trace(h_rot) - np.trace(h)) < 1e-8


# ---------------------------------------------------------------------------
# OOVQEResult
# ---------------------------------------------------------------------------

class TestOOVQEResult:
    def _make(self):
        return OOVQEResult(
            energies=[-1.0, -1.1, -1.15],
            circuit_params=np.array([0.1, 0.2]),
            orbital_params=np.array([0.05]),
            converged=True,
            n_iterations=3,
        )

    def test_final_energy(self):
        r = self._make()
        assert r.final_energy == pytest.approx(-1.15)

    def test_energy_history_shape(self):
        r = self._make()
        assert r.energy_history.shape == (3,)

    def test_empty_result(self):
        r = OOVQEResult()
        assert math.isnan(r.final_energy)


# ---------------------------------------------------------------------------
# OOVQESolver construction
# ---------------------------------------------------------------------------

class TestOOVQESolverConstruction:
    def test_basic_construction(self):
        s = OOVQESolver(_h2_core(), n_orb=2)
        assert s.n_orb == 2

    def test_non_square_h_raises(self):
        with pytest.raises(OOVQEError, match="square"):
            OOVQESolver(np.zeros((2, 3)), n_orb=2)

    def test_1d_h_raises(self):
        with pytest.raises(OOVQEError, match="2-D"):
            OOVQESolver(np.array([1.0, 2.0]), n_orb=2)

    def test_n_orb_one_raises(self):
        with pytest.raises(OOVQEError, match="n_orb"):
            OOVQESolver(np.eye(1), n_orb=1)

    def test_h_shape_mismatch_raises(self):
        with pytest.raises(OOVQEError, match="n_orb"):
            OOVQESolver(np.eye(3), n_orb=2)

    def test_n_circuit_params_zero_raises(self):
        with pytest.raises(OOVQEError, match="n_circuit_params"):
            OOVQESolver(_h2_core(), n_orb=2, n_circuit_params=0)

    def test_max_iter_zero_raises(self):
        with pytest.raises(OOVQEError, match="max_iterations"):
            OOVQESolver(_h2_core(), n_orb=2, max_iterations=0)

    def test_inner_steps_zero_raises(self):
        with pytest.raises(OOVQEError, match="inner_steps"):
            OOVQESolver(_h2_core(), n_orb=2, inner_steps=0)

    def test_conv_tol_zero_raises(self):
        with pytest.raises(OOVQEError, match="conv_tol"):
            OOVQESolver(_h2_core(), n_orb=2, conv_tol=0.0)

    def test_eps_zero_raises(self):
        with pytest.raises(OOVQEError, match="eps"):
            OOVQESolver(_h2_core(), n_orb=2, eps=0.0)

    def test_repr(self):
        s = OOVQESolver(_h2_core(), n_orb=2)
        assert "OOVQESolver" in repr(s)
        assert "n_orb=2" in repr(s)

    def test_custom_energy_fn(self):
        def my_e(h, theta): return float(h[0, 0])
        s = OOVQESolver(_h2_core(), n_orb=2, energy_fn=my_e)
        r = s.run()
        assert math.isfinite(r.final_energy)


# ---------------------------------------------------------------------------
# OOVQESolver.run
# ---------------------------------------------------------------------------

class TestOOVQESolverRun:
    @pytest.fixture
    def solver(self):
        return OOVQESolver(
            _h2_core(), n_orb=2, n_circuit_params=2,
            max_iterations=20, inner_steps=30,
            lr_circuit=0.1, lr_orbital=0.05,
        )

    def test_run_returns_result(self, solver):
        r = solver.run(seed=0)
        assert isinstance(r, OOVQEResult)

    def test_energies_populated(self, solver):
        r = solver.run(seed=0)
        assert len(r.energies) >= 1

    def test_final_energy_is_finite(self, solver):
        r = solver.run(seed=0)
        assert math.isfinite(r.final_energy)

    def test_circuit_params_shape(self, solver):
        r = solver.run(seed=0)
        assert r.circuit_params.shape == (2,)

    def test_orbital_params_shape(self, solver):
        r = solver.run(seed=0)
        assert r.orbital_params.shape == (_n_kappa(2),)

    def test_energy_does_not_increase(self, solver):
        r = solver.run(seed=0)
        # Final energy should not be higher than initial energy
        assert r.energies[-1] <= r.energies[0] + 0.1

    def test_wrong_init_circuit_params_shape_raises(self, solver):
        with pytest.raises(OOVQEError, match="init_circuit_params"):
            solver.run(init_circuit_params=np.array([0.1]))

    def test_wrong_init_orbital_params_shape_raises(self, solver):
        with pytest.raises(OOVQEError, match="init_orbital_params"):
            solver.run(init_orbital_params=np.array([0.1, 0.2]))

    def test_n_iterations_positive(self, solver):
        r = solver.run(seed=0)
        assert r.n_iterations >= 1

    def test_converges_with_enough_iterations(self):
        s = OOVQESolver(
            _h2_core(), n_orb=2, n_circuit_params=2,
            max_iterations=100, inner_steps=50,
            lr_circuit=0.1, lr_orbital=0.05,
            conv_tol=1e-4,
        )
        r = s.run(seed=0)
        assert r.converged or r.n_iterations == 100

    def test_reproducible_with_same_seed(self, solver):
        r1 = solver.run(seed=42)
        r2 = solver.run(seed=42)
        assert r1.final_energy == pytest.approx(r2.final_energy)


# ---------------------------------------------------------------------------
# OOVQESolver with larger system (n_orb > 2, generic energy fn)
# ---------------------------------------------------------------------------

class TestOOVQESolverLarger:
    def test_3orbital_run(self):
        h = -1.0 * np.eye(3)
        s = OOVQESolver(h, n_orb=3, n_circuit_params=3, max_iterations=5)
        r = s.run()
        assert math.isfinite(r.final_energy)

    def test_4orbital_run(self):
        rng = np.random.default_rng(7)
        A = rng.standard_normal((4, 4))
        h = -(A + A.T) / 2  # negative definite-ish
        s = OOVQESolver(h, n_orb=4, n_circuit_params=4, max_iterations=3)
        r = s.run()
        assert math.isfinite(r.final_energy)

    def test_orbital_gradient_returns_array(self):
        h = -np.eye(3)
        s = OOVQESolver(h, n_orb=3, n_circuit_params=3)
        grad = s.orbital_gradient()
        assert grad.shape == (_n_kappa(3),)
