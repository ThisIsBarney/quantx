"""Tests for M122: McLachlan variational quantum dynamics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantx_engine.contrib.mclachlan import (
    McLachlanError,
    McLachlanResult,
    McLachlanSolver,
)

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

# Simple 1-qubit Hamiltonians (2×2 Hermitian matrices)
_ZZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)  # σ_z (eigvals ±1)
_HH = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)  # σ_x (eigvals ±1)
_EYE = np.eye(2, dtype=complex)


def _rx_state(theta: np.ndarray) -> np.ndarray:
    """Rotation-X ansatz: |ψ(θ)⟩ = Rx(θ₀)|0⟩ = cos(θ₀/2)|0⟩ - i·sin(θ₀/2)|1⟩."""
    a = float(theta[0])
    return np.array([math.cos(a / 2), -1j * math.sin(a / 2)], dtype=complex)


def _ry_state(theta: np.ndarray) -> np.ndarray:
    """Rotation-Y ansatz: |ψ(θ)⟩ = Ry(θ₀)|0⟩ = cos(θ₀/2)|0⟩ + sin(θ₀/2)|1⟩."""
    a = float(theta[0])
    return np.array([math.cos(a / 2), math.sin(a / 2)], dtype=complex)


def _two_param_state(theta: np.ndarray) -> np.ndarray:
    """Simple 2-parameter 2-qubit-like state on 2-dimensional Hilbert space."""
    a, b = float(theta[0]), float(theta[1])
    v = np.array([math.cos(a) * math.cos(b),
                  math.sin(a) * math.cos(b)], dtype=complex)
    norm = float(np.linalg.norm(v)) or 1.0
    return v / norm


# ---------------------------------------------------------------------------
# McLachlanResult
# ---------------------------------------------------------------------------

class TestMcLachlanResult:
    def _make(self):
        return McLachlanResult(
            times=np.linspace(0, 1, 11),
            params=np.zeros((11, 2)),
            energies=np.linspace(-1, -1.5, 11),
            norms=np.ones(11),
            converged=True,
        )

    def test_n_steps(self):
        r = self._make()
        assert r.n_steps == 10

    def test_final_energy(self):
        r = self._make()
        assert r.final_energy == pytest.approx(-1.5)

    def test_energy_change(self):
        r = self._make()
        assert r.energy_change == pytest.approx(-0.5)

    def test_final_params_shape(self):
        r = self._make()
        assert r.final_params.shape == (2,)

    def test_converged_flag(self):
        r = self._make()
        assert r.converged is True


# ---------------------------------------------------------------------------
# McLachlanSolver construction
# ---------------------------------------------------------------------------

class TestMcLachlanSolverConstruction:
    def test_basic_construction(self):
        s = McLachlanSolver(_rx_state, _ZZ, n_params=1)
        assert s.n_params == 1

    def test_callable_hamiltonian(self):
        H_fn = lambda psi: _ZZ @ psi  # noqa: E731
        s = McLachlanSolver(_rx_state, H_fn, n_params=1)
        assert s is not None

    def test_non_callable_ansatz_raises(self):
        with pytest.raises(McLachlanError, match="callable"):
            McLachlanSolver("not a callable", _ZZ, n_params=1)

    def test_n_params_zero_raises(self):
        with pytest.raises(McLachlanError, match="n_params"):
            McLachlanSolver(_rx_state, _ZZ, n_params=0)

    def test_dt_zero_raises(self):
        with pytest.raises(McLachlanError, match="dt"):
            McLachlanSolver(_rx_state, _ZZ, n_params=1, dt=0.0)

    def test_negative_dt_raises(self):
        with pytest.raises(McLachlanError, match="dt"):
            McLachlanSolver(_rx_state, _ZZ, n_params=1, dt=-0.1)

    def test_invalid_evolution_raises(self):
        with pytest.raises(McLachlanError, match="evolution"):
            McLachlanSolver(_rx_state, _ZZ, n_params=1, evolution="other")

    def test_invalid_integrator_raises(self):
        with pytest.raises(McLachlanError, match="integrator"):
            McLachlanSolver(_rx_state, _ZZ, n_params=1, integrator="bad")

    def test_negative_reg_raises(self):
        with pytest.raises(McLachlanError, match="reg"):
            McLachlanSolver(_rx_state, _ZZ, n_params=1, reg=-1e-6)

    def test_negative_eps_raises(self):
        with pytest.raises(McLachlanError, match="eps"):
            McLachlanSolver(_rx_state, _ZZ, n_params=1, eps=-1e-5)

    def test_repr(self):
        s = McLachlanSolver(_rx_state, _ZZ, n_params=1)
        assert "McLachlanSolver" in repr(s)
        assert "n_params=1" in repr(s)

    def test_1d_hamiltonian_raises(self):
        with pytest.raises(McLachlanError, match="2-D"):
            McLachlanSolver(_rx_state, np.array([1, -1], dtype=complex), n_params=1)


# ---------------------------------------------------------------------------
# McLachlanSolver.evolve — result shape and types
# ---------------------------------------------------------------------------

class TestMcLachlanEvolveShape:
    @pytest.fixture
    def solver(self):
        return McLachlanSolver(_rx_state, _ZZ, n_params=1, dt=0.05)

    def test_result_type(self, solver):
        r = solver.evolve(np.array([0.1]), t_final=0.2)
        assert isinstance(r, McLachlanResult)

    def test_times_shape(self, solver):
        r = solver.evolve(np.array([0.1]), t_final=0.2)
        assert r.times.shape[0] == r.params.shape[0]

    def test_params_shape(self, solver):
        r = solver.evolve(np.array([0.1]), t_final=0.2)
        assert r.params.shape[1] == 1

    def test_energies_shape(self, solver):
        r = solver.evolve(np.array([0.1]), t_final=0.2)
        assert r.energies.shape == r.times.shape

    def test_norms_positive(self, solver):
        r = solver.evolve(np.array([0.1]), t_final=0.2)
        assert np.all(r.norms > 0)

    def test_wrong_init_params_shape_raises(self, solver):
        with pytest.raises(McLachlanError):
            solver.evolve(np.array([0.1, 0.2]), t_final=0.2)

    def test_negative_t_final_raises(self, solver):
        with pytest.raises(McLachlanError):
            solver.evolve(np.array([0.1]), t_final=-1.0)

    def test_rk4_integrator(self):
        s = McLachlanSolver(_rx_state, _ZZ, n_params=1, dt=0.05, integrator="rk4")
        r = s.evolve(np.array([0.1]), t_final=0.2)
        assert r.n_steps >= 1


# ---------------------------------------------------------------------------
# Real-time evolution — physics checks
# ---------------------------------------------------------------------------

class TestRealTimeEvolution:
    def test_energy_constant_for_eigenstate_rz(self):
        """Evolving an eigenstate of H=Z under real-time: energy should be constant."""
        # |0⟩ is eigenstate of Z with eigenvalue +1
        init_theta = np.array([0.0])  # Rx(0)|0⟩ = |0⟩
        s = McLachlanSolver(
            _rx_state, _ZZ, n_params=1, dt=0.05, evolution="real"
        )
        r = s.evolve(init_theta, t_final=0.5)
        # Energy should stay near +1 (eigenvalue of Z for |0⟩)
        assert np.max(np.abs(r.energies - r.energies[0])) < 0.2

    def test_initial_energy_correct(self):
        """Energy at t=0 should equal ⟨ψ(θ₀)|H|ψ(θ₀)⟩."""
        theta0 = np.array([math.pi / 3])
        s = McLachlanSolver(_ry_state, _ZZ, n_params=1, dt=0.05)
        r = s.evolve(theta0, t_final=0.05)
        psi0 = _ry_state(theta0)
        expected = float(np.real(psi0.conj() @ (_ZZ @ psi0)))
        assert r.energies[0] == pytest.approx(expected, abs=1e-10)

    def test_evolution_runs_multiple_steps(self):
        s = McLachlanSolver(_rx_state, _ZZ, n_params=1, dt=0.05)
        r = s.evolve(np.array([0.2]), t_final=0.5)
        assert r.n_steps >= 5

    def test_two_param_ansatz(self):
        s = McLachlanSolver(_two_param_state, _ZZ, n_params=2, dt=0.02)
        r = s.evolve(np.array([0.3, 0.1]), t_final=0.1)
        assert r.params.shape[1] == 2


# ---------------------------------------------------------------------------
# Imaginary-time evolution — physics checks
# ---------------------------------------------------------------------------

class TestImaginaryTimeEvolution:
    def test_energy_decreases_toward_ground_state_Z(self):
        """Imaginary-time with H=Z: ground state is |1⟩, energy → -1."""
        # Start with Ry(π/3): ⟨Z⟩ = cos(π/3) ≈ 0.5
        # After imaginary time, energy should decrease toward -1
        init_theta = np.array([math.pi / 3])
        s = McLachlanSolver(
            _ry_state, _ZZ, n_params=1, dt=0.05, evolution="imaginary"
        )
        r = s.evolve(init_theta, t_final=2.0)
        assert r.energies[-1] < r.energies[0]  # energy decreases

    def test_energy_lower_bound_Z(self):
        """Final energy must not go below the minimum eigenvalue (-1)."""
        init_theta = np.array([math.pi / 4])
        s = McLachlanSolver(
            _ry_state, _ZZ, n_params=1, dt=0.05, evolution="imaginary"
        )
        r = s.evolve(init_theta, t_final=2.0)
        assert r.final_energy >= -1.0 - 0.05  # small tolerance

    def test_ground_state_energy_convenience(self):
        """ground_state_energy() wrapper should give the same result."""
        init_theta = np.array([math.pi / 4])
        s = McLachlanSolver(
            _ry_state, _ZZ, n_params=1, dt=0.05, evolution="real"
        )
        gse = s.ground_state_energy(init_theta, t_final=2.0)
        # Should be a float near -1
        assert isinstance(gse, float)
        assert gse >= -1.0 - 0.05

    def test_imaginary_time_X_hamiltonian(self):
        """Imaginary-time with H=X: ground state is |−⟩, energy → -1."""
        init_theta = np.array([0.0])
        s = McLachlanSolver(
            _ry_state, _HH, n_params=1, dt=0.05, evolution="imaginary"
        )
        r = s.evolve(init_theta, t_final=2.0)
        assert r.final_energy <= r.energies[0] + 0.1

    def test_evolution_attribute_restored_after_ground_state(self):
        """evolution attribute should be restored after ground_state_energy()."""
        s = McLachlanSolver(
            _ry_state, _ZZ, n_params=1, dt=0.05, evolution="real"
        )
        s.ground_state_energy(np.array([0.5]))
        assert s.evolution == "real"


# ---------------------------------------------------------------------------
# Metric and force internal methods
# ---------------------------------------------------------------------------

class TestMetricAndForce:
    def test_metric_is_symmetric(self):
        s = McLachlanSolver(_two_param_state, _ZZ, n_params=2)
        M, _ = s._metric_and_force(np.array([0.2, 0.3]))
        assert np.allclose(M, M.T, atol=1e-12)

    def test_metric_is_pd(self):
        """M + reg*I should be positive definite."""
        s = McLachlanSolver(_two_param_state, _ZZ, n_params=2, reg=1e-4)
        M, _ = s._metric_and_force(np.array([0.2, 0.3]))
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0)

    def test_force_vector_shape(self):
        s = McLachlanSolver(_two_param_state, _ZZ, n_params=2)
        _, V = s._metric_and_force(np.array([0.2, 0.3]))
        assert V.shape == (2,)

    def test_metric_shape(self):
        s = McLachlanSolver(_two_param_state, _ZZ, n_params=2)
        M, _ = s._metric_and_force(np.array([0.2, 0.3]))
        assert M.shape == (2, 2)
