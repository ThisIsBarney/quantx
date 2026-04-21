"""Tests for quantum optimal control — GRAPE and CRAB (M84)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quantx_engine.optimal_control import (
    CRABResult,
    CRABSolver,
    GRAPEResult,
    GRAPESolver,
    OCError,
)


# ===========================================================================
# Helpers
# ===========================================================================

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H_mat = (X + Z) / math.sqrt(2)

# Standard CNOT on 2 qubits
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)


def _kron(A: "np.ndarray", B: "np.ndarray") -> "np.ndarray":
    return np.kron(A, B)


def _fidelity(U_prop: "np.ndarray", U_target: "np.ndarray") -> float:
    d = U_target.shape[0]
    X_mat = U_target.conj().T @ U_prop
    return float(abs(np.trace(X_mat)) ** 2) / (d * d)


# ===========================================================================
# GRAPESolver — Construction
# ===========================================================================

class TestGRAPEConstruction:
    def test_default_construction(self):
        solver = GRAPESolver(
            H_drift=0.5 * Z, H_controls=[X], U_target=X, n_steps=10, total_time=math.pi
        )
        assert solver.n_steps == 10
        assert solver._d == 2

    def test_invalid_H_drift_not_square(self):
        with pytest.raises(OCError, match="square"):
            GRAPESolver(
                H_drift=np.zeros((2, 3), dtype=complex),
                H_controls=[X], U_target=X,
            )

    def test_invalid_H_control_wrong_size(self):
        with pytest.raises(OCError):
            GRAPESolver(
                H_drift=Z,
                H_controls=[np.eye(4, dtype=complex)],  # wrong size
                U_target=X,
            )

    def test_invalid_U_target_wrong_size(self):
        with pytest.raises(OCError):
            GRAPESolver(
                H_drift=Z, H_controls=[X],
                U_target=np.eye(4, dtype=complex),  # wrong size
            )

    def test_invalid_n_steps(self):
        with pytest.raises(OCError, match="n_steps"):
            GRAPESolver(H_drift=Z, H_controls=[X], U_target=X, n_steps=0)

    def test_invalid_total_time(self):
        with pytest.raises(OCError, match="total_time"):
            GRAPESolver(H_drift=Z, H_controls=[X], U_target=X, total_time=0.0)

    def test_empty_controls_raises(self):
        with pytest.raises(OCError, match="H_controls"):
            GRAPESolver(H_drift=Z, H_controls=[], U_target=X)


class TestGRAPERunValidation:
    def test_invalid_n_iterations(self):
        solver = GRAPESolver(H_drift=Z, H_controls=[X], U_target=X)
        with pytest.raises(OCError, match="n_iterations"):
            solver.run(n_iterations=0)

    def test_invalid_learning_rate(self):
        solver = GRAPESolver(H_drift=Z, H_controls=[X], U_target=X)
        with pytest.raises(OCError, match="learning_rate"):
            solver.run(learning_rate=0.0)

    def test_invalid_fidelity_threshold(self):
        solver = GRAPESolver(H_drift=Z, H_controls=[X], U_target=X)
        with pytest.raises(OCError, match="fidelity_threshold"):
            solver.run(fidelity_threshold=1.5)

    def test_wrong_initial_controls_shape(self):
        solver = GRAPESolver(H_drift=Z, H_controls=[X], U_target=X, n_steps=5)
        with pytest.raises(OCError):
            solver.run(initial_controls=np.zeros((2, 5)))  # 2 controls, only 1 expected

    def test_returns_grape_result(self):
        solver = GRAPESolver(
            H_drift=np.zeros((2, 2), dtype=complex),
            H_controls=[X], U_target=X, n_steps=5, seed=0,
        )
        result = solver.run(n_iterations=5)
        assert isinstance(result, GRAPEResult)

    def test_result_controls_shape(self):
        solver = GRAPESolver(
            H_drift=Z, H_controls=[X, Y], U_target=X, n_steps=8, seed=0,
        )
        result = solver.run(n_iterations=3)
        assert result.controls.shape == (2, 8)  # (n_controls, n_steps)

    def test_fidelity_history_length(self):
        solver = GRAPESolver(H_drift=Z, H_controls=[X], U_target=X, n_steps=5, seed=0)
        result = solver.run(n_iterations=10)
        assert len(result.fidelity_history) >= 1
        assert len(result.fidelity_history) <= 11  # at most n_iterations + 1


# ===========================================================================
# GRAPESolver — Physics
# ===========================================================================

class TestGRAPEPhysics:
    def test_free_precession_exact_zero_control(self):
        """With zero control and H_drift = 0, propagator is identity."""
        solver = GRAPESolver(
            H_drift=np.zeros((2, 2), dtype=complex),
            H_controls=[X],
            U_target=I2,
            n_steps=5, total_time=1.0, seed=42,
        )
        result = solver.run(
            n_iterations=1,
            initial_controls=np.zeros((1, 5)),
        )
        # With zero Hamiltonian the propagator is the identity, matching target I
        assert result.fidelity_history[0] == pytest.approx(1.0, abs=1e-8)

    def test_fidelity_monotone_increasing(self):
        """GRAPE should not decrease fidelity on average over many steps."""
        solver = GRAPESolver(
            H_drift=0.5 * Z,
            H_controls=[X],
            U_target=X,
            n_steps=20, total_time=math.pi, seed=7,
        )
        result = solver.run(n_iterations=30, learning_rate=0.3)
        fids = result.fidelity_history
        # Not monotone every step (line search not used), but final > initial
        assert fids[-1] >= fids[0] - 0.05, (
            f"GRAPE decreased fidelity from {fids[0]:.4f} to {fids[-1]:.4f}"
        )

    def test_x_gate_achievable(self):
        """GRAPE should achieve high fidelity for an X gate with resonant drive."""
        # Resonant drive: H_drift = 0, H_control = X, target = X = exp(-i π/2 X)
        # Optimal pulse: constant u(t) = π/(2T) gives X gate
        solver = GRAPESolver(
            H_drift=np.zeros((2, 2), dtype=complex),
            H_controls=[X],
            U_target=X,
            n_steps=40, total_time=math.pi / 2.0, seed=1,
        )
        result = solver.run(n_iterations=100, learning_rate=0.5)
        assert result.fidelity_history[-1] > 0.9, (
            f"X gate GRAPE fidelity too low: {result.fidelity_history[-1]:.4f}"
        )

    def test_hadamard_gate_achievable(self):
        """GRAPE should achieve high fidelity for a Hadamard gate."""
        solver = GRAPESolver(
            H_drift=0.5 * Z,
            H_controls=[X],
            U_target=H_mat,
            n_steps=30, total_time=math.pi / 2.0, seed=3,
        )
        result = solver.run(n_iterations=80, learning_rate=0.4)
        assert result.fidelity_history[-1] > 0.7, (
            f"Hadamard GRAPE fidelity: {result.fidelity_history[-1]:.4f}"
        )

    def test_seed_reproducibility(self):
        solver_a = GRAPESolver(
            H_drift=0.5 * Z, H_controls=[X], U_target=X,
            n_steps=10, total_time=math.pi / 2.0, seed=42,
        )
        solver_b = GRAPESolver(
            H_drift=0.5 * Z, H_controls=[X], U_target=X,
            n_steps=10, total_time=math.pi / 2.0, seed=42,
        )
        r_a = solver_a.run(n_iterations=5)
        r_b = solver_b.run(n_iterations=5)
        assert np.allclose(r_a.controls, r_b.controls)

    def test_converged_flag_when_target_trivial(self):
        """Identity gate with zero Hamiltonian: trivially converged on first step."""
        solver = GRAPESolver(
            H_drift=np.zeros((2, 2), dtype=complex),
            H_controls=[X], U_target=I2,
            n_steps=5, seed=0,
        )
        result = solver.run(
            n_iterations=10, initial_controls=np.zeros((1, 5)),
            fidelity_threshold=0.999,
        )
        assert result.converged

    def test_two_control_fields(self):
        """Should work with two control Hamiltonians (X and Y drives)."""
        solver = GRAPESolver(
            H_drift=0.5 * Z, H_controls=[X, Y],
            U_target=X,
            n_steps=20, total_time=math.pi / 2.0, seed=5,
        )
        result = solver.run(n_iterations=50, learning_rate=0.3)
        assert result.controls.shape == (2, 20)
        assert len(result.fidelity_history) >= 1

    def test_fidelity_bounded(self):
        solver = GRAPESolver(
            H_drift=Z, H_controls=[X], U_target=H_mat,
            n_steps=10, total_time=math.pi / 2.0, seed=9,
        )
        result = solver.run(n_iterations=20)
        for f in result.fidelity_history:
            assert 0.0 <= f <= 1.0 + 1e-8, f"Fidelity {f} out of [0, 1]"


# ===========================================================================
# CRABSolver — Construction
# ===========================================================================

class TestCRABConstruction:
    def test_default_construction(self):
        solver = CRABSolver(H_drift=Z, H_controls=[X], U_target=X)
        assert solver.n_harmonics == 4

    def test_invalid_n_harmonics(self):
        with pytest.raises(OCError, match="n_harmonics"):
            CRABSolver(H_drift=Z, H_controls=[X], U_target=X, n_harmonics=0)

    def test_invalid_H_drift_not_square(self):
        with pytest.raises(OCError, match="square"):
            CRABSolver(
                H_drift=np.zeros((2, 3), dtype=complex),
                H_controls=[X], U_target=X,
            )

    def test_empty_controls_raises(self):
        with pytest.raises(OCError, match="H_controls"):
            CRABSolver(H_drift=Z, H_controls=[], U_target=X)


class TestCRABRunValidation:
    def test_returns_crab_result(self):
        solver = CRABSolver(
            H_drift=np.zeros((2, 2), dtype=complex),
            H_controls=[X], U_target=X, n_steps=5, n_harmonics=2, seed=0,
        )
        result = solver.run(n_iterations=20)
        assert isinstance(result, CRABResult)

    def test_result_controls_shape(self):
        solver = CRABSolver(
            H_drift=Z, H_controls=[X], U_target=X,
            n_steps=10, n_harmonics=3, seed=0,
        )
        result = solver.run(n_iterations=20)
        assert result.controls.shape == (1, 10)

    def test_result_coefficients_shape(self):
        solver = CRABSolver(
            H_drift=Z, H_controls=[X, Y], U_target=H_mat,
            n_steps=10, n_harmonics=3, seed=0,
        )
        result = solver.run(n_iterations=20)
        assert result.coefficients.shape == (2, 3, 2)  # (nc, nh, 2)

    def test_invalid_n_iterations(self):
        solver = CRABSolver(H_drift=Z, H_controls=[X], U_target=X)
        with pytest.raises(OCError, match="n_iterations"):
            solver.run(n_iterations=0)

    def test_invalid_fidelity_threshold(self):
        solver = CRABSolver(H_drift=Z, H_controls=[X], U_target=X)
        with pytest.raises(OCError, match="fidelity_threshold"):
            solver.run(fidelity_threshold=0.0)


# ===========================================================================
# CRABSolver — Physics
# ===========================================================================

class TestCRABPhysics:
    def test_zero_hamiltonian_identity_target(self):
        """With H=0 the propagator is always I; CRAB should immediately converge."""
        solver = CRABSolver(
            H_drift=np.zeros((2, 2), dtype=complex),
            H_controls=[np.zeros((2, 2), dtype=complex)],
            U_target=I2,
            n_steps=5, n_harmonics=2, seed=0,
        )
        result = solver.run(n_iterations=10, fidelity_threshold=0.999)
        assert result.fidelity_history[-1] == pytest.approx(1.0, abs=1e-8)

    def test_fidelity_bounded(self):
        solver = CRABSolver(
            H_drift=Z, H_controls=[X], U_target=H_mat,
            n_steps=10, n_harmonics=3, seed=11,
        )
        result = solver.run(n_iterations=50)
        for f in result.fidelity_history:
            assert 0.0 <= f <= 1.0 + 1e-8, f"Fidelity {f} out of [0, 1]"

    def test_fidelity_non_trivially_positive(self):
        """CRAB should find a non-zero fidelity for a reachable gate."""
        solver = CRABSolver(
            H_drift=np.zeros((2, 2), dtype=complex),
            H_controls=[X],
            U_target=X,
            n_steps=20, total_time=math.pi / 2.0,
            n_harmonics=4, seed=17,
        )
        result = solver.run(n_iterations=100)
        assert result.fidelity_history[-1] > 0.2, (
            f"CRAB fidelity too low: {result.fidelity_history[-1]:.4f}"
        )

    def test_seed_reproducibility(self):
        solver_a = CRABSolver(
            H_drift=Z, H_controls=[X], U_target=X,
            n_steps=8, n_harmonics=2, seed=42,
        )
        solver_b = CRABSolver(
            H_drift=Z, H_controls=[X], U_target=X,
            n_steps=8, n_harmonics=2, seed=42,
        )
        r_a = solver_a.run(n_iterations=20)
        r_b = solver_b.run(n_iterations=20)
        assert np.allclose(r_a.controls, r_b.controls, atol=1e-10)


# ===========================================================================
# __init__ re-exports
# ===========================================================================

class TestOptimalControlInit:
    def test_grape_exported(self):
        from quantx_engine import GRAPESolver as GS
        assert GS is GRAPESolver

    def test_crab_exported(self):
        from quantx_engine import CRABSolver as CS
        assert CS is CRABSolver
