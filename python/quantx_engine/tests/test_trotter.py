"""Tests for trotter.py — M101 Suzuki-Trotter Hamiltonian simulation."""
from __future__ import annotations

import math
import numpy as np
import pytest

from quantx_engine.trotter import (
    TrotterError,
    TrotterTerm,
    TrotterResult,
    TrotterSimulator,
    heisenberg_trotter,
    ising_trotter,
    _pauli_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_unitary(U: np.ndarray, atol: float = 1e-9) -> bool:
    n = U.shape[0]
    return (
        np.allclose(U.conj().T @ U, np.eye(n), atol=atol) and
        np.allclose(U @ U.conj().T, np.eye(n), atol=atol)
    )


def _exact_evolve(H: np.ndarray, psi0: np.ndarray, t: float) -> np.ndarray:
    """Compute exp(-iHt)|ψ₀⟩ via scipy.linalg.expm (skips if scipy missing)."""
    scipy_linalg = pytest.importorskip("scipy.linalg",
                                        reason="scipy required for exact evolution")
    return scipy_linalg.expm(-1j * H * t) @ psi0


# ─────────────────────────────────────────────────────────────────────────────
# _pauli_matrix
# ─────────────────────────────────────────────────────────────────────────────

class TestPauliMatrix:
    def test_single_I(self):
        np.testing.assert_allclose(_pauli_matrix("I"), np.eye(2))

    def test_single_X(self):
        np.testing.assert_allclose(_pauli_matrix("X"), [[0, 1], [1, 0]])

    def test_single_Y(self):
        np.testing.assert_allclose(_pauli_matrix("Y"), [[0, -1j], [1j, 0]])

    def test_single_Z(self):
        np.testing.assert_allclose(_pauli_matrix("Z"), [[1, 0], [0, -1]])

    def test_ZZ_diagonal(self):
        np.testing.assert_allclose(np.diag(_pauli_matrix("ZZ")),
                                   [1, -1, -1, 1])

    def test_three_qubit_shape(self):
        assert _pauli_matrix("XYZ").shape == (8, 8)

    def test_pauli_squares_to_identity(self):
        for ps in ("X", "Y", "Z", "XX", "YY", "ZZ", "XYZ"):
            P = _pauli_matrix(ps)
            np.testing.assert_allclose(P @ P, np.eye(P.shape[0]), atol=1e-12)

    def test_empty_string_raises(self):
        with pytest.raises(TrotterError):
            _pauli_matrix("")

    def test_invalid_char_raises(self):
        with pytest.raises(TrotterError, match="Invalid Pauli"):
            _pauli_matrix("A")


# ─────────────────────────────────────────────────────────────────────────────
# TrotterTerm
# ─────────────────────────────────────────────────────────────────────────────

class TestTrotterTerm:
    def test_basic_construction(self):
        t = TrotterTerm(1.0, "ZZ")
        assert t.coefficient == 1.0
        assert t.pauli_string == "ZZ"

    def test_lowercase_normalised(self):
        t = TrotterTerm(0.5, "xx")
        assert t.pauli_string == "XX"

    def test_invalid_char_raises(self):
        with pytest.raises(TrotterError):
            TrotterTerm(1.0, "AB")

    def test_zero_coefficient_ok(self):
        assert TrotterTerm(0.0, "ZZ").coefficient == 0.0

    def test_negative_coefficient_ok(self):
        assert TrotterTerm(-1.5, "ZZ").coefficient == -1.5


# ─────────────────────────────────────────────────────────────────────────────
# TrotterSimulator — construction
# ─────────────────────────────────────────────────────────────────────────────

class TestTrotterSimulatorConstruction:
    def test_single_qubit(self):
        sim = TrotterSimulator(1, [TrotterTerm(1.0, "Z")])
        assert sim.n_qubits == 1

    def test_two_qubit(self):
        sim = TrotterSimulator(2, [TrotterTerm(0.5, "ZZ")])
        assert sim.n_qubits == 2

    def test_zero_qubits_raises(self):
        with pytest.raises(TrotterError, match="n_qubits"):
            TrotterSimulator(0, [TrotterTerm(1.0, "Z")])

    def test_empty_terms_raises(self):
        with pytest.raises(TrotterError, match="at least one"):
            TrotterSimulator(1, [])

    def test_wrong_pauli_length_raises(self):
        with pytest.raises(TrotterError, match="length"):
            TrotterSimulator(2, [TrotterTerm(1.0, "Z")])

    def test_custom_initial_state(self):
        psi = np.array([0, 1], dtype=complex)
        sim = TrotterSimulator(1, [TrotterTerm(1.0, "Z")], initial_state=psi)
        np.testing.assert_allclose(sim.initial_state, psi)

    def test_initial_state_auto_normalised(self):
        psi = np.array([2.0, 0.0], dtype=complex)
        sim = TrotterSimulator(1, [TrotterTerm(1.0, "Z")], initial_state=psi)
        assert abs(np.linalg.norm(sim.initial_state) - 1.0) < 1e-12

    def test_zero_initial_state_raises(self):
        with pytest.raises(TrotterError, match="zero norm"):
            TrotterSimulator(1, [TrotterTerm(1.0, "Z")],
                             initial_state=np.zeros(2, dtype=complex))

    def test_wrong_initial_state_length_raises(self):
        with pytest.raises(TrotterError):
            TrotterSimulator(1, [TrotterTerm(1.0, "Z")],
                             initial_state=np.array([1, 0, 0], dtype=complex))

    def test_hamiltonian_hermitian(self):
        sim = TrotterSimulator(2, [
            TrotterTerm(0.5, "XX"), TrotterTerm(0.5, "YY"), TrotterTerm(0.5, "ZZ"),
        ])
        H = sim.hamiltonian
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_hamiltonian_shape(self):
        sim = TrotterSimulator(3, [TrotterTerm(1.0, "ZZZ")])
        assert sim.hamiltonian.shape == (8, 8)


# ─────────────────────────────────────────────────────────────────────────────
# TrotterResult
# ─────────────────────────────────────────────────────────────────────────────

class TestTrotterResult:
    def test_n_qubits_property(self):
        sim = TrotterSimulator(3, [TrotterTerm(1.0, "ZZZ")])
        r = sim.run(t=1.0, n_steps=10)
        assert r.n_qubits == 3

    def test_probs_shape(self):
        sim = TrotterSimulator(2, [TrotterTerm(1.0, "ZZ")])
        r = sim.run(t=1.0, n_steps=10)
        assert r.probs.shape == (4,)

    def test_metadata_keys(self):
        sim = TrotterSimulator(2, [TrotterTerm(1.0, "ZZ")])
        r = sim.run(t=1.0, n_steps=10)
        assert {"n_qubits", "n_terms", "dt"} <= r.metadata.keys()

    def test_no_exact_by_default(self):
        sim = TrotterSimulator(1, [TrotterTerm(1.0, "Z")])
        r = sim.run(t=1.0, n_steps=10)
        assert r.trotter_error is None


# ─────────────────────────────────────────────────────────────────────────────
# Single-term Hamiltonians (Trotter is exact — no splitting error)
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleTermExact:
    def test_x_rotation_probs_order1(self):
        """H = X: |0⟩ → probs = [cos²t, sin²t]."""
        t = math.pi / 4
        sim = TrotterSimulator(1, [TrotterTerm(1.0, "X")])
        r = sim.run(t=t, n_steps=1, order=1)
        np.testing.assert_allclose(
            r.probs, [math.cos(t) ** 2, math.sin(t) ** 2], atol=1e-12
        )

    def test_x_rotation_probs_order2(self):
        t = math.pi / 3
        sim = TrotterSimulator(1, [TrotterTerm(1.0, "X")])
        r = sim.run(t=t, n_steps=5, order=2)
        np.testing.assert_allclose(
            r.probs, [math.cos(t) ** 2, math.sin(t) ** 2], atol=1e-12
        )

    def test_z_eigenstate_unchanged(self):
        """H = Z, initial |0⟩: exp(-itZ)|0⟩ = e^{-it}|0⟩ → probs[0] = 1."""
        sim = TrotterSimulator(1, [TrotterTerm(1.0, "Z")])
        r = sim.run(t=2.345, n_steps=1, order=1)
        np.testing.assert_allclose(r.probs, [1.0, 0.0], atol=1e-12)

    def test_zz_two_qubit_normalised(self):
        sim = TrotterSimulator(2, [TrotterTerm(0.5, "ZZ")])
        r = sim.run(t=1.0, n_steps=1, order=2)
        assert abs(r.probs.sum() - 1.0) < 1e-12

    def test_single_term_all_orders_agree(self):
        """For a single term, orders 1/2/4 must produce identical output."""
        sim = TrotterSimulator(2, [TrotterTerm(0.7, "XX")])
        t = 0.83
        states = [sim.run(t=t, n_steps=5, order=o).state for o in (1, 2, 4)]
        np.testing.assert_allclose(states[0], states[1], atol=1e-12)
        np.testing.assert_allclose(states[0], states[2], atol=1e-12)

    def test_single_term_exact_vs_scipy(self):
        """Single-term Trotter must match scipy.linalg.expm exactly."""
        sim = TrotterSimulator(2, [TrotterTerm(0.5, "XX")])
        r = sim.run(t=1.0, n_steps=1, order=2)
        exact = _exact_evolve(sim.hamiltonian, sim.initial_state, 1.0)
        np.testing.assert_allclose(r.state, exact, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Order 1
# ─────────────────────────────────────────────────────────────────────────────

class TestOrder1:
    def _sim(self):
        return TrotterSimulator(2, [
            TrotterTerm(0.5, "XX"), TrotterTerm(0.5, "YY"),
        ])

    def test_probs_non_negative(self):
        r = self._sim().run(t=1.0, n_steps=50, order=1)
        assert np.all(r.probs >= -1e-12)

    def test_probs_sum_to_one(self):
        r = self._sim().run(t=1.0, n_steps=50, order=1)
        assert abs(r.probs.sum() - 1.0) < 1e-12

    def test_state_normalised(self):
        r = self._sim().run(t=2.0, n_steps=100, order=1)
        assert abs(np.linalg.norm(r.state) - 1.0) < 1e-10

    def test_step_matrix_unitary(self):
        sim = self._sim()
        assert _is_unitary(sim._step1(0.1))

    def test_invalid_order_raises(self):
        with pytest.raises(TrotterError, match="order"):
            self._sim().run(t=1.0, n_steps=10, order=3)

    def test_zero_steps_raises(self):
        with pytest.raises(TrotterError, match="n_steps"):
            self._sim().run(t=1.0, n_steps=0, order=1)

    def test_less_accurate_than_order2(self):
        """Order-1 error should be larger than order-2 for the same n_steps."""
        sim = self._sim()
        H = sim.hamiltonian
        psi0 = sim.initial_state
        t = 1.0
        exact = _exact_evolve(H, psi0, t)
        e1 = float(np.linalg.norm(sim.run(t=t, n_steps=20, order=1).state - exact))
        e2 = float(np.linalg.norm(sim.run(t=t, n_steps=20, order=2).state - exact))
        assert e1 > e2


# ─────────────────────────────────────────────────────────────────────────────
# Order 2
# ─────────────────────────────────────────────────────────────────────────────

class TestOrder2:
    def _sim(self):
        return TrotterSimulator(2, [
            TrotterTerm(1.0, "XX"), TrotterTerm(1.0, "YY"),
        ])

    def test_probs_sum_to_one(self):
        r = self._sim().run(t=1.0, n_steps=50, order=2)
        assert abs(r.probs.sum() - 1.0) < 1e-12

    def test_state_normalised(self):
        r = self._sim().run(t=2.0, n_steps=50, order=2)
        assert abs(np.linalg.norm(r.state) - 1.0) < 1e-10

    def test_step_matrix_unitary(self):
        assert _is_unitary(self._sim()._step2(0.1))

    def test_error_decreases_with_more_steps(self):
        sim = self._sim()
        H, psi0 = sim.hamiltonian, sim.initial_state
        exact = _exact_evolve(H, psi0, 1.0)
        e_coarse = float(np.linalg.norm(sim.run(1.0, 10, 2).state - exact))
        e_fine   = float(np.linalg.norm(sim.run(1.0, 100, 2).state - exact))
        assert e_fine < e_coarse

    def test_error_scales_quadratically(self):
        """Order-2 error ∝ 1/n² → 10× more steps → ~100× smaller error."""
        sim = self._sim()
        H, psi0 = sim.hamiltonian, sim.initial_state
        exact = _exact_evolve(H, psi0, 1.0)
        e10  = float(np.linalg.norm(sim.run(1.0,  10, 2).state - exact))
        e100 = float(np.linalg.norm(sim.run(1.0, 100, 2).state - exact))
        if e100 > 1e-15:
            ratio = e10 / e100
            assert 20 < ratio < 500  # loosely 100, allow numerical variation

    def test_result_metadata(self):
        r = self._sim().run(t=1.0, n_steps=20, order=2)
        assert r.order == 2 and r.n_steps == 20


# ─────────────────────────────────────────────────────────────────────────────
# Order 4
# ─────────────────────────────────────────────────────────────────────────────

class TestOrder4:
    def _sim(self):
        return TrotterSimulator(2, [
            TrotterTerm(1.0, "XX"),
            TrotterTerm(1.0, "YY"),
            TrotterTerm(1.0, "ZZ"),
        ])

    def test_probs_sum_to_one(self):
        r = self._sim().run(t=1.0, n_steps=10, order=4)
        assert abs(r.probs.sum() - 1.0) < 1e-12

    def test_state_normalised(self):
        r = self._sim().run(t=1.0, n_steps=10, order=4)
        assert abs(np.linalg.norm(r.state) - 1.0) < 1e-10

    def test_step_matrix_unitary(self):
        assert _is_unitary(self._sim()._step4(0.1), atol=1e-8)

    def test_more_accurate_than_order2_per_cost(self):
        """Order-4 with fewer steps beats order-2 with more."""
        sim = self._sim()
        H, psi0 = sim.hamiltonian, sim.initial_state
        exact = _exact_evolve(H, psi0, 1.0)
        e2 = float(np.linalg.norm(sim.run(1.0, 30, 2).state - exact))
        e4 = float(np.linalg.norm(sim.run(1.0, 10, 4).state - exact))
        assert e4 < e2


# ─────────────────────────────────────────────────────────────────────────────
# Physics: free precession
# ─────────────────────────────────────────────────────────────────────────────

class TestFreePrecess:
    def test_z_precession_plus_state(self):
        """H = Z, |+⟩: probs stay [0.5, 0.5] for all t (Z acts as dephasing)."""
        psi_plus = np.array([1, 1], dtype=complex) / math.sqrt(2)
        sim = TrotterSimulator(1, [TrotterTerm(1.0, "Z")], initial_state=psi_plus)
        r = sim.run(t=1.234, n_steps=1, order=2)
        np.testing.assert_allclose(r.probs, [0.5, 0.5], atol=1e-12)

    def test_x_pi_over_2_pulse(self):
        """H = X, t = π/2: |0⟩ → |1⟩ with prob 1."""
        sim = TrotterSimulator(1, [TrotterTerm(1.0, "X")])
        r = sim.run(t=math.pi / 2, n_steps=1, order=1)
        np.testing.assert_allclose(r.probs, [0.0, 1.0], atol=1e-12)

    def test_backward_evolution_inverts(self):
        """Evolve forward then backward → recover initial state."""
        sim = TrotterSimulator(2, [
            TrotterTerm(0.5, "XX"), TrotterTerm(0.5, "YY"),
        ])
        r_fwd = sim.run(t=1.0, n_steps=100, order=2)
        sim2 = TrotterSimulator(2, [
            TrotterTerm(0.5, "XX"), TrotterTerm(0.5, "YY"),
        ], initial_state=r_fwd.state)
        r_back = sim2.run(t=-1.0, n_steps=100, order=2)
        np.testing.assert_allclose(r_back.probs, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_zero_coefficient_no_evolution(self):
        """H = 0·ZZ: state unchanged."""
        sim = TrotterSimulator(2, [TrotterTerm(0.0, "ZZ")])
        r = sim.run(t=10.0, n_steps=50, order=2)
        np.testing.assert_allclose(r.probs, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Physics: Heisenberg dimer eigenstate
# ─────────────────────────────────────────────────────────────────────────────

class TestHeisenbergDimerPhysics:
    def _sim(self, initial_state=None):
        return TrotterSimulator(2, [
            TrotterTerm(1.0, "XX"),
            TrotterTerm(1.0, "YY"),
            TrotterTerm(1.0, "ZZ"),
        ], initial_state=initial_state)

    def test_probs_normalised_generic(self):
        r = self._sim().run(t=1.0, n_steps=100, order=2)
        assert abs(r.probs.sum() - 1.0) < 1e-12

    def test_singlet_is_eigenstate(self):
        """Singlet (|01⟩ − |10⟩)/√2 is an eigenstate of H_Heisenberg.
        Probs must remain [0, 0.5, 0.5, 0] for any t."""
        singlet = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
        r = self._sim(initial_state=singlet).run(t=1.234, n_steps=200, order=2)
        np.testing.assert_allclose(r.probs, [0.0, 0.5, 0.5, 0.0], atol=1e-3)

    def test_non_eigenstate_evolves(self):
        """|01⟩ is NOT an eigenstate of H_Heisenberg (it mixes triplet/singlet).
        Probs must change from the initial [0,1,0,0]."""
        psi01 = np.array([0, 1, 0, 0], dtype=complex)
        r = self._sim(initial_state=psi01).run(t=math.pi / 4, n_steps=200, order=2)
        # After t=π/4, prob should have transferred to |10⟩
        assert r.probs[1] < 0.5  # substantial population left |01⟩


# ─────────────────────────────────────────────────────────────────────────────
# Error bound
# ─────────────────────────────────────────────────────────────────────────────

class TestTrotterErrorBound:
    # [ZI, XI] ≠ 0 since both act on qubit 0 with [Z,X] = 2iY ≠ 0
    def _noncomm_terms(self):
        return [TrotterTerm(1.0, "ZI"), TrotterTerm(1.0, "XI")]

    # [XX, YY] = (XY)⊗(XY) - (YX)⊗(YX) = (iZ)⊗(iZ) - (-iZ)⊗(-iZ) = 0
    def _comm_terms(self):
        return [TrotterTerm(0.5, "XX"), TrotterTerm(0.5, "YY")]

    def test_order1_non_negative(self):
        assert TrotterSimulator.trotter_error_bound(2, self._noncomm_terms(), 1.0, 50, 1) >= 0

    def test_order2_non_negative(self):
        assert TrotterSimulator.trotter_error_bound(2, self._noncomm_terms(), 1.0, 50, 2) >= 0

    def test_order4_non_negative(self):
        assert TrotterSimulator.trotter_error_bound(2, self._noncomm_terms(), 1.0, 50, 4) >= 0

    def test_order1_decreases_with_steps(self):
        terms = self._noncomm_terms()
        b10  = TrotterSimulator.trotter_error_bound(2, terms, 1.0, 10,  1)
        b100 = TrotterSimulator.trotter_error_bound(2, terms, 1.0, 100, 1)
        assert b10 > 0 and b100 < b10

    def test_order2_decreases_with_steps(self):
        terms = self._noncomm_terms()
        b10  = TrotterSimulator.trotter_error_bound(2, terms, 1.0, 10,  2)
        b100 = TrotterSimulator.trotter_error_bound(2, terms, 1.0, 100, 2)
        assert b10 > 0 and b100 < b10

    def test_order1_linear_scaling(self):
        """Doubling n_steps should halve the order-1 bound."""
        terms = self._noncomm_terms()
        b10 = TrotterSimulator.trotter_error_bound(2, terms, 1.0, 10, 1)
        b20 = TrotterSimulator.trotter_error_bound(2, terms, 1.0, 20, 1)
        if b20 > 0:
            assert 1.8 < b10 / b20 < 2.2

    def test_order2_quadratic_scaling(self):
        """10× more steps → ~100× smaller order-2 bound."""
        terms = self._noncomm_terms()
        b10  = TrotterSimulator.trotter_error_bound(2, terms, 1.0, 10,  2)
        b100 = TrotterSimulator.trotter_error_bound(2, terms, 1.0, 100, 2)
        if b100 > 0:
            assert 80 < b10 / b100 < 120

    def test_commuting_terms_zero_order1(self):
        """Commuting terms have zero commutator → zero order-1 bound."""
        terms = [TrotterTerm(1.0, "ZZ"), TrotterTerm(0.5, "IZ")]
        b = TrotterSimulator.trotter_error_bound(2, terms, 1.0, 10, 1)
        assert b == pytest.approx(0.0, abs=1e-12)

    def test_invalid_order_raises(self):
        with pytest.raises(TrotterError):
            TrotterSimulator.trotter_error_bound(2, self._noncomm_terms(), 1.0, 10, 3)


# ─────────────────────────────────────────────────────────────────────────────
# optimal_steps
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimalSteps:
    # Use non-commuting terms so the error bound is non-zero
    def _terms(self):
        return [TrotterTerm(1.0, "ZI"), TrotterTerm(1.0, "XI")]

    def test_returns_positive_int(self):
        n = TrotterSimulator.optimal_steps(2, self._terms(), 1.0, 0.01, 2)
        assert isinstance(n, int) and n >= 1

    def test_bound_satisfied(self):
        terms = self._terms()
        n = TrotterSimulator.optimal_steps(2, terms, 1.0, 0.01, 2)
        b = TrotterSimulator.trotter_error_bound(2, terms, 1.0, n, 2)
        assert b <= 0.01

    def test_tighter_target_needs_more_steps(self):
        terms = self._terms()
        n_loose = TrotterSimulator.optimal_steps(2, terms, 1.0, 0.1,   2)
        n_tight = TrotterSimulator.optimal_steps(2, terms, 1.0, 0.001, 2)
        assert n_tight >= n_loose

    def test_zero_target_raises(self):
        with pytest.raises(TrotterError, match="target_error"):
            TrotterSimulator.optimal_steps(2, self._terms(), 1.0, 0.0, 2)

    def test_commuting_terms_return_one(self):
        """Commuting H has zero error bound → minimum 1 step."""
        terms = [TrotterTerm(1.0, "ZZ"), TrotterTerm(0.5, "IZ")]
        assert TrotterSimulator.optimal_steps(2, terms, 1.0, 1e-6, 1) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Factory: heisenberg_trotter
# ─────────────────────────────────────────────────────────────────────────────

class TestHeisenbergTrotter:
    def test_constructs_n2(self):
        assert heisenberg_trotter(n=2).n_qubits == 2

    def test_constructs_n4(self):
        assert heisenberg_trotter(n=4).n_qubits == 4

    def test_n1_raises(self):
        with pytest.raises(TrotterError):
            heisenberg_trotter(n=1)

    def test_run_normalised(self):
        r = heisenberg_trotter(n=3, J=1.0, hz=0.5).run(t=1.0, n_steps=50, order=2)
        assert abs(r.probs.sum() - 1.0) < 1e-12

    def test_zero_coupling_no_evolution(self):
        """J=0, hz=0 → H=0 → |000…⟩ unchanged."""
        sim = heisenberg_trotter(n=2, J=0.0, hz=0.0)
        r = sim.run(t=5.0, n_steps=50, order=2)
        np.testing.assert_allclose(r.probs, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_with_longitudinal_field(self):
        r = heisenberg_trotter(n=2, J=1.0, hz=0.5).run(t=1.0, n_steps=50)
        assert r.probs.sum() == pytest.approx(1.0, abs=1e-12)

    def test_custom_initial_state(self):
        psi = np.zeros(4, dtype=complex); psi[1] = 1.0
        r = heisenberg_trotter(n=2, J=1.0, initial_state=psi).run(t=1.0, n_steps=50)
        assert abs(r.probs.sum() - 1.0) < 1e-12

    def test_converges_to_exact(self):
        sim = heisenberg_trotter(n=2, J=1.0)
        H, psi0 = sim.hamiltonian, sim.initial_state
        exact = _exact_evolve(H, psi0, 1.0)
        e_c = float(np.linalg.norm(sim.run(1.0,  10, 2).state - exact))
        e_f = float(np.linalg.norm(sim.run(1.0, 200, 2).state - exact))
        assert e_f < e_c and e_f < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Factory: ising_trotter
# ─────────────────────────────────────────────────────────────────────────────

class TestIsingTrotter:
    def test_constructs_n2(self):
        assert ising_trotter(n=2).n_qubits == 2

    def test_constructs_n4(self):
        assert ising_trotter(n=4).n_qubits == 4

    def test_n1_raises(self):
        with pytest.raises(TrotterError):
            ising_trotter(n=1)

    def test_run_normalised(self):
        r = ising_trotter(n=3, J=1.0, h=0.5).run(t=1.0, n_steps=50, order=2)
        assert abs(r.probs.sum() - 1.0) < 1e-12

    def test_zero_field_zz_eigenstate(self):
        """|000⟩ is ZZ eigenstate → prob[0] = 1 for h=0."""
        r = ising_trotter(n=3, J=1.0, h=0.0).run(t=2.0, n_steps=20, order=2)
        np.testing.assert_allclose(r.probs[0], 1.0, atol=1e-12)

    def test_converges_to_exact(self):
        sim = ising_trotter(n=2, J=1.0, h=1.0)
        H, psi0 = sim.hamiltonian, sim.initial_state
        exact = _exact_evolve(H, psi0, 1.0)
        e_c = float(np.linalg.norm(sim.run(1.0,  10, 2).state - exact))
        e_f = float(np.linalg.norm(sim.run(1.0, 200, 2).state - exact))
        assert e_f < e_c and e_f < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────

class TestExports:
    def test_importable_from_module(self):
        from quantx_engine.trotter import (
            TrotterError, TrotterTerm, TrotterResult,
            TrotterSimulator, heisenberg_trotter, ising_trotter,
        )
        assert TrotterSimulator is not None

    def test_exported_from_package(self):
        from quantx_engine import (
            TrotterError, TrotterTerm, TrotterResult,
            TrotterSimulator, heisenberg_trotter, ising_trotter,
        )
        assert TrotterSimulator is not None
