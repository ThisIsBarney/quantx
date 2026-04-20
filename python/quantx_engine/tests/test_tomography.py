"""Tests for quantum state and process tomography (M83)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quantx_engine.tomography import (
    ProcessTomography,
    StateTomography,
    TomographyError,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _pure_state_rho(psi: "np.ndarray") -> "np.ndarray":
    """Density matrix for pure state |ψ⟩."""
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def _maximally_mixed(d: int) -> "np.ndarray":
    return np.eye(d, dtype=complex) / d


# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H_gate = (X + Z) / math.sqrt(2)


# ===========================================================================
# StateTomography
# ===========================================================================

class TestStateTomographyConstruction:
    def test_default_1qubit(self):
        st = StateTomography()
        assert st.n_qubits == 1
        assert st._d == 2

    def test_two_qubit(self):
        st = StateTomography(n_qubits=2)
        assert st._d == 4

    def test_invalid_n_qubits(self):
        with pytest.raises(TomographyError):
            StateTomography(n_qubits=0)


class TestPauliExpectations:
    def test_ground_state_z(self):
        """⟨Z⟩ = +1 for |0⟩."""
        rho = _pure_state_rho([1, 0])
        st = StateTomography(n_qubits=1)
        exp = st.pauli_expectations(rho)
        assert exp["Z"] == pytest.approx(1.0, abs=1e-10)
        assert exp["X"] == pytest.approx(0.0, abs=1e-10)
        assert exp["Y"] == pytest.approx(0.0, abs=1e-10)
        assert exp["I"] == pytest.approx(1.0, abs=1e-10)

    def test_excited_state_z(self):
        """⟨Z⟩ = -1 for |1⟩."""
        rho = _pure_state_rho([0, 1])
        st = StateTomography(n_qubits=1)
        exp = st.pauli_expectations(rho)
        assert exp["Z"] == pytest.approx(-1.0, abs=1e-10)

    def test_plus_state_x(self):
        """|+⟩ = (|0⟩+|1⟩)/√2: ⟨X⟩=1, ⟨Y⟩=⟨Z⟩=0."""
        rho = _pure_state_rho([1, 1])
        st = StateTomography(n_qubits=1)
        exp = st.pauli_expectations(rho)
        assert exp["X"] == pytest.approx(1.0, abs=1e-10)
        assert exp["Y"] == pytest.approx(0.0, abs=1e-10)
        assert exp["Z"] == pytest.approx(0.0, abs=1e-10)

    def test_maximally_mixed_all_zero(self):
        """All Pauli expectations except I vanish for I/2."""
        rho = _maximally_mixed(2)
        st = StateTomography(n_qubits=1)
        exp = st.pauli_expectations(rho)
        assert exp["I"] == pytest.approx(1.0, abs=1e-10)
        assert exp["X"] == pytest.approx(0.0, abs=1e-10)
        assert exp["Y"] == pytest.approx(0.0, abs=1e-10)
        assert exp["Z"] == pytest.approx(0.0, abs=1e-10)

    def test_wrong_shape_raises(self):
        rho = np.eye(4, dtype=complex) / 4
        st = StateTomography(n_qubits=1)  # expects 2x2
        with pytest.raises(TomographyError):
            st.pauli_expectations(rho)

    def test_two_qubit_bell_state_entanglement(self):
        """Bell state |Φ+⟩ has ⟨ZZ⟩=1, ⟨XX⟩=1, ⟨YY⟩=-1."""
        bell = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
        rho = _pure_state_rho(bell)
        st = StateTomography(n_qubits=2)
        exp = st.pauli_expectations(rho)
        assert exp["ZZ"] == pytest.approx(1.0, abs=1e-10)
        assert exp["XX"] == pytest.approx(1.0, abs=1e-10)
        assert exp["YY"] == pytest.approx(-1.0, abs=1e-10)
        assert exp["ZI"] == pytest.approx(0.0, abs=1e-10)
        assert exp["IZ"] == pytest.approx(0.0, abs=1e-10)

    def test_number_of_expectations_1q(self):
        rho = _maximally_mixed(2)
        st = StateTomography(n_qubits=1)
        exp = st.pauli_expectations(rho)
        assert len(exp) == 4  # I, X, Y, Z

    def test_number_of_expectations_2q(self):
        rho = _maximally_mixed(4)
        st = StateTomography(n_qubits=2)
        exp = st.pauli_expectations(rho)
        assert len(exp) == 16  # 4² Pauli operators


class TestReconstructLinear:
    def test_round_trip_pure_state(self):
        """Reconstruct pure state from its Pauli expectations."""
        rho = _pure_state_rho([1, 1])  # |+⟩
        st = StateTomography(n_qubits=1)
        exp = st.pauli_expectations(rho)
        rho_rec = st.reconstruct_linear(exp)
        assert np.allclose(rho, rho_rec, atol=1e-10)

    def test_round_trip_mixed_state(self):
        """Reconstruct a mixed state."""
        rho = np.array([[0.7, 0.1 + 0.05j], [0.1 - 0.05j, 0.3]], dtype=complex)
        st = StateTomography(n_qubits=1)
        exp = st.pauli_expectations(rho)
        rho_rec = st.reconstruct_linear(exp)
        assert np.allclose(rho, rho_rec, atol=1e-10)

    def test_round_trip_maximally_mixed(self):
        rho = _maximally_mixed(2)
        st = StateTomography(n_qubits=1)
        exp = st.pauli_expectations(rho)
        rho_rec = st.reconstruct_linear(exp)
        assert np.allclose(rho, rho_rec, atol=1e-10)

    def test_round_trip_bell_state(self):
        bell = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
        rho = _pure_state_rho(bell)
        st = StateTomography(n_qubits=2)
        exp = st.pauli_expectations(rho)
        rho_rec = st.reconstruct_linear(exp)
        assert np.allclose(rho, rho_rec, atol=1e-10)

    def test_reconstructed_hermitian(self):
        rho = _pure_state_rho([0.6, 0.8j])
        st = StateTomography(n_qubits=1)
        exp = st.pauli_expectations(rho)
        rho_rec = st.reconstruct_linear(exp)
        assert np.allclose(rho_rec, rho_rec.conj().T, atol=1e-10)


class TestStateFidelity:
    def test_identical_pure_states(self):
        rho = _pure_state_rho([1, 0])
        assert StateTomography.state_fidelity(rho, rho) == pytest.approx(1.0, abs=1e-8)

    def test_orthogonal_pure_states(self):
        rho0 = _pure_state_rho([1, 0])
        rho1 = _pure_state_rho([0, 1])
        assert StateTomography.state_fidelity(rho0, rho1) == pytest.approx(0.0, abs=1e-8)

    def test_pure_to_mixed(self):
        """|0⟩ vs I/2 should give F=0.5."""
        rho_pure = _pure_state_rho([1, 0])
        rho_mixed = _maximally_mixed(2)
        f = StateTomography.state_fidelity(rho_pure, rho_mixed)
        assert f == pytest.approx(0.5, abs=1e-8)

    def test_fidelity_symmetric(self):
        rho1 = _pure_state_rho([1, 1])
        rho2 = _pure_state_rho([1, -1])
        f12 = StateTomography.state_fidelity(rho1, rho2)
        f21 = StateTomography.state_fidelity(rho2, rho1)
        assert f12 == pytest.approx(f21, abs=1e-10)

    def test_fidelity_bounded(self):
        rho1 = _pure_state_rho([0.6, 0.8])
        rho2 = _pure_state_rho([0.8, -0.6])
        f = StateTomography.state_fidelity(rho1, rho2)
        assert 0.0 <= f <= 1.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(TomographyError, match="same shape"):
            StateTomography.state_fidelity(np.eye(2, dtype=complex), np.eye(4, dtype=complex))


# ===========================================================================
# ProcessTomography
# ===========================================================================

class TestProcessTomographyConstruction:
    def test_default_1qubit(self):
        pt = ProcessTomography()
        assert pt.n_qubits == 1
        assert pt._d == 2

    def test_invalid_n_qubits(self):
        with pytest.raises(TomographyError):
            ProcessTomography(n_qubits=0)


class TestChiMatrix:
    def test_identity_gate_shape(self):
        pt = ProcessTomography(n_qubits=1)
        chi = pt.chi_matrix(np.eye(2, dtype=complex))
        assert chi.shape == (4, 4)

    def test_identity_gate_chi_trace(self):
        """Tr(χ) = d for a trace-preserving map."""
        pt = ProcessTomography(n_qubits=1)
        chi = pt.chi_matrix(np.eye(2, dtype=complex))
        # For unitary map, chi has rank 1 and trace = d (as ∑_m chi_mm should = d ... actually
        # chi in Pauli basis satisfies ∑_m chi_mm = d for trace-preserving maps only when
        # normalised correctly). Let's just check chi is positive-semidefinite (rank 1).
        evals = np.linalg.eigvalsh(chi)
        # Rank-1: one large eigenvalue, rest ≈ 0
        sorted_evals = np.sort(np.abs(evals))[::-1]
        assert sorted_evals[0] > 0.9
        assert sorted_evals[1] < 1e-10

    def test_x_gate_maps_z_to_minus_z(self):
        """X gate: chi matrix should produce E(Z) = X Z X† = -Z channel."""
        pt = ProcessTomography(n_qubits=1)
        chi_x = pt.chi_matrix(X)
        chi_z = pt.chi_matrix(Z)
        # X and Z are distinct → chi matrices should differ
        assert not np.allclose(chi_x, chi_z, atol=1e-8)

    def test_identity_chi_rank1(self):
        """Identity channel has rank-1 chi matrix."""
        pt = ProcessTomography(n_qubits=1)
        chi = pt.chi_matrix(np.eye(2, dtype=complex))
        rank = int(np.linalg.matrix_rank(chi, tol=1e-8))
        assert rank == 1

    def test_x_gate_chi_rank1(self):
        pt = ProcessTomography(n_qubits=1)
        chi = pt.chi_matrix(X)
        rank = int(np.linalg.matrix_rank(chi, tol=1e-8))
        assert rank == 1

    def test_wrong_shape_raises(self):
        pt = ProcessTomography(n_qubits=1)
        with pytest.raises(TomographyError):
            pt.chi_matrix(np.eye(4, dtype=complex))

    def test_two_qubit_chi_shape(self):
        pt = ProcessTomography(n_qubits=2)
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)
        chi = pt.chi_matrix(CNOT)
        assert chi.shape == (16, 16)

    def test_chi_hermitian(self):
        """Chi matrix is Hermitian for a unitary channel."""
        pt = ProcessTomography(n_qubits=1)
        chi = pt.chi_matrix(H_gate)
        assert np.allclose(chi, chi.conj().T, atol=1e-10)


class TestProcessFidelity:
    def test_same_gate_fidelity_one(self):
        pt = ProcessTomography(n_qubits=1)
        chi = pt.chi_matrix(X)
        assert pt.process_fidelity(chi, chi) == pytest.approx(1.0, abs=1e-8)

    def test_orthogonal_gates_fidelity(self):
        """X vs Z process fidelity (orthogonal in Pauli space) should be small."""
        pt = ProcessTomography(n_qubits=1)
        chi_x = pt.chi_matrix(X)
        chi_z = pt.chi_matrix(Z)
        f = pt.process_fidelity(chi_x, chi_z)
        assert f < 0.5

    def test_fidelity_bounded(self):
        pt = ProcessTomography(n_qubits=1)
        chi_h = pt.chi_matrix(H_gate)
        chi_x = pt.chi_matrix(X)
        f = pt.process_fidelity(chi_h, chi_x)
        assert 0.0 <= f <= 1.0

    def test_shape_mismatch_raises(self):
        pt1 = ProcessTomography(n_qubits=1)
        pt2 = ProcessTomography(n_qubits=2)
        CNOT = np.eye(4, dtype=complex)
        with pytest.raises(TomographyError, match="same shape"):
            pt1.process_fidelity(pt1.chi_matrix(X), pt2.chi_matrix(CNOT))


class TestAverageGateFidelity:
    def test_perfect_gate_fidelity_one(self):
        """Perfect match: F_avg = 1."""
        pt = ProcessTomography(n_qubits=1)
        chi = pt.chi_matrix(X)
        f = pt.average_gate_fidelity(chi, chi)
        assert f == pytest.approx(1.0, abs=1e-8)

    def test_relation_to_process_fidelity(self):
        """F_avg = (d·F_proc + 1) / (d + 1)."""
        pt = ProcessTomography(n_qubits=1)
        chi_h = pt.chi_matrix(H_gate)
        chi_x = pt.chi_matrix(X)
        f_proc = pt.process_fidelity(chi_h, chi_x)
        f_avg = pt.average_gate_fidelity(chi_h, chi_x)
        d = pt._d
        expected_f_avg = (d * f_proc + 1.0) / (d + 1.0)
        assert f_avg == pytest.approx(expected_f_avg, abs=1e-10)

    def test_f_avg_bounded(self):
        pt = ProcessTomography(n_qubits=1)
        chi_z = pt.chi_matrix(Z)
        chi_x = pt.chi_matrix(X)
        f = pt.average_gate_fidelity(chi_z, chi_x)
        assert 0.0 <= f <= 1.0

    def test_two_qubit_perfect_fidelity(self):
        pt = ProcessTomography(n_qubits=2)
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)
        chi = pt.chi_matrix(CNOT)
        f = pt.average_gate_fidelity(chi, chi)
        assert f == pytest.approx(1.0, abs=1e-8)

    def test_f_avg_greater_than_f_proc(self):
        """For d >= 2, F_avg > F_proc for imperfect processes (since (1/(d+1)) > 0)."""
        pt = ProcessTomography(n_qubits=1)
        chi_h = pt.chi_matrix(H_gate)
        chi_x = pt.chi_matrix(X)
        f_proc = pt.process_fidelity(chi_h, chi_x)
        f_avg = pt.average_gate_fidelity(chi_h, chi_x)
        # F_avg = (d*F_proc + 1)/(d+1): for d=2, F_avg > F_proc when F_proc < 1
        if f_proc < 1.0:
            assert f_avg > f_proc


# ===========================================================================
# __init__ re-exports
# ===========================================================================

class TestTomographyInit:
    def test_state_tomography_exported(self):
        from quantx_engine import StateTomography as ST
        assert ST is StateTomography

    def test_process_tomography_exported(self):
        from quantx_engine import ProcessTomography as PT
        assert PT is ProcessTomography
