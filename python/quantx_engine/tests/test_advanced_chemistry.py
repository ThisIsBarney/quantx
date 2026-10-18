"""Tests for advanced quantum chemistry solvers (M79).

Covers: build_hamiltonian_matrix, hf_state, UCCSDSolver, ADAPTVQESolver,
QAOASolver, QITESolver, AdvChemError, and public __init__ exports.
"""
from __future__ import annotations

import math
import pytest

import numpy as np

from ..chemistry import PauliTerm, PauliHamiltonian, MolecularVQE, ChemistryError
from ..advanced_chemistry import (
    AdvChemError,
    build_hamiltonian_matrix,
    hf_state,
    UCCSDResult,
    UCCSDSolver,
    ADAPTVQEResult,
    ADAPTVQESolver,
    QAOAResult,
    QAOASolver,
    QITEResult,
    QITESolver,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_z_ham(n_qubits: int, qubit: int = 0, coeff: float = 1.0) -> PauliHamiltonian:
    """Z_qubit Hamiltonian on n_qubits qubits."""
    ham = PauliHamiltonian(n_qubits=n_qubits)
    ham.add(PauliTerm(coeff=complex(coeff), ops={qubit: "Z"}))
    return ham


def _make_zz_ham(n_qubits: int = 2) -> PauliHamiltonian:
    """Z₀⊗Z₁ Hamiltonian (2 qubits)."""
    ham = PauliHamiltonian(n_qubits=n_qubits)
    ham.add(PauliTerm(coeff=1.0+0j, ops={0: "Z", 1: "Z"}))
    return ham


def _h2_data():
    """Return H₂ Hamiltonian matrix, matrix ground-state energy, and raw ham.

    Note: ``MolecularVQE.h2().fci_energy`` is the reference FCI value from
    the molecular dataset (-1.1373 Ha).  The Hamiltonian here is a
    reduced-active-space representation whose lowest eigenvalue may differ
    by ~0.02 Ha.  We use the exact matrix eigenvalue as the ground-truth
    for variational-bound assertions.
    """
    problem = MolecularVQE.h2()
    H = build_hamiltonian_matrix(problem.hamiltonian)
    gs_energy = float(np.linalg.eigvalsh(H)[0])   # exact matrix ground state
    return H, gs_energy, problem.hamiltonian


# ===========================================================================
# AdvChemError
# ===========================================================================

class TestAdvChemError:
    def test_is_chemistry_error_subclass(self):
        assert issubclass(AdvChemError, ChemistryError)

    def test_raises_as_exception(self):
        with pytest.raises(AdvChemError):
            raise AdvChemError("test error")

    def test_caught_by_chemistry_error(self):
        with pytest.raises(ChemistryError):
            raise AdvChemError("caught as ChemistryError")


# ===========================================================================
# build_hamiltonian_matrix
# ===========================================================================

class TestBuildHamiltonianMatrix:
    def test_single_z_eigenvalues(self):
        """Single-qubit Z: eigenvalues {-1, +1}."""
        ham = _make_z_ham(n_qubits=1)
        H = build_hamiltonian_matrix(ham)
        assert H.shape == (2, 2)
        evals = sorted(np.linalg.eigvalsh(H))
        assert np.allclose(evals, [-1.0, 1.0])

    def test_single_z_matrix_entries(self):
        """Z gives diag(1, -1) for qubit 0 in MSB-first ordering."""
        ham = _make_z_ham(n_qubits=1)
        H = build_hamiltonian_matrix(ham)
        expected = np.diag([1.0, -1.0]).astype(complex)
        assert np.allclose(H, expected)

    def test_coeff_scaling(self):
        """Coefficient is applied correctly."""
        ham = _make_z_ham(n_qubits=1, coeff=3.0)
        H = build_hamiltonian_matrix(ham)
        assert np.allclose(H, np.diag([3.0, -3.0]))

    def test_x_eigenvalues(self):
        """Single-qubit X: eigenvalues {-1, +1}."""
        ham = PauliHamiltonian(n_qubits=1)
        ham.add(PauliTerm(coeff=1.0+0j, ops={0: "X"}))
        H = build_hamiltonian_matrix(ham)
        evals = sorted(np.linalg.eigvalsh(H))
        assert np.allclose(evals, [-1.0, 1.0])

    def test_hermitian(self):
        """Hamiltonian matrix is Hermitian."""
        problem = MolecularVQE.h2()
        H = build_hamiltonian_matrix(problem.hamiltonian)
        assert np.allclose(H, H.conj().T)

    def test_h2_shape(self):
        """H₂ Hamiltonian is 16×16 (4 qubits → 2⁴ = 16)."""
        problem = MolecularVQE.h2()
        H = build_hamiltonian_matrix(problem.hamiltonian)
        assert H.shape == (16, 16)

    def test_h2_minimum_eigenvalue(self):
        """H₂ matrix ground state is close to -1.1 Ha (negative, physically reasonable)."""
        H, gs_energy, _ = _h2_data()
        # The reduced-active-space H2 ground state is around -1.115 Ha.
        assert -1.2 < gs_energy < -1.0

    def test_two_qubit_zz(self):
        """Z₀⊗Z₁: eigenvalues {-1, -1, +1, +1}."""
        H = build_hamiltonian_matrix(_make_zz_ham())
        evals = sorted(np.linalg.eigvalsh(H))
        assert np.allclose(evals, [-1.0, -1.0, 1.0, 1.0])

    def test_identity_term(self):
        """All-identity term adds a constant shift."""
        ham = PauliHamiltonian(n_qubits=1)
        ham.add(PauliTerm(coeff=2.0+0j, ops={}))  # 2 * I
        H = build_hamiltonian_matrix(ham)
        assert np.allclose(H, 2.0 * np.eye(2))

    def test_sum_of_terms(self):
        """Sum of Z and X on 1 qubit: H = Z + X."""
        ham = PauliHamiltonian(n_qubits=1)
        ham.add(PauliTerm(coeff=1.0+0j, ops={0: "Z"}))
        ham.add(PauliTerm(coeff=1.0+0j, ops={0: "X"}))
        H = build_hamiltonian_matrix(ham)
        expected = np.array([[1, 1], [1, -1]], dtype=complex)
        assert np.allclose(H, expected)


# ===========================================================================
# hf_state
# ===========================================================================

class TestHFState:
    def test_one_electron_two_qubits(self):
        """1 electron in 2 qubits: |10⟩ = index 2."""
        psi = hf_state(1, 2)
        assert psi.shape == (4,)
        assert abs(psi[2]) == pytest.approx(1.0)  # |10⟩
        assert np.sum(np.abs(psi) ** 2) == pytest.approx(1.0)

    def test_two_electrons_four_qubits(self):
        """2 electrons in 4 qubits: |1100⟩ = index 12."""
        psi = hf_state(2, 4)
        assert psi.shape == (16,)
        assert abs(psi[12]) == pytest.approx(1.0)  # |1100⟩
        assert np.sum(np.abs(psi) ** 2) == pytest.approx(1.0)

    def test_normalization(self):
        for ne in range(1, 5):
            psi = hf_state(ne, 5)
            assert np.linalg.norm(psi) == pytest.approx(1.0)

    def test_single_nonzero_element(self):
        """HF state is a single computational basis vector."""
        psi = hf_state(2, 4)
        nonzero = np.nonzero(psi)[0]
        assert len(nonzero) == 1

    def test_dtype_complex(self):
        psi = hf_state(1, 2)
        assert np.iscomplexobj(psi)

    def test_three_electrons_four_qubits(self):
        """3 electrons in 4 qubits: |1110⟩ = index 14."""
        psi = hf_state(3, 4)
        assert abs(psi[14]) == pytest.approx(1.0)  # |1110⟩ = 8+4+2 = 14


# ===========================================================================
# UCCSDSolver
# ===========================================================================

class TestUCCSDSolverConstruction:
    def test_h2_param_counts(self):
        """H₂: 2 elec, 4 qubits → 4 singles + 1 double = 5 params."""
        solver = UCCSDSolver(n_electrons=2, n_qubits=4)
        # occ=[0,1], virt=[2,3]
        # singles: 2*2 = 4
        # doubles: C(2,2)*C(2,2) = 1
        assert solver.n_singles == 4
        assert solver.n_doubles == 1
        assert solver.n_params == 5

    def test_one_electron_two_orbitals(self):
        """1 elec, 2 qubits: 1 single + 0 doubles."""
        solver = UCCSDSolver(n_electrons=1, n_qubits=2)
        assert solver.n_singles == 1
        assert solver.n_doubles == 0
        assert solver.n_params == 1

    def test_invalid_n_electrons_zero(self):
        with pytest.raises(AdvChemError):
            UCCSDSolver(n_electrons=0, n_qubits=4)

    def test_invalid_n_electrons_equals_n_qubits(self):
        with pytest.raises(AdvChemError):
            UCCSDSolver(n_electrons=4, n_qubits=4)

    def test_generators_are_antihermitian(self):
        """All generators should satisfy K = -K†."""
        solver = UCCSDSolver(n_electrons=2, n_qubits=4)
        for K in solver._generators:
            assert np.allclose(K, -K.conj().T, atol=1e-10), "Generator not anti-Hermitian"


class TestUCCSDSolverRun:
    def setup_method(self):
        self.H, self.fci_energy, _ = _h2_data()
        self.solver = UCCSDSolver(n_electrons=2, n_qubits=4)
        # HF energy
        psi_hf = hf_state(2, 4)
        self.E_hf = float(np.real(psi_hf.conj() @ self.H @ psi_hf))

    def test_returns_uccsd_result(self):
        result = self.solver.run(self.H, n_iter=60)
        assert isinstance(result, UCCSDResult)

    def test_energy_below_hf(self):
        """UCCSD energy must be ≤ HF energy (variational principle)."""
        result = self.solver.run(self.H, n_iter=60)
        assert result.energy <= self.E_hf + 1e-8

    def test_energy_close_to_fci(self):
        """UCCSD is exact for H₂ (2-electron, 2-orbital): should reach matrix ground state."""
        result = self.solver.run(self.H, n_iter=100)
        assert abs(result.energy - self.fci_energy) < 0.03

    def test_result_fields(self):
        result = self.solver.run(self.H, n_iter=20)
        assert result.n_params == 5
        assert result.n_singles == 4
        assert result.n_doubles == 1
        assert len(result.params) == 5
        assert isinstance(result.converged, bool)

    def test_energy_is_real_finite(self):
        result = self.solver.run(self.H, n_iter=20)
        assert math.isfinite(result.energy)


# ===========================================================================
# ADAPTVQESolver
# ===========================================================================

class TestADAPTVQESolverConstruction:
    def test_default_pool_from_uccsd(self):
        """Default pool = UCCSD singles+doubles pool."""
        uccsd = UCCSDSolver(n_electrons=2, n_qubits=4)
        adapt = ADAPTVQESolver(n_electrons=2, n_qubits=4)
        assert len(adapt._pool) == uccsd.n_params

    def test_custom_pool_accepted(self):
        pool = [np.zeros((16, 16), dtype=complex)]
        adapt = ADAPTVQESolver(n_electrons=2, n_qubits=4, pool=pool)
        assert len(adapt._pool) == 1


class TestADAPTVQESolverRun:
    def setup_method(self):
        self.H, self.fci_energy, _ = _h2_data()
        psi_hf = hf_state(2, 4)
        self.E_hf = float(np.real(psi_hf.conj() @ self.H @ psi_hf))

    def test_returns_adapt_result(self):
        adapt = ADAPTVQESolver(n_electrons=2, n_qubits=4)
        result = adapt.run(self.H, max_operators=3, n_iter_per_step=20)
        assert isinstance(result, ADAPTVQEResult)

    def test_energy_below_hf(self):
        """ADAPT-VQE energy must be ≤ HF energy."""
        adapt = ADAPTVQESolver(n_electrons=2, n_qubits=4)
        result = adapt.run(self.H, max_operators=5, n_iter_per_step=30)
        assert result.energy <= self.E_hf + 1e-6

    def test_result_structure(self):
        adapt = ADAPTVQESolver(n_electrons=2, n_qubits=4)
        result = adapt.run(self.H, max_operators=2, n_iter_per_step=10)
        assert result.n_operators == len(result.operator_indices)
        assert result.n_operators == len(result.params)
        assert result.n_operators <= 2
        assert math.isfinite(result.energy)
        assert result.gradient_norm >= 0.0

    def test_threshold_stops_early(self):
        """With a very high threshold, ADAPT stops after 0 operators."""
        adapt = ADAPTVQESolver(n_electrons=2, n_qubits=4)
        result = adapt.run(self.H, threshold=1e9, max_operators=10)
        assert result.n_operators == 0


# ===========================================================================
# QAOASolver
# ===========================================================================

class TestQAOASolverConstruction:
    def test_builds_mixer(self):
        solver = QAOASolver(n_qubits=2)
        H_B = solver._H_mixer
        # H_B = X₀⊗I₁ + I₀⊗X₁: eigenvalues = {-2, 0, 0, +2}
        evals = sorted(np.linalg.eigvalsh(H_B))
        assert np.allclose(evals, [-2.0, 0.0, 0.0, 2.0])

    def test_initial_state_is_uniform(self):
        solver = QAOASolver(n_qubits=2)
        psi = solver._initial_state()
        expected = 0.5 * np.ones(4, dtype=complex)
        assert np.allclose(psi, expected)


class TestQAOASolverRun:
    def setup_method(self):
        # Z₀+Z₁: minimum is |11⟩ with energy -2
        self.ham = PauliHamiltonian(n_qubits=2)
        self.ham.add(PauliTerm(coeff=1.0+0j, ops={0: "Z"}))
        self.ham.add(PauliTerm(coeff=1.0+0j, ops={1: "Z"}))
        self.solver = QAOASolver(n_qubits=2)

    def test_returns_qaoa_result(self):
        result = self.solver.run(self.ham, p=1, n_iter=30, seed=0)
        assert isinstance(result, QAOAResult)

    def test_p_stored_in_result(self):
        for p in (1, 2, 3):
            result = self.solver.run(self.ham, p=p, n_iter=20, seed=0)
            assert result.p == p

    def test_gamma_beta_lengths(self):
        p = 2
        result = self.solver.run(self.ham, p=p, n_iter=20, seed=0)
        assert len(result.gamma) == p
        assert len(result.beta) == p

    def test_energy_finite(self):
        result = self.solver.run(self.ham, p=1, n_iter=40, seed=0)
        assert math.isfinite(result.energy)

    def test_energy_below_exact_minimum(self):
        """QAOA p=1 energy must be ≥ true minimum (variational lower bound)."""
        result = self.solver.run(self.ham, p=1, n_iter=60, seed=0)
        # True minimum eigenvalue for Z₀+Z₁ is -2
        assert result.energy >= -2.0 - 1e-8

    def test_energy_improves_with_p(self):
        """More layers should not worsen the minimum (or at least comparable)."""
        r1 = self.solver.run(self.ham, p=1, n_iter=60, seed=0)
        r2 = self.solver.run(self.ham, p=2, n_iter=60, seed=0)
        # p=2 can't do worse than a variational method with more params
        # (This is a soft bound since optimization is not exact)
        assert r2.energy <= r1.energy + 0.5  # generous tolerance

    def test_seed_reproducibility(self):
        r1 = self.solver.run(self.ham, p=1, n_iter=30, seed=42)
        r2 = self.solver.run(self.ham, p=1, n_iter=30, seed=42)
        assert r1.energy == r2.energy
        assert r1.gamma == r2.gamma


# ===========================================================================
# QITESolver
# ===========================================================================

class TestQITESolverSimple:
    def test_z_hamiltonian_converges_to_minus1(self):
        """QITE on Z (1 qubit) should converge to ground state energy -1."""
        ham = _make_z_ham(n_qubits=1)
        H = build_hamiltonian_matrix(ham)
        solver = QITESolver(n_qubits=1)
        result = solver.run(H, delta_tau=0.1, n_steps=100)
        assert result.final_energy == pytest.approx(-1.0, abs=1e-4)

    def test_energies_decrease_monotonically(self):
        """Each QITE step should (weakly) lower the energy."""
        H, _, _ = _h2_data()
        solver = QITESolver(n_qubits=4)
        psi0 = hf_state(2, 4)
        result = solver.run(H, delta_tau=0.05, n_steps=50, initial_state=psi0)
        for i in range(1, len(result.energies)):
            assert result.energies[i] <= result.energies[i - 1] + 1e-8

    def test_tau_values_spaced_correctly(self):
        solver = QITESolver(n_qubits=1)
        H = build_hamiltonian_matrix(_make_z_ham(n_qubits=1))
        result = solver.run(H, delta_tau=0.1, n_steps=5)
        expected = [0.0, 0.1, 0.2, 0.3, 0.4]
        assert len(result.tau_values) == len(result.energies)
        for got, want in zip(result.tau_values, expected):
            assert got == pytest.approx(want, abs=1e-10)


class TestQITESolverH2:
    def setup_method(self):
        self.H, self.fci_energy, _ = _h2_data()
        self.solver = QITESolver(n_qubits=4)

    def test_returns_qite_result(self):
        result = self.solver.run(self.H, n_steps=5)
        assert isinstance(result, QITEResult)

    def test_energy_approaches_fci(self):
        """QITE starting from HF should converge to the matrix ground state."""
        psi_hf = hf_state(2, 4)
        result = self.solver.run(
            self.H, delta_tau=0.05, n_steps=200, initial_state=psi_hf
        )
        assert result.final_energy < self.fci_energy + 0.01

    def test_converged_property(self):
        psi_hf = hf_state(2, 4)
        result = self.solver.run(
            self.H, delta_tau=0.05, n_steps=300, initial_state=psi_hf
        )
        # Long enough run should converge
        assert result.converged

    def test_initial_state_from_hf(self):
        """n_electrons parameter selects the HF state automatically."""
        r_explicit = self.solver.run(
            self.H, delta_tau=0.05, n_steps=20,
            initial_state=hf_state(2, 4)
        )
        r_implicit = self.solver.run(
            self.H, delta_tau=0.05, n_steps=20,
            n_electrons=2
        )
        assert r_explicit.energies[0] == pytest.approx(r_implicit.energies[0], abs=1e-10)

    def test_equal_superposition_initial(self):
        """Default initial state is equal superposition; first energy is real."""
        result = self.solver.run(self.H, delta_tau=0.02, n_steps=10)
        assert math.isfinite(result.energies[0])

    def test_final_energy_less_than_hf(self):
        psi_hf = hf_state(2, 4)
        E_hf = float(np.real(psi_hf.conj() @ self.H @ psi_hf))
        result = self.solver.run(
            self.H, delta_tau=0.05, n_steps=100, initial_state=psi_hf
        )
        assert result.final_energy <= E_hf + 1e-8


class TestQITEResultConvergedProperty:
    def test_too_few_steps(self):
        solver = QITESolver(n_qubits=1)
        H = build_hamiltonian_matrix(_make_z_ham(n_qubits=1))
        result = solver.run(H, n_steps=3)
        assert not result.converged  # fewer than 5 steps → never converged

    def test_converged_after_many_steps(self):
        solver = QITESolver(n_qubits=1)
        H = build_hamiltonian_matrix(_make_z_ham(n_qubits=1))
        # Starting from near-ground-state should plateau quickly
        result = solver.run(
            H, delta_tau=0.2, n_steps=200,
            initial_state=np.array([0.01, 0.9999], dtype=complex),
        )
        assert result.converged


# ===========================================================================
# Cross-solver consistency: UCCSD ≤ FCI, QITE ≤ FCI
# ===========================================================================

class TestCrossMethodConsistency:
    def setup_method(self):
        self.H, self.fci_energy, _ = _h2_data()

    def test_uccsd_above_fci(self):
        """UCCSD energy ≥ FCI (variational upper bound)."""
        result = UCCSDSolver(n_electrons=2, n_qubits=4).run(self.H, n_iter=80)
        assert result.energy >= self.fci_energy - 1e-5

    def test_qite_above_fci(self):
        """QITE energy ≥ exact ground state."""
        psi_hf = hf_state(2, 4)
        result = QITESolver(n_qubits=4).run(
            self.H, delta_tau=0.05, n_steps=200, initial_state=psi_hf
        )
        assert result.final_energy >= self.fci_energy - 1e-5

    def test_adapt_above_fci(self):
        """ADAPT-VQE energy ≥ FCI (variational upper bound)."""
        adapt = ADAPTVQESolver(n_electrons=2, n_qubits=4)
        result = adapt.run(self.H, max_operators=5, n_iter_per_step=40)
        assert result.energy >= self.fci_energy - 1e-5
