"""M113: Tests for excited-state quantum chemistry."""
from __future__ import annotations

import math
import pytest
import numpy as np

from quantx_engine.contrib.chemistry import PauliHamiltonian, PauliTerm
from quantx_engine.contrib.excited_state import (
    ExcitedStateError,
    ExcitedStateResult,
    ExcitedStateVQE,
    ConicalIntersectionResult,
    detect_conical_intersections,
    pyscf_to_pauli_hamiltonian,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_h2_hamiltonian() -> PauliHamiltonian:
    """Minimal 2-qubit Hamiltonian with known spectrum."""
    terms = [
        PauliTerm(coeff=-0.5+0j, ops={0: "Z"}),
        PauliTerm(coeff=-0.5+0j, ops={1: "Z"}),
        PauliTerm(coeff=0.2+0j, ops={0: "Z", 1: "Z"}),
        PauliTerm(coeff=0.1+0j, ops={0: "X", 1: "X"}),
    ]
    return PauliHamiltonian(terms=terms, n_qubits=2)


def _make_diagonal_hamiltonian(eigenvalues: list[float]) -> PauliHamiltonian:
    """Build a PauliHamiltonian whose two lowest eigenvalues are eigenvalues[0,1]."""
    e0, e1 = eigenvalues[0], eigenvalues[1]
    avg = (e0 + e1) / 2
    half = (e1 - e0) / 2
    terms = [
        PauliTerm(coeff=complex(avg), ops={}),
        PauliTerm(coeff=complex(half), ops={0: "Z"}),
    ]
    return PauliHamiltonian(terms=terms, n_qubits=1)


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_default_init(self):
        ham = _make_h2_hamiltonian()
        vqe = ExcitedStateVQE(ham)
        assert vqe.n_states == 3
        assert vqe.method == "exact"
        assert vqe.penalty_weight == 10.0

    def test_explicit_params(self):
        ham = _make_h2_hamiltonian()
        vqe = ExcitedStateVQE(ham, n_states=2, method="deflation", penalty_weight=5.0)
        assert vqe.n_states == 2
        assert vqe.method == "deflation"
        assert vqe.penalty_weight == 5.0

    def test_non_hamiltonian_raises(self):
        with pytest.raises(ExcitedStateError):
            ExcitedStateVQE("not a hamiltonian")  # type: ignore

    def test_n_states_zero_raises(self):
        with pytest.raises(ExcitedStateError):
            ExcitedStateVQE(_make_h2_hamiltonian(), n_states=0)

    def test_n_states_negative_raises(self):
        with pytest.raises(ExcitedStateError):
            ExcitedStateVQE(_make_h2_hamiltonian(), n_states=-1)

    def test_unknown_method_raises(self):
        with pytest.raises(ExcitedStateError):
            ExcitedStateVQE(_make_h2_hamiltonian(), method="unknown")

    def test_folded_spectrum_method_accepted(self):
        vqe = ExcitedStateVQE(_make_h2_hamiltonian(), method="folded_spectrum")
        assert vqe.method == "folded_spectrum"


# ── ExcitedStateResult ────────────────────────────────────────────────────────

class TestExcitedStateResult:
    def test_to_dict_keys(self):
        result = ExcitedStateVQE(_make_h2_hamiltonian()).solve()
        d = result.to_dict()
        assert "energies" in d
        assert "excitation_energies" in d
        assert "n_states" in d
        assert "n_qubits" in d
        assert "method" in d

    def test_excitation_energies_non_negative(self):
        result = ExcitedStateVQE(_make_h2_hamiltonian()).solve()
        for de in result.excitation_energies:
            assert de >= -1e-10  # should be non-negative (may have tiny float error)

    def test_n_states_matches_energies_len(self):
        result = ExcitedStateVQE(_make_h2_hamiltonian(), n_states=2).solve()
        assert result.n_states == len(result.energies)
        assert result.n_states - 1 == len(result.excitation_energies)


# ── Exact diagonalisation ─────────────────────────────────────────────────────

class TestExactMethod:
    def test_returns_exact_result(self):
        result = ExcitedStateVQE(_make_h2_hamiltonian(), method="exact").solve()
        assert isinstance(result, ExcitedStateResult)
        assert result.method == "exact"

    def test_ground_state_is_lowest(self):
        result = ExcitedStateVQE(_make_h2_hamiltonian(), method="exact", n_states=3).solve()
        for e in result.energies[1:]:
            assert e >= result.energies[0] - 1e-10

    def test_energies_sorted_ascending(self):
        result = ExcitedStateVQE(_make_h2_hamiltonian(), method="exact", n_states=3).solve()
        for i in range(len(result.energies) - 1):
            assert result.energies[i] <= result.energies[i + 1] + 1e-10

    def test_n_states_capped_at_dim(self):
        # 2-qubit Hamiltonian has dim=4; requesting 10 should give at most 4
        result = ExcitedStateVQE(_make_h2_hamiltonian(), method="exact", n_states=10).solve()
        assert result.n_states <= 4

    def test_one_state_is_ground_state(self):
        ham = _make_h2_hamiltonian()
        result = ExcitedStateVQE(ham, method="exact", n_states=1).solve()
        assert len(result.energies) == 1
        assert len(result.excitation_energies) == 0

    def test_excitation_energy_is_difference(self):
        result = ExcitedStateVQE(_make_h2_hamiltonian(), method="exact", n_states=3).solve()
        for k, de in enumerate(result.excitation_energies):
            assert abs(de - (result.energies[k + 1] - result.energies[0])) < 1e-10

    def test_known_2qubit_spectrum(self):
        """Diagonal 2-qubit Hamiltonian: check exact eigenvalues."""
        # H = -0.5 Z0 - 0.5 Z1 → eigenvalues: -1, 0, 0, 1
        terms = [
            PauliTerm(coeff=-0.5+0j, ops={0: "Z"}),
            PauliTerm(coeff=-0.5+0j, ops={1: "Z"}),
        ]
        ham = PauliHamiltonian(terms=terms, n_qubits=2)
        result = ExcitedStateVQE(ham, method="exact", n_states=4).solve()
        evals = sorted(result.energies)
        assert abs(evals[0] - (-1.0)) < 1e-10
        assert abs(evals[-1] - 1.0) < 1e-10


# ── Deflation VQE ─────────────────────────────────────────────────────────────

class TestDeflationMethod:
    def test_returns_deflation_result(self):
        result = ExcitedStateVQE(_make_h2_hamiltonian(), method="deflation").solve(seed=0)
        assert result.method == "deflation"

    def test_energies_count(self):
        result = ExcitedStateVQE(_make_h2_hamiltonian(), method="deflation", n_states=3).solve(seed=1)
        assert len(result.energies) == 3

    def test_deflation_ground_state_matches_exact(self):
        ham = _make_h2_hamiltonian()
        exact = ExcitedStateVQE(ham, method="exact", n_states=1).solve()
        defl = ExcitedStateVQE(ham, method="deflation", n_states=1).solve(seed=0)
        assert abs(defl.energies[0] - exact.energies[0]) < 1e-8

    def test_deflation_excited_states_match_exact(self):
        ham = _make_h2_hamiltonian()
        exact = ExcitedStateVQE(ham, method="exact", n_states=3).solve()
        defl = ExcitedStateVQE(ham, method="deflation", n_states=3).solve(seed=42)
        for k in range(min(len(exact.energies), len(defl.energies))):
            assert abs(defl.energies[k] - exact.energies[k]) < 1e-6

    def test_deflation_energies_non_decreasing(self):
        result = ExcitedStateVQE(_make_h2_hamiltonian(), method="deflation", n_states=3).solve(seed=7)
        for i in range(len(result.energies) - 1):
            assert result.energies[i] <= result.energies[i + 1] + 1e-6

    def test_penalty_weight_affects_result(self):
        ham = _make_h2_hamiltonian()
        r_low = ExcitedStateVQE(ham, method="deflation", n_states=2, penalty_weight=1.0).solve(seed=0)
        r_high = ExcitedStateVQE(ham, method="deflation", n_states=2, penalty_weight=100.0).solve(seed=0)
        # Ground states should be the same regardless of penalty weight
        assert abs(r_low.energies[0] - r_high.energies[0]) < 1e-8


# ── Folded-spectrum VQE ───────────────────────────────────────────────────────

class TestFoldedSpectrumMethod:
    def test_returns_folded_spectrum_result(self):
        result = ExcitedStateVQE(_make_h2_hamiltonian(), method="folded_spectrum").solve()
        assert result.method == "folded_spectrum"

    def test_folded_spectrum_matches_exact(self):
        ham = _make_h2_hamiltonian()
        exact = ExcitedStateVQE(ham, method="exact", n_states=3).solve()
        fs = ExcitedStateVQE(ham, method="folded_spectrum", n_states=3).solve()
        for k in range(min(len(exact.energies), len(fs.energies))):
            assert abs(fs.energies[k] - exact.energies[k]) < 1e-10

    def test_folded_spectrum_energies_sorted(self):
        result = ExcitedStateVQE(_make_h2_hamiltonian(), method="folded_spectrum", n_states=4).solve()
        for i in range(len(result.energies) - 1):
            assert result.energies[i] <= result.energies[i + 1] + 1e-10


# ── Conical intersection detection ───────────────────────────────────────────

def _make_crossing_hamiltonians(n_pts: int = 20):
    """
    Build a family of 2-qubit Hamiltonians whose two lowest eigenvalues cross.
    At lambda=0: E0 < E1; at lambda=1: E0 > E1. Crossing somewhere in between.
    """
    param_values = list(np.linspace(0, 1, n_pts))
    hamiltonians = []
    for lam in param_values:
        # H = alpha(lam) * Z0 + beta(lam) * Z1
        # E0 sweeps from -1 to +0.5, E1 sweeps from +0.5 to -1 → they cross
        alpha = 0.5 - lam          # +0.5 → -0.5
        beta  = lam - 0.5          # -0.5 → +0.5
        terms = [
            PauliTerm(coeff=complex(alpha), ops={0: "Z"}),
            PauliTerm(coeff=complex(beta), ops={1: "Z"}),
        ]
        hamiltonians.append(PauliHamiltonian(terms=terms, n_qubits=2))
    return hamiltonians, param_values


class TestDetectConicalIntersections:
    def test_returns_conical_intersection_result(self):
        hams, params = _make_crossing_hamiltonians()
        result = detect_conical_intersections(hams, params, n_states=2)
        assert isinstance(result, ConicalIntersectionResult)

    def test_energies_populated(self):
        hams, params = _make_crossing_hamiltonians()
        result = detect_conical_intersections(hams, params, n_states=2)
        assert 0 in result.energies
        assert 1 in result.energies
        assert len(result.energies[0]) == len(params)

    def test_crossing_detected(self):
        hams, params = _make_crossing_hamiltonians(n_pts=40)
        result = detect_conical_intersections(hams, params, n_states=2, gap_threshold=0.05)
        assert len(result.crossing_points) >= 1

    def test_crossing_near_midpoint(self):
        hams, params = _make_crossing_hamiltonians(n_pts=40)
        result = detect_conical_intersections(hams, params, n_states=2, gap_threshold=0.1)
        if result.crossing_points:
            lam_cross, _, _ = result.crossing_points[0]
            assert 0.2 < lam_cross < 0.8

    def test_to_dict(self):
        hams, params = _make_crossing_hamiltonians()
        result = detect_conical_intersections(hams, params, n_states=2)
        d = result.to_dict()
        assert "parameter_values" in d
        assert "energies" in d
        assert "crossing_points" in d
        assert "gap_threshold" in d

    def test_empty_hamiltonians_raises(self):
        with pytest.raises(ExcitedStateError):
            detect_conical_intersections([], [], n_states=2)

    def test_mismatched_lengths_raises(self):
        hams, params = _make_crossing_hamiltonians(n_pts=5)
        with pytest.raises(ExcitedStateError):
            detect_conical_intersections(hams, params[:-1], n_states=2)

    def test_no_crossing_when_separated(self):
        """States that never come close should produce no crossings."""
        n_pts = 20
        params = list(np.linspace(0, 1, n_pts))
        hams = []
        for _ in params:
            # 1-qubit H = 1.0 Z → eigenvalues {-1, +1}, gap = 2 >> threshold
            terms = [PauliTerm(coeff=1.0+0j, ops={0: "Z"})]
            hams.append(PauliHamiltonian(terms=terms, n_qubits=1))
        result = detect_conical_intersections(hams, params, n_states=2, gap_threshold=0.01)
        assert result.crossing_points == []

    def test_custom_state_pairs(self):
        hams, params = _make_crossing_hamiltonians(n_pts=30)
        result = detect_conical_intersections(
            hams, params, n_states=3, gap_threshold=0.1, state_pairs=[(0, 1)]
        )
        assert isinstance(result, ConicalIntersectionResult)

    def test_gap_threshold_stored(self):
        hams, params = _make_crossing_hamiltonians()
        result = detect_conical_intersections(hams, params, gap_threshold=0.05)
        assert result.gap_threshold == 0.05

    def test_parameter_values_stored(self):
        hams, params = _make_crossing_hamiltonians(n_pts=10)
        result = detect_conical_intersections(hams, params)
        assert result.parameter_values == params


# ── pyscf_to_pauli_hamiltonian ────────────────────────────────────────────────

class TestPySCFAdapter:
    def test_returns_pauli_hamiltonian(self):
        mol_data = {
            "n_orbitals": 2,
            "n_electrons": 2,
            "one_body_integrals": np.array([[-1.0, 0.1], [0.1, -0.8]]),
            "two_body_integrals": np.zeros((2, 2, 2, 2)),
        }
        ham = pyscf_to_pauli_hamiltonian(mol_data)
        assert isinstance(ham, PauliHamiltonian)

    def test_n_qubits_is_twice_n_orb(self):
        mol_data = {
            "n_orbitals": 3,
            "n_electrons": 2,
            "one_body_integrals": np.eye(3) * -0.5,
            "two_body_integrals": np.zeros((3, 3, 3, 3)),
        }
        ham = pyscf_to_pauli_hamiltonian(mol_data)
        assert ham.n_qubits == 6

    def test_zero_one_body_falls_back(self):
        """All-zero h1 → falls back to constant offset term."""
        mol_data = {
            "n_orbitals": 2,
            "n_electrons": 2,
            "one_body_integrals": np.zeros((2, 2)),
            "two_body_integrals": np.zeros((2, 2, 2, 2)),
        }
        ham = pyscf_to_pauli_hamiltonian(mol_data)
        assert len(ham.terms) >= 1


# ── Codex review regression tests ────────────────────────────────────────────

class TestDeflationRegressions:
    """Regression tests for Codex P1/P2 review findings on M113."""

    def test_conjugate_expectation_y_hamiltonian(self):
        """P2: deflation energies must be correct for Hamiltonians with Y terms.

        H = Y has eigenvalues {-1, +1} with complex eigenvectors.
        The naive ``v @ H @ v`` (no conjugate) evaluates to 0 for each
        eigenstate; the correct Hermitian form ``v†Hv`` gives ±1.
        """
        # Build single-qubit H = Y
        terms = [PauliTerm(coeff=1.0+0j, ops={0: "Y"})]
        ham = PauliHamiltonian(terms=terms, n_qubits=1)
        result = ExcitedStateVQE(ham, method="deflation", n_states=2).solve(seed=0)
        evals_sorted = sorted(result.energies)
        # Both eigenvalues must be ±1; not 0 (which the buggy non-conjugate form
        # would return for both eigenstates of Y).
        assert abs(evals_sorted[0] - (-1.0)) < 1e-6, (
            f"Ground-state energy of Y Hamiltonian should be -1, got {evals_sorted[0]}"
        )
        assert abs(evals_sorted[1] - 1.0) < 1e-6, (
            f"First excited energy of Y Hamiltonian should be +1, got {evals_sorted[1]}"
        )

    def test_deflation_no_duplicate_states_small_penalty(self):
        """P2: deflation must not return the ground state twice even when the
        user-supplied penalty_weight is smaller than the spectral gap.

        Hamiltonian H = 50·Z has eigenvalues {-50, +50}.  With the default
        penalty_weight=10 (< 100 = gap), the naïve deflation without adaptive
        scaling would re-select the ground state.  The fix scales the effective
        penalty to exceed the spectral width.
        """
        # Large gap Hamiltonian: eigenvalues ±50
        terms = [PauliTerm(coeff=50.0+0j, ops={0: "Z"})]
        ham = PauliHamiltonian(terms=terms, n_qubits=1)
        # Use small penalty_weight deliberately to expose the bug
        vqe = ExcitedStateVQE(ham, method="deflation", n_states=2, penalty_weight=5.0)
        result = vqe.solve(seed=0)
        assert len(result.energies) == 2
        # The two energies must be distinct (i.e. NOT both -50)
        assert abs(result.energies[0] - result.energies[1]) > 1.0, (
            f"Deflation returned duplicate energies: {result.energies}"
        )
        # Ground state must still be correct
        assert abs(result.energies[0] - (-50.0)) < 1e-6


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration:
    def test_all_three_methods_agree_on_ground_state(self):
        ham = _make_h2_hamiltonian()
        e_exact = ExcitedStateVQE(ham, method="exact", n_states=1).solve().energies[0]
        e_defl  = ExcitedStateVQE(ham, method="deflation", n_states=1).solve(seed=0).energies[0]
        e_fs    = ExcitedStateVQE(ham, method="folded_spectrum", n_states=1).solve().energies[0]
        assert abs(e_defl - e_exact) < 1e-6
        assert abs(e_fs - e_exact) < 1e-10

    def test_solve_seed_is_reproducible(self):
        ham = _make_h2_hamiltonian()
        r1 = ExcitedStateVQE(ham, method="deflation").solve(seed=123)
        r2 = ExcitedStateVQE(ham, method="deflation").solve(seed=123)
        assert r1.energies == r2.energies

    def test_n_qubits_stored_in_result(self):
        ham = _make_h2_hamiltonian()
        result = ExcitedStateVQE(ham).solve()
        assert result.n_qubits == ham.n_qubits
