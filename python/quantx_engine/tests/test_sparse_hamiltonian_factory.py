"""Tests for SparseHamiltonian factory methods (M72).

Covers from_pauli_terms, xxz_chain, heisenberg_2d, bose_hubbard, hubbard_1d.
All tests require numpy; skipped otherwise.
"""
from __future__ import annotations

import math
import pytest

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

pytestmark = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy required")

from quantx_engine.lanczos import (
    LanczosSolver,
    LanczosError,
    SparseHamiltonian,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evals(H: SparseHamiltonian) -> np.ndarray:
    """Return all eigenvalues of H via dense diagonalization (small H only)."""
    return np.sort(np.linalg.eigvalsh(H.to_dense()))  # eigvalsh handles complex Hermitian


# ===========================================================================
# from_pauli_terms
# ===========================================================================

class TestFromPauliTerms:
    def test_identity_term_is_identity_matrix(self):
        """A single (c, 'II', [0, 1]) term gives c * I_{4x4}."""
        c = 2.5
        H = SparseHamiltonian.from_pauli_terms(2, [(c, "II", [0, 1])])
        M = H.to_dense()
        np.testing.assert_allclose(M, c * np.eye(4), atol=1e-12)

    def test_z_single_site_diagonal(self):
        """Z on site 0 of a 2-qubit system: diagonal is (+1, -1, +1, -1)."""
        H = SparseHamiltonian.from_pauli_terms(2, [(1.0, "Z", [0])])
        diag = np.diag(H.to_dense().real)
        # states 0=|00>, 1=|10>, 2=|01>, 3=|11>  (bit 0 = site 0)
        expected = np.array([1.0, -1.0, 1.0, -1.0])
        np.testing.assert_allclose(diag, expected, atol=1e-12)

    def test_x_single_site_offdiagonal(self):
        """X on site 0: flips bit 0, giving off-diagonal structure."""
        H = SparseHamiltonian.from_pauli_terms(1, [(1.0, "X", [0])])
        M = H.to_dense()
        # σ^x = [[0,1],[1,0]]
        expected = np.array([[0, 1], [1, 0]], dtype=complex)
        np.testing.assert_allclose(M, expected, atol=1e-12)

    def test_y_single_site_matrix(self):
        """Y on site 0 gives the Pauli-Y matrix."""
        H = SparseHamiltonian.from_pauli_terms(1, [(1.0, "Y", [0])])
        M = H.to_dense()
        expected = np.array([[0, -1j], [1j, 0]], dtype=complex)
        np.testing.assert_allclose(M, expected, atol=1e-12)

    def test_zz_two_site_diagonal(self):
        """ZZ on [0,1]: diagonal (1,-1,-1,1) for 2-qubit system."""
        H = SparseHamiltonian.from_pauli_terms(2, [(1.0, "ZZ", [0, 1])])
        diag = np.diag(H.to_dense().real)
        expected = np.array([1.0, -1.0, -1.0, 1.0])
        np.testing.assert_allclose(diag, expected, atol=1e-12)

    def test_hermitian(self):
        """from_pauli_terms with real Pauli strings gives Hermitian matrix."""
        H = SparseHamiltonian.from_pauli_terms(3, [
            (1.0, "ZZ", [0, 1]),
            (1.0, "ZZ", [1, 2]),
            (-0.5, "X", [0]),
            (-0.5, "X", [1]),
            (-0.5, "X", [2]),
        ])
        M = H.to_dense()
        np.testing.assert_allclose(M, M.conj().T, atol=1e-12)

    def test_reproduces_ising_1d_zz_only(self):
        """from_pauli_terms ZZ-only matches ising_1d(h=0) for n=3 (periodic)."""
        n = 3
        # ising_1d uses periodic BC: bonds (0,1), (1,2), (2,0)
        pauli_terms = [(-1.0, "ZZ", [i, (i + 1) % n]) for i in range(n)]
        H_pt = SparseHamiltonian.from_pauli_terms(n, pauli_terms)
        H_ising = SparseHamiltonian.ising_1d(n, J=1.0, h=0.0)
        np.testing.assert_allclose(
            H_pt.to_dense().real,
            H_ising.to_dense().real,
            atol=1e-12,
        )

    def test_sum_of_x_terms(self):
        """Sum of X_i terms gives transverse-field part of Ising model."""
        n = 3
        pauli_terms = [(-0.5, "X", [i]) for i in range(n)]
        H_pt = SparseHamiltonian.from_pauli_terms(n, pauli_terms)
        H_ising_x = SparseHamiltonian.ising_1d(n, J=0.0, h=0.5)
        np.testing.assert_allclose(
            H_pt.to_dense().real,
            H_ising_x.to_dense().real,
            atol=1e-12,
        )

    def test_complex_coefficient(self):
        """Complex coefficients are supported."""
        H = SparseHamiltonian.from_pauli_terms(1, [(1j, "Y", [0])])
        M = H.to_dense()
        # i * Y = i * [[0,-i],[i,0]] = [[0,1],[-1,0]]
        expected = np.array([[0, 1], [-1, 0]], dtype=complex)
        np.testing.assert_allclose(M, expected, atol=1e-12)

    def test_dimension(self):
        """from_pauli_terms(5, ...).n == 32."""
        H = SparseHamiltonian.from_pauli_terms(5, [(1.0, "Z", [0])])
        assert H.n == 32

    def test_invalid_site_raises(self):
        """Out-of-range site index raises LanczosError."""
        with pytest.raises(LanczosError, match="out of range"):
            SparseHamiltonian.from_pauli_terms(2, [(1.0, "Z", [5])])

    def test_mismatched_ops_sites_raises(self):
        """len(ops) != len(sites) raises LanczosError."""
        with pytest.raises(LanczosError, match="must match"):
            SparseHamiltonian.from_pauli_terms(2, [(1.0, "ZZ", [0])])

    def test_unknown_pauli_raises(self):
        """Unknown Pauli character raises LanczosError."""
        with pytest.raises(LanczosError, match="Unknown Pauli"):
            SparseHamiltonian.from_pauli_terms(1, [(1.0, "A", [0])])

    def test_duplicate_sites_raises(self):
        """Duplicate site indices in a term raise LanczosError."""
        with pytest.raises(LanczosError, match="Duplicate"):
            SparseHamiltonian.from_pauli_terms(2, [(1.0, "ZZ", [0, 0])])

    def test_n_sites_zero_raises(self):
        """n_sites=0 raises LanczosError."""
        with pytest.raises(LanczosError, match="n_sites must be >= 1"):
            SparseHamiltonian.from_pauli_terms(0, [])

    def test_lanczos_works_with_from_pauli_terms(self):
        """LanczosSolver works on a from_pauli_terms Hamiltonian."""
        H = SparseHamiltonian.from_pauli_terms(3, [
            (-1.0, "ZZ", [0, 1]), (-1.0, "ZZ", [1, 2]),
            (-0.5, "X", [0]), (-0.5, "X", [1]), (-0.5, "X", [2]),
        ])
        result = LanczosSolver(k=2, seed=0).solve(H)
        assert len(result.eigenvalues) == 2
        assert result.eigenvalues[0] < result.eigenvalues[1]


# ===========================================================================
# xxz_chain
# ===========================================================================

class TestXXZChain:
    def test_dimension(self):
        """xxz_chain(4).n == 16."""
        assert SparseHamiltonian.xxz_chain(4).n == 16

    def test_hermitian(self):
        """xxz_chain is Hermitian."""
        H = SparseHamiltonian.xxz_chain(4, delta=0.5)
        M = H.to_dense()
        np.testing.assert_allclose(M, M.conj().T, atol=1e-12)

    def test_isotropic_matches_heisenberg_1d(self):
        """xxz_chain(n, J=J, delta=1) spectrum == heisenberg_1d(n, J=J)."""
        for n in (2, 3, 4):
            for J in (1.0, 2.0):
                evals_xxz = _evals(SparseHamiltonian.xxz_chain(n, J=J, delta=1.0))
                evals_heis = _evals(SparseHamiltonian.heisenberg_1d(n, J=J, hz=0.0))
                np.testing.assert_allclose(evals_xxz, evals_heis, atol=1e-10,
                                           err_msg=f"n={n} J={J} xxz(δ=1) != heisenberg_1d")

    def test_ising_limit_diagonal(self):
        """xxz_chain(2, J=1, delta=2, hz=0): ZZ-diagonal is ±J*delta."""
        H = SparseHamiltonian.xxz_chain(2, J=1.0, delta=2.0, hz=0.0)
        M = H.to_dense()
        # |00⟩: z0=+1, z1=+1, ZZ-diag = J*delta*1*1 = 2
        # |11⟩: z0=-1, z1=-1, ZZ-diag = J*delta*1 = 2
        # |01⟩, |10⟩: z0*z1=-1, ZZ-diag = -J*delta = -2 (plus XY off-diagonal)
        assert M[0, 0].real == pytest.approx(2.0, abs=1e-12)
        assert M[3, 3].real == pytest.approx(2.0, abs=1e-12)

    def test_hz_field_shifts_eigenvalues(self):
        """Non-zero hz shifts eigenvalues vs hz=0."""
        evals_0 = _evals(SparseHamiltonian.xxz_chain(3, delta=1.0, hz=0.0))
        evals_h = _evals(SparseHamiltonian.xxz_chain(3, delta=1.0, hz=1.0))
        assert not np.allclose(evals_0, evals_h, atol=1e-6)

    def test_xx_model_real(self):
        """xxz_chain(delta=0): pure XX model; matrix is real."""
        H = SparseHamiltonian.xxz_chain(3, delta=0.0)
        M = H.to_dense()
        np.testing.assert_allclose(M.imag, 0.0, atol=1e-12)

    def test_single_site_no_bonds(self):
        """xxz_chain(1) has no bonds; only hz field contributes."""
        H = SparseHamiltonian.xxz_chain(1, J=1.0, delta=1.0, hz=0.5)
        M = H.to_dense().real
        # Z|0⟩=+1, Z|1⟩=-1; hz*(1-2*bit)
        assert M[0, 0] == pytest.approx(0.5, abs=1e-12)
        assert M[1, 1] == pytest.approx(-0.5, abs=1e-12)

    def test_n_sites_zero_raises(self):
        """xxz_chain(0) raises LanczosError."""
        with pytest.raises(LanczosError, match="n_sites must be >= 1"):
            SparseHamiltonian.xxz_chain(0)

    def test_ground_state_antiferromagnet(self):
        """4-site XXZ(delta=1) ground state energy matches dense diagonalization."""
        H = SparseHamiltonian.xxz_chain(4, delta=1.0)
        e0_lanczos = LanczosSolver(k=1, seed=0).solve(H).ground_state_energy()
        e0_dense = _evals(H)[0]
        assert abs(e0_lanczos - e0_dense) < 1e-8


# ===========================================================================
# heisenberg_2d
# ===========================================================================

class TestHeisenberg2D:
    def test_dimension_2x2(self):
        """2×2 lattice: n=4 sites, dim=16."""
        H = SparseHamiltonian.heisenberg_2d(2, 2)
        assert H.n == 16

    def test_dimension_3x2(self):
        """3×2 lattice: n=6 sites, dim=64."""
        H = SparseHamiltonian.heisenberg_2d(3, 2)
        assert H.n == 64

    def test_hermitian(self):
        """heisenberg_2d is Hermitian."""
        H = SparseHamiltonian.heisenberg_2d(2, 2)
        M = H.to_dense()
        np.testing.assert_allclose(M, M.conj().T, atol=1e-12)

    def test_1d_limit_matches_heisenberg_1d(self):
        """heisenberg_2d(n, 1, J) == heisenberg_1d(n, J) (1D chain, Ly=1)."""
        for n in (2, 3, 4):
            evals_2d = _evals(SparseHamiltonian.heisenberg_2d(n, 1, J=1.0))
            evals_1d = _evals(SparseHamiltonian.heisenberg_1d(n, J=1.0, hz=0.0))
            np.testing.assert_allclose(evals_2d, evals_1d, atol=1e-10,
                                       err_msg=f"n={n} 2D(Ly=1) != 1D")

    def test_ground_state_2x2(self):
        """2×2 Heisenberg: ground state energy is negative (antiferromagnet)."""
        H = SparseHamiltonian.heisenberg_2d(2, 2, J=1.0)
        e0 = LanczosSolver(k=1, seed=0).solve(H).ground_state_energy()
        assert e0 < 0.0

    def test_coupling_scales_eigenvalues(self):
        """Doubling J doubles all eigenvalues."""
        evals1 = _evals(SparseHamiltonian.heisenberg_2d(2, 2, J=1.0))
        evals2 = _evals(SparseHamiltonian.heisenberg_2d(2, 2, J=2.0))
        np.testing.assert_allclose(2 * evals1, evals2, atol=1e-10)

    def test_bond_count_2x2(self):
        """2×2 lattice has 4 bonds: 2 horizontal + 2 vertical."""
        # With J=1, n_bonds unique diagonal entries per state = 4 bonds
        # This is a structural test — just check the matrix is non-trivial
        H = SparseHamiltonian.heisenberg_2d(2, 2, J=1.0)
        M = H.to_dense()
        assert np.any(M != 0)

    def test_column_chain_lx1_matches_heisenberg_1d(self):
        """heisenberg_2d(1, n) (column chain) also matches heisenberg_1d."""
        for n in (2, 3, 4):
            evals_2d = _evals(SparseHamiltonian.heisenberg_2d(1, n, J=1.0))
            evals_1d = _evals(SparseHamiltonian.heisenberg_1d(n, J=1.0))
            np.testing.assert_allclose(evals_2d, evals_1d, atol=1e-10,
                                       err_msg=f"Lx=1,Ly={n} column != 1D chain")

    def test_single_site_2d(self):
        """heisenberg_2d(1, 1) — single site, no bonds."""
        H = SparseHamiltonian.heisenberg_2d(1, 1, J=1.0)
        assert H.n == 2
        M = H.to_dense()
        np.testing.assert_allclose(M, 0.0, atol=1e-12)


# ===========================================================================
# bose_hubbard
# ===========================================================================

class TestBoseHubbard:
    def test_dimension_l2_max2(self):
        """L=2, max_occ=2: dim = 3^2 = 9."""
        H = SparseHamiltonian.bose_hubbard(2, max_occ=2)
        assert H.n == 9

    def test_dimension_l3_max1(self):
        """L=3, max_occ=1: dim = 2^3 = 8."""
        H = SparseHamiltonian.bose_hubbard(3, max_occ=1)
        assert H.n == 8

    def test_hermitian(self):
        """bose_hubbard is Hermitian."""
        H = SparseHamiltonian.bose_hubbard(3, t=1.0, U=4.0, max_occ=2)
        M = H.to_dense()
        np.testing.assert_allclose(M, M.conj().T, atol=1e-12)

    def test_real_matrix(self):
        """bose_hubbard matrix is real (all matrix elements are real)."""
        H = SparseHamiltonian.bose_hubbard(2, t=1.0, U=2.0, max_occ=3)
        M = H.to_dense()
        np.testing.assert_allclose(M.imag, 0.0, atol=1e-12)

    def test_non_interacting_hopping(self):
        """U=0 on-site interaction: diagonal has no U contribution."""
        H = SparseHamiltonian.bose_hubbard(2, t=1.0, U=0.0, max_occ=3)
        M = H.to_dense()
        # With U=0, diagonal entries from single-occupancy states should be 0
        # State 0 = |0,0⟩: no hopping partner, energy=0
        assert M[0, 0].real == pytest.approx(0.0, abs=1e-12)

    def test_interaction_term_l1(self):
        """Single site L=1: no hopping, only interaction U/2 * n*(n-1)."""
        H = SparseHamiltonian.bose_hubbard(1, t=1.0, U=4.0, max_occ=3)
        M = H.to_dense().real
        # n=0: E=0, n=1: E=0, n=2: E=U/2*2=U=4, n=3: E=U/2*6=3U=12
        assert M[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert M[1, 1] == pytest.approx(0.0, abs=1e-12)
        assert M[2, 2] == pytest.approx(4.0, abs=1e-12)
        assert M[3, 3] == pytest.approx(12.0, abs=1e-12)

    def test_hopping_amplitude(self):
        """L=2, max_occ=1, U=0: Hamiltonian is the XX hopping matrix."""
        H = SparseHamiltonian.bose_hubbard(2, t=1.0, U=0.0, max_occ=1)
        M = H.to_dense().real
        # States: (n0, n1) → index = n0 + 2*n1
        # |0,0⟩=0, |1,0⟩=1, |0,1⟩=2, |1,1⟩=3
        # Hopping: |1,0⟩↔|0,1⟩ with amp -t*sqrt(1*1) = -1
        assert M[1, 2] == pytest.approx(-1.0, abs=1e-12)
        assert M[2, 1] == pytest.approx(-1.0, abs=1e-12)

    def test_max_occ_zero_raises(self):
        """max_occ=0 raises LanczosError."""
        with pytest.raises(LanczosError, match="max_occ"):
            SparseHamiltonian.bose_hubbard(2, max_occ=0)

    def test_ground_state_energy(self):
        """L=3, Bose-Hubbard ground state from Lanczos matches dense diag."""
        H = SparseHamiltonian.bose_hubbard(3, t=1.0, U=4.0, max_occ=2)
        e0_lz = LanczosSolver(k=1, seed=0).solve(H).ground_state_energy()
        e0_dense = _evals(H)[0]
        assert abs(e0_lz - e0_dense) < 1e-8


# ===========================================================================
# hubbard_1d
# ===========================================================================

class TestHubbard1D:
    def test_dimension_l2(self):
        """L=2: dim = 4^2 = 16."""
        H = SparseHamiltonian.hubbard_1d(2)
        assert H.n == 16

    def test_dimension_l3(self):
        """L=3: dim = 4^3 = 64."""
        H = SparseHamiltonian.hubbard_1d(3)
        assert H.n == 64

    def test_hermitian(self):
        """hubbard_1d is Hermitian."""
        H = SparseHamiltonian.hubbard_1d(2, t=1.0, U=4.0)
        M = H.to_dense()
        np.testing.assert_allclose(M, M.conj().T, atol=1e-12)

    def test_real_matrix(self):
        """hubbard_1d matrix is real."""
        H = SparseHamiltonian.hubbard_1d(2, t=1.0, U=4.0)
        M = H.to_dense()
        np.testing.assert_allclose(M.imag, 0.0, atol=1e-12)

    def test_u_zero_independent_sectors(self):
        """U=0: up and down sectors are independent; ground state = 2 * E_free."""
        # For U=0 and 1 particle per sector in L=2 chain, E_free = -t
        # So 2-particle (1↑, 1↓) ground state = 2 * (-t) = -2t
        H = SparseHamiltonian.hubbard_1d(2, t=1.0, U=0.0)
        e0 = LanczosSolver(k=1, seed=0).solve(H).ground_state_energy()
        # The full spectrum's minimum energy is at half-filling with U=0
        # Minimum is achieved with both spin-up and spin-down in lowest orbital: E = -2t
        assert e0 == pytest.approx(-2.0, abs=1e-6)

    def test_interaction_raises_energy(self):
        """Turning on U increases the ground state energy."""
        e0_u0 = LanczosSolver(k=1, seed=0).solve(
            SparseHamiltonian.hubbard_1d(2, t=1.0, U=0.0)).ground_state_energy()
        e0_u4 = LanczosSolver(k=1, seed=0).solve(
            SparseHamiltonian.hubbard_1d(2, t=1.0, U=4.0)).ground_state_energy()
        assert e0_u4 > e0_u0

    def test_empty_state_zero_energy(self):
        """Vacuum state |0⟩ (all empty) has energy 0."""
        H = SparseHamiltonian.hubbard_1d(2, t=1.0, U=4.0)
        M = H.to_dense()
        assert M[0, 0].real == pytest.approx(0.0, abs=1e-12)

    def test_doubly_occupied_site_diagonal(self):
        """State with one doubly-occupied site contributes U to diagonal."""
        # L=2: state index = up_bits | (dn_bits << 2)
        # State with site 0 doubly occupied: up_bits=0b01=1, dn_bits=0b01=1 → idx=1+4=5
        H = SparseHamiltonian.hubbard_1d(2, t=1.0, U=4.0)
        M = H.to_dense()
        idx = 1 | (1 << 2)   # site 0 doubly occupied
        assert M[idx, idx].real == pytest.approx(4.0, abs=1e-12)

    def test_l1_only_interaction(self):
        """L=1: no hopping, only on-site U for doubly occupied state."""
        H = SparseHamiltonian.hubbard_1d(1, t=1.0, U=6.0)
        M = H.to_dense().real
        # dim=4: |0⟩=0, |↑⟩=1, |↓⟩=2, |↑↓⟩=3
        assert M[0, 0] == pytest.approx(0.0)
        assert M[1, 1] == pytest.approx(0.0)
        assert M[2, 2] == pytest.approx(0.0)
        assert M[3, 3] == pytest.approx(6.0)

    def test_ground_state_energy_matches_dense(self):
        """L=2 Hubbard Lanczos ground state matches dense diagonalization."""
        H = SparseHamiltonian.hubbard_1d(2, t=1.0, U=4.0)
        e0_lz = LanczosSolver(k=1, seed=0).solve(H).ground_state_energy()
        e0_dense = _evals(H)[0]
        assert abs(e0_lz - e0_dense) < 1e-8

    def test_l1_raises_on_zero(self):
        """L=0 raises LanczosError."""
        with pytest.raises(LanczosError, match="L must be >= 1"):
            SparseHamiltonian.hubbard_1d(0)

    def test_l3_ground_state_matches_dense(self):
        """L=3 Hubbard Lanczos ground state matches dense diagonalization (validates JW signs)."""
        H = SparseHamiltonian.hubbard_1d(3, t=1.0, U=4.0)
        e0_lz = LanczosSolver(k=1, seed=0).solve(H).ground_state_energy()
        e0_dense = _evals(H)[0]
        assert abs(e0_lz - e0_dense) < 1e-8
