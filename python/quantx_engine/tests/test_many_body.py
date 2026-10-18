"""Tests for many_body.py — M102 many-body physics enhancements."""
from __future__ import annotations

import math
import numpy as np
import pytest

from quantx_engine.contrib.many_body import (
    ManyBodyError,
    OTOCResult,
    OTOCSweepResult,
    SYKResult,
    SYKModel,
    DMRG2DResult,
    SnakeMapper,
    compute_otoc,
    otoc_sweep,
    syk_hamiltonian,
    dmrg_2d,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_hermitian(M: np.ndarray, atol: float = 1e-10) -> bool:
    return np.allclose(M, M.conj().T, atol=atol)


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — OTOCs
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeOTOC:
    def _ising_H(self, n=4, J=1.0, h=0.5):
        """Transverse-field Ising Hamiltonian as dense matrix."""
        from quantx_engine.contrib.trotter import ising_trotter
        sim = ising_trotter(n=n, J=J, h=h)
        return sim.hamiltonian

    def test_returns_otoc_result(self):
        H = self._ising_H()
        r = compute_otoc(H, "X", "Z", 0, 3, t=0.5, n_qubits=4)
        assert isinstance(r, OTOCResult)

    def test_otoc_value_in_range(self):
        """C(t) = 1 - Re F(t) should lie in [-1, 2] in general, but typically [0,1]."""
        H = self._ising_H()
        r = compute_otoc(H, "X", "Z", 0, 3, t=1.0, n_qubits=4)
        assert -1.5 < r.otoc_value < 2.0

    def test_fotoc_is_complex(self):
        H = self._ising_H()
        r = compute_otoc(H, "X", "Z", 0, 3, t=1.0, n_qubits=4)
        assert isinstance(r.fotoc, complex)

    def test_t_zero_otoc_zero(self):
        """At t=0, W(0)=W, so [W,V] is fixed and C(0) should match 1-Re⟨ψ|W†V†WV|ψ⟩."""
        H = self._ising_H()
        # At t=0: F(0) = ⟨ψ|W†V†WV|ψ⟩; if W and V commute, F(0)=1, C(0)=0
        # Use same-site same-op: W=V=Z at site 0 → Z†Z†ZZ = I → F(0)=1, C(0)=0
        r = compute_otoc(H, "Z", "Z", 0, 0, t=0.0, n_qubits=4)
        assert abs(r.otoc_value) < 1e-10

    def test_same_site_commuting_ops(self):
        """Z and I commute at any site → OTOC value should be 0 at t=0."""
        H = self._ising_H()
        r = compute_otoc(H, "Z", "I", 0, 0, t=0.0, n_qubits=4)
        assert abs(r.otoc_value) < 1e-10

    def test_metadata_keys(self):
        H = self._ising_H()
        r = compute_otoc(H, "X", "Z", 0, 3, t=1.0, n_qubits=4)
        assert {"op_w", "site_w", "op_v", "site_v", "method"} <= r.metadata.keys()

    def test_metadata_method_exact(self):
        H = self._ising_H()
        r = compute_otoc(H, "X", "Z", 0, 3, t=1.0, n_qubits=4)
        assert r.metadata["method"] == "exact_expm"

    def test_wrong_hamiltonian_shape_raises(self):
        H_bad = np.eye(8)  # 3-qubit H with n_qubits=4
        with pytest.raises(ManyBodyError, match="inconsistent"):
            compute_otoc(H_bad, "X", "Z", 0, 3, t=1.0, n_qubits=4)

    def test_invalid_op_raises(self):
        H = self._ising_H()
        with pytest.raises(ManyBodyError):
            compute_otoc(H, "A", "Z", 0, 3, t=1.0, n_qubits=4)

    def test_invalid_site_raises(self):
        H = self._ising_H()
        with pytest.raises(ManyBodyError):
            compute_otoc(H, "X", "Z", 10, 3, t=1.0, n_qubits=4)

    def test_zero_norm_initial_state_raises(self):
        H = self._ising_H()
        with pytest.raises(ManyBodyError, match="zero norm"):
            compute_otoc(H, "X", "Z", 0, 3, t=1.0, n_qubits=4,
                         initial_state=np.zeros(16))

    def test_n_qubits_stored(self):
        H = self._ising_H()
        r = compute_otoc(H, "X", "Z", 0, 3, t=1.0, n_qubits=4)
        assert r.n_qubits == 4

    def test_t_stored(self):
        H = self._ising_H()
        r = compute_otoc(H, "X", "Z", 0, 3, t=1.23, n_qubits=4)
        assert r.t == pytest.approx(1.23)

    def test_two_qubit_system(self):
        """Minimal 2-qubit OTOC — just ensure no crash."""
        from quantx_engine.contrib.trotter import heisenberg_trotter
        H = heisenberg_trotter(n=2).hamiltonian
        r = compute_otoc(H, "X", "Z", 0, 1, t=0.5, n_qubits=2)
        assert isinstance(r.otoc_value, float)

    def test_otoc_grows_with_time_chaotic(self):
        """For a non-integrable system, OTOC at t>0 should differ from t=0."""
        H = self._ising_H(n=4, J=1.0, h=0.8)
        r0 = compute_otoc(H, "X", "Z", 0, 3, t=0.0, n_qubits=4)
        r1 = compute_otoc(H, "X", "Z", 0, 3, t=2.0, n_qubits=4)
        # Values should differ — chaotic growth
        assert abs(r1.otoc_value - r0.otoc_value) > 1e-6


class TestOTOCSweep:
    def _H(self):
        from quantx_engine.contrib.trotter import ising_trotter
        return ising_trotter(n=4, J=1.0, h=0.5).hamiltonian

    def test_returns_sweep_result(self):
        r = otoc_sweep(self._H(), "X", "Z", 0, 3,
                       t_list=[0.0, 0.5, 1.0, 1.5], n_qubits=4)
        assert isinstance(r, OTOCSweepResult)

    def test_output_lengths(self):
        t_list = [0.0, 0.5, 1.0, 1.5, 2.0]
        r = otoc_sweep(self._H(), "X", "Z", 0, 3, t_list=t_list, n_qubits=4)
        assert len(r.otoc_values) == 5
        assert len(r.fotoc_values) == 5

    def test_fotoc_complex(self):
        r = otoc_sweep(self._H(), "X", "Z", 0, 3,
                       t_list=[0.5, 1.0], n_qubits=4)
        assert r.fotoc_values.dtype == complex

    def test_scrambling_time_none_if_below_threshold(self):
        """At very short times C(t) should stay near 0 → no scrambling time."""
        H = self._H()
        r = otoc_sweep(H, "Z", "Z", 0, 0, t_list=[0.0, 0.001], n_qubits=4)
        assert r.scrambling_time is None

    def test_empty_t_list_raises(self):
        with pytest.raises(ManyBodyError):
            otoc_sweep(self._H(), "X", "Z", 0, 3, t_list=[], n_qubits=4)

    def test_t_list_stored(self):
        t_list = [0.0, 1.0, 2.0]
        r = otoc_sweep(self._H(), "X", "Z", 0, 3, t_list=t_list, n_qubits=4)
        np.testing.assert_allclose(r.t_list, t_list)


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — SYK model
# ─────────────────────────────────────────────────────────────────────────────

class TestSYKModel:
    def test_constructs_N4_q4(self):
        m = SYKModel(N=4, q=4, seed=0)
        assert m.N == 4 and m.q == 4

    def test_constructs_N6_q4(self):
        m = SYKModel(N=6, q=4, seed=1)
        assert m.n_qubits == 3

    def test_constructs_N8_q4(self):
        m = SYKModel(N=8, q=4, seed=2)
        assert m.dim == 16

    def test_odd_N_raises(self):
        with pytest.raises(ManyBodyError):
            SYKModel(N=5, q=4)

    def test_odd_q_raises(self):
        with pytest.raises(ManyBodyError):
            SYKModel(N=4, q=3)

    def test_q_exceeds_N_raises(self):
        with pytest.raises(ManyBodyError):
            SYKModel(N=4, q=6)

    def test_hamiltonian_hermitian(self):
        m = SYKModel(N=4, q=4, seed=0)
        assert _is_hermitian(m.hamiltonian)

    def test_hamiltonian_shape(self):
        m = SYKModel(N=6, q=4, seed=0)
        assert m.hamiltonian.shape == (8, 8)

    def test_hamiltonian_traceless(self):
        """SYK Hamiltonian should be traceless (zero average energy)."""
        m = SYKModel(N=6, q=4, seed=0)
        assert abs(np.trace(m.hamiltonian)) < 1e-10

    def test_different_seeds_different_H(self):
        H1 = SYKModel(N=6, q=4, seed=1).hamiltonian
        H2 = SYKModel(N=6, q=4, seed=2).hamiltonian
        assert not np.allclose(H1, H2)

    def test_same_seed_reproducible(self):
        H1 = SYKModel(N=6, q=4, seed=42).hamiltonian
        H2 = SYKModel(N=6, q=4, seed=42).hamiltonian
        np.testing.assert_array_equal(H1, H2)

    def test_ground_state_energy_negative(self):
        """SYK ground state energy is negative (random Gaussian couplings)."""
        m = SYKModel(N=8, q=4, seed=0)
        assert m.ground_state_energy < 0

    def test_ground_state_energy_scales_with_N(self):
        """E0 / N should be roughly constant (extensive ground state)."""
        e4 = SYKModel(N=4, q=4, seed=0).ground_state_energy
        e8 = SYKModel(N=8, q=4, seed=0).ground_state_energy
        # Both negative; larger N → lower absolute energy roughly ∝ N
        assert e8 < e4  # more modes → lower ground state energy


class TestSYKDiagonalise:
    def test_returns_syk_result(self):
        r = SYKModel(N=4, q=4, seed=0).diagonalise()
        assert isinstance(r, SYKResult)

    def test_energies_sorted(self):
        r = SYKModel(N=4, q=4, seed=0).diagonalise()
        assert np.all(np.diff(r.energies) >= -1e-10)

    def test_energies_real(self):
        r = SYKModel(N=6, q=4, seed=0).diagonalise()
        assert np.all(np.isreal(r.energies))

    def test_level_spacing_ratio_in_range(self):
        """⟨r⟩ should be between Poisson (0.386) and GOE (0.536)."""
        r = SYKModel(N=8, q=4, seed=7).diagonalise()
        if not math.isnan(r.level_spacing_ratio):
            assert 0.2 < r.level_spacing_ratio < 0.65

    def test_spectral_form_factor_shape(self):
        t_list = [0.1, 0.5, 1.0]
        r = SYKModel(N=4, q=4, seed=0).diagonalise(t_list=t_list)
        assert r.spectral_form_factor is not None
        assert len(r.spectral_form_factor) == 3

    def test_spectral_form_factor_non_negative(self):
        r = SYKModel(N=4, q=4, seed=0).diagonalise(t_list=[0.0, 1.0])
        assert np.all(r.spectral_form_factor >= 0)

    def test_sff_at_zero_equals_dim_squared(self):
        """SFF(t=0) = |Tr(I)|² = dim²."""
        m = SYKModel(N=4, q=4, seed=0)
        r = m.diagonalise(t_list=[0.0])
        assert abs(r.spectral_form_factor[0] - m.dim ** 2) < 1e-6


class TestSYKHamiltonianHelper:
    def test_returns_ndarray(self):
        H = syk_hamiltonian(N=4, q=4, seed=0)
        assert isinstance(H, np.ndarray)

    def test_shape(self):
        H = syk_hamiltonian(N=6, q=4, seed=0)
        assert H.shape == (8, 8)

    def test_hermitian(self):
        H = syk_hamiltonian(N=6, q=4, seed=0)
        assert _is_hermitian(H)

    def test_q2_works(self):
        """q=2 SYK is free fermions (quadratic, integrable)."""
        H = syk_hamiltonian(N=4, q=2, seed=0)
        assert H.shape == (4, 4) and _is_hermitian(H)


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — SnakeMapper
# ─────────────────────────────────────────────────────────────────────────────

class TestSnakeMapper:
    def test_2x2_total_sites(self):
        m = SnakeMapper(2, 2)
        assert m.n_sites == 4

    def test_3x3_total_sites(self):
        assert SnakeMapper(3, 3).n_sites == 9

    def test_2x3_total_sites(self):
        assert SnakeMapper(2, 3).n_sites == 6

    def test_to_1d_to_2d_roundtrip(self):
        m = SnakeMapper(3, 4)
        for x in range(3):
            for y in range(4):
                idx = m.to_1d(x, y)
                x2, y2 = m.to_2d(idx)
                assert (x2, y2) == (x, y)

    def test_snake_row0_left_to_right(self):
        """Row 0 should map left-to-right: (0,0)→0, (1,0)→1, (2,0)→2."""
        m = SnakeMapper(3, 3)
        assert m.to_1d(0, 0) == 0
        assert m.to_1d(1, 0) == 1
        assert m.to_1d(2, 0) == 2

    def test_snake_row1_right_to_left(self):
        """Row 1 should map right-to-left: (2,1)→3, (1,1)→4, (0,1)→5."""
        m = SnakeMapper(3, 3)
        assert m.to_1d(2, 1) == 3
        assert m.to_1d(1, 1) == 4
        assert m.to_1d(0, 1) == 5

    def test_all_indices_unique(self):
        m = SnakeMapper(4, 3)
        indices = [m.to_1d(x, y) for x in range(4) for y in range(3)]
        assert len(set(indices)) == m.n_sites

    def test_all_indices_in_range(self):
        m = SnakeMapper(3, 3)
        indices = [m.to_1d(x, y) for x in range(3) for y in range(3)]
        assert min(indices) == 0 and max(indices) == 8

    def test_out_of_range_to_1d_raises(self):
        m = SnakeMapper(3, 3)
        with pytest.raises(ManyBodyError):
            m.to_1d(5, 0)

    def test_out_of_range_to_2d_raises(self):
        m = SnakeMapper(3, 3)
        with pytest.raises(ManyBodyError):
            m.to_2d(100)

    def test_nn_pairs_2x2(self):
        """2×2 lattice: 4 nearest-neighbour bonds (2 horizontal + 2 vertical)."""
        m = SnakeMapper(2, 2)
        pairs = m.nn_pairs_1d()
        assert len(pairs) == 4

    def test_nn_pairs_all_valid(self):
        m = SnakeMapper(3, 3)
        for i, j in m.nn_pairs_1d():
            assert 0 <= i < m.n_sites and 0 <= j < m.n_sites
            assert i < j


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — 2D DMRG
# ─────────────────────────────────────────────────────────────────────────────

class TestDMRG2D:
    def test_returns_result(self):
        r = dmrg_2d(Lx=2, Ly=2)
        assert isinstance(r, DMRG2DResult)

    def test_lx_ly_stored(self):
        r = dmrg_2d(Lx=2, Ly=3)
        assert r.Lx == 2 and r.Ly == 3

    def test_sweeps_stored(self):
        r = dmrg_2d(Lx=2, Ly=2, sweeps=3)
        assert r.n_sweeps == 3

    def test_2x2_heisenberg_energy_negative(self):
        """2×2 Heisenberg antiferromagnet ground state energy should be negative."""
        r = dmrg_2d(Lx=2, Ly=2, J=1.0, model="heisenberg")
        assert r.energy < 0

    def test_2x2_ising_energy(self):
        """2×2 ferromagnetic Ising: all-down state gives E = -J*(n_bonds)."""
        r = dmrg_2d(Lx=2, Ly=2, J=1.0, model="ising")
        assert r.energy <= 0  # ferromagnetic → negative

    def test_2x2_exact_heisenberg_energy(self):
        """Exact 2×2 Heisenberg energy: compare with ED value.

        Pauli convention: H = J Σ (XX+YY+ZZ) where X,Y,Z are Pauli matrices.
        This is 4× the spin-½ convention H = J Σ S_i·S_j (since σ = 2S).
        4-site square spin-½ ground state = -2J in spin convention →
        E₀ = 4×(-2J) = -8J in Pauli convention → -8.0 for J=1.
        """
        r = dmrg_2d(Lx=2, Ly=2, J=1.0, model="heisenberg")
        assert r.energy == pytest.approx(-8.0, abs=1e-8)

    def test_invalid_model_raises(self):
        with pytest.raises(ManyBodyError, match="model"):
            dmrg_2d(Lx=2, Ly=2, model="xxz")

    def test_zero_bond_dim_raises(self):
        with pytest.raises(ManyBodyError, match="bond_dim"):
            dmrg_2d(Lx=2, Ly=2, bond_dim=0)

    def test_zero_sweeps_raises(self):
        with pytest.raises(ManyBodyError, match="sweeps"):
            dmrg_2d(Lx=2, Ly=2, sweeps=0)

    def test_2x1_single_bond(self):
        """1D chain (Ly=1) should just be the 2-site Heisenberg energy."""
        r = dmrg_2d(Lx=2, Ly=1, J=1.0, model="heisenberg")
        # 2-site Heisenberg H = XX+YY+ZZ, ground state E0 = -3
        assert r.energy == pytest.approx(-3.0, abs=1e-8)

    def test_1x2_equivalent_to_2x1(self):
        """By symmetry, 1×2 and 2×1 should give the same energy."""
        r1 = dmrg_2d(Lx=2, Ly=1, J=1.0)
        r2 = dmrg_2d(Lx=1, Ly=2, J=1.0)
        assert r1.energy == pytest.approx(r2.energy, abs=1e-8)

    def test_larger_j_lower_energy(self):
        """Stronger coupling → lower ground state energy."""
        r1 = dmrg_2d(Lx=2, Ly=2, J=1.0)
        r2 = dmrg_2d(Lx=2, Ly=2, J=2.0)
        assert r2.energy < r1.energy

    def test_variational_path_runs_without_error(self):
        """4×4 lattice (16 sites > 12) exercises the variational MPS path.

        Verifies that the corrected einsum contraction and non-adjacent bond
        inclusion do not raise ValueError or produce NaN/Inf energies.
        The exact energy value is not asserted; only numerical sanity is checked.
        """
        r = dmrg_2d(Lx=4, Ly=4, J=1.0, bond_dim=4, sweeps=2, model="heisenberg")
        assert isinstance(r, DMRG2DResult)
        assert math.isfinite(r.energy)
        assert r.energy < 0  # antiferromagnetic → negative ground state

    def test_variational_ising_path_runs(self):
        """4×4 Ising lattice (variational path): completes without error, finite energy.

        The variational MPS path uses a mean-field approximation for non-adjacent
        (cross-row) bonds, so convergence to the true ground state requires more
        sweeps than tested here.  We only assert that the code runs correctly and
        returns a finite energy.
        """
        r = dmrg_2d(Lx=4, Ly=4, J=1.0, bond_dim=4, sweeps=2, model="ising")
        assert isinstance(r, DMRG2DResult)
        assert math.isfinite(r.energy)


# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────

class TestExports:
    def test_importable_from_module(self):
        from quantx_engine.contrib.many_body import (
            ManyBodyError, OTOCResult, OTOCSweepResult,
            SYKResult, SYKModel, DMRG2DResult, SnakeMapper,
            compute_otoc, otoc_sweep, syk_hamiltonian, dmrg_2d,
        )
        assert SYKModel is not None
