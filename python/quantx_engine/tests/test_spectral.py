"""Tests for SpectralResult and KrylovEvolver (M73).

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

from quantx_engine.contrib.spectral import (
    KrylovEvolver,
    KrylovResult,
    SpectralError,
    SpectralResult,
)
from quantx_engine.contrib.lanczos import LanczosSolver, SparseHamiltonian


# ===========================================================================
# SpectralResult
# ===========================================================================

class TestSpectralResult:
    def _mock_raw(self, n_k=3, n_w=10):
        omega = np.linspace(0.0, 4.0, n_w)
        kpoints = np.linspace(0, np.pi, n_k)
        # Gaussian peak at each k
        S_kw = np.array([
            np.exp(-((omega - kpoints[k]) ** 2) / 0.2)
            for k in range(n_k)
        ])
        return {"omega": omega.tolist(), "kpoints": kpoints.tolist(),
                "S_kw": S_kw.tolist()}

    def test_from_engine_output_basic(self):
        raw = self._mock_raw(n_k=3, n_w=10)
        result = SpectralResult.from_engine_output(raw)
        assert result.n_kpoints == 3
        assert result.n_omega == 10

    def test_omega_shape(self):
        raw = self._mock_raw()
        result = SpectralResult.from_engine_output(raw)
        assert result.omega.shape == (10,)

    def test_kpoints_shape(self):
        raw = self._mock_raw(n_k=5)
        result = SpectralResult.from_engine_output(raw)
        assert result.kpoints.shape == (5,)

    def test_S_kw_shape(self):
        raw = self._mock_raw(n_k=4, n_w=8)
        result = SpectralResult.from_engine_output(raw)
        assert result.S_kw.shape == (4, 8)

    def test_gap_parsed(self):
        raw = self._mock_raw()
        raw["gap"] = 1.23
        result = SpectralResult.from_engine_output(raw)
        assert result.gap == pytest.approx(1.23)

    def test_gap_none_when_absent(self):
        result = SpectralResult.from_engine_output(self._mock_raw())
        assert result.gap is None

    def test_gap_zero_preserved(self):
        """gap=0.0 (gapless system) must not be silently dropped by falsy-or."""
        raw = self._mock_raw()
        raw["gap"] = 0.0
        result = SpectralResult.from_engine_output(raw)
        assert result.gap is not None
        assert result.gap == pytest.approx(0.0)

    def test_gap_alias_spectral_gap(self):
        """spectral_gap alias is accepted."""
        raw = self._mock_raw()
        raw["spectral_gap"] = 2.5
        result = SpectralResult.from_engine_output(raw)
        assert result.gap == pytest.approx(2.5)

    def test_gap_alias_energy_gap(self):
        """energy_gap alias is accepted."""
        raw = self._mock_raw()
        raw["energy_gap"] = 0.0  # gapless via energy_gap alias
        result = SpectralResult.from_engine_output(raw)
        assert result.gap == pytest.approx(0.0)

    def test_alternate_field_names(self):
        """Accept 'frequencies', 'k_points', 'spectral_weight' aliases."""
        omega = [0.0, 1.0, 2.0]
        raw = {
            "frequencies": omega,
            "k_points": [0.0, 1.0],
            "spectral_weight": [[1.0, 2.0, 1.0], [0.5, 1.0, 0.5]],
        }
        result = SpectralResult.from_engine_output(raw)
        assert result.n_omega == 3
        assert result.n_kpoints == 2

    def test_1d_S_kw_promoted_to_2d(self):
        """S_kw as a flat 1D list is promoted to (1, n_omega)."""
        raw = {"omega": [0.0, 1.0, 2.0], "S_kw": [0.1, 0.5, 0.1], "kpoints": [0.0]}
        result = SpectralResult.from_engine_output(raw)
        assert result.S_kw.shape == (1, 3)

    def test_empty_raw_gives_zeros(self):
        """Empty raw dict gives zero-filled S_kw."""
        result = SpectralResult.from_engine_output({})
        assert result.S_kw.shape[0] >= 1
        assert np.all(result.S_kw == 0.0)

    def test_raw_stored(self):
        raw = self._mock_raw()
        result = SpectralResult.from_engine_output(raw)
        assert result.raw is raw

    def test_integrated_weight(self):
        """integrated_weight() returns array of length n_kpoints."""
        result = SpectralResult.from_engine_output(self._mock_raw(n_k=3, n_w=20))
        w = result.integrated_weight()
        assert w.shape == (3,)
        assert np.all(w >= 0.0)

    def test_peak_frequencies(self):
        """peak_frequencies() returns argmax of S_kw per k-point."""
        omega = np.array([0.0, 1.0, 2.0, 3.0])
        S_kw = np.array([[0.0, 1.0, 0.0, 0.0],  # peak at omega=1
                         [0.0, 0.0, 0.0, 1.0]])  # peak at omega=3
        raw = {"omega": omega.tolist(), "kpoints": [0.0, 1.0], "S_kw": S_kw.tolist()}
        result = SpectralResult.from_engine_output(raw)
        peaks = result.peak_frequencies()
        assert peaks[0] == pytest.approx(1.0)
        assert peaks[1] == pytest.approx(3.0)

    def test_direct_construction(self):
        """SpectralResult can be constructed directly without from_engine_output."""
        omega = np.array([0.0, 1.0])
        S_kw = np.array([[1.0, 0.5]])
        kpoints = np.array([0.0])
        result = SpectralResult(omega=omega, S_kw=S_kw, kpoints=kpoints, gap=0.5)
        assert result.gap == pytest.approx(0.5)
        assert result.n_kpoints == 1


# ===========================================================================
# KrylovEvolver — basic API
# ===========================================================================

class TestKrylovEvolverAPI:
    def test_invalid_m_raises(self):
        with pytest.raises(SpectralError, match="m must be >= 1"):
            KrylovEvolver(m=0)

    def test_invalid_dt_raises(self):
        with pytest.raises(SpectralError, match="dt must be > 0"):
            KrylovEvolver(dt=-0.1)

    def test_negative_t_raises(self):
        H = np.diag([1.0, 2.0])
        psi0 = np.array([1.0, 0.0], dtype=complex)
        with pytest.raises(SpectralError, match="t must be >= 0"):
            KrylovEvolver().evolve(H, psi0, t=-1.0)

    def test_negative_tau_raises(self):
        H = np.diag([1.0, 2.0])
        psi0 = np.array([1.0, 0.0], dtype=complex)
        with pytest.raises(SpectralError, match="tau must be >= 0"):
            KrylovEvolver().imaginary_evolve(H, psi0, tau=-1.0)

    def test_zero_t_returns_copy(self):
        """evolve with t=0 returns a copy of psi0."""
        H = np.diag([1.0, 2.0])
        psi0 = np.array([1.0, 0.0], dtype=complex)
        result = KrylovEvolver().evolve(H, psi0, t=0.0)
        np.testing.assert_allclose(result, psi0)

    def test_zero_tau_returns_normalised(self):
        H = np.diag([1.0, 2.0])
        psi0 = np.array([1.0, 0.0], dtype=complex)
        result = KrylovEvolver().imaginary_evolve(H, psi0, tau=0.0)
        np.testing.assert_allclose(result, psi0)

    def test_unsupported_H_raises(self):
        with pytest.raises(SpectralError, match="numpy array"):
            KrylovEvolver().evolve("not_a_matrix", np.array([1.0, 0.0], dtype=complex), 1.0)


# ===========================================================================
# KrylovEvolver — physics: real-time evolution
# ===========================================================================

class TestKrylovRealTime:
    def test_unit_norm_preserved(self):
        """||e^{-iHt}|ψ₀⟩|| = ||ψ₀|| for unitary evolution."""
        H = SparseHamiltonian.heisenberg_1d(3, J=1.0)
        rng = np.random.default_rng(42)
        psi0 = rng.standard_normal(8).astype(complex)
        psi0 /= np.linalg.norm(psi0)
        evolver = KrylovEvolver(m=20, dt=0.1)
        psi_t = evolver.evolve(H, psi0, t=1.0)
        assert abs(np.linalg.norm(psi_t) - 1.0) < 1e-6

    def test_free_rotation_sigma_z(self):
        """H = ω σ_z / 2: e^{-iHt}|+⟩ oscillates with known period."""
        # H = ω/2 * Z; ω = 1.0
        omega = 1.0
        H = np.array([[omega / 2, 0], [0, -omega / 2]], dtype=complex)
        # |+⟩ = (|0⟩ + |1⟩)/√2
        psi0 = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)
        evolver = KrylovEvolver(m=10, dt=0.05)
        t = math.pi / omega  # half-period: |+⟩ → |−⟩
        psi_t = evolver.evolve(H, psi0, t=t)
        # At t=π/ω: e^{-i π/2 σ_z}|+⟩ = e^{-iπ/4}(|0⟩ - |1⟩)/√2 up to phase
        expected_abs = 1.0 / math.sqrt(2)
        assert abs(abs(psi_t[0]) - expected_abs) < 1e-4
        assert abs(abs(psi_t[1]) - expected_abs) < 1e-4
        # Relative phase between components should flip sign
        phase_before = psi0[0] / psi0[1]
        phase_after = psi_t[0] / psi_t[1]
        assert abs(phase_after + phase_before) < 1e-4  # phases are negatives

    def test_rabi_oscillation_period(self):
        """H = Ω σ_x: population returns to |0⟩ after full Rabi cycle."""
        omega = 2.0
        H = np.array([[0, omega / 2], [omega / 2, 0]], dtype=complex)  # Ω/2 * X
        psi0 = np.array([1.0, 0.0], dtype=complex)
        evolver = KrylovEvolver(m=15, dt=0.05)
        T_rabi = 2 * math.pi / omega   # full period
        psi_T = evolver.evolve(H, psi0, t=T_rabi)
        # After one full Rabi cycle, population returns to |0⟩
        assert abs(abs(psi_T[0]) ** 2 - 1.0) < 1e-4

    def test_eigenstate_is_stationary(self):
        """An eigenstate |ψ⟩ with energy E evolves as e^{-iEt}|ψ⟩."""
        H = SparseHamiltonian.heisenberg_1d(3, J=1.0)
        # Get ground state from Lanczos
        result = LanczosSolver(k=1, seed=0).solve(H)
        E0 = result.eigenvalues[0]
        psi0 = result.eigenvectors[:, 0]
        evolver = KrylovEvolver(m=20, dt=0.1)
        t = 1.5
        psi_t = evolver.evolve(H, psi0, t=t)
        # Should differ only by global phase e^{-iE0*t}.
        # Use assert_allclose on the full vector to avoid division-by-zero
        # on near-zero components.
        expected = np.exp(-1j * E0 * t) * psi0
        np.testing.assert_allclose(psi_t, expected, atol=1e-5)

    def test_sparse_hamiltonian_input(self):
        """SparseHamiltonian is accepted as input."""
        H = SparseHamiltonian.ising_1d(3, J=1.0, h=0.5)
        psi0 = np.zeros(8, dtype=complex)
        psi0[0] = 1.0
        evolver = KrylovEvolver(m=15, dt=0.1)
        psi_t = evolver.evolve(H, psi0, t=0.5)
        assert abs(np.linalg.norm(psi_t) - 1.0) < 1e-6

    def test_autocorrelation_decays_from_neel(self):
        """Néel state autocorrelation |⟨Néel|ψ(t)⟩|² should decay from 1."""
        H = SparseHamiltonian.heisenberg_1d(4, J=1.0)
        neel = np.zeros(16, dtype=complex)
        neel[0b0101] = 1.0   # |↑↓↑↓⟩
        evolver = KrylovEvolver(m=20, dt=0.2)
        result = evolver.evolve_trajectory(H, neel, t_max=4.0, n_steps=10)
        assert result.autocorrelation[0] == pytest.approx(1.0, abs=1e-10)
        # Should have non-trivial time dependence
        assert not all(abs(a - 1.0) < 1e-3 for a in result.autocorrelation[1:])

    def test_trajectory_length(self):
        H = SparseHamiltonian.heisenberg_1d(2, J=1.0)
        psi0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        evolver = KrylovEvolver(m=10, dt=0.1)
        result = evolver.evolve_trajectory(H, psi0, t_max=1.0, n_steps=5)
        assert len(result) == 5
        assert len(result.times) == 5
        assert len(result.states) == 5
        assert result.times[0] == pytest.approx(0.0)


# ===========================================================================
# KrylovEvolver — physics: imaginary-time evolution
# ===========================================================================

class TestKrylovImaginaryTime:
    def test_converges_to_ground_state(self):
        """Imaginary-time evolution converges to the Lanczos ground state."""
        H = SparseHamiltonian.heisenberg_1d(4, J=1.0)
        rng = np.random.default_rng(7)
        psi0 = rng.standard_normal(16).astype(complex)
        psi0 /= np.linalg.norm(psi0)

        evolver = KrylovEvolver(m=20, dt=0.2)
        psi_gs = evolver.imaginary_evolve(H, psi0, tau=20.0)

        # Compare energy with Lanczos
        e0_krylov = float(np.real(psi_gs.conj() @ H.matvec(psi_gs)))
        e0_lanczos = LanczosSolver(k=1, seed=0).solve(H).ground_state_energy()
        assert abs(e0_krylov - e0_lanczos) < 1e-4

    def test_ground_state_norm_one(self):
        """imaginary_evolve result is normalised."""
        H = SparseHamiltonian.ising_1d(3, J=1.0, h=0.5)
        psi0 = np.ones(8, dtype=complex)
        psi0 /= np.linalg.norm(psi0)
        evolver = KrylovEvolver(m=15, dt=0.5)
        psi_gs = evolver.imaginary_evolve(H, psi0, tau=5.0)
        assert abs(np.linalg.norm(psi_gs) - 1.0) < 1e-10

    def test_longer_tau_more_accurate(self):
        """Larger tau gives lower ground-state energy (better convergence)."""
        H = SparseHamiltonian.heisenberg_1d(4, J=1.0)
        psi0 = np.ones(16, dtype=complex) / 4.0
        evolver = KrylovEvolver(m=20, dt=0.5)

        e_short = float(np.real(evolver.imaginary_evolve(H, psi0, tau=1.0).conj()
                                 @ H.matvec(evolver.imaginary_evolve(H, psi0, tau=1.0))))
        e_long = float(np.real(evolver.imaginary_evolve(H, psi0, tau=10.0).conj()
                                @ H.matvec(evolver.imaginary_evolve(H, psi0, tau=10.0))))
        assert e_long <= e_short + 1e-6  # longer τ gives lower or equal energy

    def test_zero_norm_psi_raises(self):
        H = np.diag([1.0, 2.0])
        psi0 = np.zeros(2, dtype=complex)
        with pytest.raises(SpectralError, match="near-zero norm"):
            KrylovEvolver().imaginary_evolve(H, psi0, tau=1.0)


# ===========================================================================
# KrylovResult
# ===========================================================================

class TestKrylovResult:
    def test_len(self):
        result = KrylovResult(
            times=[0.0, 1.0, 2.0],
            states=[np.zeros(4)] * 3,
            autocorrelation=[1.0, 0.8, 0.6],
        )
        assert len(result) == 3

    def test_n_steps_raises_if_too_small(self):
        H = SparseHamiltonian.heisenberg_1d(2)
        psi0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        evolver = KrylovEvolver(m=10)
        with pytest.raises(SpectralError, match="n_steps must be >= 2"):
            evolver.evolve_trajectory(H, psi0, t_max=1.0, n_steps=1)
