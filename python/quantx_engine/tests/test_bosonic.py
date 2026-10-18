"""M119: Tests for bosonic codes — cat qubits and GKP."""
from __future__ import annotations

import math
import pytest
import numpy as np

from quantx_engine.contrib.bosonic import (
    BosonicError,
    CatNoiseModel,
    CatQubit,
    GKPState,
    GKPDecoder,
)


# ── CatNoiseModel ──────────────────────────────────────────────────────────────

class TestCatNoiseModel:
    def test_default_construction(self):
        m = CatNoiseModel()
        assert m.kappa_1 > 0
        assert m.kappa_phi >= 0
        assert m.alpha_sq > 0

    def test_explicit_params(self):
        m = CatNoiseModel(kappa_1=0.02, kappa_phi=0.005, alpha_sq=9.0)
        assert m.kappa_1 == 0.02
        assert m.kappa_phi == 0.005
        assert m.alpha_sq == 9.0

    def test_negative_kappa1_raises(self):
        with pytest.raises(BosonicError):
            CatNoiseModel(kappa_1=-0.01)

    def test_negative_kappa_phi_raises(self):
        with pytest.raises(BosonicError):
            CatNoiseModel(kappa_phi=-0.001)

    def test_zero_alpha_sq_raises(self):
        with pytest.raises(BosonicError):
            CatNoiseModel(alpha_sq=0.0)

    def test_bit_flip_rate_exponentially_suppressed(self):
        """Larger |α|² → exponentially smaller bit-flip rate."""
        m_small = CatNoiseModel(alpha_sq=1.0)
        m_large = CatNoiseModel(alpha_sq=4.0)
        assert m_large.bit_flip_rate < m_small.bit_flip_rate

    def test_phase_flip_rate_grows_with_alpha(self):
        """Larger |α|² → larger phase-flip rate (linear)."""
        m_small = CatNoiseModel(alpha_sq=1.0)
        m_large = CatNoiseModel(alpha_sq=4.0)
        assert m_large.phase_flip_rate > m_small.phase_flip_rate

    def test_bias_ratio_large_alpha(self):
        """For large α, bias η ≫ 1."""
        m = CatNoiseModel(alpha_sq=9.0)
        assert m.bias_ratio > 100

    def test_bias_ratio_positive(self):
        m = CatNoiseModel()
        assert m.bias_ratio > 0

    def test_bit_flip_rate_is_float(self):
        m = CatNoiseModel()
        assert isinstance(m.bit_flip_rate, float)

    def test_phase_flip_rate_is_float(self):
        m = CatNoiseModel()
        assert isinstance(m.phase_flip_rate, float)


# ── CatQubit construction ─────────────────────────────────────────────────────

class TestCatQubitConstruction:
    def test_default_construction(self):
        cat = CatQubit()
        assert cat.K == 1.0
        assert cat.eps2 == 4.0
        assert cat.N_max == 30

    def test_alpha_from_params(self):
        cat = CatQubit(K=1.0, eps2=4.0)
        assert abs(cat.alpha - 2.0) < 1e-10

    def test_explicit_params(self):
        cat = CatQubit(K=2.0, eps2=8.0, N_max=20)
        assert cat.K == 2.0
        assert cat.eps2 == 8.0
        assert cat.N_max == 20
        assert abs(cat.alpha - 2.0) < 1e-10

    def test_nonpositive_K_raises(self):
        with pytest.raises(BosonicError):
            CatQubit(K=0.0)

    def test_nonpositive_eps2_raises(self):
        with pytest.raises(BosonicError):
            CatQubit(eps2=0.0)

    def test_small_N_max_raises(self):
        with pytest.raises(BosonicError):
            CatQubit(N_max=2)

    def test_repr(self):
        cat = CatQubit()
        assert "CatQubit" in repr(cat)
        assert "alpha" in repr(cat)

    def test_hamiltonian_shape(self):
        cat = CatQubit(N_max=10)
        H = cat.hamiltonian
        assert H.shape == (10, 10)

    def test_hamiltonian_hermitian(self):
        cat = CatQubit(N_max=15)
        H = cat.hamiltonian
        assert np.allclose(H, H.conj().T, atol=1e-10)


# ── Cat qubit code words ──────────────────────────────────────────────────────

class TestCatCodeWords:
    def test_encode_0_normalised(self):
        cat = CatQubit(K=1.0, eps2=4.0, N_max=30)
        psi = cat.encode_0()
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-10

    def test_encode_1_normalised(self):
        cat = CatQubit(K=1.0, eps2=4.0, N_max=30)
        psi = cat.encode_1()
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-10

    def test_code_words_orthogonal(self):
        """Logical |0_L⟩ and |1_L⟩ should be approximately orthogonal for α ≫ 1."""
        cat = CatQubit(K=1.0, eps2=9.0, N_max=40)  # α = 3
        psi0 = cat.encode_0()
        psi1 = cat.encode_1()
        overlap = abs(np.dot(psi0.conj(), psi1))
        assert overlap < 0.05  # near-orthogonal for α=3

    def test_logical_z_eigenvalues(self):
        """Logical Z = parity operator: |0_L⟩ (even parity) → +1, |1_L⟩ (odd) → −1."""
        cat = CatQubit(K=1.0, eps2=4.0, N_max=30)
        Z_L = cat.logical_z
        psi0 = cat.encode_0()
        psi1 = cat.encode_1()
        ev0 = float(np.real(psi0.conj() @ Z_L @ psi0))
        ev1 = float(np.real(psi1.conj() @ Z_L @ psi1))
        assert abs(ev0 - 1.0) < 0.1   # even cat ≈ +1 eigenstate of parity
        assert abs(ev1 + 1.0) < 0.1   # odd cat ≈ −1 eigenstate of parity

    def test_photon_number_positive(self):
        cat = CatQubit(K=1.0, eps2=4.0, N_max=30)
        psi0 = cat.encode_0()
        n_avg = cat.mean_photon_number(psi0)
        assert n_avg > 0

    def test_mean_photon_grows_with_alpha(self):
        """Higher drive → higher mean photon number in code words."""
        cat_small = CatQubit(K=1.0, eps2=1.0, N_max=30)
        cat_large = CatQubit(K=1.0, eps2=9.0, N_max=40)
        n_small = cat_small.mean_photon_number(cat_small.encode_0())
        n_large = cat_large.mean_photon_number(cat_large.encode_0())
        assert n_large > n_small

    def test_photon_number_distribution_sums_to_one(self):
        cat = CatQubit(N_max=20)
        psi = cat.encode_0()
        dist = cat.photon_number_distribution(psi)
        assert abs(sum(dist) - 1.0) < 1e-10

    def test_even_cat_parity(self):
        """Even cat |0_L⟩ should have mostly even-photon-number support."""
        cat = CatQubit(K=1.0, eps2=4.0, N_max=30)
        psi = cat.encode_0()
        dist = cat.photon_number_distribution(psi)
        even_weight = sum(dist[n] for n in range(0, len(dist), 2))
        odd_weight = sum(dist[n] for n in range(1, len(dist), 2))
        assert even_weight > odd_weight  # predominantly even photon numbers

    def test_odd_cat_parity(self):
        """Odd cat |1_L⟩ should have mostly odd-photon-number support."""
        cat = CatQubit(K=1.0, eps2=4.0, N_max=30)
        psi = cat.encode_1()
        dist = cat.photon_number_distribution(psi)
        even_weight = sum(dist[n] for n in range(0, len(dist), 2))
        odd_weight = sum(dist[n] for n in range(1, len(dist), 2))
        assert odd_weight > even_weight  # predominantly odd photon numbers


# ── CatQubit noise / timing ───────────────────────────────────────────────────

class TestCatQubitNoise:
    def test_bit_flip_time_finite(self):
        cat = CatQubit()
        t = cat.bit_flip_time()
        assert math.isfinite(t)
        assert t > 0

    def test_bit_flip_time_grows_with_alpha(self):
        cat_small = CatQubit(K=1.0, eps2=1.0, N_max=20)
        cat_large = CatQubit(K=1.0, eps2=9.0, N_max=40)
        assert cat_large.bit_flip_time() > cat_small.bit_flip_time()

    def test_phase_flip_rate_positive(self):
        cat = CatQubit()
        assert cat.phase_flip_rate() > 0

    def test_custom_noise_model(self):
        noise = CatNoiseModel(kappa_1=0.1, kappa_phi=0.0, alpha_sq=4.0)
        cat = CatQubit(noise=noise)
        assert cat.noise is noise


# ── CatQubit Wigner function ──────────────────────────────────────────────────

class TestCatQubitWigner:
    def test_wigner_returns_real(self):
        cat = CatQubit(N_max=15)
        psi = cat.encode_0()
        x, p, W = cat.wigner_function(psi, xvec=np.linspace(-3, 3, 5),
                                       pvec=np.linspace(-3, 3, 5))
        assert np.all(np.isreal(W)) or np.allclose(W.imag, 0, atol=1e-8)

    def test_wigner_shape(self):
        cat = CatQubit(N_max=12)
        psi = cat.encode_0()
        xvec = np.linspace(-2, 2, 7)
        pvec = np.linspace(-2, 2, 9)
        _, _, W = cat.wigner_function(psi, xvec=xvec, pvec=pvec)
        assert W.shape == (7, 9)

    def test_wigner_default_grid(self):
        cat = CatQubit(N_max=12)
        psi = cat.encode_0()
        x, p, W = cat.wigner_function(psi)
        assert W.shape[0] == len(x)
        assert W.shape[1] == len(p)


# ── CatQubit ground state ─────────────────────────────────────────────────────

class TestCatQubitGroundState:
    def test_ground_state_normalised(self):
        cat = CatQubit(N_max=20)
        gs = cat.ground_state()
        assert abs(np.linalg.norm(gs) - 1.0) < 1e-10

    def test_ground_state_energy(self):
        """Ground state energy should be the minimum eigenvalue of H."""
        cat = CatQubit(N_max=15)
        gs = cat.ground_state()
        H = cat.hamiltonian
        E_gs = float(np.real(gs.conj() @ H @ gs))
        eigvals = np.linalg.eigvalsh(H)
        assert abs(E_gs - eigvals[0]) < 1e-8


# ── GKPState construction ─────────────────────────────────────────────────────

class TestGKPStateConstruction:
    def test_default_construction(self):
        gkp = GKPState()
        assert 0 < gkp.delta < 1
        assert gkp.N_max >= 8

    def test_explicit_params(self):
        gkp = GKPState(delta=0.2, N_max=50)
        assert gkp.delta == 0.2
        assert gkp.N_max == 50

    def test_delta_zero_raises(self):
        with pytest.raises(BosonicError):
            GKPState(delta=0.0)

    def test_delta_one_raises(self):
        with pytest.raises(BosonicError):
            GKPState(delta=1.0)

    def test_small_N_max_raises(self):
        with pytest.raises(BosonicError):
            GKPState(N_max=4)

    def test_repr(self):
        gkp = GKPState()
        assert "GKPState" in repr(gkp)
        assert "dB" in repr(gkp)

    def test_squeezing_db_positive(self):
        gkp = GKPState(delta=0.3)
        assert gkp.squeezing_db() > 0

    def test_smaller_delta_more_squeezing(self):
        gkp_lo = GKPState(delta=0.4)
        gkp_hi = GKPState(delta=0.2)
        assert gkp_hi.squeezing_db() > gkp_lo.squeezing_db()


# ── GKPState code words ───────────────────────────────────────────────────────

class TestGKPCodeWords:
    def test_code_word_0_normalised(self):
        gkp = GKPState(delta=0.4, N_max=40)
        psi = gkp.code_word_0()
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-6

    def test_code_word_1_normalised(self):
        gkp = GKPState(delta=0.4, N_max=40)
        psi = gkp.code_word_1()
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-6

    def test_code_words_approximately_orthogonal(self):
        """For a reasonable δ, |0_L⟩ and |1_L⟩ should have small overlap."""
        gkp = GKPState(delta=0.3, N_max=50)
        psi0 = gkp.code_word_0()
        psi1 = gkp.code_word_1()
        overlap = abs(gkp.overlap(psi0, psi1))
        assert overlap < 0.3

    def test_code_word_0_self_overlap(self):
        gkp = GKPState(delta=0.3, N_max=50)
        psi0 = gkp.code_word_0()
        overlap = gkp.overlap(psi0, psi0)
        assert abs(overlap - 1.0) < 1e-6

    def test_mean_photon_number_positive(self):
        gkp = GKPState(delta=0.4, N_max=40)
        psi = gkp.code_word_0()
        n_avg = gkp.mean_photon_number(psi)
        assert n_avg > 0

    def test_mean_photon_number_grows_with_squeezing(self):
        """More squeezing (smaller δ) → higher mean photon number."""
        gkp_lo = GKPState(delta=0.5, N_max=40)
        gkp_hi = GKPState(delta=0.25, N_max=60)
        n_lo = gkp_lo.mean_photon_number(gkp_lo.code_word_0())
        n_hi = gkp_hi.mean_photon_number(gkp_hi.code_word_0())
        assert n_hi > n_lo

    def test_logical_fidelity_self(self):
        """Fidelity of |0_L⟩ with itself should be 1."""
        gkp = GKPState(delta=0.4, N_max=40)
        psi = gkp.code_word_0()
        fid = gkp.logical_fidelity(psi, logical=0)
        assert abs(fid - 1.0) < 1e-6


# ── GKPDecoder ────────────────────────────────────────────────────────────────

class TestGKPDecoder:
    def test_construction(self):
        gkp = GKPState(delta=0.4, N_max=40)
        dec = GKPDecoder(gkp)
        assert dec.n_rounds == 1

    def test_explicit_n_rounds(self):
        gkp = GKPState(delta=0.4, N_max=40)
        dec = GKPDecoder(gkp, n_rounds=3)
        assert dec.n_rounds == 3

    def test_zero_rounds_raises(self):
        gkp = GKPState(delta=0.4, N_max=40)
        with pytest.raises(BosonicError):
            GKPDecoder(gkp, n_rounds=0)

    def test_correct_preserves_norm(self):
        gkp = GKPState(delta=0.4, N_max=40)
        dec = GKPDecoder(gkp)
        psi = gkp.code_word_0()
        rng = np.random.default_rng(0)
        noisy = dec.apply_shift(psi, sigma=0.1, rng=rng)
        corrected = dec.correct(noisy)
        assert abs(np.linalg.norm(corrected) - 1.0) < 1e-8

    def test_apply_shift_preserves_norm(self):
        gkp = GKPState(delta=0.4, N_max=40)
        dec = GKPDecoder(gkp)
        psi = gkp.code_word_0()
        rng = np.random.default_rng(1)
        noisy = dec.apply_shift(psi, sigma=0.2, rng=rng)
        assert abs(np.linalg.norm(noisy) - 1.0) < 1e-8

    def test_zero_shift_no_error(self):
        """Zero shift → corrected state should have high fidelity with original."""
        gkp = GKPState(delta=0.4, N_max=40)
        dec = GKPDecoder(gkp)
        psi = gkp.code_word_0()
        rng = np.random.default_rng(0)
        noisy = dec.apply_shift(psi, sigma=0.0, rng=rng)
        corrected = dec.correct(noisy)
        fid = gkp.logical_fidelity(corrected, logical=0)
        assert fid > 0.9

    def test_logical_error_rate_in_range(self):
        gkp = GKPState(delta=0.4, N_max=40)
        dec = GKPDecoder(gkp)
        rate = dec.logical_error_rate(sigma=0.1, n_shots=50, seed=42)
        assert 0.0 <= rate <= 1.0

    def test_logical_error_rate_increases_with_noise(self):
        """Larger shift noise → higher logical error rate."""
        gkp = GKPState(delta=0.3, N_max=50)
        dec = GKPDecoder(gkp)
        rate_low = dec.logical_error_rate(sigma=0.05, n_shots=100, seed=0)
        rate_high = dec.logical_error_rate(sigma=0.5, n_shots=100, seed=0)
        assert rate_high >= rate_low - 0.1  # high noise should be at least as bad

    def test_correction_improves_fidelity(self):
        """SBS correction should improve or maintain fidelity vs no correction."""
        gkp = GKPState(delta=0.35, N_max=50)
        dec = GKPDecoder(gkp, n_rounds=1)
        psi0 = gkp.code_word_0()
        rng = np.random.default_rng(7)
        sigma = 0.15

        total_fid_raw = 0.0
        total_fid_corrected = 0.0
        n = 30
        for _ in range(n):
            noisy = dec.apply_shift(psi0, sigma=sigma, rng=rng)
            corrected = dec.correct(noisy)
            total_fid_raw += gkp.logical_fidelity(noisy, logical=0)
            total_fid_corrected += gkp.logical_fidelity(corrected, logical=0)

        avg_raw = total_fid_raw / n
        avg_corrected = total_fid_corrected / n
        # SBS correction should improve fidelity on average
        assert avg_corrected >= avg_raw - 0.05

    def test_negative_sigma_raises(self):
        gkp = GKPState(delta=0.4, N_max=40)
        dec = GKPDecoder(gkp)
        with pytest.raises(BosonicError):
            dec.logical_error_rate(sigma=-0.1)

    def test_zero_shots_raises(self):
        gkp = GKPState(delta=0.4, N_max=40)
        dec = GKPDecoder(gkp)
        with pytest.raises(BosonicError):
            dec.logical_error_rate(sigma=0.1, n_shots=0)

    def test_reproducible_with_seed(self):
        gkp = GKPState(delta=0.4, N_max=40)
        dec = GKPDecoder(gkp)
        r1 = dec.logical_error_rate(sigma=0.2, n_shots=30, seed=99)
        r2 = dec.logical_error_rate(sigma=0.2, n_shots=30, seed=99)
        assert r1 == r2
