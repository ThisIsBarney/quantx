"""Tests for finite-size scaling and quantum phase-transition tools (M74).

All tests require numpy; they are skipped otherwise.
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

from quantx_engine.finite_size import (
    EntanglementResult,
    FiniteSizeError,
    FiniteSizeScaling,
    ScalingResult,
    binder_cumulant,
    entanglement_scaling,
)


# ===========================================================================
# binder_cumulant
# ===========================================================================

class TestBinderCumulant:
    def test_gaussian_samples_near_zero(self):
        """Gaussian samples → U ≈ 0 (disordered phase)."""
        rng = np.random.default_rng(0)
        samples = rng.normal(0.0, 1.0, 50_000)
        # Theoretical: <m^4> = 3*sigma^4, <m^2> = sigma^2, U = 0
        u = binder_cumulant(samples)
        assert abs(u) < 0.02

    def test_bimodal_samples_near_two_thirds(self):
        """Bimodal (perfectly ordered) samples → U ≈ 2/3."""
        # m = ±1 with equal probability
        rng = np.random.default_rng(1)
        signs = rng.choice([-1.0, 1.0], size=20_000)
        u = binder_cumulant(signs)
        assert abs(u - 2.0 / 3.0) < 0.02

    def test_constant_samples(self):
        """All samples equal → <m^4> = m^4, <m^2> = m^2, U = 2/3."""
        samples = np.ones(1000) * 3.7
        u = binder_cumulant(samples)
        assert abs(u - 2.0 / 3.0) < 1e-10

    def test_range_in_zero_two_thirds(self):
        """Binder cumulant should be in [0, 2/3] for physical distributions."""
        rng = np.random.default_rng(2)
        for _ in range(10):
            # Mixture of bimodal and Gaussian
            m0 = rng.uniform(0.5, 2.0)
            noise = rng.normal(0.0, 0.3, 5000)
            samples = m0 * rng.choice([-1.0, 1.0], 5000) + noise
            u = binder_cumulant(samples)
            assert -0.05 <= u <= 2.0 / 3.0 + 0.05  # small numerical tolerance

    def test_empty_raises(self):
        with pytest.raises(FiniteSizeError, match="at least 2 samples"):
            binder_cumulant([])

    def test_single_element_raises(self):
        with pytest.raises(FiniteSizeError, match="at least 2 samples"):
            binder_cumulant([1.0])

    def test_zero_order_parameter_raises(self):
        """All-zero samples → ⟨m²⟩ = 0 → raises."""
        with pytest.raises(FiniteSizeError, match="order parameter is identically zero"):
            binder_cumulant([0.0, 0.0, 0.0])

    def test_2d_input_ravelled(self):
        """2D input is ravelled before processing."""
        samples = np.ones((10, 10))  # constant 1 → U = 2/3
        u = binder_cumulant(samples)
        assert abs(u - 2.0 / 3.0) < 1e-10

    def test_known_value_exact(self):
        """Manual computation: samples = [1, -1, 1, -1] → U = 2/3."""
        samples = [1.0, -1.0, 1.0, -1.0]
        # <m^2> = 1.0, <m^4> = 1.0, U = 1 - 1/3 = 2/3
        u = binder_cumulant(samples)
        assert abs(u - 2.0 / 3.0) < 1e-12

    def test_asymmetric_distribution(self):
        """Positive-only samples (biased): U = 2/3 when all same sign."""
        samples = np.full(100, 2.5)
        u = binder_cumulant(samples)
        assert abs(u - 2.0 / 3.0) < 1e-10


# ===========================================================================
# entanglement_scaling
# ===========================================================================

class TestEntanglementScaling:
    # ── Basic correctness ─────────────────────────────────────────────────

    def test_product_state_zero_entropy(self):
        """Product state |0⟩⊗L → S(A) = 0 for all A."""
        for L in [2, 3, 4]:
            psi = np.zeros(2 ** L, dtype=complex)
            psi[0] = 1.0
            result = entanglement_scaling(psi, L)
            np.testing.assert_allclose(result.entropy, 0.0, atol=1e-10)

    def test_bell_state_entropy_ln2(self):
        """Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2 → S(1) = ln(2)."""
        psi = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / math.sqrt(2)
        result = entanglement_scaling(psi, L=2)
        assert result.subsystem_sizes.tolist() == [1]
        assert result.entropy[0] == pytest.approx(math.log(2), abs=1e-10)

    def test_ghz_state_entropy_ln2_all_bipartitions(self):
        """GHZ state → S(A) = ln(2) for every bipartition."""
        for L in [3, 4, 5]:
            psi = np.zeros(2 ** L, dtype=complex)
            psi[0] = 1.0 / math.sqrt(2)
            psi[-1] = 1.0 / math.sqrt(2)
            result = entanglement_scaling(psi, L)
            np.testing.assert_allclose(result.entropy, math.log(2), atol=1e-10)

    def test_subsystem_sizes_correct(self):
        """subsystem_sizes = [1, 2, ..., L-1]."""
        psi = np.ones(8, dtype=complex) / math.sqrt(8)
        result = entanglement_scaling(psi, L=3)
        np.testing.assert_array_equal(result.subsystem_sizes, [1, 2])

    def test_normalisation_applied(self):
        """Un-normalised psi gives the same entropy as normalised psi."""
        psi = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex)  # norm = sqrt(2)
        result_unnorm = entanglement_scaling(psi, L=2)
        psi_norm = psi / math.sqrt(2)
        result_norm = entanglement_scaling(psi_norm, L=2)
        np.testing.assert_allclose(result_unnorm.entropy, result_norm.entropy, atol=1e-12)

    # ── Shape and dimension checks ────────────────────────────────────────

    def test_wrong_dimension_raises(self):
        psi = np.ones(5, dtype=complex)
        with pytest.raises(FiniteSizeError, match="expected d\\^L"):
            entanglement_scaling(psi, L=3)  # expects 8

    def test_near_zero_norm_raises(self):
        psi = np.zeros(4, dtype=complex)
        with pytest.raises(FiniteSizeError, match="near-zero norm"):
            entanglement_scaling(psi, L=2)

    def test_invalid_L_raises(self):
        with pytest.raises(FiniteSizeError, match="L must be >= 1"):
            entanglement_scaling(np.array([1.0]), L=0)

    def test_invalid_d_raises(self):
        with pytest.raises(FiniteSizeError, match="d must be >= 2"):
            entanglement_scaling(np.array([1.0]), L=1, d=1)

    # ── CFT fit ───────────────────────────────────────────────────────────

    def test_central_charge_none_for_small_L(self):
        """L < 4 → no CFT fit → central_charge is None."""
        psi = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / math.sqrt(2)
        result = entanglement_scaling(psi, L=2)
        assert result.central_charge is None
        assert result.fit_residual is None

    def test_central_charge_returned_for_large_L(self):
        """L >= 4 → central_charge is not None."""
        # Use a random entangled state — we only check a value is returned
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(2 ** 4).astype(complex)
        psi /= np.linalg.norm(psi)
        result = entanglement_scaling(psi, L=4)
        assert result.central_charge is not None
        assert result.fit_residual is not None

    def test_product_state_central_charge_near_zero(self):
        """For a product state S(A) = 0 everywhere → c ≈ 0."""
        psi = np.zeros(2 ** 4, dtype=complex)
        psi[0] = 1.0
        result = entanglement_scaling(psi, L=4)
        # S = 0 → 0 = c/3 * log_xi + s0 → c = 0 (and s0 = 0)
        assert result.central_charge == pytest.approx(0.0, abs=1e-8)

    def test_max_entropy_property(self):
        """max_entropy returns the maximum S(A) over all A."""
        psi = np.zeros(2 ** 4, dtype=complex)
        psi[0] = 1.0 / math.sqrt(2)
        psi[-1] = 1.0 / math.sqrt(2)
        result = entanglement_scaling(psi, L=4)
        assert result.max_entropy == pytest.approx(math.log(2), abs=1e-10)

    def test_qutrit_d3(self):
        """d=3 local dimension: |0⟩⊗2 has zero entropy."""
        psi = np.zeros(9, dtype=complex)
        psi[0] = 1.0
        result = entanglement_scaling(psi, L=2, d=3)
        np.testing.assert_allclose(result.entropy, 0.0, atol=1e-10)

    def test_heisenberg_central_charge_is_finite(self):
        """Heisenberg chain: entanglement_scaling runs and returns a finite c.

        Note: the CFT formula S = (c/3) ln[...] assumes periodic boundary
        conditions.  The Lanczos ground state uses open BC, which introduces
        staggered finite-size corrections that make the extracted c
        unreliable for small L.  This test only verifies that the function
        runs without error and returns a finite central-charge estimate.
        """
        from quantx_engine.lanczos import LanczosSolver, SparseHamiltonian
        L = 6
        H = SparseHamiltonian.heisenberg_1d(L, J=1.0)
        result_lanczos = LanczosSolver(k=1, seed=0).solve(H)
        psi_gs = result_lanczos.eigenvectors[:, 0]
        result = entanglement_scaling(psi_gs, L=L)
        assert result.central_charge is not None
        assert math.isfinite(result.central_charge)
        # Entropies should be non-negative
        assert np.all(result.entropy >= -1e-12)


# ===========================================================================
# FiniteSizeScaling — construction and validation
# ===========================================================================

class TestFiniteSizeScalingConstruction:
    def _make_fss(self):
        """Simple FSS with known crossing at J=1.0."""
        params = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
        # L=8: rising curve, L=16: falling curve, cross at J=1.0
        u8  = 0.2 * (params - 0.5) + 0.2    # 0.20 → 0.40
        u16 = -0.2 * (params - 1.5) + 0.2   # 0.40 → 0.20
        binder = np.array([u8, u16])
        return FiniteSizeScaling([8, 16], params, binder)

    def test_construction_succeeds(self):
        fss = self._make_fss()
        assert fss.sizes == [8, 16]
        assert len(fss.params) == 5

    def test_too_few_sizes_raises(self):
        with pytest.raises(FiniteSizeError, match="at least 2 system sizes"):
            FiniteSizeScaling([8], np.linspace(0, 2, 5), np.zeros((1, 5)))

    def test_too_few_params_raises(self):
        with pytest.raises(FiniteSizeError, match="at least 2 parameter values"):
            FiniteSizeScaling([8, 16], [1.0], np.zeros((2, 1)))

    def test_shape_mismatch_raises(self):
        with pytest.raises(FiniteSizeError, match="does not match"):
            FiniteSizeScaling([8, 16], np.linspace(0, 2, 5), np.zeros((2, 4)))

    def test_binder_crossings_returns_list(self):
        fss = self._make_fss()
        crossings = fss.binder_crossings()
        assert isinstance(crossings, list)
        assert len(crossings) == 1

    def test_crossing_at_known_point(self):
        """Curves constructed to cross at J=1.0."""
        fss = self._make_fss()
        crossings = fss.binder_crossings()
        assert crossings[0] == pytest.approx(1.0, abs=1e-10)

    def test_no_crossings_in_range(self):
        """Both curves monotonically increasing → no crossing."""
        params = np.linspace(0, 2, 10)
        binder = np.array([0.1 * params, 0.2 * params])
        fss = FiniteSizeScaling([8, 16], params, binder)
        assert fss.binder_crossings() == []

    def test_fit_critical_point_basic(self):
        fss = self._make_fss()
        result = fss.fit_critical_point()
        assert isinstance(result, ScalingResult)
        assert result.critical_point == pytest.approx(1.0, abs=1e-10)
        assert result.critical_point_std == pytest.approx(0.0, abs=1e-10)

    def test_fit_critical_point_no_crossings_raises(self):
        params = np.linspace(0, 2, 10)
        binder = np.array([0.1 * params, 0.2 * params])
        fss = FiniteSizeScaling([8, 16], params, binder)
        with pytest.raises(FiniteSizeError, match="No Binder cumulant crossings"):
            fss.fit_critical_point()

    def test_fit_critical_point_multiple_sizes(self):
        """4 sizes → 3 crossings → averaged critical point."""
        J_c = 1.0
        nu = 1.0
        params = np.linspace(0.7, 1.3, 40)
        sizes = [8, 16, 32, 64]

        def logistic(x):
            return (2.0 / 3.0) / (1.0 + np.exp(-4.0 * x))

        binder = np.array([logistic((params - J_c) * L ** (1.0 / nu)) for L in sizes])
        fss = FiniteSizeScaling(sizes, params, binder)
        result = fss.fit_critical_point()
        assert result.critical_point == pytest.approx(J_c, abs=0.01)
        assert len(result.binder_crossings) == 3

    def test_raw_data_stored(self):
        fss = self._make_fss()
        result = fss.fit_critical_point()
        assert "sizes" in result.raw
        assert "params" in result.raw
        assert "binder" in result.raw


# ===========================================================================
# FiniteSizeScaling.data_collapse
# ===========================================================================

class TestDataCollapse:
    def _make_fss_logistic(self, J_c=1.0, nu=1.0):
        params = np.linspace(0.7, 1.3, 40)
        sizes = [8, 16, 32]

        def logistic(x):
            return (2.0 / 3.0) / (1.0 + np.exp(-4.0 * x))

        binder = np.array([logistic((params - J_c) * L ** (1.0 / nu)) for L in sizes])
        return FiniteSizeScaling(sizes, params, binder), J_c, nu

    def test_cost_at_true_params_is_small(self):
        """data_collapse cost near zero at true (J_c, ν)."""
        fss, J_c, nu = self._make_fss_logistic()
        cost = fss.data_collapse(J_c, nu)
        assert cost < 1e-5

    def test_cost_minimized_at_true_nu(self):
        """data_collapse cost is minimized near the true ν = 1.0."""
        fss, J_c, nu = self._make_fss_logistic()
        cost_true = fss.data_collapse(J_c, nu=1.0)
        cost_wrong = fss.data_collapse(J_c, nu=0.5)
        assert cost_true < cost_wrong

    def test_cost_at_wrong_jc_is_larger(self):
        """Using wrong J_c increases the data collapse cost."""
        fss, J_c, nu = self._make_fss_logistic()
        cost_true = fss.data_collapse(J_c, nu)
        cost_wrong = fss.data_collapse(J_c + 0.2, nu)
        assert cost_true < cost_wrong

    def test_invalid_nu_raises(self):
        fss, J_c, _ = self._make_fss_logistic()
        with pytest.raises(FiniteSizeError, match="nu must be > 0"):
            fss.data_collapse(J_c, nu=-1.0)

    def test_observable_wrong_shape_raises(self):
        fss, J_c, nu = self._make_fss_logistic()
        wrong_obs = np.zeros((2, 40))  # wrong n_sizes
        with pytest.raises(FiniteSizeError, match="shape"):
            fss.data_collapse(J_c, nu, observable=wrong_obs)

    def test_returns_float(self):
        fss, J_c, nu = self._make_fss_logistic()
        cost = fss.data_collapse(J_c, nu)
        assert isinstance(cost, float)
        assert cost >= 0.0

    def test_custom_observable(self):
        """Can pass custom observable array."""
        fss, J_c, nu = self._make_fss_logistic()
        # Use the Binder cumulant itself as custom observable
        cost = fss.data_collapse(J_c, nu, observable=fss.binder, beta_over_nu=0.0)
        assert cost < 1e-5


# ===========================================================================
# Integration: binder_cumulant + FiniteSizeScaling
# ===========================================================================

class TestIntegration:
    def test_mc_samples_to_fss_pipeline(self):
        """End-to-end: compute Binder from MC samples, run FSS."""
        rng = np.random.default_rng(99)
        J_c = 1.0
        sizes = [16, 32]
        params = np.linspace(0.6, 1.4, 15)

        binder_rows = []
        for L in sizes:
            row = []
            for J in params:
                # Simulate order parameter: bimodal near J > J_c, Gaussian near J < J_c
                # Simple model: m ~ N(m0(J), 0.1) where m0 = max(0, J - J_c)
                m0 = max(0.0, (J - J_c) * L ** 0.5)
                if m0 > 0.3:
                    samples = rng.choice([-1.0, 1.0], size=500) * m0
                else:
                    samples = rng.normal(0.0, 0.5, 500)
                row.append(binder_cumulant(samples))
            binder_rows.append(row)

        binder = np.array(binder_rows)
        fss = FiniteSizeScaling(sizes, params, binder)
        result = fss.fit_critical_point()
        # Critical point should be somewhere near J_c = 1.0
        assert 0.5 < result.critical_point < 1.5
