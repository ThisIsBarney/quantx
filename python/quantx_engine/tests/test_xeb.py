"""Tests for Cross-Entropy Benchmarking (M82)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quantx_engine.contrib.xeb import CrossEntropyBenchmarking, XEBError, XEBResult


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestCEBConstruction:
    def test_default_construction(self):
        xeb = CrossEntropyBenchmarking()
        assert xeb.n_qubits == 2
        assert xeb.p_error == pytest.approx(0.001)
        assert xeb.seed is None

    def test_custom_params(self):
        xeb = CrossEntropyBenchmarking(n_qubits=3, p_error=0.02, seed=7)
        assert xeb.n_qubits == 3
        assert xeb.p_error == pytest.approx(0.02)
        assert xeb.seed == 7

    def test_invalid_n_qubits_too_small(self):
        with pytest.raises(XEBError, match="n_qubits must be >= 2"):
            CrossEntropyBenchmarking(n_qubits=1)

    def test_invalid_n_qubits_zero(self):
        with pytest.raises(XEBError):
            CrossEntropyBenchmarking(n_qubits=0)

    def test_invalid_p_error_negative(self):
        with pytest.raises(XEBError):
            CrossEntropyBenchmarking(p_error=-0.01)

    def test_invalid_p_error_one(self):
        with pytest.raises(XEBError):
            CrossEntropyBenchmarking(p_error=1.0)

    def test_p_error_zero_accepted(self):
        xeb = CrossEntropyBenchmarking(p_error=0.0)
        assert xeb.p_error == 0.0


# ---------------------------------------------------------------------------
# Run validation
# ---------------------------------------------------------------------------

class TestXEBRunValidation:
    def test_empty_depths_raises(self):
        xeb = CrossEntropyBenchmarking(seed=0)
        with pytest.raises(XEBError, match="depths must not be empty"):
            xeb.run([])

    def test_depth_zero_raises(self):
        xeb = CrossEntropyBenchmarking(seed=0)
        with pytest.raises(XEBError, match="depths must be >= 1"):
            xeb.run([0])

    def test_n_trials_zero_raises(self):
        xeb = CrossEntropyBenchmarking(seed=0)
        with pytest.raises(XEBError, match="n_trials must be >= 1"):
            xeb.run([1], n_trials=0)

    def test_returns_xeb_result(self):
        xeb = CrossEntropyBenchmarking(p_error=0.01, seed=42)
        result = xeb.run([1, 2], n_trials=5)
        assert isinstance(result, XEBResult)

    def test_result_depths_match(self):
        xeb = CrossEntropyBenchmarking(seed=0)
        depths = [1, 3, 5]
        result = xeb.run(depths, n_trials=5)
        assert result.depths == depths
        assert len(result.xeb_fidelities) == len(depths)

    def test_result_n_qubits(self):
        xeb = CrossEntropyBenchmarking(n_qubits=3, seed=0)
        result = xeb.run([1], n_trials=5)
        assert result.n_qubits == 3

    def test_result_n_trials(self):
        xeb = CrossEntropyBenchmarking(seed=0)
        result = xeb.run([1, 2], n_trials=10)
        assert result.n_trials == 10


# ---------------------------------------------------------------------------
# Physics / fidelity properties
# ---------------------------------------------------------------------------

class TestXEBFidelityProperties:
    def test_noiseless_fidelity_positive(self):
        """Noiseless XEB: F_XEB = d·∑P²−1 > 0 (≈ (d−1)/(d+1) by Porter–Thomas).

        For Haar-random circuits on d=4 dimensions the expected value is
        (d−1)/(d+1) = 3/5 = 0.6, not 1.  The test checks positivity and that
        the value is consistent with that statistical prediction.
        """
        xeb = CrossEntropyBenchmarking(n_qubits=2, p_error=0.0, seed=11)
        d = 1 << xeb.n_qubits  # 4
        expected = (d - 1) / (d + 1)  # 0.6
        result = xeb.run([1, 2, 4], n_trials=80)
        for f in result.xeb_fidelities:
            assert f > 0.0, f"Noiseless F_XEB should be positive, got {f}"
            assert abs(f - expected) < 0.25, (
                f"Noiseless F_XEB={f:.3f} too far from (d-1)/(d+1)={expected:.3f}"
            )

    def test_noisy_fidelity_below_noiseless(self):
        """Noisy XEB should produce fidelity below noiseless."""
        xeb_clean = CrossEntropyBenchmarking(n_qubits=2, p_error=0.0, seed=5)
        xeb_noisy = CrossEntropyBenchmarking(n_qubits=2, p_error=0.1, seed=5)
        depths = [2, 4]
        r_clean = xeb_clean.run(depths, n_trials=30)
        r_noisy = xeb_noisy.run(depths, n_trials=30)
        for fc, fn in zip(r_clean.xeb_fidelities, r_noisy.xeb_fidelities):
            assert fc > fn, (
                f"Noiseless F_XEB ({fc:.4f}) should exceed noisy ({fn:.4f})"
            )

    def test_fidelity_decreases_with_depth(self):
        """XEB fidelity should decay with circuit depth under noise."""
        xeb = CrossEntropyBenchmarking(n_qubits=2, p_error=0.05, seed=99)
        result = xeb.run([1, 2, 4, 8], n_trials=40)
        # Deeper circuits under noise should have strictly lower average fidelity
        avg_shallow = np.mean(result.xeb_fidelities[:2])
        avg_deep = np.mean(result.xeb_fidelities[2:])
        assert avg_shallow > avg_deep, (
            f"Shallow circuits (F={avg_shallow:.4f}) should have higher "
            f"fidelity than deep (F={avg_deep:.4f})"
        )

    def test_fidelity_bounded(self):
        """XEB fidelity should stay in a physically valid range."""
        xeb = CrossEntropyBenchmarking(n_qubits=2, p_error=0.02, seed=17)
        result = xeb.run([1, 2, 4], n_trials=20)
        for f in result.xeb_fidelities:
            # F_XEB ∈ (-1, 1] for exact simulation; allow small numerical slack
            assert -0.2 < f <= 1.1, f"F_XEB out of expected range: {f}"

    def test_mean_fidelity_is_average(self):
        xeb = CrossEntropyBenchmarking(seed=3)
        result = xeb.run([1, 2, 3], n_trials=5)
        assert result.mean_fidelity == pytest.approx(
            float(np.mean(result.xeb_fidelities)), abs=1e-10
        )

    def test_seed_reproducibility(self):
        """Same seed must give identical results."""
        xeb_a = CrossEntropyBenchmarking(n_qubits=2, p_error=0.01, seed=42)
        xeb_b = CrossEntropyBenchmarking(n_qubits=2, p_error=0.01, seed=42)
        r_a = xeb_a.run([1, 3], n_trials=10)
        r_b = xeb_b.run([1, 3], n_trials=10)
        assert r_a.xeb_fidelities == pytest.approx(r_b.xeb_fidelities)

    def test_different_seeds_give_different_results(self):
        xeb_a = CrossEntropyBenchmarking(n_qubits=2, seed=1)
        xeb_b = CrossEntropyBenchmarking(n_qubits=2, seed=2)
        r_a = xeb_a.run([2], n_trials=5)
        r_b = xeb_b.run([2], n_trials=5)
        assert r_a.xeb_fidelities != pytest.approx(r_b.xeb_fidelities)

    def test_fully_depolarised_fidelity_near_zero(self):
        """Very high noise should drive F_XEB toward 0."""
        xeb = CrossEntropyBenchmarking(n_qubits=2, p_error=0.9, seed=77)
        result = xeb.run([4], n_trials=50)
        # With near-maximal depolarising, fidelity should be close to 0
        assert abs(result.mean_fidelity) < 0.4, (
            f"Expected F_XEB ≈ 0 under extreme noise, got {result.mean_fidelity:.4f}"
        )

    def test_three_qubit_xeb(self):
        """XEB should work for 3 qubits.

        Note: for n=3 qubits the brickwork applies only one 2-qubit gate per
        layer (floor(n/2)=1 pair), so at depth 1 the third qubit stays in |0⟩
        and the effective space is C⁴ rather than C⁸.  We therefore use
        depth≥4 to ensure all qubits are entangled and check only that the
        noiseless fidelity is positive and noisy < noiseless.
        """
        xeb_clean = CrossEntropyBenchmarking(n_qubits=3, p_error=0.0, seed=13)
        xeb_noisy = CrossEntropyBenchmarking(n_qubits=3, p_error=0.05, seed=13)
        r_clean = xeb_clean.run([4], n_trials=20)
        r_noisy = xeb_noisy.run([4], n_trials=20)
        f_clean = r_clean.xeb_fidelities[0]
        f_noisy = r_noisy.xeb_fidelities[0]
        assert f_clean > 0.0, f"Noiseless 3-qubit F_XEB should be positive, got {f_clean}"
        assert f_clean > f_noisy, (
            f"Noiseless F_XEB ({f_clean:.3f}) should exceed noisy ({f_noisy:.3f})"
        )
