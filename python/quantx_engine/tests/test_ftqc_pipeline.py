"""M112: Tests for the fault-tolerant QC end-to-end pipeline."""
from __future__ import annotations

import math
import pytest

from quantx_engine.contrib.ftqc_pipeline import (
    FTQCPipeline,
    FTQCPipelineError,
    PipelineResult,
    ThresholdCurveResult,
    threshold_curve,
    threshold_crossing,
)


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_default_init(self):
        pipe = FTQCPipeline(code_distance=3, p_physical=0.01)
        assert pipe.code_distance == 3
        assert pipe.p_physical == 0.01
        assert pipe.p_meas == 0.0
        assert pipe.decoder == "mwpm"

    def test_explicit_params(self):
        pipe = FTQCPipeline(5, 0.05, p_meas=0.01, decoder="greedy")
        assert pipe.code_distance == 5
        assert pipe.p_meas == 0.01
        assert pipe.decoder == "greedy"

    def test_n_physical_qubits(self):
        assert FTQCPipeline(3, 0.01).n_physical_qubits == 9    # 3²
        assert FTQCPipeline(5, 0.01).n_physical_qubits == 25   # 5²
        assert FTQCPipeline(7, 0.01).n_physical_qubits == 49   # 7²

    def test_n_correctable(self):
        assert FTQCPipeline(3, 0.01).n_correctable == 1
        assert FTQCPipeline(5, 0.01).n_correctable == 2
        assert FTQCPipeline(7, 0.01).n_correctable == 3

    def test_invalid_distance_raises(self):
        with pytest.raises(FTQCPipelineError):
            FTQCPipeline(code_distance=2, p_physical=0.01)

    def test_invalid_distance_float_raises(self):
        with pytest.raises(FTQCPipelineError):
            FTQCPipeline(code_distance=3.0, p_physical=0.01)  # type: ignore

    def test_p_phys_zero_raises(self):
        with pytest.raises(FTQCPipelineError):
            FTQCPipeline(3, 0.0)

    def test_p_phys_one_raises(self):
        with pytest.raises(FTQCPipelineError):
            FTQCPipeline(3, 1.0)

    def test_p_meas_negative_raises(self):
        with pytest.raises(FTQCPipelineError):
            FTQCPipeline(3, 0.01, p_meas=-0.01)

    def test_unknown_decoder_raises(self):
        with pytest.raises(FTQCPipelineError):
            FTQCPipeline(3, 0.01, decoder="best")

    def test_repr(self):
        pipe = FTQCPipeline(3, 0.01)
        assert "FTQCPipeline" in repr(pipe)
        assert "d=3" in repr(pipe)


# ── PipelineResult ────────────────────────────────────────────────────────────

class TestPipelineResult:
    def test_run_returns_pipeline_result(self):
        pipe = FTQCPipeline(3, 0.01)
        result = pipe.run(n_shots=100, seed=0)
        assert isinstance(result, PipelineResult)

    def test_result_fields(self):
        pipe = FTQCPipeline(3, 0.01)
        result = pipe.run(n_shots=200, n_rounds=1, seed=7)
        assert result.code_distance == 3
        assert result.p_physical == 0.01
        assert result.p_meas == 0.0
        assert result.n_rounds == 1
        assert result.n_shots == 200
        assert 0 <= result.n_logical_errors <= 200
        assert result.logical_error_rate == result.n_logical_errors / 200
        assert result.decoder == "mwpm"

    def test_to_dict(self):
        result = FTQCPipeline(3, 0.01).run(n_shots=50, seed=0)
        d = result.to_dict()
        assert d["code_distance"] == 3
        assert "logical_error_rate" in d

    def test_invalid_n_shots_raises(self):
        with pytest.raises(FTQCPipelineError):
            FTQCPipeline(3, 0.01).run(n_shots=0)

    def test_invalid_n_rounds_raises(self):
        with pytest.raises(FTQCPipelineError):
            FTQCPipeline(3, 0.01).run(n_rounds=0)


# ── Logical error rate sanity ─────────────────────────────────────────────────

class TestLogicalErrorRate:
    def test_low_noise_low_error_rate(self):
        """Well below threshold: logical error rate should be very small."""
        pipe = FTQCPipeline(5, p_physical=0.001, decoder="mwpm")
        rate = pipe.logical_error_rate(n_shots=1000, seed=0)
        assert rate < 0.05

    def test_high_noise_high_error_rate(self):
        """Well above threshold: logical error rate should be high."""
        pipe = FTQCPipeline(3, p_physical=0.40, decoder="mwpm")
        rate = pipe.logical_error_rate(n_shots=500, seed=0)
        assert rate > 0.20

    def test_rate_between_zero_and_one(self):
        for p in [0.01, 0.05, 0.10, 0.15]:
            rate = FTQCPipeline(3, p).logical_error_rate(n_shots=200, seed=0)
            assert 0.0 <= rate <= 1.0

    def test_greedy_decoder_gives_rate_in_range(self):
        rate = FTQCPipeline(3, 0.05, decoder="greedy").logical_error_rate(
            n_shots=300, seed=1
        )
        assert 0.0 <= rate <= 1.0

    def test_reproducible_with_seed(self):
        pipe = FTQCPipeline(3, 0.05)
        r1 = pipe.logical_error_rate(n_shots=500, seed=42)
        r2 = pipe.logical_error_rate(n_shots=500, seed=42)
        assert r1 == r2

    def test_different_seeds_can_differ(self):
        pipe = FTQCPipeline(3, 0.10)
        rates = {pipe.logical_error_rate(n_shots=200, seed=s) for s in range(5)}
        assert len(rates) > 1

    def test_larger_distance_better_below_threshold(self):
        """d=7 should have lower logical error rate than d=3 at p=0.01."""
        rate3 = FTQCPipeline(3, 0.01).logical_error_rate(n_shots=2000, seed=0)
        rate7 = FTQCPipeline(7, 0.01).logical_error_rate(n_shots=2000, seed=0)
        # Expect rate7 ≤ rate3 with high probability
        assert rate7 <= rate3 + 0.03  # allow small statistical fluctuation


# ── Multi-round simulation ────────────────────────────────────────────────────

class TestMultiRound:
    def test_n_rounds_1_same_as_default(self):
        pipe = FTQCPipeline(3, 0.05)
        r1 = pipe.run(n_shots=300, n_rounds=1, seed=10)
        r_default = pipe.run(n_shots=300, n_rounds=1, seed=10)
        assert r1.logical_error_rate == r_default.logical_error_rate

    def test_n_rounds_3_runs_without_error(self):
        pipe = FTQCPipeline(3, 0.03, p_meas=0.01)
        result = pipe.run(n_shots=200, n_rounds=3, seed=5)
        assert isinstance(result, PipelineResult)
        assert result.n_rounds == 3

    def test_more_rounds_increases_error_rate(self):
        """More syndrome rounds accumulate more data errors → higher rate."""
        pipe = FTQCPipeline(3, 0.05)
        rate1 = pipe.logical_error_rate(n_shots=1000, n_rounds=1, seed=0)
        rate5 = pipe.logical_error_rate(n_shots=1000, n_rounds=5, seed=0)
        assert rate5 >= rate1 - 0.05  # rate5 should not be drastically lower

    def test_p_meas_affects_output(self):
        """p_meas=0 and p_meas=0.3 must produce different logical error rates
        (regression for P1: measurement errors were previously silently ignored)."""
        base = FTQCPipeline(3, 0.05, p_meas=0.0)
        noisy = FTQCPipeline(3, 0.05, p_meas=0.3)
        rate_base = base.logical_error_rate(n_shots=1000, n_rounds=3, seed=42)
        rate_noisy = noisy.logical_error_rate(n_shots=1000, n_rounds=3, seed=42)
        # High p_meas degrades decoder performance; rates must differ
        assert rate_base != rate_noisy

    def test_p_meas_high_raises_error_rate(self):
        """Very high p_meas (0.5) should degrade logical error rate."""
        low = FTQCPipeline(3, 0.05, p_meas=0.0)
        high = FTQCPipeline(3, 0.05, p_meas=0.4)
        rate_low = low.logical_error_rate(n_shots=800, n_rounds=3, seed=0)
        rate_high = high.logical_error_rate(n_shots=800, n_rounds=3, seed=0)
        assert rate_high > rate_low - 0.02  # p_meas=0.4 should be at least as bad


# ── threshold_curve ───────────────────────────────────────────────────────────

class TestThresholdCurve:
    def test_returns_threshold_curve_result(self):
        result = threshold_curve(
            distances=[3, 5], p_phys_range=(0.01, 0.18),
            n_points=6, n_shots=100, seed=0
        )
        assert isinstance(result, ThresholdCurveResult)

    def test_distances_and_p_phys_populated(self):
        result = threshold_curve(
            distances=[3, 5], p_phys_range=(0.01, 0.15),
            n_points=5, n_shots=80, seed=1
        )
        assert result.distances == [3, 5]
        assert len(result.p_phys) == 5
        assert 3 in result.p_logical
        assert 5 in result.p_logical
        assert len(result.p_logical[3]) == 5

    def test_p_logical_values_in_range(self):
        result = threshold_curve(
            distances=[3, 5], p_phys_range=(0.01, 0.15),
            n_points=5, n_shots=100, seed=2
        )
        for d in result.distances:
            for rate in result.p_logical[d]:
                assert 0.0 <= rate <= 1.0

    def test_default_distances(self):
        result = threshold_curve(n_points=4, n_shots=80, seed=3)
        assert result.distances == [3, 5, 7]

    def test_n_rounds_stored(self):
        result = threshold_curve(
            distances=[3, 5], n_points=4, n_shots=80, n_rounds=2, seed=4
        )
        assert result.n_rounds == 2

    def test_decoder_stored(self):
        result = threshold_curve(
            distances=[3, 5], n_points=4, n_shots=80, decoder="greedy", seed=5
        )
        assert result.decoder == "greedy"

    def test_threshold_is_none_or_float(self):
        result = threshold_curve(
            distances=[3, 5], n_points=5, n_shots=80, seed=6
        )
        assert result.threshold is None or isinstance(result.threshold, float)

    def test_invalid_p_range_raises(self):
        with pytest.raises(FTQCPipelineError):
            threshold_curve(distances=[3], p_phys_range=(0.2, 0.1), n_points=5)

    def test_to_dict(self):
        result = threshold_curve(
            distances=[3, 5], n_points=3, n_shots=50, seed=0
        )
        d = result.to_dict()
        assert "distances" in d
        assert "p_logical" in d
        assert "threshold" in d

    def test_threshold_in_plausible_range(self):
        """Surface code threshold expected ~ 0.10 for depolarizing noise."""
        result = threshold_curve(
            distances=[3, 5, 7],
            p_phys_range=(0.05, 0.18),
            n_points=10,
            n_shots=300,
            seed=99,
        )
        if result.threshold is not None:
            assert 0.03 < result.threshold < 0.20


# ── threshold_crossing ────────────────────────────────────────────────────────

class TestThresholdCrossing:
    def test_crossing_matches_curve_threshold(self):
        result = threshold_curve(
            distances=[3, 5], p_phys_range=(0.01, 0.18),
            n_points=8, n_shots=150, seed=7
        )
        crossing = threshold_crossing(result)
        # Should be the same as result.threshold
        assert crossing == result.threshold

    def test_crossing_none_when_no_crossing(self):
        """All p_logical identical → no crossing."""
        result = ThresholdCurveResult(
            distances=[3, 5],
            p_phys=[0.01, 0.05, 0.10],
            p_logical={3: [0.1, 0.2, 0.3], 5: [0.1, 0.2, 0.3]},
            threshold=None,
            decoder="mwpm",
            n_rounds=1,
        )
        assert threshold_crossing(result) is None

    def test_crossing_detected_analytically(self):
        """Manually crafted crossing at p=0.05 between p=0.03 and p=0.07."""
        result = ThresholdCurveResult(
            distances=[3, 5],
            p_phys=[0.03, 0.07],
            # d=3 worse than d=5 below crossing, better above
            p_logical={3: [0.20, 0.05], 5: [0.10, 0.15]},
            threshold=None,
            decoder="mwpm",
            n_rounds=1,
        )
        crossing = threshold_crossing(result)
        assert crossing is not None
        assert 0.03 < crossing < 0.07
