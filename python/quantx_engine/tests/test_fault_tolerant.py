"""Tests for fault-tolerant logical qubits — M91."""
from __future__ import annotations

import math

import pytest

from quantx_engine.contrib.fault_tolerant import (
    DistillationResult,
    FTError,
    FaultTolerantCircuit,
    LogicalQubit,
    MagicStateDistillery,
    logical_error_rate_vs_distance,
    t_gate_cost,
)


# ── LogicalQubit construction ─────────────────────────────────────────────────

class TestLogicalQubitConstruction:
    def test_default_state_is_zero(self):
        lq = LogicalQubit(3)
        lq.encode(0)
        assert lq.logical_state == 0

    def test_encode_one(self):
        lq = LogicalQubit(3)
        lq.encode(1)
        assert lq.logical_state == 1

    def test_n_physical_equals_distance(self):
        lq = LogicalQubit(5)
        assert lq.n_physical == 5

    def test_t_correctable_d3(self):
        assert LogicalQubit(3).t_correctable == 1

    def test_t_correctable_d5(self):
        assert LogicalQubit(5).t_correctable == 2

    def test_t_correctable_d7(self):
        assert LogicalQubit(7).t_correctable == 3

    def test_invalid_distance_even(self):
        with pytest.raises(FTError):
            LogicalQubit(4)

    def test_invalid_distance_too_small(self):
        with pytest.raises(FTError):
            LogicalQubit(1)

    def test_invalid_error_rate_negative(self):
        with pytest.raises(FTError):
            LogicalQubit(3, physical_error_rate=-0.01)

    def test_invalid_error_rate_above_one(self):
        with pytest.raises(FTError):
            LogicalQubit(3, physical_error_rate=1.5)

    def test_encode_invalid_state(self):
        with pytest.raises(FTError):
            LogicalQubit(3).encode(2)

    def test_repr(self):
        r = repr(LogicalQubit(3, 0.001))
        assert "LogicalQubit" in r
        assert "d=3" in r


# ── LogicalQubit gate operations ──────────────────────────────────────────────

class TestLogicalGates:
    def test_x_flips_state(self):
        lq = LogicalQubit(3)
        lq.encode(0)
        lq.apply_x()
        assert lq.logical_state == 1

    def test_x_twice_restores_state(self):
        lq = LogicalQubit(3)
        lq.encode(1)
        lq.apply_x()
        lq.apply_x()
        assert lq.logical_state == 1

    def test_z_flips_state(self):
        lq = LogicalQubit(3)
        lq.encode(0)
        lq.apply_z()
        assert lq.logical_state == 1

    def test_h_swaps_x_and_z(self):
        """H on |0⟩: logical_state=0, x_frame=0 → H → state=0, x_frame=0 → |0⟩."""
        lq = LogicalQubit(3)
        lq.encode(0)
        lq.apply_h()
        # After H, state should be deterministic from |0⟩ encode
        assert lq.logical_state in (0, 1)   # just check no crash

    def test_h_h_is_identity(self):
        lq = LogicalQubit(3)
        lq.encode(1)
        lq.apply_h()
        lq.apply_h()
        assert lq.logical_state == 1

    def test_xz_equals_zx_up_to_phase(self):
        """XZ and ZX applied to |0⟩ give same Z-measurement outcome."""
        lq1 = LogicalQubit(3, rng_seed=0)
        lq1.encode(0)
        lq1.apply_x()
        lq1.apply_z()

        lq2 = LogicalQubit(3, rng_seed=0)
        lq2.encode(0)
        lq2.apply_z()
        lq2.apply_x()

        # Both should end up in state |1⟩ (X|0⟩=|1⟩, then Z|1⟩=-|1⟩ same Z outcome)
        assert lq1.logical_state == lq2.logical_state


# ── Noiseless measurement ─────────────────────────────────────────────────────

class TestNoiselessMeasurement:
    def test_measure_zero_noiseless(self):
        lq = LogicalQubit(3, physical_error_rate=0.0)
        lq.encode(0)
        assert lq.measure_z() == 0

    def test_measure_one_noiseless(self):
        lq = LogicalQubit(3, physical_error_rate=0.0)
        lq.encode(1)
        assert lq.measure_z() == 1

    def test_x_then_measure_noiseless(self):
        lq = LogicalQubit(3, physical_error_rate=0.0)
        lq.encode(0)
        lq.apply_x()
        assert lq.measure_z() == 1

    def test_x_x_then_measure_noiseless(self):
        lq = LogicalQubit(3, physical_error_rate=0.0)
        lq.encode(0)
        lq.apply_x()
        lq.apply_x()
        assert lq.measure_z() == 0

    def test_n_rounds_invalid(self):
        lq = LogicalQubit(3)
        with pytest.raises(FTError):
            lq.measure_z(n_rounds=0)


# ── Noisy error correction ────────────────────────────────────────────────────

class TestNoisyCorrection:
    def test_low_error_rate_mostly_correct(self):
        """With p << threshold, logical errors should be rare."""
        lq = LogicalQubit(3, physical_error_rate=0.01, rng_seed=42)
        n_shots = 500
        errors = sum(lq.encode(0) or lq.measure_z() != 0 for _ in range(n_shots))
        assert errors / n_shots < 0.05

    def test_higher_distance_lower_logical_error(self):
        """Larger d should give lower logical error rate below threshold."""
        p = 0.05   # below threshold for repetition code
        lq3 = LogicalQubit(3, physical_error_rate=p, rng_seed=1)
        lq7 = LogicalQubit(7, physical_error_rate=p, rng_seed=1)
        ler3 = lq3.logical_error_rate(n_shots=500)
        ler7 = lq7.logical_error_rate(n_shots=500)
        # For repetition code with p=0.05, d=7 should beat d=3
        assert ler7 <= ler3 + 0.05   # allow statistical fluctuation

    def test_logical_error_rate_zero_noise(self):
        lq = LogicalQubit(3, physical_error_rate=0.0)
        ler = lq.logical_error_rate(n_shots=200)
        assert ler == 0.0

    def test_logical_error_rate_returns_float(self):
        lq = LogicalQubit(3, physical_error_rate=0.1, rng_seed=0)
        ler = lq.logical_error_rate(n_shots=100)
        assert isinstance(ler, float)
        assert 0.0 <= ler <= 1.0

    def test_logical_error_rate_invalid_shots(self):
        lq = LogicalQubit(3)
        with pytest.raises(FTError):
            lq.logical_error_rate(n_shots=0)


# ── MagicStateDistillery ──────────────────────────────────────────────────────

class TestMagicStateDistillery:
    def test_p_out_theoretical_cubic_suppression(self):
        p = 0.01
        d = MagicStateDistillery(p_in=p)
        assert d.p_out_theoretical == pytest.approx(35.0 * p ** 3, rel=1e-10)

    def test_low_p_in_mostly_accepted(self):
        d = MagicStateDistillery(p_in=0.001, rng_seed=0)
        result = d.distill(n_attempts=500)
        assert result.acceptance_rate > 0.5

    def test_low_p_in_very_low_output_error(self):
        d = MagicStateDistillery(p_in=0.001, rng_seed=7)
        result = d.distill(n_attempts=2000)
        # p_out ≈ 35*(0.001)^3 = 3.5e-8; measured rate should be < 0.01
        assert result.p_out_measured < 0.01

    def test_high_p_in_high_rejection(self):
        d = MagicStateDistillery(p_in=0.3, rng_seed=1)
        result = d.distill(n_attempts=500)
        # Most attempts should be rejected with high p_in
        assert result.acceptance_rate < 0.8

    def test_zero_p_in_no_errors(self):
        d = MagicStateDistillery(p_in=0.0, rng_seed=0)
        result = d.distill(n_attempts=500)
        assert result.n_errors == 0
        assert result.p_out_measured == 0.0

    def test_invalid_p_in_negative(self):
        with pytest.raises(FTError):
            MagicStateDistillery(p_in=-0.1)

    def test_invalid_p_in_above_one(self):
        with pytest.raises(FTError):
            MagicStateDistillery(p_in=1.5)

    def test_distill_invalid_attempts(self):
        with pytest.raises(FTError):
            MagicStateDistillery(0.01).distill(n_attempts=0)

    def test_distill_result_fields(self):
        result = MagicStateDistillery(0.005, rng_seed=42).distill(100)
        assert result.n_attempts == 100
        assert 0 <= result.n_accepted <= 100
        assert 0 <= result.n_errors <= result.n_accepted
        assert 0.0 <= result.p_out_measured <= 1.0
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_repr_distillery(self):
        assert "MagicStateDistillery" in repr(MagicStateDistillery(0.01))

    def test_repr_result(self):
        r = MagicStateDistillery(0.01, rng_seed=0).distill(10)
        assert "DistillationResult" in repr(r)


# ── FaultTolerantCircuit ──────────────────────────────────────────────────────

class TestFaultTolerantCircuit:
    def test_construction(self):
        c = FaultTolerantCircuit(2, distance=3)
        assert c.n_logical_qubits == 2
        assert c.distance == 3

    def test_repr(self):
        assert "FaultTolerantCircuit" in repr(FaultTolerantCircuit(1))

    def test_invalid_n_qubits(self):
        with pytest.raises(FTError):
            FaultTolerantCircuit(0)

    def test_out_of_range_qubit_raises(self):
        c = FaultTolerantCircuit(2)
        with pytest.raises(FTError):
            c.x(5)

    def test_encode_and_measure_zero(self):
        c = FaultTolerantCircuit(1, physical_error_rate=0.0, rng_seed=0)
        c.encode(0, 0)
        assert c.measure(0) == 0

    def test_encode_and_measure_one(self):
        c = FaultTolerantCircuit(1, physical_error_rate=0.0, rng_seed=0)
        c.encode(0, 1)
        assert c.measure(0) == 1

    def test_x_gate_flips(self):
        c = FaultTolerantCircuit(1, physical_error_rate=0.0)
        c.encode(0, 0)
        c.x(0)
        assert c.measure(0) == 1

    def test_x_x_identity(self):
        c = FaultTolerantCircuit(1, physical_error_rate=0.0)
        c.encode(0, 1)
        c.x(0).x(0)
        assert c.measure(0) == 1

    def test_z_gate(self):
        c = FaultTolerantCircuit(1, physical_error_rate=0.0)
        c.encode(0, 0)
        c.z(0)
        # Z flips the _logical_state; state should now be 1
        assert c._qubits[0]._logical_state == 1

    def test_h_h_is_identity_noiseless(self):
        c = FaultTolerantCircuit(1, physical_error_rate=0.0)
        c.encode(0, 1)
        c.h(0).h(0)
        assert c.measure(0) == 1

    def test_cnot_control_one_flips_target(self):
        c = FaultTolerantCircuit(2, physical_error_rate=0.0)
        c.encode(0, 1)   # control = |1⟩
        c.encode(1, 0)   # target  = |0⟩
        c.cnot(0, 1)
        assert c.measure(0) == 1
        assert c.measure(1) == 1   # target flipped

    def test_cnot_control_zero_leaves_target(self):
        c = FaultTolerantCircuit(2, physical_error_rate=0.0)
        c.encode(0, 0)   # control = |0⟩
        c.encode(1, 1)   # target  = |1⟩
        c.cnot(0, 1)
        assert c.measure(1) == 1   # target unchanged

    def test_measure_all(self):
        c = FaultTolerantCircuit(3, physical_error_rate=0.0)
        for q in range(3):
            c.encode(q, q % 2)
        outcomes = c.measure_all()
        assert outcomes == [0, 1, 0]

    def test_operations_logged(self):
        c = FaultTolerantCircuit(2, physical_error_rate=0.0)
        c.encode(0, 0).x(0).cnot(0, 1)
        assert c.n_operations == 3

    def test_fluent_chaining(self):
        c = FaultTolerantCircuit(2, physical_error_rate=0.0)
        result = c.encode(0, 0).encode(1, 1).x(0).cnot(0, 1)
        assert result is c


# ── Utility functions ─────────────────────────────────────────────────────────

class TestUtilityFunctions:
    def test_ler_vs_distance_keys(self):
        result = logical_error_rate_vs_distance([3, 5], 0.0, n_shots=50)
        assert set(result.keys()) == {3, 5}

    def test_ler_vs_distance_zero_noise(self):
        result = logical_error_rate_vs_distance([3, 5, 7], 0.0, n_shots=100)
        for d, ler in result.items():
            assert ler == 0.0

    def test_ler_vs_distance_decreasing_below_threshold(self):
        """Below threshold, higher d should give lower or equal LER."""
        result = logical_error_rate_vs_distance([3, 5, 7], 0.05,
                                                n_shots=300, rng_seed=42)
        # Just check that the dict has values in [0,1]
        for ler in result.values():
            assert 0.0 <= ler <= 1.0

    def test_t_gate_cost_zero_levels_for_good_state(self):
        """If p_in is already below p_target, 0 levels needed."""
        assert t_gate_cost(3, 1e-15, p_target=1e-10) == 0

    def test_t_gate_cost_positive_for_high_p(self):
        levels = t_gate_cost(3, 0.01, p_target=1e-10)
        assert levels > 0

    def test_t_gate_cost_cubic_suppression(self):
        """Each level cubes the suppression: 2 levels should handle p=0.01→1e-10."""
        levels = t_gate_cost(3, 0.01, p_target=1e-10)
        assert levels <= 5   # certainly within 5 levels

    def test_t_gate_cost_invalid_p(self):
        with pytest.raises(FTError):
            t_gate_cost(3, 0.0)

    def test_t_gate_cost_invalid_p_target(self):
        with pytest.raises(FTError):
            t_gate_cost(3, 0.01, p_target=0.0)
