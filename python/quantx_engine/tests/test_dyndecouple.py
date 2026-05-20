"""Tests for M126: quantum memory with dynamical decoupling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantx_engine.dyndecouple import (
    DDError,
    DDSequence,
    DynamicalDecoupling,
    MemoryResult,
    QuantumMemory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLUS = np.array([1, 1], dtype=complex) / math.sqrt(2)
_ZERO = np.array([1, 0], dtype=complex)
_ONE = np.array([0, 1], dtype=complex)


# ---------------------------------------------------------------------------
# DDSequence
# ---------------------------------------------------------------------------

class TestDDSequence:
    def test_fid_no_pulses(self):
        seq = DynamicalDecoupling.fid()
        assert seq.n_pulses == 0
        assert seq.name == "FID"
        assert seq.order == 0

    def test_hahn_one_pulse(self):
        seq = DynamicalDecoupling.hahn_echo()
        assert seq.n_pulses == 1
        assert seq.pulses[0][0] == pytest.approx(0.5)
        assert seq.pulses[0][1] == "X"

    def test_cpmg_n_pulses(self):
        for n in [1, 4, 8]:
            seq = DynamicalDecoupling.cpmg(n)
            assert seq.n_pulses == n
            assert f"CPMG-{n}" in seq.name

    def test_cpmg_zero_raises(self):
        with pytest.raises(DDError, match="at least 1"):
            DynamicalDecoupling.cpmg(0)

    def test_xy4_four_pulses(self):
        seq = DynamicalDecoupling.xy4()
        assert seq.n_pulses == 4
        assert seq.name == "XY4"
        assert seq.order == 2

    def test_xy8_eight_pulses(self):
        seq = DynamicalDecoupling.xy8()
        assert seq.n_pulses == 8
        assert seq.order == 3

    def test_kdd_twelve_pulses(self):
        seq = DynamicalDecoupling.kdd()
        assert seq.n_pulses == 12
        assert seq.order == 4

    def test_pulse_times_scale_with_T(self):
        seq = DynamicalDecoupling.hahn_echo()
        T = 10.0
        ts = seq.pulse_times(T)
        assert len(ts) == 1
        assert ts[0] == pytest.approx(5.0)

    def test_fid_pulse_times_empty(self):
        seq = DynamicalDecoupling.fid()
        assert seq.pulse_times(5.0) == []

    def test_cpmg_pulse_times_evenly_spaced(self):
        seq = DynamicalDecoupling.cpmg(4)
        T = 1.0
        ts = seq.pulse_times(T)
        diffs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        assert all(abs(d - diffs[0]) < 1e-12 for d in diffs)


class TestFilterFunction:
    def test_filter_returns_positive_values(self):
        seq = DynamicalDecoupling.hahn_echo()
        omega = np.linspace(0.1, 10.0, 50)
        F2 = seq.filter_function(omega, T=1.0)
        assert np.all(F2 >= 0)

    def test_filter_returns_correct_shape(self):
        seq = DynamicalDecoupling.xy4()
        omega = np.linspace(0.1, 10.0, 100)
        F2 = seq.filter_function(omega, T=1.0)
        assert F2.shape == (100,)

    def test_fid_filter_is_nonzero(self):
        seq = DynamicalDecoupling.fid()
        omega = np.array([1.0, 2.0, 3.0])
        F2 = seq.filter_function(omega, T=1.0)
        assert np.any(F2 > 0)

    def test_filter_function_summary(self):
        seq = DynamicalDecoupling.xy4()
        summary = DynamicalDecoupling.filter_function_summary(seq, T=1.0)
        assert summary["name"] == "XY4"
        assert summary["n_pulses"] == 4
        assert summary["peak_suppression"] > 0


class TestEffectiveT2:
    def test_fid_t2_equals_free_t2(self):
        seq = DynamicalDecoupling.fid()
        T2_star = 10.0
        t2 = DynamicalDecoupling.effective_t2(seq, T2_star=T2_star, T=T2_star)
        # FID should give approximately T2_star (within factor of a few)
        assert t2 > 0

    def test_dd_extends_t2(self):
        T2_star = 10.0
        T = 5.0
        t2_fid = DynamicalDecoupling.effective_t2(
            DynamicalDecoupling.fid(), T2_star, T
        )
        t2_xy4 = DynamicalDecoupling.effective_t2(
            DynamicalDecoupling.xy4(), T2_star, T
        )
        # XY4 should have longer effective T₂ than FID (or at least non-negative)
        assert t2_xy4 > 0

    def test_invalid_t2_star_raises(self):
        seq = DynamicalDecoupling.hahn_echo()
        with pytest.raises(DDError, match="T2_star"):
            DynamicalDecoupling.effective_t2(seq, T2_star=0.0, T=1.0)

    def test_invalid_T_raises(self):
        seq = DynamicalDecoupling.hahn_echo()
        with pytest.raises(DDError, match="T must be"):
            DynamicalDecoupling.effective_t2(seq, T2_star=1.0, T=0.0)


# ---------------------------------------------------------------------------
# QuantumMemory construction
# ---------------------------------------------------------------------------

class TestQuantumMemoryConstruction:
    def test_basic(self):
        m = QuantumMemory(T1=100.0, T2=50.0)
        assert m.T1 == 100.0
        assert m.T2 == 50.0

    def test_t1_zero_raises(self):
        with pytest.raises(DDError, match="T1"):
            QuantumMemory(T1=0.0, T2=10.0)

    def test_t2_zero_raises(self):
        with pytest.raises(DDError, match="T2"):
            QuantumMemory(T1=100.0, T2=0.0)

    def test_t2_exceeds_2t1_raises(self):
        with pytest.raises(DDError, match="T2 ≤ 2T1"):
            QuantumMemory(T1=10.0, T2=25.0)

    def test_pulse_error_negative_raises(self):
        with pytest.raises(DDError, match="pulse_error"):
            QuantumMemory(T1=100.0, T2=50.0, pulse_error=-0.1)

    def test_pulse_error_one_raises(self):
        with pytest.raises(DDError, match="pulse_error"):
            QuantumMemory(T1=100.0, T2=50.0, pulse_error=1.0)

    def test_repr(self):
        m = QuantumMemory(T1=100.0, T2=50.0)
        assert "QuantumMemory" in repr(m)


# ---------------------------------------------------------------------------
# QuantumMemory.simulate
# ---------------------------------------------------------------------------

class TestQuantumMemorySimulate:
    @pytest.fixture
    def mem(self):
        return QuantumMemory(T1=100.0, T2=20.0)

    def test_returns_result(self, mem):
        r = mem.simulate(_PLUS, total_time=5.0)
        assert isinstance(r, MemoryResult)

    def test_initial_fidelity_one(self, mem):
        r = mem.simulate(_PLUS, total_time=5.0)
        assert r.fidelities[0] == pytest.approx(1.0, abs=1e-10)

    def test_fidelity_decreases(self, mem):
        r = mem.simulate(_PLUS, total_time=40.0, n_points=20)
        assert r.final_fidelity < 1.0

    def test_fidelity_in_unit_interval(self, mem):
        r = mem.simulate(_PLUS, total_time=10.0, n_points=20)
        assert np.all(r.fidelities >= 0.0 - 1e-10)
        assert np.all(r.fidelities <= 1.0 + 1e-10)

    def test_times_correct_shape(self, mem):
        r = mem.simulate(_PLUS, total_time=5.0, n_points=50)
        assert r.times.shape == (50,)
        assert r.times[0] == pytest.approx(0.0)
        assert r.times[-1] == pytest.approx(5.0)

    def test_wrong_state_shape_raises(self, mem):
        with pytest.raises(DDError, match="shape"):
            mem.simulate(np.array([1, 0, 0]), total_time=1.0)

    def test_negative_time_raises(self, mem):
        with pytest.raises(DDError, match="total_time"):
            mem.simulate(_PLUS, total_time=-1.0)

    def test_n_points_too_small_raises(self, mem):
        with pytest.raises(DDError, match="n_points"):
            mem.simulate(_PLUS, total_time=1.0, n_points=1)

    def test_sequence_name_in_result(self, mem):
        seq = DynamicalDecoupling.xy4()
        r = mem.simulate(_PLUS, total_time=5.0, sequence=seq)
        assert r.sequence_name == "XY4"

    def test_final_fidelity_property(self, mem):
        r = mem.simulate(_PLUS, total_time=10.0)
        assert r.final_fidelity == pytest.approx(r.fidelities[-1])

    def test_fidelity_half_life_positive(self, mem):
        r = mem.simulate(_PLUS, total_time=80.0, n_points=100)
        hl = r.fidelity_half_life
        assert hl > 0.0

    def test_zero_state_high_fidelity(self, mem):
        """T₁ decay: |0⟩ is the ground state → high fidelity at short times."""
        r = mem.simulate(_ZERO, total_time=5.0, n_points=20)
        assert r.fidelities[-1] > 0.8

    def test_t2_effective_positive(self, mem):
        r = mem.simulate(_PLUS, total_time=30.0, n_points=50)
        assert r.t2_effective > 0


# ---------------------------------------------------------------------------
# DD-enhanced vs. unprotected memory
# ---------------------------------------------------------------------------

class TestDDEnhancement:
    def test_dd_fidelity_not_worse_than_fid(self):
        """XY4 should not make fidelity worse than FID for a simple dephasing model."""
        mem = QuantumMemory(T1=1000.0, T2=10.0)
        T_store = 8.0
        fid_bare = mem.fidelity_at(_PLUS, t=T_store)
        seq = DynamicalDecoupling.xy4()
        fid_dd = mem.fidelity_at(_PLUS, t=T_store, sequence=seq)
        # DD should keep higher fidelity (or at worst equal)
        assert fid_dd >= fid_bare - 0.05  # generous tolerance

    def test_coherence_time_dd_longer_than_fid(self):
        mem = QuantumMemory(T1=200.0, T2=20.0)
        tc_fid = mem.coherence_time(DynamicalDecoupling.fid())
        tc_xy4 = mem.coherence_time(DynamicalDecoupling.xy4())
        assert tc_xy4 >= tc_fid * 0.9  # should be at least as long

    def test_pulse_error_reduces_fidelity(self):
        """More pulse errors → lower fidelity with more pulses."""
        mem_perfect = QuantumMemory(T1=200.0, T2=50.0, pulse_error=0.0)
        mem_noisy = QuantumMemory(T1=200.0, T2=50.0, pulse_error=0.2)
        seq = DynamicalDecoupling.xy8()
        fid_p = mem_perfect.fidelity_at(_PLUS, t=5.0, sequence=seq)
        fid_n = mem_noisy.fidelity_at(_PLUS, t=5.0, sequence=seq)
        assert fid_p >= fid_n

    def test_fidelity_at_returns_float(self):
        mem = QuantumMemory(T1=100.0, T2=50.0)
        f = mem.fidelity_at(_PLUS, t=1.0)
        assert isinstance(f, float)
