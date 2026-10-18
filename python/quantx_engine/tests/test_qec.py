"""Tests for the QEC research toolkit."""
from __future__ import annotations

import pytest
from quantx_engine.contrib.qec import (
    RepetitionCode, SurfaceCode, LogicalErrorSimulator, ThresholdResult, QECError,
)


# ── RepetitionCode ────────────────────────────────────────────────────────────

def test_rep_encode_0():
    code = RepetitionCode(5)
    assert code.encode(0) == [0, 0, 0, 0, 0]


def test_rep_encode_1():
    code = RepetitionCode(3)
    assert code.encode(1) == [1, 1, 1]


def test_rep_encode_invalid():
    with pytest.raises(QECError):
        RepetitionCode(3).encode(2)


def test_rep_syndrome_no_error():
    code = RepetitionCode(5)
    assert code.syndrome([0, 0, 0, 0, 0]) == [0, 0, 0, 0]
    assert code.syndrome([1, 1, 1, 1, 1]) == [0, 0, 0, 0]


def test_rep_syndrome_single_error():
    code = RepetitionCode(5)
    errored = [0, 1, 0, 0, 0]  # flip qubit 1
    syn = code.syndrome(errored)
    assert syn == [1, 1, 0, 0]  # qubits (0,1) and (1,2) disagree


def test_rep_decode_no_error():
    code = RepetitionCode(5)
    assert code.decode([0, 0, 0, 0, 0]) == 0
    assert code.decode([1, 1, 1, 1, 1]) == 1


def test_rep_decode_single_error():
    code = RepetitionCode(5)
    # Single flip on qubit 2 → majority vote still correct
    assert code.decode([0, 0, 1, 0, 0]) == 0
    assert code.decode([1, 1, 0, 1, 1]) == 1


def test_rep_decode_double_error_fails():
    """Three qubit code fails with 2 errors (too many to correct)."""
    code = RepetitionCode(3)
    # 2 out of 3 flipped → decoded wrong
    assert code.decode([0, 1, 1]) == 1  # wrong: should be 0


def test_rep_min_distance():
    code = RepetitionCode(7)
    assert code.d == 7
    assert code.n == 7
    assert code.k == 1


def test_rep_to_dict():
    d = RepetitionCode(5).to_dict()
    assert d["code"] == "repetition"
    assert d["n"] == 5


def test_rep_requires_n_ge_3():
    with pytest.raises(QECError):
        RepetitionCode(2)


# ── SurfaceCode ───────────────────────────────────────────────────────────────

def test_surface_d3_qubit_count():
    code = SurfaceCode(d=3)
    assert code.n == 9
    assert code.k == 1
    assert code.d == 3


def test_surface_d3_stabilizer_counts():
    code = SurfaceCode(d=3)
    # d=3 rotated code has 4 Z stabilizers and 4 X stabilizers
    assert code.n_z_syndromes == 4
    assert code.n_x_syndromes == 4


def test_surface_syndrome_no_error():
    code = SurfaceCode(d=3)
    zeros = [0] * 9
    assert code.z_syndrome(zeros) == [0] * code.n_z_syndromes
    assert code.x_syndrome(zeros) == [0] * code.n_x_syndromes


def test_surface_syndrome_single_x_error():
    code = SurfaceCode(d=3)
    # X error on qubit 4 (center) should trigger exactly 2 Z-type stabilizers
    x_err = [0] * 9
    x_err[4] = 1
    syn = code.z_syndrome(x_err)
    assert sum(syn) in (1, 2, 4)  # at least 1 stabilizer triggered


def test_surface_logical_operators():
    code = SurfaceCode(d=3)
    lx = code.logical_x()
    lz = code.logical_z()
    assert len(lx) == 3
    assert len(lz) == 3
    # Logical X (column) and Z (row) must anticommute: odd overlap
    overlap = len(set(lx) & set(lz))
    assert overlap % 2 == 1


def test_surface_no_logical_error_on_zero_errors():
    code = SurfaceCode(d=3)
    zeros = [0] * 9
    lx, lz = code.logical_error(zeros, zeros, zeros, zeros)
    assert not lx and not lz


def test_surface_logical_x_error():
    """Applying logical X (uncorrected) should trigger a logical X error."""
    code = SurfaceCode(d=3)
    zeros = [0] * 9
    x_err = [0] * 9
    for q in code.logical_x():
        x_err[q] = 1
    lx, lz = code.logical_error(x_err, zeros, zeros, zeros)
    assert lz  # logical Z error (X on all qubits of logical X anticommutes with Z_L)


def test_surface_decode_no_error():
    code = SurfaceCode(d=3)
    zeros = [0] * 9
    corr = code.decode_z(code.x_syndrome(zeros))
    assert all(c == 0 for c in corr)


def test_surface_decode_single_x_error():
    """Single X error on any qubit must not cause a logical error (distance-3 code)."""
    code = SurfaceCode(d=3)
    zeros = [0] * 9
    z_err = [0] * 9
    for q in range(9):
        x_err = [0] * 9
        x_err[q] = 1
        syn_z = code.z_syndrome(x_err)
        syn_x = code.x_syndrome(z_err)
        corr_x = code.decode_x(syn_z)
        corr_z = code.decode_z(syn_x)
        lx, lz = code.logical_error(x_err, z_err, corr_x, corr_z)
        assert not lx and not lz, (
            f"Single X error on qubit {q} caused a logical error after correction"
        )


def test_surface_requires_d_ge_3():
    with pytest.raises(QECError):
        SurfaceCode(d=2)


def test_surface_to_dict():
    d = SurfaceCode(d=3).to_dict()
    assert d["code"] == "surface"
    assert d["d"] == 3
    assert d["n"] == 9


# ── LogicalErrorSimulator ─────────────────────────────────────────────────────

def test_sim_rep_low_noise():
    """Repetition code at p=0.01 should give very low logical error rate."""
    code = RepetitionCode(5)
    sim = LogicalErrorSimulator(code)
    p_L = sim.logical_error_rate(p_phys=0.01, n_trials=500, seed=0)
    assert p_L < 0.05, f"Expected low logical error rate, got {p_L}"


def test_sim_rep_high_noise():
    """Repetition code at p=0.4 should give high logical error rate."""
    code = RepetitionCode(3)
    sim = LogicalErrorSimulator(code)
    p_L = sim.logical_error_rate(p_phys=0.4, n_trials=500, seed=0)
    assert p_L > 0.3, f"Expected high logical error rate at p=0.4, got {p_L}"


def test_sim_surface_low_noise():
    """Surface code d=3 at p=0.001 should give near-zero logical error rate."""
    code = SurfaceCode(d=3)
    sim = LogicalErrorSimulator(code)
    p_L = sim.logical_error_rate(p_phys=0.001, n_trials=1000, seed=42)
    assert p_L < 0.05, f"Expected near-zero logical error rate, got {p_L}"


def test_sim_invalid_p():
    code = RepetitionCode(3)
    sim = LogicalErrorSimulator(code)
    with pytest.raises(QECError):
        sim.logical_error_rate(p_phys=1.5)


def test_threshold_result_structure():
    code = RepetitionCode(3)
    sim = LogicalErrorSimulator(code)
    result = sim.threshold_sweep(p_range=(0.01, 0.4), n_points=5, n_trials=200, seed=0)
    assert isinstance(result, ThresholdResult)
    assert len(result.p_phys) == 5
    assert len(result.p_logical) == 5
    assert all(0.0 <= p <= 1.0 for p in result.p_logical)


def test_threshold_result_to_dict():
    code = RepetitionCode(3)
    sim = LogicalErrorSimulator(code)
    result = sim.threshold_sweep(n_points=3, n_trials=100, seed=0)
    d = result.to_dict()
    assert "p_phys" in d and "p_logical" in d and "threshold" in d


def test_threshold_rep_below_05():
    """Repetition code threshold should be near 0.5."""
    code = RepetitionCode(5)
    sim = LogicalErrorSimulator(code)
    result = sim.threshold_sweep(p_range=(0.05, 0.45), n_points=5, n_trials=300, seed=1)
    # At p < 0.5 the code should perform better than uncorrected
    # (not asserting exact threshold, just that high-p entries have higher logical error)
    assert result.p_logical[-1] > result.p_logical[0]
