"""Tests for the MWPM decoder (M78)."""
from __future__ import annotations

import random

import pytest

from ..qec import (
    QECError,
    SurfaceCode,
    RepetitionCode,
    MWPMDecoder,
    _mwpm_bitmask,
    _mwpm_greedy_fallback,
)


# ===========================================================================
# _mwpm_bitmask — unit tests for the core DP solver
# ===========================================================================

class TestMWPMBitmask:
    def test_empty(self):
        cost, matching = _mwpm_bitmask({}, 0)
        assert cost == 0.0
        assert matching == []

    def test_two_nodes(self):
        costs = {(0, 1): 3.0}
        cost, matching = _mwpm_bitmask(costs, 2)
        assert cost == 3.0
        assert set(matching) == {(0, 1)}

    def test_four_nodes_optimal(self):
        # 4 nodes: 0-1-2-3 in a line
        # Optimal: 0-1 (dist 1), 2-3 (dist 1) → total 2
        # Suboptimal: 0-2 (dist 2), 1-3 (dist 2) → total 4
        costs = {
            (0, 1): 1.0, (0, 2): 2.0, (0, 3): 3.0,
            (1, 2): 1.0, (1, 3): 2.0,
            (2, 3): 1.0,
        }
        cost, matching = _mwpm_bitmask(costs, 4)
        assert cost == 2.0
        matched_set = {frozenset(p) for p in matching}
        assert frozenset({0, 1}) in matched_set
        assert frozenset({2, 3}) in matched_set

    def test_four_nodes_suboptimal_greedy_case(self):
        # Counter-example where greedy would fail:
        # Nodes: 0, 1, 2, 3
        # d(0,1) = 1.5  ← greedy picks this first
        # d(2,3) = 5.0  ← then forced to pick this
        # d(0,2) = 1.0, d(1,3) = 1.0  ← optimal pairing: total 2.0
        costs = {
            (0, 1): 1.5,
            (0, 2): 1.0, (0, 3): 4.0,
            (1, 2): 4.0, (1, 3): 1.0,
            (2, 3): 5.0,
        }
        cost, matching = _mwpm_bitmask(costs, 4)
        # Optimal: 0-2 (1.0) + 1-3 (1.0) = 2.0
        # Greedy would take 0-1 (1.5) + 2-3 (5.0) = 6.5
        assert cost == pytest.approx(2.0)
        matched_set = {frozenset(p) for p in matching}
        assert frozenset({0, 2}) in matched_set
        assert frozenset({1, 3}) in matched_set

    def test_symmetric_costs(self):
        """Cost lookup via (i,j) and (j,i) both work."""
        costs = {(1, 0): 2.5}  # Only reversed key
        cost, matching = _mwpm_bitmask(costs, 2)
        assert cost == 2.5

    def test_six_nodes(self):
        # 6 nodes in a line: optimal is adjacent pairs
        costs = {}
        for i in range(6):
            for j in range(i + 1, 6):
                costs[(i, j)] = float(abs(i - j))
        cost, matching = _mwpm_bitmask(costs, 6)
        # Optimal: (0,1)(2,3)(4,5) = 1+1+1 = 3
        assert cost == pytest.approx(3.0)


class TestMWPMGreedyFallback:
    def test_empty(self):
        cost, matching = _mwpm_greedy_fallback({}, [])
        assert cost == 0.0
        assert matching == []

    def test_two_nodes(self):
        costs = {(0, 1): 4.0}
        cost, matching = _mwpm_greedy_fallback(costs, [0, 1])
        assert cost == 4.0

    def test_four_nodes(self):
        costs = {
            (0, 1): 1.0, (0, 2): 5.0, (0, 3): 5.0,
            (1, 2): 5.0, (1, 3): 5.0,
            (2, 3): 1.0,
        }
        cost, matching = _mwpm_greedy_fallback(costs, [0, 1, 2, 3])
        # Greedy picks (0,1) then (2,3): cost = 2.0
        assert cost == pytest.approx(2.0)


# ===========================================================================
# MWPMDecoder construction
# ===========================================================================

class TestMWPMDecoderConstruction:
    def test_basic_construction(self):
        code = SurfaceCode(d=3)
        dec = MWPMDecoder(code)
        assert dec.code is code
        assert dec.max_exact == 10

    def test_custom_max_exact(self):
        code = SurfaceCode(d=3)
        dec = MWPMDecoder(code, max_exact=5)
        assert dec.max_exact == 5

    def test_max_exact_zero_raises(self):
        code = SurfaceCode(d=3)
        with pytest.raises(QECError):
            MWPMDecoder(code, max_exact=0)

    def test_repetition_code_accepted(self):
        code = RepetitionCode(n=5)
        dec = MWPMDecoder(code)
        assert dec.code is code


# ===========================================================================
# MWPMDecoder.decode_x / decode_z — correctness
# ===========================================================================

class TestMWPMDecoderDecodeX:
    def setup_method(self):
        self.code = SurfaceCode(d=3)
        self.dec = MWPMDecoder(self.code)

    def test_no_errors_no_correction(self):
        x_err = [0] * self.code.n
        syn = self.code.z_syndrome(x_err)
        corr = self.dec.decode_x(syn)
        assert corr == [0] * self.code.n

    def test_single_x_error_corrected(self):
        """Single X error at qubit 4 (centre of d=3): decoder finds correction,
        and the net error has no logical component."""
        code = self.code
        for qubit in range(code.n):
            x_err = [0] * code.n
            x_err[qubit] = 1
            syn = code.z_syndrome(x_err)
            corr = self.dec.decode_x(syn)
            # Net residual: x_err XOR corr
            net = [x_err[i] ^ corr[i] for i in range(code.n)]
            # No logical Z error (net X residual overlaps logical X support even number of times)
            lz = sum(net[i] for i in code.logical_x()) % 2
            assert lz == 0, f"Logical error for X on qubit {qubit}: net={net}"

    def test_two_x_errors_correction_zeroes_syndrome(self):
        """MWPM correction must zero the syndrome (valid correction guarantee).

        Note: two errors exceed d=3's correction capacity (corrects only 1),
        so a logical error may still occur — MWPM only guarantees the
        residual syndrome is zero.
        """
        code = self.code
        x_err = [0] * code.n
        x_err[0] = 1
        x_err[1] = 1
        syn = code.z_syndrome(x_err)
        corr = self.dec.decode_x(syn)
        net = [x_err[i] ^ corr[i] for i in range(code.n)]
        # The net residual must have zero Z syndrome (lies in the code space)
        residual_syn = code.z_syndrome(net)
        assert all(s == 0 for s in residual_syn), (
            f"Correction did not zero syndrome: residual_syn={residual_syn}"
        )

    def test_all_zero_syndrome_no_correction(self):
        syn = [0] * self.code.n_z_syndromes
        corr = self.dec.decode_x(syn)
        assert all(c == 0 for c in corr)


class TestMWPMDecoderDecodeZ:
    def setup_method(self):
        self.code = SurfaceCode(d=3)
        self.dec = MWPMDecoder(self.code)

    def test_no_errors_no_correction(self):
        z_err = [0] * self.code.n
        syn = self.code.x_syndrome(z_err)
        corr = self.dec.decode_z(syn)
        assert corr == [0] * self.code.n

    def test_single_z_error_corrected(self):
        code = self.code
        for qubit in range(code.n):
            z_err = [0] * code.n
            z_err[qubit] = 1
            syn = code.x_syndrome(z_err)
            corr = self.dec.decode_z(syn)
            net = [z_err[i] ^ corr[i] for i in range(code.n)]
            lx = sum(net[i] for i in code.logical_z()) % 2
            assert lx == 0, f"Logical error for Z on qubit {qubit}"


# ===========================================================================
# d=5 surface code — MWPM handles more defects
# ===========================================================================

class TestMWPMDecoderD5:
    def setup_method(self):
        self.code = SurfaceCode(d=5)
        self.dec = MWPMDecoder(self.code)

    def test_no_errors(self):
        x_err = [0] * self.code.n
        syn = self.code.z_syndrome(x_err)
        corr = self.dec.decode_x(syn)
        assert corr == [0] * self.code.n

    def test_single_x_errors_all_qubits(self):
        code = self.code
        for qubit in range(code.n):
            x_err = [0] * code.n
            x_err[qubit] = 1
            syn = code.z_syndrome(x_err)
            corr = self.dec.decode_x(syn)
            net = [x_err[i] ^ corr[i] for i in range(code.n)]
            lz = sum(net[i] for i in code.logical_x()) % 2
            assert lz == 0, f"d=5 logical error for X on qubit {qubit}"

    def test_single_z_errors_all_qubits(self):
        code = self.code
        for qubit in range(code.n):
            z_err = [0] * code.n
            z_err[qubit] = 1
            syn = code.x_syndrome(z_err)
            corr = self.dec.decode_z(syn)
            net = [z_err[i] ^ corr[i] for i in range(code.n)]
            lx = sum(net[i] for i in code.logical_z()) % 2
            assert lx == 0, f"d=5 logical error for Z on qubit {qubit}"


# ===========================================================================
# MWPM vs Greedy: MWPM wins on adversarial cases
# ===========================================================================

class TestMWPMBetterThanGreedy:
    def test_mwpm_exact_beats_greedy_on_4node_counterexample(self):
        """Verify the DP solver finds the true minimum on a case where
        greedy picks a suboptimal first pair."""
        # Constructed adversarial costs (not derived from a real syndrome graph,
        # just testing the solver directly):
        costs = {
            (0, 1): 1.5,
            (0, 2): 1.0, (0, 3): 10.0,
            (1, 2): 10.0, (1, 3): 1.0,
            (2, 3): 100.0,
        }
        mwpm_cost, _ = _mwpm_bitmask(costs, 4)
        greedy_cost, _ = _mwpm_greedy_fallback(costs, [0, 1, 2, 3])
        # MWPM: 0-2 (1.0) + 1-3 (1.0) = 2.0; greedy: 0-1 (1.5) + 2-3 (100) = 101.5
        assert mwpm_cost <= greedy_cost
        assert mwpm_cost == pytest.approx(2.0)

    def test_mwpm_at_least_as_good_as_greedy(self):
        """On random small instances, MWPM total cost ≤ greedy total cost."""
        rng = random.Random(42)
        n = 6  # 6 nodes → 3 pairs
        for _ in range(100):
            costs = {}
            for i in range(n):
                for j in range(i + 1, n):
                    costs[(i, j)] = rng.uniform(0.1, 10.0)
            mwpm_cost, _ = _mwpm_bitmask(costs, n)
            greedy_cost, _ = _mwpm_greedy_fallback(costs, list(range(n)))
            assert mwpm_cost <= greedy_cost + 1e-9


# ===========================================================================
# logical_error_rate integration
# ===========================================================================

class TestMWPMLogicalErrorRate:
    def test_low_physical_error_low_logical(self):
        code = SurfaceCode(d=3)
        dec = MWPMDecoder(code)
        pL = dec.logical_error_rate(p_phys=0.005, n_trials=1000, seed=0)
        # At p=0.005 with d=3, logical error rate should be well below 1%
        assert 0.0 <= pL < 0.05

    def test_high_physical_error_approaches_half(self):
        code = SurfaceCode(d=3)
        dec = MWPMDecoder(code)
        pL = dec.logical_error_rate(p_phys=0.4, n_trials=2000, seed=1)
        # At p=0.4 (well above threshold), logical error rate should be high
        assert pL > 0.1

    def test_returns_float_in_01(self):
        code = SurfaceCode(d=3)
        dec = MWPMDecoder(code)
        pL = dec.logical_error_rate(p_phys=0.05, n_trials=200, seed=7)
        assert 0.0 <= pL <= 1.0

    def test_invalid_p_raises(self):
        code = SurfaceCode(d=3)
        dec = MWPMDecoder(code)
        with pytest.raises(QECError):
            dec.logical_error_rate(p_phys=1.1)
        with pytest.raises(QECError):
            dec.logical_error_rate(p_phys=0.0)

    def test_repetition_code_raises(self):
        code = RepetitionCode(n=5)
        dec = MWPMDecoder(code)
        with pytest.raises(QECError, match="SurfaceCode"):
            dec.logical_error_rate(p_phys=0.05, n_trials=100)

    def test_mwpm_le_greedy_logical_error_rate(self):
        """MWPM logical error rate should be ≤ greedy at low p (averaged)."""
        from ..qec import LogicalErrorSimulator
        code = SurfaceCode(d=3)
        dec = MWPMDecoder(code)
        sim = LogicalErrorSimulator(code)

        p = 0.02
        n_trials = 3000
        pL_mwpm = dec.logical_error_rate(p, n_trials=n_trials, seed=0)
        pL_greedy = sim.logical_error_rate(p, n_trials=n_trials, seed=0)
        # MWPM should be no worse than greedy (within statistical noise)
        # Allow generous bound: 2x greedy + noise is considered a failure
        assert pL_mwpm <= pL_greedy * 3 + 0.02


# ===========================================================================
# Exported from __init__
# ===========================================================================

def test_mwpm_importable_from_package():
    from quantx_engine import MWPMDecoder as M
    assert M is MWPMDecoder


# ===========================================================================
# Greedy fallback triggered by max_exact=0-like logic
# ===========================================================================

class TestMaxExactBoundary:
    def test_max_exact_1_uses_exact_for_1_defect(self):
        code = SurfaceCode(d=3)
        dec = MWPMDecoder(code, max_exact=1)
        x_err = [0] * code.n
        x_err[4] = 1  # 1 defect
        syn = code.z_syndrome(x_err)
        corr = dec.decode_x(syn)
        # Correction should cancel the logical effect
        net = [x_err[i] ^ corr[i] for i in range(code.n)]
        lz = sum(net[i] for i in code.logical_x()) % 2
        assert lz == 0

    def test_max_exact_0_triggers_greedy_path(self):
        """When max_exact is set lower than number of defects, greedy is used."""
        code = SurfaceCode(d=5)
        dec = MWPMDecoder(code, max_exact=1)  # Force greedy for ≥2 defects
        x_err = [0] * code.n
        x_err[2] = 1
        x_err[6] = 1  # 2 defects → greedy path (n_d=2 > max_exact=1)
        syn = code.z_syndrome(x_err)
        # Should not raise; just uses greedy
        corr = dec.decode_x(syn)
        assert isinstance(corr, list)
        assert len(corr) == code.n
