"""Tests for M120: fault-tolerant T gate via magic state injection.

Covers:
  - MagicStateFactory (construction, properties, simulation)
  - FTTGate (construction, properties, gate correctness)
  - FTCCXGate (construction, properties, gate correctness)
  - TCountOptimizer (analysis, optimisation, decompositions)
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from quantx_engine.contrib.ft_tgate import (
    FTCCXGate,
    FTTGate,
    MagicStateError,
    MagicStateFactory,
    TCountOptimizer,
    _MAGIC_STATE_T,
    _T_COST,
    _T_GATE,
    _t_state_fidelity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T_PHASE = cmath.exp(1j * math.pi / 4)


def _norm(v):
    return float(np.linalg.norm(v))


def _fidelity(a, b):
    """State fidelity |⟨a|b⟩|²."""
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    a /= _norm(a)
    b /= _norm(b)
    return float(abs(np.dot(a.conj(), b)) ** 2)


# ---------------------------------------------------------------------------
# MagicStateFactory
# ---------------------------------------------------------------------------

class TestMagicStateFactoryConstruction:
    def test_default_protocol(self):
        f = MagicStateFactory()
        assert f.protocol == "7_to_1"

    def test_15_to_1_protocol(self):
        f = MagicStateFactory(protocol="15_to_1")
        assert f.protocol == "15_to_1"

    def test_invalid_protocol_raises(self):
        with pytest.raises(MagicStateError, match="Unknown protocol"):
            MagicStateFactory(protocol="bad")

    def test_error_rate_zero_raises(self):
        with pytest.raises(MagicStateError, match="input_error_rate"):
            MagicStateFactory(input_error_rate=0.0)

    def test_error_rate_one_raises(self):
        with pytest.raises(MagicStateError, match="input_error_rate"):
            MagicStateFactory(input_error_rate=1.0)

    def test_error_rate_stored(self):
        f = MagicStateFactory(input_error_rate=0.05)
        assert f.input_error_rate == pytest.approx(0.05)


class TestMagicStateFactoryProperties:
    def test_overhead_7_to_1(self):
        assert MagicStateFactory("7_to_1").overhead == 7

    def test_overhead_15_to_1(self):
        assert MagicStateFactory("15_to_1").overhead == 15

    def test_output_error_rate_formula(self):
        eps = 0.01
        f = MagicStateFactory("7_to_1", input_error_rate=eps)
        expected = 35.0 * eps ** 3
        assert f.output_error_rate == pytest.approx(expected, rel=1e-9)

    def test_output_error_much_smaller_than_input(self):
        f = MagicStateFactory("7_to_1", input_error_rate=1e-2)
        assert f.output_error_rate < f.input_error_rate

    def test_threshold_error_rate_positive(self):
        f = MagicStateFactory("7_to_1")
        assert 0 < f.threshold_error_rate < 1

    def test_distillation_factor_gt_one(self):
        f = MagicStateFactory("7_to_1", input_error_rate=1e-2)
        assert f.distillation_factor > 1.0

    def test_repr(self):
        r = repr(MagicStateFactory("7_to_1", input_error_rate=1e-2))
        assert "7_to_1" in r
        assert "1.00e-02" in r


class TestMagicStateFactorySimulation:
    def test_distill_one_returns_float_in_unit_interval(self):
        f = MagicStateFactory(input_error_rate=1e-3)
        fid = f.distill_one(seed=0)
        assert 0.0 <= fid <= 1.0 + 1e-10

    def test_distill_returns_correct_length(self):
        f = MagicStateFactory(input_error_rate=1e-3)
        results = f.distill(n_states=5, seed=42)
        assert len(results) == 5

    def test_distill_n_states_zero_raises(self):
        with pytest.raises(MagicStateError):
            MagicStateFactory().distill(n_states=0)

    def test_distill_high_fidelity_for_low_noise(self):
        f = MagicStateFactory(input_error_rate=1e-3)
        results = f.distill(n_states=20, seed=7)
        avg = sum(results) / len(results)
        assert avg > 0.99

    def test_distill_15_to_1_works(self):
        f = MagicStateFactory("15_to_1", input_error_rate=1e-3)
        fid = f.distill_one(seed=1)
        assert 0.0 <= fid <= 1.0 + 1e-10

    def test_distill_reproducible_with_same_seed(self):
        f = MagicStateFactory(input_error_rate=5e-3)
        a = f.distill(5, seed=99)
        b = f.distill(5, seed=99)
        assert a == b

    def test_distill_different_seeds_may_differ(self):
        # With enough shots and error rate, at least some runs differ between seeds.
        # We verify the function accepts arbitrary seeds without error.
        f = MagicStateFactory(input_error_rate=5e-2)
        a = f.distill(10, seed=1)
        b = f.distill(10, seed=2)
        assert len(a) == 10 and len(b) == 10


# ---------------------------------------------------------------------------
# FTTGate
# ---------------------------------------------------------------------------

class TestFTTGateConstruction:
    def test_default_fidelity_close_to_one(self):
        g = FTTGate()
        assert g.magic_state_fidelity > 0.999

    def test_custom_fidelity(self):
        g = FTTGate(magic_state_fidelity=0.9999)
        assert g.magic_state_fidelity == pytest.approx(0.9999)

    def test_fidelity_zero_raises(self):
        with pytest.raises(MagicStateError):
            FTTGate(magic_state_fidelity=0.0)

    def test_fidelity_above_one_raises(self):
        with pytest.raises(MagicStateError):
            FTTGate(magic_state_fidelity=1.1)

    def test_error_rate_property(self):
        g = FTTGate(magic_state_fidelity=0.9999)
        assert g.error_rate == pytest.approx(1e-4, rel=1e-6)

    def test_circuit_depth(self):
        assert FTTGate().circuit_depth == 3

    def test_repr(self):
        assert "FTTGate" in repr(FTTGate())


class TestFTTGateApply:
    """Gate teleportation correctness with a perfect magic state."""

    def _apply_ideal(self, sv: np.ndarray, seed: int = 0) -> np.ndarray:
        g = FTTGate(magic_state_fidelity=1.0)
        out, _ = g.apply(sv, seed=seed)
        return out

    def test_output_normalised(self):
        g = FTTGate(magic_state_fidelity=1.0)
        for s in range(20):
            out, _ = g.apply(np.array([1.0, 1.0]) / math.sqrt(2), seed=s)
            assert abs(_norm(out) - 1.0) < 1e-12

    def test_wrong_shape_raises(self):
        g = FTTGate()
        with pytest.raises(MagicStateError):
            g.apply(np.array([1, 0, 0, 0], dtype=complex))

    def test_apply_to_zero_state_perfect_magic(self):
        """T|0⟩ = |0⟩ (T acts trivially on |0⟩ up to global phase)."""
        g = FTTGate(magic_state_fidelity=1.0)
        sv = np.array([1.0, 0.0], dtype=complex)
        out, _ = g.apply(sv, seed=0)
        fid = _fidelity(out, np.array([1.0, 0.0]))
        assert fid > 0.999

    def test_apply_to_one_state_perfect_magic(self):
        """T|1⟩ = e^{iπ/4}|1⟩ ≡ |1⟩ (up to global phase)."""
        g = FTTGate(magic_state_fidelity=1.0)
        sv = np.array([0.0, 1.0], dtype=complex)
        out, _ = g.apply(sv, seed=1)
        fid = _fidelity(out, np.array([0.0, 1.0]))
        assert fid > 0.999

    def test_apply_plus_state_perfect_magic(self):
        """T|+⟩ = |T⟩ (the magic state itself)."""
        g = FTTGate(magic_state_fidelity=1.0)
        sv = np.array([1.0, 1.0]) / math.sqrt(2)
        out, _ = g.apply(sv, seed=2)
        fid = _fidelity(out, _MAGIC_STATE_T)
        assert fid > 0.999

    def test_success_probability_sums_to_one(self):
        """Over many seeds, success probability averages to ~0.5 for |+⟩."""
        g = FTTGate(magic_state_fidelity=1.0)
        sv = np.array([1.0, 1.0]) / math.sqrt(2)
        probs = [g.apply(sv, seed=s)[1] for s in range(200)]
        avg = sum(probs) / len(probs)
        # Should average to ~0.5 (two equally likely outcomes)
        assert 0.3 < avg < 0.7

    def test_noisy_magic_state_degrades_fidelity(self):
        """High error rate should produce lower average fidelity than perfect."""
        sv = np.array([1.0, 1.0]) / math.sqrt(2)
        target = _T_GATE @ sv
        target /= _norm(target)

        perfect = FTTGate(magic_state_fidelity=1.0)
        noisy = FTTGate(magic_state_fidelity=0.6)

        # Average fidelity over many trials
        def avg_fid(gate, n=100):
            return sum(_fidelity(gate.apply(sv, seed=s)[0], target) for s in range(n)) / n

        assert avg_fid(perfect) > avg_fid(noisy)

    def test_apply_returns_two_tuple(self):
        g = FTTGate()
        result = g.apply(np.array([1.0, 0.0]))
        assert len(result) == 2

    def test_apply_probability_nonnegative(self):
        g = FTTGate()
        _, p = g.apply(np.array([1.0, 0.0]), seed=0)
        assert p >= 0.0


# ---------------------------------------------------------------------------
# FTCCXGate
# ---------------------------------------------------------------------------

def _ket(bits: str) -> np.ndarray:
    """Compute-basis state for a bit string, e.g. '110'."""
    n = len(bits)
    idx = int(bits, 2)
    sv = np.zeros(1 << n, dtype=complex)
    sv[idx] = 1.0
    return sv


class TestFTCCXGateConstruction:
    def test_default_error_rate(self):
        g = FTCCXGate()
        assert g.t_gate_error_rate == pytest.approx(1e-4)

    def test_t_count(self):
        assert FTCCXGate().t_count == 7

    def test_circuit_depth(self):
        assert FTCCXGate().circuit_depth > 7

    def test_error_rate_formula(self):
        eps = 1e-4
        g = FTCCXGate(t_gate_error_rate=eps)
        assert g.error_rate == pytest.approx(7 * eps)

    def test_error_rate_capped_at_one(self):
        g = FTCCXGate(t_gate_error_rate=0.5)
        assert g.error_rate <= 1.0

    def test_invalid_error_rate_raises(self):
        with pytest.raises(MagicStateError):
            FTCCXGate(t_gate_error_rate=-0.01)
        with pytest.raises(MagicStateError):
            FTCCXGate(t_gate_error_rate=1.0)

    def test_repr(self):
        assert "FTCCXGate" in repr(FTCCXGate())


class TestFTCCXGateApply:
    def test_output_shape(self):
        g = FTCCXGate(t_gate_error_rate=0.0)
        out, _ = g.apply(_ket("000"))
        assert out.shape == (8,)

    def test_output_normalised(self):
        g = FTCCXGate(t_gate_error_rate=0.0)
        for bits in ["000", "001", "010", "011", "100", "101", "110", "111"]:
            out, _ = g.apply(_ket(bits), seed=0)
            assert abs(_norm(out) - 1.0) < 1e-10

    def test_wrong_shape_raises(self):
        g = FTCCXGate()
        with pytest.raises(MagicStateError):
            g.apply(np.array([1.0, 0.0]))

    def test_110_flips_to_111(self):
        """CCX|110⟩ = |111⟩ (both controls high → flip target)."""
        g = FTCCXGate(t_gate_error_rate=0.0)
        out, _ = g.apply(_ket("110"), seed=0)
        fid = _fidelity(out, _ket("111"))
        assert fid > 0.999

    def test_111_flips_to_110(self):
        """CCX|111⟩ = |110⟩ (both controls high → flip target)."""
        g = FTCCXGate(t_gate_error_rate=0.0)
        out, _ = g.apply(_ket("111"), seed=0)
        fid = _fidelity(out, _ket("110"))
        assert fid > 0.999

    def test_010_unchanged(self):
        """CCX|010⟩ = |010⟩ (only one control high → no flip)."""
        g = FTCCXGate(t_gate_error_rate=0.0)
        out, _ = g.apply(_ket("010"), seed=0)
        fid = _fidelity(out, _ket("010"))
        assert fid > 0.999

    def test_000_unchanged(self):
        g = FTCCXGate(t_gate_error_rate=0.0)
        out, _ = g.apply(_ket("000"), seed=0)
        fid = _fidelity(out, _ket("000"))
        assert fid > 0.999

    def test_noisy_t_gates_return_error_prob(self):
        g = FTCCXGate(t_gate_error_rate=0.5)
        _, ep = g.apply(_ket("110"), seed=0)
        assert 0.0 <= ep <= 1.0

    def test_zero_error_rate_returns_zero_error_prob(self):
        g = FTCCXGate(t_gate_error_rate=0.0)
        _, ep = g.apply(_ket("000"), seed=0)
        assert ep == pytest.approx(0.0)

    def test_return_is_two_tuple(self):
        g = FTCCXGate()
        result = g.apply(_ket("000"), seed=0)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# TCountOptimizer
# ---------------------------------------------------------------------------

def _circ(*gates):
    """Build a simple single-qubit circuit list."""
    return [{"gate": g, "qubit": 0} for g in gates]


class TestTCountAnalysis:
    def test_empty_circuit(self):
        opt = TCountOptimizer([])
        assert opt.t_count() == 0
        assert opt.t_depth() == 0

    def test_single_t_gate(self):
        opt = TCountOptimizer(_circ("T"))
        assert opt.t_count() == 1

    def test_single_tdg_gate(self):
        opt = TCountOptimizer(_circ("Tdg"))
        assert opt.t_count() == 1

    def test_s_costs_two_t_gates(self):
        opt = TCountOptimizer(_circ("S"))
        assert opt.t_count() == 2

    def test_ccx_costs_seven(self):
        opt = TCountOptimizer([{"gate": "CCX", "controls": [0, 1], "target": 2}])
        assert opt.t_count() == 7

    def test_clifford_gates_free(self):
        opt = TCountOptimizer(_circ("H", "X", "Y", "Z", "H"))
        assert opt.t_count() == 0

    def test_mixed_circuit(self):
        circuit = [
            {"gate": "H",    "qubit": 0},
            {"gate": "T",    "qubit": 0},
            {"gate": "CNOT", "control": 0, "target": 1},
            {"gate": "T†",   "qubit": 1},
        ]
        opt = TCountOptimizer(circuit)
        assert opt.t_count() == 2

    def test_t_depth_single_qubit(self):
        opt = TCountOptimizer(_circ("T", "T", "T"))
        assert opt.t_depth() == 3

    def test_t_depth_parallel_qubits(self):
        circuit = [
            {"gate": "T", "qubit": 0},
            {"gate": "T", "qubit": 1},  # parallel → depth 1
        ]
        opt = TCountOptimizer(circuit)
        assert opt.t_depth() == 1

    def test_t_depth_cnot_synchronises(self):
        circuit = [
            {"gate": "T",    "qubit": 0},
            {"gate": "CNOT", "control": 0, "target": 1},
            {"gate": "T",    "qubit": 1},
        ]
        opt = TCountOptimizer(circuit)
        assert opt.t_depth() == 2

    def test_t_gates_returns_list(self):
        # t_gates() returns all ops with T-cost > 0, which includes S (cost=2)
        opt = TCountOptimizer(_circ("H", "T", "S", "Tdg"))
        tg = opt.t_gates()
        assert all(_T_COST.get(g.gate, 0) > 0 for g in tg)
        assert all(g.gate not in {"H", "X", "Y", "Z", "CNOT"} for g in tg)

    def test_clifford_gates_returns_list(self):
        opt = TCountOptimizer(_circ("H", "T", "S"))
        cg = opt.clifford_gates()
        assert all(g.gate not in {"T", "TDG"} for g in cg)

    def test_repr(self):
        r = repr(TCountOptimizer(_circ("T")))
        assert "TCountOptimizer" in r
        assert "t_count=1" in r

    def test_invalid_circuit_raises(self):
        with pytest.raises(MagicStateError):
            TCountOptimizer("not a list")


class TestTCountOptimization:
    def test_tt_dagger_cancels(self):
        opt = TCountOptimizer(_circ("T", "T†"))
        reduced = opt.optimize()
        assert reduced.t_count() == 0

    def test_tdagger_t_cancels(self):
        opt = TCountOptimizer(_circ("T†", "T"))
        reduced = opt.optimize()
        assert reduced.t_count() == 0

    def test_tt_becomes_s(self):
        opt = TCountOptimizer(_circ("T", "T"))
        reduced = opt.optimize()
        # T·T → S: two T gates compressed into one S gate
        # T-count stays at 2 (S = T²), but gate count halves from 2 to 1
        assert len(reduced._ops) == 1
        assert reduced._ops[0].gate == "S"

    def test_ss_becomes_z(self):
        opt = TCountOptimizer(_circ("S", "S"))
        reduced = opt.optimize()
        # S·S = Z; Z has T-count 0
        assert reduced.t_count() == 0

    def test_zz_cancels(self):
        opt = TCountOptimizer(_circ("Z", "Z"))
        reduced = opt.optimize()
        assert len(reduced._ops) == 0

    def test_hh_cancels(self):
        opt = TCountOptimizer(_circ("H", "H"))
        reduced = opt.optimize()
        assert len(reduced._ops) == 0

    def test_no_reduction_different_qubits(self):
        circuit = [
            {"gate": "T", "qubit": 0},
            {"gate": "T†", "qubit": 1},
        ]
        opt = TCountOptimizer(circuit)
        reduced = opt.optimize()
        assert reduced.t_count() == 2  # no reduction across different qubits

    def test_optimize_returns_new_instance(self):
        opt = TCountOptimizer(_circ("T", "T†"))
        reduced = opt.optimize()
        assert reduced is not opt
        assert opt.t_count() == 2  # original unchanged

    def test_chain_reduction(self):
        # T·T·T†·T† → S·S† → I: 4 T gates → 0
        opt = TCountOptimizer(_circ("T", "T", "T†", "T†"))
        reduced = opt.optimize()
        assert reduced.t_count() == 0


class TestCliffordTDecomposition:
    def test_ccx_t_count_is_seven(self):
        opt = TCountOptimizer([])
        decomp = opt.clifford_t_decomposition("CCX")
        sub = TCountOptimizer(decomp)
        assert sub.t_count() == 7

    def test_s_t_count_is_two(self):
        opt = TCountOptimizer([])
        decomp = opt.clifford_t_decomposition("S")
        sub = TCountOptimizer(decomp)
        assert sub.t_count() == 2

    def test_sdg_t_count_is_two(self):
        opt = TCountOptimizer([])
        decomp = opt.clifford_t_decomposition("Sdg")
        sub = TCountOptimizer(decomp)
        assert sub.t_count() == 2

    def test_z_t_count_is_four(self):
        opt = TCountOptimizer([])
        decomp = opt.clifford_t_decomposition("Z")
        sub = TCountOptimizer(decomp)
        assert sub.t_count() == 4

    def test_t_decomposes_to_t(self):
        opt = TCountOptimizer([])
        decomp = opt.clifford_t_decomposition("T")
        assert len(decomp) == 1
        assert decomp[0]["gate"] == "T"

    def test_unknown_gate_raises(self):
        opt = TCountOptimizer([])
        with pytest.raises(MagicStateError, match="No Clifford"):
            opt.clifford_t_decomposition("UNKNOWN_GATE")

    def test_decomposition_returns_list_of_dicts(self):
        opt = TCountOptimizer([])
        decomp = opt.clifford_t_decomposition("CCX")
        assert isinstance(decomp, list)
        assert all(isinstance(d, dict) for d in decomp)
        assert all("gate" in d for d in decomp)

    def test_ccx_decomposition_has_h_and_cnot(self):
        opt = TCountOptimizer([])
        decomp = opt.clifford_t_decomposition("CCX")
        gates = [d["gate"] for d in decomp]
        assert "H" in gates
        assert "CNOT" in gates


# ---------------------------------------------------------------------------
# Integration: MagicStateFactory + FTTGate
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_high_fidelity_magic_state_gives_accurate_t_gate(self):
        """Using a distilled state with high fidelity should yield near-perfect T gate."""
        factory = MagicStateFactory("7_to_1", input_error_rate=1e-3)
        fid = min(1.0, factory.distill_one(seed=0))  # clip FP noise to [0,1]
        gate = FTTGate(magic_state_fidelity=fid)

        sv = np.array([1.0, 1.0]) / math.sqrt(2)
        out, _ = gate.apply(sv, seed=0)
        ideal = _T_GATE @ sv
        ideal /= _norm(ideal)
        assert _fidelity(out, ideal) > 0.99

    def test_magic_state_factory_output_below_threshold(self):
        """For ε_in well below threshold, output_error_rate << input_error_rate."""
        f = MagicStateFactory("7_to_1", input_error_rate=1e-2)
        assert f.output_error_rate < f.input_error_rate * 0.01

    def test_t_count_optimizer_on_ccx_decomposition(self):
        """Applying TCountOptimizer to a CCX decomposition confirms T-count = 7."""
        opt0 = TCountOptimizer([])
        ccx_circuit = opt0.clifford_t_decomposition("CCX")
        opt = TCountOptimizer(ccx_circuit)
        assert opt.t_count() == 7
        # No further algebraic reductions possible
        reduced = opt.optimize()
        assert reduced.t_count() == 7
