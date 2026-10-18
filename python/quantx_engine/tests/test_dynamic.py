"""Tests for dynamic circuits — M86."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quantx_engine.contrib.dynamic import (
    DynamicCircuit,
    DynamicCircuitError,
    DynamicResult,
    DynamicSimulator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bell_dynamic() -> DynamicCircuit:
    """Bell circuit with final measurements."""
    return DynamicCircuit(2, 2).h(0).cx(0, 1).measure_all()


def _ground_state() -> DynamicCircuit:
    return DynamicCircuit(1, 1).measure(0, 0)


# ---------------------------------------------------------------------------
# DynamicCircuit — construction
# ---------------------------------------------------------------------------

class TestDynamicCircuitConstruction:
    def test_default_cbits_equals_qubits(self):
        c = DynamicCircuit(3)
        assert c.n_cbits == 3

    def test_explicit_cbits(self):
        c = DynamicCircuit(3, 5)
        assert c.n_cbits == 5

    def test_invalid_n_qubits(self):
        with pytest.raises(DynamicCircuitError):
            DynamicCircuit(0)

    def test_add_gate(self):
        c = DynamicCircuit(1).h(0)
        assert len(c.operations) == 1
        assert c.operations[0]["name"] == "h"

    def test_fluent_chaining(self):
        c = DynamicCircuit(2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
        assert len(c.operations) == 4

    def test_measure_all(self):
        c = DynamicCircuit(3, 3).measure_all()
        assert len(c.operations) == 3

    def test_out_of_range_qubit(self):
        with pytest.raises(DynamicCircuitError, match="Qubit index"):
            DynamicCircuit(2).h(5)

    def test_out_of_range_cbit(self):
        with pytest.raises(DynamicCircuitError, match="Classical bit"):
            DynamicCircuit(2, 1).measure(0, 5)

    def test_if_bit_invalid_value(self):
        with pytest.raises(DynamicCircuitError, match="0 or 1"):
            DynamicCircuit(2, 1).if_bit(cbit=0, value=2, name="x", qubits=[1])

    def test_repeat_until_invalid_max_iter(self):
        with pytest.raises(DynamicCircuitError, match="max_iterations"):
            DynamicCircuit(2, 1).repeat_until(0, 0, [], max_iterations=0)

    def test_repr(self):
        c = DynamicCircuit(2)
        assert "DynamicCircuit" in repr(c)


# ---------------------------------------------------------------------------
# DynamicSimulator — construction
# ---------------------------------------------------------------------------

class TestDynamicSimulatorConstruction:
    def test_invalid_shots(self):
        with pytest.raises(DynamicCircuitError, match="shots"):
            DynamicSimulator(shots=0)

    def test_invalid_circuit_type(self):
        with pytest.raises(DynamicCircuitError):
            DynamicSimulator(shots=1).run({"not": "a circuit"})

    def test_returns_dynamic_result(self):
        r = DynamicSimulator(shots=1).run(_ground_state())
        assert isinstance(r, DynamicResult)


# ---------------------------------------------------------------------------
# Measurement — basic
# ---------------------------------------------------------------------------

class TestMeasurementBasic:
    def test_ground_state_always_zero(self):
        r = DynamicSimulator(shots=50, seed=0).run(_ground_state())
        assert r.counts == {"0": 50}

    def test_x_gate_always_one(self):
        circ = DynamicCircuit(1, 1).x(0).measure(0, 0)
        r = DynamicSimulator(shots=50).run(circ)
        assert r.counts == {"1": 50}

    def test_h_gate_approx_50_50(self):
        circ = DynamicCircuit(1, 1).h(0).measure(0, 0)
        r = DynamicSimulator(shots=1000, seed=7).run(circ)
        p0 = r.counts.get("0", 0) / r.n_shots
        assert abs(p0 - 0.5) < 0.07

    def test_counts_sum_to_n_shots(self):
        r = DynamicSimulator(shots=100, seed=1).run(_bell_dynamic())
        assert sum(r.counts.values()) == 100

    def test_bell_only_00_and_11(self):
        r = DynamicSimulator(shots=200, seed=2).run(_bell_dynamic())
        assert set(r.counts.keys()).issubset({"00", "11"})

    def test_bell_approx_50_50(self):
        r = DynamicSimulator(shots=1000, seed=3).run(_bell_dynamic())
        p00 = r.counts.get("00", 0) / r.n_shots
        assert abs(p00 - 0.5) < 0.07

    def test_bitstring_length_equals_n_cbits(self):
        circ = DynamicCircuit(3, 3).measure_all()
        r = DynamicSimulator(shots=10).run(circ)
        for key in r.counts:
            assert len(key) == 3

    def test_quasi_dists_sums_to_one(self):
        r = DynamicSimulator(shots=100, seed=0).run(_bell_dynamic())
        assert sum(r.quasi_dists.values()) == pytest.approx(1.0, abs=1e-10)

    def test_most_likely_returns_string(self):
        circ = DynamicCircuit(1, 1).x(0).measure(0, 0)
        r = DynamicSimulator(shots=10).run(circ)
        assert r.most_likely() == "1"

    def test_cbits_history_length(self):
        r = DynamicSimulator(shots=5, seed=0).run(_bell_dynamic())
        assert len(r.cbits_history) == 5

    def test_seed_reproducibility(self):
        circ = DynamicCircuit(1, 1).h(0).measure(0, 0)
        ra = DynamicSimulator(shots=100, seed=42).run(circ)
        rb = DynamicSimulator(shots=100, seed=42).run(circ)
        assert ra.counts == rb.counts

    def test_metadata_fields(self):
        r = DynamicSimulator(shots=5).run(_bell_dynamic())
        assert r.metadata["n_qubits"] == 2
        assert r.metadata["shots"] == 5


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_ground_state_unchanged(self):
        circ = DynamicCircuit(1, 1).reset(0).measure(0, 0)
        r = DynamicSimulator(shots=50).run(circ)
        assert r.counts == {"0": 50}

    def test_reset_after_x_returns_zero(self):
        circ = DynamicCircuit(1, 1).x(0).reset(0).measure(0, 0)
        r = DynamicSimulator(shots=50).run(circ)
        assert r.counts == {"0": 50}

    def test_reset_after_measure_one_gives_zero(self):
        # X flips to |1⟩; measure, reset, measure again → always 0
        circ = (
            DynamicCircuit(1, 2)
            .x(0)
            .measure(0, 0)
            .reset(0)
            .measure(0, 1)
        )
        r = DynamicSimulator(shots=50).run(circ)
        # MSB-first key: key[0]=cbit_1, key[-1]=cbit_0
        # cbit_1 = 0 (after reset), cbit_0 = 1 (X result before reset)
        for key in r.counts:
            assert key[0] == "0", f"Expected cbit_1=0 (leftmost), got {key}"
            assert key[-1] == "1", f"Expected cbit_0=1 (rightmost), got {key}"


# ---------------------------------------------------------------------------
# Classical feed-forward (if_bit)
# ---------------------------------------------------------------------------

class TestIfBit:
    def test_if_bit_applies_when_condition_met(self):
        # |1⟩ after X, cbit=1 → X again → back to |0⟩
        circ = (
            DynamicCircuit(1, 1)
            .x(0)
            .measure(0, 0)
            .if_bit(cbit=0, value=1, name="x", qubits=[0])
            .measure(0, 0)
        )
        # cbit is overwritten on second measure
        r = DynamicSimulator(shots=50).run(circ)
        assert r.counts == {"0": 50}

    def test_if_bit_no_apply_when_condition_not_met(self):
        # |0⟩, cbit=0, condition=value=1 → gate NOT applied → stays 0
        circ = (
            DynamicCircuit(1, 1)
            .measure(0, 0)
            .if_bit(cbit=0, value=1, name="x", qubits=[0])
            .measure(0, 0)
        )
        r = DynamicSimulator(shots=50).run(circ)
        assert r.counts == {"0": 50}

    def test_teleportation_feedforward(self):
        """Simple Bell-then-feedforward: X correction on qubit 1 restores state.

        Prepare |0⟩ on q1, entangle, measure q0.
        If q0==1 apply X on q1 → q1 should always be |0⟩ at end.
        """
        circ = (
            DynamicCircuit(n_qubits=2, n_cbits=2)
            .h(0)
            .cx(0, 1)
            .measure(0, 0)
            .if_bit(cbit=0, value=1, name="x", qubits=[1])
            .measure(1, 1)
        )
        r = DynamicSimulator(shots=200, seed=11).run(circ)
        # MSB-first key: key[0]=cbit_1 (q1 after correction), key[-1]=cbit_0 (random)
        # cbit_1 must always be 0 (q1 corrected to |0⟩)
        for key in r.counts:
            assert key[0] == "0", (
                f"Feed-forward failed: q1 ended in |1⟩, bitstring {key}"
            )

    def test_x_correction_after_bell_basis_measure(self):
        """Measure q1 → if 1, apply X on q0 → q0 always |0⟩ at end."""
        # H(0), CX(0,1) → Bell (|00⟩+|11⟩)/√2
        # Measure q1 → cbit_1: 50% |00⟩, 50% |11⟩
        # if cbit_1=1 → X on q0: |11⟩ → X(q0) → |01⟩, q0=|0⟩
        # Either way q0=|0⟩, measure → cbit_0=0 always
        circ = (
            DynamicCircuit(n_qubits=2, n_cbits=2)
            .h(0)
            .cx(0, 1)
            .measure(1, 1)
            .if_bit(cbit=1, value=1, name="x", qubits=[0])
            .measure(0, 0)
        )
        r = DynamicSimulator(shots=200, seed=22).run(circ)
        # MSB-first key: key[0]=cbit_1 (random), key[-1]=cbit_0=0 always
        for key in r.counts:
            assert key[-1] == "0", (
                f"X correction failed: q0 not in |0⟩, bitstring {key}"
            )


# ---------------------------------------------------------------------------
# Repeat-until-success
# ---------------------------------------------------------------------------

class TestRepeatUntilSuccess:
    def test_immediate_success(self):
        """Body always produces success_value=0 on first try (empty body)."""
        # Classical register initialised to 0, success_value=0 → immediate exit
        circ = DynamicCircuit(1, 1).repeat_until(
            success_cbit=0,
            success_value=0,
            body=[],          # no ops → cbit stays 0 (init value)
        )
        r = DynamicSimulator(shots=10).run(circ)
        assert r.n_shots == 10

    def test_x_then_measure_success_on_one(self):
        """Apply X + measure; success when cbit=1 (always on first try)."""
        body = [
            {"type": "gate", "name": "x", "qubits": [0], "params": []},
            {"type": "measure", "qubit": 0, "cbit": 0},
        ]
        circ = DynamicCircuit(1, 1).repeat_until(
            success_cbit=0,
            success_value=1,
            body=body,
        )
        r = DynamicSimulator(shots=50).run(circ)
        # Each shot exits after the first body execution, cbit should be 1
        for cbs in r.cbits_history:
            assert cbs[0] == 1

    def test_probabilistic_success_terminates(self):
        """H + measure: success=1 with p=0.5; RUS loop terminates."""
        body = [
            {"type": "gate", "name": "h", "qubits": [0], "params": []},
            {"type": "measure", "qubit": 0, "cbit": 0},
        ]
        circ = DynamicCircuit(1, 1).repeat_until(
            success_cbit=0,
            success_value=1,
            body=body,
            reset_qubits=[0],
            max_iterations=50,
        )
        r = DynamicSimulator(shots=100, seed=5).run(circ)
        # Every shot should succeed (cbit 0 = 1) within 50 tries
        for cbs in r.cbits_history:
            assert cbs[0] == 1


# ---------------------------------------------------------------------------
# Noisy simulation
# ---------------------------------------------------------------------------

class TestNoisyDynamic:
    def test_noise_blurs_deterministic_circuit(self):
        from quantx_engine.contrib.noise import DepolarizingNoise
        noise = DepolarizingNoise(0.3)
        circ = DynamicCircuit(1, 1).x(0).measure(0, 0)
        r = DynamicSimulator(noise_model=noise, shots=500, seed=9).run(circ)
        # Some shots may have X on qubit after noise, ending in |0⟩
        p1 = r.counts.get("1", 0) / r.n_shots
        # Should be dominated by 1 but not 100% due to noise
        assert p1 > 0.5

    def test_noise_counts_sum_to_shots(self):
        from quantx_engine.contrib.noise import DepolarizingNoise
        noise = DepolarizingNoise(0.05)
        r = DynamicSimulator(noise_model=noise, shots=100, seed=0).run(_bell_dynamic())
        assert sum(r.counts.values()) == 100


# ---------------------------------------------------------------------------
# Parametric gates in dynamic circuits
# ---------------------------------------------------------------------------

class TestParametricGatesDynamic:
    def test_rx_pi_flips_qubit(self):
        circ = (
            DynamicCircuit(1, 1)
            .add_gate("rx", [0], [math.pi])
            .measure(0, 0)
        )
        r = DynamicSimulator(shots=50).run(circ)
        assert r.counts.get("1", 0) == 50

    def test_ry_pi_over_2_superposition(self):
        circ = (
            DynamicCircuit(1, 1)
            .add_gate("ry", [0], [math.pi / 2])
            .measure(0, 0)
        )
        r = DynamicSimulator(shots=1000, seed=13).run(circ)
        p0 = r.counts.get("0", 0) / r.n_shots
        assert abs(p0 - 0.5) < 0.07
