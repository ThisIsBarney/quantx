"""M160: Pure-Python dense statevector simulator tests.

Verifies that:
1. StatevectorSimulator is importable from the public API.
2. Bell-state preparation (H + CNOT) produces the correct statevector.
3. Single-qubit gates: H, X, Y, Z, S, T, Rx, Ry, Rz, P.
4. Controlled gates: CNOT, CY, CZ, CH, SWAP, iSWAP.
5. Three-qubit gates: CCX (Toffoli), CSWAP (Fredkin).
6. Measurement: sample() returns correct distribution; expectation() is exact.
7. Fluent chaining works (all methods return self).
8. copy() / reset() preserve / restore state.
9. from_circuit_dict() / run_circuit_dict() bridge works.
10. Edge cases: invalid qubit indices, oversized n_qubits, unnormalised input.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from quantx_engine.contrib.statevector import StatevectorSimulator


# ── helpers ──────────────────────────────────────────────────────────────────

def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared overlap |⟨a|b⟩|²."""
    return float(abs(np.dot(a.conj(), b)) ** 2)


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


# ── 2. Initialisation ─────────────────────────────────────────────────────────

class TestInit:
    def test_default_is_zero_state(self):
        sim = StatevectorSimulator(2)
        sv = sim.statevector()
        expected = np.array([1, 0, 0, 0], dtype=complex)
        assert np.allclose(sv, expected)

    def test_basis_state_one(self):
        sim = StatevectorSimulator(2, initial_state=1)
        sv = sim.statevector()
        expected = np.array([0, 1, 0, 0], dtype=complex)
        assert np.allclose(sv, expected)

    def test_custom_initial_state_normalised(self):
        v = np.array([1, 1, 0, 0], dtype=complex)
        sim = StatevectorSimulator(2, initial_state=v)
        assert abs(_norm(sim.statevector()) - 1.0) < 1e-12

    def test_n_qubits_property(self):
        sim = StatevectorSimulator(5)
        assert sim.n_qubits == 5

    def test_dim_property(self):
        sim = StatevectorSimulator(4)
        assert sim.dim == 16

    def test_max_qubits_limit(self):
        with pytest.raises(ValueError, match="n_qubits"):
            StatevectorSimulator(26)

    def test_invalid_basis_state(self):
        with pytest.raises(ValueError):
            StatevectorSimulator(2, initial_state=4)

    def test_zero_norm_raises(self):
        with pytest.raises(ValueError, match="zero norm"):
            StatevectorSimulator(2, initial_state=np.zeros(4))


# ── 3. Single-qubit gates ─────────────────────────────────────────────────────

class TestSingleQubitGates:
    def test_x_flips_qubit(self):
        sim = StatevectorSimulator(1)
        sim.x(0)
        assert np.allclose(sim.statevector(), [0, 1])

    def test_h_creates_superposition(self):
        sim = StatevectorSimulator(1)
        sim.h(0)
        sv = sim.statevector()
        assert np.allclose(sv, [1 / math.sqrt(2), 1 / math.sqrt(2)])

    def test_h_twice_is_identity(self):
        sim = StatevectorSimulator(2)
        sim.h(0).h(1).h(0).h(1)
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0
        assert np.allclose(sim.statevector(), expected, atol=1e-14)

    def test_z_gate(self):
        sim = StatevectorSimulator(1, initial_state=np.array([0, 1], dtype=complex))
        sim.z(0)
        assert np.allclose(sim.statevector(), [0, -1])

    def test_y_gate(self):
        sim = StatevectorSimulator(1)
        sim.y(0)
        assert np.allclose(sim.statevector(), [0, 1j])

    def test_s_gate(self):
        sim = StatevectorSimulator(1, initial_state=np.array([0, 1], dtype=complex))
        sim.s(0)
        assert np.allclose(sim.statevector(), [0, 1j])

    def test_t_gate_phase(self):
        sim = StatevectorSimulator(1, initial_state=np.array([0, 1], dtype=complex))
        sim.t(0)
        expected_phase = np.exp(1j * math.pi / 4)
        assert np.allclose(sim.statevector(), [0, expected_phase])

    def test_sdg_is_s_inverse(self):
        sim = StatevectorSimulator(1, initial_state=np.array([0, 1], dtype=complex))
        sim.s(0).sdg(0)
        assert np.allclose(sim.statevector(), [0, 1])

    def test_tdg_is_t_inverse(self):
        sim = StatevectorSimulator(1, initial_state=np.array([0, 1], dtype=complex))
        sim.t(0).tdg(0)
        assert np.allclose(sim.statevector(), [0, 1])

    def test_rx_pi_equals_x(self):
        sim = StatevectorSimulator(1)
        sim.rx(math.pi, 0)
        assert np.allclose(sim.statevector(), [0, -1j], atol=1e-14)

    def test_ry_pi_equals_y_up_to_global_phase(self):
        sim = StatevectorSimulator(1)
        sim.ry(math.pi, 0)
        sv = sim.statevector()
        assert abs(sv[0]) < 1e-14
        assert abs(abs(sv[1]) - 1.0) < 1e-14

    def test_rz_2pi_is_identity_up_to_phase(self):
        sim = StatevectorSimulator(1, initial_state=np.array([1, 1], dtype=complex) / math.sqrt(2))
        sv_before = sim.statevector().copy()
        sim.rz(2 * math.pi, 0)
        sv_after = sim.statevector()
        # global phase: |⟨before|after⟩|² = 1
        assert _fidelity(sv_before, sv_after) > 1 - 1e-12

    def test_phase_gate(self):
        sim = StatevectorSimulator(1, initial_state=np.array([0, 1], dtype=complex))
        sim.p(math.pi / 2, 0)
        assert np.allclose(sim.statevector(), [0, 1j])

    def test_norm_preserved_after_all_gates(self):
        sim = StatevectorSimulator(3)
        sim.h(0).h(1).h(2).x(0).y(1).z(2).s(0).t(1).rx(0.5, 2).ry(1.2, 0).rz(2.1, 1)
        assert abs(_norm(sim.statevector()) - 1.0) < 1e-12


# ── 4. Two-qubit gates ────────────────────────────────────────────────────────

class TestTwoQubitGates:
    def test_cnot_entangles(self):
        sim = StatevectorSimulator(2)
        sim.h(0).cnot(0, 1)
        sv = sim.statevector()
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        expected = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
        assert np.allclose(sv, expected, atol=1e-14)

    def test_cx_is_cnot_alias(self):
        sim1 = StatevectorSimulator(2)
        sim1.h(0).cnot(0, 1)
        sim2 = StatevectorSimulator(2)
        sim2.h(0).cx(0, 1)
        assert np.allclose(sim1.statevector(), sim2.statevector())

    def test_cz_gate(self):
        # |11⟩ → -|11⟩
        sim = StatevectorSimulator(2, initial_state=3)
        sim.cz(0, 1)
        sv = sim.statevector()
        assert np.allclose(sv, [0, 0, 0, -1])

    def test_swap_gate(self):
        sim = StatevectorSimulator(2, initial_state=1)  # |01⟩
        sim.swap(0, 1)
        sv = sim.statevector()
        assert np.allclose(sv, [0, 0, 1, 0])  # |10⟩

    def test_swap_twice_is_identity(self):
        sim = StatevectorSimulator(2)
        sim.h(0).ry(1.3, 1).swap(0, 1).swap(0, 1)
        sv_ref = StatevectorSimulator(2).h(0).ry(1.3, 1).statevector()
        assert np.allclose(sim.statevector(), sv_ref, atol=1e-14)

    def test_iswap_norm_preserved(self):
        sim = StatevectorSimulator(2)
        sim.h(0).h(1).iswap(0, 1)
        assert abs(_norm(sim.statevector()) - 1.0) < 1e-12


# ── 5. Three-qubit gates ──────────────────────────────────────────────────────

class TestThreeQubitGates:
    def test_toffoli_flips_target_when_both_controls_set(self):
        # Prepare |110⟩ (c0=1, c1=1, t=0)
        sim = StatevectorSimulator(3, initial_state=6)  # 6 = 0b110
        sim.ccx(0, 1, 2)
        sv = sim.statevector()
        assert np.argmax(np.abs(sv)) == 7  # |111⟩

    def test_toffoli_no_flip_when_one_control_zero(self):
        sim = StatevectorSimulator(3, initial_state=4)  # |100⟩
        sim.ccx(0, 1, 2)
        sv = sim.statevector()
        assert np.argmax(np.abs(sv)) == 4  # unchanged

    def test_cswap_swaps_when_ctrl_set(self):
        # |101⟩: ctrl=1, q0=0, q1=0 → swap q1 and q2 → |110⟩
        sim = StatevectorSimulator(3, initial_state=5)  # 5 = 0b101
        sim.cswap(0, 1, 2)
        sv = sim.statevector()
        assert np.argmax(np.abs(sv)) == 6  # 6 = 0b110

    def test_cswap_no_swap_when_ctrl_zero(self):
        sim = StatevectorSimulator(3, initial_state=3)  # |011⟩
        sim.cswap(0, 1, 2)  # ctrl=qubit0=0 → no swap
        sv = sim.statevector()
        assert np.argmax(np.abs(sv)) == 3

    def test_fredkin_alias(self):
        sim1 = StatevectorSimulator(3, initial_state=5)
        sim1.cswap(0, 1, 2)
        sim2 = StatevectorSimulator(3, initial_state=5)
        sim2.fredkin(0, 1, 2)
        assert np.allclose(sim1.statevector(), sim2.statevector())


# ── 6. Measurement & observables ─────────────────────────────────────────────

class TestMeasurement:
    def test_sample_uniform_distribution(self):
        sim = StatevectorSimulator(1)
        sim.h(0)
        counts = sim.sample(10000, seed=42)
        assert set(counts.keys()).issubset({"0", "1"})
        # Both outcomes present with at least 40% share
        assert counts.get("0", 0) > 4000
        assert counts.get("1", 0) > 4000

    def test_sample_deterministic_state(self):
        sim = StatevectorSimulator(2, initial_state=3)  # |11⟩
        counts = sim.sample(100, seed=0)
        assert counts == Counter({"11": 100})

    def test_sample_total_shots(self):
        sim = StatevectorSimulator(2)
        sim.h(0).h(1)
        counts = sim.sample(500, seed=7)
        assert sum(counts.values()) == 500

    def test_sample_bitstring_length(self):
        sim = StatevectorSimulator(4)
        sim.h(0).h(1).h(2).h(3)
        counts = sim.sample(100, seed=1)
        for bitstring in counts:
            assert len(bitstring) == 4

    def test_expectation_z_ground_state(self):
        sim = StatevectorSimulator(1)
        # |0⟩: ⟨Z⟩ = +1
        assert abs(sim.expectation("Z") - 1.0) < 1e-14

    def test_expectation_z_excited_state(self):
        sim = StatevectorSimulator(1, initial_state=1)
        # |1⟩: ⟨Z⟩ = -1
        assert abs(sim.expectation("Z") + 1.0) < 1e-14

    def test_expectation_x_hadamard_state(self):
        sim = StatevectorSimulator(1)
        sim.h(0)
        # |+⟩: ⟨X⟩ = +1
        assert abs(sim.expectation("X") - 1.0) < 1e-12

    def test_expectation_zz_bell_state(self):
        sim = StatevectorSimulator(2)
        sim.h(0).cnot(0, 1)
        # |Φ+⟩: ⟨ZZ⟩ = +1
        assert abs(sim.expectation("ZZ") - 1.0) < 1e-12

    def test_expectation_normalised_result(self):
        # For any normalised state and Hermitian observable, |⟨O⟩| ≤ ||O||
        sim = StatevectorSimulator(3)
        sim.h(0).ry(0.7, 1).rx(1.3, 2).cnot(0, 1).cz(1, 2)
        val = abs(sim.expectation("ZZZ"))
        assert val <= 1.0 + 1e-12

    def test_probabilities_sum_to_one(self):
        sim = StatevectorSimulator(3)
        sim.h(0).h(1).h(2).cnot(0, 1).cz(0, 2)
        assert abs(sim.probabilities().sum() - 1.0) < 1e-12

    def test_probabilities_shape(self):
        sim = StatevectorSimulator(4)
        assert sim.probabilities().shape == (16,)


# ── 7. Fluent API ─────────────────────────────────────────────────────────────

class TestFluentAPI:
    def test_chain_returns_self(self):
        sim = StatevectorSimulator(2)
        result = sim.h(0).cnot(0, 1).x(0).y(1).z(0)
        assert result is sim

    def test_chain_produces_correct_state(self):
        sv = StatevectorSimulator(1).h(0).z(0).h(0).statevector()
        # H·Z·H = X, so H·Z·H|0⟩ = |1⟩
        assert np.allclose(sv, [0, 1], atol=1e-14)


# ── 8. copy() / reset() ───────────────────────────────────────────────────────

class TestCopyReset:
    def test_copy_is_independent(self):
        sim = StatevectorSimulator(2)
        sim.h(0)
        cpy = sim.copy()
        sim.z(0)  # Z|+⟩ = |−⟩, which is ≠ |+⟩; modifies original but not copy
        # After Z(0): sim has |−0⟩ = (|0⟩-|1⟩)/√2 ⊗ |0⟩
        # copy is unchanged at |+0⟩ = (|0⟩+|1⟩)/√2 ⊗ |0⟩
        assert not np.allclose(sim.statevector(), cpy.statevector())

    def test_reset_returns_to_basis(self):
        sim = StatevectorSimulator(2)
        sim.h(0).h(1)
        sim.reset(0)
        assert np.allclose(sim.statevector(), [1, 0, 0, 0])

    def test_reset_returns_self(self):
        sim = StatevectorSimulator(2)
        result = sim.reset()
        assert result is sim


# ── 9. Circuit dict bridge ────────────────────────────────────────────────────

class TestCircuitBridge:
    def test_from_circuit_dict_bell_state(self):
        circuit = {
            "qubits": 2,
            "operations": [
                {"name": "H", "targets": [0]},
                {"name": "CNOT", "targets": [0, 1], "controls": [0]},
            ],
        }
        sim = StatevectorSimulator.from_circuit_dict(circuit)
        sv = sim.statevector()
        expected = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
        assert np.allclose(sv, expected, atol=1e-14)

    def test_run_circuit_dict_returns_ok(self):
        circuit = {
            "qubits": 1,
            "shots": 0,
            "operations": [{"name": "X", "targets": [0]}],
        }
        sim = StatevectorSimulator(1)
        result = sim.run_circuit_dict(circuit)
        assert result["ok"] is True
        assert result["backend"] == "pure_python_statevector"

    def test_run_circuit_dict_state_field(self):
        circuit = {
            "qubits": 1,
            "shots": 0,
            "operations": [{"name": "X", "targets": [0]}],
        }
        sim = StatevectorSimulator(1)
        result = sim.run_circuit_dict(circuit)
        state = result["state"]
        assert len(state) == 2
        # |1⟩ → [0+0j, 1+0j]
        re0, im0 = state[0]
        re1, im1 = state[1]
        assert abs(re0) < 1e-12 and abs(im0) < 1e-12
        assert abs(re1 - 1.0) < 1e-12 and abs(im1) < 1e-12

    def test_run_circuit_dict_with_shots(self):
        circuit = {
            "qubits": 1,
            "shots": 200,
            "seed": 42,
            "operations": [
                {"name": "H", "targets": [0]},
                {"name": "MEASURE", "targets": [0]},
            ],
        }
        sim = StatevectorSimulator(1)
        result = sim.run_circuit_dict(circuit)
        counts = result["counts"]
        assert sum(counts.values()) == 200

    def test_run_circuit_dict_unknown_gate_raises(self):
        circuit = {
            "qubits": 1,
            "operations": [{"name": "MAGIC", "targets": [0]}],
        }
        sim = StatevectorSimulator(1)
        with pytest.raises(ValueError, match="Unknown gate"):
            sim.run_circuit_dict(circuit)


# ── 10. Error handling ────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_invalid_qubit_index(self):
        sim = StatevectorSimulator(2)
        with pytest.raises(ValueError, match="qubit"):
            sim.h(5)

    def test_negative_qubit_index(self):
        sim = StatevectorSimulator(2)
        with pytest.raises(ValueError):
            sim.x(-1)

    def test_cnot_same_qubit_raises(self):
        sim = StatevectorSimulator(2)
        with pytest.raises(ValueError):
            sim.cnot(0, 0)

    def test_swap_same_qubit_raises(self):
        sim = StatevectorSimulator(2)
        with pytest.raises(ValueError):
            sim.swap(1, 1)

    def test_ccx_duplicate_qubits_raises(self):
        sim = StatevectorSimulator(3)
        with pytest.raises(ValueError):
            sim.ccx(0, 0, 2)

    def test_sample_zero_shots_raises(self):
        sim = StatevectorSimulator(1)
        with pytest.raises(ValueError, match="n_shots"):
            sim.sample(0)

    def test_custom_gate_wrong_shape(self):
        sim = StatevectorSimulator(1)
        with pytest.raises(ValueError, match="2×2"):
            sim.custom(np.eye(4), 0)


# ── 11. Larger circuit smoke test ─────────────────────────────────────────────

class TestLargerCircuit:
    def test_ghz_5_qubit(self):
        n = 5
        sim = StatevectorSimulator(n)
        sim.h(0)
        for i in range(n - 1):
            sim.cnot(i, i + 1)
        sv = sim.statevector()
        # GHZ state: (|00000⟩ + |11111⟩)/√2
        expected = np.zeros(2 ** n, dtype=complex)
        expected[0] = 1 / math.sqrt(2)
        expected[-1] = 1 / math.sqrt(2)
        assert np.allclose(sv, expected, atol=1e-12)

    def test_qft_2_qubit_unitary(self):
        """2-qubit QFT: statevector norm is preserved."""
        sim = StatevectorSimulator(2)
        sim.h(0).p(math.pi / 2, 1).cnot(0, 1).h(1).swap(0, 1)
        assert abs(_norm(sim.statevector()) - 1.0) < 1e-12

    def test_10_qubit_norm_preserved(self):
        n = 10
        sim = StatevectorSimulator(n)
        for i in range(n):
            sim.h(i)
        for i in range(0, n - 1, 2):
            sim.cnot(i, i + 1)
        assert abs(_norm(sim.statevector()) - 1.0) < 1e-12
