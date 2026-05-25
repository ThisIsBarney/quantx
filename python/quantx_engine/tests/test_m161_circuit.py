"""Tests for M161: QuantumCircuit fluent builder API.

Covers: construction, all gates, Bell/GHZ state, run(), counts,
statevector, QASM3 round-trip, ASCII draw, Symbol parameters,
CircuitResult fields, error handling, copy, __len__.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pytest

from quantx_engine import QuantumCircuit, CircuitResult
from quantx_engine.circuit import Symbol, _GateOp, _fmt_angle, _parse_angle


# ── Helpers ────────────────────────────────────────────────────────────────────

_SQRT2I = 1.0 / math.sqrt(2)


def _bell_sv() -> np.ndarray:
    """Expected Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
    sv = np.zeros(4, dtype=complex)
    sv[0] = _SQRT2I
    sv[3] = _SQRT2I
    return sv


# ── Construction & basic properties ───────────────────────────────────────────


def test_construction_default():
    qc = QuantumCircuit(2)
    assert qc.n_qubits == 2
    assert len(qc) == 0
    assert qc.operations == []


def test_construction_with_name():
    qc = QuantumCircuit(3, name="my_circuit")
    assert qc.name == "my_circuit"
    assert qc.n_qubits == 3


def test_construction_single_qubit():
    qc = QuantumCircuit(1)
    assert qc.n_qubits == 1


def test_construction_max_qubits():
    qc = QuantumCircuit(25)
    assert qc.n_qubits == 25


def test_construction_invalid_zero():
    with pytest.raises(ValueError, match="n_qubits"):
        QuantumCircuit(0)


def test_construction_invalid_negative():
    with pytest.raises(ValueError, match="n_qubits"):
        QuantumCircuit(-1)


def test_construction_invalid_too_large():
    with pytest.raises(ValueError, match="n_qubits"):
        QuantumCircuit(26)


def test_repr_and_str():
    qc = QuantumCircuit(2)
    qc.h(0)
    r = repr(qc)
    assert "n_qubits=2" in r
    assert "gates=1" in r


# ── Single-qubit gate fluent chaining ─────────────────────────────────────────


def test_fluent_chain_returns_self():
    qc = QuantumCircuit(1)
    result = qc.h(0)
    assert result is qc


def test_gate_count():
    qc = QuantumCircuit(2)
    qc.h(0).x(1).y(0).z(1).s(0).t(1).sdg(0).tdg(1)
    assert len(qc) == 8


def test_rx_ry_rz_p_gates():
    qc = QuantumCircuit(1)
    qc.rx(math.pi, 0).ry(math.pi / 2, 0).rz(math.pi / 4, 0).p(math.pi / 3, 0)
    assert len(qc) == 4


def test_u3_gate():
    qc = QuantumCircuit(1)
    qc.u3(math.pi / 2, 0.0, math.pi, 0)
    assert len(qc) == 1


def test_identity_gate():
    qc = QuantumCircuit(1)
    qc.i(0)
    assert len(qc) == 1


def test_invalid_qubit_raises():
    qc = QuantumCircuit(2)
    with pytest.raises(ValueError, match="Qubit index"):
        qc.h(2)
    with pytest.raises(ValueError, match="Qubit index"):
        qc.h(-1)


# ── Two-qubit gates ────────────────────────────────────────────────────────────


def test_cx_cnot_alias():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    assert len(qc) == 1
    qc2 = QuantumCircuit(2)
    qc2.cnot(0, 1)
    assert qc2.operations[0].name == "CX"


def test_cy_cz_ch():
    qc = QuantumCircuit(2)
    qc.cy(0, 1).cz(0, 1).ch(0, 1)
    assert len(qc) == 3


def test_swap():
    qc = QuantumCircuit(2)
    qc.swap(0, 1)
    assert len(qc) == 1


def test_iswap():
    qc = QuantumCircuit(2)
    qc.iswap(0, 1)
    assert len(qc) == 1


def test_cx_same_qubit_raises():
    qc = QuantumCircuit(2)
    with pytest.raises(ValueError, match="differ"):
        qc.cx(1, 1)


def test_swap_same_qubit_raises():
    qc = QuantumCircuit(2)
    with pytest.raises(ValueError, match="differ"):
        qc.swap(0, 0)


# ── Three-qubit gates ──────────────────────────────────────────────────────────


def test_ccx_toffoli_alias():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    assert len(qc) == 1
    qc2 = QuantumCircuit(3)
    qc2.toffoli(0, 1, 2)
    assert qc2.operations[0].name == "CCX"


def test_cswap_fredkin_alias():
    qc = QuantumCircuit(3)
    qc.cswap(0, 1, 2)
    assert len(qc) == 1
    qc2 = QuantumCircuit(3)
    qc2.fredkin(0, 1, 2)
    assert qc2.operations[0].name == "CSWAP"


def test_ccx_duplicate_qubit_raises():
    qc = QuantumCircuit(3)
    with pytest.raises(ValueError, match="distinct"):
        qc.ccx(0, 0, 2)


# ── Measurement ────────────────────────────────────────────────────────────────


def test_measure_single():
    qc = QuantumCircuit(2)
    qc.measure(0)
    # measure ops are not counted in len()
    assert len(qc) == 0
    measure_ops = [op for op in qc.operations if op.is_measure]
    assert len(measure_ops) == 1


def test_measure_all():
    qc = QuantumCircuit(3)
    qc.h(0).measure_all()
    assert len(qc) == 1  # only H counted
    measure_ops = [op for op in qc.operations if op.is_measure]
    assert len(measure_ops) == 1
    assert measure_ops[0].name == "MEASURE_ALL"


# ── Bell state (run + statevector) ─────────────────────────────────────────────


def test_bell_state_statevector():
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1)
    result = qc.run()
    assert isinstance(result, CircuitResult)
    assert result.n_qubits == 2
    assert result.statevector.shape == (4,)
    np.testing.assert_allclose(np.abs(result.statevector), np.abs(_bell_sv()), atol=1e-10)


def test_bell_state_counts_distribution():
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1).measure_all()
    result = qc.run(n_shots=4096, seed=42)
    assert set(result.counts.keys()) <= {"00", "11"}
    total = sum(result.counts.values())
    assert total == 4096
    # Both outcomes should appear
    assert "00" in result.counts and "11" in result.counts


def test_zero_shots_gives_empty_counts():
    qc = QuantumCircuit(1)
    qc.h(0)
    result = qc.run(n_shots=0)
    assert result.counts == {}


# ── GHZ state ─────────────────────────────────────────────────────────────────


def test_ghz_3qubit_statevector():
    qc = QuantumCircuit(3)
    qc.h(0).cx(0, 1).cx(0, 2)
    result = qc.run()
    sv = result.statevector
    assert sv.shape == (8,)
    expected = np.zeros(8, dtype=complex)
    expected[0] = _SQRT2I
    expected[7] = _SQRT2I
    np.testing.assert_allclose(np.abs(sv), np.abs(expected), atol=1e-10)


# ── X, H correctness ──────────────────────────────────────────────────────────


def test_x_gate_flips_state():
    qc = QuantumCircuit(1)
    qc.x(0)
    sv = qc.run().statevector
    assert abs(sv[1]) == pytest.approx(1.0, abs=1e-10)


def test_h_gate_superposition():
    qc = QuantumCircuit(1)
    qc.h(0)
    sv = qc.run().statevector
    np.testing.assert_allclose(sv, [_SQRT2I, _SQRT2I], atol=1e-10)


def test_h_twice_returns_to_zero():
    qc = QuantumCircuit(1)
    qc.h(0).h(0)
    sv = qc.run().statevector
    np.testing.assert_allclose(np.abs(sv), [1.0, 0.0], atol=1e-10)


# ── Rotation gates ─────────────────────────────────────────────────────────────


def test_rx_pi_equals_x():
    qc_rx = QuantumCircuit(1)
    qc_rx.rx(math.pi, 0)
    qc_x = QuantumCircuit(1)
    qc_x.x(0)
    sv_rx = qc_rx.run().statevector
    sv_x = qc_x.run().statevector
    np.testing.assert_allclose(np.abs(sv_rx), np.abs(sv_x), atol=1e-10)


def test_rz_pi_equals_z_up_to_global_phase():
    qc = QuantumCircuit(1)
    qc.h(0).rz(math.pi, 0).h(0)
    sv = qc.run().statevector
    # h rz(pi) h = X up to global phase
    np.testing.assert_allclose(np.abs(sv), [0.0, 1.0], atol=1e-10)


# ── S/T gates ─────────────────────────────────────────────────────────────────


def test_s_and_sdg_cancel():
    qc = QuantumCircuit(1)
    qc.h(0).s(0).sdg(0).h(0)
    sv = qc.run().statevector
    np.testing.assert_allclose(np.abs(sv), [1.0, 0.0], atol=1e-10)


def test_t_and_tdg_cancel():
    qc = QuantumCircuit(1)
    qc.h(0).t(0).tdg(0).h(0)
    sv = qc.run().statevector
    np.testing.assert_allclose(np.abs(sv), [1.0, 0.0], atol=1e-10)


# ── CircuitResult fields ───────────────────────────────────────────────────────


def test_circuit_result_has_statevector_field():
    result = QuantumCircuit(1).h(0).run()
    assert hasattr(result, "statevector")
    assert isinstance(result.statevector, np.ndarray)


def test_circuit_result_has_counts_field():
    result = QuantumCircuit(1).h(0).run(n_shots=10, seed=0)
    assert hasattr(result, "counts")
    assert isinstance(result.counts, dict)


def test_circuit_result_has_n_qubits_field():
    result = QuantumCircuit(3).run()
    assert result.n_qubits == 3


def test_circuit_result_backend_field():
    result = QuantumCircuit(1).run()
    assert isinstance(result.backend, str)
    assert len(result.backend) > 0


# ── Symbolic parameters ────────────────────────────────────────────────────────


def test_symbol_repr():
    s = Symbol("theta")
    assert "theta" in repr(s)
    assert str(s) == "theta"


def test_symbol_float_raises():
    s = Symbol("theta")
    with pytest.raises(TypeError, match="not been bound"):
        float(s)


def test_parametric_rx_via_symbol():
    theta = Symbol("theta")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    result = qc.run(params={"theta": math.pi})
    # Rx(pi)|0⟩ = -i|1⟩  → amplitude 1 at index 1
    np.testing.assert_allclose(abs(result.statevector[1]), 1.0, atol=1e-10)


def test_unbound_symbol_raises_on_run():
    theta = Symbol("theta")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    with pytest.raises(ValueError, match="not bound"):
        qc.run()


def test_multiple_symbols_bound():
    theta = Symbol("t")
    phi = Symbol("p")
    qc = QuantumCircuit(1)
    qc.u3(theta, phi, 0.0, 0)
    result = qc.run(params={"t": math.pi, "p": 0.0})
    # u3(pi, 0, 0)|0⟩ ≈ X|0⟩ = |1⟩
    np.testing.assert_allclose(abs(result.statevector[1]), 1.0, atol=1e-10)


# ── QASM3 export ───────────────────────────────────────────────────────────────


def test_to_qasm3_header():
    qc = QuantumCircuit(2)
    qc.h(0)
    qasm = qc.to_qasm3()
    assert "OPENQASM 3.0" in qasm
    assert "stdgates.inc" in qasm
    assert "qubit[2] q" in qasm


def test_to_qasm3_contains_gate():
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1)
    qasm = qc.to_qasm3()
    assert "h q[0]" in qasm
    assert "cx q[0], q[1]" in qasm


def test_to_qasm3_rotation_gate():
    qc = QuantumCircuit(1)
    qc.rx(math.pi / 2, 0)
    qasm = qc.to_qasm3()
    assert "rx(" in qasm
    assert "q[0]" in qasm


def test_to_qasm3_measure_all():
    qc = QuantumCircuit(2)
    qc.h(0).measure_all()
    qasm = qc.to_qasm3()
    assert "bit[2] c" in qasm
    assert "measure" in qasm


def test_to_qasm3_no_measure_no_bit_register():
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1)
    qasm = qc.to_qasm3()
    # "qubit[" legitimately contains "bit[" as substring, so check for the
    # classical register declaration specifically (starts at line boundary).
    assert "\nbit[" not in qasm


# ── QASM3 import ───────────────────────────────────────────────────────────────


def test_from_qasm3_basic():
    qasm = """\
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;

h q[0];
cx q[0], q[1];
"""
    qc = QuantumCircuit.from_qasm3(qasm)
    assert qc.n_qubits == 2
    assert len(qc) == 2


def test_from_qasm3_gate_names():
    qasm = """\
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;

h q[0];
cx q[0], q[1];
"""
    qc = QuantumCircuit.from_qasm3(qasm)
    names = [op.name for op in qc.operations]
    assert names == ["H", "CX"]


def test_from_qasm3_no_qubit_declaration_raises():
    with pytest.raises(ValueError, match="qubit declaration"):
        QuantumCircuit.from_qasm3("OPENQASM 3.0;\nh q[0];\n")


def test_from_qasm3_with_angles():
    qasm = """\
OPENQASM 3.0;
include "stdgates.inc";

qubit[1] q;

rx(pi/2) q[0];
"""
    qc = QuantumCircuit.from_qasm3(qasm)
    assert len(qc) == 1
    op = qc.operations[0]
    assert op.name == "RX"
    assert abs(op.params[0] - math.pi / 2) < 1e-10


def test_from_qasm3_with_measurement():
    qasm = """\
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

h q[0];
cx q[0], q[1];
c = measure q;
"""
    qc = QuantumCircuit.from_qasm3(qasm)
    assert len(qc) == 2
    measure_ops = [op for op in qc.operations if op.is_measure]
    assert len(measure_ops) == 1


# ── QASM3 round-trip ──────────────────────────────────────────────────────────


def test_qasm3_roundtrip_bell():
    """to_qasm3 → from_qasm3 → run should give the same statevector."""
    qc_orig = QuantumCircuit(2)
    qc_orig.h(0).cx(0, 1)
    qasm = qc_orig.to_qasm3()

    qc_parsed = QuantumCircuit.from_qasm3(qasm)
    sv_orig = qc_orig.run().statevector
    sv_parsed = qc_parsed.run().statevector
    np.testing.assert_allclose(sv_orig, sv_parsed, atol=1e-10)


def test_qasm3_roundtrip_rotations():
    qc_orig = QuantumCircuit(1)
    qc_orig.rx(math.pi / 3, 0).ry(math.pi / 5, 0).rz(math.pi / 7, 0)
    qasm = qc_orig.to_qasm3()
    qc_parsed = QuantumCircuit.from_qasm3(qasm)
    sv_orig = qc_orig.run().statevector
    sv_parsed = qc_parsed.run().statevector
    np.testing.assert_allclose(sv_orig, sv_parsed, atol=1e-9)


def test_qasm3_roundtrip_toffoli():
    qc_orig = QuantumCircuit(3)
    qc_orig.x(0).x(1).ccx(0, 1, 2)
    qasm = qc_orig.to_qasm3()
    qc_parsed = QuantumCircuit.from_qasm3(qasm)
    sv_orig = qc_orig.run().statevector
    sv_parsed = qc_parsed.run().statevector
    np.testing.assert_allclose(np.abs(sv_orig), np.abs(sv_parsed), atol=1e-10)


# ── ASCII draw ─────────────────────────────────────────────────────────────────


def test_draw_returns_string():
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1)
    diagram = qc.draw()
    assert isinstance(diagram, str)
    assert len(diagram) > 0


def test_draw_has_correct_qubit_labels():
    qc = QuantumCircuit(3)
    qc.h(0)
    diagram = qc.draw()
    assert "q[0]" in diagram
    assert "q[1]" in diagram
    assert "q[2]" in diagram


def test_draw_contains_gate_label():
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1)
    diagram = qc.draw()
    assert "H" in diagram
    assert "CX" in diagram


def test_draw_n_rows_equals_n_qubits():
    qc = QuantumCircuit(4)
    qc.h(0).cx(0, 1).cx(1, 2).cx(2, 3)
    diagram = qc.draw()
    rows = diagram.splitlines()
    assert len(rows) == 4


def test_str_equals_draw():
    qc = QuantumCircuit(2)
    qc.h(0)
    assert str(qc) == qc.draw()


def test_draw_with_measurement():
    qc = QuantumCircuit(2)
    qc.h(0).measure_all()
    diagram = qc.draw()
    assert "M" in diagram


# ── Copy ──────────────────────────────────────────────────────────────────────


def test_copy_is_independent():
    qc = QuantumCircuit(2, name="original")
    qc.h(0).cx(0, 1)
    copy = qc.copy()
    assert copy.n_qubits == qc.n_qubits
    assert len(copy) == len(qc)
    # Mutate the copy — original must be unaffected
    copy.x(0)
    assert len(qc) == 2
    assert len(copy) == 3


def test_copy_preserves_name():
    qc = QuantumCircuit(1, name="bell")
    copy = qc.copy()
    assert copy.name == "bell"


def test_copy_ops_list_is_independent():
    qc = QuantumCircuit(1)
    qc.h(0)
    copy = qc.copy()
    copy._ops.clear()
    # Original must still have its op
    assert len(qc) == 1


# ── _fmt_angle / _parse_angle helpers ─────────────────────────────────────────


def test_fmt_angle_zero():
    assert _fmt_angle(0.0) == "0"


def test_fmt_angle_pi():
    assert _fmt_angle(math.pi) == "pi"


def test_fmt_angle_half_pi():
    s = _fmt_angle(math.pi / 2)
    assert "pi" in s


def test_fmt_angle_symbol():
    s = Symbol("alpha")
    assert _fmt_angle(s) == "alpha"


def test_parse_angle_pi_over_2():
    v = _parse_angle("pi/2")
    assert abs(v - math.pi / 2) < 1e-12


def test_parse_angle_negative():
    v = _parse_angle("-pi/4")
    assert abs(v - (-math.pi / 4)) < 1e-12


def test_parse_angle_plain_float():
    v = _parse_angle("1.5707963267948966")
    assert abs(v - math.pi / 2) < 1e-10


# ── QFT (5-qubit) end-to-end ─────────────────────────────────────────────────


def _build_qft(n: int) -> QuantumCircuit:
    """Construct an n-qubit QFT circuit using QuantumCircuit API."""
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            angle = math.pi / (2 ** (k - j))
            qc.p(angle, k)
            qc.cx(j, k)
            qc.p(-angle, k)
            qc.cx(j, k)
    return qc


def test_qft_5qubit_state_is_normalised():
    qc = _build_qft(5)
    sv = qc.run().statevector
    assert abs(np.linalg.norm(sv) - 1.0) < 1e-9


def test_qft_1qubit_equals_hadamard():
    qc = QuantumCircuit(1)
    qc.h(0)
    sv_h = qc.run().statevector

    qft = _build_qft(1)
    sv_qft = qft.run().statevector
    np.testing.assert_allclose(np.abs(sv_qft), np.abs(sv_h), atol=1e-10)


# ── Public API exports ────────────────────────────────────────────────────────


def test_quantumcircuit_exported_from_package():
    from quantx_engine import QuantumCircuit as QC
    assert QC is QuantumCircuit


def test_circuitresult_exported_from_package():
    from quantx_engine import CircuitResult as CR
    assert CR is CircuitResult


def test_symbol_exported_from_package():
    from quantx_engine import Symbol as S
    assert S is Symbol


def test_in_dunder_all():
    import quantx_engine
    assert "QuantumCircuit" in quantx_engine.__all__
    assert "CircuitResult" in quantx_engine.__all__
    assert "Symbol" in quantx_engine.__all__
