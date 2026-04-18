"""Tests for the Cirq and Braket circuit adapters (M77).

All tests use mock objects so the test suite runs even when cirq and
amazon-braket-sdk are not installed.
"""
from __future__ import annotations

import math
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from ..transpiler import GateOp


# ---------------------------------------------------------------------------
# Helper: build a minimal mock cirq module
# ---------------------------------------------------------------------------

def _make_mock_cirq() -> ModuleType:
    """Return a mock that covers the cirq API used by the adapters."""
    cirq = MagicMock(name="cirq")

    # LineQubit
    class FakeLineQubit:
        def __init__(self, x: int):
            self.x = x
        def __repr__(self) -> str:
            return f"q({self.x})"

    cirq.LineQubit = MagicMock()
    cirq.LineQubit.range = lambda n: [FakeLineQubit(i) for i in range(n)]

    # Simple gates (callable to create operations)
    class FakeGate:
        def __init__(self, name: str, exp: float = 1.0, rads: float | None = None):
            self.__class__ = type(name, (FakeGate,), {})  # dynamic class name
            self.__class__.__name__ = name
            self.exponent = exp
            if rads is not None:
                self.rads = rads

        def __call__(self, *qubits):
            op = MagicMock()
            op.gate = self
            op.qubits = list(qubits)
            return op

        def __pow__(self, exp):
            return FakeGate(self.__class__.__name__, exp)

    def _gate(name, **kw):
        g = FakeGate(name, **kw)
        g.__class__.__name__ = name
        return g

    cirq.H      = _gate("HPowGate")
    cirq.X      = _gate("XPowGate")
    cirq.Y      = _gate("YPowGate")
    cirq.Z      = _gate("ZPowGate")
    cirq.S      = _gate("ZPowGate", exp=0.5)
    cirq.T      = _gate("ZPowGate", exp=0.25)
    cirq.I      = _gate("IdentityGate")
    cirq.CNOT   = _gate("CNotPowGate")
    cirq.CZ     = _gate("CZPowGate")
    cirq.SWAP   = _gate("SwapPowGate")
    cirq.ISWAP  = _gate("ISwapPowGate")
    cirq.CCX    = _gate("CCXPowGate")
    cirq.CSWAP  = _gate("CSwapGate")

    def _rx(rads):
        return _gate("Rx", rads=rads)
    def _ry(rads):
        return _gate("Ry", rads=rads)
    def _rz(rads):
        return _gate("Rz", rads=rads)

    cirq.rx = _rx
    cirq.ry = _ry
    cirq.rz = _rz

    # Circuit class
    class FakeCircuit:
        def __init__(self, ops=()):
            self._ops = list(ops)
        def all_operations(self):
            return iter(self._ops)

    cirq.Circuit = FakeCircuit
    return cirq


# ---------------------------------------------------------------------------
# Helper: build a minimal mock braket module
# ---------------------------------------------------------------------------

def _make_mock_braket() -> ModuleType:
    """Return a minimal braket mock covering the adapters API."""
    braket_pkg = MagicMock(name="braket")

    class FakeGate:
        def __init__(self, class_name: str, angle: float | None = None):
            self.__class__ = type(class_name, (FakeGate,), {})
            self.__class__.__name__ = class_name
            if angle is not None:
                self.angle = angle

    class FakeInstruction:
        def __init__(self, gate_cls: str, qubits: list[int], angle: float | None = None):
            self.operator = FakeGate(gate_cls, angle)
            self.target = qubits

    class FakeCircuit:
        def __init__(self):
            self.instructions: list[FakeInstruction] = []
            # Install gate methods dynamically
            self._define_gates()

        def _define_gates(self):
            one_q_no_param = {
                "i": "I", "h": "H", "x": "X", "y": "Y", "z": "Z",
                "s": "S", "si": "Si", "t": "T", "ti": "Ti",
                "v": "V", "vi": "Vi",
            }
            one_q_param = {
                "rx": "Rx", "ry": "Ry", "rz": "Rz", "phaseshift": "PhaseShift",
            }
            two_q_no_param = {
                "cnot": "CNot", "cz": "CZ", "cy": "CY", "swap": "Swap",
                "iswap": "ISwap",
            }
            three_q = {"ccnot": "CCNot", "cswap": "CSwap"}

            def _make_1q(cls_name):
                def method(q):
                    self.instructions.append(FakeInstruction(cls_name, [q]))
                    return self
                return method

            def _make_1q_param(cls_name):
                def method(q, angle):
                    self.instructions.append(FakeInstruction(cls_name, [q], angle))
                    return self
                return method

            def _make_2q(cls_name):
                def method(q0, q1):
                    self.instructions.append(FakeInstruction(cls_name, [q0, q1]))
                    return self
                return method

            def _make_3q(cls_name):
                def method(q0, q1, q2):
                    self.instructions.append(FakeInstruction(cls_name, [q0, q1, q2]))
                    return self
                return method

            for method_name, cls_name in one_q_no_param.items():
                setattr(self, method_name, _make_1q(cls_name))
            for method_name, cls_name in one_q_param.items():
                setattr(self, method_name, _make_1q_param(cls_name))
            for method_name, cls_name in two_q_no_param.items():
                setattr(self, method_name, _make_2q(cls_name))
            for method_name, cls_name in three_q.items():
                setattr(self, method_name, _make_3q(cls_name))

    circuits_mod = MagicMock(name="braket.circuits")
    circuits_mod.Circuit = FakeCircuit

    braket_pkg.circuits = circuits_mod
    return braket_pkg, circuits_mod


# ---------------------------------------------------------------------------
# Fixtures that patch sys.modules
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_cirq():
    cirq_mod = _make_mock_cirq()
    with patch.dict(sys.modules, {"cirq": cirq_mod}):
        yield cirq_mod


@pytest.fixture()
def mock_braket():
    braket_mod, circuits_mod = _make_mock_braket()
    with patch.dict(sys.modules, {
        "braket": braket_mod,
        "braket.circuits": circuits_mod,
    }):
        yield braket_mod


# ---------------------------------------------------------------------------
# AdapterError import test (no mocks needed)
# ---------------------------------------------------------------------------

def test_adapter_error_is_value_error():
    from ..adapters import AdapterError
    assert issubclass(AdapterError, ValueError)


def test_adapter_error_when_cirq_missing():
    """AdapterError raised if cirq is not installed."""
    from ..adapters import to_cirq
    with patch.dict(sys.modules, {"cirq": None}):  # None = import fails
        with pytest.raises(Exception):  # AdapterError or ImportError
            to_cirq([GateOp("h", [0])])


def test_adapter_error_when_braket_missing():
    """AdapterError raised if braket is not installed."""
    from ..adapters import to_braket
    with patch.dict(sys.modules, {"braket": None, "braket.circuits": None}):
        with pytest.raises(Exception):
            to_braket([GateOp("h", [0])])


# ===========================================================================
# Cirq adapter tests
# ===========================================================================

class TestToCirq:
    def test_empty_circuit(self, mock_cirq):
        from ..adapters import to_cirq
        result = to_cirq([])
        # Should return a FakeCircuit with no ops
        ops = list(result.all_operations())
        assert ops == []

    def test_single_h_gate(self, mock_cirq):
        from ..adapters import to_cirq
        circ = to_cirq([GateOp("h", [0])], n_qubits=1)
        ops = list(circ.all_operations())
        assert len(ops) == 1
        assert type(ops[0].gate).__name__ == "HPowGate"
        assert ops[0].qubits[0].x == 0

    def test_cx_gate(self, mock_cirq):
        from ..adapters import to_cirq
        circ = to_cirq([GateOp("cx", [0, 1])], n_qubits=2)
        ops = list(circ.all_operations())
        assert type(ops[0].gate).__name__ == "CNotPowGate"
        assert [q.x for q in ops[0].qubits] == [0, 1]

    def test_cz_gate(self, mock_cirq):
        from ..adapters import to_cirq
        circ = to_cirq([GateOp("cz", [0, 1])], n_qubits=2)
        ops = list(circ.all_operations())
        assert type(ops[0].gate).__name__ == "CZPowGate"

    def test_swap_gate(self, mock_cirq):
        from ..adapters import to_cirq
        circ = to_cirq([GateOp("swap", [0, 1])], n_qubits=2)
        ops = list(circ.all_operations())
        assert type(ops[0].gate).__name__ == "SwapPowGate"

    def test_rx_gate_with_param(self, mock_cirq):
        from ..adapters import to_cirq
        circ = to_cirq([GateOp("rx", [0], [1.5707963])], n_qubits=1)
        ops = list(circ.all_operations())
        assert type(ops[0].gate).__name__ == "Rx"
        assert math.isclose(ops[0].gate.rads, 1.5707963)

    def test_ry_gate_with_param(self, mock_cirq):
        from ..adapters import to_cirq
        circ = to_cirq([GateOp("ry", [0], [math.pi / 4])], n_qubits=1)
        ops = list(circ.all_operations())
        assert type(ops[0].gate).__name__ == "Ry"

    def test_rz_gate_with_param(self, mock_cirq):
        from ..adapters import to_cirq
        circ = to_cirq([GateOp("rz", [0], [math.pi])], n_qubits=1)
        ops = list(circ.all_operations())
        assert type(ops[0].gate).__name__ == "Rz"

    def test_ccx_gate(self, mock_cirq):
        from ..adapters import to_cirq
        circ = to_cirq([GateOp("ccx", [0, 1, 2])], n_qubits=3)
        ops = list(circ.all_operations())
        assert type(ops[0].gate).__name__ == "CCXPowGate"

    def test_toffoli_alias(self, mock_cirq):
        from ..adapters import to_cirq
        circ = to_cirq([GateOp("toffoli", [0, 1, 2])], n_qubits=3)
        ops = list(circ.all_operations())
        assert type(ops[0].gate).__name__ == "CCXPowGate"

    def test_cnot_alias(self, mock_cirq):
        from ..adapters import to_cirq
        circ = to_cirq([GateOp("cnot", [0, 1])], n_qubits=2)
        ops = list(circ.all_operations())
        assert type(ops[0].gate).__name__ == "CNotPowGate"

    def test_n_qubits_inferred(self, mock_cirq):
        from ..adapters import to_cirq
        # Should not raise; n_qubits inferred = 3
        circ = to_cirq([GateOp("h", [2])])
        ops = list(circ.all_operations())
        assert ops[0].qubits[0].x == 2

    def test_unsupported_gate_raises(self, mock_cirq):
        from ..adapters import to_cirq, AdapterError
        with pytest.raises(AdapterError, match="not supported"):
            to_cirq([GateOp("my_gate", [0])])

    def test_multi_gate_circuit(self, mock_cirq):
        from ..adapters import to_cirq
        circ = to_cirq([
            GateOp("h", [0]),
            GateOp("cx", [0, 1]),
            GateOp("rz", [1], [math.pi / 2]),
        ], n_qubits=2)
        ops = list(circ.all_operations())
        assert len(ops) == 3

    def test_x_y_z_gates(self, mock_cirq):
        from ..adapters import to_cirq
        for name, expected in [("x", "XPowGate"), ("y", "YPowGate"), ("z", "ZPowGate")]:
            circ = to_cirq([GateOp(name, [0])], n_qubits=1)
            ops = list(circ.all_operations())
            assert type(ops[0].gate).__name__ == expected


class TestFromCirq:
    def _make_op(self, gate_cls_name: str, qubit_xs: list[int],
                 exponent: float = 1.0, rads: float | None = None):
        """Build a mock Cirq operation with the correct gate class name."""
        # Create a real class so type(gate).__name__ == gate_cls_name
        attrs: dict = {"exponent": exponent}
        if rads is not None:
            attrs["rads"] = rads
        GateClass = type(gate_cls_name, (), attrs)
        gate = GateClass()

        qubit_mock = [MagicMock(x=x) for x in qubit_xs]
        op = MagicMock()
        op.gate = gate
        op.qubits = qubit_mock
        return op

    def _make_circuit(self, ops):
        circ = MagicMock()
        circ.all_operations = lambda: iter(ops)
        return circ

    def test_h_gate(self, mock_cirq):
        from ..adapters import from_cirq
        op = self._make_op("HPowGate", [0], exponent=1.0)
        circ = self._make_circuit([op])
        result = from_cirq(circ)
        assert len(result) == 1
        assert result[0].name == "h"
        assert result[0].qubits == [0]

    def test_cx_gate(self, mock_cirq):
        from ..adapters import from_cirq
        op = self._make_op("CNotPowGate", [0, 1], exponent=1.0)
        circ = self._make_circuit([op])
        result = from_cirq(circ)
        assert result[0].name == "cx"
        assert result[0].qubits == [0, 1]

    def test_cz_gate(self, mock_cirq):
        from ..adapters import from_cirq
        op = self._make_op("CZPowGate", [1, 2], exponent=1.0)
        circ = self._make_circuit([op])
        result = from_cirq(circ)
        assert result[0].name == "cz"

    def test_swap_gate(self, mock_cirq):
        from ..adapters import from_cirq
        op = self._make_op("SwapPowGate", [0, 2], exponent=1.0)
        circ = self._make_circuit([op])
        result = from_cirq(circ)
        assert result[0].name == "swap"

    def test_identity_gate(self, mock_cirq):
        from ..adapters import from_cirq
        op = self._make_op("IdentityGate", [0])
        circ = self._make_circuit([op])
        result = from_cirq(circ)
        assert result[0].name == "id"

    def test_ccx_gate(self, mock_cirq):
        from ..adapters import from_cirq
        op = self._make_op("CCXPowGate", [0, 1, 2])
        circ = self._make_circuit([op])
        result = from_cirq(circ)
        assert result[0].name == "ccx"

    def test_rx_parametric(self, mock_cirq):
        from ..adapters import from_cirq
        op = self._make_op("Rx", [0], rads=1.5707963)
        circ = self._make_circuit([op])
        result = from_cirq(circ)
        assert result[0].name == "rx"
        assert math.isclose(result[0].params[0], 1.5707963)

    def test_ry_parametric(self, mock_cirq):
        from ..adapters import from_cirq
        op = self._make_op("Ry", [0], rads=math.pi / 4)
        circ = self._make_circuit([op])
        result = from_cirq(circ)
        assert result[0].name == "ry"

    def test_rz_parametric(self, mock_cirq):
        from ..adapters import from_cirq
        op = self._make_op("Rz", [0], rads=math.pi)
        circ = self._make_circuit([op])
        result = from_cirq(circ)
        assert result[0].name == "rz"

    def test_zpowgate_exp1_is_z(self, mock_cirq):
        from ..adapters import from_cirq
        op = self._make_op("ZPowGate", [0], exponent=1.0)
        circ = self._make_circuit([op])
        result = from_cirq(circ)
        assert result[0].name == "z"

    def test_zpowgate_exp_half_is_s(self, mock_cirq):
        from ..adapters import from_cirq
        op = self._make_op("ZPowGate", [0], exponent=0.5)
        circ = self._make_circuit([op])
        result = from_cirq(circ)
        assert result[0].name == "s"

    def test_zpowgate_exp_quarter_is_t(self, mock_cirq):
        from ..adapters import from_cirq
        op = self._make_op("ZPowGate", [0], exponent=0.25)
        circ = self._make_circuit([op])
        result = from_cirq(circ)
        assert result[0].name == "t"

    def test_zpowgate_neg_quarter_is_tdg(self, mock_cirq):
        from ..adapters import from_cirq
        op = self._make_op("ZPowGate", [0], exponent=-0.25)
        circ = self._make_circuit([op])
        result = from_cirq(circ)
        assert result[0].name == "tdg"

    def test_unknown_op_raises(self, mock_cirq):
        from ..adapters import from_cirq, AdapterError
        op = MagicMock()
        op.gate = MagicMock()
        op.gate.__class__.__name__ = "SomeExoticGate"
        op.qubits = [MagicMock(x=0)]
        circ = self._make_circuit([op])
        with pytest.raises(AdapterError):
            from_cirq(circ)

    def test_none_gate_raises(self, mock_cirq):
        from ..adapters import from_cirq, AdapterError
        op = MagicMock()
        op.gate = None
        op.qubits = []
        circ = self._make_circuit([op])
        with pytest.raises(AdapterError):
            from_cirq(circ)

    def test_empty_circuit(self, mock_cirq):
        from ..adapters import from_cirq
        circ = self._make_circuit([])
        assert from_cirq(circ) == []


# ===========================================================================
# Braket adapter tests
# ===========================================================================

class TestToBraket:
    def test_empty_circuit(self, mock_braket):
        from ..adapters import to_braket
        bk = to_braket([])
        assert bk.instructions == []

    def test_single_h_gate(self, mock_braket):
        from ..adapters import to_braket
        bk = to_braket([GateOp("h", [0])])
        assert len(bk.instructions) == 1
        assert type(bk.instructions[0].operator).__name__ == "H"
        assert bk.instructions[0].target == [0]

    def test_cx_gate(self, mock_braket):
        from ..adapters import to_braket
        bk = to_braket([GateOp("cx", [0, 1])])
        assert type(bk.instructions[0].operator).__name__ == "CNot"

    def test_cnot_alias(self, mock_braket):
        from ..adapters import to_braket
        bk = to_braket([GateOp("cnot", [0, 1])])
        assert type(bk.instructions[0].operator).__name__ == "CNot"

    def test_cz_gate(self, mock_braket):
        from ..adapters import to_braket
        bk = to_braket([GateOp("cz", [0, 1])])
        assert type(bk.instructions[0].operator).__name__ == "CZ"

    def test_swap_gate(self, mock_braket):
        from ..adapters import to_braket
        bk = to_braket([GateOp("swap", [0, 1])])
        assert type(bk.instructions[0].operator).__name__ == "Swap"

    def test_rx_parametric(self, mock_braket):
        from ..adapters import to_braket
        bk = to_braket([GateOp("rx", [0], [1.5707963])])
        inst = bk.instructions[0]
        assert type(inst.operator).__name__ == "Rx"
        assert math.isclose(inst.operator.angle, 1.5707963)

    def test_ry_parametric(self, mock_braket):
        from ..adapters import to_braket
        bk = to_braket([GateOp("ry", [0], [math.pi / 3])])
        inst = bk.instructions[0]
        assert type(inst.operator).__name__ == "Ry"

    def test_rz_parametric(self, mock_braket):
        from ..adapters import to_braket
        bk = to_braket([GateOp("rz", [0], [math.pi])])
        inst = bk.instructions[0]
        assert type(inst.operator).__name__ == "Rz"

    def test_s_sdg_t_tdg(self, mock_braket):
        from ..adapters import to_braket
        for name, cls in [("s", "S"), ("sdg", "Si"), ("t", "T"), ("tdg", "Ti")]:
            bk = to_braket([GateOp(name, [0])])
            assert type(bk.instructions[0].operator).__name__ == cls, name

    def test_sx_sxdg(self, mock_braket):
        from ..adapters import to_braket
        bk = to_braket([GateOp("sx", [0])])
        assert type(bk.instructions[0].operator).__name__ == "V"
        bk2 = to_braket([GateOp("sxdg", [0])])
        assert type(bk2.instructions[0].operator).__name__ == "Vi"

    def test_ccx_toffoli(self, mock_braket):
        from ..adapters import to_braket
        bk = to_braket([GateOp("ccx", [0, 1, 2])])
        assert type(bk.instructions[0].operator).__name__ == "CCNot"
        bk2 = to_braket([GateOp("toffoli", [0, 1, 2])])
        assert type(bk2.instructions[0].operator).__name__ == "CCNot"

    def test_multi_gate_circuit(self, mock_braket):
        from ..adapters import to_braket
        circ = [
            GateOp("h", [0]),
            GateOp("cnot", [0, 1]),
            GateOp("rz", [1], [math.pi / 4]),
        ]
        bk = to_braket(circ)
        assert len(bk.instructions) == 3

    def test_unsupported_gate_raises(self, mock_braket):
        from ..adapters import to_braket, AdapterError
        with pytest.raises(AdapterError, match="not supported"):
            to_braket([GateOp("my_gate", [0])])

    def test_x_y_z_gates(self, mock_braket):
        from ..adapters import to_braket
        for name, cls in [("x", "X"), ("y", "Y"), ("z", "Z")]:
            bk = to_braket([GateOp(name, [0])])
            assert type(bk.instructions[0].operator).__name__ == cls


class TestFromBraket:
    def _make_circ(self, instructions):
        circ = MagicMock()
        circ.instructions = instructions
        return circ

    def _make_inst(self, gate_cls: str, qubits: list[int], angle: float | None = None):
        # Create a real class so type(gate).__name__ == gate_cls
        attrs: dict = {}
        if angle is not None:
            attrs["angle"] = angle
        GateClass = type(gate_cls, (), attrs)
        gate = GateClass()

        inst = MagicMock()
        inst.operator = gate
        inst.target = qubits
        return inst

    def test_h_gate(self, mock_braket):
        from ..adapters import from_braket
        circ = self._make_circ([self._make_inst("H", [0])])
        result = from_braket(circ)
        assert result[0].name == "h"
        assert result[0].qubits == [0]

    def test_cnot_gate(self, mock_braket):
        from ..adapters import from_braket
        circ = self._make_circ([self._make_inst("CNot", [0, 1])])
        result = from_braket(circ)
        assert result[0].name == "cx"
        assert result[0].qubits == [0, 1]

    def test_cz_gate(self, mock_braket):
        from ..adapters import from_braket
        circ = self._make_circ([self._make_inst("CZ", [0, 1])])
        result = from_braket(circ)
        assert result[0].name == "cz"

    def test_swap_gate(self, mock_braket):
        from ..adapters import from_braket
        circ = self._make_circ([self._make_inst("Swap", [0, 1])])
        result = from_braket(circ)
        assert result[0].name == "swap"

    def test_rx_parametric(self, mock_braket):
        from ..adapters import from_braket
        circ = self._make_circ([self._make_inst("Rx", [0], angle=1.5707963)])
        result = from_braket(circ)
        assert result[0].name == "rx"
        assert math.isclose(result[0].params[0], 1.5707963)

    def test_ry_parametric(self, mock_braket):
        from ..adapters import from_braket
        circ = self._make_circ([self._make_inst("Ry", [0], angle=math.pi / 4)])
        result = from_braket(circ)
        assert result[0].name == "ry"

    def test_rz_parametric(self, mock_braket):
        from ..adapters import from_braket
        circ = self._make_circ([self._make_inst("Rz", [0], angle=math.pi)])
        result = from_braket(circ)
        assert result[0].name == "rz"

    def test_s_si_t_ti(self, mock_braket):
        from ..adapters import from_braket
        for cls, name in [("S", "s"), ("Si", "sdg"), ("T", "t"), ("Ti", "tdg")]:
            circ = self._make_circ([self._make_inst(cls, [0])])
            result = from_braket(circ)
            assert result[0].name == name, f"{cls} should map to {name}"

    def test_ccnot_gate(self, mock_braket):
        from ..adapters import from_braket
        circ = self._make_circ([self._make_inst("CCNot", [0, 1, 2])])
        result = from_braket(circ)
        assert result[0].name == "ccx"

    def test_v_vi_gates(self, mock_braket):
        from ..adapters import from_braket
        circ = self._make_circ([self._make_inst("V", [0])])
        result = from_braket(circ)
        assert result[0].name == "sx"
        circ2 = self._make_circ([self._make_inst("Vi", [0])])
        result2 = from_braket(circ2)
        assert result2[0].name == "sxdg"

    def test_empty_circuit(self, mock_braket):
        from ..adapters import from_braket
        circ = self._make_circ([])
        assert from_braket(circ) == []

    def test_unknown_gate_raises(self, mock_braket):
        from ..adapters import from_braket, AdapterError
        circ = self._make_circ([self._make_inst("FancyGate", [0])])
        with pytest.raises(AdapterError):
            from_braket(circ)

    def test_multi_gate_circuit(self, mock_braket):
        from ..adapters import from_braket
        circ = self._make_circ([
            self._make_inst("H", [0]),
            self._make_inst("CNot", [0, 1]),
            self._make_inst("Rz", [1], angle=math.pi / 4),
        ])
        result = from_braket(circ)
        assert [g.name for g in result] == ["h", "cx", "rz"]

    def test_xyzgates(self, mock_braket):
        from ..adapters import from_braket
        for cls, name in [("X", "x"), ("Y", "y"), ("Z", "z")]:
            circ = self._make_circ([self._make_inst(cls, [0])])
            result = from_braket(circ)
            assert result[0].name == name


# ===========================================================================
# __init__ export smoke-test
# ===========================================================================

def test_adapters_importable_from_package():
    """All adapter symbols are exported from quantx_engine."""
    import quantx_engine  # noqa: F401
    # Only check the AdapterError since the rest require the lazy imports
    from quantx_engine.adapters import AdapterError
    assert AdapterError is not None
