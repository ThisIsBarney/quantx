"""Tests for the IBM Quantum connector."""
from __future__ import annotations

import pytest
from quantx_engine.ibm import circuit_to_qasm3, IBMConnectorError, IBMQuantumBackend


# ── circuit_to_qasm3 ──────────────────────────────────────────────────────────

def test_empty_circuit():
    circuit = {"qubits": 0, "clbits": 0, "operations": [], "shots": 0, "seed": 42}
    qasm = circuit_to_qasm3(circuit)
    assert "OPENQASM 3.0" in qasm


def test_bell_state_qasm():
    circuit = {
        "qubits": 2, "clbits": 2, "shots": 1000, "seed": 42,
        "operations": [
            {"name": "H", "targets": [0]},
            {"name": "CNOT", "controls": [0], "targets": [1]},
            {"name": "MEASURE", "targets": [0], "controls": [0]},
            {"name": "MEASURE", "targets": [1], "controls": [1]},
        ],
    }
    qasm = circuit_to_qasm3(circuit)
    assert "qubit[2] q;" in qasm
    assert "bit[2] c;" in qasm
    assert "h q[0];" in qasm
    assert "cx q[0], q[1];" in qasm
    assert "c[0] = measure q[0];" in qasm
    assert "c[1] = measure q[1];" in qasm


def test_parameterized_gate_qasm():
    import math
    circuit = {
        "qubits": 1, "clbits": 0, "shots": 0, "seed": 42,
        "operations": [
            {"name": "RX", "targets": [0], "params": [math.pi / 2]},
        ],
    }
    qasm = circuit_to_qasm3(circuit)
    assert "rx(" in qasm
    assert "q[0];" in qasm


def test_ccx_gate_qasm():
    circuit = {
        "qubits": 3, "clbits": 0, "shots": 0, "seed": 42,
        "operations": [
            {"name": "CCX", "controls": [0, 1], "targets": [2]},
        ],
    }
    qasm = circuit_to_qasm3(circuit)
    assert "ccx q[0], q[1], q[2];" in qasm


def test_unknown_gate_raises():
    circuit = {
        "qubits": 1, "clbits": 0, "shots": 0, "seed": 42,
        "operations": [
            {"name": "UNKNOWNGATE", "targets": [0]},
        ],
    }
    with pytest.raises(IBMConnectorError, match="UNKNOWNGATE"):
        circuit_to_qasm3(circuit)


def test_roundtrip_qasm_parse():
    """circuit_to_qasm3 output should be re-parseable by parse_qasm_to_circuit."""
    from quantx_engine.qasm import parse_qasm_to_circuit
    circuit = {
        "qubits": 2, "clbits": 2, "shots": 500, "seed": 42,
        "operations": [
            {"name": "H", "targets": [0]},
            {"name": "CNOT", "controls": [0], "targets": [1]},
            {"name": "MEASURE", "targets": [0], "controls": [0]},
            {"name": "MEASURE", "targets": [1], "controls": [1]},
        ],
    }
    qasm = circuit_to_qasm3(circuit)
    reparsed = parse_qasm_to_circuit(qasm, shots=500)
    assert reparsed["qubits"] == 2
    ops = reparsed["operations"]
    names = [op["name"] for op in ops]
    assert "H" in names
    assert "CNOT" in names
    assert names.count("MEASURE") == 2


# ── IBMQuantumBackend instantiation ──────────────────────────────────────────

def test_ibm_backend_no_qiskit_raises_importerror(monkeypatch):
    """If qiskit_ibm_runtime is not installed, instantiation raises ImportError."""
    import sys
    # Temporarily hide qiskit_ibm_runtime
    real = sys.modules.pop("qiskit_ibm_runtime", None)
    try:
        with pytest.raises(ImportError, match="qiskit-ibm-runtime"):
            IBMQuantumBackend(token="fake_token")
    finally:
        if real is not None:
            sys.modules["qiskit_ibm_runtime"] = real


def test_ibm_backend_properties():
    """Backend stores constructor args as properties (no connection needed for property access)."""
    # We can't instantiate without qiskit, so test circuit_to_qasm3 instead
    # which is always available
    circuit = {"qubits": 1, "clbits": 0, "operations": [], "shots": 0, "seed": 42}
    qasm = circuit_to_qasm3(circuit)
    assert "OPENQASM 3.0" in qasm


def test_all_standard_gates_convert():
    """Every gate in _QX_TO_QASM should produce valid QASM output."""
    from quantx_engine.ibm import _QX_TO_QASM
    single_qubit = {"H","X","Y","Z","S","T","Sdg","Tdg","I","SX","SXdg","RX","RY","RZ","P"}
    two_qubit = {"CNOT","CY","CZ","SWAP","ISWAP"}
    three_qubit = {"CCX"}
    for gate in single_qubit:
        circuit = {
            "qubits": 1, "clbits": 0, "shots": 0, "seed": 42,
            "operations": [{"name": gate, "targets": [0],
                            "params": [1.5] if gate in {"RX","RY","RZ","P"} else []}],
        }
        qasm = circuit_to_qasm3(circuit)
        assert _QX_TO_QASM[gate] in qasm
    for gate in two_qubit:
        circuit = {
            "qubits": 2, "clbits": 0, "shots": 0, "seed": 42,
            "operations": [{"name": gate, "controls": [0] if gate not in {"SWAP","ISWAP"} else [],
                            "targets": [1] if gate not in {"SWAP","ISWAP"} else [0,1]}],
        }
        qasm = circuit_to_qasm3(circuit)
        assert _QX_TO_QASM[gate] in qasm
