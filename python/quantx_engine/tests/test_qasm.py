"""Tests for the OpenQASM 3.0 parser / transpiler."""
from __future__ import annotations

import math
import pytest
from quantx_engine.qasm import parse_qasm_to_circuit, ParseError


def test_empty_program():
    circuit = parse_qasm_to_circuit("OPENQASM 3.0;")
    assert circuit["qubits"] == 0
    assert circuit["operations"] == []


def test_qubit_declaration():
    circuit = parse_qasm_to_circuit("OPENQASM 3.0;\nqubit[3] q;")
    assert circuit["qubits"] == 3


def test_qubit_scalar_declaration():
    circuit = parse_qasm_to_circuit("OPENQASM 3.0;\nqubit q;")
    assert circuit["qubits"] == 1


def test_bell_state():
    src = """
    OPENQASM 3.0;
    qubit[2] q;
    bit[2] c;
    h q[0];
    cx q[0], q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];
    """
    circuit = parse_qasm_to_circuit(src, shots=1000)
    assert circuit["qubits"] == 2
    assert circuit["shots"] == 1000
    ops = circuit["operations"]
    assert ops[0] == {"name": "H", "targets": [0]}
    assert ops[1] == {"name": "CNOT", "controls": [0], "targets": [1]}
    assert ops[2]["name"] == "MEASURE"
    assert ops[3]["name"] == "MEASURE"


def test_parameterized_gate():
    src = "OPENQASM 3.0;\nqubit q;\nrx(pi/2) q;"
    circuit = parse_qasm_to_circuit(src)
    ops = circuit["operations"]
    assert ops[0]["name"] == "RX"
    assert abs(ops[0]["params"][0] - math.pi / 2) < 1e-10


def test_multiple_registers():
    src = """
    OPENQASM 3.0;
    qubit[2] a;
    qubit[2] b;
    h a[0];
    cx a[1], b[0];
    """
    circuit = parse_qasm_to_circuit(src)
    assert circuit["qubits"] == 4
    ops = circuit["operations"]
    assert ops[0] == {"name": "H", "targets": [0]}
    assert ops[1] == {"name": "CNOT", "controls": [1], "targets": [2]}


def test_measure_arrow_syntax():
    src = """
    OPENQASM 3.0;
    qubit[2] q;
    bit[2] c;
    h q[0];
    measure q[0] -> c[0];
    """
    circuit = parse_qasm_to_circuit(src)
    ops = circuit["operations"]
    assert ops[1]["name"] == "MEASURE"
    assert ops[1]["targets"] == [0]


def test_barrier_ignored():
    src = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\nbarrier q;\nx q[1];"
    circuit = parse_qasm_to_circuit(src)
    names = [op["name"] for op in circuit["operations"]]
    assert "H" in names
    assert "X" in names


def test_comments_stripped():
    src = """
    // This is a comment
    OPENQASM 3.0;
    qubit[1] q; /* block comment */
    h q[0]; // inline comment
    """
    circuit = parse_qasm_to_circuit(src)
    assert circuit["qubits"] == 1
    assert circuit["operations"][0]["name"] == "H"


def test_gate_definition_skipped():
    src = """
    OPENQASM 3.0;
    gate mygate q { h q; }
    qubit[1] q;
    mygate q[0];
    """
    # mygate is not in _GATE_MAP, so it is silently skipped
    circuit = parse_qasm_to_circuit(src)
    assert circuit["qubits"] == 1
    assert circuit["operations"] == []


def test_include_ignored():
    src = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit q;\nh q;'
    circuit = parse_qasm_to_circuit(src)
    assert circuit["operations"][0]["name"] == "H"


def test_qasm2_style_qreg():
    src = "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\nh q[0];\n"
    circuit = parse_qasm_to_circuit(src)
    assert circuit["qubits"] == 2
    assert circuit["operations"][0]["name"] == "H"


def test_ccx_toffoli():
    src = "OPENQASM 3.0;\nqubit[3] q;\nccx q[0], q[1], q[2];"
    circuit = parse_qasm_to_circuit(src)
    op = circuit["operations"][0]
    assert op["name"] == "CCX"
    assert op["controls"] == [0, 1]
    assert op["targets"] == [2]


def test_pi_expressions():
    for expr, expected in [
        ("pi", math.pi),
        ("2*pi", 2 * math.pi),
        ("pi/4", math.pi / 4),
        ("3.14159265", 3.14159265),
    ]:
        src = f"OPENQASM 3.0;\nqubit q;\nrx({expr}) q;"
        circuit = parse_qasm_to_circuit(src)
        assert abs(circuit["operations"][0]["params"][0] - expected) < 1e-5


def test_parse_error_on_unknown_statement():
    with pytest.raises(ParseError):
        parse_qasm_to_circuit("OPENQASM 3.0;\nif (c == 0) { h q[0]; }")


def test_parse_error_undefined_register():
    with pytest.raises(ParseError):
        parse_qasm_to_circuit("OPENQASM 3.0;\nh q[0];")


def test_shots_and_seed_propagated():
    src = "OPENQASM 3.0;\nqubit q;\nh q;"
    circuit = parse_qasm_to_circuit(src, shots=512, seed=99)
    assert circuit["shots"] == 512
    assert circuit["seed"] == 99


def test_parameter_map_in_output():
    src = "OPENQASM 3.0;\nqubit q;\nrx(pi/2) q;"
    circuit = parse_qasm_to_circuit(src, parameter_map={"theta": 1.5})
    assert "parameter_map" in circuit
    assert circuit["parameter_map"]["theta"] == 1.5


def test_grover_diffuser_structure():
    """3-qubit circuit with multiple gate types parses cleanly."""
    src = """
    OPENQASM 3.0;
    qubit[3] q;
    bit[3] c;
    h q[0];
    h q[1];
    h q[2];
    x q[0];
    x q[1];
    x q[2];
    ccx q[0], q[1], q[2];
    x q[0];
    x q[1];
    x q[2];
    h q[0];
    h q[1];
    h q[2];
    c[0] = measure q[0];
    c[1] = measure q[1];
    c[2] = measure q[2];
    """
    circuit = parse_qasm_to_circuit(src, shots=1024)
    assert circuit["qubits"] == 3
    assert circuit["shots"] == 1024
    # Check we have the CCX in there
    names = [op["name"] for op in circuit["operations"]]
    assert "CCX" in names
    assert names.count("H") == 6
    assert names.count("X") == 6
    assert names.count("MEASURE") == 3
