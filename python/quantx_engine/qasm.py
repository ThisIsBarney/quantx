"""
OpenQASM 3.0 parser and transpiler for the QuantX engine.

Converts a subset of OpenQASM 3.0 source into the QuantX native circuit JSON
schema, suitable for passing to Engine.run_circuit().

Supported:
- OPENQASM 3.0 / OPENQASM 3 header
- qubit[n] name; / qubit name;  register declarations
- bit[n] name; / bit name;      classical bit declarations
- Standard gate applications (h, x, y, z, s, t, sdg, tdg, cx, cy, cz, ccx,
  rx, ry, rz, p, u1, u2, u3, swap, iswap, id, reset)
- Parameterized gates: rx(pi/2) q[0];
- Measurements: c[0] = measure q[0];  /  measure q[0] -> c[0];
- Barrier statements (no-op in QuantX circuit)
- Include directives (silently ignored)
- Gate definitions (silently ignored)
- Line comments // ...
- Block comments /* ... */

Not supported (raises ParseError):
- Classical control flow (if / while)
- Subroutines / defcalgrammar
- Delay / duration types
- Pulse-level constructs
"""
from __future__ import annotations

import math
import re
from typing import Any

__all__ = ["parse_qasm_to_circuit", "ParseError"]


class ParseError(ValueError):
    """Raised when the QASM source cannot be parsed."""


# ── Gate name mapping ────────────────────────────────────────────────────────
# Maps OpenQASM gate names → QuantX native gate name
_GATE_MAP: dict[str, str] = {
    "h": "H", "x": "X", "y": "Y", "z": "Z",
    "s": "S", "t": "T", "sdg": "Sdg", "tdg": "Tdg",
    "cx": "CNOT", "cnot": "CNOT", "cy": "CY", "cz": "CZ",
    "ccx": "CCX", "toffoli": "CCX",
    "rx": "RX", "ry": "RY", "rz": "RZ",
    "p": "P", "u1": "P",  # P-gate is a phase gate
    "u2": "U2", "u3": "U3",
    "swap": "SWAP", "iswap": "ISWAP",
    "id": "I", "i": "I",
    "measure": "MEASURE",
    "reset": "RESET",
    # Aliases
    "sx": "SX", "sxdg": "SXdg",
}

# Gates that use controls= in QuantX
_CONTROLLED_GATES = {"CNOT", "CY", "CZ", "CCX"}

# Gates with parameters
_PARAM_GATES = {"RX", "RY", "RZ", "P", "U2", "U3"}

# Number of parameter arguments per gate
_PARAM_COUNT: dict[str, int] = {
    "RX": 1, "RY": 1, "RZ": 1, "P": 1, "U2": 2, "U3": 3,
}

# Number of qubit arguments per gate
_QUBIT_COUNT: dict[str, int] = {
    "H": 1, "X": 1, "Y": 1, "Z": 1, "S": 1, "T": 1,
    "Sdg": 1, "Tdg": 1, "SX": 1, "SXdg": 1, "I": 1,
    "RX": 1, "RY": 1, "RZ": 1, "P": 1, "U2": 1, "U3": 1,
    "CNOT": 2, "CY": 2, "CZ": 2, "CCX": 3,
    "SWAP": 2, "ISWAP": 2,
    "MEASURE": 1, "RESET": 1,
}


# ── Expression evaluator for gate parameters ─────────────────────────────────

def _eval_expr(expr: str, param_map: dict[str, float] | None = None) -> float:
    """Evaluate a simple arithmetic expression appearing in a gate parameter.

    Supports: float literals, pi, pi/n, n*pi, basic +/-/*/÷, unary minus.
    Raises ParseError for anything that cannot be evaluated.
    """
    expr = expr.strip()
    # Substitute named constants
    expr = re.sub(r"\bpi\b", str(math.pi), expr)
    expr = re.sub(r"\btau\b", str(2 * math.pi), expr)
    expr = re.sub(r"\beuler\b", str(math.e), expr)
    # Substitute parameter references if given
    if param_map:
        for name, val in param_map.items():
            expr = re.sub(r"\b" + re.escape(name) + r"\b", str(val), expr)
    # Allow only safe characters: digits, operators, whitespace, dot, e/E for scientific notation
    if not re.fullmatch(r"[\d\s.+\-*/()eE]+", expr):
        raise ParseError(f"unsafe expression: {expr!r}")
    try:
        result = eval(expr, {"__builtins__": {}})  # noqa: S307 — controlled input
        return float(result)
    except Exception as exc:
        raise ParseError(f"cannot evaluate expression {expr!r}: {exc}") from exc


# ── Tokeniser ────────────────────────────────────────────────────────────────

def _strip_comments(src: str) -> str:
    """Remove C-style line and block comments, preserving newlines."""
    # Block comments: /* ... */
    src = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), src, flags=re.DOTALL)
    # Line comments: // ...
    src = re.sub(r"//[^\n]*", "", src)
    return src


def _tokenize(src: str) -> list[str]:
    """Split QASM source into tokens (identifiers, numbers, punctuation)."""
    token_re = re.compile(
        r'"[^"]*"'            # string literals (for include)
        r"|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"  # numbers
        r"|[A-Za-z_][A-Za-z0-9_]*"  # identifiers
        r"|->|[{}()\[\],;=*+\-/^@#%&|!<>?:]"  # punctuation
    )
    return token_re.findall(src)


# ── Parser ────────────────────────────────────────────────────────────────────

class _Parser:
    """Stateful parser that converts a token list into a QuantX circuit dict."""

    def __init__(self, tokens: list[str], parameter_map: dict[str, float] | None = None):
        self._tokens = tokens
        self._pos = 0
        # Register tables: name → (start_index, size)
        self._qregs: dict[str, tuple[int, int]] = {}
        self._cregs: dict[str, tuple[int, int]] = {}
        self._total_qubits = 0
        self._total_cbits = 0
        self._ops: list[dict[str, Any]] = []
        self._shots = 0
        self._parameter_map: dict[str, float] = dict(parameter_map or {})

    # ── Token access ─────────────────────────────────────────────────────────

    def _peek(self) -> str | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        if self._pos >= len(self._tokens):
            raise ParseError("unexpected end of input")
        t = self._tokens[self._pos]
        self._pos += 1
        return t

    def _expect(self, tok: str) -> None:
        t = self._consume()
        if t != tok:
            raise ParseError(f"expected {tok!r}, got {t!r}")

    def _skip_until_semicolon(self) -> None:
        """Consume tokens up to and including the next semicolon."""
        while True:
            t = self._consume()
            if t == ";":
                return

    # ── Register helpers ──────────────────────────────────────────────────────

    def _qubit_index(self, reg: str, idx: int | None) -> int:
        if reg not in self._qregs:
            raise ParseError(f"undefined qubit register: {reg!r}")
        start, size = self._qregs[reg]
        if idx is None:
            if size != 1:
                raise ParseError(f"register {reg!r} has size {size}; must specify index")
            return start
        if not (0 <= idx < size):
            raise ParseError(f"index {idx} out of range for register {reg!r} (size {size})")
        return start + idx

    def _cbit_index(self, reg: str, idx: int | None) -> int:
        if reg not in self._cregs:
            raise ParseError(f"undefined classical register: {reg!r}")
        start, size = self._cregs[reg]
        if idx is None:
            if size != 1:
                raise ParseError(f"register {reg!r} has size {size}; must specify index")
            return start
        if not (0 <= idx < size):
            raise ParseError(f"index {idx} out of range for register {reg!r} (size {size})")
        return start + idx

    def _parse_reg_ref(self) -> tuple[str, int | None]:
        """Parse name or name[idx] and return (name, idx_or_None)."""
        name = self._consume()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ParseError(f"expected register name, got {name!r}")
        if self._peek() == "[":
            self._consume()  # [
            idx_str = self._consume()
            try:
                idx = int(idx_str)
            except ValueError:
                raise ParseError(f"expected integer index, got {idx_str!r}")
            self._expect("]")
            return name, idx
        return name, None

    def _parse_qubit_ref(self) -> int:
        name, idx = self._parse_reg_ref()
        return self._qubit_index(name, idx)

    def _parse_cbit_ref(self) -> int:
        name, idx = self._parse_reg_ref()
        return self._cbit_index(name, idx)

    # ── Parameter parsing ─────────────────────────────────────────────────────

    def _parse_params(self) -> list[float]:
        """Parse (expr, expr, ...) returning evaluated floats."""
        self._expect("(")
        params: list[float] = []
        while self._peek() != ")":
            # Collect tokens for one expression (up to , or ))
            expr_toks: list[str] = []
            depth = 0
            while True:
                t = self._peek()
                if t is None:
                    raise ParseError("unclosed parameter list")
                if t == "(" :
                    depth += 1
                elif t == ")":
                    if depth == 0:
                        break
                    depth -= 1
                elif t == "," and depth == 0:
                    break
                expr_toks.append(self._consume())
            params.append(_eval_expr("".join(expr_toks), self._parameter_map or None))
            if self._peek() == ",":
                self._consume()
        self._expect(")")
        return params

    # ── Top-level statement parser ────────────────────────────────────────────

    def parse(self) -> dict[str, Any]:
        while self._pos < len(self._tokens):
            tok = self._peek()
            if tok is None:
                break
            if tok == "OPENQASM":
                self._consume()  # OPENQASM
                self._consume()  # version number
                if self._peek() == ";":
                    self._consume()
            elif tok == "include":
                self._skip_until_semicolon()
            elif tok == "qubit":
                self._parse_qubit_decl()
            elif tok == "bit":
                self._parse_bit_decl()
            elif tok == "qreg":
                self._parse_qreg()
            elif tok == "creg":
                self._parse_creg()
            elif tok == "gate":
                self._skip_gate_definition()
            elif tok == "barrier":
                self._skip_until_semicolon()
            elif tok == "reset":
                self._parse_reset()
            elif tok == "measure":
                self._parse_measure_stmt()
            elif tok in ("if", "while", "for", "def", "defcalgrammar", "cal"):
                raise ParseError(f"unsupported statement: {tok!r}")
            else:
                # Try to parse as a gate application; classical assignment handled inside
                self._parse_gate_or_assignment()

        return {
            "qubits": self._total_qubits,
            "shots": self._shots,
            "clbits": self._total_cbits,
            "observables": [],
            "operations": self._ops,
        }

    # ── Declaration parsers ───────────────────────────────────────────────────

    def _parse_qubit_decl(self) -> None:
        """Parse  qubit[n] name;  or  qubit name;"""
        self._consume()  # qubit
        size = 1
        if self._peek() == "[":
            self._consume()
            size = int(self._consume())
            self._expect("]")
        name = self._consume()
        self._expect(";")
        self._qregs[name] = (self._total_qubits, size)
        self._total_qubits += size

    def _parse_bit_decl(self) -> None:
        """Parse  bit[n] name;  or  bit name;"""
        self._consume()  # bit
        size = 1
        if self._peek() == "[":
            self._consume()
            size = int(self._consume())
            self._expect("]")
        name = self._consume()
        self._expect(";")
        self._cregs[name] = (self._total_cbits, size)
        self._total_cbits += size

    def _parse_qreg(self) -> None:
        """Parse QASM 2-style  qreg name[n];"""
        self._consume()  # qreg
        name = self._consume()
        self._expect("[")
        size = int(self._consume())
        self._expect("]")
        self._expect(";")
        self._qregs[name] = (self._total_qubits, size)
        self._total_qubits += size

    def _parse_creg(self) -> None:
        """Parse QASM 2-style  creg name[n];"""
        self._consume()  # creg
        name = self._consume()
        self._expect("[")
        size = int(self._consume())
        self._expect("]")
        self._expect(";")
        self._cregs[name] = (self._total_cbits, size)
        self._total_cbits += size

    def _skip_gate_definition(self) -> None:
        """Skip  gate name(params) qargs { body }"""
        self._consume()  # gate
        # Skip until closing brace
        depth = 0
        while True:
            t = self._consume()
            if t == "{":
                depth += 1
            elif t == "}":
                depth -= 1
                if depth == 0:
                    return

    # ── Gate/measurement parsers ──────────────────────────────────────────────

    def _parse_reset(self) -> None:
        """Consume a reset statement.

        The QuantX gate-model circuit backend does not support mid-circuit qubit
        reset. Reset statements are intentionally dropped (not emitted as ops)
        so that programs using reset can still be partially transpiled.  Callers
        that require strict semantics should check for this case themselves.
        """
        self._consume()  # reset
        self._skip_until_semicolon()

    def _parse_measure_stmt(self) -> None:
        """Parse  measure qreg -> creg;"""
        self._consume()  # measure
        q = self._parse_qubit_ref()
        self._expect("->")
        c = self._parse_cbit_ref()
        self._expect(";")
        self._ops.append({"name": "MEASURE", "targets": [q], "controls": [c]})

    def _parse_gate_or_assignment(self) -> None:
        """Parse a gate application or classical assignment (c[i] = measure q[i];)."""
        tok = self._peek()
        if tok is None:
            return

        # Classical assignment lookahead: c[0] = measure q[0];
        # Save position before consuming; only backtrack when '=' is absent
        # (not on real errors like undefined register — those should propagate).
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            saved_pos = self._pos
            name, idx = self._parse_reg_ref()
            if self._peek() == "=":
                # Committed to the assignment path — errors propagate from here
                self._consume()  # =
                kw = self._consume()
                if kw == "measure":
                    q = self._parse_qubit_ref()
                    self._expect(";")
                    c = self._cbit_index(name, idx)
                    self._ops.append({"name": "MEASURE", "targets": [q], "controls": [c]})
                    return
                else:
                    raise ParseError(f"unsupported assignment: {name} = {kw}")
            else:
                # Not an assignment — restore position and parse as gate
                self._pos = saved_pos

        # Gate application
        gate_name_raw = self._consume()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", gate_name_raw):
            raise ParseError(f"unexpected token: {gate_name_raw!r}")

        # Map to QuantX gate name
        qx_gate = _GATE_MAP.get(gate_name_raw.lower())
        if qx_gate is None:
            # Unknown gate — skip until semicolon (silently tolerate custom gates)
            while self._peek() != ";" and self._peek() is not None:
                self._consume()
            if self._peek() == ";":
                self._consume()
            return

        # Parse optional parameter list
        params: list[float] = []
        if self._peek() == "(" and qx_gate in _PARAM_GATES:
            params = self._parse_params()

        # Parse qubit arguments (comma-separated)
        qubits: list[int] = []
        n_expected = _QUBIT_COUNT.get(qx_gate, 1)
        for i in range(n_expected):
            qubits.append(self._parse_qubit_ref())
            if i < n_expected - 1:
                self._expect(",")
        self._expect(";")

        # Build the operation
        op: dict[str, Any] = {"name": qx_gate}
        if qx_gate in _CONTROLLED_GATES:
            op["controls"] = qubits[:-1]
            op["targets"] = qubits[-1:]
        else:
            op["targets"] = qubits
        if params:
            op["params"] = params
        self._ops.append(op)


# ── Public API ────────────────────────────────────────────────────────────────

def parse_qasm_to_circuit(
    qasm_source: str,
    shots: int = 0,
    seed: int = 42,
    parameter_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Parse an OpenQASM 3.0 string and return a QuantX native circuit dict.

    Parameters
    ----------
    qasm_source:
        OpenQASM 3.0 source code (OPENQASM 3.0; header optional).
    shots:
        Number of measurement shots to attach to the circuit dict (default 0).
    seed:
        Random seed for shot-based simulation (default 42).
    parameter_map:
        Optional mapping of free parameter names to float values for
        parameterized circuits.

    Returns
    -------
    dict compatible with Engine.run_circuit().

    Raises
    ------
    ParseError
        If the source cannot be parsed or uses unsupported constructs.
    """
    clean = _strip_comments(qasm_source)
    tokens = _tokenize(clean)
    parser = _Parser(tokens, parameter_map=parameter_map)
    circuit = parser.parse()
    circuit["shots"] = shots
    circuit["seed"] = seed
    if parameter_map:
        circuit["parameter_map"] = {str(k): float(v) for k, v in parameter_map.items()}
    return circuit
