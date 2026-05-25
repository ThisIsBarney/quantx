"""QuantumCircuit fluent builder API (M161).

A Qiskit/Cirq-style circuit construction layer built on top of
:class:`~quantx_engine.statevector.StatevectorSimulator`.

Example
-------
>>> from quantx_engine import QuantumCircuit
>>> qc = QuantumCircuit(3)
>>> qc.h(0).cx(0, 1).cx(0, 2).measure_all()
>>> result = qc.run(n_shots=1024)
>>> print(result.counts)         # {'000': 510, '111': 514}
>>> print(result.statevector)    # np.ndarray shape (8,)

Parametric gates (for VQE etc.)
--------------------------------
>>> from quantx_engine.circuit import Symbol
>>> theta = Symbol("theta")
>>> qc = QuantumCircuit(1)
>>> qc.rx(theta, 0)
>>> result = qc.run(params={"theta": 1.5707963})
"""
from __future__ import annotations

import dataclasses
import math
import re
from typing import Sequence

import numpy as np

from .statevector import StatevectorSimulator

# ── Symbolic parameters ────────────────────────────────────────────────────────


class Symbol:
    """Named symbolic parameter for parametric gates (e.g. in VQE).

    Use a ``Symbol`` wherever a rotation angle is expected.  Before calling
    :meth:`QuantumCircuit.run` you must bind all symbols to numeric values via
    the *params* argument::

        theta = Symbol("theta")
        qc    = QuantumCircuit(1)
        qc.rx(theta, 0)
        result = qc.run(n_shots=0, params={"theta": math.pi / 2})

    Parameters
    ----------
    name : str
        Human-readable name used as the key in the ``params`` dict.
    """

    def __init__(self, name: str) -> None:
        self.name = str(name)

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"

    def __str__(self) -> str:
        return self.name

    def __float__(self) -> float:
        raise TypeError(
            f"Symbol {self.name!r} has not been bound to a numeric value. "
            "Pass params={...} to QuantumCircuit.run()."
        )


_Param = float | Symbol  # type alias used throughout this module


# ── Internal gate operation record ────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class _GateOp:
    """Immutable record of a single gate or measurement in the circuit."""

    name: str                      # canonical upper-case name, e.g. "H", "CX", "RX"
    qubits: tuple[int, ...]
    params: tuple[_Param, ...]     # rotation angles, phases, etc.
    is_measure: bool = False


# ── Circuit result ─────────────────────────────────────────────────────────────


@dataclasses.dataclass
class CircuitResult:
    """Result returned by :meth:`QuantumCircuit.run`.

    Attributes
    ----------
    statevector : np.ndarray
        Final state as a complex128 array, shape ``(2**n_qubits,)``.
    counts : dict[str, int]
        Measurement counts mapping bitstring → count.  Empty when
        ``n_shots=0``.
    n_qubits : int
        Number of qubits in the circuit.
    backend : str
        Name of the execution backend.
    """

    statevector: np.ndarray
    counts: dict[str, int]
    n_qubits: int
    backend: str = "pure_python_statevector"


# ── QASM3 helpers ─────────────────────────────────────────────────────────────

# Map internal gate names → QASM3 gate identifiers
_TO_QASM3: dict[str, str] = {
    "H": "h", "X": "x", "Y": "y", "Z": "z",
    "S": "s", "T": "t", "SDG": "sdg", "TDG": "tdg", "I": "id",
    "RX": "rx", "RY": "ry", "RZ": "rz", "P": "p", "U3": "u",
    "CX": "cx", "CNOT": "cx", "CY": "cy", "CZ": "cz", "CH": "ch",
    "SWAP": "swap", "ISWAP": "iswap",
    "CCX": "ccx", "TOFFOLI": "ccx",
    "CSWAP": "cswap", "FREDKIN": "cswap",
}

# Map QASM3 gate identifiers → internal gate names (used by from_qasm3)
_FROM_QASM3: dict[str, str] = {
    "h": "H", "x": "X", "y": "Y", "z": "Z",
    "s": "S", "t": "T", "sdg": "SDG", "tdg": "TDG", "id": "I",
    "rx": "RX", "ry": "RY", "rz": "RZ", "p": "P", "u": "U3",
    "cx": "CX", "cy": "CY", "cz": "CZ", "ch": "CH",
    "swap": "SWAP", "iswap": "ISWAP",
    "ccx": "CCX", "cswap": "CSWAP",
}


def _fmt_angle(v: _Param) -> str:
    """Format a gate angle for QASM3 output."""
    if isinstance(v, Symbol):
        return v.name
    x = float(v)
    if x == 0.0:
        return "0"
    # Try common pi multiples: ±k*pi/d for small k,d
    for d in (1, 2, 3, 4, 6, 8):
        for k in range(1, 5):
            for sign in (1, -1):
                candidate = sign * math.pi * k / d
                if abs(x - candidate) < 1e-12:
                    frac = "" if k == 1 else f"{k}*"
                    denom = "" if d == 1 else f"/{d}"
                    prefix = "-" if sign < 0 else ""
                    return f"{prefix}{frac}pi{denom}"
    return repr(x)


def _parse_angle(tok: str) -> float:
    """Parse a QASM3 angle token (may contain 'pi') to float."""
    tok = tok.strip()
    safe = {"pi": math.pi, "sqrt": math.sqrt, "cos": math.cos,
            "sin": math.sin, "tan": math.tan, "exp": math.exp}
    return float(eval(tok, {"__builtins__": {}}, safe))  # noqa: S307


def _gate_label(op: _GateOp) -> str:
    """Short human-readable label used in ASCII diagrams."""
    name = op.name
    if name in ("RX", "RY", "RZ", "P") and op.params:
        return f"{name}({_fmt_angle(op.params[0])})"
    if name == "U3" and len(op.params) >= 3:
        ps = ",".join(_fmt_angle(p) for p in op.params[:3])
        return f"U3({ps})"
    return name


# ── QuantumCircuit ─────────────────────────────────────────────────────────────


class QuantumCircuit:
    """Pythonic fluent circuit builder.

    All gate methods return *self* for chaining::

        sv = QuantumCircuit(2).h(0).cx(0, 1).run().statevector

    Parameters
    ----------
    n_qubits : int
        Number of qubits (1–25).
    name : str | None
        Optional human-readable label for the circuit.
    """

    def __init__(self, n_qubits: int, name: str | None = None) -> None:
        if not 1 <= n_qubits <= 25:
            raise ValueError(f"n_qubits must be in [1, 25], got {n_qubits}.")
        self._n: int = n_qubits
        self.name: str = name or f"circuit_{n_qubits}q"
        self._ops: list[_GateOp] = []
        self._measured: list[bool] = [False] * n_qubits

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        return self._n

    @property
    def operations(self) -> list[_GateOp]:
        """Snapshot of the gate/measurement operations list."""
        return list(self._ops)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check(self, *qubits: int) -> None:
        for q in qubits:
            if not 0 <= q < self._n:
                raise ValueError(
                    f"Qubit index {q} out of range [0, {self._n})."
                )

    def _add(
        self,
        name: str,
        qubits: tuple[int, ...],
        params: tuple[_Param, ...] = (),
    ) -> "QuantumCircuit":
        self._ops.append(_GateOp(name=name, qubits=qubits, params=params))
        return self

    # ── Single-qubit gates ────────────────────────────────────────────────────

    def i(self, qubit: int) -> "QuantumCircuit":
        """Identity gate."""
        self._check(qubit)
        return self._add("I", (qubit,))

    def h(self, qubit: int) -> "QuantumCircuit":
        """Hadamard gate."""
        self._check(qubit)
        return self._add("H", (qubit,))

    def x(self, qubit: int) -> "QuantumCircuit":
        """Pauli-X (NOT) gate."""
        self._check(qubit)
        return self._add("X", (qubit,))

    def y(self, qubit: int) -> "QuantumCircuit":
        """Pauli-Y gate."""
        self._check(qubit)
        return self._add("Y", (qubit,))

    def z(self, qubit: int) -> "QuantumCircuit":
        """Pauli-Z gate."""
        self._check(qubit)
        return self._add("Z", (qubit,))

    def s(self, qubit: int) -> "QuantumCircuit":
        """S gate (√Z, phase 90°)."""
        self._check(qubit)
        return self._add("S", (qubit,))

    def t(self, qubit: int) -> "QuantumCircuit":
        """T gate (√S, phase 45°)."""
        self._check(qubit)
        return self._add("T", (qubit,))

    def sdg(self, qubit: int) -> "QuantumCircuit":
        """S† (conjugate of S)."""
        self._check(qubit)
        return self._add("SDG", (qubit,))

    def tdg(self, qubit: int) -> "QuantumCircuit":
        """T† (conjugate of T)."""
        self._check(qubit)
        return self._add("TDG", (qubit,))

    def rx(self, theta: _Param, qubit: int) -> "QuantumCircuit":
        """Rx(θ) rotation gate."""
        self._check(qubit)
        return self._add("RX", (qubit,), (theta,))

    def ry(self, theta: _Param, qubit: int) -> "QuantumCircuit":
        """Ry(θ) rotation gate."""
        self._check(qubit)
        return self._add("RY", (qubit,), (theta,))

    def rz(self, theta: _Param, qubit: int) -> "QuantumCircuit":
        """Rz(θ) rotation gate."""
        self._check(qubit)
        return self._add("RZ", (qubit,), (theta,))

    def p(self, lam: _Param, qubit: int) -> "QuantumCircuit":
        """Phase gate P(λ): |1⟩ → e^{iλ}|1⟩."""
        self._check(qubit)
        return self._add("P", (qubit,), (lam,))

    def u3(
        self, theta: _Param, phi: _Param, lam: _Param, qubit: int
    ) -> "QuantumCircuit":
        """General single-qubit U3(θ, φ, λ) gate (OpenQASM U3/U)."""
        self._check(qubit)
        return self._add("U3", (qubit,), (theta, phi, lam))

    # ── Two-qubit gates ───────────────────────────────────────────────────────

    def cx(self, control: int, target: int) -> "QuantumCircuit":
        """CNOT (CX) gate."""
        self._check(control, target)
        if control == target:
            raise ValueError("control and target qubits must differ.")
        return self._add("CX", (control, target))

    def cnot(self, control: int, target: int) -> "QuantumCircuit":
        """Alias for cx()."""
        return self.cx(control, target)

    def cy(self, control: int, target: int) -> "QuantumCircuit":
        """Controlled-Y gate."""
        self._check(control, target)
        if control == target:
            raise ValueError("control and target qubits must differ.")
        return self._add("CY", (control, target))

    def cz(self, control: int, target: int) -> "QuantumCircuit":
        """Controlled-Z gate."""
        self._check(control, target)
        if control == target:
            raise ValueError("control and target qubits must differ.")
        return self._add("CZ", (control, target))

    def ch(self, control: int, target: int) -> "QuantumCircuit":
        """Controlled-H gate."""
        self._check(control, target)
        if control == target:
            raise ValueError("control and target qubits must differ.")
        return self._add("CH", (control, target))

    def swap(self, q0: int, q1: int) -> "QuantumCircuit":
        """SWAP gate."""
        self._check(q0, q1)
        if q0 == q1:
            raise ValueError("SWAP qubits must differ.")
        return self._add("SWAP", (q0, q1))

    def iswap(self, q0: int, q1: int) -> "QuantumCircuit":
        """iSWAP gate."""
        self._check(q0, q1)
        if q0 == q1:
            raise ValueError("iSWAP qubits must differ.")
        return self._add("ISWAP", (q0, q1))

    # ── Three-qubit gates ─────────────────────────────────────────────────────

    def ccx(self, c0: int, c1: int, target: int) -> "QuantumCircuit":
        """Toffoli (CCX) gate."""
        self._check(c0, c1, target)
        if len({c0, c1, target}) != 3:
            raise ValueError("CCX requires three distinct qubits.")
        return self._add("CCX", (c0, c1, target))

    def toffoli(self, c0: int, c1: int, target: int) -> "QuantumCircuit":
        """Alias for ccx()."""
        return self.ccx(c0, c1, target)

    def cswap(self, ctrl: int, q0: int, q1: int) -> "QuantumCircuit":
        """Fredkin (CSWAP) gate: swap q0 and q1 when ctrl=|1⟩."""
        self._check(ctrl, q0, q1)
        if len({ctrl, q0, q1}) != 3:
            raise ValueError("CSWAP requires three distinct qubits.")
        return self._add("CSWAP", (ctrl, q0, q1))

    def fredkin(self, ctrl: int, q0: int, q1: int) -> "QuantumCircuit":
        """Alias for cswap()."""
        return self.cswap(ctrl, q0, q1)

    # ── Measurement ───────────────────────────────────────────────────────────

    def measure(self, qubit: int) -> "QuantumCircuit":
        """Mark a single qubit for measurement (used in shots sampling)."""
        self._check(qubit)
        self._measured[qubit] = True
        self._ops.append(
            _GateOp("MEASURE", (qubit,), (), is_measure=True)
        )
        return self

    def measure_all(self) -> "QuantumCircuit":
        """Mark all qubits for measurement."""
        for q in range(self._n):
            self._measured[q] = True
        self._ops.append(
            _GateOp("MEASURE_ALL", tuple(range(self._n)), (), is_measure=True)
        )
        return self

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(
        self,
        n_shots: int = 0,
        seed: int | None = None,
        params: dict[str, float] | None = None,
    ) -> CircuitResult:
        """Execute the circuit and return a :class:`CircuitResult`.

        Parameters
        ----------
        n_shots : int
            Number of measurement shots.  0 = statevector only (no sampling).
        seed : int | None
            RNG seed for shot sampling.
        params : dict[str, float] | None
            Bindings for :class:`Symbol` parameters, e.g.
            ``{"theta": 1.5707963}``.

        Returns
        -------
        CircuitResult
        """
        params = params or {}
        sim = StatevectorSimulator(self._n)

        for op in self._ops:
            if op.is_measure:
                continue  # handled by shots sampling below

            resolved: list[float] = []
            for p in op.params:
                if isinstance(p, Symbol):
                    if p.name not in params:
                        raise ValueError(
                            f"Symbol {p.name!r} is not bound. "
                            f"Pass params={{{p.name!r}: <value>}} to run()."
                        )
                    resolved.append(float(params[p.name]))
                else:
                    resolved.append(float(p))

            _apply_to_sim(sim, op.name, op.qubits, tuple(resolved))

        sv = sim.statevector()
        counts: dict[str, int] = {}
        if n_shots > 0:
            counts = dict(sim.sample(n_shots, seed=seed))

        return CircuitResult(
            statevector=sv,
            counts=counts,
            n_qubits=self._n,
        )

    # ── QASM3 export ─────────────────────────────────────────────────────────

    def to_qasm3(self) -> str:
        """Serialise the circuit to an OpenQASM 3.0 string.

        Returns
        -------
        str
            Valid QASM3 that can be round-tripped via :meth:`from_qasm3`.
        """
        lines: list[str] = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            "",
            f"qubit[{self._n}] q;",
        ]
        has_measure = any(op.is_measure for op in self._ops)
        if has_measure:
            lines.append(f"bit[{self._n}] c;")
        lines.append("")

        for op in self._ops:
            if op.name == "MEASURE_ALL":
                lines.append("c = measure q;")
            elif op.name == "MEASURE":
                q = op.qubits[0]
                lines.append(f"c[{q}] = measure q[{q}];")
            else:
                qasm_name = _TO_QASM3.get(op.name, op.name.lower())
                qubit_args = ", ".join(f"q[{q}]" for q in op.qubits)
                if op.params:
                    param_str = "(" + ", ".join(_fmt_angle(p) for p in op.params) + ")"
                    lines.append(f"{qasm_name}{param_str} {qubit_args};")
                else:
                    lines.append(f"{qasm_name} {qubit_args};")

        return "\n".join(lines) + "\n"

    # ── QASM3 import ─────────────────────────────────────────────────────────

    @classmethod
    def from_qasm3(cls, qasm: str) -> "QuantumCircuit":
        """Parse a basic OpenQASM 3.0 string and return a :class:`QuantumCircuit`.

        Supports: qubit declarations, standard gates (H/X/Y/Z/S/T/Sdg/Tdg/
        Rx/Ry/Rz/P/U/CX/CY/CZ/CH/SWAP/iSWAP/CCX/CSWAP/id), measurements,
        and ``//`` comments.  Symbolic angles (e.g. ``pi/2``) are evaluated
        to floats.

        Parameters
        ----------
        qasm : str
            OpenQASM 3.0 source string.

        Returns
        -------
        QuantumCircuit
        """
        n_qubits: int | None = None
        ops: list[_GateOp] = []
        measured: list[bool] = []

        for raw in qasm.splitlines():
            line = raw.split("//")[0].strip().rstrip(";").strip()
            if not line:
                continue
            if line.startswith("OPENQASM") or line.startswith("include"):
                continue

            # qubit[N] name;
            m = re.match(r"qubit\[(\d+)\]\s+\w+$", line)
            if m:
                n_qubits = int(m.group(1))
                measured = [False] * n_qubits
                continue

            # bit[N] name;  — ignore classical registers
            if re.match(r"bit\[", line):
                continue

            if n_qubits is None:
                continue  # ignore gates before qubit declaration

            # Measurement:  c = measure q;   or   c[k] = measure q[k];
            if "measure" in line:
                if re.search(r"measure\s+\w+\s*$", line):
                    # measure all qubits
                    ops.append(
                        _GateOp("MEASURE_ALL", tuple(range(n_qubits)), (), True)
                    )
                    measured = [True] * n_qubits
                else:
                    m2 = re.search(r"measure\s+\w+\[(\d+)\]", line)
                    if m2:
                        q = int(m2.group(1))
                        ops.append(_GateOp("MEASURE", (q,), (), True))
                        measured[q] = True
                continue

            # Gate:  name[(params)]  qubit_args
            gm = re.match(r"([a-zA-Z][a-zA-Z0-9_]*)\s*(\([^)]*\))?\s+(.+)$", line)
            if not gm:
                continue
            raw_name = gm.group(1).lower()
            params_str = (gm.group(2) or "").strip("()")
            qubits_str = gm.group(3)

            internal = _FROM_QASM3.get(raw_name)
            if internal is None:
                continue  # unknown gate — skip gracefully

            # Parse angles
            params_list: list[float] = []
            if params_str:
                for tok in re.split(r",\s*", params_str):
                    if tok.strip():
                        params_list.append(_parse_angle(tok))

            # Parse qubit indices
            qubit_indices = [
                int(x) for x in re.findall(r"\[(\d+)\]", qubits_str)
            ]

            ops.append(
                _GateOp(internal, tuple(qubit_indices), tuple(params_list))
            )

        if n_qubits is None:
            raise ValueError("No qubit declaration found in QASM3 source.")

        qc = cls(n_qubits)
        qc._ops = ops
        qc._measured = measured
        return qc

    # ── ASCII diagram ─────────────────────────────────────────────────────────

    def draw(self) -> str:
        """Return a text-mode ASCII circuit diagram (no matplotlib required).

        Returns
        -------
        str
            Multi-line string where each qubit occupies one row.  Example::

                q[0] : --[H]---[*]--
                q[1] : --------[CX]-

        Qubit 0 is the topmost row.  ``[*]`` denotes a control qubit;
        ``[GATE]`` marks the target.
        """
        gate_ops = [op for op in self._ops if not op.is_measure]
        measure_qubits: set[int] = {
            q
            for op in self._ops
            if op.is_measure
            for q in op.qubits
        }

        # Build column → {qubit: cell_label} mapping
        columns: list[dict[int, str]] = []
        for op in gate_ops:
            col: dict[int, str] = {}
            label = _gate_label(op)
            if len(op.qubits) == 1:
                col[op.qubits[0]] = label
            elif len(op.qubits) == 2:
                col[op.qubits[0]] = "*"      # control
                col[op.qubits[1]] = label    # target
            else:
                # e.g. CCX or CSWAP: first N-1 are controls, last is target
                for idx, q in enumerate(op.qubits[:-1]):
                    col[q] = "*"
                col[op.qubits[-1]] = label
            columns.append(col)

        if measure_qubits:
            col = {q: "M" for q in measure_qubits}
            columns.append(col)

        # Determine per-column cell width (label + 2 brackets + 2 dashes each side)
        col_widths: list[int] = []
        for col in columns:
            w = max((len(v) for v in col.values()), default=1)
            col_widths.append(w)

        # Render rows
        q_label_w = len(f"q[{self._n - 1}]")
        rows: list[str] = []
        for q in range(self._n):
            header = f"q[{q}]".ljust(q_label_w) + " : "
            cells: list[str] = []
            for i, col in enumerate(columns):
                w = col_widths[i]
                total = w + 4           # "[" + label(w) + "]" + 2× "-" padding
                cell_label = col.get(q, "")
                if cell_label:
                    inner = f"[{cell_label}]"
                    # Pad inner to (w+2) then add one "-" each side
                    inner = inner.center(w + 2, "-")
                    cells.append(f"-{inner}-")
                else:
                    cells.append("-" * total)
            rows.append(header + "-" + "".join(cells) + "-")

        return "\n".join(rows)

    # ── Utility ───────────────────────────────────────────────────────────────

    def copy(self) -> "QuantumCircuit":
        """Return a deep copy (independent ops list and measured flags)."""
        new = QuantumCircuit(self._n, self.name)
        new._ops = list(self._ops)
        new._measured = list(self._measured)
        return new

    def __len__(self) -> int:
        """Number of gate operations (measurements excluded)."""
        return sum(1 for op in self._ops if not op.is_measure)

    def __repr__(self) -> str:
        return (
            f"QuantumCircuit(n_qubits={self._n}, gates={len(self)}, "
            f"name={self.name!r})"
        )

    def __str__(self) -> str:
        return self.draw()


# ── Gate dispatcher (module-level, reused by Engine bridge) ──────────────────


def _apply_to_sim(
    sim: StatevectorSimulator,
    name: str,
    qubits: tuple[int, ...],
    params: tuple[float, ...],
) -> None:
    """Apply one gate operation to a :class:`StatevectorSimulator`."""
    if name == "I":
        sim.i(qubits[0])
    elif name == "H":
        sim.h(qubits[0])
    elif name == "X":
        sim.x(qubits[0])
    elif name == "Y":
        sim.y(qubits[0])
    elif name == "Z":
        sim.z(qubits[0])
    elif name == "S":
        sim.s(qubits[0])
    elif name == "T":
        sim.t(qubits[0])
    elif name == "SDG":
        sim.sdg(qubits[0])
    elif name == "TDG":
        sim.tdg(qubits[0])
    elif name == "RX":
        sim.rx(params[0], qubits[0])
    elif name == "RY":
        sim.ry(params[0], qubits[0])
    elif name == "RZ":
        sim.rz(params[0], qubits[0])
    elif name == "P":
        sim.p(params[0], qubits[0])
    elif name == "U3":
        sim.u3(params[0], params[1], params[2], qubits[0])
    elif name in ("CX", "CNOT"):
        sim.cnot(qubits[0], qubits[1])
    elif name == "CY":
        sim.cy(qubits[0], qubits[1])
    elif name == "CZ":
        sim.cz(qubits[0], qubits[1])
    elif name == "CH":
        sim.ch(qubits[0], qubits[1])
    elif name == "SWAP":
        sim.swap(qubits[0], qubits[1])
    elif name == "ISWAP":
        sim.iswap(qubits[0], qubits[1])
    elif name in ("CCX", "TOFFOLI"):
        sim.ccx(qubits[0], qubits[1], qubits[2])
    elif name in ("CSWAP", "FREDKIN"):
        sim.cswap(qubits[0], qubits[1], qubits[2])
    else:
        raise ValueError(
            f"Unknown gate {name!r}. "
            "Supported: H/X/Y/Z/S/T/Sdg/Tdg/I, Rx/Ry/Rz/P/U3, "
            "CX/CY/CZ/CH/SWAP/iSWAP, CCX/CSWAP."
        )
