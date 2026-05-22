"""Circuit adapters for Google Cirq and Amazon Braket (M77).

Converts between QuantX :class:`~quantx_engine.transpiler.GateOp` lists and
the native circuit objects of each framework.

Both directions are supported::

    # QuantX → Cirq
    from quantx_engine.adapters import to_cirq, from_cirq
    cirq_circ = to_cirq([GateOp("h", [0]), GateOp("cx", [0, 1])])
    back      = from_cirq(cirq_circ)

    # QuantX → Braket
    from quantx_engine.adapters import to_braket, from_braket
    bk_circ = to_braket([GateOp("h", [0]), GateOp("cnot", [0, 1])])
    back    = from_braket(bk_circ)

All imports of ``cirq`` and ``braket`` are deferred so that the rest of
QuantX remains importable even if those packages are not installed.
:class:`AdapterError` is raised when a required package is missing or a gate
is not supported.

References
----------
.. [1] Cirq Developers (2023). Cirq: A Python library for writing, manipulating,
       and optimizing quantum circuits. *Zenodo*.
       https://doi.org/10.5281/zenodo.8161252

.. [2] Amazon Web Services (2021). Amazon Braket: Exploring quantum computing.
       https://aws.amazon.com/braket/ (accessed 2024)

.. [3] Qiskit Community (2023). Qiskit: An open-source framework for quantum
       computing. *Zenodo*. https://doi.org/10.5281/zenodo.2573505
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .transpiler import GateOp

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

__all__ = [
    "AdapterError",
    "to_cirq",
    "from_cirq",
    "to_braket",
    "from_braket",
]


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class AdapterError(ValueError):
    """Raised for unsupported gates or missing optional dependencies."""


# ---------------------------------------------------------------------------
# Shared gate name normalisation
# ---------------------------------------------------------------------------

#: Canonical QuantX lower-case gate name → (n_qubits, n_params)
_GATE_SPEC: "dict[str, tuple[int, int]]" = {
    "id":   (1, 0), "h":   (1, 0), "x":   (1, 0), "y":   (1, 0), "z":   (1, 0),
    "s":    (1, 0), "sdg": (1, 0), "t":   (1, 0), "tdg": (1, 0),
    "sx":   (1, 0), "sxdg":(1, 0),
    "rx":   (1, 1), "ry":  (1, 1), "rz":  (1, 1),
    "u1":   (1, 1), "u2":  (1, 2), "u3":  (1, 3),
    "cx":   (2, 0), "cnot":(2, 0), "cz":  (2, 0), "cy":  (2, 0),
    "swap": (2, 0), "iswap":(2, 0),
    "ccx":  (3, 0), "toffoli":(3, 0), "cswap":(3, 0),
}

def _canonical(name: str) -> str:
    """Lower-case and normalise gate name aliases."""
    n = name.lower()
    if n == "toffoli":
        return "ccx"
    return n


# ===========================================================================
# Cirq adapter
# ===========================================================================

# QuantX canonical name → callable that returns a cirq Gate (given params list)
# Cache keyed by the cirq module's id so a reload is handled correctly.
_CIRQ_GATE_MAP_CACHE: "dict[int, dict[str, Any]]" = {}


def _get_cirq_gate_map() -> "dict[str, Any]":
    """Return the cached name→gate-factory map, building it on first use."""
    import cirq  # noqa: PLC0415
    key = id(cirq)
    if key not in _CIRQ_GATE_MAP_CACHE:
        _CIRQ_GATE_MAP_CACHE[key] = {
            "id":    lambda p: cirq.I,
            "h":     lambda p: cirq.H,
            "x":     lambda p: cirq.X,
            "y":     lambda p: cirq.Y,
            "z":     lambda p: cirq.Z,
            "s":     lambda p: cirq.S,
            "sdg":   lambda p: cirq.S**-1,
            "t":     lambda p: cirq.T,
            "tdg":   lambda p: cirq.T**-1,
            "sx":    lambda p: cirq.X**0.5,
            "sxdg":  lambda p: cirq.X**-0.5,
            "rx":    lambda p: cirq.rx(p[0]),
            "ry":    lambda p: cirq.ry(p[0]),
            "rz":    lambda p: cirq.rz(p[0]),
            "cx":    lambda p: cirq.CNOT,
            "cnot":  lambda p: cirq.CNOT,
            "cz":    lambda p: cirq.CZ,
            "cy":    lambda p: cirq.Y.controlled(1),
            "swap":  lambda p: cirq.SWAP,
            "iswap": lambda p: cirq.ISWAP,
            "ccx":   lambda p: cirq.CCX,
            "toffoli": lambda p: cirq.CCX,
            "cswap": lambda p: cirq.CSWAP,
        }
    return _CIRQ_GATE_MAP_CACHE[key]


def to_cirq(
    circuit: "list[GateOp]",
    n_qubits: "int | None" = None,
) -> "Any":
    """Convert a QuantX gate list to a :class:`cirq.Circuit`.

    Parameters
    ----------
    circuit:
        List of :class:`~quantx_engine.transpiler.GateOp` (logical or
        physical qubit indices).
    n_qubits:
        Total number of qubits.  Inferred from the circuit if ``None``.

    Returns
    -------
    cirq.Circuit

    Raises
    ------
    AdapterError
        If ``cirq`` is not installed or a gate is not supported.
    """
    try:
        import cirq  # noqa: PLC0415
    except ImportError as exc:
        raise AdapterError(
            "cirq is not installed.  Run: pip install cirq"
        ) from exc

    if n_qubits is None:
        if not circuit:
            n_qubits = 0
        else:
            n_qubits = max(q for g in circuit for q in g.qubits) + 1

    qubits = cirq.LineQubit.range(n_qubits)
    gate_map = _get_cirq_gate_map()

    ops = []
    for gate in circuit:
        name = _canonical(gate.name)
        if name not in gate_map:
            raise AdapterError(
                f"Gate '{gate.name}' is not supported by the Cirq adapter."
            )
        cirq_gate = gate_map[name](gate.params)
        qs = [qubits[q] for q in gate.qubits]
        ops.append(cirq_gate(*qs))

    return cirq.Circuit(ops)


# Cirq gate type → QuantX canonical name (for from_cirq)
_CIRQ_TYPE_TO_NAME: "dict[str, str]" = {
    "IdentityGate": "id",
    "HPowGate":     "h",
    "XPowGate":     "x",
    "YPowGate":     "y",
    "ZPowGate":     "z",
    "CNotPowGate":  "cx",
    "CZPowGate":    "cz",
    "SwapPowGate":  "swap",
    "ISwapPowGate": "iswap",
    "CCXPowGate":   "ccx",
    "CSwapGate":    "cswap",
    "Rx":           "rx",
    "Ry":           "ry",
    "Rz":           "rz",
}


def _qubit_index(q: "Any") -> int:
    """Extract an integer qubit index from a Cirq qubit object.

    Supports :class:`cirq.LineQubit` (attribute ``x``) and
    :class:`cirq.GridQubit` when it falls on row 0 (attribute ``col``).
    Raises :class:`AdapterError` for types that cannot be mapped to a
    single integer index.
    """
    if hasattr(q, "x"):           # LineQubit
        return int(q.x)
    if hasattr(q, "col") and hasattr(q, "row"):  # GridQubit
        if q.row != 0:
            raise AdapterError(
                f"Cannot convert GridQubit({q.row},{q.col}) to a linear "
                "qubit index: only row-0 GridQubits are supported."
            )
        return int(q.col)
    raise AdapterError(
        f"Unsupported Cirq qubit type {type(q).__name__!r}; "
        "only LineQubit (and row-0 GridQubit) are supported."
    )


def _cirq_op_to_gateop(op: "Any") -> "GateOp | None":
    """Convert a single Cirq Operation to GateOp; return None if unknown."""
    gate = op.gate
    if gate is None:
        return None
    type_name = type(gate).__name__
    qubits = [_qubit_index(q) for q in op.qubits]

    # Parametric rotation gates
    if type_name == "Rx":
        return GateOp("rx", qubits, [float(gate.rads)])
    if type_name == "Ry":
        return GateOp("ry", qubits, [float(gate.rads)])
    if type_name == "Rz":
        return GateOp("rz", qubits, [float(gate.rads)])

    # Power gates need special treatment
    if type_name in ("XPowGate", "YPowGate", "ZPowGate", "HPowGate"):
        exp = float(gate.exponent)
        base = type_name[0].lower()  # "x", "y", "z", or "h"
        if math.isclose(exp, 1.0):
            return GateOp(base, qubits)
        if math.isclose(exp, -1.0) and base == "z":
            # ZPowGate(exponent=-1) is equivalent to Z (global-phase difference).
            return GateOp(base, qubits)
        if math.isclose(exp, 0.5):
            if base == "z":
                return GateOp("s", qubits)
            if base == "x":
                return GateOp("sx", qubits)
            if base == "y":
                # YPowGate(0.5) is √Y = Ry(π/2); "sx" is wrong axis.
                return GateOp("ry", qubits, [math.pi / 2])
            return GateOp(base, qubits)
        if math.isclose(exp, 0.25):
            return GateOp("t" if base == "z" else base, qubits)
        if math.isclose(exp, -0.25):
            return GateOp("tdg" if base == "z" else base, qubits)
        if math.isclose(exp, -0.5):
            return GateOp("sdg" if base == "z" else "sxdg", qubits)
        # General rotation: rz(π*exp) etc.
        if base == "z":
            return GateOp("rz", qubits, [math.pi * exp])
        if base == "x":
            return GateOp("rx", qubits, [math.pi * exp])
        if base == "y":
            return GateOp("ry", qubits, [math.pi * exp])
        if base == "h":
            return GateOp("h", qubits)  # best effort

    if type_name in ("CNotPowGate",):
        exp = float(gate.exponent)
        if math.isclose(exp, 1.0):
            return GateOp("cx", qubits)
    if type_name in ("CZPowGate",):
        exp = float(gate.exponent)
        if math.isclose(exp, 1.0):
            return GateOp("cz", qubits)
    if type_name in ("SwapPowGate",):
        exp = float(gate.exponent)
        if math.isclose(exp, 1.0):
            return GateOp("swap", qubits)
    if type_name in ("ISwapPowGate",):
        exp = float(gate.exponent)
        if math.isclose(exp, 1.0):
            return GateOp("iswap", qubits)
    if type_name in ("CCXPowGate",):
        return GateOp("ccx", qubits)
    if type_name in ("CSwapGate",):
        return GateOp("cswap", qubits)
    if type_name == "IdentityGate":
        return GateOp("id", qubits)

    # ControlledGate: e.g. cirq.Y.controlled(1) emitted by to_cirq for "cy".
    # Guard against multi-controlled or non-default-control variants that
    # would have a different arity or semantics from a plain CY gate.
    if type_name == "ControlledGate":
        sub_gate = getattr(gate, "sub_gate", None)
        num_controls = getattr(gate, "num_controls", lambda: None)()
        # Accept only single-control (num_controls == 1) with default all-1
        # control values (cirq default: ((1,),) or similar).
        ctrl_vals = getattr(gate, "control_values", None)
        default_ctrl = (ctrl_vals is None) or all(
            set(cv) == {1} for cv in ctrl_vals
        )
        if (
            sub_gate is not None
            and num_controls == 1
            and default_ctrl
        ):
            sub_type = type(sub_gate).__name__
            sub_exp = float(getattr(sub_gate, "exponent", 1))
            if sub_type == "YPowGate" and math.isclose(sub_exp, 1.0):
                return GateOp("cy", qubits)

    # Unknown gate: return None to let caller handle
    return None


def from_cirq(cirq_circuit: "Any") -> "list[GateOp]":
    """Convert a :class:`cirq.Circuit` to a QuantX gate list.

    Parameters
    ----------
    cirq_circuit:
        A ``cirq.Circuit`` instance.

    Returns
    -------
    list[GateOp]

    Raises
    ------
    AdapterError
        If ``cirq`` is not installed or an operation cannot be converted.
    """
    try:
        import cirq  # noqa: F401, PLC0415
    except ImportError as exc:
        raise AdapterError(
            "cirq is not installed.  Run: pip install cirq"
        ) from exc

    result: "list[GateOp]" = []
    for op in cirq_circuit.all_operations():
        gop = _cirq_op_to_gateop(op)
        if gop is None:
            raise AdapterError(
                f"Cannot convert Cirq operation {op!r} to a QuantX GateOp."
            )
        result.append(gop)
    return result


# ===========================================================================
# Braket adapter
# ===========================================================================

# QuantX canonical name → Braket Circuit method name
_QUANTX_TO_BRAKET: "dict[str, str]" = {
    "id":    "i",
    "h":     "h",
    "x":     "x",
    "y":     "y",
    "z":     "z",
    "s":     "s",
    "sdg":   "si",     # S-dagger = S†
    "t":     "t",
    "tdg":   "ti",     # T-dagger
    "sx":    "v",      # √X
    "sxdg":  "vi",     # (√X)†
    "rx":    "rx",
    "ry":    "ry",
    "rz":    "rz",
    "cx":    "cnot",
    "cnot":  "cnot",
    "cz":    "cz",
    "cy":    "cy",
    "swap":  "swap",
    "iswap": "iswap",
    "ccx":   "ccnot",
    "toffoli": "ccnot",
    "cswap": "cswap",
}

# Braket gate class name → QuantX canonical name
_BRAKET_TO_QUANTX: "dict[str, str]" = {
    "I":     "id",
    "H":     "h",
    "X":     "x",
    "Y":     "y",
    "Z":     "z",
    "S":     "s",
    "Si":    "sdg",
    "T":     "t",
    "Ti":    "tdg",
    "V":     "sx",
    "Vi":    "sxdg",
    "Rx":    "rx",
    "Ry":    "ry",
    "Rz":    "rz",
    "CNot":  "cx",
    "CZ":    "cz",
    "CY":    "cy",
    "Swap":  "swap",
    "ISwap": "iswap",
    "CCNot": "ccx",
    "CSwap": "cswap",
    "PhaseShift": "rz",  # approximate: PhaseShift(angle) ≈ Rz(angle) up to global phase
}

# Braket gates that carry a single angle parameter
_BRAKET_PARAMETRIC: "frozenset[str]" = frozenset({
    "Rx", "Ry", "Rz", "PhaseShift",
})


def to_braket(
    circuit: "list[GateOp]",
    n_qubits: "int | None" = None,
) -> "Any":
    """Convert a QuantX gate list to an Amazon Braket :class:`Circuit`.

    Parameters
    ----------
    circuit:
        List of :class:`~quantx_engine.transpiler.GateOp`.
    n_qubits:
        Ignored (Braket infers qubit count automatically).

    Returns
    -------
    braket.circuits.Circuit

    Raises
    ------
    AdapterError
        If ``amazon-braket-sdk`` is not installed or a gate is unsupported.
    """
    try:
        from braket.circuits import Circuit  # noqa: PLC0415
    except ImportError as exc:
        raise AdapterError(
            "amazon-braket-sdk is not installed.  "
            "Run: pip install amazon-braket-sdk"
        ) from exc

    bk_circ = Circuit()
    for gate in circuit:
        name = _canonical(gate.name)
        bk_name = _QUANTX_TO_BRAKET.get(name)
        if bk_name is None:
            raise AdapterError(
                f"Gate '{gate.name}' is not supported by the Braket adapter."
            )
        method = getattr(bk_circ, bk_name, None)
        if method is None:
            raise AdapterError(
                f"Braket Circuit has no method '{bk_name}' (gate '{gate.name}')."
            )
        if gate.params:
            method(*gate.qubits, *gate.params)
        else:
            method(*gate.qubits)
    return bk_circ


def from_braket(braket_circuit: "Any") -> "list[GateOp]":
    """Convert an Amazon Braket :class:`Circuit` to a QuantX gate list.

    Parameters
    ----------
    braket_circuit:
        A ``braket.circuits.Circuit`` instance.

    Returns
    -------
    list[GateOp]

    Raises
    ------
    AdapterError
        If ``amazon-braket-sdk`` is not installed or an instruction cannot
        be converted.
    """
    try:
        from braket.circuits import Circuit as _BC  # noqa: F401, PLC0415
    except ImportError as exc:
        raise AdapterError(
            "amazon-braket-sdk is not installed.  "
            "Run: pip install amazon-braket-sdk"
        ) from exc

    result: "list[GateOp]" = []
    for inst in braket_circuit.instructions:
        gate = inst.operator
        gate_class = type(gate).__name__
        quantx_name = _BRAKET_TO_QUANTX.get(gate_class)
        if quantx_name is None:
            raise AdapterError(
                f"Cannot convert Braket gate '{gate_class}' to a QuantX GateOp."
            )
        # Qubit indices: Braket uses int-valued qubit objects
        qubits = [int(q) for q in inst.target]
        # Parameters
        params: "list[float]" = []
        if gate_class in _BRAKET_PARAMETRIC:
            params = [float(gate.angle)]
        result.append(GateOp(quantx_name, qubits, params))
    return result
