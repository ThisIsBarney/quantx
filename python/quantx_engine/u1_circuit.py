"""Charge-conserving qubit circuits on the block-sparse engine (M16).

A circuit whose gates all conserve the number of qubits in ``|1>`` (a particle-number-
conserving circuit, such as a fermionic simulation after a Jordan-Wigner map) runs on the
engine's block-sparse MPS with that number as an exact charge. It is conserved at any
truncation, where a dense MPS loses it at the scale of its truncation, and the MPS is
block-diagonal in it::

    from quantx_engine import u1_circuit

    circ = u1_circuit.from_qasm(open("circuit.qasm").read())   # fuse into U(1) gates
    r = u1_circuit.run(circ, max_bond=800)
    r.measurements[-1].z                                        # <Z_i> after the last gate

:func:`from_qasm` reads OpenQASM 2 (one register, the qelib1 one- and two-qubit gates) and
fuses its gates, greedily, into clusters of at most two adjacent qubits whose product
conserves the charge. A transpiled circuit breaks the symmetry gate by gate (an H or a CX
alone does not conserve it) but keeps it cluster by cluster, which is what the fusion
recovers. A cluster that cannot close within two adjacent qubits is an error, as is a
circuit that ends with one open. X gates that prepare the initial state become its
bitstring (see :func:`from_qasm`).

The gates of :class:`U1Circuit` are 2x2 phases or 4x4 two-qubit gates, row-major in the basis
``|s_q s_q+1> = |00>, |01>, |10>, |11>`` with qubit q the high bit; ``|0>`` has Z = +1.
"""
from __future__ import annotations

import ast
import bisect
import json
import math
import operator
import re
from ctypes import c_char_p
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = ["U1CircuitError", "Gate", "U1Circuit", "U1Measurement", "U1Result",
           "from_qasm", "from_gates", "measure_points", "run"]

# A cluster closes once its charge-changing entries are this small. The engine refuses a
# gate whose charge-changing entries exceed 1e-10, or that is not unitary to 1e-10.
_TOL = 1e-10


class U1CircuitError(RuntimeError):
    """A circuit that cannot run as a charge-conserving one, or a failure in the engine."""


@dataclass(frozen=True)
class Gate:
    """One fused gate: a 2x2 phase on qubit ``q``, or a 4x4 gate on qubits (q, q+1)."""
    q: int
    u: np.ndarray

    @property
    def arity(self) -> int:
        return 1 if self.u.shape == (2, 2) else 2


@dataclass(frozen=True)
class U1Circuit:
    """A charge-conserving circuit: the initial bitstring and its fused gates.

    Fused gate k holds source gates (as read) from ``source_start[k]`` to
    ``source_end[k] - 1``, those on its qubits; ``initial_end`` is one past the last X gate
    taken into the bitstring. :func:`measure_points` uses them to turn a position in the
    source into one here.
    """
    n_qubits: int
    initial: tuple
    gates: tuple
    source_start: tuple = ()
    source_end: tuple = ()
    source_gates: int = 0
    initial_end: int = 0

    @property
    def ones(self) -> int:
        """The conserved number of qubits in ``|1>``."""
        return int(sum(self.initial))


@dataclass(frozen=True)
class U1Measurement:
    after: int                 # measured after this many fused gates
    z: np.ndarray              # <Z_i>, normalized by the state's norm
    norm2: float               # squared norm: the weight the truncations left
    max_bond: int              # the largest bond of the MPS at this point


@dataclass(frozen=True)
class U1Result:
    measurements: list
    ones: int                  # the conserved number of |1>s, read off the MPS
    max_bond: int              # largest bond any gate produced
    truncation_sum: float      # discarded weight summed over the gates
    truncation_max: float      # the largest single discard
    wall_time_s: float = 0.0


# ---- the qelib1 gates -------------------------------------------------------------------
_S2 = 1 / math.sqrt(2)
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)
_H = np.array([[_S2, _S2], [_S2, -_S2]], dtype=complex)
_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])


def _rx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def _ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def _phase(t):
    return np.diag([1, np.exp(1j * t)])


def _u3(t, p, lam):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * p) * s, np.exp(1j * (p + lam)) * c]])


def _ctrl(u):
    """``u`` on the second qubit, controlled by the first (the high bit)."""
    out = np.eye(4, dtype=complex)
    out[2:, 2:] = u
    return out


# name -> (number of parameters, number of qubits, matrix from the parameters)
_GATES = {
    "id": (0, 1, lambda: _I2), "x": (0, 1, lambda: _X), "y": (0, 1, lambda: _Y),
    "z": (0, 1, lambda: _Z), "h": (0, 1, lambda: _H),
    "s": (0, 1, lambda: np.diag([1, 1j])), "sdg": (0, 1, lambda: np.diag([1, -1j])),
    "t": (0, 1, lambda: _phase(math.pi / 4)), "tdg": (0, 1, lambda: _phase(-math.pi / 4)),
    "sx": (0, 1, lambda: _SX), "sxdg": (0, 1, lambda: _SX.conj().T),
    "rx": (1, 1, _rx), "ry": (1, 1, _ry), "rz": (1, 1, _rz),
    "p": (1, 1, _phase), "u1": (1, 1, _phase), "u0": (1, 1, lambda g: _I2),
    "u2": (2, 1, lambda p, lam: _u3(math.pi / 2, p, lam)), "u3": (3, 1, _u3), "u": (3, 1, _u3),
    "cx": (0, 2, lambda: _ctrl(_X)), "cy": (0, 2, lambda: _ctrl(_Y)),
    "cz": (0, 2, lambda: _ctrl(_Z)), "ch": (0, 2, lambda: _ctrl(_H)),
    "csx": (0, 2, lambda: _ctrl(_SX)),
    "swap": (0, 2, lambda: np.eye(4, dtype=complex)[[0, 2, 1, 3]]),
    "crx": (1, 2, lambda t: _ctrl(_rx(t))), "cry": (1, 2, lambda t: _ctrl(_ry(t))),
    "crz": (1, 2, lambda t: _ctrl(_rz(t))),
    "cp": (1, 2, lambda t: _ctrl(_phase(t))), "cu1": (1, 2, lambda t: _ctrl(_phase(t))),
    "cu3": (3, 2, lambda t, p, lam: _ctrl(_u3(t, p, lam))),
    "cu": (4, 2, lambda t, p, lam, g: _ctrl(np.exp(1j * g) * _u3(t, p, lam))),
    "rzz": (1, 2, lambda t: np.diag(np.exp(-0.5j * t * np.array([1, -1, -1, 1])))),
    "rxx": (1, 2, lambda t: math.cos(t / 2) * np.eye(4) - 1j * math.sin(t / 2) * np.kron(_X, _X)),
}

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}


def _param(text: str) -> float:
    """A gate parameter: numbers and ``pi`` under + - * / and parentheses."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return math.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError
    try:
        return ev(ast.parse(text.strip(), mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError):
        raise U1CircuitError(f"unsupported gate parameter {text!r}") from None


def _parse_qasm(text: str):
    n, reg, gates = None, None, []
    for k, raw in enumerate(text.splitlines(), 1):
        line = raw.split("//", 1)[0].strip()
        for stmt in filter(None, (s.strip() for s in line.split(";"))):
            if stmt.startswith(("OPENQASM", "include", "creg", "barrier")):
                continue
            m = re.fullmatch(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]", stmt)
            if m:
                if n is not None:
                    raise U1CircuitError("one quantum register only")
                reg, n = m.group(1), int(m.group(2))
                continue
            word = re.match(r"\w*", stmt).group(0)
            if word in ("measure", "reset", "if", "gate", "opaque"):
                raise U1CircuitError(f"line {k}: {word} is not supported: the circuit must "
                                     "be unitary and use the qelib1 gates")
            m = re.fullmatch(r"(\w+)\s*(?:\(([^)]*)\))?\s+(.+)", stmt)
            if not m:
                raise U1CircuitError(f"line {k}: cannot read {stmt!r}")
            name, par, args = m.group(1).lower(), m.group(2), m.group(3)
            if name not in _GATES:
                raise U1CircuitError(f"line {k}: unsupported gate {name!r}")
            n_par, arity, fn = _GATES[name]
            pars = [] if par is None or not par.strip() else [_param(p) for p in par.split(",")]
            if len(pars) != n_par:
                raise U1CircuitError(f"line {k}: {name} takes {n_par} parameter(s)")
            refs = re.findall(r"(\w+)\s*\[\s*(\d+)\s*\]", args)
            if len(refs) != arity or len(args.split(",")) != arity:
                raise U1CircuitError(f"line {k}: {name} acts on {arity} single qubit(s)")
            if reg is None:
                raise U1CircuitError(f"line {k}: a gate before the qreg")
            if any(r != reg for r, _ in refs):
                raise U1CircuitError(f"line {k}: {name} acts on a register other than the "
                                     f"qreg {reg!r}")
            qs = [int(i) for _, i in refs]
            if arity == 2 and qs[0] == qs[1]:
                raise U1CircuitError(f"line {k}: {name} on the same qubit twice")
            gates.append((np.asarray(fn(*pars), dtype=complex), qs))
    if n is None:
        raise U1CircuitError("no qreg")
    for _, qs in gates:
        if max(qs) >= n:
            raise U1CircuitError(f"qubit index {max(qs)} past the register's {n}")
    return n, gates


def _embed(mat, gate_qs, cluster_qs):
    """``mat`` on gate_qs (in that order) as an operator on cluster_qs (first = high bit)."""
    if list(gate_qs) == list(cluster_qs):
        return mat
    n, k = len(cluster_qs), len(gate_qs)
    pos = [cluster_qs.index(q) for q in gate_qs]
    out = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for col in range(2 ** n):
        bits = [(col >> (n - 1 - i)) & 1 for i in range(n)]
        sub = 0
        for p in pos:
            sub = (sub << 1) | bits[p]
        for rs in range(2 ** k):
            amp = mat[rs, sub]
            if amp == 0:
                continue
            nb = list(bits)
            for j, p in enumerate(pos):
                nb[p] = (rs >> (k - 1 - j)) & 1
            row = 0
            for b in nb:
                row = (row << 1) | b
            out[row, col] += amp
    return out


def _leak_mask(dim):
    ones = np.array([bin(i).count("1") for i in range(dim)])
    return ones[:, None] != ones[None, :]


def _charge_leak(u):
    """The largest entry of ``u`` that changes the number of qubits in ``|1>``."""
    mask = _leak_mask(u.shape[0])
    return float(np.abs(u[mask]).max()) if mask.any() else 0.0


def _close(u, qs):
    """The cluster as a Gate on its lowest qubit, its charge-changing entries made zero."""
    u = u.copy()
    u[_leak_mask(u.shape[0])] = 0
    if len(qs) == 1:
        return Gate(q=qs[0], u=u)
    lo, hi = sorted(qs)
    return Gate(q=lo, u=_embed(u, list(qs), [lo, hi]))


def from_qasm(text: str) -> U1Circuit:
    """Read OpenQASM 2 and fuse its gates into charge-conserving one- and two-qubit gates.

    The X gates that open the circuit, before any other gate, become the initial bitstring.
    If the circuit then cannot be fused -- preparation X gates that a transpiler emitted
    among other gates -- every X gate on a qubit that nothing has acted on yet is taken
    instead. (Not the first choice: it also takes the X of an X U X cluster that opens on
    such a qubit, and leaves the closing X without a partner. Neither choice can tell such a
    cluster from preparation when it directly follows the opening X gates.)
    """
    n, raw = _parse_qasm(text)
    lead = next((i for i, (mat, qs) in enumerate(raw)
                 if not (len(qs) == 1 and _is_flip(mat))), len(raw))
    try:
        return _fuse(n, raw, lead)
    except U1CircuitError as first:
        try:
            return _fuse(n, raw, len(raw))
        except U1CircuitError:
            hint = (f" (the {lead} X gates that open the circuit were taken as its initial "
                    "bitstring)") if lead else ""
            raise U1CircuitError(f"{first}{hint}") from None


def _is_flip(mat):
    """X up to phases: on a basis state, a bit flip and a global phase."""
    return mat.shape == (2, 2) and np.allclose(np.abs(mat), _X)


def _fuse(n, raw, absorb_before):
    """Fuse ``raw``; X gates before source index ``absorb_before`` on untouched qubits
    go into the initial bitstring."""
    initial = [0] * n
    initial_end = 0
    touched = [False] * n
    open_of = {}                 # qubit -> id of the open cluster that holds it
    clusters = {}                # id -> {"qs": [...], "u": matrix, "start": source index}
    out, starts, ends = [], [], []
    nid = 0
    for idx, (mat, qs) in enumerate(raw):
        if idx < absorb_before and len(qs) == 1 and not touched[qs[0]] and _is_flip(mat):
            initial[qs[0]] ^= 1
            initial_end = idx + 1
            continue
        for q in qs:
            touched[q] = True
        cids = sorted({open_of[q] for q in qs if q in open_of})
        if not cids:
            cid, nid = nid, nid + 1
            clusters[cid] = {"qs": list(qs), "u": np.eye(2 ** len(qs), dtype=complex),
                             "start": idx}
        else:
            cid = cids[0]
            a = clusters[cid]
            for other in cids[1:]:   # clusters on disjoint qubits commute: a tensor product
                b = clusters.pop(other)
                qs_new = a["qs"] + b["qs"]
                a["u"] = _embed(b["u"], b["qs"], qs_new) @ _embed(a["u"], a["qs"], qs_new)
                a["qs"], a["start"] = qs_new, min(a["start"], b["start"])
            for q in qs:
                if q not in a["qs"]:
                    a["u"] = _embed(a["u"], a["qs"], a["qs"] + [q])
                    a["qs"] = a["qs"] + [q]
        c = clusters[cid]
        if len(c["qs"]) > 2 or (len(c["qs"]) == 2 and abs(c["qs"][0] - c["qs"][1]) != 1):
            raise U1CircuitError(
                f"gate {idx}: the charge-conserving cluster it belongs to spans qubits "
                f"{sorted(c['qs'])}, not one qubit or two adjacent ones")
        c["u"] = _embed(mat, qs, c["qs"]) @ c["u"]
        for q in c["qs"]:
            open_of[q] = cid
        if _charge_leak(c["u"]) <= _TOL:
            out.append(_close(c["u"], c["qs"]))
            starts.append(c["start"])
            ends.append(idx + 1)
            for q in c["qs"]:
                del open_of[q]
            del clusters[cid]
    if clusters:
        qs = sorted(q for c in clusters.values() for q in c["qs"])
        raise U1CircuitError(f"the circuit ends with gates on qubits {qs} that do not "
                             "conserve the number of |1>s")
    return U1Circuit(n_qubits=n, initial=tuple(initial), gates=tuple(out),
                     source_start=tuple(starts), source_end=tuple(ends),
                     source_gates=len(raw), initial_end=initial_end)


def from_gates(n_qubits: int, gates: Sequence,
               initial: Optional[Sequence[int]] = None) -> U1Circuit:
    """A circuit from (q, matrix) pairs that already conserve the number of qubits in ``|1>``.

    A 2x2 matrix is a phase on qubit q, a 4x4 one a gate on (q, q+1) in the basis
    ``|00>, |01>, |10>, |11>`` with qubit q the high bit.
    """
    bits = tuple(int(b) for b in (initial if initial is not None else [0] * n_qubits))
    if len(bits) != n_qubits or any(b not in (0, 1) for b in bits):
        raise U1CircuitError("initial must give one bit, 0 or 1, per qubit")
    out = []
    for k, (q, u) in enumerate(gates):
        u = np.asarray(u, dtype=complex)
        if u.shape not in ((2, 2), (4, 4)):
            raise U1CircuitError(f"gate {k}: a 2x2 or 4x4 matrix")
        if _charge_leak(u) > _TOL:
            raise U1CircuitError(f"gate {k}: does not conserve the number of |1>s")
        out.append(Gate(q=operator.index(q), u=u))
    return U1Circuit(n_qubits=n_qubits, initial=bits, gates=tuple(out),
                     source_start=tuple(range(len(out))),
                     source_end=tuple(range(1, len(out) + 1)), source_gates=len(out))


def measure_points(circuit: U1Circuit, source_counts: Sequence[int]) -> list:
    """The fused-gate counts equal to having applied the first ``source_counts`` source gates.

    A position that splits a fused gate has no equivalent, and is an error; so is one
    before the last X gate taken into the initial bitstring.
    """
    starts, ends = circuit.source_start, circuit.source_end
    first_after = [math.inf] * (len(starts) + 1)   # earliest source gate of fused k, k+1, ...
    for k in range(len(starts) - 1, -1, -1):
        first_after[k] = min(starts[k], first_after[k + 1])
    out = []
    for c in source_counts:
        c = operator.index(c)
        if not 0 <= c <= circuit.source_gates:
            raise U1CircuitError(f"source position {c} outside [0, {circuit.source_gates}]")
        if c < circuit.initial_end:
            raise U1CircuitError(f"source position {c} comes before the X gates taken into "
                                 f"the initial bitstring end, at {circuit.initial_end}")
        k = bisect.bisect_right(ends, c)
        if first_after[k] < c:
            raise U1CircuitError(f"source position {c} splits a fused gate: gates before "
                                 "and after it were fused together")
        out.append(k)
    return out


def _lib(engine):
    from quantx_engine.dmrg import _resolve_engine
    lib = getattr(_resolve_engine(engine, "u1_circuit", U1CircuitError), "_lib", None)
    if lib is None or not hasattr(lib, "qx_u1_circuit"):
        raise U1CircuitError("the loaded native library does not export qx_u1_circuit")
    lib.qx_u1_circuit.argtypes = [c_char_p, c_char_p]
    lib.qx_u1_circuit.restype = c_char_p
    return lib


def run(circuit: U1Circuit, *, max_bond: int = 256, cutoff: float = 1e-10,
        measure_after: Optional[Sequence[int]] = None, engine=None) -> U1Result:
    """Run ``circuit`` on the block-sparse MPS and measure <Z_i>.

    Parameters
    ----------
    circuit : U1Circuit
    max_bond : int
        The largest bond dimension any gate may leave.
    cutoff : float
        Schmidt values below cutoff times the norm of the two-site block are dropped.
    measure_after : sequence of int, optional
        Fused-gate counts to measure after (default: the end), each measured once and in
        increasing order; :func:`measure_points` gives them for positions in the source
        circuit. The whole circuit runs either way.

    Raises
    ------
    U1CircuitError
        No native library, a gate the engine refuses, or a failure in the engine.
    """
    try:
        if isinstance(max_bond, bool):
            raise TypeError
        max_bond = operator.index(max_bond)
        when = ([len(circuit.gates)] if measure_after is None
                else [operator.index(k) for k in measure_after])
    except TypeError:
        raise U1CircuitError("max_bond and measure_after must be integers") from None
    prob = {"n_qubits": circuit.n_qubits, "initial": list(circuit.initial),
            "gates": [{"q": g.q, "u": [[float(z.real), float(z.imag)] for z in g.u.reshape(-1)]}
                      for g in circuit.gates],
            "measure_after": when}
    opts = {"max_bond": max_bond, "cutoff": float(cutoff)}
    out = json.loads(_lib(engine).qx_u1_circuit(json.dumps(prob).encode(),
                                                json.dumps(opts).encode()))
    if not out.get("ok", False):
        raise U1CircuitError(out.get("error", "unknown error from the native engine"))
    ms = [U1Measurement(after=int(m["after"]), z=np.asarray(m["z"], dtype=float),
                        norm2=float(m["norm2"]), max_bond=int(m["max_bond"]))
          for m in out["measurements"]]
    return U1Result(measurements=ms, ones=int(out["ones"]), max_bond=int(out["max_bond"]),
                    truncation_sum=float(out["truncation_sum"]),
                    truncation_max=float(out["truncation_max"]),
                    wall_time_s=float(out.get("wall_time_s", 0.0)))
