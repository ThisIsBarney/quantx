"""Hardware-aware circuit transpilation (M76).

Three main components:

**CouplingMap** — hardware qubit connectivity (undirected graph).  Factory
methods for common topologies: :meth:`linear`, :meth:`ring`, :meth:`grid`,
:meth:`all_to_all`.  Provides BFS shortest-path for routing.

**HardwareTranspiler** — compiles a circuit (list of :class:`GateOp`) to a
target device:

1. **Gate decomposition** — rewrites non-native gates into the device's
   native gate set ({CX+Rz} or {CZ+Rz}).
2. **SWAP insertion** — moves logical qubits to physically-adjacent
   positions before each non-adjacent two-qubit gate (greedy path-following
   algorithm).
3. **Peephole optimisation** (``optimization_level ≥ 2``) — cancels
   adjacent inverse-gate pairs and redundant SWAP·SWAP sequences.

**TranspileResult** — reports the transpiled circuit, logical→physical
qubit map, number of inserted SWAPs, and gate/depth metrics.

Quick start::

    from quantx_engine.transpiler import (
        CouplingMap, GateOp, HardwareTranspiler,
    )

    dev = CouplingMap.linear(3)            # 0-1-2
    circ = [GateOp("cx", [0, 2])]          # non-adjacent CX: needs SWAP
    transpiler = HardwareTranspiler(dev, native_gates="cx_rz")
    result = transpiler.transpile(circ)
    print(result.n_swaps)                  # 1
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TranspilerError",
    "GateOp",
    "CouplingMap",
    "TranspileResult",
    "HardwareTranspiler",
]


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class TranspilerError(ValueError):
    """Raised for invalid transpilation parameters or unsupported gates."""


# ---------------------------------------------------------------------------
# GateOp
# ---------------------------------------------------------------------------

@dataclass
class GateOp:
    """A single gate operation.

    Attributes
    ----------
    name:
        Gate name (case-insensitive): ``"h"``, ``"x"``, ``"cx"``, ``"cz"``,
        ``"swap"``, ``"rz"``, ``"t"``, ``"tdg"``, ``"ccx"``, …
    qubits:
        Qubit indices this gate acts on (logical or physical depending on
        context).
    params:
        Gate parameters (e.g. rotation angle for ``"rz"``).
    """
    name: str
    qubits: "list[int]"
    params: "list[float]" = field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GateOp):
            return NotImplemented
        return (
            self.name.lower() == other.name.lower()
            and self.qubits == other.qubits
            and self.params == other.params
        )

    def __hash__(self) -> int:
        return hash((self.name.lower(), tuple(self.qubits), tuple(self.params)))

    def __repr__(self) -> str:
        p = f", params={self.params}" if self.params else ""
        return f"GateOp({self.name!r}, {self.qubits}{p})"


# ---------------------------------------------------------------------------
# CouplingMap
# ---------------------------------------------------------------------------

@dataclass
class CouplingMap:
    """Hardware qubit connectivity — an undirected graph.

    Parameters
    ----------
    n_qubits:
        Total number of physical qubits.
    edges:
        List of (q₁, q₂) pairs representing physical connections.
        Edges are undirected: (0,1) implies both 0→1 and 1→0.

    Examples
    --------
    >>> cm = CouplingMap.linear(4)   # 0-1-2-3
    >>> cm.is_connected(0, 1)
    True
    >>> cm.distance(0, 3)
    3
    """
    n_qubits: int
    edges: "list[tuple[int, int]]"

    def __post_init__(self) -> None:
        self._adj_set: "set[tuple[int, int]]" = set()
        self._adj_list: "dict[int, list[int]]" = {
            i: [] for i in range(self.n_qubits)
        }
        for a, b in self.edges:
            self._adj_set.add((a, b))
            self._adj_set.add((b, a))
            if b not in self._adj_list[a]:
                self._adj_list[a].append(b)
            if a not in self._adj_list[b]:
                self._adj_list[b].append(a)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def linear(cls, n: int) -> "CouplingMap":
        """Linear chain: 0-1-2-…-(n−1)."""
        if n < 1:
            raise TranspilerError(f"n must be >= 1, got {n}")
        return cls(n, [(i, i + 1) for i in range(n - 1)])

    @classmethod
    def ring(cls, n: int) -> "CouplingMap":
        """Cyclic ring: 0-1-…-(n−1)-0."""
        if n < 2:
            raise TranspilerError(f"ring requires n >= 2, got {n}")
        return cls(n, [(i, (i + 1) % n) for i in range(n)])

    @classmethod
    def grid(cls, rows: int, cols: int) -> "CouplingMap":
        """2-D rectangular grid with open boundary conditions."""
        if rows < 1 or cols < 1:
            raise TranspilerError(
                f"rows and cols must be >= 1, got {rows}, {cols}"
            )
        edges = []
        for r in range(rows):
            for c in range(cols):
                q = r * cols + c
                if c + 1 < cols:
                    edges.append((q, q + 1))       # horizontal bond
                if r + 1 < rows:
                    edges.append((q, q + cols))    # vertical bond
        return cls(rows * cols, edges)

    @classmethod
    def all_to_all(cls, n: int) -> "CouplingMap":
        """Fully connected (all-to-all) topology."""
        if n < 1:
            raise TranspilerError(f"n must be >= 1, got {n}")
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return cls(n, edges)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def is_connected(self, q1: int, q2: int) -> bool:
        """Return True if qubits q1 and q2 share a physical edge."""
        return (q1, q2) in self._adj_set

    def neighbors(self, q: int) -> "list[int]":
        """Return the list of physical qubits adjacent to *q*."""
        return list(self._adj_list.get(q, []))

    def shortest_path(self, q1: int, q2: int) -> "list[int]":
        """BFS shortest path from q1 to q2.

        Returns
        -------
        list[int]
            Physical qubit indices along the path, inclusive of both
            endpoints.

        Raises
        ------
        TranspilerError
            If no path exists (disconnected graph).
        """
        if q1 == q2:
            return [q1]
        prev: "dict[int, int | None]" = {q1: None}
        queue: "deque[int]" = deque([q1])
        while queue:
            curr = queue.popleft()
            for nb in self._adj_list.get(curr, []):
                if nb not in prev:
                    prev[nb] = curr
                    if nb == q2:
                        path: "list[int]" = []
                        node: "int | None" = q2
                        while node is not None:
                            path.append(node)
                            node = prev[node]
                        return path[::-1]
                    queue.append(nb)
        raise TranspilerError(
            f"No path between physical qubits {q1} and {q2}. "
            "Check that the coupling map is connected."
        )

    def distance(self, q1: int, q2: int) -> int:
        """SWAP distance (number of edges) between qubits q1 and q2."""
        return len(self.shortest_path(q1, q2)) - 1

    def __repr__(self) -> str:  # pragma: no cover
        return f"CouplingMap(n_qubits={self.n_qubits}, edges={self.edges})"


# ---------------------------------------------------------------------------
# Native gate sets and decompositions
# ---------------------------------------------------------------------------

# Lower-case canonical names accepted in each set.
_CX_NATIVE: "frozenset[str]" = frozenset({
    "cx", "cnot", "h", "x", "y", "z", "s", "sdg",
    "t", "tdg", "rz", "rx", "ry", "u1", "u2", "u3", "id",
})
_CZ_NATIVE: "frozenset[str]" = frozenset({
    "cz", "h", "x", "y", "z", "s", "sdg",
    "t", "tdg", "rz", "rx", "ry", "sx", "sxdg", "id",
})
_ALL_NATIVE: "frozenset[str]" = _CX_NATIVE | _CZ_NATIVE | frozenset({
    "swap", "ccx", "toffoli", "cswap",
})

_NATIVE_SETS: "dict[str, frozenset[str]]" = {
    "cx_rz": _CX_NATIVE,
    "cz_rz": _CZ_NATIVE,
    "all": _ALL_NATIVE,
}

# Pairs of mutually-inverse single-qubit gates (both orders)
_INVERSE_PAIRS: "set[frozenset[str]]" = {
    frozenset({"h", "h"}),
    frozenset({"x", "x"}),
    frozenset({"y", "y"}),
    frozenset({"z", "z"}),
    frozenset({"s", "sdg"}),
    frozenset({"t", "tdg"}),
    frozenset({"sx", "sxdg"}),
}


def _decompose_single(gate: GateOp, native: "frozenset[str]") -> "list[GateOp]":
    """Return native-gate decomposition of *gate*.  Raises if not possible."""
    name = gate.name.lower()
    if name in native:
        return [gate]
    q = gate.qubits

    # ── Two-qubit gate inter-conversions ─────────────────────────────────
    if name == "cz" and "cx" in native:
        # CZ = H(target) · CX · H(target)
        return [
            GateOp("h", [q[1]]),
            GateOp("cx", [q[0], q[1]]),
            GateOp("h", [q[1]]),
        ]
    if name in ("cx", "cnot") and "cz" in native:
        # CX = H(target) · CZ · H(target)
        return [
            GateOp("h", [q[1]]),
            GateOp("cz", [q[0], q[1]]),
            GateOp("h", [q[1]]),
        ]

    # ── SWAP ──────────────────────────────────────────────────────────────
    if name == "swap":
        if "cx" in native:
            return [
                GateOp("cx", [q[0], q[1]]),
                GateOp("cx", [q[1], q[0]]),
                GateOp("cx", [q[0], q[1]]),
            ]
        if "cz" in native:
            # SWAP = (H₁ CZ H₁) (H₀ CZ H₀) (H₁ CZ H₁)
            # Equivalent to 3 × CX with target-H wrapping, then convert
            # Here: SWAP(a,b) via CZ: CZ is symmetric so we need to use
            # H to convert CX(a,b)→CZ. Same result as above with CZ.
            return [
                GateOp("h", [q[1]]),
                GateOp("cz", [q[0], q[1]]),
                GateOp("h", [q[1]]),
                GateOp("h", [q[0]]),
                GateOp("cz", [q[0], q[1]]),
                GateOp("h", [q[0]]),
                GateOp("h", [q[1]]),
                GateOp("cz", [q[0], q[1]]),
                GateOp("h", [q[1]]),
            ]

    # ── Toffoli (CCX): standard 6-CNOT decomposition ─────────────────────
    if name in ("ccx", "toffoli") and "cx" in native:
        c0, c1, t = q[0], q[1], q[2]
        return [
            GateOp("h", [t]),
            GateOp("cx", [c1, t]),   GateOp("tdg", [t]),
            GateOp("cx", [c0, t]),   GateOp("t",   [t]),
            GateOp("cx", [c1, t]),   GateOp("tdg", [t]),
            GateOp("cx", [c0, t]),
            GateOp("t",  [c1]),      GateOp("t",   [t]),
            GateOp("h",  [t]),
            GateOp("cx", [c0, c1]),
            GateOp("t",  [c0]),      GateOp("tdg", [c1]),
            GateOp("cx", [c0, c1]),
        ]

    raise TranspilerError(
        f"Cannot decompose gate '{gate.name}' into native set {sorted(native)}"
    )


# ---------------------------------------------------------------------------
# Circuit utilities
# ---------------------------------------------------------------------------

def _n_logical_qubits(circuit: "list[GateOp]") -> int:
    """Return the number of logical qubits referenced in *circuit*."""
    if not circuit:
        return 0
    return max(q for gate in circuit for q in gate.qubits) + 1


def _circuit_depth(circuit: "list[GateOp]") -> int:
    """Compute circuit depth as the longest sequential layer."""
    if not circuit:
        return 0
    time: "dict[int, int]" = {}
    for gate in circuit:
        t = max((time.get(q, 0) for q in gate.qubits), default=0)
        t += 1
        for q in gate.qubits:
            time[q] = t
    return max(time.values()) if time else 0


def _cancel_inverses(circuit: "list[GateOp]") -> "list[GateOp]":
    """Peephole: cancel adjacent single-qubit inverse-gate pairs."""
    changed = True
    ops = list(circuit)
    while changed:
        changed = False
        new_ops: "list[GateOp]" = []
        i = 0
        while i < len(ops):
            if i + 1 < len(ops):
                a, b = ops[i], ops[i + 1]
                # Same qubit, inverse names, no params
                if (
                    len(a.qubits) == 1
                    and a.qubits == b.qubits
                    and not a.params
                    and not b.params
                    and frozenset({a.name.lower(), b.name.lower()}) in _INVERSE_PAIRS
                ):
                    # Cancel the pair
                    i += 2
                    changed = True
                    continue
                # Adjacent SWAP · SWAP = identity
                if (
                    a.name.lower() == "swap"
                    and b.name.lower() == "swap"
                    and sorted(a.qubits) == sorted(b.qubits)
                ):
                    i += 2
                    changed = True
                    continue
            new_ops.append(ops[i])
            i += 1
        ops = new_ops
    return ops


# ---------------------------------------------------------------------------
# TranspileResult
# ---------------------------------------------------------------------------

@dataclass
class TranspileResult:
    """Result of a transpilation run.

    Attributes
    ----------
    circuit:
        Transpiled gate list in terms of **physical** qubit indices.
    qubit_map:
        Logical → physical qubit mapping used at the start of routing.
    n_swaps:
        Number of SWAP gates inserted during routing.
    n_gates_before:
        Gate count in the original circuit (before any transformation).
    n_gates_after:
        Gate count after transpilation (includes decomposition + SWAPs).
    depth_before:
        Circuit depth before transpilation.
    depth_after:
        Circuit depth after transpilation.
    """
    circuit: "list[GateOp]"
    qubit_map: "dict[int, int]"
    n_swaps: int
    n_gates_before: int
    n_gates_after: int
    depth_before: int
    depth_after: int


# ---------------------------------------------------------------------------
# HardwareTranspiler
# ---------------------------------------------------------------------------

class HardwareTranspiler:
    """Hardware-aware circuit transpiler.

    Parameters
    ----------
    coupling_map:
        Physical device topology.
    native_gates:
        Specifies the device's native gate set.  Either a string shorthand
        (``"cx_rz"``, ``"cz_rz"``, ``"all"``) or a custom
        ``frozenset[str]`` of lower-case gate names.

        * ``"cx_rz"`` — CX/CNOT-based devices (e.g. IBM superconducting):
          ``{cx, h, x, y, z, s, sdg, t, tdg, rz, rx, ry, …}``
        * ``"cz_rz"`` — CZ-based devices (e.g. Google, Rigetti):
          ``{cz, h, x, y, z, s, sdg, t, tdg, rz, rx, ry, sx, …}``
        * ``"all"`` — accept all standard gates (no decomposition).

    Examples
    --------
    >>> dev = CouplingMap.linear(3)
    >>> t = HardwareTranspiler(dev)
    >>> result = t.transpile([GateOp("cx", [0, 2])])
    >>> result.n_swaps
    1
    """

    def __init__(
        self,
        coupling_map: CouplingMap,
        native_gates: "str | frozenset[str]" = "cx_rz",
    ):
        self.coupling_map = coupling_map
        if isinstance(native_gates, str):
            if native_gates not in _NATIVE_SETS:
                raise TranspilerError(
                    f"Unknown native_gates preset '{native_gates}'. "
                    f"Choose from {list(_NATIVE_SETS)}"
                )
            self._native: "frozenset[str]" = _NATIVE_SETS[native_gates]
        else:
            self._native = frozenset(g.lower() for g in native_gates)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transpile(
        self,
        circuit: "list[GateOp]",
        initial_mapping: "dict[int, int] | None" = None,
        optimization_level: int = 1,
    ) -> TranspileResult:
        """Compile *circuit* to the target hardware.

        Parameters
        ----------
        circuit:
            Input gate list using **logical** qubit indices (0, 1, …).
        initial_mapping:
            Logical → physical qubit assignment.  If ``None``, the identity
            mapping (logical i → physical i) is used.  The number of
            logical qubits must not exceed ``coupling_map.n_qubits``.
        optimization_level:
            0 = no optimisation; 1 = default (basic); 2 = peephole
            cancellation of adjacent inverse gates.

        Returns
        -------
        TranspileResult

        Raises
        ------
        TranspilerError
            If the circuit requires more logical qubits than the device
            has physical qubits, or if a gate cannot be decomposed.
        """
        if not circuit:
            n_phys = self.coupling_map.n_qubits
            identity = {i: i for i in range(n_phys)}
            return TranspileResult([], initial_mapping or identity, 0, 0, 0, 0, 0)

        n_logical = _n_logical_qubits(circuit)
        if n_logical > self.coupling_map.n_qubits:
            raise TranspilerError(
                f"Circuit uses {n_logical} logical qubits but device has "
                f"only {self.coupling_map.n_qubits} physical qubits."
            )

        n_before = len(circuit)
        depth_before = _circuit_depth(circuit)

        # ── Step 1: gate decomposition ────────────────────────────────
        decomposed = self._decompose_all(circuit)

        # ── Step 2: SWAP routing ──────────────────────────────────────
        if initial_mapping is None:
            initial_mapping = {i: i for i in range(n_logical)}
        routed, final_mapping, n_swaps = self._route(decomposed, initial_mapping)

        # ── Step 2b: decompose inserted SWAPs into native gates ───────
        # _route() inserts raw GateOp("swap", ...) for routing; these must
        # be decomposed so the output only contains native gates.
        routed = self._decompose_all(routed)

        # ── Step 3: optimisation ──────────────────────────────────────
        if optimization_level >= 2:
            routed = _cancel_inverses(routed)

        return TranspileResult(
            circuit=routed,
            qubit_map=initial_mapping,
            n_swaps=n_swaps,
            n_gates_before=n_before,
            n_gates_after=len(routed),
            depth_before=depth_before,
            depth_after=_circuit_depth(routed),
        )

    def decompose(self, gate: GateOp) -> "list[GateOp]":
        """Decompose a single gate into native gates."""
        return _decompose_single(gate, self._native)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decompose_all(self, circuit: "list[GateOp]") -> "list[GateOp]":
        """Recursively decompose all non-native gates."""
        result: "list[GateOp]" = []
        for gate in circuit:
            if gate.name.lower() in self._native:
                result.append(gate)
            else:
                sub = _decompose_single(gate, self._native)
                # Recursively decompose in case the decomposition itself
                # contains non-native gates (e.g. SWAP → CX via CZ path)
                for sub_gate in sub:
                    if sub_gate.name.lower() in self._native:
                        result.append(sub_gate)
                    else:
                        result.extend(self._decompose_all([sub_gate]))
        return result

    def _route(
        self,
        circuit: "list[GateOp]",
        initial_mapping: "dict[int, int]",
    ) -> "tuple[list[GateOp], dict[int, int], int]":
        """Greedy SWAP-insertion routing.

        For each two-qubit gate on non-adjacent physical qubits, walks
        logical qubit l1 toward l2 along the BFS-shortest path, inserting
        one SWAP per hop.

        Returns (routed_circuit, final_mapping, n_swaps_inserted).
        """
        cm = self.coupling_map

        # Validate that every gate references a mapped logical qubit
        for gate in circuit:
            for lq in gate.qubits:
                if lq not in initial_mapping:
                    raise TranspilerError(
                        f"Logical qubit {lq} referenced in circuit but not "
                        "present in initial_mapping."
                    )

        # Build mutable mapping tables
        l2p: "dict[int, int]" = dict(initial_mapping)  # logical → physical
        p2l: "dict[int, int]" = {p: l for l, p in l2p.items()}  # inverse

        routed: "list[GateOp]" = []
        n_swaps = 0

        for gate in circuit:
            name = gate.name.lower()
            qs = gate.qubits

            if len(qs) == 1:
                # Single-qubit gate: just remap the qubit
                p = l2p[qs[0]]
                routed.append(GateOp(gate.name, [p], list(gate.params)))

            elif len(qs) == 2:
                l1, l2 = qs[0], qs[1]
                p1 = l2p[l1]
                p2 = l2p[l2]

                if not cm.is_connected(p1, p2):
                    # Move l1 toward l2 along the shortest path
                    path = cm.shortest_path(p1, p2)
                    # Insert SWAPs: move p1 hop-by-hop toward p2
                    for i in range(len(path) - 2):
                        pa, pb = path[i], path[i + 1]
                        # SWAP physical qubits pa and pb
                        routed.append(GateOp("swap", [pa, pb]))
                        n_swaps += 1
                        # Update mappings
                        la = p2l.get(pa)
                        lb = p2l.get(pb)
                        # Remove stale entries before updating to keep
                        # p2l bijective: pa and pb exchange their occupants.
                        if la is not None:
                            del p2l[pa]
                        if lb is not None:
                            del p2l[pb]
                        if la is not None:
                            l2p[la] = pb
                            p2l[pb] = la
                        if lb is not None:
                            l2p[lb] = pa
                            p2l[pa] = lb

                # Emit the gate on the (now-adjacent) physical qubits
                p1_final = l2p[l1]
                p2_final = l2p[l2]
                routed.append(GateOp(gate.name, [p1_final, p2_final], list(gate.params)))

            else:
                # ≥3-qubit gate: emit on remapped qubits (must already be native)
                routed.append(
                    GateOp(gate.name, [l2p[q] for q in qs], list(gate.params))
                )

        return routed, l2p, n_swaps
