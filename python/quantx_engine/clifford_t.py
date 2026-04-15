"""
Clifford+T fault-tolerant gate compilation module for the QuantX engine.

Implements symbolic single-qubit gate representation, Clifford group identification,
Solovay-Kitaev approximation, and T-count optimization utilities.

Classes:
    GateSequence     — symbolic gate sequence with matrix evaluation
    CliffordGroup    — single-qubit Clifford group (24 elements)
    SolovayKitaev    — depth-limited Clifford+T approximation compiler
    TCountOptimizer  — peephole T-count reduction via rewrite rules

Usage::

    from quantx_engine.clifford_t import GateSequence, CliffordGroup, SolovayKitaev

    # Identify a Clifford gate
    seq = CliffordGroup.identify(H_GATE)

    # Compile arbitrary rotation
    import numpy as np
    sk = SolovayKitaev(epsilon=0.01, max_depth=3)
    result = sk.compile(np.array([[0,1],[1,0]], dtype=complex))

    # Optimize T-count
    from quantx_engine.clifford_t import TCountOptimizer
    opt = TCountOptimizer()
    reduced = opt.optimize(GateSequence(["T", "T", "T", "T"]))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

__all__ = [
    "CliffordTError",
    "GateSequence",
    "CliffordGroup",
    "SolovayKitaev",
    "TCountOptimizer",
    "I_GATE",
    "X_GATE",
    "Y_GATE",
    "Z_GATE",
    "H_GATE",
    "S_GATE",
    "T_GATE",
    "S_DAG",
    "T_DAG",
    "CNOT_GATE",
]


class CliffordTError(ValueError):
    """Raised for invalid Clifford+T inputs or configuration."""


# ── Gate set constants (guarded behind _HAS_NUMPY) ───────────────────────────

if _HAS_NUMPY:
    I_GATE = np.eye(2, dtype=complex)
    X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
    Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z_GATE = np.diag([1.0 + 0j, -1.0 + 0j])
    H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S_GATE = np.diag([1.0 + 0j, 1j])
    T_GATE = np.diag([1.0 + 0j, np.exp(1j * np.pi / 4)])
    S_DAG = np.diag([1.0 + 0j, -1j])
    T_DAG = np.diag([1.0 + 0j, np.exp(-1j * np.pi / 4)])
    CNOT_GATE = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )
else:  # pragma: no cover
    I_GATE = None
    X_GATE = None
    Y_GATE = None
    Z_GATE = None
    H_GATE = None
    S_GATE = None
    T_GATE = None
    S_DAG = None
    T_DAG = None
    CNOT_GATE = None

# ── Gate name → matrix mapping ────────────────────────────────────────────────

def _gate_map() -> dict:
    """Return name-to-matrix mapping (requires numpy)."""
    return {
        "I": I_GATE,
        "X": X_GATE,
        "Y": Y_GATE,
        "Z": Z_GATE,
        "H": H_GATE,
        "S": S_GATE,
        "T": T_GATE,
        "S†": S_DAG,
        "T†": T_DAG,
        "CNOT": CNOT_GATE,
    }


_INVERSE_GATE: dict[str, str] = {
    "I": "I",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "H": "H",
    "S": "S†",
    "S†": "S",
    "T": "T†",
    "T†": "T",
    "CNOT": "CNOT",
}

_T_GATES = frozenset({"T", "T†"})
_CLIFFORD_GATES = frozenset({"I", "X", "Y", "Z", "H", "S", "S†", "CNOT"})


# ── Helper ────────────────────────────────────────────────────────────────────

def _unitary_equal_up_to_phase(A: "np.ndarray", B: "np.ndarray") -> bool:
    """Return True if A == e^{iφ} B for some real φ."""
    # Find index of max |A[i,j]|
    idx = int(np.argmax(np.abs(A)))
    i, j = divmod(idx, A.shape[1])
    b_val = B[i, j]
    if abs(b_val) < 1e-12:
        return bool(np.allclose(A, 0, atol=1e-10))
    phase = A[i, j] / b_val
    return bool(np.allclose(A, phase * B, atol=1e-10))


# ── GateSequence ──────────────────────────────────────────────────────────────

@dataclass
class GateSequence:
    """Symbolic sequence of named quantum gates."""

    gates: list[str]
    n_qubits: int = 1

    def t_count(self) -> int:
        """Number of T and T† gates."""
        return sum(1 for g in self.gates if g in _T_GATES)

    def clifford_count(self) -> int:
        """Number of non-T gates (Clifford gates)."""
        return sum(1 for g in self.gates if g not in _T_GATES)

    def total_count(self) -> int:
        """Total number of gates."""
        return len(self.gates)

    def to_matrix(self) -> "np.ndarray":
        """Compute unitary matrix; gates[0] applied first (rightmost in product).

        U = gates[-1] @ ... @ gates[0]
        """
        if not _HAS_NUMPY:  # pragma: no cover
            raise CliffordTError("numpy is required for to_matrix()")
        gmap = _gate_map()
        if not self.gates:
            dim = 2 if self.n_qubits == 1 else (2 ** self.n_qubits)
            return np.eye(dim, dtype=complex)
        expected_dim = 2 ** self.n_qubits
        result = None
        for name in self.gates:
            mat = gmap.get(name)
            if mat is None:
                raise CliffordTError(f"Unknown gate: {name!r}")
            if mat.shape != (expected_dim, expected_dim):
                raise CliffordTError(
                    f"Gate {name!r} has shape {mat.shape}, expected "
                    f"({expected_dim}, {expected_dim}) for n_qubits={self.n_qubits}"
                )
            if result is None:
                result = mat.copy()
            else:
                result = mat @ result
        return result

    def append(self, gate: str) -> "GateSequence":
        """Return new GateSequence with gate appended."""
        return GateSequence(self.gates + [gate], self.n_qubits)

    def prepend(self, gate: str) -> "GateSequence":
        """Return new GateSequence with gate prepended."""
        return GateSequence([gate] + self.gates, self.n_qubits)

    def inverse(self) -> "GateSequence":
        """Reverse gate list, replacing each gate with its inverse."""
        inv_gates = [_INVERSE_GATE.get(g, g) for g in reversed(self.gates)]
        return GateSequence(inv_gates, self.n_qubits)

    def __repr__(self) -> str:
        return f"GateSequence([{', '.join(self.gates)}], t_count={self.t_count()})"

    def to_dict(self) -> dict:
        """Serialise to plain dict."""
        return {
            "gates": list(self.gates),
            "t_count": self.t_count(),
            "clifford_count": self.clifford_count(),
            "n_qubits": self.n_qubits,
        }


# ── CliffordGroup ─────────────────────────────────────────────────────────────

class CliffordGroup:
    """Single-qubit Clifford group (24 elements)."""

    GENERATORS: list[str] = ["H", "S"]

    # Lazily computed via generate()
    _elements_cache: list[GateSequence] | None = None

    @classmethod
    def generate(cls) -> list[GateSequence]:
        """BFS from identity using H, S, S†, X, Z; deduplicate up to global phase.

        Returns all 24 single-qubit Clifford gates as GateSequences.
        """
        if not _HAS_NUMPY:  # pragma: no cover
            raise CliffordTError("numpy is required for CliffordGroup.generate()")

        if cls._elements_cache is not None:
            return list(cls._elements_cache)

        generators = ["H", "S", "S†", "X", "Z"]
        # BFS
        identity = GateSequence([])
        queue: list[GateSequence] = [identity]
        unique_matrices: list["np.ndarray"] = [identity.to_matrix()]
        unique_seqs: list[GateSequence] = [identity]

        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            for g in generators:
                candidate = current.append(g)
                mat = candidate.to_matrix()
                already_seen = False
                for existing in unique_matrices:
                    if _unitary_equal_up_to_phase(mat, existing):
                        already_seen = True
                        break
                if not already_seen:
                    unique_matrices.append(mat)
                    unique_seqs.append(candidate)
                    queue.append(candidate)
                    if len(unique_seqs) == 24:
                        break
            if len(unique_seqs) == 24:
                break

        cls._elements_cache = unique_seqs
        return list(unique_seqs)

    @property
    def ELEMENTS(self) -> list[GateSequence]:  # pragma: no cover
        return self.generate()

    @classmethod
    def identify(cls, U: "np.ndarray") -> GateSequence | None:
        """Find which Clifford best matches U (up to global phase).

        Returns the matching GateSequence, or None if U is not a Clifford gate.
        """
        if not _HAS_NUMPY:  # pragma: no cover
            raise CliffordTError("numpy is required for CliffordGroup.identify()")
        U = np.asarray(U, dtype=complex)
        if U.ndim != 2 or U.shape != (2, 2):
            return None
        elements = cls.generate()
        for seq in elements:
            mat = seq.to_matrix()
            if _unitary_equal_up_to_phase(U, mat):
                return seq
        return None

    @classmethod
    def random(cls, seed: int | None = None) -> GateSequence:
        """Return a random Clifford gate sequence."""
        if not _HAS_NUMPY:  # pragma: no cover
            raise CliffordTError("numpy is required for CliffordGroup.random()")
        rng = np.random.default_rng(seed)
        elements = cls.generate()
        idx = int(rng.integers(0, len(elements)))
        return elements[idx]


# ── SolovayKitaev ─────────────────────────────────────────────────────────────

@dataclass
class SolovayKitaev:
    """Depth-limited Clifford+T approximation compiler.

    Uses exhaustive search over a pre-built basis of all Clifford+T sequences
    up to max_depth gates from {H, S, T, T†, S†}.
    """

    epsilon: float = 0.01
    max_depth: int = 3
    _basis: list[GateSequence] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise CliffordTError("epsilon must be > 0")
        if _HAS_NUMPY:
            self._basis = self.build_basis(self.max_depth)

    @staticmethod
    def _matrix_distance(A: "np.ndarray", B: "np.ndarray") -> float:
        """Normalised trace distance between two unitaries (phase-invariant)."""
        product = A.conj().T @ B
        tr = np.trace(product)
        val = 1.0 - abs(tr) ** 2 / 4.0
        return float(np.sqrt(max(0.0, val)))

    def _closest_basis(self, U: "np.ndarray") -> GateSequence:
        """Find the sequence in self._basis minimising distance to U."""
        best_seq = self._basis[0] if self._basis else GateSequence([])
        best_dist = float("inf")
        for seq in self._basis:
            mat = seq.to_matrix()
            d = self._matrix_distance(U, mat)
            if d < best_dist:
                best_dist = d
                best_seq = seq
        return best_seq

    def _su2_project(self, U: "np.ndarray") -> "np.ndarray":
        """Project 2×2 unitary to SU(2) by normalising determinant."""
        try:
            det = np.linalg.det(U)
            if abs(det) < 1e-15:
                return U.copy()
            return U / np.sqrt(det)
        except Exception:
            return U.copy()

    def compile(self, U: "np.ndarray") -> GateSequence:
        """Compile an arbitrary 2×2 unitary to a Clifford+T sequence.

        Finds the closest sequence in the pre-built basis (depth-limited
        exhaustive search).
        """
        if not _HAS_NUMPY:  # pragma: no cover
            raise CliffordTError("numpy is required for SolovayKitaev.compile()")
        U_su2 = self._su2_project(U)
        return self._closest_basis(U_su2)

    def build_basis(self, max_length: int) -> list[GateSequence]:
        """Generate all Clifford+T sequences up to max_length gates.

        Generators: {H, S, T, T†, S†}. Deduplicates up to global phase.
        Caps max_length at 4 for practical use.
        """
        if not _HAS_NUMPY:  # pragma: no cover
            raise CliffordTError("numpy is required for build_basis()")
        max_length = min(max_length, 4)
        generators = ["H", "S", "T", "T†", "S†"]

        # Start with identity
        unique_matrices: list["np.ndarray"] = [np.eye(2, dtype=complex)]
        unique_seqs: list[GateSequence] = [GateSequence([])]

        current_layer: list[GateSequence] = [GateSequence([])]

        for _depth in range(max_length):
            next_layer: list[GateSequence] = []
            for seq in current_layer:
                for g in generators:
                    candidate = seq.append(g)
                    mat = candidate.to_matrix()
                    already_seen = False
                    for existing in unique_matrices:
                        if _unitary_equal_up_to_phase(mat, existing):
                            already_seen = True
                            break
                    if not already_seen:
                        unique_matrices.append(mat)
                        unique_seqs.append(candidate)
                        next_layer.append(candidate)
            current_layer = next_layer

        return unique_seqs

    def approximation_error(self, U: "np.ndarray") -> float:
        """Return the matrix distance between U and compile(U).to_matrix()."""
        approx = self.compile(U)
        return self._matrix_distance(U, approx.to_matrix())


# ── TCountOptimizer ───────────────────────────────────────────────────────────

# Peephole rewrite rules: (pattern_tuple) → replacement_list
_REWRITE_RULES: list[tuple[tuple[str, ...], list[str]]] = [
    (("T", "T"), ["S"]),
    (("T†", "T†"), ["S†"]),
    (("T", "T†"), []),
    (("T†", "T"), []),
    (("S", "S"), ["Z"]),
    (("S†", "S†"), ["Z"]),
    (("S", "S†"), []),
    (("S†", "S"), []),
    (("Z", "Z"), []),
    (("X", "X"), []),
    (("Y", "Y"), []),
    (("H", "H"), []),
]


@dataclass
class TCountOptimizer:
    """Peephole T-count optimizer using local rewrite rules."""

    def optimize(self, seq: GateSequence) -> GateSequence:
        """Apply rewrite rules repeatedly until stable.

        Returns a new GateSequence with reduced T-count.
        """
        gates = list(seq.gates)
        changed = True
        while changed:
            changed = False
            for pattern, replacement in _REWRITE_RULES:
                plen = len(pattern)
                i = 0
                while i <= len(gates) - plen:
                    if tuple(gates[i: i + plen]) == pattern:
                        gates[i: i + plen] = replacement
                        changed = True
                        break  # restart scan after any change
                    i += 1
        return GateSequence(gates, seq.n_qubits)

    def t_count_reduction(self, seq: GateSequence) -> int:
        """Return original t_count minus optimized t_count."""
        return seq.t_count() - self.optimize(seq).t_count()

    def to_dict(self) -> dict:
        """Return metadata dict."""
        return {"optimizer": "peephole", "rules_count": len(_REWRITE_RULES)}
