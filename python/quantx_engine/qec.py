"""
Quantum Error Correction research toolkit for the QuantX engine.

Implements stabilizer codes, syndrome decoding, and logical error rate estimation
in pure Python / NumPy.

Codes included:
    RepetitionCode(n)     — [[n, 1, n]] (odd n): protects against bit-flip errors
    SurfaceCode(d)        — rotated planar [[d², 1, d]]: industry standard for FTQC

Decoders:
    RepetitionCode  — majority vote (optimal for independent errors)
    SurfaceCode     — minimum-weight syndrome graph matching (greedy MWPM)

Usage:
    from quantx_engine.qec import SurfaceCode, LogicalErrorSimulator

    code = SurfaceCode(d=3)
    sim = LogicalErrorSimulator(code)

    # Single-shot logical error rate at p=0.01
    p_L = sim.logical_error_rate(p_phys=0.01, n_trials=10000)

    # Threshold sweep
    result = sim.threshold_sweep(p_range=(0.001, 0.15), n_points=12, n_trials=5000)
    print(f"Estimated threshold: {result.threshold:.4f}")
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

__all__ = [
    "QECError",
    "RepetitionCode",
    "SurfaceCode",
    "LogicalErrorSimulator",
    "ThresholdResult",
    "MWPMDecoder",
]


class QECError(ValueError):
    """Raised for invalid QEC inputs or unsupported configurations."""


# ── RepetitionCode ─────────────────────────────────────────────────────────────

class RepetitionCode:
    """[[n, 1, n]] repetition code (odd n recommended).

    Encodes one logical qubit into n physical qubits. Protects against
    ⌊(n-1)/2⌋ bit-flip (Z-basis) errors. Phase-flip errors are undetected.

    Parameters
    ----------
    n:
        Number of physical qubits. Use odd n for symmetric correction.
    """

    def __init__(self, n: int) -> None:
        if n < 3:
            raise QECError(f"RepetitionCode requires n >= 3, got {n}")
        self.n = n
        self.k = 1
        self.d = n  # distance (detects n-1 errors, corrects ⌊(n-1)/2⌋)
        # Stabilizers: Z_i Z_{i+1} for i in 0..n-2
        self.stabilizers: list[list[int]] = [[i, i + 1] for i in range(n - 1)]

    # ── Encoding / decoding ───────────────────────────────────────────────────

    def encode(self, logical: int) -> list[int]:
        """Encode logical 0 or 1 into n physical qubits.

        Parameters
        ----------
        logical:
            0 or 1.

        Returns
        -------
        Physical codeword, list of n bits.
        """
        if logical not in (0, 1):
            raise QECError(f"logical must be 0 or 1, got {logical!r}")
        return [logical] * self.n

    def syndrome(self, codeword: list[int]) -> list[int]:
        """Compute parity-check syndrome of a (possibly errored) codeword.

        Returns
        -------
        Syndrome bits: 1 where Z_i Z_{i+1} anticommutes (a parity flip), else 0.
        Length = n-1.
        """
        return [codeword[i] ^ codeword[i + 1] for i in range(self.n - 1)]

    def decode(self, codeword: list[int]) -> int:
        """Majority-vote decoder: return the most common bit value.

        Optimal for independent, identically distributed bit-flip errors with
        p < 0.5.

        Returns
        -------
        Decoded logical bit (0 or 1).
        """
        return int(sum(codeword) > self.n / 2)

    def correct(self, codeword: list[int]) -> list[int]:
        """Apply syndrome-based correction and return the corrected codeword.

        Uses the first non-zero syndrome pair to identify the error location.
        Handles single errors exactly; applies greedy correction for multiple.

        Returns
        -------
        Corrected codeword, list of n bits.
        """
        syn = self.syndrome(codeword)
        corrected = list(codeword)
        i = 0
        while i < len(syn):
            if syn[i]:
                # Determine which qubit flipped: check if i+1 syndrome also set
                if i + 1 < len(syn) and syn[i + 1]:
                    # Error on qubit i+1 (inner qubit)
                    corrected[i + 1] ^= 1
                    syn = self.syndrome(corrected)
                    i = 0
                    continue
                else:
                    # Error on qubit i (boundary) or qubit i+1 (boundary)
                    # Flip the lower-weight endpoint
                    corrected[i] ^= 1
                    syn = self.syndrome(corrected)
                    i = 0
                    continue
            i += 1
        return corrected

    def logical_z(self) -> list[int]:
        """Logical Z operator: indices of qubits in the Z-type logical operator."""
        return list(range(self.n))  # Z on all qubits

    def to_dict(self) -> dict[str, Any]:
        return {"code": "repetition", "n": self.n, "k": self.k, "d": self.d}


# ── SurfaceCode ───────────────────────────────────────────────────────────────

def _surface_data_qubits(d: int) -> list[tuple[int, int]]:
    """Return (row, col) positions of data qubits in the rotated surface code."""
    # Data qubits occupy all d×d positions
    return [(r, c) for r in range(d) for c in range(d)]


def _surface_z_stabilizers(d: int) -> list[list[int]]:
    """Z-type stabilizers for the rotated surface code (bulk + boundary plaquettes).

    Each stabilizer is a list of data qubit indices (row-major order).
    Z stabilizers are weight-4 in the bulk, weight-2 at top/bottom boundaries.

    Boundary assignment (matching the rotated-code checkerboard convention):
    - Top boundary half-plaquettes: odd column offsets (c=1, 3, ...)
    - Bottom boundary half-plaquettes: even column offsets (c=0, 2, ...)
    This ensures no boundary stabilizer overlaps a bulk stabilizer, guaranteeing
    distinct syndromes for all single-qubit X errors.
    """
    stabilizers = []
    for r in range(d - 1):
        for c in range(d - 1):
            # Even checkerboard plaquettes → Z type
            if (r + c) % 2 == 0:
                qubits = [
                    r * d + c,
                    r * d + c + 1,
                    (r + 1) * d + c,
                    (r + 1) * d + c + 1,
                ]
                stabilizers.append(qubits)
    # Top boundary (odd c): half-plaquettes between virtual row -1 and row 0
    for c in range(1, d - 1, 2):
        stabilizers.append([c, c + 1])
    # Bottom boundary (even c): half-plaquettes between row d-1 and virtual row d
    for c in range(0, d - 1, 2):
        stabilizers.append([(d - 1) * d + c, (d - 1) * d + c + 1])
    return stabilizers


def _surface_x_stabilizers(d: int) -> list[list[int]]:
    """X-type stabilizers for the rotated surface code."""
    stabilizers = []
    for r in range(d - 1):
        for c in range(d - 1):
            if (r + c) % 2 == 1:  # Odd checkerboard plaquettes → X type
                qubits = [
                    r * d + c,
                    r * d + c + 1,
                    (r + 1) * d + c,
                    (r + 1) * d + c + 1,
                ]
                stabilizers.append(qubits)
    # Left and right boundary half-plaquettes
    for r in range(0, d - 1, 2):
        stabilizers.append([r * d, (r + 1) * d])
    for r in range(1, d - 1, 2):
        stabilizers.append([r * d + d - 1, (r + 1) * d + d - 1])
    return stabilizers


class SurfaceCode:
    """Rotated planar surface code [[d², 1, d]].

    Encodes one logical qubit with code distance d. Corrects up to ⌊(d-1)/2⌋
    errors of each type (X or Z).

    Parameters
    ----------
    d:
        Code distance (odd recommended; d=3 is the smallest non-trivial case).
    """

    def __init__(self, d: int) -> None:
        if d < 3:
            raise QECError(f"SurfaceCode requires d >= 3, got {d}")
        self.d = d
        self.n = d * d          # number of data qubits
        self.k = 1              # logical qubits encoded
        self.z_stabs = _surface_z_stabilizers(d)
        self.x_stabs = _surface_x_stabilizers(d)
        self.n_z_syndromes = len(self.z_stabs)
        self.n_x_syndromes = len(self.x_stabs)
        # Positions of data qubits (row, col) for distance calculations
        self._positions = _surface_data_qubits(d)

    # ── Syndrome extraction ───────────────────────────────────────────────────

    def z_syndrome(self, x_errors: list[int] | "np.ndarray") -> list[int]:
        """Compute Z-stabilizer syndrome for a given X-error pattern.

        Parameters
        ----------
        x_errors:
            Binary array of length n (1 = X error on data qubit i).

        Returns
        -------
        Syndrome bits: 1 if stabilizer anticommutes with errors, else 0.
        """
        return [
            int(sum(x_errors[q] for q in stab) % 2)
            for stab in self.z_stabs
        ]

    def x_syndrome(self, z_errors: list[int] | "np.ndarray") -> list[int]:
        """Compute X-stabilizer syndrome for a given Z-error pattern."""
        return [
            int(sum(z_errors[q] for q in stab) % 2)
            for stab in self.x_stabs
        ]

    # ── Syndrome graph + greedy decoder ──────────────────────────────────────

    def _build_syndrome_graph(
        self, stabs: list[list[int]]
    ) -> tuple[dict[int, list[tuple[int, int]]], set[int]]:
        """Build adjacency list for the syndrome graph.

        Nodes are stabilizer indices 0..len(stabs)-1 plus one virtual boundary
        node BOUNDARY = len(stabs).

        Each data qubit q that appears in exactly one stabilizer s is a
        boundary edge: (s, BOUNDARY) with label q.  Each qubit q shared by
        exactly two stabilizers s_i, s_j is an interior edge: (s_i, s_j)
        with label q.  Qubits shared by more than two stabilizers are split
        across the first two containing stabilizers (degenerate, but never
        occurs in the rotated surface code).

        Returns (adj, boundary_qubits) where adj maps each node to a list of
        (neighbour_node, qubit_label) and boundary_qubits is the set of data
        qubits that are boundary edges.
        """
        BOUNDARY = len(stabs)
        # Map qubit → list of stabilizer indices that contain it
        stab_of: dict[int, list[int]] = {}
        for idx, stab in enumerate(stabs):
            for q in stab:
                stab_of.setdefault(q, []).append(idx)

        adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(BOUNDARY + 1)}
        boundary_qubits: set[int] = set()

        for q, containing in stab_of.items():
            if len(containing) == 1:
                # Boundary qubit: connects this stabilizer to the virtual node
                s = containing[0]
                adj[s].append((BOUNDARY, q))
                adj[BOUNDARY].append((s, q))
                boundary_qubits.add(q)
            elif len(containing) >= 2:
                # Interior qubit: connects two stabilizers
                si, sj = containing[0], containing[1]
                adj[si].append((sj, q))
                adj[sj].append((si, q))

        return adj, boundary_qubits

    def _syndrome_graph_bfs(
        self,
        start: int,
        goal: int,
        adj: dict[int, list[tuple[int, int]]],
    ) -> list[int]:
        """BFS shortest path in the syndrome graph from `start` to `goal`.

        Returns the list of data-qubit labels (edge labels) along the path.
        Each returned qubit, when added to the correction operator, anticommutes
        with exactly the two syndrome nodes it connects (start and goal, through
        possibly intermediate nodes).
        """
        from collections import deque
        # visited: node -> (parent_node, edge_qubit_to_parent)
        visited: dict[int, tuple[int | None, int | None]] = {start: (None, None)}
        queue: deque[int] = deque([start])
        while queue:
            curr = queue.popleft()
            if curr == goal:
                break
            for nb, q in adj.get(curr, []):
                if nb not in visited:
                    visited[nb] = (curr, q)
                    queue.append(nb)

        if goal not in visited:
            return []
        # Reconstruct path (collect edge qubits, skip virtual node's edge label)
        path: list[int] = []
        curr = goal
        while visited[curr][0] is not None:
            parent, q = visited[curr]
            if q is not None:
                path.append(q)
            curr = parent
        return path

    def _greedy_match(
        self,
        syndrome: list[int],
        stabs: list[list[int]],
        error_type: str,
    ) -> list[int]:
        """Greedy minimum-weight syndrome matching decoder via syndrome graph.

        Builds the syndrome graph (stabilizers as nodes, data qubits as edges,
        virtual boundary node for open boundary conditions), then greedily pairs
        defects by centroid distance and routes corrections along BFS shortest
        paths in the syndrome graph.  Routing in the syndrome graph guarantees
        that the correction anticommutes with exactly the paired stabilizers and
        commutes with all others, regardless of which source qubit is dequeued
        first.

        Parameters
        ----------
        syndrome:
            Binary syndrome vector indexed against `stabs`.
        stabs:
            List of stabilizers.
        error_type:
            Unused (kept for API compatibility; routing is handled by the
            syndrome graph structure, which is inherently boundary-aware).

        Returns
        -------
        Correction: binary array of length n.
        """
        defects = [i for i, s in enumerate(syndrome) if s]
        correction = [0] * self.n

        if not defects:
            return correction

        BOUNDARY = len(stabs)
        adj, _ = self._build_syndrome_graph(stabs)

        def centroid(stab_idx: int) -> tuple[float, float]:
            qubits = stabs[stab_idx]
            r = sum(self._positions[q][0] for q in qubits) / len(qubits)
            c = sum(self._positions[q][1] for q in qubits) / len(qubits)
            return r, c

        remaining = list(defects)
        while len(remaining) >= 2:
            # Pair nearest defects by centroid distance
            best_i, best_j, best_dist = 0, 1, float("inf")
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    ri, ci = centroid(remaining[i])
                    rj, cj = centroid(remaining[j])
                    dist = abs(ri - rj) + abs(ci - cj)
                    if dist < best_dist:
                        best_i, best_j, best_dist = i, j, dist

            si, sj = remaining[best_i], remaining[best_j]
            path = self._syndrome_graph_bfs(si, sj, adj)
            for q in path:
                correction[q] ^= 1

            remaining.pop(best_j)
            remaining.pop(best_i)

        # Unpaired defect: route to virtual BOUNDARY node in syndrome graph.
        # The syndrome graph ensures exactly the correct boundary qubit is chosen
        # (the one connecting this stabilizer to the virtual boundary node),
        # eliminating the first-source-bias of the previous qubit-grid BFS.
        if remaining:
            si = remaining[0]
            path = self._syndrome_graph_bfs(si, BOUNDARY, adj)
            for q in path:
                correction[q] ^= 1

        return correction

    def decode_x(self, z_syndrome: list[int]) -> list[int]:
        """Decode Z-stabilizer syndrome (caused by X errors) → X correction operator.

        The z_syndrome is indexed against z_stabs (Z stabilizers detect X errors).
        """
        return self._greedy_match(z_syndrome, self.z_stabs, "X")

    def decode_z(self, x_syndrome: list[int]) -> list[int]:
        """Decode X-stabilizer syndrome (caused by Z errors) → Z correction operator.

        The x_syndrome is indexed against x_stabs (X stabilizers detect Z errors).
        """
        return self._greedy_match(x_syndrome, self.x_stabs, "Z")

    # ── Logical operators ─────────────────────────────────────────────────────

    def logical_x(self) -> list[int]:
        """Logical X operator: first column of data qubits."""
        return [r * self.d for r in range(self.d)]

    def logical_z(self) -> list[int]:
        """Logical Z operator: first row of data qubits."""
        return list(range(self.d))

    def logical_error(
        self,
        x_errors: list[int],
        z_errors: list[int],
        x_correction: list[int],
        z_correction: list[int],
    ) -> tuple[bool, bool]:
        """Check if the net error + correction causes a logical error.

        Parameters
        ----------
        x_errors, z_errors:
            Physical error patterns (length n).
        x_correction, z_correction:
            Decoder correction patterns (length n).

        Returns
        -------
        (logical_x_error, logical_z_error): True if a logical error occurred.
        """
        # Net X/Z error after correction
        net_x = [x_errors[i] ^ x_correction[i] for i in range(self.n)]
        net_z = [z_errors[i] ^ z_correction[i] for i in range(self.n)]
        # Logical X error: net Z error overlaps logical Z operator an odd number of times
        lx = bool(sum(net_z[i] for i in self.logical_z()) % 2)
        # Logical Z error: net X error overlaps logical X operator an odd number of times
        lz = bool(sum(net_x[i] for i in self.logical_x()) % 2)
        return lx, lz

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": "surface",
            "d": self.d,
            "n": self.n,
            "k": self.k,
            "n_z_stabilizers": self.n_z_syndromes,
            "n_x_stabilizers": self.n_x_syndromes,
        }


# ── LogicalErrorSimulator ─────────────────────────────────────────────────────

@dataclass
class ThresholdResult:
    """Result of a threshold sweep simulation.

    Attributes
    ----------
    p_phys:
        Physical error rates swept.
    p_logical:
        Estimated logical error rates at each p_phys.
    threshold:
        Estimated threshold error rate (crossing point of logical error curves),
        or None if not detectable from the sweep.
    code_distances:
        List of distances used (for multi-distance threshold sweeps).
    """
    p_phys: list[float]
    p_logical: list[float]
    threshold: float | None = None
    code_distances: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_phys": self.p_phys,
            "p_logical": self.p_logical,
            "threshold": self.threshold,
            "code_distances": self.code_distances,
        }


class LogicalErrorSimulator:
    """Monte Carlo logical error rate estimator for stabilizer codes.

    Parameters
    ----------
    code:
        A `SurfaceCode` or `RepetitionCode` instance.
    """

    def __init__(self, code: SurfaceCode | RepetitionCode) -> None:
        self.code = code

    def logical_error_rate(
        self,
        p_phys: float,
        n_trials: int = 10000,
        seed: int | None = None,
        error_model: str = "depolarizing",
    ) -> float:
        """Estimate the logical error rate under the given physical error rate.

        Parameters
        ----------
        p_phys:
            Physical error probability per qubit per round (0 < p_phys < 0.5).
        n_trials:
            Number of Monte Carlo trials.
        seed:
            Random seed for reproducibility.
        error_model:
            'depolarizing' (symmetric X/Z errors) or 'bit_flip' (X only).

        Returns
        -------
        Estimated logical error probability per round.
        """
        if not 0 < p_phys < 1:
            raise QECError(f"p_phys must be in (0, 1), got {p_phys}")

        rng = random.Random(seed)
        n_logical_errors = 0

        if isinstance(self.code, RepetitionCode):
            n = self.code.n
            for _ in range(n_trials):
                errors = [1 if rng.random() < p_phys else 0 for _ in range(n)]
                codeword = [b ^ e for b, e in zip(self.code.encode(0), errors)]
                decoded = self.code.decode(codeword)
                if decoded != 0:
                    n_logical_errors += 1

        elif isinstance(self.code, SurfaceCode):
            n = self.code.n
            for _ in range(n_trials):
                if error_model == "depolarizing":
                    # Independent X and Z errors
                    x_err = [1 if rng.random() < p_phys else 0 for _ in range(n)]
                    z_err = [1 if rng.random() < p_phys else 0 for _ in range(n)]
                else:  # bit_flip
                    x_err = [1 if rng.random() < p_phys else 0 for _ in range(n)]
                    z_err = [0] * n

                syn_x = self.code.x_syndrome(z_err)
                syn_z = self.code.z_syndrome(x_err)
                corr_x = self.code.decode_x(syn_z)
                corr_z = self.code.decode_z(syn_x)
                lx, lz = self.code.logical_error(x_err, z_err, corr_x, corr_z)
                if lx or lz:
                    n_logical_errors += 1
        else:
            raise QECError(f"Unsupported code type: {type(self.code).__name__}")

        return n_logical_errors / n_trials

    def threshold_sweep(
        self,
        p_range: tuple[float, float] = (0.001, 0.15),
        n_points: int = 12,
        n_trials: int = 5000,
        seed: int | None = 0,
        error_model: str = "depolarizing",
    ) -> ThresholdResult:
        """Sweep physical error rates and return logical error rate curve.

        Parameters
        ----------
        p_range:
            (p_min, p_max) range of physical error rates to sweep.
        n_points:
            Number of sample points.
        n_trials:
            Monte Carlo trials per point.
        seed:
            Random seed.
        error_model:
            See `logical_error_rate`.

        Returns
        -------
        ThresholdResult with p_phys, p_logical, and estimated threshold.
        """
        p_min, p_max = p_range
        p_values = [
            p_min + (p_max - p_min) * i / (n_points - 1)
            for i in range(n_points)
        ]
        p_logical = []
        for i, p in enumerate(p_values):
            p_L = self.logical_error_rate(
                p, n_trials=n_trials,
                seed=None if seed is None else seed + i,
                error_model=error_model,
            )
            p_logical.append(p_L)

        # Estimate threshold: largest p where p_logical < p_phys
        threshold = None
        for p, pL in zip(p_values, p_logical):
            if pL < p:
                threshold = p

        return ThresholdResult(
            p_phys=p_values,
            p_logical=p_logical,
            threshold=threshold,
            code_distances=[self.code.d if hasattr(self.code, "d") else self.code.n],
        )


# ── MWPMDecoder ───────────────────────────────────────────────────────────────

def _mwpm_bitmask(
    costs: "dict[tuple[int, int], float]",
    n: int,
) -> "tuple[float, list[tuple[int, int]]]":
    """Bitmask DP minimum-weight perfect matching on *n* nodes (n must be even).

    Parameters
    ----------
    costs:
        Mapping (i, j) or (j, i) → edge weight for any pair of nodes.
    n:
        Total number of nodes.  All must be matched.

    Returns
    -------
    (min_cost, matching)  where *matching* is a list of (i, j) pairs.

    Complexity
    ----------
    O(2ⁿ · n) time and space.  Feasible for n ≤ 24 (≈ 12 actual defects).
    """
    if n == 0:
        return 0.0, []

    memo: "dict[int, tuple[float, int]]" = {}  # bitmask → (min_cost, first_node's partner)

    def _dp(mask: int) -> "tuple[float, int]":
        if mask == 0:
            return 0.0, -1
        if mask in memo:
            return memo[mask]
        # Lowest set bit = first unmatched node
        lsb = mask & (-mask)
        first = lsb.bit_length() - 1
        remaining = mask ^ lsb
        best_cost: float = float("inf")
        best_partner: int = -1
        tmp = remaining
        while tmp:
            j_lsb = tmp & (-tmp)
            partner = j_lsb.bit_length() - 1
            c = costs.get((first, partner), costs.get((partner, first), float("inf")))
            sub_cost, _ = _dp(remaining ^ j_lsb)
            total = c + sub_cost
            if total < best_cost:
                best_cost = total
                best_partner = partner
            tmp ^= j_lsb
        memo[mask] = (best_cost, best_partner)
        return best_cost, best_partner

    full_mask = (1 << n) - 1
    _dp(full_mask)

    # Reconstruct matching
    matching: "list[tuple[int, int]]" = []
    mask = full_mask
    while mask:
        lsb = mask & (-mask)
        first = lsb.bit_length() - 1
        _, partner = memo[mask]
        matching.append((first, partner))
        mask ^= lsb
        mask ^= (1 << partner)

    return memo[full_mask][0], matching


def _mwpm_greedy_fallback(
    costs: "dict[tuple[int, int], float]",
    nodes: "list[int]",
) -> "tuple[float, list[tuple[int, int]]]":
    """Greedy nearest-neighbour MWPM (fallback for large syndrome sizes)."""
    remaining = list(nodes)
    total = 0.0
    matching: "list[tuple[int, int]]" = []
    while len(remaining) >= 2:
        bi, bj, bd = 0, 1, float("inf")
        for ii in range(len(remaining)):
            for jj in range(ii + 1, len(remaining)):
                d = costs.get(
                    (remaining[ii], remaining[jj]),
                    costs.get((remaining[jj], remaining[ii]), float("inf")),
                )
                if d < bd:
                    bi, bj, bd = ii, jj, d
        matching.append((remaining[bi], remaining[bj]))
        total += bd
        remaining.pop(bj)
        remaining.pop(bi)
    return total, matching


class MWPMDecoder:
    """Minimum-Weight Perfect Matching decoder for stabilizer codes (M78).

    Uses an exact bitmask-DP MWPM solver for small syndrome sizes and a
    greedy nearest-neighbour fallback for larger ones.  The syndrome graph
    infrastructure from :class:`SurfaceCode` is reused directly, so the
    decoder is compatible with any code that exposes
    ``_build_syndrome_graph``, ``_syndrome_graph_bfs``, ``z_stabs``,
    ``x_stabs``, and ``n``.

    Unlike the built-in greedy decoder in :class:`SurfaceCode`, MWPM finds
    the globally optimal correction (minimum total edge weight), which
    reduces the logical error rate — especially at low physical error rates.

    Parameters
    ----------
    code:
        A :class:`SurfaceCode` or :class:`RepetitionCode` instance.
    max_exact:
        Maximum number of *actual* defects for which the exact bitmask-DP
        solver is used.  The DP operates on ``2 * n_defects`` nodes; the
        default of 10 defects corresponds to 20 DP nodes, which runs in
        ~20 ms.  For larger syndrome sizes the greedy fallback is used.

    Examples
    --------
    >>> code = SurfaceCode(d=5)
    >>> decoder = MWPMDecoder(code)
    >>> x_errors = [0] * 25
    >>> x_errors[3] = 1
    >>> syn = code.z_syndrome(x_errors)
    >>> correction = decoder.decode_x(syn)
    """

    def __init__(
        self,
        code: "SurfaceCode | RepetitionCode",
        max_exact: int = 10,
    ) -> None:
        if max_exact < 1:
            raise QECError(f"max_exact must be >= 1, got {max_exact}")
        self.code = code
        self.max_exact = max_exact

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode(
        self,
        syndrome: "list[int]",
        stabs: "list[list[int]]",
    ) -> "list[int]":
        """Core MWPM decoding on any syndrome + stabilizer list."""
        defects = [i for i, s in enumerate(syndrome) if s]
        n_data = self.code.n
        correction = [0] * n_data

        if not defects:
            return correction

        BOUNDARY = len(stabs)
        adj, _ = self.code._build_syndrome_graph(stabs)
        n_d = len(defects)

        # ── Precompute BFS paths & distances ──────────────────────────
        # Node scheme (MWPM on 2*n_d nodes):
        #   0 .. n_d-1        → actual defects
        #   n_d .. 2*n_d-1    → boundary proxy for defect i (index n_d+i)
        # Cost(i, n_d+i) = BFS dist from defect[i] to BOUNDARY
        # Cost(n_d+i, n_d+j) = 0 (two boundary proxies cancel freely)
        # Cost(i, j) for 0<=i<j<n_d = BFS dist between defects

        costs: "dict[tuple[int, int], float]" = {}
        paths: "dict[tuple[int, int], list[int]]" = {}

        # Defect–defect edges
        for i in range(n_d):
            for j in range(i + 1, n_d):
                p = self.code._syndrome_graph_bfs(defects[i], defects[j], adj)
                d = float(len(p)) if p else float("inf")
                costs[(i, j)] = costs[(j, i)] = d
                paths[(i, j)] = paths[(j, i)] = p

        # Defect–boundary proxy edges
        for i in range(n_d):
            p = self.code._syndrome_graph_bfs(defects[i], BOUNDARY, adj)
            d = float(len(p)) if p else float("inf")
            costs[(i, n_d + i)] = costs[(n_d + i, i)] = d
            paths[(i, n_d + i)] = paths[(n_d + i, i)] = p

        # Boundary proxy–proxy edges (cost 0, no correction)
        for i in range(n_d):
            for j in range(n_d):
                if i != j:
                    costs[(n_d + i, n_d + j)] = 0.0
                    paths[(n_d + i, n_d + j)] = []

        # ── Solve MWPM ────────────────────────────────────────────────
        all_nodes = list(range(2 * n_d))
        n_mwpm = 2 * n_d

        if n_d <= self.max_exact:
            _, matching = _mwpm_bitmask(costs, n_mwpm)
        else:
            _, matching = _mwpm_greedy_fallback(costs, all_nodes)

        # ── Apply corrections ─────────────────────────────────────────
        for u, v in matching:
            p = paths.get((u, v), paths.get((v, u), []))
            for q in p:
                correction[q] ^= 1

        return correction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode_x(self, z_syndrome: "list[int]") -> "list[int]":
        """Decode Z-stabilizer syndrome (X errors) → X correction operator."""
        return self._decode(z_syndrome, self.code.z_stabs)

    def decode_z(self, x_syndrome: "list[int]") -> "list[int]":
        """Decode X-stabilizer syndrome (Z errors) → Z correction operator."""
        return self._decode(x_syndrome, self.code.x_stabs)

    def logical_error_rate(
        self,
        p_phys: float,
        n_trials: int = 5000,
        seed: "int | None" = None,
        error_model: str = "depolarizing",
    ) -> float:
        """Monte Carlo logical error rate using MWPM decoding.

        Compatible with :meth:`LogicalErrorSimulator.logical_error_rate` but
        uses this decoder instead of the built-in greedy one.
        """
        if not 0 < p_phys < 1:
            raise QECError(f"p_phys must be in (0, 1), got {p_phys}")

        code = self.code
        if not isinstance(code, SurfaceCode):
            raise QECError("MWPMDecoder.logical_error_rate only supports SurfaceCode")

        rng = random.Random(seed)
        n_logical_errors = 0
        n = code.n

        for _ in range(n_trials):
            if error_model == "depolarizing":
                x_err = [1 if rng.random() < p_phys else 0 for _ in range(n)]
                z_err = [1 if rng.random() < p_phys else 0 for _ in range(n)]
            else:
                x_err = [1 if rng.random() < p_phys else 0 for _ in range(n)]
                z_err = [0] * n

            syn_z = code.z_syndrome(x_err)
            syn_x = code.x_syndrome(z_err)
            corr_x = self.decode_x(syn_z)
            corr_z = self.decode_z(syn_x)
            lx, lz = code.logical_error(x_err, z_err, corr_x, corr_z)
            if lx or lz:
                n_logical_errors += 1

        return n_logical_errors / n_trials
