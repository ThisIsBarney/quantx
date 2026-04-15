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
    # Top and bottom boundary half-plaquettes
    for c in range(0, d - 1, 2):
        # top boundary (row -1 to row 0)
        stabilizers.append([c, c + 1])
    for c in range(1, d - 1, 2):
        # bottom boundary
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

    # ── Decoder (greedy MWPM) ─────────────────────────────────────────────────

    def _greedy_match(
        self,
        syndrome: list[int],
        stabs: list[list[int]],
        error_type: str,
    ) -> list[int]:
        """Greedy minimum-weight syndrome matching decoder.

        Pairs up syndrome defects (violated stabilizers) by minimum Manhattan
        distance on the syndrome graph and returns a correction operator.

        Parameters
        ----------
        syndrome:
            Binary syndrome vector.
        stabs:
            List of stabilizers (each a list of qubit indices).
        error_type:
            'X' or 'Z' — type of correction to generate.

        Returns
        -------
        Correction: binary array of length n (1 = apply error_type on qubit i).
        """
        # Identify violated stabilizers (defects)
        defects = [i for i, s in enumerate(syndrome) if s]
        correction = [0] * self.n

        if not defects:
            return correction

        # Compute stabilizer centers for distance estimation
        def center(stab_idx: int):
            qubits = stabs[stab_idx]
            r = sum(self._positions[q][0] for q in qubits) / len(qubits)
            c = sum(self._positions[q][1] for q in qubits) / len(qubits)
            return r, c

        # Greedy: repeatedly pair the nearest unpaired defects
        remaining = list(defects)
        while len(remaining) >= 2:
            # Find pair with minimum distance
            best_i, best_j, best_dist = 0, 1, float("inf")
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    ri, ci = center(remaining[i])
                    rj, cj = center(remaining[j])
                    dist = abs(ri - rj) + abs(ci - cj)
                    if dist < best_dist:
                        best_i, best_j, best_dist = i, j, dist

            # Connect the pair by flipping qubits along a path
            si, sj = remaining[best_i], remaining[best_j]
            qi = stabs[si][0]  # representative qubit from each stabilizer
            qj = stabs[sj][0]
            ri, ci = self._positions[qi]
            rj, cj = self._positions[qj]

            # Walk from (ri,ci) to (rj,cj) flipping corrections
            r, c = ri, ci
            while (r, c) != (rj, cj):
                if r != rj:
                    r += 1 if rj > r else -1
                elif c != cj:
                    c += 1 if cj > c else -1
                idx = r * self.d + c
                if 0 <= idx < self.n:
                    correction[idx] ^= 1

            remaining.pop(best_j)
            remaining.pop(best_i)

        # Unpaired defect (odd number): connect to nearest boundary
        if remaining:
            si = remaining[0]
            qi = stabs[si][0]
            ri, ci = self._positions[qi]
            # Walk to nearest boundary (row 0 or row d-1 for Z errors)
            if error_type == "X":
                target_r = 0 if ri < self.d // 2 else self.d - 1
                r = ri
                while r != target_r:
                    r += 1 if target_r > r else -1
                    idx = r * self.d + ci
                    if 0 <= idx < self.n:
                        correction[idx] ^= 1
            else:
                target_c = 0 if ci < self.d // 2 else self.d - 1
                c = ci
                while c != target_c:
                    c += 1 if target_c > c else -1
                    idx = ri * self.d + c
                    if 0 <= idx < self.n:
                        correction[idx] ^= 1

        return correction

    def decode_x(self, z_syndrome: list[int]) -> list[int]:
        """Decode X-stabilizer syndrome → X correction operator."""
        return self._greedy_match(z_syndrome, self.x_stabs, "X")

    def decode_z(self, x_syndrome: list[int]) -> list[int]:
        """Decode Z-stabilizer syndrome → Z correction operator."""
        return self._greedy_match(x_syndrome, self.z_stabs, "Z")

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
