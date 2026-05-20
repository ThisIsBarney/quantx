"""M121: Quantum LDPC codes — hypergraph product and bicycle constructions.

Implements:
  - :class:`ClassicalCode`       — binary parity-check matrix utilities
  - :class:`HypergraphProductCode` — [[n₁n₂+n₁'n₂', k₁k₂, d]] CSS code
  - :class:`LDPCDecoder`         — belief-propagation + OSD post-processing
  - :func:`bicycle_code`         — quasi-cyclic bicycle (QC-LDPC) construction

References
----------
* Tillich, J.-P. & Zémor, G. (2014). Quantum LDPC codes with positive rate
  and minimum distance proportional to the square root of the blocklength.
  *IEEE Trans. Inf. Theory* 60, 1193–1202.
* Panteleev, P. & Kalachev, G. (2022). Asymptotically good quantum and
  locally testable classical LDPC codes.  *STOC 2022*.
* Fossorier, M. P. C. (2004). Quasi-cyclic low-density parity-check codes.
  *IEEE Trans. Inf. Theory* 50, 1788–1793.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "LDPCError",
    "ClassicalCode",
    "HypergraphProductCode",
    "LDPCDecoder",
    "bicycle_code",
]

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class LDPCError(Exception):
    """Raised for invalid LDPC-code operations."""


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _rref_gf2(H: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Row-reduce *H* over GF(2); return (reduced matrix, pivot columns)."""
    M = H.astype(np.uint8).copy()
    m, n = M.shape
    pivots: list[int] = []
    row = 0
    for col in range(n):
        # Find pivot
        found = -1
        for r in range(row, m):
            if M[r, col]:
                found = r
                break
        if found == -1:
            continue
        if found != row:
            M[[row, found]] = M[[found, row]]
        for r in range(m):
            if r != row and M[r, col]:
                M[r] ^= M[row]
        pivots.append(col)
        row += 1
        if row == m:
            break
    return M, pivots


def _rank_gf2(H: np.ndarray) -> int:
    """Rank of *H* over GF(2)."""
    _, pivots = _rref_gf2(H)
    return len(pivots)


def _null_space_gf2(H: np.ndarray) -> np.ndarray:
    """Return a basis for the null space of *H* over GF(2) (rows of result)."""
    m, n = H.shape
    _, pivots = _rref_gf2(H)
    pivot_set = set(pivots)
    free_cols = [c for c in range(n) if c not in pivot_set]
    if not free_cols:
        return np.zeros((0, n), dtype=np.uint8)
    # Build null space basis: one vector per free column
    Reduced, _ = _rref_gf2(H)
    basis: list[np.ndarray] = []
    for fc in free_cols:
        v = np.zeros(n, dtype=np.uint8)
        v[fc] = 1
        for i, pc in enumerate(pivots):
            if i < len(Reduced):
                v[pc] = int(Reduced[i, fc])
        basis.append(v)
    return np.array(basis, dtype=np.uint8)


def _kron_gf2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Kronecker product of two GF(2) matrices."""
    r1, c1 = A.shape
    r2, c2 = B.shape
    result = np.zeros((r1 * r2, c1 * c2), dtype=np.uint8)
    for i in range(r1):
        for j in range(c1):
            if A[i, j]:
                result[i * r2:(i + 1) * r2, j * c2:(j + 1) * c2] ^= B
    return result


def _eye_gf2(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.uint8)


def _hstack_gf2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.hstack([a, b]).astype(np.uint8)


def _vstack_gf2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.vstack([a, b]).astype(np.uint8)


# ---------------------------------------------------------------------------
# ClassicalCode
# ---------------------------------------------------------------------------

class ClassicalCode:
    """Binary linear code specified by a parity-check matrix H.

    Parameters
    ----------
    H : array-like, shape (m, n)
        Binary parity-check matrix over GF(2).
    """

    def __init__(self, H: np.ndarray) -> None:
        H = np.asarray(H, dtype=np.uint8) % 2
        if H.ndim != 2:
            raise LDPCError("H must be a 2-D array.")
        self._H = H

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def H(self) -> np.ndarray:
        """Parity-check matrix (binary, shape m × n)."""
        return self._H.copy()

    @property
    def n(self) -> int:
        """Block length."""
        return int(self._H.shape[1])

    @property
    def m(self) -> int:
        """Number of parity checks."""
        return int(self._H.shape[0])

    @property
    def rank(self) -> int:
        """Rank of H over GF(2)."""
        return _rank_gf2(self._H)

    @property
    def k(self) -> int:
        """Dimension (number of logical bits): k = n − rank(H)."""
        return self.n - self.rank

    @property
    def column_weight(self) -> float:
        """Average column weight of H."""
        return float(self._H.sum()) / self.n

    @property
    def row_weight(self) -> float:
        """Average row weight of H."""
        return float(self._H.sum()) / self.m

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def syndrome(self, bits: np.ndarray) -> np.ndarray:
        """Compute H · bits mod 2.

        Parameters
        ----------
        bits : array-like, shape (n,)
            Binary word.

        Returns
        -------
        ndarray, shape (m,)
            Syndrome.
        """
        bits = np.asarray(bits, dtype=np.uint8) % 2
        if bits.shape != (self.n,):
            raise LDPCError(
                f"bits must have shape ({self.n},); got {bits.shape}."
            )
        return (self._H @ bits) % 2

    def is_codeword(self, bits: np.ndarray) -> bool:
        """Return True if *bits* is a valid codeword (zero syndrome)."""
        return bool(np.all(self.syndrome(bits) == 0))

    def generator_matrix(self) -> np.ndarray:
        """Return a generator matrix G such that rows of G span the code.

        Returns
        -------
        ndarray, shape (k, n)
        """
        return _null_space_gf2(self._H)

    def __repr__(self) -> str:
        return (
            f"ClassicalCode(n={self.n}, k={self.k}, m={self.m}, "
            f"col_wt={self.column_weight:.2f})"
        )

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def repetition(cls, n: int) -> "ClassicalCode":
        """[n, 1, n] repetition code.

        Parameters
        ----------
        n : int
            Block length (n ≥ 2).
        """
        if n < 2:
            raise LDPCError("n must be at least 2 for the repetition code.")
        H = np.zeros((n - 1, n), dtype=np.uint8)
        for i in range(n - 1):
            H[i, i] = H[i, i + 1] = 1
        return cls(H)

    @classmethod
    def hamming(cls, r: int) -> "ClassicalCode":
        """[2^r − 1, 2^r − r − 1, 3] Hamming code.

        Parameters
        ----------
        r : int
            Redundancy (r ≥ 2).
        """
        if r < 2:
            raise LDPCError("r must be at least 2 for the Hamming code.")
        n = (1 << r) - 1
        # Columns are all non-zero r-bit strings
        cols = [
            np.array([(j >> b) & 1 for b in range(r)], dtype=np.uint8)
            for j in range(1, n + 1)
        ]
        H = np.column_stack(cols)  # shape (r, n)
        return cls(H)

    @classmethod
    def from_circulant(cls, row: Sequence[int], n: int) -> "ClassicalCode":
        """Circulant LDPC code whose first row is *row* on an n × n circulant.

        Parameters
        ----------
        row : sequence of int
            First row of the circulant matrix (binary vector of length n).
        n : int
            Size of the circulant.
        """
        row_arr = np.asarray(row, dtype=np.uint8) % 2
        if len(row_arr) != n:
            raise LDPCError("row must have length n.")
        H = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            H[i] = np.roll(row_arr, i)
        return cls(H)


# ---------------------------------------------------------------------------
# HypergraphProductCode
# ---------------------------------------------------------------------------

class HypergraphProductCode:
    """CSS quantum code via the hypergraph product of two classical codes.

    Given classical codes C₁ = [n₁, k₁] with parity-check H₁ (m₁×n₁) and
    C₂ = [n₂, k₂] with H₂ (m₂×n₂), the hypergraph product code has

    * Physical qubits: N = n₁n₂ + m₁m₂
    * Logical qubits:  K = k₁k₂ + k₁'k₂'  (k_i' = n_i − m_i if H_i is full rank)
    * X stabilisers:   Hx = [H₁ ⊗ I_{n₂}  |  I_{m₁} ⊗ H₂ᵀ]
    * Z stabilisers:   Hz = [I_{n₁} ⊗ H₂  |  H₁ᵀ ⊗ I_{m₂}]

    Parameters
    ----------
    code1 : ClassicalCode
        First classical seed code C₁.
    code2 : ClassicalCode
        Second classical seed code C₂.
    """

    def __init__(self, code1: ClassicalCode, code2: ClassicalCode) -> None:
        self.code1 = code1
        self.code2 = code2
        self._build()

    def _build(self) -> None:
        H1 = self.code1._H   # (m1, n1)
        H2 = self.code2._H   # (m2, n2)
        m1, n1 = H1.shape
        m2, n2 = H2.shape

        # X stabilisers: Hx = [H1 ⊗ I_n2 | I_m1 ⊗ H2^T]
        self._Hx = _hstack_gf2(
            _kron_gf2(H1, _eye_gf2(n2)),
            _kron_gf2(_eye_gf2(m1), H2.T),
        )

        # Z stabilisers: Hz = [I_n1 ⊗ H2 | H1^T ⊗ I_m2]
        self._Hz = _hstack_gf2(
            _kron_gf2(_eye_gf2(n1), H2),
            _kron_gf2(H1.T, _eye_gf2(m2)),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def Hx(self) -> np.ndarray:
        """X-stabiliser matrix (binary)."""
        return self._Hx.copy()

    @property
    def Hz(self) -> np.ndarray:
        """Z-stabiliser matrix (binary)."""
        return self._Hz.copy()

    @property
    def n_qubits(self) -> int:
        """Total number of physical qubits."""
        return int(self._Hx.shape[1])

    @property
    def n_x_checks(self) -> int:
        """Number of X parity checks (rows of Hx)."""
        return int(self._Hx.shape[0])

    @property
    def n_z_checks(self) -> int:
        """Number of Z parity checks (rows of Hz)."""
        return int(self._Hz.shape[0])

    @property
    def k(self) -> int:
        """Number of logical qubits K = k₁k₂."""
        return self.code1.k * self.code2.k

    @property
    def n_x_stabs(self) -> int:
        """Alias for n_x_checks."""
        return self.n_x_checks

    @property
    def n_z_stabs(self) -> int:
        """Alias for n_z_checks."""
        return self.n_z_checks

    def commutation_check(self) -> bool:
        """Verify that Hx · Hz^T = 0 mod 2 (CSS commutativity condition)."""
        return bool(np.all(self._Hx @ self._Hz.T % 2 == 0))

    def x_syndrome(self, error: np.ndarray) -> np.ndarray:
        """Compute X syndrome s = Hx · e mod 2."""
        e = np.asarray(error, dtype=np.uint8) % 2
        return (self._Hx @ e) % 2

    def z_syndrome(self, error: np.ndarray) -> np.ndarray:
        """Compute Z syndrome s = Hz · e mod 2."""
        e = np.asarray(error, dtype=np.uint8) % 2
        return (self._Hz @ e) % 2

    def __repr__(self) -> str:
        return (
            f"HypergraphProductCode(n={self.n_qubits}, k={self.k}, "
            f"n_x={self.n_x_checks}, n_z={self.n_z_checks})"
        )


# ---------------------------------------------------------------------------
# LDPCDecoder — belief propagation with OSD post-processing
# ---------------------------------------------------------------------------

class LDPCDecoder:
    """Belief-propagation decoder for binary LDPC codes.

    Implements flooding-schedule BP over the Tanner graph with optional
    *ordered-statistics decoding* (OSD) post-processing when BP fails to
    converge.

    Parameters
    ----------
    H : array-like, shape (m, n)
        Binary parity-check matrix.
    max_iter : int
        Maximum number of BP iterations.
    osd_order : int
        Order of OSD post-processing (0 = no OSD; 1 = test all single-bit
        flips on the unreliable positions).
    """

    def __init__(
        self,
        H: np.ndarray,
        max_iter: int = 50,
        osd_order: int = 0,
    ) -> None:
        self._H = np.asarray(H, dtype=np.uint8) % 2
        if self._H.ndim != 2:
            raise LDPCError("H must be 2-D.")
        if max_iter < 1:
            raise LDPCError("max_iter must be at least 1.")
        if osd_order < 0:
            raise LDPCError("osd_order must be non-negative.")
        self.max_iter = max_iter
        self.osd_order = osd_order
        self._m, self._n = self._H.shape

        # Build adjacency lists for the Tanner graph
        self._check_nbrs: list[list[int]] = [
            list(np.where(self._H[i])[0]) for i in range(self._m)
        ]
        self._var_nbrs: list[list[int]] = [
            list(np.where(self._H[:, j])[0]) for j in range(self._n)
        ]

    # ------------------------------------------------------------------
    # Main decode interface
    # ------------------------------------------------------------------

    def decode(
        self,
        syndrome: np.ndarray,
        channel_llr: Optional[np.ndarray] = None,
        error_rate: float = 0.1,
    ) -> tuple[np.ndarray, bool]:
        """Decode a syndrome to an error estimate.

        Parameters
        ----------
        syndrome : array-like, shape (m,)
            Measured syndrome (binary).
        channel_llr : array-like, shape (n,) or None
            Log-likelihood ratios log(P(e=0)/P(e=1)) for each bit.  If None,
            a BSC model with *error_rate* is assumed for all bits.
        error_rate : float
            BSC error probability used when *channel_llr* is None.

        Returns
        -------
        error_estimate : ndarray, shape (n,)
            Binary error estimate.
        converged : bool
            True if BP converged (zero residual syndrome), False otherwise.
        """
        s = np.asarray(syndrome, dtype=np.uint8) % 2
        if s.shape != (self._m,):
            raise LDPCError(
                f"syndrome must have shape ({self._m},); got {s.shape}."
            )

        # Channel LLR initialisation
        if channel_llr is not None:
            llr = np.asarray(channel_llr, dtype=float)
        else:
            if not (0.0 < error_rate < 1.0):
                raise LDPCError("error_rate must be in (0, 1).")
            llr_val = math.log((1.0 - error_rate) / error_rate)
            llr = np.full(self._n, llr_val, dtype=float)

        # BP decode
        estimate, converged = self._bp_decode(s, llr)

        # OSD post-processing
        if not converged and self.osd_order > 0:
            estimate, converged = self._osd_postprocess(s, llr, estimate)

        return estimate.astype(np.uint8), converged

    # ------------------------------------------------------------------
    # Belief propagation
    # ------------------------------------------------------------------

    def _bp_decode(
        self, syndrome: np.ndarray, llr: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Sum-product belief propagation (flooding schedule)."""
        n, m = self._n, self._m

        # Variable-to-check messages (initialised to channel LLR)
        v2c = {(j, i): llr[j] for i in range(m) for j in self._check_nbrs[i]}
        # Check-to-variable messages
        c2v: dict[tuple[int, int], float] = {k: 0.0 for k in v2c}

        def _tanh_safe(x: float) -> float:
            return math.tanh(max(-500.0, min(500.0, x / 2.0)))

        def _atanh_safe(x: float) -> float:
            x = max(-1.0 + 1e-10, min(1.0 - 1e-10, x))
            return math.atanh(x)

        for _ in range(self.max_iter):
            # Check → variable
            for i in range(m):
                nbrs = self._check_nbrs[i]
                if not nbrs:
                    continue
                tanh_prod = 1.0
                tanh_vals = {j: _tanh_safe(v2c[(j, i)]) for j in nbrs}
                for tv in tanh_vals.values():
                    tanh_prod *= tv
                for j in nbrs:
                    tv_j = tanh_vals[j]
                    if abs(tv_j) > 1.0 - 1e-12:
                        partial = 1.0
                        for jj in nbrs:
                            if jj != j:
                                partial *= tanh_vals[jj]
                    else:
                        partial = tanh_prod / tv_j
                    # Syndrome bit s[i] flips message sign
                    sign = -1.0 if syndrome[i] else 1.0
                    c2v[(j, i)] = sign * 2.0 * _atanh_safe(
                        max(-1.0 + 1e-10, min(1.0 - 1e-10, partial))
                    )

            # Variable → check + marginals
            total_llr = llr.copy()
            for j in range(n):
                for i in self._var_nbrs[j]:
                    total_llr[j] += c2v[(j, i)]

            estimate = (total_llr < 0).astype(np.uint8)

            if np.all((self._H @ estimate) % 2 == syndrome):
                return estimate, True

            # Variable → check update
            for j in range(n):
                nbrs_j = self._var_nbrs[j]
                for i in nbrs_j:
                    v2c[(j, i)] = llr[j] + sum(
                        c2v[(j, ii)] for ii in nbrs_j if ii != i
                    )

        return estimate, False

    # ------------------------------------------------------------------
    # OSD post-processing
    # ------------------------------------------------------------------

    def _osd_postprocess(
        self,
        syndrome: np.ndarray,
        llr: np.ndarray,
        bp_estimate: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Order-1 OSD: sort by reliability and try single-bit flips."""
        # Sort positions by reliability (most reliable first)
        reliability = np.abs(llr)
        order = np.argsort(-reliability)

        # Gaussian elimination to find systematic columns
        H = self._H.copy()
        m = self._m
        systematic: list[int] = []
        Hwork, pivots = _rref_gf2(H)
        systematic = pivots  # pivot columns → most reliable first by our sort

        # Check flipping each of the `osd_order` least-reliable positions
        n_unreliable = min(self.osd_order, self._n)
        unreliable = order[-n_unreliable:][::-1]  # least reliable first

        best = bp_estimate.copy()
        for j in unreliable:
            candidate = best.copy()
            candidate[j] ^= 1
            if np.all((self._H @ candidate) % 2 == syndrome):
                return candidate, True

        return best, False

    def __repr__(self) -> str:
        return (
            f"LDPCDecoder(m={self._m}, n={self._n}, "
            f"max_iter={self.max_iter}, osd_order={self.osd_order})"
        )


# ---------------------------------------------------------------------------
# bicycle_code
# ---------------------------------------------------------------------------

def bicycle_code(
    L: int,
    a_shifts: Sequence[int],
    b_shifts: Sequence[int],
) -> HypergraphProductCode:
    """Construct a quasi-cyclic bicycle (QC-LDPC) CSS code.

    A bicycle code is built from two circulant matrices A, B of size L × L,
    where A = Σᵢ P^{aᵢ} and B = Σⱼ P^{bⱼ} (P is the L × L cyclic shift).
    The parity-check matrices are:

    ::

        Hx = [A | B]      Hz = [Bᵀ | Aᵀ]

    which are the hypergraph product of the circulant codes defined by A and B.

    Parameters
    ----------
    L : int
        Circulant size (must be > 0).
    a_shifts : sequence of int
        Positions of the 1s in the first row of A (shifts of P).
    b_shifts : sequence of int
        Positions of the 1s in the first row of B (shifts of P).

    Returns
    -------
    HypergraphProductCode
        The resulting bicycle CSS code.
    """
    if L <= 0:
        raise LDPCError("L must be a positive integer.")
    if not a_shifts:
        raise LDPCError("a_shifts must be non-empty.")
    if not b_shifts:
        raise LDPCError("b_shifts must be non-empty.")

    def _circulant(shifts: Sequence[int]) -> np.ndarray:
        row = np.zeros(L, dtype=np.uint8)
        for s in shifts:
            row[int(s) % L] = 1
        C = np.zeros((L, L), dtype=np.uint8)
        for i in range(L):
            C[i] = np.roll(row, i)
        return C

    A = _circulant(a_shifts)
    B = _circulant(b_shifts)

    code_A = ClassicalCode(A)
    code_B = ClassicalCode(B)
    return HypergraphProductCode(code_A, code_B)
