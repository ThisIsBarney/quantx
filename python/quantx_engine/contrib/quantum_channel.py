"""Quantum channels and steady-state solver (M75).

Two main components:

**QuantumChannel** — a completely-positive trace-preserving (CPTP) map
in the Kraus representation {K_i} with ρ' = Σ_i K_i ρ K_i†.

Multiple representations are available:

* :meth:`choi_matrix` — Choi-Jamiołkowski matrix  J = Σ_n |ψ_n⟩⟨ψ_n|
  where |ψ_n⟩ = vec_F(K_n) (column-major vectorisation).  Size d²×d².
* :meth:`superoperator` — Liouville superoperator L where
  vec_C(ρ') = L vec_C(ρ) (row-major vectorisation).
  L = Σ_n K_n ⊗ K_n*.  Size d²×d².
* :meth:`apply` — direct application Σ_n K_n ρ K_n†.

Factory methods cover common channels: :meth:`identity`,
:meth:`depolarizing`, :meth:`amplitude_damping`, :meth:`dephasing`,
:meth:`bit_flip`, :meth:`phase_flip`, :meth:`reset`.

Channels can be composed (:meth:`compose`) and tensored (:meth:`tensor`).

**SteadyStateSolver** — finds the unique fixed point ρ_ss satisfying
Φ(ρ_ss) = ρ_ss by computing the eigenvector of the superoperator with
eigenvalue 1 (always exists for CPTP maps).

Quick start::

    import numpy as np
    from quantx_engine.contrib.quantum_channel import QuantumChannel, SteadyStateSolver

    # Amplitude damping channel
    ch = QuantumChannel.amplitude_damping(gamma=0.3)
    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)  # |+⟩⟨+|
    rho1 = ch.apply(rho0)

    # Steady state: should be |0⟩⟨0|
    solver = SteadyStateSolver()
    rho_ss = solver.solve(ch)
    print(np.abs(rho_ss - np.array([[1,0],[0,0]])).max())  # ≈ 0

References
----------
.. [1] Choi, M.-D. (1975). Completely positive linear maps on complex
       matrices. *Linear Algebra Appl.* 10, 285–290.
       https://doi.org/10.1016/0024-3795(75)90075-0

.. [2] Kraus, K. (1983). *States, Effects, and Operations*.
       Springer. https://doi.org/10.1007/3-540-12732-1

.. [3] Wolf, M. M. & Cirac, J. I. (2008). Dividing quantum channels.
       *Commun. Math. Phys.* 279, 147.
       https://doi.org/10.1007/s00220-008-0411-y

Physical validation
-------------------
Choi-Kraus representation: a channel is completely positive iff its Choi
matrix is positive semi-definite [Choi1975]. The Choi matrix of the
identity channel I_n is the maximally entangled state |omega><omega| * d,
where d = 2^n and |omega> = (1/sqrt(d)) * sum_i |ii>.
Steady-state solver: Liouvillian L rho_ss = 0 (zero eigenvalue of L).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

__all__ = [
    "ChannelError",
    "QuantumChannel",
    "SteadyStateSolver",
]


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class ChannelError(ValueError):
    """Raised for invalid quantum channel parameters."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _weyl_operator(d: int, m: int, n: int) -> "np.ndarray":
    """Clock-shift Weyl unitary W_{mn}: |j⟩ → e^{2πi·j·m/d} |(j+n) mod d⟩.

    In matrix form: W_{mn}[k, j] = e^{2πi·j·m/d} · δ_{k, (j+n) mod d}.

    The d² operators {W_{mn}, 0 ≤ m,n < d} form an orthogonal basis for
    the space of d×d matrices and are all unitary.
    """
    W = np.zeros((d, d), dtype=complex)
    for j in range(d):
        k = (j + n) % d
        W[k, j] = np.exp(2j * np.pi * j * m / d)
    return W


def _require_numpy() -> None:
    if not _HAS_NUMPY:
        raise ChannelError("numpy is required for QuantumChannel")


# ---------------------------------------------------------------------------
# QuantumChannel
# ---------------------------------------------------------------------------

class QuantumChannel:
    """A completely-positive trace-preserving (CPTP) quantum channel.

    Represented by a list of Kraus operators {K_i} satisfying
    Σ_i K_i† K_i = I.  The channel acts on density matrices as
    ρ → Σ_i K_i ρ K_i†.

    Parameters
    ----------
    kraus_ops:
        List of (d×d) complex Kraus operators.  At least one required.
        The CPTP condition Σ K_i† K_i = I is verified by
        :meth:`is_trace_preserving`.

    Raises
    ------
    ChannelError
        If ``kraus_ops`` is empty, contains non-square or inconsistent
        shapes, or numpy is not available.
    """

    def __init__(self, kraus_ops: "list[np.ndarray]"):
        _require_numpy()
        if not kraus_ops:
            raise ChannelError("kraus_ops must contain at least one operator")
        ops = [np.asarray(K, dtype=complex) for K in kraus_ops]
        d = ops[0].shape[0]
        for i, K in enumerate(ops):
            if K.ndim != 2 or K.shape[0] != K.shape[1]:
                raise ChannelError(
                    f"Kraus operator {i} is not square: shape {K.shape}"
                )
            if K.shape[0] != d:
                raise ChannelError(
                    f"Kraus operator {i} has shape {K.shape}, expected ({d}, {d})"
                )
        self.kraus_ops: "list[np.ndarray]" = ops

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Local Hilbert-space dimension d."""
        return self.kraus_ops[0].shape[0]

    @property
    def n_kraus(self) -> int:
        """Number of Kraus operators."""
        return len(self.kraus_ops)

    # ------------------------------------------------------------------
    # Factory methods — specific channels
    # ------------------------------------------------------------------

    @classmethod
    def from_kraus(
        cls, ops: "list[np.ndarray] | np.ndarray"
    ) -> "QuantumChannel":
        """Construct from an explicit list (or array) of Kraus operators.

        Parameters
        ----------
        ops:
            Iterable of (d×d) arrays, or a (n_kraus, d, d) array.
        """
        _require_numpy()
        arr = np.asarray(ops)
        if arr.ndim == 3:
            return cls([arr[i] for i in range(arr.shape[0])])
        return cls(list(ops))

    @classmethod
    def from_choi(
        cls, choi: "np.ndarray", tol: float = 1e-10
    ) -> "QuantumChannel":
        """Construct from the Choi-Jamiołkowski matrix.

        The Choi matrix J must be (d²×d²), Hermitian, and positive
        semi-definite (for a CP channel).  Eigendecomposition yields
        Kraus operators K_n = √λ_n · reshape_F(ψ_n).

        Parameters
        ----------
        choi:
            Choi matrix, shape (d², d²).  Must be PSD and Hermitian.
        tol:
            Threshold below which eigenvalues are considered zero.

        Raises
        ------
        ChannelError
            If the matrix is not square-dimension or has no positive
            eigenvalues.
        """
        _require_numpy()
        J = np.asarray(choi, dtype=complex)
        if J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ChannelError(f"Choi matrix must be square, got shape {J.shape}")
        d2 = J.shape[0]
        d = int(round(d2 ** 0.5))
        if d * d != d2:
            raise ChannelError(
                f"Choi matrix size {d2} is not a perfect square; "
                "expected d² for integer d"
            )
        # Symmetrize to handle small numerical asymmetries
        J_herm = (J + J.conj().T) / 2
        evals, evecs = np.linalg.eigh(J_herm)
        ops = []
        for i, lam in enumerate(evals):
            if lam < -tol:
                raise ChannelError(
                    f"Choi matrix has eigenvalue {lam:.3e} < −{tol:.3e}; "
                    "the input matrix is not completely positive (not a valid "
                    "quantum channel). Provide a PSD Choi matrix."
                )
            if lam > tol:
                psi = evecs[:, i] * np.sqrt(lam)
                # K = psi reshaped in column-major (Fortran) order
                K = psi.reshape(d, d, order="F")
                ops.append(K)
        if not ops:
            raise ChannelError(
                "No positive eigenvalues found in Choi matrix; "
                "the channel may not be completely positive"
            )
        return cls(ops)

    @classmethod
    def identity(cls, d: "int | None" = None, *, n_qubits: "int | None" = None) -> "QuantumChannel":
        """Identity channel Φ(ρ) = ρ, one Kraus operator K = I_d.

        Parameters
        ----------
        d : int
            Hilbert-space dimension.
        n_qubits : int
            Alternative to ``d``; sets ``d = 2**n_qubits``.
        """
        _require_numpy()
        if n_qubits is not None and d is None:
            d = 1 << n_qubits
        if d is None:
            raise ChannelError("d (or n_qubits) must be provided")
        return cls([np.eye(d, dtype=complex)])

    @classmethod
    def depolarizing(cls, d: "int | None" = None, p: float = 0.0, *, n_qubits: "int | None" = None) -> "QuantumChannel":
        """Depolarizing channel Φ(ρ) = (1−p)ρ + p·I/d.

        Uses the Weyl-operator Kraus decomposition:

            K_{(0,0)} = √(1 − p + p/d²) · I,
            K_{(m,n)} = √(p/d²) · W_{mn}  for (m,n) ≠ (0,0).

        Valid for p ∈ [0, 1].  At p = 1, every state maps to I/d.

        Parameters
        ----------
        d:
            Hilbert-space dimension (d ≥ 2).
        p:
            Depolarizing probability ∈ [0, 1].
        """
        _require_numpy()
        if n_qubits is not None and d is None:
            d = 1 << n_qubits
        if d is None:
            raise ChannelError("d (or n_qubits) must be provided")
        if d < 2:
            raise ChannelError(f"d must be >= 2, got {d}")
        if not 0.0 <= p <= 1.0:
            raise ChannelError(f"p must be in [0, 1], got {p}")
        p_each = p / d ** 2          # weight on each Weyl term
        p_id = 1.0 - (d ** 2 - 1) * p_each   # = 1 - p + p/d²
        ops = []
        for m in range(d):
            for n in range(d):
                weight = p_id if (m == 0 and n == 0) else p_each
                if weight > 0.0:
                    W = _weyl_operator(d, m, n)
                    ops.append(np.sqrt(weight) * W)
        return cls(ops)

    @classmethod
    def amplitude_damping(cls, gamma: float) -> "QuantumChannel":
        """Qubit amplitude-damping channel with decay probability γ.

        Kraus operators::

            K₀ = [[1, 0], [0, √(1−γ)]],  K₁ = [[0, √γ], [0, 0]].

        The steady state is |0⟩⟨0| (ground state).

        Parameters
        ----------
        gamma:
            Decay probability ∈ [0, 1].  At γ = 1, ρ → |0⟩⟨0|
            completely.
        """
        _require_numpy()
        if not 0.0 <= gamma <= 1.0:
            raise ChannelError(f"gamma must be in [0, 1], got {gamma}")
        K0 = np.array([[1, 0], [0, np.sqrt(1.0 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        return cls([K0, K1])

    @classmethod
    def dephasing(cls, p: float, d: int = 2) -> "QuantumChannel":
        """Dephasing (phase-damping) channel.

        Diagonal elements are unchanged; off-diagonal elements are
        multiplied by (1−p).

        For d = 2 uses K₀ = √(1−p/2) I, K₁ = √(p/2) Z (standard form
        that preserves |0⟩⟨0| and |1⟩⟨1|).

        For d > 2 uses K₀ = √(1−p) I, Kᵢ = √p |i⟩⟨i| (i = 0…d−1).

        Parameters
        ----------
        p:
            Dephasing probability ∈ [0, 1].
        d:
            Hilbert-space dimension (default 2).
        """
        _require_numpy()
        if not 0.0 <= p <= 1.0:
            raise ChannelError(f"p must be in [0, 1], got {p}")
        if d < 2:
            raise ChannelError(f"d must be >= 2, got {d}")
        if d == 2:
            K0 = np.sqrt(1.0 - p / 2.0) * np.eye(2, dtype=complex)
            K1 = np.sqrt(p / 2.0) * np.array([[1, 0], [0, -1]], dtype=complex)
            return cls([K0, K1])
        else:
            ops = []
            # Only include the identity term when p < 1 (at p=1 it would be
            # the zero operator, which is non-standard and wastes computation).
            if p < 1.0:
                ops.append(np.sqrt(1.0 - p) * np.eye(d, dtype=complex))
            for i in range(d):
                Ki = np.zeros((d, d), dtype=complex)
                Ki[i, i] = np.sqrt(p)
                ops.append(Ki)
            return cls(ops)

    @classmethod
    def bit_flip(cls, p: float) -> "QuantumChannel":
        """Qubit bit-flip channel: applies X with probability p.

        K₀ = √(1−p) I,  K₁ = √p X.
        """
        _require_numpy()
        if not 0.0 <= p <= 1.0:
            raise ChannelError(f"p must be in [0, 1], got {p}")
        K0 = np.sqrt(1.0 - p) * np.eye(2, dtype=complex)
        K1 = np.sqrt(p) * np.array([[0, 1], [1, 0]], dtype=complex)
        return cls([K0, K1])

    @classmethod
    def phase_flip(cls, p: float) -> "QuantumChannel":
        """Qubit phase-flip channel: applies Z with probability p.

        K₀ = √(1−p) I,  K₁ = √p Z.
        """
        _require_numpy()
        if not 0.0 <= p <= 1.0:
            raise ChannelError(f"p must be in [0, 1], got {p}")
        K0 = np.sqrt(1.0 - p) * np.eye(2, dtype=complex)
        K1 = np.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=complex)
        return cls([K0, K1])

    @classmethod
    def reset(cls, d: int = 2) -> "QuantumChannel":
        """Reset channel: maps any state to |0⟩⟨0|.

        Kraus operators: Kⱼ = |0⟩⟨j|, j = 0…d−1.
        """
        _require_numpy()
        if d < 2:
            raise ChannelError(f"d must be >= 2, got {d}")
        ops = []
        for j in range(d):
            K = np.zeros((d, d), dtype=complex)
            K[0, j] = 1.0
            ops.append(K)
        return cls(ops)

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(self, rho: "np.ndarray") -> "np.ndarray":
        """Apply the channel: ρ → Σ_i K_i ρ K_i†.

        Parameters
        ----------
        rho:
            Input density matrix, shape (d, d).

        Returns
        -------
        np.ndarray
            Output density matrix, shape (d, d), dtype complex128.
        """
        rho_arr = np.asarray(rho, dtype=complex)
        result = np.zeros_like(rho_arr)
        for K in self.kraus_ops:
            result += K @ rho_arr @ K.conj().T
        return result

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------

    def superoperator(self) -> "np.ndarray":
        """Liouville superoperator L (d²×d²) in row-major vectorisation.

        Satisfies ``rho'.ravel() == L @ rho.ravel()`` where ravel uses
        the default NumPy C (row-major) order.

        Computed as L = Σ_n K_n ⊗ K_n*.
        """
        d = self.dim
        L = np.zeros((d ** 2, d ** 2), dtype=complex)
        for K in self.kraus_ops:
            L += np.kron(K, K.conj())
        return L

    def choi_matrix(self) -> "np.ndarray":
        """Choi-Jamiołkowski matrix J (d²×d²).

        Computed as J = Σ_n |ψ_n⟩⟨ψ_n| where
        |ψ_n⟩ = K_n.ravel(order='F') (column-major vectorisation of K_n).

        Properties of J for a CPTP channel:

        * J is positive semi-definite (completely positive).
        * Tr(J) = d (trace-preserving).
        * The partial trace Tr_B(J) = I_A/d (for a unital channel).
        """
        d = self.dim
        J = np.zeros((d ** 2, d ** 2), dtype=complex)
        for K in self.kraus_ops:
            psi = K.ravel(order="F")
            J += np.outer(psi, psi.conj())
        return J

    # ------------------------------------------------------------------
    # Composition and tensor product
    # ------------------------------------------------------------------

    def compose(self, other: "QuantumChannel") -> "QuantumChannel":
        """Return the composed channel other(self(ρ)).

        Apply *self* first, *other* second.  Kraus operators of the
        composed channel are {other.K_j @ self.K_i} for all pairs (i,j).

        Parameters
        ----------
        other:
            Channel to apply after *self*.  Must have the same ``dim``.

        Raises
        ------
        ChannelError
            If dimensions don't match.
        """
        if self.dim != other.dim:
            raise ChannelError(
                f"Cannot compose channels of different dimensions "
                f"({self.dim} vs {other.dim})"
            )
        ops = [K2 @ K1 for K2 in other.kraus_ops for K1 in self.kraus_ops]
        return QuantumChannel(ops)

    def tensor(self, other: "QuantumChannel") -> "QuantumChannel":
        """Tensor-product channel Φ_self ⊗ Φ_other.

        Acts on a bipartite system A⊗B:
        (Φ_A ⊗ Φ_B)(ρ) = Σ_{i,j} (K_A^i ⊗ K_B^j) ρ (K_A^i ⊗ K_B^j)†.

        Kraus operators: {K_self^i ⊗ K_other^j} for all pairs (i,j).
        """
        ops = [
            np.kron(K1, K2)
            for K1 in self.kraus_ops
            for K2 in other.kraus_ops
        ]
        return QuantumChannel(ops)

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def is_trace_preserving(self, tol: float = 1e-10) -> bool:
        """Check Σ_i K_i† K_i = I (trace-preserving condition)."""
        d = self.dim
        check = sum(K.conj().T @ K for K in self.kraus_ops)
        return bool(np.allclose(check, np.eye(d), atol=tol))

    def is_unital(self, tol: float = 1e-10) -> bool:
        """Check Σ_i K_i K_i† = I (unital condition: maps I to I)."""
        d = self.dim
        check = sum(K @ K.conj().T for K in self.kraus_ops)
        return bool(np.allclose(check, np.eye(d), atol=tol))

    def is_completely_positive(self, tol: float = 1e-10) -> bool:
        """Check that the Choi matrix is positive semi-definite."""
        J = self.choi_matrix()
        evals = np.linalg.eigvalsh((J + J.conj().T) / 2)
        return bool(np.all(evals >= -tol))

    def __repr__(self) -> str:  # pragma: no cover
        return f"QuantumChannel(dim={self.dim}, n_kraus={self.n_kraus})"


# ---------------------------------------------------------------------------
# SteadyStateSolver
# ---------------------------------------------------------------------------

class SteadyStateSolver:
    """Find a fixed point ρ_ss of a quantum channel: Φ(ρ_ss) = ρ_ss.

    Algorithm:

    1. Build the Liouville superoperator L = Σ K_n ⊗ K_n*.
    2. Find eigenvalues/vectors of L.
    3. Among eigenvectors with eigenvalue closest to 1, select the one
       whose reshaped matrix is most positive (smallest negative
       eigenvalue mass).
    4. Project to Hermitian, project to PSD, normalise trace to 1.

    For CPTP maps the eigenvalue 1 is always present; the corresponding
    eigenvector(s) span the set of fixed points.

    .. note::
       For channels with a **degenerate fixed-point subspace** (e.g. the
       identity channel, or p = 0 dephasing) the solver returns *one*
       valid fixed-point density matrix.  The result is determined by
       the eigenvector ordering from :func:`numpy.linalg.eig` and is
       not guaranteed to be a specific state.  If you need all fixed
       points, compute the nullspace of (L − I) manually.

    Parameters
    ----------
    tol:
        Threshold for identifying eigenvalues as "equal to 1".
    """

    def __init__(
        self,
        tol: float = 1e-8,
        *,
        H: "np.ndarray | None" = None,
        jump_operators=None,
    ):
        _require_numpy()
        if tol <= 0:
            raise ChannelError(f"tol must be > 0, got {tol}")
        self.tol = tol
        self._H = H
        self._jump_operators = jump_operators

    def _build_lindblad_channel(self) -> "QuantumChannel":
        """Build a QuantumChannel from H and jump_operators using the Lindblad map."""
        H = self._H
        jump_ops = self._jump_operators or []
        d = H.shape[0]

        # Collect Kraus operators of the Lindblad map (first-order Euler step approximation)
        # More accurately, we build the superoperator directly.
        # The Lindblad steady-state superoperator L acts as:
        #   L[rho] = -i[H,rho] + sum_k (L_k rho L_k† - 1/2 {L_k†L_k, rho})
        # We vectorize rho (row-major) → vec(rho) of length d²
        # Then L is a d²×d² matrix.

        d2 = d * d
        I_d = np.eye(d, dtype=complex)

        # Coherent part: -i(H⊗I - I⊗H^T) acting on vec(rho)
        L_super = -1j * (np.kron(H, I_d) - np.kron(I_d, H.T))

        # Dissipative part
        for op in jump_ops:
            # Get the Kraus matrix
            if hasattr(op, 'matrix') and op.matrix is not None:
                Lk = op.matrix.astype(complex)
            elif hasattr(op, 'kind') and op.kind is not None:
                # Build from named channel
                if op.kind == "amplitude_damping":
                    gamma = float(op.rate)
                    Lk = np.sqrt(gamma) * np.array([[0, 1], [0, 0]], dtype=complex)
                elif op.kind == "dephasing":
                    gamma = float(op.rate)
                    Lk = np.sqrt(gamma) * np.array([[1, 0], [0, -1]], dtype=complex) / 2
                else:
                    continue
            else:
                continue

            LkH = Lk.conj().T
            # L_k rho L_k† → (L_k ⊗ L_k*) vec(rho)
            # -1/2 {L_k†L_k, rho} → -1/2 ((L_k†L_k ⊗ I) + (I ⊗ (L_k†L_k)^T)) vec(rho)
            LkHLk = LkH @ Lk
            L_super += np.kron(Lk, Lk.conj())
            L_super -= 0.5 * np.kron(LkHLk, I_d)
            L_super -= 0.5 * np.kron(I_d, LkHLk.T)

        # Build a QuantumChannel whose superoperator IS L_super
        # We wrap it by creating a dummy channel with a solve method
        # Instead, just store and use L_super directly in solve()
        self._L_super = L_super
        self._d = d
        return None  # signal to solve() that L_super is available

    def solve(self, channel: "QuantumChannel | None" = None) -> "np.ndarray":
        """Find the steady-state density matrix.

        Parameters
        ----------
        channel:
            A :class:`QuantumChannel` instance.  May be omitted if *H* and
            *jump_operators* were provided at construction time.

        Returns
        -------
        np.ndarray
            Density matrix ρ_ss of shape (d, d): Hermitian, PSD,
            trace = 1, satisfying Φ(ρ_ss) ≈ ρ_ss.

        Raises
        ------
        ChannelError
            If no physically valid fixed point can be found.
        """
        # ── Lindblad steady-state path ───────────────────────────────────
        if channel is None:
            if self._H is None:
                raise ChannelError(
                    "solve() requires either a 'channel' argument or 'H' and "
                    "'jump_operators' provided at construction time."
                )
            self._build_lindblad_channel()
            L_super = self._L_super
            d = self._d
            # Find null space of L_super (steady state satisfies L[rho] = 0)
            # Use SVD to find null vectors
            _, s, Vh = np.linalg.svd(L_super)
            # Null vector = right singular vector for smallest singular value
            null_vec = Vh[-1].conj()
            rho = null_vec.reshape(d, d)
            # Symmetrize and normalize
            rho = (rho + rho.conj().T) / 2.0
            ev, U = np.linalg.eigh(rho)
            ev = np.maximum(ev, 0.0)
            rho = U @ np.diag(ev) @ U.conj().T
            tr = float(np.real(np.trace(rho)))
            if tr < 1e-14:
                raise ChannelError(
                    "SteadyStateSolver: Lindblad steady-state density matrix has "
                    "near-zero trace."
                )
            return rho / tr

        # ── CPTP fixed-point path (original) ────────────────────────────
        d = channel.dim
        L = channel.superoperator()
        evals, evecs = np.linalg.eig(L)

        # ── Identify eigenvalues near 1 ───────────────────────────────
        dist = np.abs(evals - 1.0)
        candidates = np.where(dist < max(self.tol * 100, 1e-6))[0]
        if len(candidates) == 0:
            # Fall back to the single closest eigenvalue
            candidates = np.array([int(np.argmin(dist))])

        # ── Select the most physical fixed point ─────────────────────
        best_rho: "np.ndarray | None" = None
        best_neg: float = float("inf")

        for idx in candidates:
            v = evecs[:, idx]
            M = v.reshape(d, d)
            # Phase-normalize before symmetrizing: the eigenvector is only
            # defined up to a global complex phase e^{iφ}, which can make
            # (M + M†)/2 ≈ 0 even for a valid ρ_ss (e.g. v = i·vec(ρ_ss)).
            # Rotate so that Tr(M) is real and positive, which maximises
            # Re(Tr(H)) and ensures H has non-zero trace.
            tr_M = np.trace(M)
            if abs(tr_M) > 1e-14:
                M = M * (abs(tr_M) / tr_M)  # multiply by conj(tr_M)/|tr_M|
            # Project to Hermitian
            H = (M + M.conj().T) / 2.0
            tr = float(np.real(np.trace(H)))
            if abs(tr) < 1e-14:
                continue
            H = H / tr
            # Measure "unphysicality" via negative eigenvalue mass
            ev = np.linalg.eigvalsh(H)
            neg = float(np.sum(np.abs(np.minimum(ev, 0.0))))
            if neg < best_neg:
                best_neg = neg
                best_rho = H

        if best_rho is None:
            # Last resort: use closest eigenvalue regardless of trace
            idx = int(np.argmin(dist))
            v = evecs[:, idx]
            best_rho = v.reshape(d, d)

        # ── Final projection to valid density matrix ─────────────────
        rho = (best_rho + best_rho.conj().T) / 2.0
        # Project to PSD by zeroing negative eigenvalues
        ev, U = np.linalg.eigh(rho)
        ev = np.maximum(ev, 0.0)
        rho = U @ np.diag(ev) @ U.conj().T
        tr = float(np.real(np.trace(rho)))
        if tr < 1e-14:
            raise ChannelError(
                "SteadyStateSolver: fixed-point density matrix has "
                "near-zero trace; the channel may lack a unique steady state."
            )
        return rho / tr
