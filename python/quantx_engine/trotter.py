"""Hamiltonian simulation via Suzuki-Trotter product formulas (M101).

Provides first-, second-, and fourth-order Suzuki-Trotter decompositions
for Hamiltonians expressed as weighted sums of Pauli strings::

    H = Σ_k  c_k  P_k,   P_k ∈ {I, X, Y, Z}^⊗n

Each term is exponentiated analytically using the identity::

    exp(-i θ P) = cos(θ) I - i sin(θ) P

which holds for any Pauli string because P² = I.  No full matrix
exponentiation is needed for individual terms.

Quick start::

    from quantx_engine.trotter import TrotterSimulator, TrotterTerm

    # Heisenberg XY dimer: H = XX + YY
    sim = TrotterSimulator(
        n_qubits=2,
        terms=[TrotterTerm(1.0, "XX"), TrotterTerm(1.0, "YY")],
    )
    result = sim.run(t=1.0, n_steps=50, order=2, compute_exact=True)
    print(result.trotter_error)   # ≈ 0 for small dt

    # 1D Heisenberg chain factory
    from quantx_engine.trotter import heisenberg_trotter
    sim2 = heisenberg_trotter(n=4, J=1.0)
    r2 = sim2.run(t=2.0, n_steps=100, order=4)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:
    from scipy.linalg import expm as _scipy_expm
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


# ── Error ──────────────────────────────────────────────────────────────────────

class TrotterError(ValueError):
    """Raised for invalid Trotter simulation parameters."""


# ── Pauli primitives ───────────────────────────────────────────────────────────

_I2 = np.eye(2, dtype=complex)
_X  = np.array([[0, 1], [1, 0]], dtype=complex)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z  = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI_MAP: dict[str, np.ndarray] = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}


def _pauli_matrix(pauli_string: str) -> np.ndarray:
    """Return the 2^n × 2^n matrix for a Pauli tensor-product string."""
    s = pauli_string.upper()
    if not s:
        raise TrotterError("Pauli string must not be empty")
    for ch in s:
        if ch not in _PAULI_MAP:
            raise TrotterError(
                f"Invalid Pauli character {ch!r}; use I, X, Y, or Z"
            )
    mat = _PAULI_MAP[s[0]]
    for ch in s[1:]:
        mat = np.kron(mat, _PAULI_MAP[ch])
    return mat


def _term_exp(coeff: float, P: np.ndarray, theta: float) -> np.ndarray:
    """Return exp(-i * coeff * P * theta) using exp(-iθP) = cosθ I - i sinθ P."""
    angle = coeff * theta
    n = P.shape[0]
    return math.cos(angle) * np.eye(n, dtype=complex) - 1j * math.sin(angle) * P


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class TrotterTerm:
    """One term ``c · P`` in a Pauli Hamiltonian ``H = Σ c_k P_k``.

    Parameters
    ----------
    coefficient : float
        Real scalar weight *c*.  Must be real for H to be Hermitian.
    pauli_string : str
        Pauli tensor-product string such as ``'ZZ'``, ``'XYZ'``, ``'IZI'``.
        Length must equal the ``n_qubits`` of the :class:`TrotterSimulator`.
    """
    coefficient: float
    pauli_string: str

    def __post_init__(self) -> None:
        self.pauli_string = self.pauli_string.upper()
        for ch in self.pauli_string:
            if ch not in _PAULI_MAP:
                raise TrotterError(
                    f"Invalid Pauli character {ch!r} in {self.pauli_string!r}"
                )


@dataclass
class TrotterResult:
    """Outcome of a :class:`TrotterSimulator` run.

    Attributes
    ----------
    state : np.ndarray
        Final state vector, shape ``(2**n_qubits,)``.
    probs : np.ndarray
        Measurement probability distribution ``|state|²``.
    n_steps : int
        Number of Trotter steps applied.
    order : int
        Product-formula order (1, 2, or 4).
    t : float
        Total simulated time.
    trotter_error : float or None
        ``‖ψ_trotter − ψ_exact‖₂`` when *compute_exact* was requested;
        ``None`` otherwise.
    metadata : dict
        Diagnostics: ``n_qubits``, ``n_terms``, ``dt``.
    """
    state: np.ndarray
    probs: np.ndarray
    n_steps: int
    order: int
    t: float
    trotter_error: Optional[float]
    metadata: dict = field(default_factory=dict)

    @property
    def n_qubits(self) -> int:
        """System size inferred from state-vector length."""
        return int(round(math.log2(len(self.state))))


# ── TrotterSimulator ───────────────────────────────────────────────────────────

class TrotterSimulator:
    """Hamiltonian simulation via Suzuki-Trotter product formulas.

    The Hamiltonian is given as a sum of weighted Pauli strings::

        H = Σ_k  c_k  P_k

    The unitary ``exp(-iHt)`` is approximated by composing single-term
    exponentials ``exp(-i c_k P_k Δt)`` (each exact analytically) arranged
    according to a first-, second-, or fourth-order product formula.

    Parameters
    ----------
    n_qubits : int
        System size (≥ 1).
    terms : sequence of TrotterTerm
        Pauli decomposition of *H*.  Each term's ``pauli_string`` length
        must equal *n_qubits*.
    initial_state : np.ndarray or None
        Initial state vector of length ``2**n_qubits``.
        Defaults to the all-zero computational basis state ``|0…0⟩``.
    """

    def __init__(
        self,
        n_qubits: int,
        terms: Sequence[TrotterTerm],
        initial_state: Optional[np.ndarray] = None,
    ) -> None:
        if n_qubits < 1:
            raise TrotterError("n_qubits must be ≥ 1")
        self._n = n_qubits
        self._dim = 2 ** n_qubits

        if not terms:
            raise TrotterError("Hamiltonian must have at least one TrotterTerm")
        for term in terms:
            if len(term.pauli_string) != n_qubits:
                raise TrotterError(
                    f"Pauli string {term.pauli_string!r} has length "
                    f"{len(term.pauli_string)}, expected {n_qubits}"
                )
        self._terms: list[TrotterTerm] = list(terms)

        # Pre-build Pauli matrices and full Hamiltonian
        self._paulis: list[np.ndarray] = [
            _pauli_matrix(t.pauli_string) for t in self._terms
        ]
        self._H = sum(
            t.coefficient * P for t, P in zip(self._terms, self._paulis)
        ).astype(complex)

        # Initial state
        if initial_state is None:
            psi0 = np.zeros(self._dim, dtype=complex)
            psi0[0] = 1.0
        else:
            psi0 = np.asarray(initial_state, dtype=complex)
            if psi0.shape != (self._dim,):
                raise TrotterError(
                    f"initial_state must have length {self._dim}, "
                    f"got {psi0.shape}"
                )
            nm = np.linalg.norm(psi0)
            if nm < 1e-12:
                raise TrotterError("initial_state has zero norm")
            psi0 = psi0 / nm
        self._psi0 = psi0

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def hamiltonian(self) -> np.ndarray:
        """Full Hamiltonian matrix (read-only copy)."""
        return self._H.copy()

    @property
    def initial_state(self) -> np.ndarray:
        """Initial state vector (read-only copy)."""
        return self._psi0.copy()

    # ── Step matrix builders ───────────────────────────────────────────────

    def _step1(self, dt: float) -> np.ndarray:
        """First-order Lie-Trotter step: ∏_k exp(-i c_k P_k dt)."""
        U = np.eye(self._dim, dtype=complex)
        for term, P in zip(self._terms, self._paulis):
            U = _term_exp(term.coefficient, P, dt) @ U
        return U

    def _step2(self, dt: float) -> np.ndarray:
        """Symmetric (2nd-order) Suzuki-Trotter step.

        S₂(dt) = [∏_k exp(-i c_k P_k dt/2)] · [∏_k (reversed) exp(-i c_k P_k dt/2)]

        For K terms this gives: U₁^(½) … U_K^(½) · U_K^(½) … U₁^(½)
        which collapses to the palindrome U₁^(½) … U_{K-1}^(½) U_K U_{K-1}^(½) … U₁^(½).
        """
        half = dt / 2.0
        U_fwd = np.eye(self._dim, dtype=complex)
        for term, P in zip(self._terms, self._paulis):
            U_fwd = _term_exp(term.coefficient, P, half) @ U_fwd
        U_bwd = np.eye(self._dim, dtype=complex)
        for term, P in zip(reversed(self._terms), reversed(self._paulis)):
            U_bwd = _term_exp(term.coefficient, P, half) @ U_bwd
        return U_bwd @ U_fwd

    def _step4(self, dt: float) -> np.ndarray:
        """Fourth-order Suzuki-Trotter step.

        S₄(dt) = S₂(p·dt)² · S₂((1−4p)·dt) · S₂(p·dt)²
        where p = 1/(4 − 4^{1/3}).
        """
        p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        s = self._step2
        return s(p * dt) @ s(p * dt) @ s((1.0 - 4.0 * p) * dt) @ s(p * dt) @ s(p * dt)

    # ── Public run ─────────────────────────────────────────────────────────

    def run(
        self,
        t: float,
        n_steps: int = 100,
        order: int = 2,
        compute_exact: bool = False,
    ) -> TrotterResult:
        """Simulate time evolution ``|ψ(t)⟩ ≈ exp(-iHt)|ψ₀⟩``.

        Parameters
        ----------
        t : float
            Total simulation time.  Negative values evolve backward.
        n_steps : int
            Number of Trotter steps.  Larger values give higher accuracy
            at proportionally larger cost.
        order : {1, 2, 4}
            Product-formula order.
        compute_exact : bool
            If ``True``, compute the exact state via ``scipy.linalg.expm``
            and store the error ``‖ψ_trotter − ψ_exact‖₂`` in the result.
            Requires *scipy*.

        Returns
        -------
        TrotterResult
        """
        if n_steps < 1:
            raise TrotterError(f"n_steps must be ≥ 1, got {n_steps}")
        if order not in (1, 2, 4):
            raise TrotterError(f"order must be 1, 2, or 4; got {order}")

        dt = t / n_steps
        if order == 1:
            U_step = self._step1(dt)
        elif order == 2:
            U_step = self._step2(dt)
        else:
            U_step = self._step4(dt)

        state = self._psi0.copy()
        for _ in range(n_steps):
            state = U_step @ state

        probs = np.abs(state) ** 2

        trotter_error: Optional[float] = None
        if compute_exact:
            if not _HAVE_SCIPY:
                raise TrotterError(
                    "compute_exact requires scipy; install with: pip install scipy"
                )
            exact_state = _scipy_expm(-1j * self._H * t) @ self._psi0
            trotter_error = float(np.linalg.norm(state - exact_state))

        return TrotterResult(
            state=state,
            probs=probs,
            n_steps=n_steps,
            order=order,
            t=t,
            trotter_error=trotter_error,
            metadata={
                "n_qubits": self._n,
                "n_terms": len(self._terms),
                "dt": dt,
            },
        )

    # ── Static utilities ───────────────────────────────────────────────────

    @staticmethod
    def trotter_error_bound(
        n_qubits: int,
        terms: Sequence[TrotterTerm],
        t: float,
        n_steps: int,
        order: int = 2,
    ) -> float:
        """Compute an analytical upper bound on the Trotter error.

        Uses spectral-norm commutator sums:

        * **Order 1**: ``‖err‖ ≤ (t²/2n) Σ_{j<k} |c_j c_k| ‖[P_j,P_k]‖``
        * **Order 2**: ``‖err‖ ≤ (t³/12n²) Σ_{j,k,j≠k} |c_j² c_k| ‖[[P_j,P_k],P_j]‖``
        * **Order 4**: rough bound via ``‖H‖⁵ t⁵ / (120 n⁴)``

        Parameters
        ----------
        n_qubits, terms, t, n_steps, order
            Same semantics as :meth:`run`.

        Returns
        -------
        float
            Non-negative error bound.
        """
        if order not in (1, 2, 4):
            raise TrotterError("order must be 1, 2, or 4")
        if n_steps < 1:
            raise TrotterError("n_steps must be ≥ 1")

        paulis = [_pauli_matrix(term.pauli_string) for term in terms]
        coeffs = [term.coefficient for term in terms]
        K = len(terms)
        dim = 2 ** n_qubits

        if order == 1:
            bound = 0.0
            for j in range(K):
                for k in range(j + 1, K):
                    comm = paulis[j] @ paulis[k] - paulis[k] @ paulis[j]
                    bound += abs(coeffs[j] * coeffs[k]) * float(
                        np.linalg.norm(comm, ord=2)
                    )
            return (t ** 2 / (2 * n_steps)) * bound

        elif order == 2:
            bound = 0.0
            for j in range(K):
                for k in range(K):
                    if j == k:
                        continue
                    comm_jk = paulis[j] @ paulis[k] - paulis[k] @ paulis[j]
                    nested  = comm_jk @ paulis[j] - paulis[j] @ comm_jk
                    bound += abs(coeffs[j] ** 2 * coeffs[k]) * float(
                        np.linalg.norm(nested, ord=2)
                    )
            return (abs(t) ** 3 / (12 * n_steps ** 2)) * bound

        else:  # order 4
            H = sum(c * P for c, P in zip(coeffs, paulis))
            H_norm = float(np.linalg.norm(H, ord=2))
            return (H_norm ** 5) * (abs(t) ** 5) / (120 * n_steps ** 4)

    @staticmethod
    def optimal_steps(
        n_qubits: int,
        terms: Sequence[TrotterTerm],
        t: float,
        target_error: float = 1e-3,
        order: int = 2,
    ) -> int:
        """Return the minimum *n_steps* whose error bound ≤ *target_error*.

        Uses binary search over :meth:`trotter_error_bound`.

        Parameters
        ----------
        target_error : float
            Desired ``‖ψ_trotter − ψ_exact‖₂ ≤ target_error``.
        """
        if target_error <= 0:
            raise TrotterError("target_error must be positive")

        _bound = TrotterSimulator.trotter_error_bound

        # If a single step is already sufficient, return 1
        if _bound(n_qubits, terms, t, 1, order) <= target_error:
            return 1

        # Double hi until bound is satisfied
        lo, hi = 1, 2
        while _bound(n_qubits, terms, t, hi, order) > target_error:
            hi *= 2
            if hi > 100_000:
                return hi  # safety cap

        # Binary search in [lo, hi]
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if _bound(n_qubits, terms, t, mid, order) <= target_error:
                hi = mid
            else:
                lo = mid
        return hi


# ── Convenience factory functions ──────────────────────────────────────────────

def heisenberg_trotter(
    n: int,
    J: float = 1.0,
    hz: float = 0.0,
    initial_state: Optional[np.ndarray] = None,
) -> TrotterSimulator:
    """Build a :class:`TrotterSimulator` for the 1D Heisenberg chain (open BC).

    ``H = J Σᵢ (XXᵢ,ᵢ₊₁ + YYᵢ,ᵢ₊₁ + ZZᵢ,ᵢ₊₁) + hz Σᵢ Zᵢ``

    Parameters
    ----------
    n : int
        Number of spins (≥ 2).
    J : float
        Exchange coupling strength.
    hz : float
        Longitudinal magnetic field.
    """
    if n < 2:
        raise TrotterError("Heisenberg chain requires n ≥ 2")
    terms: list[TrotterTerm] = []
    for i in range(n - 1):
        for pauli in ("X", "Y", "Z"):
            ps = "I" * i + pauli + pauli + "I" * (n - i - 2)
            terms.append(TrotterTerm(J, ps))
    if hz != 0.0:
        for i in range(n):
            ps = "I" * i + "Z" + "I" * (n - i - 1)
            terms.append(TrotterTerm(hz, ps))
    return TrotterSimulator(n, terms, initial_state=initial_state)


def ising_trotter(
    n: int,
    J: float = 1.0,
    h: float = 1.0,
    initial_state: Optional[np.ndarray] = None,
) -> TrotterSimulator:
    """Build a :class:`TrotterSimulator` for the transverse-field Ising model (open BC).

    ``H = −J Σᵢ ZZᵢ,ᵢ₊₁ − h Σᵢ Xᵢ``

    Parameters
    ----------
    n : int
        Number of spins (≥ 2).
    J : float
        ZZ coupling (positive = ferromagnetic).
    h : float
        Transverse field strength.
    """
    if n < 2:
        raise TrotterError("Ising chain requires n ≥ 2")
    terms: list[TrotterTerm] = []
    for i in range(n - 1):
        ps = "I" * i + "ZZ" + "I" * (n - i - 2)
        terms.append(TrotterTerm(-J, ps))
    if h != 0.0:
        for i in range(n):
            ps = "I" * i + "X" + "I" * (n - i - 1)
            terms.append(TrotterTerm(-h, ps))
    return TrotterSimulator(n, terms, initial_state=initial_state)
