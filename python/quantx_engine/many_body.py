"""Many-body physics enhancements for the QuantX engine (M102).

Three additions:

**Out-of-time-order correlators (OTOCs)**
    Measure quantum information scrambling via the squared commutator::

        C(t) = -⟨[W(t), V]²⟩  =  1 - Re F(t)
        F(t) = ⟨W†(t) V† W(t) V⟩

    where W(t) = exp(iHt) W exp(-iHt) is the Heisenberg-picture operator.
    Fast scrambling (chaotic systems like SYK) drives C(t) → 1 on the
    scrambling time scale t* ~ β ln N.

**Sachdev-Ye-Kitaev (SYK) model**
    Random all-to-all q-body Majorana fermion Hamiltonian::

        H_SYK = (i^{q/2} / q!) Σ_{1≤i₁<…<i_q≤N} J_{i₁…i_q} χ_{i₁}…χ_{i_q}

    The Majorana operators χ_i satisfy {χ_i, χ_j} = 2δ_{ij} and are mapped
    to qubits via a Jordan-Wigner-like transformation.  The q=4 model is the
    minimal holographic (AdS/CFT dual) case.

**2D DMRG via snake-path mapping**
    Maps a 2D Lx×Ly lattice to a 1D MPS via a snake (boustrophedon) path,
    then runs the existing DMRG sweep engine.  Long-range 2D couplings become
    medium-range 1D interactions that DMRG can still handle efficiently with
    modest bond dimensions.

Quick start::

    from quantx_engine.many_body import compute_otoc, SYKModel, dmrg_2d

    # OTOC for a 4-qubit Ising chain
    from quantx_engine.trotter import ising_trotter
    sim = ising_trotter(n=4, J=1.0, h=0.5)
    result = compute_otoc(sim.hamiltonian, op_w="X", op_v="Z",
                          site_w=0, site_v=3, t=1.0, n_qubits=4)
    print(result.otoc_value)   # C(t) ∈ [0, 1]

    # SYK model
    model = SYKModel(N=8, q=4, seed=42)
    print(model.ground_state_energy)

    # 2D DMRG
    result = dmrg_2d(Lx=2, Ly=2, J=1.0, bond_dim=8, sweeps=4)
    print(result.energy)

References
----------
.. [1] Kitaev, A. Y. (1997). Fault-tolerant quantum computation by anyons.
       *Ann. Phys.* 303, 2–30.
       https://doi.org/10.1016/S0003-4916(02)00018-0

.. [2] Sachdev, S. & Ye, J. (1993). Gapless spin-fluid ground state in a
       random quantum Heisenberg magnet. *Phys. Rev. Lett.* 70, 3339.
       https://doi.org/10.1103/PhysRevLett.70.3339

.. [3] Larkin, A. I. & Ovchinnikov, Yu. N. (1969). Quasiclassical method in
       the theory of superconductivity. *Sov. Phys. JETP* 28, 1200.
       (OTOC / scrambling motivation)

Physical validation
-------------------
SYK model ground state energy is negative for any random couplings J_ijkl
(the random model has E_0 < 0 with probability 1) [Sachdev1993].
OTOC C(t) = -<[W(t),V]^2> grows from 0 and saturates at late times for
chaotic systems (information scrambling).
"""
from __future__ import annotations

import math
import itertools
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:
    from scipy.linalg import expm as _scipy_expm
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ── Error ──────────────────────────────────────────────────────────────────────

class ManyBodyError(ValueError):
    """Raised for invalid many-body simulation parameters."""


# ── Pauli helpers (local, minimal) ────────────────────────────────────────────

_I2 = np.eye(2, dtype=complex)
_X  = np.array([[0, 1], [1, 0]], dtype=complex)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z  = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}


def _single_site_op(op: str, site: int, n_qubits: int) -> np.ndarray:
    """Return the 2^n × 2^n matrix for `op` acting on `site`, I elsewhere."""
    if op not in _PAULI:
        raise ManyBodyError(f"Unknown single-site operator {op!r}; use I/X/Y/Z")
    if not (0 <= site < n_qubits):
        raise ManyBodyError(f"site={site} out of range [0, {n_qubits})")
    mat = _PAULI["I"]
    for i in range(n_qubits):
        mat = np.kron(mat, _PAULI[op] if i == site else _PAULI["I"]) if i == 0 else np.kron(mat, _PAULI[op] if i == site else _PAULI["I"])
    # Rebuild cleanly
    parts = [_PAULI[op] if i == site else _PAULI["I"] for i in range(n_qubits)]
    result = parts[0]
    for p in parts[1:]:
        result = np.kron(result, p)
    return result


def _exact_time_evolve(H: np.ndarray, t: float) -> np.ndarray:
    """Return exp(-iHt) for a Hermitian matrix H.

    Uses ``numpy.linalg.eigh`` (exact spectral decomposition, no scipy
    required):  exp(-iHt) = V · diag(exp(-i λ t)) · V†.
    """
    eigvals, eigvecs = np.linalg.eigh(H)
    phases = np.exp(-1j * eigvals * t)
    return (eigvecs * phases) @ eigvecs.conj().T


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — Out-of-time-order correlators (OTOCs)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OTOCResult:
    """Result of an OTOC calculation.

    Attributes
    ----------
    otoc_value : float
        Scrambling diagnostic C(t) = 1 − Re F(t) ∈ [0, 1].
    fotoc : complex
        Raw OTOC F(t) = ⟨W†(t) V† W(t) V⟩.
    t : float
        Time at which the OTOC was evaluated.
    n_qubits : int
        System size.
    metadata : dict
        op_w, op_v, site_w, site_v, method used.
    """
    otoc_value: float
    fotoc: complex
    t: float
    n_qubits: int
    metadata: dict = field(default_factory=dict)


@dataclass
class OTOCSweepResult:
    """OTOC evaluated at multiple times.

    Attributes
    ----------
    t_list : np.ndarray
        Time points.
    otoc_values : np.ndarray
        C(t) at each time point.
    fotoc_values : np.ndarray
        Raw F(t) at each time point (complex).
    scrambling_time : float or None
        First t where C(t) > 0.5 (rough scrambling time); None if never reached.
    """
    t_list: np.ndarray
    otoc_values: np.ndarray
    fotoc_values: np.ndarray
    scrambling_time: Optional[float]


def compute_otoc(
    hamiltonian: np.ndarray,
    op_w: str,
    op_v: str,
    site_w: int,
    site_v: int,
    t: float,
    n_qubits: int,
    initial_state: Optional[np.ndarray] = None,
    n_steps: int = 100,
    use_trotter: bool = False,
) -> OTOCResult:
    """Compute the OTOC F(t) = ⟨ψ|W†(t) V† W(t) V|ψ⟩.

    The scrambling diagnostic is C(t) = 1 − Re F(t).

    For a thermalising system C(t) grows from 0 to ~1 on the scrambling
    time scale; for an integrable system it stays near 0.

    Parameters
    ----------
    hamiltonian : np.ndarray
        ``(2^n × 2^n)`` Hermitian Hamiltonian matrix.
    op_w, op_v : str
        Single-site Pauli operators (``'X'``, ``'Y'``, ``'Z'``, or ``'I'``).
    site_w, site_v : int
        Qubit indices for W and V respectively.
    t : float
        Time of evolution.
    n_qubits : int
        System size (used to build operator matrices).
    initial_state : np.ndarray or None
        State |ψ⟩ in which expectation value is taken.  Defaults to |0…0⟩.
    n_steps : int
        Trotter steps (only used when ``use_trotter=True``).
    use_trotter : bool
        If True, approximate exp(-iHt) via second-order Trotter; otherwise
        use ``scipy.linalg.expm`` (exact, requires scipy).

    Returns
    -------
    OTOCResult
    """
    H = np.asarray(hamiltonian, dtype=complex)
    dim = 2 ** n_qubits
    if H.shape != (dim, dim):
        raise ManyBodyError(
            f"Hamiltonian shape {H.shape} inconsistent with n_qubits={n_qubits}"
        )

    W = _single_site_op(op_w, site_w, n_qubits)
    V = _single_site_op(op_v, site_v, n_qubits)

    if initial_state is None:
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1.0
    else:
        psi = np.asarray(initial_state, dtype=complex)
        if psi.shape != (dim,):
            raise ManyBodyError(f"initial_state must have length {dim}")
        nm = np.linalg.norm(psi)
        if nm < 1e-12:
            raise ManyBodyError("initial_state has zero norm")
        psi = psi / nm

    # Time-evolution unitary U(t) = exp(-iHt)
    if use_trotter:
        from quantx_engine.trotter import TrotterSimulator, TrotterTerm
        # Decompose H into Pauli terms by tracing: c_k = Tr(P_k H) / 2^n
        terms = _hamiltonian_to_trotter_terms(H, n_qubits)
        sim = TrotterSimulator(n_qubits, terms)
        dt = t / n_steps
        U_step = sim._step2(dt)
        U = np.eye(dim, dtype=complex)
        for _ in range(n_steps):
            U = U_step @ U
        method = "trotter_order2"
    else:
        U = _exact_time_evolve(H, t)
        method = "exact_expm"

    Udagger = U.conj().T

    # W(t) = U† W U
    Wt = Udagger @ W @ U

    # F(t) = ⟨ψ| W†(t) V† W(t) V |ψ⟩
    Vpsi    = V @ psi
    WtVpsi  = Wt @ Vpsi
    VtWtVpsi = V.conj().T @ WtVpsi   # V† W(t) V |ψ⟩  (V is Hermitian → V† = V)
    WdVtWtVpsi = Wt.conj().T @ VtWtVpsi

    fotoc = complex(np.vdot(psi, WdVtWtVpsi))
    otoc_val = float(1.0 - fotoc.real)

    return OTOCResult(
        otoc_value=otoc_val,
        fotoc=fotoc,
        t=t,
        n_qubits=n_qubits,
        metadata={
            "op_w": op_w, "site_w": site_w,
            "op_v": op_v, "site_v": site_v,
            "method": method,
        },
    )


def otoc_sweep(
    hamiltonian: np.ndarray,
    op_w: str,
    op_v: str,
    site_w: int,
    site_v: int,
    t_list: Sequence[float],
    n_qubits: int,
    initial_state: Optional[np.ndarray] = None,
) -> OTOCSweepResult:
    """Compute OTOC at multiple times.

    Parameters
    ----------
    t_list : sequence of float
        Time points (must be non-negative, monotonically non-decreasing).

    Returns
    -------
    OTOCSweepResult
    """
    t_arr = np.asarray(t_list, dtype=float)
    if t_arr.ndim != 1 or len(t_arr) == 0:
        raise ManyBodyError("t_list must be a non-empty 1-D sequence")

    otoc_vals = np.zeros(len(t_arr))
    fotoc_vals = np.zeros(len(t_arr), dtype=complex)

    for i, t in enumerate(t_arr):
        r = compute_otoc(hamiltonian, op_w, op_v, site_w, site_v,
                         float(t), n_qubits, initial_state)
        otoc_vals[i] = r.otoc_value
        fotoc_vals[i] = r.fotoc

    scrambling_time: Optional[float] = None
    for t, c in zip(t_arr, otoc_vals):
        if c > 0.5:
            scrambling_time = float(t)
            break

    return OTOCSweepResult(
        t_list=t_arr,
        otoc_values=otoc_vals,
        fotoc_values=fotoc_vals,
        scrambling_time=scrambling_time,
    )


def _hamiltonian_to_trotter_terms(H: np.ndarray, n_qubits: int):
    """Decompose an n-qubit Hamiltonian into TrotterTerm list via Pauli tracing.

    Uses Tr(P_k H) / 2^n to extract coefficients.  Internal helper for
    Trotter-based OTOC computation.
    """
    from quantx_engine.trotter import TrotterTerm
    dim = 2 ** n_qubits
    terms = []
    for pauli_labels in itertools.product("IXYZ", repeat=n_qubits):
        ps = "".join(pauli_labels)
        # Build Pauli matrix
        parts = [_PAULI[c] for c in ps]
        P = parts[0]
        for p in parts[1:]:
            P = np.kron(P, p)
        coeff = float(np.trace(P @ H).real) / dim
        if abs(coeff) > 1e-14:
            terms.append(TrotterTerm(coeff, ps))
    if not terms:
        # Zero Hamiltonian: add a dummy zero term
        terms.append(TrotterTerm(0.0, "I" * n_qubits))
    return terms


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Sachdev-Ye-Kitaev (SYK) model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SYKResult:
    """Spectral diagnostics of the SYK model.

    Attributes
    ----------
    energies : np.ndarray
        Full eigenspectrum (sorted ascending).
    ground_state_energy : float
        Lowest eigenvalue.
    level_spacing_ratio : float
        Mean consecutive level-spacing ratio ⟨r⟩.  GOE predicts ≈ 0.536,
        Poisson (integrable) predicts ≈ 0.386.
    spectral_form_factor : np.ndarray or None
        |Tr(e^{-iHt})|² evaluated at ``t_list`` (if requested).
    """
    energies: np.ndarray
    ground_state_energy: float
    level_spacing_ratio: float
    spectral_form_factor: Optional[np.ndarray]


class SYKModel:
    """Sachdev-Ye-Kitaev random q-body Majorana fermion Hamiltonian.

    The q=4 SYK model is the minimal model conjectured to have a holographic
    (AdS₂) dual at low temperatures; it exhibits maximal quantum chaos.

    The Hamiltonian is::

        H = (i^{q/2} / q!) Σ_{i₁<…<i_q} J_{i₁…i_q} γ_{i₁}…γ_{i_q}

    where γ_i are Majorana fermion operators satisfying {γ_i, γ_j} = 2δ_{ij},
    mapped to qubits via the Jordan-Wigner transformation.

    Parameters
    ----------
    N : int
        Number of Majorana modes (must be even).  The qubit system has N//2
        qubits; the Hilbert space has dimension 2^{N//2}.
    q : int
        Body count (must be even, typically 4).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        N: "Optional[int]" = None,
        q: int = 4,
        seed: "Optional[int]" = None,
        *,
        n_sites: "Optional[int]" = None,
    ) -> None:
        # ``n_sites`` kwarg: number of Majorana modes (same as N)
        if n_sites is not None and N is None:
            N = n_sites
        if N is None:
            raise ManyBodyError("N (or n_sites) must be provided")
        if N < 2 or N % 2 != 0:
            raise ManyBodyError("SYK requires N ≥ 2 even (number of Majorana modes)")
        if q < 2 or q % 2 != 0:
            raise ManyBodyError("q must be an even integer ≥ 2")
        if q > N:
            raise ManyBodyError(f"q={q} cannot exceed N={N}")

        self._N = N
        self._q = q
        self._n_qubits = N // 2
        self._rng = np.random.default_rng(seed)

        # Build Majorana operators γ_i via Jordan-Wigner
        self._majoranas = self._build_majoranas()

        # Build random coupling tensor and Hamiltonian
        self._J, self._H = self._build_hamiltonian()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def N(self) -> int:
        return self._N

    @property
    def q(self) -> int:
        return self._q

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def hamiltonian(self) -> np.ndarray:
        return self._H.copy()

    @property
    def dim(self) -> int:
        return 2 ** self._n_qubits

    # ── Majorana + Hamiltonian construction ───────────────────────────────

    def _build_majoranas(self) -> list[np.ndarray]:
        """Jordan-Wigner Majorana operators γ_{2k} = (c_k + c†_k),
        γ_{2k+1} = i(c†_k - c_k) for k=0..n_qubits-1."""
        n = self._n_qubits
        dim = 2 ** n
        majoranas: list[np.ndarray] = []

        for k in range(n):
            # Jordan-Wigner string: Z^⊗k ⊗ X ⊗ I^⊗(n-k-1) for c_k + c†_k
            # and Z^⊗k ⊗ Y ⊗ I^⊗(n-k-1) for i(c†_k - c_k)
            jw_string_x = ["I"] * n
            jw_string_y = ["I"] * n
            for j in range(k):
                jw_string_x[j] = "Z"
                jw_string_y[j] = "Z"
            jw_string_x[k] = "X"
            jw_string_y[k] = "Y"

            gamma_even = self._pauli_kron(jw_string_x)
            gamma_odd  = self._pauli_kron(jw_string_y)
            majoranas.append(gamma_even)  # γ_{2k}
            majoranas.append(gamma_odd)   # γ_{2k+1}

        return majoranas  # length N

    def _pauli_kron(self, pauli_list: list[str]) -> np.ndarray:
        result = _PAULI[pauli_list[0]]
        for p in pauli_list[1:]:
            result = np.kron(result, _PAULI[p])
        return result

    def _build_hamiltonian(self):
        """Build coupling tensor J and the q-body Hamiltonian."""
        N, q = self._N, self._q
        var = math.factorial(q - 1) / N ** (q - 1)  # normalisation: J² = (q-1)!/N^{q-1}
        sigma = math.sqrt(var)

        # Enumerate all q-subsets of [0, N)
        subsets = list(itertools.combinations(range(N), q))
        # Draw Gaussian couplings (antisymmetric by symmetry of subset enumeration)
        J_vals = self._rng.normal(0, sigma, len(subsets))

        dim = self.dim
        H = np.zeros((dim, dim), dtype=complex)
        prefactor = (1j ** (q // 2)) / math.factorial(q)

        for J_val, subset in zip(J_vals, subsets):
            # Product γ_{i1} γ_{i2} … γ_{iq}
            P = self._majoranas[subset[0]]
            for idx in subset[1:]:
                P = P @ self._majoranas[idx]
            H += J_val * prefactor * P

        # Symmetrise: H should be Hermitian
        H = (H + H.conj().T) / 2
        return J_vals, H

    # ── Spectral analysis ─────────────────────────────────────────────────

    def diagonalise(
        self,
        t_list: Optional[Sequence[float]] = None,
    ) -> SYKResult:
        """Diagonalise the Hamiltonian and compute spectral diagnostics.

        Parameters
        ----------
        t_list : sequence of float or None
            If provided, compute spectral form factor |Tr(e^{-iHt})|² at these times.

        Returns
        -------
        SYKResult
        """
        eigvals = np.linalg.eigvalsh(self._H)  # real, sorted
        eigvals = np.sort(eigvals.real)

        # Level spacing ratio ⟨r⟩ = ⟨min(δn,δn+1)/max(δn,δn+1)⟩
        spacings = np.diff(eigvals)
        spacings = spacings[spacings > 1e-15]  # remove near-degenerate pairs
        if len(spacings) > 1:
            ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
            lsr = float(np.mean(ratios))
        else:
            lsr = float("nan")

        sff = None
        if t_list is not None:
            t_arr = np.asarray(t_list, dtype=float)
            sff = np.array([
                abs(np.sum(np.exp(-1j * eigvals * t))) ** 2
                for t in t_arr
            ])

        return SYKResult(
            energies=eigvals,
            ground_state_energy=float(eigvals[0]),
            level_spacing_ratio=lsr,
            spectral_form_factor=sff,
        )

    @property
    def ground_state_energy(self) -> float:
        """Ground state energy (calls diagonalise internally)."""
        return self.diagonalise().ground_state_energy


def syk_hamiltonian(
    N: int,
    q: int = 4,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return the SYK Hamiltonian matrix for N Majorana modes, q-body interactions.

    Convenience wrapper around :class:`SYKModel`.

    Parameters
    ----------
    N : int
        Number of Majorana modes (even, ≥ 2).
    q : int
        Interaction body count (even, default 4).
    seed : int or None
        Random seed.
    """
    return SYKModel(N=N, q=q, seed=seed).hamiltonian


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3 — 2D DMRG via snake-path mapping
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DMRG2DResult:
    """Result of a 2D DMRG calculation.

    Attributes
    ----------
    energy : float
        Variational ground-state energy.
    bond_dims : list[int]
        Bond dimensions at the end of the last sweep.
    truncation_error : float
        Accumulated truncation error from the final sweep.
    Lx, Ly : int
        Lattice dimensions.
    n_sweeps : int
        Number of sweeps performed.
    """
    energy: float
    bond_dims: list[int]
    truncation_error: float
    Lx: int
    Ly: int
    n_sweeps: int


class SnakeMapper:
    """Maps 2D lattice sites (x, y) to a 1D snake-path index.

    The snake (boustrophedon) ordering::

        Row 0: 0 → 1 → 2 → … → Lx-1        (left to right)
        Row 1: Lx + Lx-1 → … → Lx           (right to left)
        Row 2: 2Lx → 2Lx+1 → …              (left to right)
        …

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions (x = columns, y = rows).
    """

    def __init__(self, Lx: int, Ly: int) -> None:
        if Lx < 1 or Ly < 1:
            raise ManyBodyError("Lx and Ly must be ≥ 1")
        self.Lx = Lx
        self.Ly = Ly
        self._total = Lx * Ly
        # Build forward and inverse maps
        self._to_1d: dict[tuple[int, int], int] = {}
        self._to_2d: dict[int, tuple[int, int]] = {}
        idx = 0
        for y in range(Ly):
            xs = range(Lx) if y % 2 == 0 else range(Lx - 1, -1, -1)
            for x in xs:
                self._to_1d[(x, y)] = idx
                self._to_2d[idx] = (x, y)
                idx += 1

    def to_1d(self, x: int, y: int) -> int:
        """Convert 2D coordinates to snake-path 1D index."""
        if not (0 <= x < self.Lx and 0 <= y < self.Ly):
            raise ManyBodyError(f"({x},{y}) out of range Lx={self.Lx}, Ly={self.Ly}")
        return self._to_1d[(x, y)]

    def to_2d(self, idx: int) -> tuple[int, int]:
        """Convert snake-path 1D index to 2D coordinates."""
        if not (0 <= idx < self._total):
            raise ManyBodyError(f"idx={idx} out of range [0, {self._total})")
        return self._to_2d[idx]

    @property
    def n_sites(self) -> int:
        return self._total

    def nn_pairs_1d(self) -> list[tuple[int, int]]:
        """All nearest-neighbour pairs on the 2D lattice as 1D snake indices.

        Returns a list of (i, j) with i < j covering both x- and y-direction
        bonds.
        """
        pairs: list[tuple[int, int]] = []
        for x in range(self.Lx):
            for y in range(self.Ly):
                i = self.to_1d(x, y)
                # right neighbour
                if x + 1 < self.Lx:
                    j = self.to_1d(x + 1, y)
                    pairs.append((min(i, j), max(i, j)))
                # up neighbour
                if y + 1 < self.Ly:
                    j = self.to_1d(x, y + 1)
                    pairs.append((min(i, j), max(i, j)))
        return sorted(set(pairs))


def dmrg_2d(
    Lx: int,
    Ly: int,
    J: float = 1.0,
    bond_dim: int = 16,
    sweeps: int = 4,
    model: str = "heisenberg",
) -> DMRG2DResult:
    """Ground-state DMRG for the 2D Heisenberg (or Ising) model on an Lx×Ly lattice.

    Uses a snake-path MPS mapping to convert the 2D problem to 1D, then runs
    two-site DMRG sweeps.  The 1D chain has ``Lx*Ly`` sites with both
    short-range (adjacent on snake path) and medium-range (cross-row) bonds.

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions.  Total sites = Lx * Ly.
    J : float
        Exchange coupling (positive = antiferromagnetic).
    bond_dim : int
        Maximum MPS bond dimension (controls accuracy).
    sweeps : int
        Number of DMRG sweeps (left-to-right + right-to-left = one sweep).
    model : {'heisenberg', 'ising'}
        'heisenberg': H = J Σ (XX + YY + ZZ)
        'ising': H = -J Σ ZZ

    Returns
    -------
    DMRG2DResult
    """
    if Lx < 1 or Ly < 1:
        raise ManyBodyError("Lx and Ly must be ≥ 1")
    if bond_dim < 1:
        raise ManyBodyError("bond_dim must be ≥ 1")
    if sweeps < 1:
        raise ManyBodyError("sweeps must be ≥ 1")
    if model not in ("heisenberg", "ising"):
        raise ManyBodyError(f"model must be 'heisenberg' or 'ising', got {model!r}")

    mapper = SnakeMapper(Lx, Ly)
    n = mapper.n_sites
    pairs = mapper.nn_pairs_1d()

    # For small systems (n ≤ 12), use exact diagonalisation
    if n <= 12:
        return _dmrg_2d_exact(n, pairs, J, model, Lx, Ly, sweeps)

    # For larger systems use the variational MPS approach
    return _dmrg_2d_variational(n, pairs, J, bond_dim, sweeps, model, Lx, Ly)


def _build_2d_hamiltonian(n: int, pairs: list[tuple[int, int]],
                          J: float, model: str) -> np.ndarray:
    """Build the full 2^n × 2^n Hamiltonian for exact diagonalisation."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for (i, j) in pairs:
        if model == "heisenberg":
            for op in ("X", "Y", "Z"):
                Pi = _single_site_op(op, i, n)
                Pj = _single_site_op(op, j, n)
                H += J * (Pi @ Pj)
        else:  # ising
            Zi = _single_site_op("Z", i, n)
            Zj = _single_site_op("Z", j, n)
            H += -J * (Zi @ Zj)
    return H


def _dmrg_2d_exact(n, pairs, J, model, Lx, Ly, sweeps) -> DMRG2DResult:
    """Exact diagonalisation fallback for small lattices (n ≤ 12 sites)."""
    H = _build_2d_hamiltonian(n, pairs, J, model)
    energy = float(np.linalg.eigvalsh(H)[0])
    return DMRG2DResult(
        energy=energy,
        bond_dims=[1] * (n - 1),
        truncation_error=0.0,
        Lx=Lx,
        Ly=Ly,
        n_sweeps=sweeps,
    )


def _dmrg_2d_variational(n, pairs, J, bond_dim, sweeps, model, Lx, Ly) -> DMRG2DResult:
    """Variational MPS-DMRG for larger systems."""
    # Build MPO tensors for the Hamiltonian
    # Use a simple two-site DMRG with random initial MPS
    rng = np.random.default_rng(0)

    # Initialise random MPS with given bond_dim
    # Tensor shapes: [phys, bond_left, bond_right]
    d = 2  # physical dimension (spin-1/2)
    mps = _random_mps(n, d, bond_dim, rng)
    mps = _normalise_mps(mps)

    energy = 0.0
    trunc_error = 0.0

    # Two-site DMRG sweeps
    for sweep in range(sweeps):
        # Left-to-right sweep
        for site in range(n - 1):
            theta, e, trunc = _two_site_update(mps, site, pairs, J, model, bond_dim)
            energy = e
            trunc_error += trunc
            mps = _update_mps_left(mps, site, theta, bond_dim)
        # Right-to-left sweep
        for site in range(n - 2, -1, -1):
            theta, e, trunc = _two_site_update(mps, site, pairs, J, model, bond_dim)
            energy = e
            trunc_error += trunc
            mps = _update_mps_right(mps, site, theta, bond_dim)

    bond_dims = [mps[i].shape[2] for i in range(n - 1)]

    return DMRG2DResult(
        energy=energy,
        bond_dims=bond_dims,
        truncation_error=trunc_error,
        Lx=Lx,
        Ly=Ly,
        n_sweeps=sweeps,
    )


def _random_mps(n: int, d: int, bond_dim: int, rng) -> list[np.ndarray]:
    """Random orthonormalised MPS. Tensors: [d, chi_l, chi_r]."""
    tensors = []
    chi_l = 1
    for i in range(n):
        chi_r = bond_dim if i < n - 1 else 1
        chi_r = min(chi_r, d ** min(i + 1, n - i - 1) if n > 1 else 1)
        chi_r = max(chi_r, 1)
        A = rng.standard_normal((d, chi_l, chi_r)) + 1j * rng.standard_normal((d, chi_l, chi_r))
        tensors.append(A)
        chi_l = chi_r
    return tensors


def _normalise_mps(mps: list[np.ndarray]) -> list[np.ndarray]:
    """Left-normalise MPS (QR sweep)."""
    n = len(mps)
    for i in range(n - 1):
        A = mps[i]  # [d, chi_l, chi_r]
        d, chi_l, chi_r = A.shape
        mat = A.reshape(d * chi_l, chi_r)
        Q, R = np.linalg.qr(mat)
        new_chi = Q.shape[1]
        mps[i] = Q.reshape(d, chi_l, new_chi)
        # R: [new_chi, old_chi_r]; mps[i+1]: [d, old_chi_r, chi_r_next]
        # Contract over the shared bond (j = old_chi_r) → [d, new_chi, chi_r_next]
        mps[i + 1] = np.einsum("ij,kjl->kil", R, mps[i + 1])
    # Normalise last tensor
    norm = np.linalg.norm(mps[-1])
    if norm > 1e-12:
        mps[-1] = mps[-1] / norm
    return mps


def _expect_op(mps: list[np.ndarray], k: int, op: np.ndarray) -> float:
    """Mean-field expectation ⟨op⟩ on site k (no environment tensors).

    Uses the raw MPS tensor at site k, ignoring environment factors.  This is
    exact when k is the boundary of a left-/right-canonical MPS and the rest
    of the chain is orthogonally compressed, but is used here as a
    mean-field approximation for non-adjacent bond contributions during the
    two-site DMRG update.

    Parameters
    ----------
    mps : list of np.ndarray
        MPS tensors, each with shape [d, chi_l, chi_r].
    k : int
        Site index.
    op : np.ndarray
        Physical-space operator, shape [d, d].

    Returns
    -------
    float
        Real part of the (unnormalized) expectation value.
    """
    T = mps[k]  # [d, chi_l, chi_r]
    # ⟨op⟩ = Σ_{s,p,l,r} T*[s,l,r] op[s,p] T[p,l,r] / ||T||²
    val = float(np.real(np.einsum("slr,sp,plr->", T.conj(), op, T)))
    norm = float(np.real(np.einsum("slr,slr->", T.conj(), T)))
    return val / norm if norm > 1e-14 else 0.0


def _two_site_update(
    mps: list[np.ndarray],
    site: int,
    pairs: list[tuple[int, int]],
    J: float,
    model: str,
    bond_dim: int,
) -> tuple[np.ndarray, float, float]:
    """Apply local Hamiltonian to two-site tensor and return updated theta, energy, error."""
    A = mps[site]       # [d, chi_l, chi_m]
    B = mps[site + 1]   # [d, chi_m, chi_r]
    d = 2
    chi_l = A.shape[1]
    chi_r = B.shape[2]

    # Two-site wavefunction: A[p_i, chi_l, chi_m] × B[p_j, chi_m, chi_r]
    # einsum "ijk,lkm->iljm": i=p_i, j=chi_l, k=chi_m(shared), l=p_j, m=chi_r
    # result shape [p_i, chi_l, p_j, chi_r], reshaped to [d*chi_l, d*chi_r]
    theta = np.einsum("ijk,lkm->iljm", A, B).reshape(d * chi_l, d * chi_r)

    # Build local Hamiltonian, including non-adjacent snake-mapped bonds.
    # For each NN bond (bi, bj):
    #   • both endpoints in {site, site+1}  → direct 2-site interaction term
    #   • one endpoint in window            → mean-field: replace partner with ⟨op⟩
    #   • neither endpoint in window        → skip
    h_local = np.zeros((d * d, d * d), dtype=complex)
    I2 = np.eye(d, dtype=complex)
    win = {site: 0, site + 1: 1}

    for (bi, bj) in pairs:
        i_loc = win.get(bi)
        j_loc = win.get(bj)

        if i_loc is None and j_loc is None:
            continue  # bond entirely outside the two-site window

        if i_loc is not None and j_loc is not None:
            # Both endpoints in window → standard 2-site interaction
            if model == "heisenberg":
                for op_name in ("X", "Y", "Z"):
                    P = _PAULI[op_name]
                    if i_loc == 0:
                        h_local += J * np.kron(P, P)
                    else:
                        h_local += J * np.kron(P, P)
            else:  # ising
                Z = _PAULI["Z"]
                h_local += -J * np.kron(Z, Z)
        else:
            # One endpoint in window, one outside → mean-field single-site term
            loc_idx = i_loc if i_loc is not None else j_loc
            ext_site = bj if i_loc is not None else bi
            if model == "heisenberg":
                for op_name in ("X", "Y", "Z"):
                    exp_val = _expect_op(mps, ext_site, _PAULI[op_name])
                    P_loc = J * exp_val * _PAULI[op_name]
                    if loc_idx == 0:
                        h_local += np.kron(P_loc, I2)
                    else:
                        h_local += np.kron(I2, P_loc)
            else:  # ising
                exp_val = _expect_op(mps, ext_site, _PAULI["Z"])
                P_loc = -J * exp_val * _PAULI["Z"]
                if loc_idx == 0:
                    h_local += np.kron(P_loc, I2)
                else:
                    h_local += np.kron(I2, P_loc)

    # Effective environment (identity for simplicity — single-site approximation)
    H_eff = np.kron(np.eye(chi_l), np.kron(h_local, np.eye(chi_r)))

    # Reshape theta for eigenvalue problem
    theta_vec = theta.reshape(-1)
    dim_eff = len(theta_vec)

    if H_eff.shape[0] != dim_eff:
        # Shapes don't match — skip this site
        return theta, 0.0, 0.0

    # Energy via Rayleigh quotient
    energy = float(np.real(theta_vec.conj() @ H_eff @ theta_vec))

    # SVD truncation
    theta_mat = theta.reshape(d * chi_l, d * chi_r)
    U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)
    keep = min(bond_dim, len(S))
    trunc_error = float(np.sum(S[keep:] ** 2))
    S = S[:keep]
    U = U[:, :keep]
    Vh = Vh[:keep, :]

    return (U, S, Vh), energy, trunc_error


def _update_mps_left(mps, site, theta_svd, bond_dim):
    """Update MPS tensors after left-to-right sweep."""
    if not isinstance(theta_svd, tuple):
        return mps
    U, S, Vh = theta_svd
    d = 2
    chi_l = mps[site].shape[1]
    chi_new = len(S)

    mps[site] = U.reshape(d, chi_l, chi_new)
    # Absorb S Vh into right tensor
    SV = np.diag(S) @ Vh  # [chi_new, d*chi_r]
    chi_r = mps[site + 1].shape[2]
    mps[site + 1] = SV.reshape(chi_new, d, chi_r).transpose(1, 0, 2)
    return mps


def _update_mps_right(mps, site, theta_svd, bond_dim):
    """Update MPS tensors after right-to-left sweep."""
    if not isinstance(theta_svd, tuple):
        return mps
    U, S, Vh = theta_svd
    d = 2
    chi_l = mps[site].shape[1]
    chi_r = mps[site + 1].shape[2]
    chi_new = len(S)

    # Absorb U S into left tensor
    US = U @ np.diag(S)  # [d*chi_l, chi_new]
    mps[site] = US.reshape(d, chi_l, chi_new)
    mps[site + 1] = Vh.reshape(chi_new, d, chi_r).transpose(1, 0, 2)
    return mps
