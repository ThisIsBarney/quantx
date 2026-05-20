"""M119: Bosonic codes — cat qubits and GKP (Gottesman-Kitaev-Preskill) codes.

Provides hardware-biased-noise bosonic encodings that are at the centre of
near-term fault-tolerant quantum computing roadmaps (AWS, Yale cat-qubit,
INRIA/Alice&Bob).

Cat qubits
----------
A *Kerr-cat qubit* encodes a logical qubit in the two coherent-state
superpositions |C_α±⟩ ∝ |α⟩ ± |−α⟩ inside a driven Kerr oscillator:

    H = −K a†²a² + ε₂(a†² + a²)

Bit-flip errors (X_L) are exponentially suppressed ~ exp(−2|α|²) while
phase-flip errors (Z_L) scale polynomially, yielding a *biased* noise
channel ideal for concatenation with a repetition code.

GKP codes
---------
The Gottesman-Kitaev-Preskill code encodes a qubit in a harmonic oscillator
whose Hilbert space is infinite-dimensional (or truncated at N_max photons).
The logical code words are superpositions of squeezed states on a square
phase-space lattice with spacing √π.  We implement the *finite-energy*
(approximate) GKP state with Gaussian envelope parameter δ and the
small-big-small (SBS) correction sequence for shift errors.

Classes
-------
BosonicError
CatNoiseModel
CatQubit
GKPState
GKPDecoder
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "BosonicError",
    "CatNoiseModel",
    "CatQubit",
    "GKPState",
    "GKPDecoder",
]


class BosonicError(ValueError):
    """Raised for invalid bosonic-code requests."""


# ── Fock-space helpers ────────────────────────────────────────────────────────

def _annihilation(N: int) -> np.ndarray:
    """Return the N×N matrix representation of the bosonic annihilation operator a."""
    return np.diag(np.sqrt(np.arange(1, N, dtype=float)), k=1)


def _number(N: int) -> np.ndarray:
    """Return the number operator n̂ = a†a in the N-Fock truncation."""
    return np.diag(np.arange(N, dtype=float))


def _coherent_state(alpha: complex, N: int) -> np.ndarray:
    """Fock-space representation of the coherent state |α⟩ (normalised)."""
    ns = np.arange(N, dtype=float)
    # c_n = e^{-|α|²/2} α^n / sqrt(n!)
    log_c = -0.5 * abs(alpha) ** 2 + ns * np.log(abs(alpha) + 1e-300) + 1j * np.angle(alpha) * ns
    log_c -= 0.5 * np.array([sum(math.log(k) for k in range(1, int(n) + 1)) for n in ns])
    c = np.exp(log_c)
    c /= np.linalg.norm(c)
    return c.astype(complex)


def _wigner(state: np.ndarray, xvec: np.ndarray, pvec: np.ndarray) -> np.ndarray:
    """Wigner function W(x, p) via the Fock-state expansion.

    Uses the iterative formula for the Wigner function matrix elements
    W_{mn}(α) where α = (x + ip)/√2.
    """
    N = len(state)
    rho = np.outer(state, state.conj())
    X, P = np.meshgrid(xvec, pvec, indexing="ij")
    alpha = (X + 1j * P) / math.sqrt(2)

    # Build Wigner function using the laguerre-based formula
    # W(α) = (2/π) e^{-2|α|²} Σ_{mn} ρ_{mn} (-1)^m f_{mn}(α)
    # For computational efficiency we use the iterative method.
    W = np.zeros(X.shape, dtype=complex)
    for m in range(N):
        for n in range(N):
            if abs(rho[m, n]) < 1e-14:
                continue
            W += rho[m, n] * _wigner_element(m, n, alpha)
    return (2.0 / math.pi * np.real(W))


def _wigner_element(m: int, n: int, alpha: np.ndarray) -> np.ndarray:
    """W_{mn}(α) = (2/π)(−1)^m √(m!/n!) e^{-2|α|²} α^{n-m} L_m^{n-m}(4|α|²)."""
    # Ensure m ≤ n (use conjugate symmetry otherwise)
    swap = m > n
    if swap:
        m, n = n, m
        alpha = alpha.conj()
    r2 = np.abs(alpha) ** 2
    log_prefactor = (
        math.lgamma(m + 1) - math.lgamma(n + 1)
    ) / 2.0
    prefactor = (-1) ** m * math.exp(log_prefactor) * np.exp(-2 * r2)
    power = (2 * alpha) ** (n - m)
    laguerre = _assoc_laguerre_vec(m, n - m, 4 * r2)
    result = prefactor * power * laguerre
    if swap:
        result = result.conj()
    return result


def _assoc_laguerre_vec(n: int, k: int, x: np.ndarray) -> np.ndarray:
    """Associated Laguerre polynomial L_n^k(x) via recurrence."""
    if n == 0:
        return np.ones_like(x)
    L_prev = np.ones_like(x, dtype=complex)
    L_curr = 1.0 + k - x
    for i in range(1, n):
        L_next = ((2 * i + 1 + k - x) * L_curr - (i + k) * L_prev) / (i + 1)
        L_prev, L_curr = L_curr, L_next
    return L_curr


# ── CatNoiseModel ─────────────────────────────────────────────────────────────

@dataclass
class CatNoiseModel:
    """Noise model for a Kerr-cat qubit.

    Attributes
    ----------
    kappa_1 : float
        Single-photon loss rate (κ₁).  Sets the overall photon-loss timescale.
    kappa_phi : float
        Pure dephasing rate (κ_φ) in the oscillator frame.
    alpha_sq : float
        Cat-state amplitude squared |α|².  Controls the biased-noise ratio.

    Notes
    -----
    The bit-flip rate scales as exp(−2|α|²·κ₁/κ₂) and the phase-flip rate
    scales as |α|²·(κ₁ + κ_φ) (linear in cat size).
    """
    kappa_1: float = 0.01
    kappa_phi: float = 0.001
    alpha_sq: float = 4.0

    def __post_init__(self) -> None:
        if self.kappa_1 < 0:
            raise BosonicError(f"kappa_1 must be ≥ 0, got {self.kappa_1}")
        if self.kappa_phi < 0:
            raise BosonicError(f"kappa_phi must be ≥ 0, got {self.kappa_phi}")
        if self.alpha_sq <= 0:
            raise BosonicError(f"alpha_sq must be > 0, got {self.alpha_sq}")

    @property
    def bit_flip_rate(self) -> float:
        """Logical X error rate ~ exp(−2|α|²)·κ₁ (exponentially suppressed)."""
        return math.exp(-2.0 * self.alpha_sq) * self.kappa_1

    @property
    def phase_flip_rate(self) -> float:
        """Logical Z error rate ~ |α|²·(κ₁ + κ_φ) (linear in cat size)."""
        return self.alpha_sq * (self.kappa_1 + self.kappa_phi)

    @property
    def bias_ratio(self) -> float:
        """η = phase_flip_rate / bit_flip_rate (should be ≫ 1 for a useful cat)."""
        bf = self.bit_flip_rate
        if bf == 0.0:
            return float("inf")
        return self.phase_flip_rate / bf


# ── CatQubit ──────────────────────────────────────────────────────────────────

class CatQubit:
    """Kerr-cat qubit in a truncated Fock space.

    The Kerr-cat Hamiltonian is

        H = −K a†²a² + ε₂(a†² + a²)

    which stabilises a two-component cat state with amplitude α = √(ε₂/K).

    Parameters
    ----------
    K : float
        Kerr nonlinearity (anharmonicity).  Must be > 0.
    eps2 : float
        Two-photon drive amplitude.  Must be > 0.
    N_max : int
        Fock-space truncation.  Should satisfy N_max ≫ 2|α|² + 10.
    noise : CatNoiseModel | None
        Optional noise model.  If None, uses CatNoiseModel defaults with
        alpha_sq derived from eps2/K.
    """

    def __init__(
        self,
        K: float = 1.0,
        eps2: float = 4.0,
        N_max: int = 30,
        noise: Optional[CatNoiseModel] = None,
    ) -> None:
        if K <= 0:
            raise BosonicError(f"K must be > 0, got {K}")
        if eps2 <= 0:
            raise BosonicError(f"eps2 must be > 0, got {eps2}")
        if not isinstance(N_max, int) or N_max < 4:
            raise BosonicError(f"N_max must be an integer ≥ 4, got {N_max}")

        self.K = float(K)
        self.eps2 = float(eps2)
        self.N_max = N_max
        self.alpha = math.sqrt(eps2 / K)  # steady-state cat amplitude

        alpha_sq = eps2 / K
        self.noise = noise if noise is not None else CatNoiseModel(alpha_sq=alpha_sq)

        # Build Fock-space operators
        a = _annihilation(N_max)
        adag = a.T.conj()
        n = _number(N_max)

        # Kerr-cat Hamiltonian
        self._H = -K * (adag @ adag @ a @ a) + eps2 * (adag @ adag + a @ a)

        # Logical operators
        self._a = a
        self._adag = adag
        self._n = n

    # ── Encoded states ────────────────────────────────────────────────────────

    def encode_0(self) -> np.ndarray:
        """Logical |0_L⟩ = normalised even cat state |C_α+⟩."""
        plus = _coherent_state(self.alpha, self.N_max)
        minus = _coherent_state(-self.alpha, self.N_max)
        state = plus + minus
        norm = np.linalg.norm(state)
        if norm < 1e-12:
            raise BosonicError("Cat amplitude too small: |0_L⟩ is unnormalisable.")
        return state / norm

    def encode_1(self) -> np.ndarray:
        """Logical |1_L⟩ = normalised odd cat state |C_α−⟩."""
        plus = _coherent_state(self.alpha, self.N_max)
        minus = _coherent_state(-self.alpha, self.N_max)
        state = plus - minus
        norm = np.linalg.norm(state)
        if norm < 1e-12:
            raise BosonicError("Cat amplitude too small: |1_L⟩ is unnormalisable.")
        return state / norm

    # ── Logical operators ─────────────────────────────────────────────────────

    @property
    def logical_z(self) -> np.ndarray:
        """Logical Z operator: Z_L = exp(iπ n̂)  (photon-number parity)."""
        return np.diag(np.exp(1j * math.pi * np.arange(self.N_max, dtype=float)))

    @property
    def logical_x(self) -> np.ndarray:
        """Logical X operator: swaps |C_α+⟩ ↔ |C_α−⟩ within the code subspace.

        Constructs the unitary X_L = |1_L⟩⟨0_L| + |0_L⟩⟨1_L| + (I − P_code),
        which acts as Pauli X on the {|0_L⟩, |1_L⟩} code subspace and as the
        identity on its orthogonal complement.  This is exact for any α.
        """
        psi0 = self.encode_0()   # |0_L⟩ = |C_α+⟩
        psi1 = self.encode_1()   # |1_L⟩ = |C_α−⟩
        # Code-space projector P = |0_L⟩⟨0_L| + |1_L⟩⟨1_L|
        P_code = np.outer(psi0, psi0.conj()) + np.outer(psi1, psi1.conj())
        # X_L maps |0_L⟩↔|1_L⟩ and is identity outside the code space
        X_L = (
            np.outer(psi1, psi0.conj())
            + np.outer(psi0, psi1.conj())
            + (np.eye(self.N_max) - P_code)
        )
        return X_L

    @property
    def hamiltonian(self) -> np.ndarray:
        """Full Kerr-cat Hamiltonian matrix (N_max × N_max)."""
        return self._H.copy()

    # ── Observables ───────────────────────────────────────────────────────────

    def mean_photon_number(self, state: np.ndarray) -> float:
        """⟨n̂⟩ for a Fock-space state vector."""
        state = np.asarray(state, dtype=complex)
        return float(np.real(state.conj() @ self._n @ state))

    def photon_number_distribution(self, state: np.ndarray) -> np.ndarray:
        """P(n) = |⟨n|ψ⟩|² for n = 0, …, N_max−1."""
        state = np.asarray(state, dtype=complex)
        return np.abs(state) ** 2

    def wigner_function(
        self,
        state: np.ndarray,
        xvec: Optional[np.ndarray] = None,
        pvec: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wigner function W(x, p) on a phase-space grid.

        Returns
        -------
        xvec : np.ndarray, shape (Nx,)
        pvec : np.ndarray, shape (Np,)
        W    : np.ndarray, shape (Nx, Np)
        """
        if xvec is None:
            xvec = np.linspace(-3 * self.alpha - 2, 3 * self.alpha + 2, 31)
        if pvec is None:
            pvec = np.linspace(-3 * self.alpha - 2, 3 * self.alpha + 2, 31)
        state = np.asarray(state, dtype=complex)
        W = _wigner(state, xvec, pvec)
        return xvec, pvec, W

    def bit_flip_time(self) -> float:
        """Characteristic logical bit-flip time T_X ~ 1/Γ_X."""
        rate = self.noise.bit_flip_rate
        if rate == 0.0:
            return float("inf")
        return 1.0 / rate

    def phase_flip_rate(self) -> float:
        """Logical phase-flip rate Γ_Z (grows linearly with α²)."""
        return self.noise.phase_flip_rate

    # ── Ground state ──────────────────────────────────────────────────────────

    def ground_state(self) -> np.ndarray:
        """Ground state of the Kerr-cat Hamiltonian (lowest eigenvalue)."""
        eigvals, eigvecs = np.linalg.eigh(self._H)
        return eigvecs[:, 0]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _displacement(self, beta: float) -> np.ndarray:
        """Displacement operator D(β) = exp(β a† − β* a) in Fock basis."""
        # D(β) = exp(β a† − β a)  (β real here)
        exponent = beta * self._adag - beta * self._a
        return _matrix_exp(exponent)

    def __repr__(self) -> str:
        return (
            f"CatQubit(K={self.K}, eps2={self.eps2}, "
            f"alpha={self.alpha:.3f}, N_max={self.N_max})"
        )


def _matrix_exp(M: np.ndarray) -> np.ndarray:
    """Matrix exponential via eigendecomposition (safe for small Hermitian-ish M)."""
    # Use scipy if available; otherwise fall back to numpy eigendecomposition
    try:
        from scipy.linalg import expm  # type: ignore
        return expm(M)
    except ImportError:
        pass
    # Fallback: use Padé approximation via repeated squaring (numpy only)
    # For anti-Hermitian M = iA use eigh on A.
    eigvals, eigvecs = np.linalg.eig(M)
    return (eigvecs * np.exp(eigvals)) @ np.linalg.inv(eigvecs)


# ── GKPState ──────────────────────────────────────────────────────────────────

class GKPState:
    """Finite-energy GKP (Gottesman-Kitaev-Preskill) code state.

    The ideal GKP code encodes a qubit in a harmonic oscillator with
    stabilisers S_q = e^{i2√π q̂} and S_p = e^{-i2√π p̂}.  The finite-energy
    approximation uses a Gaussian envelope with width δ in Fock space.

    Parameters
    ----------
    delta : float
        Envelope/squeezing parameter δ > 0.  Smaller δ → better code words
        but more photons.  Physical squeezing in dB ≈ −20 log₁₀(δ/√(2π)).
    N_max : int
        Fock-space truncation.

    Notes
    -----
    The code words are constructed as a superposition of displaced squeezed
    vacuum states at lattice points n√π in phase space.
    """

    _SQRT_PI = math.sqrt(math.pi)

    def __init__(self, delta: float = 0.3, N_max: int = 60) -> None:
        if delta <= 0 or delta >= 1:
            raise BosonicError(
                f"delta must be in (0, 1) for a valid GKP state, got {delta}"
            )
        if not isinstance(N_max, int) or N_max < 8:
            raise BosonicError(f"N_max must be an integer ≥ 8, got {N_max}")

        self.delta = float(delta)
        self.N_max = N_max
        self._n_terms = max(5, int(4 / delta))  # number of lattice terms to include

        # Pre-build operators
        a = _annihilation(N_max)
        self._a = a
        self._adag = a.T.conj()
        self._x_op = (a + a.T.conj()) / math.sqrt(2)  # q̂ = (a+a†)/√2
        self._p_op = (a - a.T.conj()) / (1j * math.sqrt(2))  # p̂ = i(a†-a)/√2

        # Cache the code words
        self._code0: Optional[np.ndarray] = None
        self._code1: Optional[np.ndarray] = None

    def _squeezed_vacuum(self, r: float) -> np.ndarray:
        """Squeezed vacuum |r⟩ in Fock basis (squeezing parameter r)."""
        N = self.N_max
        # c_{2n} = (-1)^n sqrt((2n)!) / (2^n n!) * tanh(r)^n / cosh(r)^{1/2}
        # Only even Fock states are populated
        psi = np.zeros(N, dtype=complex)
        tanh_r = math.tanh(r)
        cosh_r = math.cosh(r)
        for n in range(N // 2):
            two_n = 2 * n
            log_c = (
                0.5 * math.lgamma(two_n + 1)
                - n * math.log(2)
                - math.lgamma(n + 1)
                + n * math.log(abs(tanh_r) + 1e-300)
                - 0.5 * math.log(cosh_r)
            )
            psi[two_n] = ((-1) ** n) * math.exp(log_c)
        norm = np.linalg.norm(psi)
        if norm > 0:
            psi /= norm
        return psi

    def _displace(self, state: np.ndarray, beta: complex) -> np.ndarray:
        """Apply displacement D(β) to a Fock-space state."""
        # D(β) = exp(β a† − β* a)
        M = beta * self._adag - beta.conjugate() * self._a
        return _matrix_exp(M) @ state

    def _envelope(self, n: int) -> float:
        """Gaussian envelope weight e^{-δ²π n²/2} for lattice site n."""
        return math.exp(-0.5 * self.delta ** 2 * math.pi * n ** 2)

    def code_word_0(self) -> np.ndarray:
        """|0_L⟩ = GKP logical zero code word."""
        if self._code0 is None:
            self._code0 = self._build_code_word(logical=0)
        return self._code0.copy()

    def code_word_1(self) -> np.ndarray:
        """|1_L⟩ = GKP logical one code word."""
        if self._code1 is None:
            self._code1 = self._build_code_word(logical=1)
        return self._code1.copy()

    def _build_code_word(self, logical: int) -> np.ndarray:
        """Build |logical_L⟩ as a sum of displaced squeezed states."""
        r = -math.log(self.delta)  # squeezing level: e^{-r} = δ
        sq_vac = self._squeezed_vacuum(r)

        psi = np.zeros(self.N_max, dtype=complex)
        M = self._n_terms
        for n in range(-M, M + 1):
            # GKP lattice: displace by (2n + logical) * √π in q
            shift = (2 * n + logical) * self._SQRT_PI
            displaced = self._displace(sq_vac, shift / math.sqrt(2))  # β = shift/√2
            psi += self._envelope(2 * n + logical) * displaced

        norm = np.linalg.norm(psi)
        if norm < 1e-10:
            raise BosonicError("GKP code word has negligible norm — increase N_max.")
        return psi / norm

    def squeezing_db(self) -> float:
        """Approximate squeezing in dB: s ≈ −20 log₁₀(δ/√(2π))."""
        return -20.0 * math.log10(self.delta / math.sqrt(2 * math.pi))

    def mean_photon_number(self, state: np.ndarray) -> float:
        """⟨n̂⟩ for a Fock-space state."""
        n_op = _number(self.N_max)
        return float(np.real(state.conj() @ n_op @ state))

    def overlap(self, state_a: np.ndarray, state_b: np.ndarray) -> complex:
        """Compute ⟨a|b⟩."""
        return complex(np.dot(state_a.conj(), state_b))

    def logical_fidelity(self, noisy_state: np.ndarray, logical: int = 0) -> float:
        """Fidelity |⟨logical_L|noisy⟩|² after a shift error."""
        target = self.code_word_0() if logical == 0 else self.code_word_1()
        return float(abs(self.overlap(target, noisy_state)) ** 2)

    def __repr__(self) -> str:
        return f"GKPState(delta={self.delta}, N_max={self.N_max}, squeezing={self.squeezing_db():.1f} dB)"


# ── GKPDecoder ────────────────────────────────────────────────────────────────

class GKPDecoder:
    """Small-Big-Small (SBS) shift error correction for GKP codes.

    The SBS sequence applies alternating small and big displacements to map
    shift errors back toward the lattice points:

        D_small(−ξ) · D_big(+ξ + correction) · D_small(−ξ)

    where the correction direction is estimated from a syndrome measurement
    of the shift-error magnitude.

    Parameters
    ----------
    gkp : GKPState
        The GKP state object (defines δ, N_max, lattice).
    n_rounds : int
        Number of SBS correction rounds to apply.
    """

    def __init__(self, gkp: GKPState, n_rounds: int = 1) -> None:
        if n_rounds < 1:
            raise BosonicError(f"n_rounds must be ≥ 1, got {n_rounds}")
        self.gkp = gkp
        self.n_rounds = n_rounds

    def apply_shift(self, state: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
        """Apply a random Gaussian shift error in q and p.

        Parameters
        ----------
        state : np.ndarray
            Input Fock-space state.
        sigma : float
            Standard deviation of the Gaussian shift distribution.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Shifted Fock-space state (normalised).
        """
        dq = float(rng.normal(0, sigma))
        dp = float(rng.normal(0, sigma))
        beta = (dq + 1j * dp) / math.sqrt(2)
        shifted = self.gkp._displace(state, beta)
        norm = np.linalg.norm(shifted)
        return shifted / norm if norm > 0 else shifted

    def correct(self, state: np.ndarray) -> np.ndarray:
        """Apply SBS correction to estimate and remove shift errors.

        Parameters
        ----------
        state : np.ndarray
            Noisy Fock-space state.

        Returns
        -------
        np.ndarray
            Corrected state (normalised).
        """
        corrected = state.copy()
        for _ in range(self.n_rounds):
            corrected = self._sbs_round(corrected)
        return corrected

    def _sbs_round(self, state: np.ndarray) -> np.ndarray:
        """One SBS round: estimate residual shift and apply correction."""
        # Estimate shift from the expectation value of q and p modulo √π
        n_op = _number(self.gkp.N_max)
        a = self.gkp._a
        adag = self.gkp._adag

        x_op = (a + adag) / math.sqrt(2)
        p_op = (a - adag) / (1j * math.sqrt(2))

        exp_x = float(np.real(state.conj() @ x_op @ state))
        exp_p = float(np.real(state.conj() @ p_op @ state))

        # Round to nearest GKP lattice point (multiples of √π)
        sqrt_pi = math.sqrt(math.pi)
        shift_q = exp_x - round(exp_x / sqrt_pi) * sqrt_pi
        shift_p = exp_p - round(exp_p / sqrt_pi) * sqrt_pi

        # Apply correction displacement
        beta = -(shift_q + 1j * shift_p) / math.sqrt(2)
        corrected = self.gkp._displace(state, beta)
        norm = np.linalg.norm(corrected)
        return corrected / norm if norm > 0 else corrected

    def logical_error_rate(
        self,
        sigma: float,
        n_shots: int = 200,
        seed: Optional[int] = None,
    ) -> float:
        """Estimate the logical error rate after SBS correction.

        Applies random shift errors of standard deviation σ to |0_L⟩,
        corrects with SBS, and measures fidelity with the original code word.

        Parameters
        ----------
        sigma : float
            Shift noise standard deviation (in units of 1/√2).
        n_shots : int
            Number of Monte Carlo samples.
        seed : int | None
            Random seed.

        Returns
        -------
        float
            Estimated logical error rate (1 - mean fidelity).
        """
        if sigma < 0:
            raise BosonicError(f"sigma must be ≥ 0, got {sigma}")
        if n_shots < 1:
            raise BosonicError(f"n_shots must be ≥ 1, got {n_shots}")

        rng = np.random.default_rng(seed)
        code0 = self.gkp.code_word_0()
        total_fidelity = 0.0

        for _ in range(n_shots):
            noisy = self.apply_shift(code0, sigma, rng)
            corrected = self.correct(noisy)
            fid = self.gkp.logical_fidelity(corrected, logical=0)
            total_fidelity += fid

        mean_fidelity = total_fidelity / n_shots
        return float(1.0 - mean_fidelity)
