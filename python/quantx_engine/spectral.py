"""Spectral functions and Krylov time evolution for the QuantX engine (M73).

Two main components:

**SpectralResult** — wraps the raw output of :meth:`Engine.run_spectral` into a
typed, queryable object.  Contains the dynamical structure factor
S(k, ω), k-point grid, frequency axis, and spectral gap.

**KrylovEvolver** — pure-Python Arnoldi-Krylov implementation for
time-evolution without full diagonalization:

* ``evolve(H, psi0, t)``   → |ψ(t)⟩ = e^{−iHt}|ψ₀⟩
* ``imaginary_evolve(H, psi0, tau)`` → normalised e^{−Hτ}|ψ₀⟩
  (converges to the ground state)

Both methods scale as O(N·m²) where N = Hilbert-space dimension and m is
the Krylov dimension (default 30), vs O(N³) for full diagonalisation.

Quick start::

    import numpy as np
    from quantx_engine.spectral import KrylovEvolver
    from quantx_engine.lanczos import SparseHamiltonian, LanczosSolver

    # Real-time evolution: 4-site Heisenberg, Néel initial state
    H = SparseHamiltonian.heisenberg_1d(4, J=1.0)
    neel = np.zeros(16, dtype=complex)
    neel[0b0101] = 1.0          # |↑↓↑↓⟩ in the binary basis
    evolver = KrylovEvolver(m=20)
    psi_t = evolver.evolve(H, neel, t=2.0)
    print(abs(neel @ psi_t.conj())**2)  # autocorrelation |⟨Néel|ψ(t)⟩|²

    # Ground-state projection via imaginary-time evolution
    psi_gs = evolver.imaginary_evolve(H, neel, tau=5.0)
    e0_krylov = float(np.real(psi_gs.conj() @ H.matvec(psi_gs)))

    # Reference
    e0_lanczos = LanczosSolver(k=1, seed=0).solve(H).ground_state_energy()
    print(abs(e0_krylov - e0_lanczos))  # < 1e-6
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

if TYPE_CHECKING:  # pragma: no cover
    from quantx_engine.lanczos import SparseHamiltonian

__all__ = [
    "SpectralError",
    "SpectralResult",
    "KrylovResult",
    "KrylovEvolver",
]


class SpectralError(ValueError):
    """Raised for invalid spectral or Krylov parameters."""


# ---------------------------------------------------------------------------
# SpectralResult
# ---------------------------------------------------------------------------

@dataclass
class SpectralResult:
    """Parsed output of a spectral-function (dynamical structure factor) run.

    Attributes
    ----------
    omega:
        Frequency axis, shape (n_omega,).
    S_kw:
        Dynamical structure factor S(k, ω), shape (n_kpoints, n_omega).
        Each row is the spectral weight at one k-point.
    kpoints:
        Momentum grid, shape (n_kpoints,).  In units of 2π/a (lattice spacing
        a=1 by default).
    gap:
        Spectral gap = minimum excitation energy.  ``None`` if not available
        in the backend output.
    raw:
        The original raw dict returned by ``Engine.run_spectral()``.

    Examples
    --------
    Build directly from a mock raw dict (e.g. in tests)::

        result = SpectralResult.from_engine_output({
            "omega": [0.0, 0.5, 1.0],
            "S_kw": [[0.0, 1.0, 0.0]],
            "kpoints": [0.0],
        })
    """
    omega: "np.ndarray"      # shape (n_omega,)
    S_kw: "np.ndarray"       # shape (n_kpoints, n_omega)
    kpoints: "np.ndarray"    # shape (n_kpoints,)
    gap: "float | None" = None
    raw: "dict[str, Any]" = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_engine_output(cls, raw: "dict[str, Any]") -> "SpectralResult":
        """Parse a raw ``Engine.run_spectral()`` result dict.

        Accepts multiple naming conventions used by different C++ backend
        versions (``omega`` / ``frequencies``, ``S_kw`` / ``spectral_weight``
        / ``dsf``, ``kpoints`` / ``k_points`` / ``k_grid``).

        Parameters
        ----------
        raw:
            Dict as returned by ``Engine.run_spectral()``.

        Returns
        -------
        SpectralResult
        """
        if not _HAS_NUMPY:
            raise SpectralError("numpy is required for SpectralResult")

        # ── Frequency axis ────────────────────────────────────────────────
        # Include "omega_vals" which is the key emitted by the native
        # C++ spectral_result_to_json serializer.
        omega_raw = (
            raw.get("omega_vals")
            or raw.get("omega")
            or raw.get("frequencies")
            or raw.get("freq")
            or []
        )
        omega = np.asarray(omega_raw, dtype=float)

        # ── k-points ──────────────────────────────────────────────────────
        # "k_vals" is emitted by the native serializer.
        kpoints_raw = (
            raw.get("k_vals")
            or raw.get("kpoints")
            or raw.get("k_points")
            or raw.get("k_grid")
            or raw.get("kvec")
            or []
        )
        kpoints = np.asarray(kpoints_raw, dtype=float)

        # ── S(k, ω) ───────────────────────────────────────────────────────
        # "spectral" is emitted by the native serializer.
        skw_raw = (
            raw.get("S_kw")
            or raw.get("spectral")
            or raw.get("spectral_weight")
            or raw.get("dsf")
            or raw.get("Skw")
            or []
        )
        if skw_raw:
            S_kw = np.asarray(skw_raw, dtype=float)
            if S_kw.ndim == 1:
                S_kw = S_kw[np.newaxis, :]  # treat as single k-point
        else:
            n_k = max(len(kpoints), 1)
            n_w = max(len(omega), 1)
            S_kw = np.zeros((n_k, n_w))

        # ── Spectral gap ──────────────────────────────────────────────────
        # Use explicit key iteration to avoid the falsy-or bug where a
        # physically meaningful gap=0.0 (gapless system) would be discarded.
        gap: "float | None" = None
        for _gap_key in ("gap", "spectral_gap", "energy_gap"):
            if _gap_key in raw:
                gap = float(raw[_gap_key])
                break

        return cls(omega=omega, S_kw=S_kw, kpoints=kpoints, gap=gap, raw=raw)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def n_kpoints(self) -> int:
        """Number of k-points."""
        return int(self.S_kw.shape[0])

    @property
    def n_omega(self) -> int:
        """Number of frequency points."""
        return int(self.S_kw.shape[1])

    def integrated_weight(self) -> "np.ndarray":
        """Return ∫ S(k, ω) dω for each k-point (trapezoidal rule)."""
        if not _HAS_NUMPY:
            raise SpectralError("numpy required")
        if len(self.omega) < 2:
            return self.S_kw.sum(axis=1)
        # np.trapezoid was added in NumPy 2.0 (np.trapz deprecated/removed)
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        return _trapz(self.S_kw, self.omega, axis=1)

    def peak_frequencies(self) -> "np.ndarray":
        """Return the frequency of the maximum spectral weight per k-point."""
        if not _HAS_NUMPY:
            raise SpectralError("numpy required")
        if len(self.omega) == 0:
            return np.array([])
        return self.omega[np.argmax(self.S_kw, axis=1)]

    def plot_dsf(self, ax: Any = None, **kwargs: Any) -> Any:
        """Plot S(k, ω) as a false-colour map.

        Parameters
        ----------
        ax:
            Matplotlib axes object.  If ``None``, creates a new figure.
        **kwargs:
            Passed to ``ax.pcolormesh``.

        Returns
        -------
        The ``AxesImage`` returned by ``pcolormesh``.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        SpectralError
            If S_kw has no data.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_dsf(). "
                "Install with: pip install matplotlib"
            ) from None
        if ax is None:
            _, ax = plt.subplots()
        im = ax.pcolormesh(self.omega, self.kpoints, self.S_kw, **kwargs)
        ax.set_xlabel("ω")
        ax.set_ylabel("k")
        ax.set_title("Dynamical Structure Factor S(k, ω)")
        return im

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SpectralResult(n_kpoints={self.n_kpoints}, n_omega={self.n_omega}, "
            f"gap={self.gap})"
        )


# ---------------------------------------------------------------------------
# KrylovResult
# ---------------------------------------------------------------------------

@dataclass
class KrylovResult:
    """Result of a Krylov time-evolution run.

    Attributes
    ----------
    times:
        Time points at which snapshots were taken.
    states:
        List of state vectors |ψ(t)⟩ at each time point.
    autocorrelation:
        |⟨ψ₀|ψ(t)⟩|² at each time point.  1.0 at t=0, decays as the state
        evolves away from the initial state.
    """
    times: list[float]
    states: "list[np.ndarray]"
    autocorrelation: list[float]

    def __len__(self) -> int:
        return len(self.times)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_matvec_and_n(H: Any):
    """Return (matvec_callable, dimension) from a variety of H types."""
    if not _HAS_NUMPY:
        raise SpectralError("numpy required")
    if isinstance(H, np.ndarray):
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise SpectralError("Dense H must be a square 2D array")
        N = H.shape[0]
        return (lambda v: H @ v), N
    if hasattr(H, "n") and hasattr(H, "matvec"):
        return H.matvec, H.n
    raise SpectralError(
        "H must be a numpy array or an object with .n and .matvec attributes"
    )


def _expm_mv_real(H_m: "np.ndarray", t: float, e1: "np.ndarray") -> "np.ndarray":
    """Compute e^{t * H_m} @ e1 using numpy eigendecomposition.

    Works for both real-time (t complex) and imaginary-time (t real negative).
    H_m is a small (m×m) matrix.

    For Hermitian H_m (the common case — Arnoldi on Hermitian H produces a
    symmetric tridiagonal Hessenberg, i.e. Lanczos), ``np.linalg.eigh`` is
    used: it guarantees real eigenvalues and an orthogonal eigenvector matrix,
    so no explicit inversion is needed and numerical stability is improved.
    For non-Hermitian H_m (general case), falls back to ``np.linalg.eig``.
    """
    H_c = H_m.astype(complex)
    if np.allclose(H_c, H_c.conj().T, atol=1e-10):
        # Hermitian case: eigh gives real eigenvalues, unitary eigenvectors.
        # e^{t H_m} = V e^{t D} V†  (no explicit inversion needed).
        evals_real, evecs = np.linalg.eigh(H_c)
        exp_tD = np.exp(t * evals_real)          # t may be complex
        return evecs @ (exp_tD * (evecs.conj().T @ e1))
    else:
        # General (non-Hermitian) case.
        # H_m = V D V^{-1}  →  e^{t H_m} = V e^{t D} V^{-1}
        evals, evecs = np.linalg.eig(H_c)
        try:
            evecs_inv = np.linalg.inv(evecs)
        except np.linalg.LinAlgError:
            # Nearly singular: use pinv
            evecs_inv = np.linalg.pinv(evecs)
        exp_tD = np.exp(t * evals)
        return evecs @ (exp_tD * (evecs_inv @ e1))


def _arnoldi_from_v0(
    matvec: Any, v0: "np.ndarray", m: int
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, int]":
    """Arnoldi process starting from given v0."""
    N = len(v0)
    V = np.zeros((N, m + 1), dtype=complex)
    H_m = np.zeros((m + 1, m), dtype=complex)

    norm0 = np.linalg.norm(v0)
    if norm0 < 1e-14:
        raise SpectralError("Initial vector has near-zero norm")
    V[:, 0] = v0 / norm0

    m_eff = m
    for j in range(m):
        w = matvec(V[:, j])
        for i in range(j + 1):
            H_m[i, j] = np.dot(V[:, i].conj(), w)
            w -= H_m[i, j] * V[:, i]
        h_next = np.linalg.norm(w)
        H_m[j + 1, j] = h_next
        if h_next < 1e-14:
            m_eff = j + 1
            break
        V[:, j + 1] = w / h_next
    else:
        m_eff = m

    return V[:, :m_eff], H_m[:m_eff, :m_eff], V[:, 0] * norm0, m_eff


# ---------------------------------------------------------------------------
# KrylovEvolver
# ---------------------------------------------------------------------------

class KrylovEvolver:
    """Arnoldi-Krylov time evolution: e^{−iHt}|ψ₀⟩ without full diagonalisation.

    The algorithm builds an *m*-dimensional Krylov subspace
    ``span{|ψ₀⟩, H|ψ₀⟩, H²|ψ₀⟩, …, H^{m-1}|ψ₀⟩}`` via the Arnoldi
    recurrence, then computes the matrix exponential of the small (m×m)
    upper Hessenberg projection.  This requires only *m* matrix-vector
    products (each O(N·nnz)) instead of O(N³) for full diagonalisation.

    Parameters
    ----------
    m:
        Krylov subspace dimension.  Larger values are more accurate but
        more expensive.  m=30 is suitable for moderate accuracy on systems
        with N ≤ 10⁶.
    dt:
        Default time step for long-time evolution (used when ``dt`` is not
        specified in :meth:`evolve`).

    Examples
    --------
    >>> evolver = KrylovEvolver(m=20)
    >>> psi_t = evolver.evolve(H, psi0, t=1.0)   # single step
    >>> result = evolver.evolve_trajectory(H, psi0, t_max=5.0, n_steps=20)
    """

    def __init__(self, m: int = 30, dt: float = 0.1):
        if not _HAS_NUMPY:
            raise SpectralError("numpy is required for KrylovEvolver")
        if m < 1:
            raise SpectralError(f"m must be >= 1, got {m}")
        if dt <= 0.0:
            raise SpectralError(f"dt must be > 0, got {dt}")
        self.m = m
        self.dt = dt

    def _step(self, matvec: Any, psi: "np.ndarray", t: float) -> "np.ndarray":
        """Single Krylov step: approximate e^{−i·t·H}|ψ⟩."""
        N = len(psi)
        norm_psi = np.linalg.norm(psi)
        if norm_psi < 1e-14:
            return psi.copy()

        m = min(self.m, N)
        V, H_m, _, m_eff = _arnoldi_from_v0(matvec, psi, m)

        e1 = np.zeros(m_eff, dtype=complex)
        e1[0] = 1.0
        coeff = _expm_mv_real(H_m, -1j * t, e1)
        return norm_psi * (V @ coeff)

    def evolve(
        self,
        H: Any,
        psi0: "np.ndarray",
        t: float,
        dt: "float | None" = None,
    ) -> "np.ndarray":
        """Compute |ψ(t)⟩ = e^{−iHt}|ψ₀⟩.

        Parameters
        ----------
        H:
            Hamiltonian.  Accepts:

            * ``np.ndarray`` — dense (N×N) matrix.
            * :class:`~quantx_engine.lanczos.SparseHamiltonian` — sparse.
            * Any object with ``.n`` (int) and ``.matvec(v)`` attributes.
        psi0:
            Initial state vector, shape (N,).  Need not be normalised.
        t:
            Total evolution time.  Must be >= 0.
        dt:
            Time step for splitting long evolutions.  If ``None``, uses
            the value set at construction time.  A single Krylov step is
            used when ``t <= dt``.

        Returns
        -------
        np.ndarray
            State vector |ψ(t)⟩, shape (N,), complex128.

        Raises
        ------
        SpectralError
            On invalid parameters.
        """
        if t < 0.0:
            raise SpectralError(f"t must be >= 0, got {t}")
        matvec, _ = _get_matvec_and_n(H)
        psi = np.array(psi0, dtype=complex)
        if t == 0.0:
            return psi.copy()

        step = dt if dt is not None else self.dt
        if step <= 0.0:
            raise SpectralError(f"dt must be > 0, got {step}")

        remaining = t
        while remaining > 1e-14:
            s = min(step, remaining)
            psi = self._step(matvec, psi, s)
            remaining -= s
        return psi

    def imaginary_evolve(
        self,
        H: Any,
        psi0: "np.ndarray",
        tau: float,
        dt: "float | None" = None,
    ) -> "np.ndarray":
        """Compute e^{−Hτ}|ψ₀⟩ / ||e^{−Hτ}|ψ₀⟩||, converging to the ground state.

        Repeated application of the imaginary-time propagator
        e^{−Hτ} damps high-energy components: at large τ only the ground
        state survives.  The result is normalised at each step to prevent
        underflow.

        Parameters
        ----------
        H:
            Hamiltonian (same types as :meth:`evolve`).
        psi0:
            Initial state, shape (N,).  Should have non-zero overlap with
            the ground state.
        tau:
            Total imaginary time.  Larger values give better convergence to
            the ground state.
        dt:
            Imaginary-time step.  Defaults to :attr:`dt`.

        Returns
        -------
        np.ndarray
            Normalised approximation to the ground state, shape (N,).
        """
        if tau < 0.0:
            raise SpectralError(f"tau must be >= 0, got {tau}")
        matvec, _ = _get_matvec_and_n(H)
        psi = np.array(psi0, dtype=complex)
        norm0 = np.linalg.norm(psi)
        if norm0 < 1e-14:
            raise SpectralError("psi0 has near-zero norm")
        psi /= norm0
        if tau == 0.0:
            return psi.copy()

        step = dt if dt is not None else self.dt
        if step <= 0.0:
            raise SpectralError(f"dt must be > 0, got {step}")

        remaining = tau
        while remaining > 1e-14:
            s = min(step, remaining)
            N = len(psi)
            m = min(self.m, N)
            V, H_m, _, m_eff = _arnoldi_from_v0(matvec, psi, m)
            e1 = np.zeros(m_eff, dtype=complex)
            e1[0] = 1.0
            coeff = _expm_mv_real(H_m, -s, e1)  # exp(-tau * H_m)
            psi = V @ coeff
            norm = np.linalg.norm(psi)
            if norm < 1e-14:
                raise SpectralError(
                    "State decayed to zero during imaginary-time evolution. "
                    "Check that psi0 has overlap with the ground state."
                )
            psi /= norm
            remaining -= s
        return psi

    def evolve_trajectory(
        self,
        H: Any,
        psi0: "np.ndarray",
        t_max: float,
        n_steps: int = 20,
    ) -> KrylovResult:
        """Evolve and record state at n_steps equally-spaced time points.

        Parameters
        ----------
        H:
            Hamiltonian.
        psi0:
            Initial state.
        t_max:
            Maximum time.
        n_steps:
            Number of time points (including t=0).

        Returns
        -------
        KrylovResult
            Contains times, states, and autocorrelation function.
        """
        if n_steps < 2:
            raise SpectralError("n_steps must be >= 2")
        if t_max <= 0:
            raise SpectralError("t_max must be > 0")

        matvec, _ = _get_matvec_and_n(H)
        psi0_arr = np.array(psi0, dtype=complex)
        # Normalise psi0 so that the autocorrelation |⟨ψ₀|ψ(t)⟩|² is in [0,1].
        norm0 = np.linalg.norm(psi0_arr)
        if norm0 > 1e-14:
            psi0_arr = psi0_arr / norm0
        psi = psi0_arr.copy()

        times = [0.0]
        states = [psi.copy()]
        auto = [1.0]

        dt_traj = t_max / (n_steps - 1)
        for _ in range(n_steps - 1):
            psi = self._step(matvec, psi, dt_traj)
            t_now = times[-1] + dt_traj
            times.append(t_now)
            states.append(psi.copy())
            overlap = abs(np.dot(psi0_arr.conj(), psi)) ** 2
            auto.append(float(overlap))

        return KrylovResult(times=times, states=states, autocorrelation=auto)
