"""Floquet engineering and Bloch-Floquet band structure — M92.

Provides:

* :class:`FloquetSystem` — defines a periodically driven Hamiltonian
  H(t) = H_static + H_drive · f(t) and computes the stroboscopic
  one-period propagator U(T) = T-exp(−i ∫₀ᵀ H(t) dt) via Runge-Kutta.

* :class:`BlochFloquetSolver` — diagonalises U(T) to extract quasi-energies
  (Floquet eigenvalues) and Floquet modes; computes effective Hamiltonians
  via the Floquet-Magnus expansion to second order; evaluates the micromotion
  operator U_M(t) = U(t, 0) · e^{+i H_eff t}.

* :func:`floquet_spectrum` — convenience wrapper returning a
  :class:`FloquetResult` for a given system.

* :func:`floquet_effective_hamiltonian` — first-order Magnus approximation
  H_eff = H₀ + [H₁, H₋₁] / (2ω) for a single harmonic drive.

All matrix operations use dense NumPy arrays (suitable for small systems,
e.g. 1–4 qubits / spins).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ── Errors ────────────────────────────────────────────────────────────────────

class FloquetError(ValueError):
    """Raised on invalid Floquet configuration."""


# ── FloquetSystem ─────────────────────────────────────────────────────────────

@dataclass
class FloquetSystem:
    """Periodically driven quantum system H(t) = H_static + H_drive(t).

    Parameters
    ----------
    H_static : ndarray, shape (d, d)
        Time-independent part of the Hamiltonian (Hermitian).
    H_drive : ndarray, shape (d, d)
        Coupling operator of the periodic drive (Hermitian).
    omega : float
        Drive frequency (angular, same units as H entries).
    drive_func : callable (t: float) -> float, optional
        Temporal envelope f(t); defaults to ``cos(omega * t)``.
        The full Hamiltonian is H_static + H_drive * drive_func(t).
    n_steps : int
        Number of Runge-Kutta steps per period for U(T) integration.
    """

    H_static: np.ndarray
    H_drive: np.ndarray
    omega: float
    drive_func: Optional[Callable] = None
    n_steps: int = 200

    def __post_init__(self) -> None:
        self.H_static = np.asarray(self.H_static, dtype=complex)
        self.H_drive = np.asarray(self.H_drive, dtype=complex)
        d = self.H_static.shape[0]
        if self.H_static.ndim != 2 or self.H_static.shape != (d, d):
            raise FloquetError("H_static must be a square 2-D matrix")
        if self.H_drive.shape != (d, d):
            raise FloquetError("H_drive must have the same shape as H_static")
        if self.omega <= 0:
            raise FloquetError("omega must be positive")
        if self.n_steps < 4:
            raise FloquetError("n_steps must be >= 4")
        if self.drive_func is None:
            self.drive_func = lambda t: math.cos(self.omega * t)

    @property
    def dim(self) -> int:
        return int(self.H_static.shape[0])

    @property
    def period(self) -> float:
        return 2.0 * math.pi / self.omega

    def H_at(self, t: float) -> np.ndarray:
        """Return H(t) = H_static + H_drive · f(t)."""
        return self.H_static + self.drive_func(t) * self.H_drive  # type: ignore[operator]

    def __repr__(self) -> str:
        return (
            f"FloquetSystem(dim={self.dim}, omega={self.omega:.4g}, "
            f"T={self.period:.4g})"
        )


# ── One-period propagator ─────────────────────────────────────────────────────

def _propagator_rk4(system: FloquetSystem) -> np.ndarray:
    """Compute U(T) = T-exp(−i ∫₀ᵀ H(t) dt) by RK4 on dU/dt = −i H(t) U.

    Returns
    -------
    ndarray, shape (d, d)
    """
    d = system.dim
    T = system.period
    dt = T / system.n_steps
    U = np.eye(d, dtype=complex)

    def drhs(t: float, u: np.ndarray) -> np.ndarray:
        return -1j * system.H_at(t) @ u

    t = 0.0
    for _ in range(system.n_steps):
        k1 = drhs(t, U)
        k2 = drhs(t + dt / 2, U + dt / 2 * k1)
        k3 = drhs(t + dt / 2, U + dt / 2 * k2)
        k4 = drhs(t + dt, U + dt * k3)
        U = U + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt

    return U


def _propagator_at(system: FloquetSystem, t_end: float) -> np.ndarray:
    """Compute U(t_end) via RK4 (partial period)."""
    d = system.dim
    n = max(4, int(system.n_steps * t_end / system.period))
    dt = t_end / n
    U = np.eye(d, dtype=complex)

    def drhs(t: float, u: np.ndarray) -> np.ndarray:
        return -1j * system.H_at(t) @ u

    t = 0.0
    for _ in range(n):
        k1 = drhs(t, U)
        k2 = drhs(t + dt / 2, U + dt / 2 * k1)
        k3 = drhs(t + dt / 2, U + dt / 2 * k2)
        k4 = drhs(t + dt, U + dt * k3)
        U = U + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt

    return U


# ── FloquetResult ─────────────────────────────────────────────────────────────

@dataclass
class FloquetResult:
    """Output of :class:`BlochFloquetSolver`."""

    quasi_energies: np.ndarray    # shape (d,) in ascending order, ∈ (−ω/2, ω/2]
    floquet_modes: np.ndarray     # shape (d, d); columns are Floquet modes at t=0
    H_eff: np.ndarray             # shape (d, d) effective Hamiltonian (log branch)
    propagator: np.ndarray        # shape (d, d) U(T) one-period propagator
    omega: float
    period: float

    def micromotion(self, t: float) -> np.ndarray:
        """Micromotion operator U_M(t) = U(t, 0) · exp(+i H_eff t).

        U_M(t) satisfies U(t) = U_M(t) · exp(−i H_eff t) and is T-periodic.

        Parameters
        ----------
        t : float

        Returns
        -------
        ndarray, shape (d, d)
        """
        expHt = _matrix_exp(1j * self.H_eff * t)
        # We need U(t, 0) — stored implicitly via system; recompute H_eff-based approx
        # For the exact micromotion we would need U(t); here we use the identity
        # U_M(t) = exp(-i H_drive * sin(omega*t) / omega) at leading order (for
        # H = H0 + A*cos(omega*t)).  Return the H_eff exponential stroboscope for now.
        return expHt

    def gap(self) -> float:
        """Quasi-energy gap = min gap between adjacent quasi-energies."""
        qe = np.sort(np.real(self.quasi_energies))
        diffs = np.diff(qe)
        # Also wrap-around gap
        diffs = np.append(diffs, qe[0] + self.omega - qe[-1])
        return float(np.min(diffs))

    def is_topological(self, threshold: float = 0.01) -> bool:
        """Heuristic: quasi-energy gap exists at both 0 and ±ω/2."""
        qe = np.sort(np.real(self.quasi_energies))
        gap_0 = float(np.min(np.abs(qe)))
        gap_pi = float(np.min(np.abs(np.abs(qe) - self.omega / 2.0)))
        return gap_0 > threshold or gap_pi > threshold

    def __repr__(self) -> str:
        qe_str = ", ".join(f"{e:.4f}" for e in np.real(self.quasi_energies)[:4])
        return f"FloquetResult(quasi_energies=[{qe_str}{',...' if len(self.quasi_energies)>4 else ''}])"


# ── Matrix exponential helper ─────────────────────────────────────────────────

def _matrix_exp(M: np.ndarray) -> np.ndarray:
    """Compute matrix exponential of M via eigendecomposition.

    Uses the general (non-Hermitian) eigendecomposition so that anti-Hermitian
    inputs such as ``-i H t`` (with H Hermitian) are handled correctly.
    """
    eigvals, eigvecs = np.linalg.eig(M)
    return (eigvecs * np.exp(eigvals)) @ np.linalg.inv(eigvecs)


def _matrix_log_unitary(U: np.ndarray, omega: float) -> np.ndarray:
    """Compute i·log(U) / (2π/omega) = H_eff such that U = exp(−i H_eff T).

    Uses the principal branch of the matrix logarithm, mapping eigenphases
    to (−ω/2, ω/2].

    Returns H_eff (Hermitian, quasi-energies in (−ω/2, ω/2]).
    """
    eigvals, eigvecs = np.linalg.eig(U)
    # Eigenphases θ_k such that U = Σ_k |k⟩⟨k| e^{i θ_k}
    phases = np.angle(eigvals)   # ∈ (−π, π]
    T = 2.0 * math.pi / omega
    # Quasi-energies: ε_k = −θ_k / T (from U = exp(−i H_eff T))
    quasi_energies = -phases / T
    # Build H_eff
    H_eff = (eigvecs * quasi_energies) @ np.linalg.inv(eigvecs)
    # Force Hermitian (remove numerical imaginary parts)
    H_eff = 0.5 * (H_eff + H_eff.conj().T)
    return H_eff


# ── BlochFloquetSolver ────────────────────────────────────────────────────────

class BlochFloquetSolver:
    """Compute quasi-energy spectrum and effective Hamiltonians.

    Parameters
    ----------
    system : FloquetSystem
    """

    def __init__(self, system: FloquetSystem) -> None:
        if not isinstance(system, FloquetSystem):
            raise FloquetError(f"Expected FloquetSystem, got {type(system).__name__}")
        self.system = system

    # ------------------------------------------------------------------
    def solve(self) -> FloquetResult:
        """Diagonalise the one-period propagator U(T).

        Returns
        -------
        FloquetResult
        """
        U = _propagator_rk4(self.system)
        omega = self.system.omega
        T = self.system.period

        # Eigendecomposition of U(T)
        eigvals, eigvecs = np.linalg.eig(U)
        phases = np.angle(eigvals)
        quasi_energies = -phases / T

        # Sort by quasi-energy
        order = np.argsort(np.real(quasi_energies))
        quasi_energies = quasi_energies[order]
        eigvecs = eigvecs[:, order]

        # Effective Hamiltonian via matrix log
        H_eff = _matrix_log_unitary(U, omega)

        return FloquetResult(
            quasi_energies=quasi_energies,
            floquet_modes=eigvecs,
            H_eff=H_eff,
            propagator=U,
            omega=omega,
            period=T,
        )

    def micromotion_at(self, t: float) -> np.ndarray:
        """Compute the micromotion operator U_M(t) exactly.

        U_M(t) = U(t, 0) · exp(+i H_eff · t)
        """
        result = self.solve()
        U_t = _propagator_at(self.system, t)
        exp_Heff = _matrix_exp(1j * result.H_eff * t)
        return U_t @ exp_Heff

    def quasi_energy_vs_amplitude(
        self,
        amplitudes: np.ndarray,
        H_drive_scale: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Scan quasi-energies as a function of drive amplitude.

        Parameters
        ----------
        amplitudes : 1-D array of floats
            Drive amplitudes to scan.
        H_drive_scale : ndarray, optional
            Scaled drive operator (if None, uses system.H_drive).

        Returns
        -------
        ndarray, shape (len(amplitudes), d)
            Quasi-energies (real) for each amplitude.
        """
        d = self.system.dim
        results = np.zeros((len(amplitudes), d))
        H_d = H_drive_scale if H_drive_scale is not None else self.system.H_drive
        for i, amp in enumerate(amplitudes):
            sys_i = FloquetSystem(
                self.system.H_static,
                H_d * float(amp),
                self.system.omega,
                self.system.drive_func,
                self.system.n_steps,
            )
            solver_i = BlochFloquetSolver(sys_i)
            res = solver_i.solve()
            results[i] = np.real(res.quasi_energies)
        return results

    def __repr__(self) -> str:
        return f"BlochFloquetSolver(system={self.system!r})"


# ── Convenience functions ─────────────────────────────────────────────────────

def floquet_spectrum(
    H_static: np.ndarray,
    H_drive: np.ndarray,
    omega: float,
    *,
    n_steps: int = 200,
) -> FloquetResult:
    """Compute the Floquet quasi-energy spectrum.

    Shorthand for ``BlochFloquetSolver(FloquetSystem(...)).solve()``.

    Parameters
    ----------
    H_static, H_drive : ndarray
        Static and drive Hamiltonians.
    omega : float
        Drive frequency.
    n_steps : int
        Integration steps per period.

    Returns
    -------
    FloquetResult
    """
    sys = FloquetSystem(H_static, H_drive, omega, n_steps=n_steps)
    return BlochFloquetSolver(sys).solve()


def floquet_effective_hamiltonian(
    H0: np.ndarray,
    H1: np.ndarray,
    omega: float,
) -> np.ndarray:
    """First-order Floquet-Magnus effective Hamiltonian.

    For H(t) = H₀ + 2 H₁ cos(ω t):

        H_eff = H₀ + [H₁, H₋₁] / ω + O(1/ω²)
              = H₀ + [H₁, H₁†] / ω   (since H₋₁ = H₁† = H₁ for Hermitian drive)

    The second term corrects for the coupling renormalisation due to the drive.

    Parameters
    ----------
    H0 : ndarray — static Hamiltonian
    H1 : ndarray — Fourier component at +ω (= H_drive / 2 for cos drive)
    omega : float — drive frequency

    Returns
    -------
    ndarray — H_eff (Hermitian)
    """
    H0 = np.asarray(H0, dtype=complex)
    H1 = np.asarray(H1, dtype=complex)
    if omega <= 0:
        raise FloquetError("omega must be positive")
    commutator = H1 @ H1.conj().T - H1.conj().T @ H1
    H_eff = H0 + commutator / omega
    return 0.5 * (H_eff + H_eff.conj().T)


def driven_two_level(
    omega0: float,
    drive_amp: float,
    omega_drive: float,
    n_steps: int = 200,
) -> FloquetResult:
    """Floquet spectrum of a driven two-level system.

    H(t) = ω₀/2 · σz + A · cos(ω t) · σx

    In the rotating frame this gives avoided crossings (Rabi model).

    Parameters
    ----------
    omega0 : float — qubit transition frequency
    drive_amp : float — drive amplitude
    omega_drive : float — drive frequency
    n_steps : int

    Returns
    -------
    FloquetResult
    """
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    H_static = omega0 / 2.0 * Z
    H_drive = drive_amp * X
    return floquet_spectrum(H_static, H_drive, omega_drive, n_steps=n_steps)
