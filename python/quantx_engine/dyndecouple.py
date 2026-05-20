"""M126: Quantum memory with dynamical decoupling.

Dynamical decoupling (DD) suppresses decoherence by applying periodic pulse
sequences that average out system–bath interactions.  This module provides:

* :class:`DynamicalDecoupling` — library of DD sequences (FID, Hahn, CPMG,
  XY4, XY8, KDD) and their filter functions.
* :class:`QuantumMemory` — single-qubit or multi-qubit quantum memory with
  configurable noise models and DD pulse sequences; simulates fidelity decay
  and coherence time enhancement.

DD Sequences
------------
* **FID** (free induction decay): no pulses, reference baseline.
* **Hahn Echo**: single X pulse at t/2; cancels static inhomogeneity.
* **CPMG**: n equally spaced X pulses; cancels slowly varying fields.
* **XY4**: π-pulses in alternating X–Y pattern; first-order robust.
* **XY8**: eight-pulse version of XY4; second-order robust.
* **KDD**: Knill dynamical decoupling; robust to pulse imperfections.

Noise Models
------------
* Dephasing (T₂*): |0⟩⟨0| + e^{-t/T2*}|1⟩⟨1| off-diagonal decay.
* Relaxation (T₁): amplitude damping.
* Combined T₁/T₂ noise.

References
----------
* Viola, L. & Lloyd, S. (1998). Dynamical suppression of decoherence.
  *Phys. Rev. A* 58, 2733.
* Gullion, T. et al. (1990). New, compensated Carr-Purcell sequences.
  *J. Magn. Reson.* 89, 479–484.
* Souza, A. M. et al. (2011). Robust dynamical decoupling for quantum
  computing and quantum memory.  *Phys. Rev. Lett.* 106, 240501.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Sequence

import numpy as np

__all__ = [
    "DDError",
    "DDSequence",
    "DynamicalDecoupling",
    "QuantumMemory",
]

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class DDError(Exception):
    """Raised for invalid dynamical-decoupling operations."""


# ---------------------------------------------------------------------------
# DD Sequence descriptor
# ---------------------------------------------------------------------------

@dataclass
class DDSequence:
    """A dynamical decoupling pulse sequence.

    Attributes
    ----------
    name : str
        Human-readable name (e.g. "CPMG-8").
    pulses : list[tuple[float, str]]
        List of (relative_time, axis) where relative_time ∈ (0, 1) is the
        normalised time within the total sequence and axis ∈ {"X", "Y", "Z"}.
    n_pulses : int
        Total number of π-pulses.
    order : int
        Decoupling order (how many leading orders of the Magnus expansion
        are cancelled).
    """
    name: str
    pulses: list[tuple[float, str]]
    n_pulses: int
    order: int = 1

    def pulse_times(self, T: float) -> list[float]:
        """Return absolute pulse times for a total storage time T."""
        return [t_rel * T for t_rel, _ in self.pulses]

    def filter_function(
        self, omega: np.ndarray, T: float
    ) -> np.ndarray:
        """Evaluate the DD filter function |F(ω)|² for the sequence.

        The filter function determines which frequency components of the
        bath noise are suppressed.

        Parameters
        ----------
        omega : ndarray
            Angular frequencies (rad/s or dimensionless units).
        T : float
            Total storage time.

        Returns
        -------
        ndarray
            |F(ω, T)|² evaluated at each frequency.
        """
        n = len(omega)
        F = np.ones(n, dtype=complex) * T  # free precession contribution

        pulse_t = self.pulse_times(T)
        pulse_t_set = set(pulse_t)

        # Build |F(ω)|² by integrating the piecewise-constant switching function exactly.
        # F(ω) = ∫₀ᵀ f(t) e^{iωt} dt
        # On each constant segment [t0, t1]: ∫ f_sign e^{iωt} dt = f_sign (e^{iωt1}-e^{iωt0})/(iω)
        # L'Hôpital limit as ω→0: f_sign * (t1 - t0)
        segments = sorted(set([0.0] + pulse_t + [T]))
        F_val = np.zeros(n, dtype=complex)
        f_sign = 1.0
        for k in range(len(segments) - 1):
            t0, t1 = segments[k], segments[k + 1]
            phase1 = np.exp(1j * omega * t1)
            phase0 = np.exp(1j * omega * t0)
            integral = np.where(
                np.abs(omega) < 1e-12,
                f_sign * (t1 - t0),
                f_sign * (phase1 - phase0) / (1j * omega),
            )
            F_val += integral
            if t1 in pulse_t_set:
                f_sign *= -1.0

        return np.abs(F_val) ** 2


# ---------------------------------------------------------------------------
# DynamicalDecoupling
# ---------------------------------------------------------------------------

class DynamicalDecoupling:
    """Library of standard dynamical decoupling sequences and utilities.

    All sequences are defined for a single qubit stored for time T.
    """

    # ------------------------------------------------------------------
    # Sequence constructors
    # ------------------------------------------------------------------

    @staticmethod
    def fid() -> DDSequence:
        """Free induction decay (no pulses — reference baseline)."""
        return DDSequence(name="FID", pulses=[], n_pulses=0, order=0)

    @staticmethod
    def hahn_echo() -> DDSequence:
        """Hahn echo: single X pulse at t = T/2."""
        return DDSequence(
            name="Hahn",
            pulses=[(0.5, "X")],
            n_pulses=1,
            order=1,
        )

    @staticmethod
    def cpmg(n: int) -> DDSequence:
        """CPMG sequence with n equally spaced X pulses.

        Parameters
        ----------
        n : int
            Number of π-pulses (must be ≥ 1).
        """
        if n < 1:
            raise DDError("CPMG requires at least 1 pulse.")
        pulses = [((2 * k - 1) / (2 * n), "X") for k in range(1, n + 1)]
        return DDSequence(name=f"CPMG-{n}", pulses=pulses, n_pulses=n, order=1)

    @staticmethod
    def xy4() -> DDSequence:
        """XY4 sequence (4 pulses: X, Y, X, Y at t/8, 3t/8, 5t/8, 7t/8)."""
        pulses = [
            (1 / 8, "X"),
            (3 / 8, "Y"),
            (5 / 8, "X"),
            (7 / 8, "Y"),
        ]
        return DDSequence(name="XY4", pulses=pulses, n_pulses=4, order=2)

    @staticmethod
    def xy8() -> DDSequence:
        """XY8 sequence (8 pulses)."""
        base = [
            (1 / 16, "X"),
            (3 / 16, "Y"),
            (5 / 16, "X"),
            (7 / 16, "Y"),
            (9 / 16, "Y"),
            (11 / 16, "X"),
            (13 / 16, "Y"),
            (15 / 16, "X"),
        ]
        return DDSequence(name="XY8", pulses=base, n_pulses=8, order=3)

    @staticmethod
    def kdd() -> DDSequence:
        """KDD sequence (12 pulses, robust to pulse-area errors)."""
        # KDD uses three XY4 blocks with relative phases 0°, 60°, 120°
        # For simplicity, represented as 12 pulses alternating X/Y with phase shifts
        phases = ["X", "Y", "X", "Y"] * 3
        times = [(2 * k + 1) / 24 for k in range(12)]
        pulses = list(zip(times, phases))
        return DDSequence(name="KDD", pulses=pulses, n_pulses=12, order=4)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    @staticmethod
    def effective_t2(
        seq: DDSequence,
        T2_star: float,
        T: float,
    ) -> float:
        """Estimate effective T₂ extended by the DD sequence.

        Uses the filter-function formalism with a Lorentzian noise spectral
        density S(ω) = S₀ / (1 + (ω τ_c)²) with τ_c = T₂* / 2.

        Parameters
        ----------
        seq : DDSequence
            The DD sequence applied.
        T2_star : float
            Free induction decay time without DD (seconds or arbitrary units).
        T : float
            Total storage time.

        Returns
        -------
        float
            Estimated effective 1/e coherence time with DD.
        """
        if T2_star <= 0:
            raise DDError("T2_star must be positive.")
        if T <= 0:
            raise DDError("T must be positive.")

        n_pts = 1000
        omega = np.linspace(0.01 / T2_star, 100.0 / T2_star, n_pts)
        dw = omega[1] - omega[0]

        # Lorentzian spectral density
        tau_c = T2_star / 2.0
        S0 = 1.0 / (2 * math.pi * T2_star)  # normalised
        S_omega = S0 / (1.0 + (omega * tau_c) ** 2)

        # Decay exponent: χ(T) = ∫ S(ω)|F(ω,T)|² dω / ω²
        F2 = seq.filter_function(omega, T)
        integrand = S_omega * F2 / (omega ** 2 + 1e-30)
        chi = float(np.sum(integrand) * dw)

        # Effective T₂: solve chi(T_eff) = 1
        # Use linear approximation: T2_eff ≈ T2_star * (1 + order * correction)
        if chi < 1e-14:
            return float("inf")

        # T2_eff from decay: e^{-chi} = e^{-T/T2_eff} → T2_eff = T/chi
        t2_eff = T / max(chi, 1e-14)
        return float(t2_eff)

    @staticmethod
    def filter_function_summary(seq: DDSequence, T: float) -> dict:
        """Return a summary of the filter function properties.

        Returns
        -------
        dict with keys: "name", "n_pulses", "order", "peak_suppression"
        """
        omega = np.logspace(-2, 2, 500) / T
        F2 = seq.filter_function(omega, T)
        fid_F2 = np.ones(len(omega)) * T ** 2  # FID reference
        peak_sup = float(np.max(fid_F2) / max(np.max(F2), 1e-14))
        return {
            "name": seq.name,
            "n_pulses": seq.n_pulses,
            "order": seq.order,
            "peak_suppression": peak_sup,
        }


# ---------------------------------------------------------------------------
# QuantumMemory
# ---------------------------------------------------------------------------

@dataclass
class MemoryResult:
    """Result of a quantum memory fidelity simulation.

    Attributes
    ----------
    times : ndarray
        Evaluation time points.
    fidelities : ndarray
        ⟨ψ|ρ(t)|ψ⟩ fidelity with the initial state at each time.
    coherence : ndarray
        Off-diagonal element ρ₀₁(t) (complex).
    t1_effective : float
        Fitted T₁ time from exponential decay of populations.
    t2_effective : float
        Fitted T₂ time from exponential decay of coherences.
    sequence_name : str
        Name of the DD sequence applied.
    """
    times: np.ndarray
    fidelities: np.ndarray
    coherence: np.ndarray
    t1_effective: float
    t2_effective: float
    sequence_name: str

    @property
    def final_fidelity(self) -> float:
        """Fidelity at the last time point."""
        return float(self.fidelities[-1])

    @property
    def fidelity_half_life(self) -> float:
        """Time at which fidelity drops below 0.5 + 0.5/e (midpoint)."""
        target = 0.5 + 0.5 / math.e
        for i, f in enumerate(self.fidelities):
            if f < target:
                return float(self.times[i])
        return float(self.times[-1])


class QuantumMemory:
    """Single-qubit quantum memory with configurable noise and DD sequences.

    Simulates the time evolution of a qubit under:
    - Dephasing (T₂ decay of off-diagonal density matrix elements).
    - Amplitude damping (T₁ decay of excited state population).
    - DD pulse sequences that partially suppress decoherence.

    Parameters
    ----------
    T1 : float
        Longitudinal relaxation time (energy relaxation).
    T2 : float
        Transverse dephasing time (T₂ ≤ 2T₁ for physical consistency).
    pulse_error : float
        Probability of a π-pulse being imperfect (depolarising error per pulse).
    """

    def __init__(
        self,
        T1: float = 100.0,
        T2: float = 50.0,
        pulse_error: float = 0.0,
    ) -> None:
        if T1 <= 0:
            raise DDError("T1 must be positive.")
        if T2 <= 0:
            raise DDError("T2 must be positive.")
        if T2 > 2 * T1:
            raise DDError("T2 must satisfy T2 ≤ 2T1 (physical Bloch constraint).")
        if not (0.0 <= pulse_error < 1.0):
            raise DDError("pulse_error must be in [0, 1).")
        self.T1 = float(T1)
        self.T2 = float(T2)
        self.pulse_error = float(pulse_error)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        init_state: np.ndarray,
        total_time: float,
        sequence: Optional[DDSequence] = None,
        n_points: int = 100,
    ) -> MemoryResult:
        """Simulate fidelity decay under a DD sequence.

        Parameters
        ----------
        init_state : array-like, shape (2,)
            Initial qubit state (normalised).
        total_time : float
            Total storage time.
        sequence : DDSequence or None
            DD sequence to apply.  If None, plain free evolution (FID).
        n_points : int
            Number of time points to evaluate.

        Returns
        -------
        MemoryResult
        """
        sv = np.asarray(init_state, dtype=complex).ravel()
        if sv.shape != (2,):
            raise DDError("init_state must have shape (2,).")
        sv = sv / np.linalg.norm(sv)
        if total_time <= 0:
            raise DDError("total_time must be positive.")
        if n_points < 2:
            raise DDError("n_points must be at least 2.")

        seq = sequence or DynamicalDecoupling.fid()
        times = np.linspace(0, total_time, n_points)
        rho0 = np.outer(sv, sv.conj())  # initial density matrix

        fidelities = np.zeros(n_points)
        coherences = np.zeros(n_points, dtype=complex)
        fidelities[0] = 1.0
        coherences[0] = complex(rho0[0, 1])

        for k, t in enumerate(times[1:], start=1):
            rho_t = self._evolve_rho(rho0, t, seq)
            fidelities[k] = float(np.real(sv.conj() @ (rho_t @ sv)))
            coherences[k] = complex(rho_t[0, 1])

        # Fit effective T₁ and T₂
        t1_eff = self._fit_t1(times, fidelities, rho0)
        t2_eff = self._fit_t2(times, coherences)

        return MemoryResult(
            times=times,
            fidelities=fidelities,
            coherence=coherences,
            t1_effective=t1_eff,
            t2_effective=t2_eff,
            sequence_name=seq.name,
        )

    def fidelity_at(
        self,
        init_state: np.ndarray,
        t: float,
        sequence: Optional[DDSequence] = None,
    ) -> float:
        """Compute the memory fidelity at a single time t.

        Parameters
        ----------
        init_state : array-like, shape (2,)
            Initial qubit state.
        t : float
            Storage time.
        sequence : DDSequence or None
            DD sequence.
        """
        if t < 0:
            raise DDError("t must be non-negative.")
        sv = np.asarray(init_state, dtype=complex).ravel()
        sv = sv / np.linalg.norm(sv)
        rho0 = np.outer(sv, sv.conj())
        seq = sequence or DynamicalDecoupling.fid()
        rho_t = self._evolve_rho(rho0, t, seq)
        return float(np.real(sv.conj() @ (rho_t @ sv)))

    def coherence_time(
        self,
        sequence: Optional[DDSequence] = None,
        target_fidelity: float = 1.0 / math.e,
    ) -> float:
        """Estimate the coherence time for a given sequence and fidelity threshold.

        Parameters
        ----------
        sequence : DDSequence or None
            DD sequence.
        target_fidelity : float
            Fidelity level at which coherence time is defined (default 1/e).

        Returns
        -------
        float
            Time at which the |+⟩ state fidelity drops below target_fidelity.
        """
        sv_plus = np.array([1, 1], dtype=complex) / math.sqrt(2)
        T_est = self.T2 * 10  # generous upper bound
        n = 500
        ts = np.linspace(0, T_est, n)
        rho0 = np.outer(sv_plus, sv_plus.conj())
        seq = sequence or DynamicalDecoupling.fid()
        for t in ts[1:]:
            rho_t = self._evolve_rho(rho0, t, seq)
            fid = float(np.real(sv_plus.conj() @ (rho_t @ sv_plus)))
            if fid <= target_fidelity:
                return float(t)
        return float(ts[-1])

    # ------------------------------------------------------------------
    # Internal: density matrix evolution
    # ------------------------------------------------------------------

    def _evolve_rho(
        self, rho0: np.ndarray, t: float, seq: DDSequence
    ) -> np.ndarray:
        """Evolve density matrix rho0 for time t under the given DD sequence."""
        # Base decoherence without DD
        rho = rho0.copy()

        # Effective T₂ enhanced by DD (heuristic: factor proportional to n_pulses)
        dd_factor = 1.0 + seq.n_pulses * seq.order * 0.3
        T2_eff = min(self.T2 * dd_factor, 2 * self.T1)

        # Account for pulse errors: additional depolarising
        if seq.n_pulses > 0 and self.pulse_error > 0:
            p_err = 1 - (1 - self.pulse_error) ** seq.n_pulses
            T2_eff = T2_eff / (1 + p_err * T2_eff / self.T2)

        # Apply T₁ amplitude damping: ρ₁₁ → ρ₁₁ e^{-t/T₁}
        # Off-diagonals already acquire a factor sqrt(e^{-t/T1}) = e^{-t/(2T1}).
        gamma1 = 1.0 / self.T1
        decay1 = math.exp(-gamma1 * t)
        rho_t = np.array([
            [rho[0, 0] + (1 - decay1) * rho[1, 1], rho[0, 1] * math.sqrt(decay1)],
            [rho[1, 0] * math.sqrt(decay1),          rho[1, 1] * decay1],
        ], dtype=complex)

        # Apply PURE dephasing only (T₁ contribution already handled above).
        # The total T₂ decomposes as 1/T₂ = 1/(2T₁) + 1/T₂_φ (pure dephasing).
        # Applying the full T₂ here would double-count the 1/(2T₁) term.
        gamma_phi = max(0.0, 1.0 / T2_eff - 1.0 / (2.0 * self.T1))
        decay_phi = math.exp(-gamma_phi * t)
        rho_t[0, 1] *= decay_phi
        rho_t[1, 0] *= decay_phi

        return rho_t

    # ------------------------------------------------------------------
    # Fitting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_t1(
        times: np.ndarray, fidelities: np.ndarray, rho0: np.ndarray
    ) -> float:
        """Estimate T₁ from the fidelity decay (exponential fit)."""
        rho11 = float(np.real(rho0[1, 1]))
        if rho11 < 1e-10:
            return float("inf")  # |0⟩ state: T₁ not observable
        # Find time at which fidelity ≈ rho11 * exp(-1) + rho00
        rho00 = float(np.real(rho0[0, 0]))
        target = rho00 + rho11 * (1 / math.e)
        for i, f in enumerate(fidelities):
            if f <= target:
                return float(times[i])
        return float(times[-1])

    @staticmethod
    def _fit_t2(
        times: np.ndarray, coherences: np.ndarray
    ) -> float:
        """Estimate T₂ from the coherence decay (exponential fit)."""
        c0 = abs(coherences[0])
        if c0 < 1e-10:
            return float("inf")
        target = c0 / math.e
        for i, c in enumerate(coherences):
            if abs(c) <= target:
                return float(times[i])
        return float(times[-1])

    def __repr__(self) -> str:
        return (
            f"QuantumMemory(T1={self.T1:.3g}, T2={self.T2:.3g}, "
            f"pulse_error={self.pulse_error:.2e})"
        )
