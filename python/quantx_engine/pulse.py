"""Pulse-level quantum simulator — M89.

Simulates time evolution under a time-dependent Hamiltonian

    H(t) = H₀ + Σ_j [Re(Ω_j(t))/2 · σx_j + Im(Ω_j(t))/2 · σy_j]

in the rotating frame.  H₀ = Σ_j Δ_j/2 · σz_j where Δ_j = ω_j − ω_drive_j
is the qubit–LO detuning.

All frequencies / amplitudes share the same angular-frequency unit (e.g. rad/ns).
T1 / T2 times are in the matching time unit (e.g. ns).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


# ── Errors ──────────────────────────────────────────────────────────────────

class PulseError(ValueError):
    """Raised on invalid pulse configuration or missing dependency."""


# ── Channels ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DriveChannel:
    """Microwave drive channel targeting a single qubit."""
    qubit: int

    def __repr__(self) -> str:
        return f"DriveChannel({self.qubit})"


@dataclass(frozen=True)
class MeasureChannel:
    """Measurement pulse channel (tracked but no Hamiltonian effect in simulation)."""
    qubit: int

    def __repr__(self) -> str:
        return f"MeasureChannel({self.qubit})"


@dataclass(frozen=True)
class ControlChannel:
    """Cross-resonance / ZZ-control channel."""
    index: int

    def __repr__(self) -> str:
        return f"ControlChannel({self.index})"


Channel = Union[DriveChannel, MeasureChannel, ControlChannel]


# ── Waveforms ────────────────────────────────────────────────────────────────

class Waveform:
    """Abstract base for pulse envelopes.

    Subclasses must expose a ``duration`` attribute (float) and implement
    :meth:`envelope`.  The base class intentionally does *not* declare
    ``duration`` as a property so that ``@dataclass`` subclasses can use it
    as a plain field without Python 3.14's property-as-default heuristic
    interfering with argument ordering.
    """

    def envelope(self, t: np.ndarray) -> np.ndarray:
        """Return (complex) amplitude at times *t*.

        Values outside [0, duration] are zero.
        """
        raise NotImplementedError


@dataclass
class ConstantWaveform(Waveform):
    """Square (constant-amplitude) pulse."""
    duration: float
    amp: float

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise PulseError("ConstantWaveform duration must be positive")

    def envelope(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.where((t >= 0) & (t <= self.duration), self.amp, 0.0)


@dataclass
class GaussianWaveform(Waveform):
    """Gaussian pulse centred at duration/2."""
    duration: float
    amp: float
    sigma: float

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise PulseError("GaussianWaveform duration must be positive")
        if self.sigma <= 0:
            raise PulseError("GaussianWaveform sigma must be positive")

    def envelope(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        mu = self.duration / 2.0
        env = self.amp * np.exp(-0.5 * ((t - mu) / self.sigma) ** 2)
        return np.where((t >= 0) & (t <= self.duration), env, 0.0)


@dataclass
class DRAGWaveform(Waveform):
    """DRAG (Derivative Removal via Adiabatic Gate) waveform.

    In-phase component  : Gaussian envelope I(t).
    Quadrature component: Q(t) = −beta · dI/dt.

    Returns a **complex** envelope I(t) + i·Q(t).
    With beta tuned to the qubit anharmonicity this suppresses leakage to |2⟩.
    """
    duration: float
    amp: float
    sigma: float
    beta: float   # DRAG coefficient (rad·time²)

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise PulseError("DRAGWaveform duration must be positive")
        if self.sigma <= 0:
            raise PulseError("DRAGWaveform sigma must be positive")

    def envelope(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        mu = self.duration / 2.0
        g = self.amp * np.exp(-0.5 * ((t - mu) / self.sigma) ** 2)
        # dI/dt = -g · (t - mu) / sigma²
        dgdt = -g * (t - mu) / self.sigma ** 2
        result = (g - 1j * self.beta * dgdt).astype(complex)
        mask = (t >= 0) & (t <= self.duration)
        return np.where(mask, result, 0.0 + 0.0j)


@dataclass
class FlatTopWaveform(Waveform):
    """Gaussian rise / flat top / Gaussian fall envelope."""
    duration: float
    amp: float
    rise_time: float   # σ of Gaussian rise / fall

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise PulseError("FlatTopWaveform duration must be positive")
        if self.rise_time <= 0:
            raise PulseError("FlatTopWaveform rise_time must be positive")

    def envelope(self, t: np.ndarray) -> np.ndarray:
        from math import erf  # noqa: PLC0415
        t = np.asarray(t, dtype=float)
        sqrt2 = math.sqrt(2.0)
        rise = 0.5 * self.amp * (1.0 + np.vectorize(erf)((t) / (self.rise_time * sqrt2)))
        fall = 0.5 * self.amp * (1.0 + np.vectorize(erf)((self.duration - t) / (self.rise_time * sqrt2)))
        env = np.minimum(rise, fall)
        return np.where((t >= 0) & (t <= self.duration), env, 0.0)


@dataclass
class ArbitraryWaveform(Waveform):
    """Waveform defined by an explicit (complex) sample array."""
    samples: np.ndarray  # shape (N,); complex supported
    dt: float = 1.0      # time step between samples

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=complex)
        if self.samples.ndim != 1 or len(self.samples) == 0:
            raise PulseError("ArbitraryWaveform samples must be a non-empty 1-D array")
        if self.dt <= 0:
            raise PulseError("ArbitraryWaveform dt must be positive")

    @property
    def duration(self) -> float:
        return len(self.samples) * self.dt

    def envelope(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        idx = np.clip((t / self.dt).astype(int), 0, len(self.samples) - 1)
        result = self.samples[idx]
        mask = (t >= 0) & (t <= self.duration)
        return np.where(mask, result, 0.0 + 0.0j)


# ── PulseInstruction / PulseSchedule ────────────────────────────────────────

@dataclass
class PulseInstruction:
    channel: Channel
    waveform: Waveform
    t_start: float


class PulseSchedule:
    """Ordered sequence of :class:`PulseInstruction` across multiple channels."""

    def __init__(self, name: str = "schedule") -> None:
        self.name = name
        self._instructions: list[PulseInstruction] = []

    # ------------------------------------------------------------------
    def add(
        self,
        channel: Channel,
        waveform: Waveform,
        t_start: float = 0.0,
    ) -> "PulseSchedule":
        """Append an instruction and return *self* for fluent chaining."""
        if t_start < 0:
            raise PulseError(f"t_start must be non-negative, got {t_start}")
        self._instructions.append(PulseInstruction(channel, waveform, t_start))
        return self

    # ------------------------------------------------------------------
    @property
    def instructions(self) -> list[PulseInstruction]:
        return list(self._instructions)

    @property
    def duration(self) -> float:
        if not self._instructions:
            return 0.0
        return max(inst.t_start + inst.waveform.duration for inst in self._instructions)

    def __repr__(self) -> str:
        return (
            f"PulseSchedule(name={self.name!r}, "
            f"n_instructions={len(self._instructions)}, "
            f"duration={self.duration:.4g})"
        )


# ── QubitSystem ──────────────────────────────────────────────────────────────

@dataclass
class QubitSystem:
    """Describes the physical multi-qubit Hamiltonian and noise parameters.

    Parameters
    ----------
    n_qubits : int
    frequencies : list[float]
        Qubit transition frequencies ω_j (angular, same unit as waveform amplitudes).
    drive_frequencies : list[float] | None
        LO frequencies ω_drive_j.  Defaults to *frequencies* (on-resonance).
    t1_times : list[float] | None
        Amplitude-damping times T1_j.  ``None`` or 0 means no T1 noise on that qubit.
    t2_times : list[float] | None
        Total dephasing times T2_j (T2 ≤ 2·T1).
    coupling_map : list[tuple[int, int, float]] | None
        Exchange couplings [(q0, q1, J), ...] added as J(σ+_q0 σ-_q1 + h.c.).
    """

    n_qubits: int
    frequencies: list
    drive_frequencies: Optional[list] = None
    t1_times: Optional[list] = None
    t2_times: Optional[list] = None
    coupling_map: Optional[list] = None

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise PulseError("n_qubits must be >= 1")
        if len(self.frequencies) != self.n_qubits:
            raise PulseError("len(frequencies) must equal n_qubits")
        if self.drive_frequencies is None:
            self.drive_frequencies = list(self.frequencies)
        if len(self.drive_frequencies) != self.n_qubits:
            raise PulseError("len(drive_frequencies) must equal n_qubits")

    @property
    def detunings(self) -> list:
        """Δ_j = ω_j − ω_drive_j."""
        return [self.frequencies[j] - self.drive_frequencies[j] for j in range(self.n_qubits)]


# ── Single-qubit matrices ────────────────────────────────────────────────────

_I2 = np.eye(2, dtype=complex)
_X2 = np.array([[0, 1], [1, 0]], dtype=complex)
_Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
_Sm = np.array([[0, 1], [0, 0]], dtype=complex)  # |0⟩⟨1|  (lowers |1⟩→|0⟩)
_Sp = np.array([[0, 0], [1, 0]], dtype=complex)  # |1⟩⟨0|


def _kron_op(op2: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Embed a 2×2 operator on *qubit* into the n-qubit Hilbert space."""
    result = np.array([[1.0 + 0j]], dtype=complex)
    # tensor product from qubit n-1 (MSB/leftmost) down to qubit 0 (LSB)
    for j in range(n - 1, -1, -1):
        result = np.kron(result, op2 if j == qubit else _I2)
    return result


def _build_static_hamiltonian(system: QubitSystem) -> np.ndarray:
    """Rotating-frame static Hamiltonian H₀ = Σ_j Δ_j/2 σz_j + coupling."""
    n = system.n_qubits
    d = 2 ** n
    H0 = np.zeros((d, d), dtype=complex)

    for j, delta in enumerate(system.detunings):
        H0 += (delta / 2.0) * _kron_op(_Z2, j, n)

    if system.coupling_map:
        for q0, q1, J in system.coupling_map:
            Sp0 = _kron_op(_Sp, q0, n)
            Sm1 = _kron_op(_Sm, q1, n)
            H0 += J * (Sp0 @ Sm1 + (Sp0 @ Sm1).conj().T)

    return H0


def _build_drive_ops(system: QubitSystem) -> dict:
    """σx_j and σy_j embedded in n-qubit space, keyed by qubit index."""
    n = system.n_qubits
    return {
        j: (_kron_op(_X2, j, n), _kron_op(_Y2, j, n))
        for j in range(n)
    }


def _build_jump_ops(system: QubitSystem) -> list:
    """Lindblad jump operators for amplitude damping (T1) and pure dephasing (T2)."""
    n = system.n_qubits
    ops: list[np.ndarray] = []

    t1s = system.t1_times or []
    t2s = system.t2_times or []

    for j in range(n):
        T1 = t1s[j] if j < len(t1s) else None
        T2 = t2s[j] if j < len(t2s) else None

        # Amplitude damping
        if T1 and T1 > 0:
            ops.append(math.sqrt(1.0 / T1) * _kron_op(_Sm, j, n))

        # Pure dephasing
        if T2 and T2 > 0:
            if T1 and T1 > 0:
                gamma_phi = 1.0 / T2 - 1.0 / (2.0 * T1)
            else:
                gamma_phi = 1.0 / T2
            if gamma_phi > 1e-20:
                # L†L contribution to d(ρ₀₁)/dt = −2·(gamma_phi/2)·ρ₀₁ = −gamma_phi·ρ₀₁
                ops.append(math.sqrt(gamma_phi / 2.0) * _kron_op(_Z2, j, n))

    return ops


# ── PulseResult ──────────────────────────────────────────────────────────────

@dataclass
class PulseResult:
    """Output of :class:`PulseSimulator`."""

    final_state: np.ndarray             # shape (2^n,) — dominant eigenvector for noisy runs
    populations: dict                    # bitstring → probability
    times: np.ndarray                    # time points used by the ODE solver
    states_at_times: Optional[np.ndarray] = None   # (n_times, 2^n) if record_states=True
    density_matrix: Optional[np.ndarray] = None    # set for Lindblad (noisy) runs
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def fidelity(self, target: np.ndarray) -> float:
        """State fidelity |⟨target|final_state⟩|²."""
        t = np.asarray(target, dtype=complex)
        norm = np.linalg.norm(t)
        if norm < 1e-12:
            raise PulseError("target state has zero norm")
        t = t / norm
        return float(abs(np.vdot(t, self.final_state)) ** 2)

    def expectation(self, op: np.ndarray) -> float:
        """Expectation value of *op*.

        Uses Tr(ρ·op) when a density matrix is available (noisy/Lindblad run),
        otherwise computes ⟨ψ|op|ψ⟩.
        """
        if self.density_matrix is not None:
            return float(np.real(np.trace(self.density_matrix @ op)))
        return float(np.real(np.vdot(self.final_state, op @ self.final_state)))

    def bloch_vector(self, qubit: int) -> tuple:
        """(⟨X⟩, ⟨Y⟩, ⟨Z⟩) for the given qubit."""
        n = int(round(math.log2(len(self.final_state))))
        bx = self.expectation(_kron_op(_X2, qubit, n))
        by = self.expectation(_kron_op(_Y2, qubit, n))
        bz = self.expectation(_kron_op(_Z2, qubit, n))
        return (bx, by, bz)

    def __repr__(self) -> str:
        top = sorted(self.populations.items(), key=lambda kv: -kv[1])[:3]
        pops = {k: round(v, 4) for k, v in top}
        return f"PulseResult(populations={pops}, n_qubits={self.metadata.get('n_qubits')})"


# ── PulseSimulator ────────────────────────────────────────────────────────────

class PulseSimulator:
    """Pulse-level simulator based on :func:`scipy.integrate.solve_ivp`.

    *Pure state* (no T1/T2 in :class:`QubitSystem`): solves the time-dependent
    Schrödinger equation.

    *Open system* (T1 and/or T2 specified): solves the Lindblad master equation
    on the density matrix.
    """

    def __init__(
        self,
        method: str = "RK45",
        atol: float = 1e-8,
        rtol: float = 1e-6,
        record_states: bool = False,
        n_record_points: int = 50,
    ) -> None:
        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.record_states = record_states
        self.n_record_points = n_record_points

    # ------------------------------------------------------------------
    def run(
        self,
        schedule: PulseSchedule,
        system: QubitSystem,
        initial_state: Optional[np.ndarray] = None,
    ) -> PulseResult:
        """Simulate *schedule* on *system* and return a :class:`PulseResult`."""
        try:
            from scipy.integrate import solve_ivp  # noqa: PLC0415
        except ImportError as exc:
            raise PulseError("scipy is required for PulseSimulator") from exc

        if not isinstance(schedule, PulseSchedule):
            raise PulseError(f"Expected PulseSchedule, got {type(schedule).__name__}")
        if not isinstance(system, QubitSystem):
            raise PulseError(f"Expected QubitSystem, got {type(system).__name__}")
        if schedule.duration <= 0:
            raise PulseError("Schedule must have positive duration")

        n = system.n_qubits
        d = 2 ** n
        T = schedule.duration

        # ── initial state ──────────────────────────────────────────────
        if initial_state is None:
            psi0 = np.zeros(d, dtype=complex)
            psi0[0] = 1.0
        else:
            psi0 = np.asarray(initial_state, dtype=complex).copy()
            if psi0.shape != (d,):
                raise PulseError(f"initial_state must have shape ({d},)")
            nm = np.linalg.norm(psi0)
            if nm < 1e-12:
                raise PulseError("initial_state has zero norm")
            psi0 /= nm

        # ── operators ─────────────────────────────────────────────────
        H0 = _build_static_hamiltonian(system)
        drive_ops = _build_drive_ops(system)
        jump_ops = _build_jump_ops(system)

        # ── per-qubit drive schedule ───────────────────────────────────
        qubit_drives: dict = {j: [] for j in range(n)}
        for inst in schedule.instructions:
            if isinstance(inst.channel, DriveChannel):
                q = inst.channel.qubit
                if 0 <= q < n:
                    qubit_drives[q].append((inst.waveform, inst.t_start))

        def _omega(qubit: int, t: float) -> complex:
            total: complex = 0.0
            for wf, t0 in qubit_drives[qubit]:
                lt = t - t0
                if 0.0 <= lt <= wf.duration:
                    arr = wf.envelope(np.array([lt]))
                    total += complex(arr[0])
            return total

        def _H_at(t: float) -> np.ndarray:
            H = H0.copy()
            for j in range(n):
                om = _omega(j, t)
                Xj, Yj = drive_ops[j]
                H += (np.real(om) / 2.0) * Xj
                im_om = np.imag(om)
                if abs(im_om) > 1e-20:
                    H += (im_om / 2.0) * Yj
            return H

        t_eval = (
            np.linspace(0.0, T, self.n_record_points) if self.record_states else None
        )

        if not jump_ops:
            # ── Pure Schrödinger ────────────────────────────────────────
            y0 = np.concatenate([np.real(psi0), np.imag(psi0)])

            def schrodinger(t: float, y: np.ndarray) -> np.ndarray:
                psi = y[:d] + 1j * y[d:]
                dpsi = -1j * (_H_at(t) @ psi)
                return np.concatenate([np.real(dpsi), np.imag(dpsi)])

            sol = solve_ivp(
                schrodinger, [0.0, T], y0,
                method=self.method, atol=self.atol, rtol=self.rtol,
                t_eval=t_eval, dense_output=False,
            )
            psi_f = sol.y[:d, -1] + 1j * sol.y[d:, -1]
            psi_f /= np.linalg.norm(psi_f)

            pops = {
                format(i, f"0{n}b"): float(v)
                for i, v in enumerate(np.abs(psi_f) ** 2)
                if v > 1e-12
            }
            sts = None
            if self.record_states and t_eval is not None:
                sts = (sol.y[:d] + 1j * sol.y[d:]).T  # (n_times, d)

            return PulseResult(
                final_state=psi_f,
                populations=pops,
                times=sol.t,
                states_at_times=sts,
                metadata={"n_qubits": n, "duration": T, "noisy": False,
                          "message": sol.message},
            )

        else:
            # ── Lindblad master equation ────────────────────────────────
            # Vectorise ρ (d²,) split real/imag → y of length 2d²
            rho0 = np.outer(psi0, psi0.conj())
            y0 = np.concatenate([np.real(rho0.ravel()), np.imag(rho0.ravel())])
            d2 = d * d

            # Pre-compute L†L for each jump operator
            LdagL = [L.conj().T @ L for L in jump_ops]

            def lindblad(t: float, y: np.ndarray) -> np.ndarray:
                rho = (y[:d2] + 1j * y[d2:]).reshape(d, d)
                H = _H_at(t)
                # Coherent part
                drho = -1j * (H @ rho - rho @ H)
                # Dissipative part
                for L, LL in zip(jump_ops, LdagL):
                    Ld = L.conj().T
                    drho += L @ rho @ Ld - 0.5 * (LL @ rho + rho @ LL)
                flat = drho.ravel()
                return np.concatenate([np.real(flat), np.imag(flat)])

            sol = solve_ivp(
                lindblad, [0.0, T], y0,
                method=self.method, atol=self.atol, rtol=self.rtol,
                t_eval=t_eval, dense_output=False,
            )
            rho_f = (sol.y[:d2, -1] + 1j * sol.y[d2:, -1]).reshape(d, d)
            # Ensure Hermitian and trace-1
            rho_f = 0.5 * (rho_f + rho_f.conj().T)
            tr = np.real(np.trace(rho_f))
            if tr > 1e-12:
                rho_f /= tr

            diag = np.maximum(np.real(np.diag(rho_f)), 0.0)
            diag /= diag.sum() if diag.sum() > 0 else 1.0
            pops = {
                format(i, f"0{n}b"): float(v)
                for i, v in enumerate(diag)
                if v > 1e-12
            }

            # dominant eigenvector as "final state"
            eigvals, eigvecs = np.linalg.eigh(rho_f)
            psi_f = eigvecs[:, -1]

            return PulseResult(
                final_state=psi_f,
                populations=pops,
                times=sol.t,
                density_matrix=rho_f,
                metadata={"n_qubits": n, "duration": T, "noisy": True,
                          "message": sol.message},
            )


# ── Convenience factory ───────────────────────────────────────────────────────

def pi_pulse_area(sigma: float) -> float:
    """Return the Gaussian amplitude that gives a π-rotation for the given σ.

    For H(t) = Ω(t)/2 · σx, the π condition is ∫ Ω dt = π.
    Gaussian area = amp · σ · √(2π), so amp = π / (σ · √(2π)).
    """
    return math.pi / (sigma * math.sqrt(2.0 * math.pi))
