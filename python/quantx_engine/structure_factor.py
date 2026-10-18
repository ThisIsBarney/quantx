"""Dynamical spin structure factor S(k, omega) via MPS time evolution.

Wraps ``qx_spectral`` (``src/dmrg_blocks.cpp`` via ``src/dmrg_capi.cpp``): computes
the time-dependent ground-state correlations::

    C_{j,l0}(t) = <psi0| sigma^z_j  e^{-i(H-E0) t}  sigma^z_{l0} |psi0>

for a central source site ``l0`` of a spin-1/2 Heisenberg chain -- DMRG for the
ground state, block-sparse (Sz-symmetric) real-time TEBD for the evolution, and
a block two-MPS overlap for the measurement. A space+time Fourier transform turns
C into the dynamical structure factor

    S(k, omega) = sum_{r,t} e^{-i k r} e^{i omega t} C_r(t).

This is distinct from ``quantx_engine.contrib.spectral`` (Lehmann/Krylov spectral
functions on dense/sparse Hamiltonians); this module is the MPS route that
scales to large chains.

Example
-------
    from quantx_engine import Engine
    from quantx_engine.structure_factor import StructureFactorSolver

    s = StructureFactorSolver(Engine())
    r = s.correlations(24, dt=0.05, n_steps=400, chi_max=128)
    k, w, S = s.structure_factor(r)        # 2-D array S[k, omega] >= 0
"""
from __future__ import annotations

import json
from ctypes import c_char_p
from dataclasses import dataclass

import numpy as np

__all__ = ["StructureFactorSolver", "DSFResult", "StructureFactorError",
           "ConductivitySolver", "ConductivityResult"]


class StructureFactorError(RuntimeError):
    """Raised when the solver fails or the native library is unavailable."""


@dataclass
class DSFResult:
    """Raw dynamical correlations from the native engine.

    Attributes
    ----------
    times : np.ndarray
        Time grid, shape (n_t,).
    C : np.ndarray
        Complex correlations ``C[t, j] = <psi0| sz_j e^{-i(H-E0)t} sz_l0 |psi0>``,
        shape (n_t, n).
    l0 : int
        Source site.
    E0 : float
        Ground-state energy.
    n : int
        Number of sites.
    max_bond : int
        Largest bond dimension during evolution.
    wall_time_s : float
        Native wall-clock time (ground state + evolution + measurement).
    """
    times: np.ndarray
    C: np.ndarray
    l0: int
    E0: float
    n: int
    max_bond: int
    wall_time_s: float


class StructureFactorSolver:
    """Dynamical structure factor S(k, omega) solver (native C++ backend)."""

    def __init__(self, engine) -> None:
        self._engine = engine
        self._lib = None
        lib = getattr(engine, "_lib", None)
        if lib is not None:
            try:
                fn = lib.qx_spectral
                fn.argtypes = [c_char_p, c_char_p]
                fn.restype = c_char_p
                self._lib = lib
            except AttributeError:
                self._lib = None

    def correlations(
        self,
        n_qubits: int,
        *,
        J: float = 1.0,
        delta: float = 1.0,
        dt: float = 0.05,
        n_steps: int = 200,
        chi_max: int = 128,
        cutoff: float = 1e-10,
        n_sweeps: int = 40,
    ) -> DSFResult:
        """Compute the time-dependent correlations C_{j,l0}(t).

        Parameters
        ----------
        n_qubits : int
            Number of spin-1/2 sites.
        J : float
            Heisenberg coupling (H = J sum sigma_i . sigma_{i+1}).
        dt : float
            Real-time step.
        n_steps : int
            Number of time steps (max time = dt * n_steps).
        chi_max : int
            Maximum bond dimension during the real-time evolution.
        cutoff : float
            SVD truncation cutoff.
        n_sweeps : int
            DMRG sweeps for the ground state.

        Returns
        -------
        DSFResult
        """
        if self._lib is None:
            raise StructureFactorError("Structure factor requires the native C++ library.")
        if n_qubits < 2:
            raise StructureFactorError("n_qubits must be >= 2")
        prob = {"n_qubits": n_qubits, "type": "heisenberg", "J": J, "delta": delta}
        opts = {"dt": dt, "n_steps": n_steps, "chi_max": chi_max,
                "cutoff": cutoff, "n_sweeps": n_sweeps}
        raw = self._lib.qx_spectral(json.dumps(prob).encode(), json.dumps(opts).encode())
        out = json.loads(raw)
        if not out.get("ok", False):
            raise StructureFactorError(out.get("error", "Unknown error from qx_spectral"))
        C = np.array(out["C_re"], dtype=float) + 1j * np.array(out["C_im"], dtype=float)
        return DSFResult(
            times=np.array(out["times"], dtype=float),
            C=C,
            l0=int(out["l0"]),
            E0=float(out["E0"]),
            n=int(out["n"]),
            max_bond=int(out.get("max_bond", 0)),
            wall_time_s=float(out.get("wall_time_s", 0.0)),
        )

    @staticmethod
    def structure_factor(
        res: DSFResult,
        *,
        n_omega: int = 512,
        omega_max: float | None = None,
        eta: float = 0.1,
    ):
        """Fourier transform C_{j,l0}(t) into S(k, omega).

        Uses a Gaussian time window exp(-(eta t)^2 / 2) to damp finite-time
        truncation ringing, the symmetry C_r(-t) = conj(C_{-r}(t)) to extend to
        negative time, and a discrete spatial transform on the open chain
        (referenced to the source site l0).

        Parameters
        ----------
        res : DSFResult
            Output of ``correlations``.
        n_omega : int
            Number of omega grid points in [0, omega_max].
        omega_max : float, optional
            Max frequency (default 2*pi).
        eta : float
            Gaussian frequency broadening (time-domain damping rate).

        Returns
        -------
        ks : np.ndarray, shape (n,)
            Momenta in [0, 2*pi).
        omegas : np.ndarray, shape (n_omega,)
        S : np.ndarray, shape (n, n_omega), real
            Dynamical structure factor.
        """
        t = res.times
        dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
        n = res.n
        l0 = res.l0
        C = res.C                                   # (n_t, n)
        win = np.exp(-0.5 * (eta * t) ** 2)         # time window

        if omega_max is None:
            omega_max = 2.0 * np.pi
        omegas = np.linspace(0.0, omega_max, n_omega)
        ks = 2.0 * np.pi * np.arange(n) / n

        # spatial transform: Ck(t) = sum_j e^{-i k (j - l0)} C[t, j]
        r = np.arange(n) - l0
        phase_space = np.exp(-1j * np.outer(ks, r))   # (n_k, n)
        Ck = C @ phase_space.T                        # (n_t, n_k)

        # temporal transform with negative-time extension Ck(-t) = conj(Ck(t)):
        #   S(k,w) = 2 Re sum_{t>=0} window(t) e^{i w t} Ck(t) dt  (t=0 once)
        ewt = np.exp(1j * np.outer(omegas, t))        # (n_w, n_t)
        wC = (win[None, :] * Ck.T)                    # (n_k, n_t)
        Skw = 2.0 * np.real(ewt @ wC.T) * dt          # (n_w, n_k)
        Skw -= np.real((win[0] * Ck[0])[None, :]) * dt  # remove double-counted t=0
        S = Skw.T                                      # (n_k, n_w)
        return ks, omegas, S


# -- Spin conductivity ---------------------------------------------------------

@dataclass
class ConductivityResult:
    """Current-current correlator C(t) = <psi0|J e^{-i(H-E0)t} J|psi0>."""
    times: object
    C: object       # complex np.ndarray
    E0: float
    n: int
    delta: float
    max_bond: int
    wall_time_s: float


class ConductivitySolver:
    """Spin conductivity of the XXZ chain (native C++ backend).

    Computes the zero-temperature current-current correlator C(t) = <J(t)J(0)>
    (J = total spin current) via DMRG ground state + block MPO x MPS apply
    (phi = J|psi0>) + block TEBD evolution + two-MPS overlap. sigma(omega) is the
    Fourier transform of C(t).
    """

    def __init__(self, engine) -> None:
        self._engine = engine
        self._lib = None
        lib = getattr(engine, "_lib", None)
        if lib is not None:
            try:
                fn = lib.qx_conductivity
                fn.argtypes = [c_char_p, c_char_p]
                fn.restype = c_char_p
                self._lib = lib
            except AttributeError:
                self._lib = None

    def correlator(self, n_qubits, *, J=1.0, delta=1.0, dt=0.0125, n_steps=400,
                   chi_max=128, cutoff=1e-10, n_sweeps=40):
        """Current-current correlator C(t) for the XXZ chain."""
        if self._lib is None:
            raise StructureFactorError("Conductivity requires the native C++ library.")
        if n_qubits < 2:
            raise StructureFactorError("n_qubits must be >= 2")
        prob = {"n_qubits": n_qubits, "type": "heisenberg", "J": J, "delta": delta}
        opts = {"dt": dt, "n_steps": n_steps, "chi_max": chi_max,
                "cutoff": cutoff, "n_sweeps": n_sweeps}
        raw = self._lib.qx_conductivity(json.dumps(prob).encode(), json.dumps(opts).encode())
        out = json.loads(raw)
        if not out.get("ok", False):
            raise StructureFactorError(out.get("error", "Unknown error from qx_conductivity"))
        C = np.array(out["C_re"], dtype=float) + 1j * np.array(out["C_im"], dtype=float)
        return ConductivityResult(
            times=np.array(out["times"], dtype=float), C=C,
            E0=float(out["E0"]), n=int(out["n"]), delta=float(out.get("delta", delta)),
            max_bond=int(out.get("max_bond", 0)),
            wall_time_s=float(out.get("wall_time_s", 0.0)))

    @staticmethod
    def sigma(res, *, n_omega=512, omega_max=None, eta=0.1):
        """Conductivity spectrum: sigma(omega) ~ (1/N) Re FT[C(t)].

        Gaussian time window exp(-(eta t)^2/2) and the C(-t)=conj(C(t)) extension
        (J Hermitian). Returns (omegas, sigma).
        """
        t = res.times
        dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
        win = np.exp(-0.5 * (eta * t) ** 2)
        if omega_max is None:
            omega_max = 2.0 * np.pi
        omegas = np.linspace(0.0, omega_max, n_omega)
        ewt = np.exp(1j * np.outer(omegas, t))           # (n_w, n_t)
        wC = win * res.C
        s = 2.0 * np.real(ewt @ wC) * dt
        s -= np.real(win[0] * res.C[0]) * dt             # de-double-count t=0
        return omegas, s / res.n
