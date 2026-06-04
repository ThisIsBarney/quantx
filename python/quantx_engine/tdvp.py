"""TDVP time evolution for MPS — Python wrapper over the native C++ engine.

Wraps ``qx_tdvp_evolve`` (``src/tdvp.hpp`` via ``src/dmrg_capi.cpp``): real- and
imaginary-time evolution of a spin-1/2 Heisenberg chain by the time-dependent
variational principle (TDVP1 one-site, or TDVP2 two-site bond-adaptive).

Example
-------
    from quantx_engine import Engine
    from quantx_engine.tdvp import TDVPSolver

    s = TDVPSolver(Engine())
    # real-time quench from the Neel state
    r = s.evolve(10, dt=0.05, n_steps=40, chi_max=64, two_site=True)
    print(r.times, r.staggered_mag)   # <m_stag>(t)

    # imaginary-time cooling to the ground state
    g = s.evolve(10, dt=0.1, n_steps=200, imag=True)
    print(g.energy)
"""
from __future__ import annotations

import json
from ctypes import c_char_p
from dataclasses import dataclass
from typing import Any

__all__ = ["TDVPSolver", "TDVPResult", "TDVPError"]


class TDVPError(RuntimeError):
    """Raised when the TDVP solver fails or the native library is unavailable."""


@dataclass
class TDVPResult:
    """Result of a TDVP evolution.

    Attributes
    ----------
    energy : float
        Final ground-state-convention energy ``<psi|H|psi>`` (H = J sum sigma.sigma).
    sz : list[float]
        Final ``<sigma^z_i>`` per site.
    times : list[float]
        Time grid (length n_steps+1, including t=0).
    staggered_mag : list[float]
        Staggered magnetization ``(1/N) sum_i (-1)^i <sigma^z_i>`` at each time.
    max_bond : int
        Largest MPS bond dimension reached.
    wall_time_s : float
        Native wall-clock time for the evolution loop.
    method : str
        ``"tdvp1"`` or ``"tdvp2"``.
    """
    energy: float
    sz: list[float]
    times: list[float]
    staggered_mag: list[float]
    max_bond: int
    wall_time_s: float
    method: str


class TDVPSolver:
    """TDVP time-evolution solver (native C++ backend; Heisenberg chain)."""

    def __init__(self, engine) -> None:
        self._engine = engine
        self._lib = None
        lib = getattr(engine, "_lib", None)
        if lib is not None:
            try:
                fn = lib.qx_tdvp_evolve
                fn.argtypes = [c_char_p, c_char_p]
                fn.restype = c_char_p
                self._lib = lib
            except AttributeError:
                self._lib = None

    def evolve(
        self,
        n_qubits: int,
        *,
        J: float = 1.0,
        delta: float = 1.0,
        dt: float = 0.05,
        n_steps: int = 20,
        chi_max: int = 64,
        imag: bool = False,
        two_site: bool = True,
        initial: str = "neel",
        block: bool = True,
        engine: str = "tebd",
        measure_stride: int = 1,
    ) -> TDVPResult:
        """Evolve a Heisenberg chain by TDVP.

        Parameters
        ----------
        n_qubits : int
            Number of spin-1/2 sites.
        J : float
            Heisenberg coupling (H = J sum_i sigma_i . sigma_{i+1}).
        dt : float
            Time step.
        n_steps : int
            Number of steps (total time = dt * n_steps).
        chi_max : int
            Maximum bond dimension (TDVP2 truncation).
        imag : bool
            Imaginary-time evolution (cooling toward the ground state) if True;
            real-time otherwise.
        two_site : bool
            TDVP2 (bond-adaptive, needed for entanglement growth) if True;
            TDVP1 (fixed bond) otherwise.
        initial : str
            Initial state: ``"neel"`` (|up dn up dn...>) or ``"up"`` (all up).

        Returns
        -------
        TDVPResult
        """
        if self._lib is None:
            raise TDVPError("TDVP requires the native C++ library (no Python fallback).")
        if n_qubits < 2:
            raise TDVPError("n_qubits must be >= 2")
        if n_steps < 1:
            raise TDVPError("n_steps must be >= 1")
        prob = {"n_qubits": n_qubits, "type": "heisenberg", "J": J, "delta": delta}
        opts = {"dt": dt, "n_steps": n_steps, "chi_max": chi_max,
                "imag": bool(imag), "two_site": bool(two_site), "initial": initial,
                "block": bool(block), "engine": engine,
                "measure_stride": int(measure_stride)}
        raw = self._lib.qx_tdvp_evolve(json.dumps(prob).encode(), json.dumps(opts).encode())
        out = json.loads(raw)
        if not out.get("ok", False):
            raise TDVPError(out.get("error", "Unknown error from qx_tdvp_evolve"))
        return TDVPResult(
            energy=float(out["energy"]),
            sz=[float(x) for x in out.get("sz", [])],
            times=[float(x) for x in out.get("times", [])],
            staggered_mag=[float(x) for x in out.get("staggered_mag", [])],
            max_bond=int(out.get("max_bond", 0)),
            wall_time_s=float(out.get("wall_time_s", 0.0)),
            method=str(out.get("method", "")),
        )
