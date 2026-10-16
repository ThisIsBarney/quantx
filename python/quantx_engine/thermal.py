"""Finite-temperature statics and dynamics of the spin and Hubbard chains.

The Gibbs state rho = e^{-beta H} / Z of an open chain is held on the native engine by
ancilla purification. Each physical site is paired with an ancilla, and the pure state
e^{-beta H/2}|I> on the doubled chain (|I> is the infinite-temperature state) is reached
by imaginary-time block TEBD. Its real-time evolution gives the dynamical correlations.
Three quantities are exposed:

- :func:`statics`: the energy, local magnetization or density, correlations and the
  uniform susceptibility over a range of inverse temperatures;
- :func:`structure_factor`: the spin correlations C_j(t) = <S^z_j(t) S^z_c(0)>_beta,
  whose Fourier transform is the dynamical structure factor S(k, omega, T);
- :func:`current_correlation`: the current-current correlation <J(t) J(0)>_beta,
  whose Fourier transform gives the optical conductivity sigma(omega, T).

Example::

    from quantx_engine import thermal
    from quantx_engine.models import HubbardChain, XXZChain

    st = thermal.statics(XXZChain(L=16), beta=2.0)
    st.energy[-1]                          # <H> at beta = 2

    sf = thermal.structure_factor(XXZChain(L=16), beta=1.0, t_max=6.0)
    k, w, S = sf.spectrum()                # S(k, omega, T)

    cc = thermal.current_correlation(HubbardChain(L=12, U=4.0, mu=2.0), beta=1.0, t_max=6.0)
    w, sigma = cc.conductivity()           # Re sigma(omega, T)

Conventions are those of :mod:`quantx_engine.models` (spins S = sigma/2). The ensemble is
grand canonical: every S^z sector, and for Hubbard every particle number, with ``mu``
setting the filling (``mu = U/2`` is half filling at every temperature).

The models are :class:`~quantx_engine.models.XXZChain`,
:class:`~quantx_engine.models.Spin1Chain` and :class:`~quantx_engine.models.HubbardChain`.
:class:`~quantx_engine.models.J1J2Chain` and the Hubbard lattices have no purified form
in the engine yet, and are refused.

Everything here is gated against exact diagonalisation of the full Hilbert space
(``tests/test_thermal_ed.py``), with the second-order Trotter error of the steps
``dbeta`` and ``dt`` as the tolerance.

References
----------
- Verstraete, Garcia-Ripoll and Cirac (2004). Phys. Rev. Lett. 93, 207204.
  DOI: 10.1103/PhysRevLett.93.207204 (ancilla purification)
- Karrasch, Bardarson and Moore (2012). Phys. Rev. Lett. 108, 227206.
  DOI: 10.1103/PhysRevLett.108.227206 (the backward ancilla evolution, ``disentangle``)
"""
from __future__ import annotations

import json
import math
import numbers
import operator
from ctypes import c_char_p
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

__all__ = ["ThermalError", "ThermalStatics", "ThermalStructureFactor",
           "ThermalCurrentCorrelation", "statics", "structure_factor",
           "current_correlation"]


class ThermalError(RuntimeError):
    """A finite-temperature run could not be done, with the reason."""


# ── Models and their unit conversions ─────────────────────────────────────────

@dataclass(frozen=True)
class _Kind:
    observable: str     # what "local" holds: "Sz" or "n"
    o_scale: float      # engine O_j (2Sz or n) -> local observable
    s_scale: float      # engine C_{j,c} of S = 2Sz -> <S^z_j(t) S^z_c>
    j_scale: float      # engine <J(t)J> -> the documented current's


def _kind(model) -> _Kind:
    from quantx_engine.models import HubbardChain, Spin1Chain, XXZChain
    if isinstance(model, XXZChain):
        # The engine runs J sigma.sigma with J/4; its current is i (J/4) sum (S+S- - h.c.),
        # half the S^z current (iJ/2) sum (S+S- - h.c.) of H.
        return _Kind("Sz", 0.5, 0.25, 4.0)
    if isinstance(model, Spin1Chain):
        # The engine current is i J sum (S+S- - h.c.), twice the S^z current of H.
        return _Kind("Sz", 0.5, 0.25, 0.25)
    if isinstance(model, HubbardChain):
        # O_j is n_j; the structure factor's S is n_up - n_dn = 2 Sz; the current is the
        # particle current (its sign is the engine's opposite, which <J J> does not see).
        return _Kind("n", 1.0, 0.25, 1.0)
    name = type(model).__name__
    raise ThermalError(f"finite temperature supports XXZChain, Spin1Chain and HubbardChain; "
                       f"the engine has no purified form for {name}")


def _check_positive(name: str, v, *, allow_zero: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise ThermalError(f"{name} must be a number, got {v!r}")
    x = float(v)
    if not math.isfinite(x) or x < 0 or (x == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ThermalError(f"{name} must be {bound}, got {v!r}")
    return x


def _check_int(name: str, v, low: int = 1) -> int:
    try:
        if isinstance(v, bool):
            raise TypeError
        i = operator.index(v)
    except TypeError:
        raise ThermalError(f"{name} must be an int, got {v!r}") from None
    if i < low:
        raise ThermalError(f"{name} must be >= {low}, got {i}")
    return i


def _run(model, options: dict, engine) -> dict:
    from quantx_engine.dmrg import _resolve_engine
    lib = getattr(_resolve_engine(engine, "thermal", ThermalError), "_lib", None)
    fn = getattr(lib, "qx_thermal", None) if lib is not None else None
    if fn is None:
        raise ThermalError("the loaded native library does not export qx_thermal")
    fn.argtypes = [c_char_p, c_char_p]
    fn.restype = c_char_p
    out = json.loads(fn(json.dumps(model.problem()).encode(), json.dumps(options).encode()))
    if not out.get("ok", False):
        raise ThermalError(out.get("error", "unknown error from qx_thermal"))
    return out


# ── Statics ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThermalStatics:
    """Thermal expectation values on a grid of inverse temperatures, from beta = 0 up.

    Attributes
    ----------
    betas : np.ndarray, shape (n_beta,)
    energy : np.ndarray, shape (n_beta,)
        <H>, for the whole chain.
    local : np.ndarray, shape (n_beta, L)
        <S^z_j> for the spin chains, <n_j> for Hubbard (see ``observable``).
    correlation : np.ndarray, shape (n_beta, L)
        <O_c O_j> with the central site c = ``center`` and O = S^z or n.
    susceptibility : np.ndarray, shape (n_beta,)
        beta Var(sum_j O_j) / L: the uniform spin susceptibility per site, or for Hubbard
        the compressibility. Zeros when not requested.
    center : int
    observable : str
        ``"Sz"`` or ``"n"``.
    max_bond : int
        The largest bond dimension of the purified state.
    wall_time_s : float
    """
    betas: np.ndarray
    energy: np.ndarray
    local: np.ndarray
    correlation: np.ndarray
    susceptibility: np.ndarray
    center: int
    observable: str
    max_bond: int
    wall_time_s: float


def statics(model, beta: float, *, dbeta: float = 0.05, max_bond: int = 128,
            cutoff: float = 1e-10, stride: int = 1, susceptibility: bool = True,
            engine=None) -> ThermalStatics:
    """Energy, local observables, correlations and susceptibility from beta = 0 to ``beta``.

    Parameters
    ----------
    model : XXZChain, Spin1Chain or HubbardChain
    beta : float
        The largest inverse temperature of the scan.
    dbeta : float
        The cooling step. ``beta / dbeta`` is rounded to a whole number of steps, and the
        step shrinks to end exactly at ``beta``. The error is second order in it.
    max_bond : int
        Bond dimension cap of the purified state.
    cutoff : float
        SVD truncation weight cutoff.
    stride : int
        Measure every ``stride``-th step (and at the end).
    susceptibility : bool
        Also compute the susceptibility, at the cost of L overlaps per measurement.
    engine : quantx_engine.Engine, optional

    Raises
    ------
    ThermalError
        An unsupported model, a bad argument, no native library, or a solver failure.
    """
    kind = _kind(model)
    opts = {"quantity": "statics", "beta": _check_positive("beta", beta),
            "dbeta": _check_positive("dbeta", dbeta),
            "max_bond": _check_int("max_bond", max_bond), "cutoff": float(cutoff),
            "stride": _check_int("stride", stride), "susceptibility": bool(susceptibility)}
    out = _run(model, opts, engine)
    s = kind.o_scale
    return ThermalStatics(
        betas=np.asarray(out["betas"], dtype=float),
        energy=np.asarray(out["energy"], dtype=float),
        local=s * np.asarray(out["O"], dtype=float),
        correlation=s * s * np.asarray(out["OO"], dtype=float),
        susceptibility=s * s * np.asarray(out["susceptibility"], dtype=float),
        center=int(out["center"]), observable=kind.observable,
        max_bond=int(out.get("max_bond", 0)), wall_time_s=float(out.get("wall_time_s", 0.0)))


# ── Dynamics ─────────────────────────────────────────────────────────────────

def _window(times: np.ndarray, eta: Optional[float]) -> np.ndarray:
    """Gaussian time window exp(-(eta t)^2 / 2); by default e^{-4.5} at the last time."""
    t_max = float(times[-1]) if len(times) > 1 else 1.0
    eta = 3.0 / t_max if eta is None else float(eta)
    return np.exp(-0.5 * (eta * times) ** 2)


def _omegas(times: np.ndarray, n_omega: int, omega_max: Optional[float], *,
            low: float = -1.0) -> np.ndarray:
    """n_omega points on [low * omega_max, omega_max]; omega_max defaults to pi / dt."""
    dt = float(times[1] - times[0]) if len(times) > 1 else 1.0
    w = math.pi / dt if omega_max is None else float(omega_max)
    return np.linspace(low * w, w, int(n_omega))


def _fourier(times: np.ndarray, C: np.ndarray, omegas: np.ndarray,
             window: np.ndarray) -> np.ndarray:
    """int dt e^{i w t} C(t) over all t, from t >= 0 and C(-t) = conj(C(t)).

    C has time along its first axis; the result has omega first.
    """
    dt = float(times[1] - times[0]) if len(times) > 1 else 1.0
    wC = window.reshape((-1,) + (1,) * (C.ndim - 1)) * C
    ewt = np.exp(1j * np.outer(omegas, times))                    # (n_w, n_t)
    F = 2.0 * np.real(np.tensordot(ewt, wC, axes=(1, 0))) * dt    # trapezoid, t = 0 twice
    return F - np.real(wC[0]) * dt


@dataclass(frozen=True)
class ThermalStructureFactor:
    """Spin correlations C_j(t) = <S^z_j(t) S^z_c(0)>_beta of a thermal state.

    Attributes
    ----------
    beta : float
    times : np.ndarray, shape (n_t,)
    correlation : np.ndarray, shape (n_t, L), complex
        ``correlation[t, j]``; the source is the central site c = ``source``.
    source : int
    energy : float
        <H> at ``beta``.
    max_bond : int
    bond_dims : np.ndarray, shape (n_t,)
        The largest bond dimension at each time.
    wall_time_s : float
    """
    beta: float
    times: np.ndarray
    correlation: np.ndarray
    source: int
    energy: float
    max_bond: int
    bond_dims: np.ndarray
    wall_time_s: float

    def spectrum(self, *, n_omega: int = 401, omega_max: Optional[float] = None,
                 eta: Optional[float] = None):
        """The dynamical structure factor S(k, omega, T).

        S(k, omega) = sum_r e^{-i k r} int dt e^{i omega t} C_{c+r}(t), on the open chain
        (r measured from the source), with C(-t) = conj(C(t)) for the negative times and
        a Gaussian time window exp(-(eta t)^2 / 2) against the cut-off at the last time.
        That extension is exact for the local correlation (H is real) and at every k on
        an odd chain, whose centre is a mirror point; on an even chain it is the usual
        open-chain approximation. At finite temperature both signs of omega carry
        weight, related by detailed balance, S(k, -omega) = e^{-beta omega} S(k, omega),
        up to the window's broadening.

        Parameters
        ----------
        n_omega : int
            Points of the omega grid on [-omega_max, omega_max].
        omega_max : float, optional
            Default pi / dt, the Nyquist frequency of the time grid.
        eta : float, optional
            Window width; default 3 / t_max.

        Returns
        -------
        ks : np.ndarray, shape (L,)
            Momenta 2 pi m / L.
        omegas : np.ndarray, shape (n_omega,)
        S : np.ndarray, shape (L, n_omega), real
        """
        L = self.correlation.shape[1]
        ks = 2.0 * np.pi * np.arange(L) / L
        r = np.arange(L) - self.source
        Ck = self.correlation @ np.exp(-1j * np.outer(ks, r)).T       # (n_t, n_k)
        omegas = _omegas(self.times, n_omega, omega_max)
        S = _fourier(self.times, Ck, omegas, _window(self.times, eta))   # (n_w, n_k)
        return ks, omegas, S.T


@dataclass(frozen=True)
class ThermalCurrentCorrelation:
    """Current-current correlation C(t) = <J(t) J(0)>_beta of a thermal state.

    J is the current of the conserved charge: for the spin chains the S^z current
    J = sum_l (i J / 2) (S+_l S-_{l+1} - h.c.), and for Hubbard the particle current
    J = sum_{l,s} i t (c+_{l+1 s} c_{l s} - h.c.).

    Attributes
    ----------
    beta : float
    times : np.ndarray, shape (n_t,)
    correlation : np.ndarray, shape (n_t,), complex
    n_sites : int
    energy : float
    max_bond : int
    bond_dims : np.ndarray, shape (n_t,)
    wall_time_s : float
    """
    beta: float
    times: np.ndarray
    correlation: np.ndarray
    n_sites: int
    energy: float
    max_bond: int
    bond_dims: np.ndarray
    wall_time_s: float

    def conductivity(self, *, n_omega: int = 401, omega_max: Optional[float] = None,
                     eta: Optional[float] = None):
        """Re sigma(omega, T), the optical conductivity per site, for omega >= 0.

        Re sigma(omega) = (1 - e^{-beta omega}) / (2 L omega) int dt e^{i omega t} C(t),
        whose omega -> 0 limit is beta / (2 L) times the integral. The Fourier transform
        is the one of :meth:`ThermalStructureFactor.spectrum`. Re sigma is even in omega,
        and only omega >= 0 is computed: below zero the Kubo factor grows as
        e^{beta |omega|}, against a windowed transform that is not exponentially small.

        Parameters
        ----------
        n_omega : int
            Points of the omega grid on [0, omega_max].
        omega_max : float, optional
            Default pi / dt, the Nyquist frequency of the time grid.
        eta : float, optional
            Window width; default 3 / t_max.

        Returns
        -------
        omegas : np.ndarray, shape (n_omega,)
        sigma : np.ndarray, shape (n_omega,)
        """
        omegas = _omegas(self.times, n_omega, omega_max, low=0.0)
        Ct = _fourier(self.times, self.correlation, omegas, _window(self.times, eta))
        b = self.beta
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(np.abs(omegas) > 1e-12, -np.expm1(-b * omegas) / omegas, b)
        return omegas, factor * Ct / (2.0 * self.n_sites)


def _dynamics_options(quantity: str, beta, t_max, dt, dbeta, max_bond, cutoff,
                      disentangle) -> dict:
    return {"quantity": quantity, "beta": _check_positive("beta", beta),
            "t_max": _check_positive("t_max", t_max, allow_zero=True),
            "dt": _check_positive("dt", dt), "dbeta": _check_positive("dbeta", dbeta),
            "max_bond": _check_int("max_bond", max_bond), "cutoff": float(cutoff),
            "disentangle": bool(disentangle)}


def structure_factor(model, beta: float, *, t_max: float = 5.0, dt: float = 0.05,
                     dbeta: float = 0.05, max_bond: int = 128, cutoff: float = 1e-10,
                     disentangle: bool = True, engine=None) -> ThermalStructureFactor:
    """Spin correlations <S^z_j(t) S^z_c(0)>_beta from the central site, up to ``t_max``.

    Parameters
    ----------
    model : XXZChain, Spin1Chain or HubbardChain
        For Hubbard, S^z = (n_up - n_dn) / 2: the spin structure factor.
    beta : float
        Inverse temperature.
    t_max, dt : float
        The real-time window and step (second-order Trotter error in ``dt``).
    dbeta : float
        Cooling step (second-order error).
    max_bond, cutoff
        Truncation of the purified states.
    disentangle : bool
        Evolve the ancillas backwards (Karrasch, Bardarson and Moore). Exact without
        truncation, and it slows the growth of the bond dimension.
    engine : quantx_engine.Engine, optional

    Raises
    ------
    ThermalError
    """
    kind = _kind(model)
    out = _run(model, _dynamics_options("structure_factor", beta, t_max, dt, dbeta, max_bond,
                                        cutoff, disentangle), engine)
    C = np.asarray(out["C_re"], dtype=float) + 1j * np.asarray(out["C_im"], dtype=float)
    return ThermalStructureFactor(
        beta=float(out["beta"]), times=np.asarray(out["times"], dtype=float),
        correlation=kind.s_scale * C, source=int(out["source"]), energy=float(out["energy"]),
        max_bond=int(out.get("max_bond", 0)),
        bond_dims=np.asarray(out.get("bond_t", []), dtype=int),
        wall_time_s=float(out.get("wall_time_s", 0.0)))


def current_correlation(model, beta: float, *, t_max: float = 5.0, dt: float = 0.05,
                        dbeta: float = 0.05, max_bond: int = 128, cutoff: float = 1e-10,
                        disentangle: bool = True, engine=None) -> ThermalCurrentCorrelation:
    """Current-current correlation <J(t) J(0)>_beta up to ``t_max``.

    The arguments are those of :func:`structure_factor`; see
    :class:`ThermalCurrentCorrelation` for the current J.

    Raises
    ------
    ThermalError
    """
    kind = _kind(model)
    out = _run(model, _dynamics_options("current", beta, t_max, dt, dbeta, max_bond, cutoff,
                                        disentangle), engine)
    C = np.asarray(out["C_re"], dtype=float) + 1j * np.asarray(out["C_im"], dtype=float)
    return ThermalCurrentCorrelation(
        beta=float(out["beta"]), times=np.asarray(out["times"], dtype=float),
        correlation=kind.j_scale * C, n_sites=int(out["n_sites"]),
        energy=float(out["energy"]), max_bond=int(out.get("max_bond", 0)),
        bond_dims=np.asarray(out.get("bond_t", []), dtype=int),
        wall_time_s=float(out.get("wall_time_s", 0.0)))
