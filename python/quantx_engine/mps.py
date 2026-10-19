"""Matrix-product states that stay in the engine (M11).

A state the engine makes stays there, under a handle, until it is freed: a ground state,
one-site operators, time evolution and measurements chain without the state passing
through Python or JSON. The methods that change a state do it in place and return it, so
they compose::

    from quantx_engine import mps
    from quantx_engine.models import XXZChain

    chain = XXZChain(L=32)
    gs = mps.ground_state(chain, max_bond=128)
    phi = gs.copy().apply("Sz", 16)                 # S^z_16 |GS>, not renormalized
    gs.evolve(chain, dt=0.05, steps=20)             # e^{-iHt} |GS> = e^{-i E0 t} |GS>
    phi.evolve(chain, dt=0.05, steps=20)
    c = gs.matrix_elements("Sz", phi)               # <GS| e^{iHt} S^z_j e^{-iHt} S^z_16 |GS>

The operators are the one-site diagonal ones of each local space (:data:`OPERATORS`), in
the conventions of :mod:`quantx_engine.models` (S = sigma/2). Evolution is two-site TDVP;
a state may be evolved under another model of the same length and local space than the
one it was made with (a quench).
"""
from __future__ import annotations

import json
import operator
from ctypes import c_char_p
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

__all__ = ["MPSError", "MPS", "GroundState", "ground_state", "product", "OPERATORS",
           "LOCAL_STATES"]


class MPSError(RuntimeError):
    """A refused request, or a failure in the engine."""


# The local states of each space, in the engine's order, and the diagonal one-site
# operators on them (physical units: S = sigma/2).
LOCAL_STATES = {
    "spin-1/2": ("up", "dn"),
    "spin-1": ("+1", "0", "-1"),
    "hubbard": ("0", "up", "dn", "ud"),
}
OPERATORS = {
    "spin-1/2": {"Id": (1, 1), "Sz": (0.5, -0.5)},
    "spin-1": {"Id": (1, 1, 1), "Sz": (1, 0, -1), "Sz2": (1, 0, 1)},
    "hubbard": {"Id": (1, 1, 1, 1), "N": (0, 1, 1, 2), "Nup": (0, 1, 0, 1),
                "Ndn": (0, 0, 1, 1), "Sz": (0, 0.5, -0.5, 0), "D": (0, 0, 0, 1)},
}


def _space(model) -> str:
    from quantx_engine.models import (HubbardChain, HubbardLattice, J1J2Chain, Spin1Chain,
                                      XXZChain)
    if isinstance(model, (XXZChain, J1J2Chain)):
        return "spin-1/2"
    if isinstance(model, Spin1Chain):
        return "spin-1"
    if isinstance(model, (HubbardChain, HubbardLattice)):
        return "hubbard"
    raise MPSError(f"not a quantx_engine.models model: {model!r}")


def _diag(space: str, op) -> list:
    """The operator's diagonal on the local space: a name from OPERATORS, or the entries."""
    if isinstance(op, str):
        try:
            return [float(x) for x in OPERATORS[space][op]]
        except KeyError:
            raise MPSError(f"no operator {op!r} on a {space} site; expected one of "
                           f"{sorted(OPERATORS[space])}, or its diagonal") from None
    d = np.asarray(op, dtype=float)
    if d.shape != (len(LOCAL_STATES[space]),):
        raise MPSError(f"a {space} operator's diagonal has {len(LOCAL_STATES[space])} "
                       f"entries, got shape {d.shape}")
    return d.tolist()


def _int(v, name: str) -> int:
    try:
        if isinstance(v, bool):
            raise TypeError
        return operator.index(v)
    except TypeError:
        raise MPSError(f"{name} must be an int, got {v!r}") from None


def _lib(engine):
    from quantx_engine.dmrg import _resolve_engine
    lib = getattr(_resolve_engine(engine, "mps", MPSError), "_lib", None)
    if lib is None or not hasattr(lib, "qx_mps"):
        raise MPSError("the loaded native library does not export qx_mps")
    lib.qx_mps.argtypes = [c_char_p]
    lib.qx_mps.restype = c_char_p
    return lib


def _call(lib, request: dict) -> dict:
    out = json.loads(lib.qx_mps(json.dumps(request).encode()))
    if not out.get("ok", False):
        raise MPSError(out.get("error", "unknown error from the native engine"))
    return out


@dataclass(frozen=True)
class GroundState:
    """How :func:`ground_state`'s DMRG went."""
    energy: float
    truncation_error: float
    n_sweeps_done: int
    sweep_energies: list = field(default_factory=list)
    sector: dict = field(default_factory=dict)
    wall_time_s: float = 0.0


class MPS:
    """A matrix-product state held by the native engine, until it is freed.

    Make one with :func:`ground_state` or :func:`product`. The methods that change it do
    so in place and return it; :meth:`copy` makes an independent state. The engine frees
    the state when this object is collected, or at :meth:`free`.
    """

    def __init__(self, handle: int, model, lib, dmrg: Optional[GroundState] = None):
        self._handle, self._lib, self.model, self.dmrg = handle, lib, model, dmrg
        self.space = _space(model)

    # -- lifetime --------------------------------------------------------------------
    def free(self) -> None:
        """Free the engine's state now; the object is unusable afterwards."""
        if self._handle is not None:
            h, self._handle = self._handle, None
            _call(self._lib, {"op": "free", "handle": h})

    def __del__(self):
        try:
            self.free()
        except Exception:      # interpreter shutdown, or a library already unloaded
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.free()

    def _h(self) -> int:
        if self._handle is None:
            raise MPSError("this state has been freed")
        return self._handle

    def _peer(self, other: "MPS") -> int:
        if not isinstance(other, MPS):
            raise MPSError(f"expected an MPS, got {type(other).__name__}")
        if other._lib is not self._lib:
            raise MPSError("the two states live in different engines")
        return other._h()

    # -- reading ---------------------------------------------------------------------
    def info(self) -> dict:
        """n_sites, local_dim, bonds (bonds 1..L-1), norm2 and charge (engine units)."""
        out = _call(self._lib, {"op": "info", "handle": self._h()})
        return {k: out[k] for k in ("n_sites", "local_dim", "bonds", "norm2", "charge")}

    @property
    def n_sites(self) -> int:
        return int(self.info()["n_sites"])

    @property
    def bonds(self) -> list:
        return [int(b) for b in self.info()["bonds"]]

    @property
    def norm(self) -> float:
        return float(np.sqrt(self.info()["norm2"]))

    def overlap(self, other: "MPS") -> complex:
        """``<self|other>``."""
        out = _call(self._lib, {"op": "overlap", "bra": self._h(), "ket": self._peer(other)})
        return complex(out["re"], out["im"])

    def matrix_elements(self, op, ket: "MPS") -> np.ndarray:
        """``<self| O_j |ket>`` for every site j, as they are: neither state is normalized."""
        out = _call(self._lib, {"op": "expect", "bra": self._h(), "ket": self._peer(ket),
                                "diag": _diag(self.space, op)})
        return np.asarray(out["re"]) + 1j * np.asarray(out["im"])

    def expect(self, op) -> np.ndarray:
        """``<O_j> = <self|O_j|self> / <self|self>`` for every site j."""
        out = _call(self._lib, {"op": "expect", "bra": self._h(),
                                "diag": _diag(self.space, op)})
        return np.asarray(out["re"]) / self.info()["norm2"]

    def energy(self, model=None) -> float:
        """``<H> / <self|self>`` for a model (default: the one this state was made with)."""
        model = self.model if model is None else model
        self._same_space(model)
        return float(_call(self._lib, {"op": "energy", "handle": self._h(),
                                       "model": model.problem()})["energy"])

    # -- changing --------------------------------------------------------------------
    def copy(self) -> "MPS":
        out = _call(self._lib, {"op": "copy", "handle": self._h()})
        return MPS(int(out["handle"]), self.model, self._lib, self.dmrg)

    def apply(self, op, site: int) -> "MPS":
        """``O_site |self>``, in place and not renormalized; O is diagonal on the local space."""
        _call(self._lib, {"op": "apply", "handle": self._h(), "site": _int(site, "site"),
                          "diag": _diag(self.space, op)})
        return self

    def evolve(self, model=None, *, dt: float, steps: int = 1, imaginary: bool = False,
               max_bond: int = 256, cutoff: float = 1e-10) -> "MPS":
        """``steps`` two-site TDVP steps of exp(-i H dt), in place.

        With ``imaginary``, of exp(-H dt), renormalized after each step. ``model`` is the
        Hamiltonian (default: the one this state was made with); another of the same length
        and local space is a quench.
        """
        model = self.model if model is None else model
        self._same_space(model)
        _call(self._lib, {"op": "evolve", "handle": self._h(), "model": model.problem(),
                          "dt": float(dt), "n_steps": _int(steps, "steps"),
                          "imaginary": bool(imaginary), "max_bond": _int(max_bond, "max_bond"),
                          "cutoff": float(cutoff)})
        return self

    def _same_space(self, model) -> None:
        if _space(model) != self.space or model.L != self.model.L:
            raise MPSError(f"the model is a {model.L}-site {_space(model)} chain, the state "
                           f"a {self.model.L}-site {self.space} one")

    def __repr__(self) -> str:
        if self._handle is None:
            return "MPS(freed)"
        b = self.bonds
        return (f"MPS({self.space}, n_sites={len(b) + 1}, max bond {max(b, default=1)}, "
                f"handle {self._handle})")


def _sector(model, sz, n_up, n_dn) -> dict:
    from quantx_engine.models import _Hubbard, _SpinChain
    if isinstance(model, _Hubbard):
        if sz is not None:
            raise MPSError(f"{type(model).__name__}'s sector is n_up / n_dn, not sz")
        return model.sector(n_up, n_dn)
    if isinstance(model, _SpinChain):
        if n_up is not None or n_dn is not None:
            raise MPSError(f"{type(model).__name__}'s sector is sz, not n_up / n_dn")
        return model.sector(sz)
    raise MPSError(f"not a quantx_engine.models model: {model!r}")


def ground_state(model, *, sz: Optional[float] = None, n_up: Optional[int] = None,
                 n_dn: Optional[int] = None, max_bond: int = 256, cutoff: float = 1e-10,
                 n_sweeps: int = 20, tol: float = 1e-10, engine=None) -> MPS:
    """The DMRG ground state of a model in one symmetry sector, as an :class:`MPS`.

    The sector and the options are those of :func:`quantx_engine.dmrg.ground_state`; how
    the DMRG went is on the result's ``dmrg``.
    """
    sector = _sector(model, sz, n_up, n_dn)
    lib = _lib(engine)
    out = _call(lib, {"op": "ground_state", "model": dict(model.problem(), sector=sector),
                      "max_bond": _int(max_bond, "max_bond"), "cutoff": float(cutoff),
                      "n_sweeps": _int(n_sweeps, "n_sweeps"), "tol": float(tol)})
    info = GroundState(energy=float(out["energy"]),
                       truncation_error=float(out["truncation_error"]),
                       n_sweeps_done=int(out["n_sweeps_done"]),
                       sweep_energies=[float(e) for e in out["sweep_energies"]],
                       sector=model.describe_sector(out["sector"]),
                       wall_time_s=float(out["wall_time_s"]))
    return MPS(int(out["handle"]), model, lib, info)


def product(model, config: Sequence[Union[int, str]], *, engine=None) -> MPS:
    """The product state with local state ``config[i]`` on site i.

    A local state is its index in the engine's order or its name in
    :data:`LOCAL_STATES` (spin 1/2 "up", "dn"; spin 1 "+1", "0", "-1"; Hubbard "0",
    "up", "dn", "ud").
    """
    space = _space(model)
    names = LOCAL_STATES[space]
    idx = []
    for i, c in enumerate(config):
        if isinstance(c, str):
            if c not in names:
                raise MPSError(f"site {i}: no {space} local state {c!r}; expected one of "
                               f"{list(names)}")
            idx.append(names.index(c))
        else:
            idx.append(_int(c, f"config[{i}]"))
    lib = _lib(engine)
    out = _call(lib, {"op": "product", "model": model.problem(), "config": idx})
    return MPS(int(out["handle"]), model, lib)
