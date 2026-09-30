"""Lattice models for the native symmetric (block-sparse) engine.

Build a model, then hand it to a solver::

    from quantx_engine.models import HubbardChain, XXZChain
    from quantx_engine.dmrg import ground_state

    gs = ground_state(HubbardChain(L=16, U=4.0), n_up=8, n_dn=8, max_bond=512)
    gs.energy

    gs = ground_state(XXZChain(L=21, delta=0.5), sz=0.5)

Every model is a 1D chain with open boundaries. The engine conserves the model's U(1)
charges exactly, so a solver works inside one symmetry sector: total S^z for the spin
chains, the numbers of up and down electrons for Hubbard.

Conventions
-----------
Spin models use physical spin operators throughout, S = sigma/2 for spin-1/2 and the
spin-1 matrices for spin-1, so couplings and energies compare directly with the
literature (the open spin-1/2 Heisenberg chain has E/L -> 1/4 - ln 2 per site).

This differs from ``DMRGSolver(hamiltonian_type="heisenberg")``, which keeps the
engine's Pauli convention, J sigma.sigma = 4 J S.S. The conversion happens here, in
:meth:`problem`: the engine's spin-1/2 builders take Pauli couplings, so J is sent as J/4.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass

__all__ = ["XXZChain", "J1J2Chain", "Spin1Chain", "HubbardChain", "HubbardLattice",
           "ModelError"]

# The C API refuses longer chains; the block engine derives its symmetry data from a
# bulk MPO site, so it needs at least three.
_MIN_SITES, _MAX_SITES = 3, 200


class ModelError(ValueError):
    """Raised for an invalid model or symmetry sector."""


def _as_int(v, name: str) -> int:
    """v as a plain int. numpy integers are welcome (np.arange loops); bools and floats
    are refused. Plain ints are also what the JSON sent to the engine can hold."""
    if isinstance(v, bool):
        raise ModelError(f"{name} must be an int, got {v!r}")
    try:
        return operator.index(v)
    except TypeError:
        raise ModelError(f"{name} must be an int, got {type(v).__name__}") from None


def _check_length(L, name: str = "L") -> int:
    L = _as_int(L, name)
    if not _MIN_SITES <= L <= _MAX_SITES:
        raise ModelError(f"{name} must be in [{_MIN_SITES}, {_MAX_SITES}], got {L}")
    return L


class _SpinChain:
    """Sector handling shared by the spin models: total S^z, in units of hbar."""

    L: int
    two_s = 1   # 2S per site

    def sector(self, sz: float | None = None) -> dict:
        """C API sector for total S^z = ``sz``; default the lowest |S^z| possible."""
        max2 = self.two_s * self.L
        sz2 = max2 % 2 if sz is None else int(round(2 * sz))
        if sz is not None and abs(2 * sz - sz2) > 1e-9:
            raise ModelError(f"sz must be a multiple of 1/2, got {sz}")
        if abs(sz2) > max2 or (max2 - sz2) % 2:
            spin = "1/2" if self.two_s == 1 else "1"
            parity = "an integer" if max2 % 2 == 0 else "a half-odd integer"
            raise ModelError(
                f"sz={sz2 / 2:g} is not reachable on {self.L} spin-{spin} sites: "
                f"need |sz| <= {max2 / 2:g} and sz {parity}")
        return {"sz2": sz2}

    @staticmethod
    def describe_sector(sector: dict) -> dict:
        return {"sz": sector["sz2"] / 2}


@dataclass(frozen=True)
class XXZChain(_SpinChain):
    """Spin-1/2 XXZ chain, H = J sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + delta Sz_i Sz_{i+1}).

    ``delta = 1`` is the isotropic Heisenberg chain.
    """
    L: int
    J: float = 1.0
    delta: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "L", _check_length(self.L))

    def problem(self) -> dict:
        """The C API problem (Pauli convention: J sigma.sigma = 4 J S.S)."""
        return {"type": "xxz", "n_sites": self.L, "J": float(self.J) / 4,
                "delta": float(self.delta)}


@dataclass(frozen=True)
class J1J2Chain(_SpinChain):
    """Frustrated spin-1/2 chain, H = sum_i (J1 h_{i,i+1} + J2 h_{i,i+2}),
    with h_{ij} = Sx_i Sx_j + Sy_i Sy_j + delta Sz_i Sz_j.

    ``J2 = J1/2`` is the Majumdar-Ghosh point, where an even open chain's ground state
    is an exact dimer product with E = -3 L J1 / 8.
    """
    L: int
    J1: float = 1.0
    J2: float = 0.0
    delta: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "L", _check_length(self.L))

    def problem(self) -> dict:
        """The C API problem (Pauli convention, so both couplings are sent over 4)."""
        return {"type": "j1j2", "n_sites": self.L, "J1": float(self.J1) / 4,
                "J2": float(self.J2) / 4, "delta": float(self.delta)}


@dataclass(frozen=True)
class Spin1Chain(_SpinChain):
    """Spin-1 Heisenberg chain, H = J sum_i S_i . S_{i+1}: the Haldane chain for J > 0."""
    L: int
    J: float = 1.0
    two_s = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "L", _check_length(self.L))

    def problem(self) -> dict:
        """The C API problem (the spin-1 builder already uses spin-1 operators)."""
        return {"type": "spin1_heisenberg", "n_sites": self.L, "J": float(self.J)}


class _Hubbard:
    """Sector handling shared by the Hubbard models: electron numbers per spin."""

    L: int

    def sector(self, n_up: int | None = None, n_dn: int | None = None) -> dict:
        """C API sector; default half filling with n_up = ceil(L/2), n_dn = floor(L/2)."""
        n_up = (self.L + 1) // 2 if n_up is None else n_up
        n_dn = self.L // 2 if n_dn is None else n_dn
        n_up, n_dn = _as_int(n_up, "n_up"), _as_int(n_dn, "n_dn")
        for name, v in (("n_up", n_up), ("n_dn", n_dn)):
            if not 0 <= v <= self.L:
                raise ModelError(f"{name} must be in [0, L={self.L}], got {v}")
        return {"n_up": n_up, "n_dn": n_dn}

    @staticmethod
    def describe_sector(sector: dict) -> dict:
        return {"n_up": sector["n_up"], "n_dn": sector["n_dn"]}


@dataclass(frozen=True)
class HubbardChain(_Hubbard):
    """Fermi-Hubbard chain,
    H = -t sum_{i,s} (c+_{i s} c_{i+1 s} + h.c.) + U sum_i n_{i up} n_{i dn} - mu sum_i n_i.

    Both the up and the down electron numbers are conserved, and the engine blocks on the
    pair (N, 2Sz), which gives the finer, faster blocks.
    """
    L: int
    t: float = 1.0
    U: float = 0.0
    mu: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "L", _check_length(self.L))

    def problem(self) -> dict:
        """The C API problem."""
        return {"type": "hubbard", "n_sites": self.L, "t": float(self.t), "U": float(self.U),
                "mu": float(self.mu)}


@dataclass(frozen=True)
class HubbardLattice(_Hubbard):
    """The Hubbard Hamiltonian of :class:`HubbardChain` on a width-W lattice of Lx columns,
    open along x: W = 1 is the chain, W = 2 the two-leg ladder, and W >= 3 a strip, or a
    cylinder with ``y_periodic``.

    Sites are numbered column by column, ``site = W*x + y`` (see :meth:`site`), and the
    engine runs them as one chain, so a hop along x has range W. Hopping is isotropic
    (``t`` on every bond).
    """
    Lx: int
    W: int = 1
    t: float = 1.0
    U: float = 0.0
    mu: float = 0.0
    y_periodic: bool = False

    def __post_init__(self) -> None:
        for name in ("Lx", "W"):
            v = _as_int(getattr(self, name), name)
            if v < 1:
                raise ModelError(f"{name} must be >= 1, got {v}")
            object.__setattr__(self, name, v)
        if self.y_periodic and self.W < 3:
            raise ModelError("y_periodic needs W >= 3: at W = 2 the wrap would double every "
                             "rung bond")
        _check_length(self.W * self.Lx, "W*Lx")

    @property
    def L(self) -> int:
        """Number of sites, W*Lx."""
        return self.W * self.Lx

    def site(self, x: int, y: int) -> int:
        """Chain index of the site in column x, row y."""
        return self.W * x + y

    def problem(self) -> dict:
        """The C API problem."""
        return {"type": "hubbard_lattice", "Lx": self.Lx, "W": self.W,
                "y_periodic": bool(self.y_periodic), "t": float(self.t), "U": float(self.U),
                "mu": float(self.mu)}
