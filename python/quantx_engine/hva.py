"""Fermionic HVA circuits on the native block-sparse engine.

The Hamiltonian Variational Ansatz is the standard VQE circuit for the Hubbard model.
Each layer applies ``exp(-i theta_g H_g)`` for every bond group ``g``, then
``exp(-i theta_U H_onsite)``. A group's bonds share no site, so its exponential factorises
into two-site gates with no Trotter error. The engine conserves both electron numbers, so
it simulates the circuit exactly, up to ``max_bond``, far beyond state-vector sizes::

    from quantx_engine.models import HubbardLattice
    from quantx_engine import hva

    cyl = HubbardLattice(Lx=4, W=3, U=4.0, y_periodic=True)
    lay = hva.layout(cyl)                          # the bond groups of this geometry
    angles = [[0.1] * lay.angles_per_layer] * 3    # three layers
    r = hva.circuit(cyl, angles, n_up=6, n_dn=6, max_bond=256)    # ~5-10 s
    r.energy, r.entropy_max, r.truncation_error, r.init_energy_error

This cylinder already truncates at max_bond 256: ``truncation_error`` is about 5e-4,
and the energy moves by 3e-3 at max_bond 1024. Raise max_bond until it stops moving.

Wider lattices cost fast. The U = 0 start itself is highly entangled in 2-D: a 6x4
cylinder prepared at max_bond 256 starts 0.05 off in energy (``init_energy_error``) and
takes minutes.

A :class:`~quantx_engine.models.HubbardChain` runs as the W = 1 lattice. Its two bond
groups are the even and the odd bonds, so each chain layer takes three angles.

Where the circuit starts: ``init="free"`` (the default) is the U = 0 ground state of the
sector. Every HVA generator commutes with S^2, so a product state can never reach a
singlet ground state. The free start is refused where the U = 0 ground state is
degenerate, since DMRG would then return an arbitrary member of the degenerate space.
``init="product"`` starts from a product state in the sector.

Precision: from a product start the circuit is exact up to ``max_bond`` and rounding
(gated against a state vector to 1e-10). The rounding depends on how the splits are
factorised. By default (``svd="gram"``) they go through the Gram matrix, the faster
route, which leaves Schmidt values below sqrt(eps) ~ 1e-8 of the largest imprecise. A
circuit whose bonds fill up then carries a rounding noise of up to ~1e-10 in its
energy: 6e-12 on Windows and 1.2e-10 on Linux for one 6-site chain of three layers.
``svd="direct"`` uses a direct SVD at every split, which keeps the energy smooth in the
angles to ~1e-14, at up to twice the cost. A free start is only as good as its DMRG
preparation, which
stops when its energy stops changing, and a circuit inherits the start's error. On the
small systems of the gate that meant 1e-8 to 7e-8 in the circuit energy; on a long chain
with a small single-particle gap it can be far worse. ``init_energy_error`` (the start's
energy against the exact U = 0 energy) catches a bad start, but a small value does not
bound the circuit's error: the energy is stationary at the ground state, so the gate's
starts were right to about 5e-14 in energy while their circuits were 1e-8 off. For the
full precision, start from ``init="product"``.

The gradient: ``gradient=True`` also returns dE/dtheta for every angle (``r.gradient``,
laid out as the angles), by the adjoint method. One pass back through the circuit
measures each step's generator between the state and H|psi>, both carried back. Like
the circuit, it costs more with depth, but it needs no circuit per angle, where central
differences need two. Its cost in energy evaluations is set by bond dimensions:
H|psi> needs up to w times the state's (w is the Hamiltonian's MPO bond dimension, 6
on a chain), so it is largest while the state's bonds are well below max_bond. On a
16-site chain at max_bond 128 the gradient costs 32 energy evaluations at 2 layers
(6 angles), against 12 for central differences, and 6 at 32 layers (96 angles),
against 192 (benchmarks/hva_gradient_cost.py). An optimizer can take it directly::

    import numpy as np
    from scipy.optimize import minimize

    def energy_and_gradient(x):
        r = hva.circuit(cyl, x.reshape(3, -1).tolist(), n_up=6, n_dn=6, gradient=True)
        return r.energy, np.ravel(r.gradient)

    res = minimize(energy_and_gradient, np.full(3 * lay.angles_per_layer, 0.1),
                   jac=True, method="L-BFGS-B")

Finite differences of the energy are a poor substitute: with the default ``svd``, a
difference step of 1e-8 turns the energy's rounding noise of ~1e-11 into an error of
1e-3 in the gradient.

What truncation costs the gradient: ``gradient_truncation_error`` is the largest weight
discarded computing it (H|psi> is held at ``max_bond``, as the state is), and
``gradient_return_error`` how far the backward pass missed the start. Both are zero
without truncation.
"""
from __future__ import annotations

import json
import operator
from ctypes import c_char_p
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = ["HVAError", "HVALayout", "HVAResult", "layout", "circuit"]


class HVAError(RuntimeError):
    """Raised when an HVA circuit cannot be run."""


@dataclass(frozen=True)
class HVALayout:
    """The bond groups of a lattice circuit, in the order each layer applies them.

    Sites are chain indices, ``site = W*x + y``.
    """
    n_sites: int
    groups: tuple

    @property
    def angles_per_layer(self) -> int:
        """One angle per bond group, then one for the on-site term."""
        return len(self.groups) + 1


@dataclass
class HVAResult:
    """Outcome of one circuit run.

    Attributes
    ----------
    energy : float
        ``<psi(theta)| H |psi(theta)>`` for the model's full Hamiltonian.
    entropy_max, entropy_mid : float or None
        Entanglement entropy (natural log) of the final state: the largest over all
        cuts of the site chain, and the balanced cut's (None for an odd number of sites).
        Both None when run with ``entropy=False``. They are the entropies of the
        ``max_bond``-capped state, so trust them only while ``truncation_error`` is small.
    truncation_error : float
        The largest single discarded weight over the circuit's truncations. Their sum
        can be larger by up to the number of truncations.
    max_bond : int
        Largest bond dimension the circuit reached (never above the ``max_bond`` cap).
    layer_bonds : list[int]
        Largest bond dimension after each layer.
    init_energy, init_energy_exact, init_energy_error : float or None
        ``init="free"`` only: the prepared state's energy, the exact U = 0 ground-state
        energy of the sector (from the single-particle levels), and their difference,
        how far the DMRG preparation stopped short. Check it; see the module notes.
    init_truncation_error : float or None
        ``init="free"`` only: the preparation's discarded weight, summed over its last
        sweep (unlike ``truncation_error``, a maximum). Not included in
        ``truncation_error``.
    init_config : list[int] or None
        ``init="product"`` only: the starting local state per site (0 empty, 1 up,
        2 down, 3 double occupancy).
    gradient : list[list[float]] or None
        ``gradient=True`` only: dE/dtheta for every angle, laid out as the angles.
    gradient_truncation_error : float or None
        ``gradient=True`` only: the largest single weight discarded computing it, in
        H|psi>'s compression to ``max_bond`` and in the backward pass's truncations, the
        gradient's counterpart of ``truncation_error``.
    gradient_return_error : float or None
        ``gradient=True`` only: 1 - |<psi_0|psi>| for the state the backward pass brings
        back to the start. It is zero without truncation; otherwise it is what the round
        trip's truncations cost the gradient.
    """
    energy: float
    entropy_max: Optional[float]
    entropy_mid: Optional[float]
    truncation_error: float
    max_bond: int
    layer_bonds: list
    wall_time_s: float
    sector: dict
    init: str
    init_energy: Optional[float] = None
    init_energy_exact: Optional[float] = None
    init_energy_error: Optional[float] = None
    init_truncation_error: Optional[float] = None
    init_config: Optional[list] = None
    gradient: Optional[list] = None
    gradient_truncation_error: Optional[float] = None
    gradient_return_error: Optional[float] = None


def _as_lattice(model):
    from quantx_engine.models import HubbardChain, HubbardLattice
    if isinstance(model, HubbardLattice):
        return model
    if isinstance(model, HubbardChain):
        return HubbardLattice(Lx=model.L, W=1, t=model.t, U=model.U, mu=model.mu)
    raise HVAError(f"HVA circuits need a HubbardChain or HubbardLattice, got {model!r}")


def _lib(engine):
    from quantx_engine.dmrg import _resolve_engine
    lib = getattr(_resolve_engine(engine, "hva", HVAError), "_lib", None)
    if lib is None or not hasattr(lib, "qx_hva_circuit"):
        raise HVAError("the loaded native library does not export qx_hva_circuit")
    lib.qx_hva_circuit.argtypes = [c_char_p, c_char_p]
    lib.qx_hva_circuit.restype = c_char_p
    lib.qx_hva_layout.argtypes = [c_char_p]
    lib.qx_hva_layout.restype = c_char_p
    return lib


def _check(out: dict) -> dict:
    if not out.get("ok", False):
        raise HVAError(out.get("error", "unknown error from the native engine"))
    return out


def layout(model, *, engine=None) -> HVALayout:
    """The bond groups of ``model``'s circuit; size each layer's angles from it."""
    lat = _as_lattice(model)
    out = _check(json.loads(_lib(engine).qx_hva_layout(json.dumps(lat.problem()).encode())))
    groups = tuple(tuple((int(i), int(j)) for i, j in g) for g in out["groups"])
    return HVALayout(n_sites=int(out["n_sites"]), groups=groups)


def circuit(
    model,
    angles: Sequence[Sequence[float]],
    *,
    n_up: Optional[int] = None,
    n_dn: Optional[int] = None,
    init: str = "free",
    max_bond: int = 256,
    cutoff: float = 1e-12,
    entropy: bool = True,
    gradient: bool = False,
    svd: str = "gram",
    engine=None,
) -> HVAResult:
    """Run an HVA circuit and measure the model's energy on its output.

    Parameters
    ----------
    model : HubbardChain or HubbardLattice
    angles : sequence of layers
        Each layer lists one angle per bond group (see :func:`layout`), then the
        on-site angle. A chain layer is ``[theta_even, theta_odd, theta_U]``.
    n_up, n_dn : int, optional
        The sector; default half filling, n_up = ceil(L/2), n_dn = floor(L/2).
    init : {"free", "product"}
        Start from the U = 0 ground state (refused where it is degenerate), or from a
        product state in the sector.
    max_bond : int
        Bond-dimension cap of the circuit's MPS. It also caps the free-state
        preparation, at no more than the 4^(L/2) an L-site chain can use.
    cutoff : float
        Truncation cutoff of the circuit (the preparation uses its own 1e-12).
    entropy : bool
        Measure the final state's entanglement entropies (one extra lossless sweep).
    gradient : bool
        Also compute dE/dtheta for every angle, by the adjoint method: one pass back
        through the circuit, carrying the state and the Hamiltonian applied to it.
        It needs no circuit per angle, where central differences need two (see the
        module notes for its cost).
    svd : {"gram", "direct"}
        How every split is factorised: through the Gram matrix (the default, faster),
        or by a direct SVD, which keeps the energy smooth in the angles to ~1e-14
        instead of ~1e-10, at up to twice the cost (see the module notes).

    Raises
    ------
    ModelError
        The sector is not valid for the model.
    HVAError
        No native library, a malformed layer, a degenerate free start, or a solver
        failure.
    """
    lat = _as_lattice(model)
    sector = lat.sector(n_up, n_dn)
    if init not in ("free", "product"):
        raise HVAError(f"init must be 'free' or 'product', got {init!r}")
    if svd not in ("gram", "direct"):
        raise HVAError(f"svd must be 'gram' or 'direct', got {svd!r}")
    try:
        if isinstance(max_bond, bool):
            raise TypeError
        max_bond = operator.index(max_bond)
    except TypeError:
        raise HVAError(f"max_bond must be an int, got {max_bond!r}") from None
    if max_bond < 1:
        raise HVAError(f"max_bond must be >= 1, got {max_bond}")
    try:
        layers = [[float(a) for a in layer] for layer in angles]
    except TypeError as exc:
        raise HVAError("angles must be a sequence of layers, each a sequence of numbers") from exc
    prob = dict(lat.problem(), sector=sector, angles=layers)
    opts = {"max_bond": max_bond, "cutoff": cutoff, "init": init, "entropy": bool(entropy),
            "gradient": bool(gradient), "svd": svd}
    out = _check(json.loads(_lib(engine).qx_hva_circuit(json.dumps(prob).encode(),
                                                        json.dumps(opts).encode())))
    return HVAResult(
        energy=float(out["energy"]),
        entropy_max=out.get("entropy_max"),
        entropy_mid=out.get("entropy_mid"),
        truncation_error=float(out["truncation_error"]),
        max_bond=int(out["max_bond"]),
        layer_bonds=[int(b) for b in out.get("layer_bonds", [])],
        wall_time_s=float(out.get("wall_time_s", 0.0)),
        sector=lat.describe_sector(out["sector"]),
        init=str(out["init"]),
        init_energy=out.get("init_energy"),
        init_energy_exact=out.get("init_energy_exact"),
        init_energy_error=out.get("init_energy_error"),
        init_truncation_error=out.get("init_truncation_error"),
        init_config=out.get("init_config"),
        gradient=out.get("gradient"),
        gradient_truncation_error=out.get("gradient_truncation_error"),
        gradient_return_error=out.get("gradient_return_error"),
    )
