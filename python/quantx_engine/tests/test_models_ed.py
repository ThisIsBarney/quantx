"""M4 ED gate: the native engine's symmetric models against exact diagonalisation.

Every model in quantx_engine.models is solved by DMRG on the block-sparse engine and
compared with an exact diagonalisation built here, independently of the engine, in
the same symmetry sector. The chains are small enough that the kept bond dimension
holds the whole sector, so DMRG is exact up to convergence: the gate is 1e-10.

The Hubbard ED orders the fermion modes spin-block-wise (all up modes, then all down),
where open-chain nearest-neighbour hopping carries no Jordan-Wigner sign at all. The
engine's MPO orders them site by site and carries the parity strings explicitly. What
this does NOT test: on an open chain with nearest-neighbour hopping those signs are a
pure gauge, so an MPO with the strings consistently dropped has the same spectrum and
would pass. Only an inconsistent slip fails here. The fermionic sign conventions need
geometries where the signs are physical, ladders and cylinders.
"""
from __future__ import annotations

import itertools
import math
import os

import numpy as np
import pytest

from quantx_engine.models import HubbardChain, J1J2Chain, ModelError, Spin1Chain, XXZChain

TOL = 1e-10
EXACT = dict(max_bond=256, cutoff=1e-14, n_sweeps=40, tol=1e-13)


def _native_or_skip():
    # Where the native library is supposed to be present (CI's native job sets
    # QX_REQUIRE_NATIVE=1), a load failure must fail the gate, not quietly skip it.
    unavailable = pytest.fail if os.environ.get("QX_REQUIRE_NATIVE") == "1" else pytest.skip
    try:
        from quantx_engine.engine import Engine
        engine = Engine()
    except Exception as exc:  # QuantXNativeUnavailable, or a missing dependency
        unavailable(f"native library unavailable: {exc}")
    if not hasattr(engine._lib, "qx_dmrg_sweep"):
        unavailable("native library has no qx_dmrg_sweep")
    return engine


@pytest.fixture(scope="module")
def engine():
    return _native_or_skip()


def _gs(model, engine, **sector):
    from quantx_engine.dmrg import ground_state
    return ground_state(model, engine=engine, **EXACT, **sector)


# ── Exact diagonalisation, independent of the engine ─────────────────────────

def _spin_sector_ed(L, two_s, sz2, bonds):
    """Lowest energy of sum_{(i,j,J,delta)} J (Sx Sx + Sy Sy + delta Sz Sz) at 2Sz_tot = sz2.

    Local states are m = S, S-1, ..., -S; the basis is every product state in the sector.
    """
    S = two_s / 2
    ms = [S - k for k in range(two_s + 1)]
    basis = [c for c in itertools.product(range(two_s + 1), repeat=L)
             if round(2 * sum(ms[k] for k in c)) == sz2]
    index = {c: n for n, c in enumerate(basis)}
    H = np.zeros((len(basis), len(basis)))

    def raise_amp(m):   # <m+1| S+ |m>
        return math.sqrt(S * (S + 1) - m * (m + 1))

    for n, c in enumerate(basis):
        for i, j, J, delta in bonds:
            mi, mj = ms[c[i]], ms[c[j]]
            H[n, n] += J * delta * mi * mj
            # (J/2)(S+_i S-_j + S-_i S+_j): level index k -> k-1 raises m.
            for ki, kj, amp in ((c[i] - 1, c[j] + 1, raise_amp(mi) * raise_amp(mj - 1)),
                                (c[i] + 1, c[j] - 1, raise_amp(mi - 1) * raise_amp(mj))):
                if 0 <= ki <= two_s and 0 <= kj <= two_s:
                    d = list(c)
                    d[i], d[j] = ki, kj
                    H[index[tuple(d)], n] += 0.5 * J * amp
    return float(np.linalg.eigvalsh(H)[0])


def _hubbard_sector_ed(L, t, U, mu, n_up, n_dn):
    """Lowest energy of the open Hubbard chain with n_up up and n_dn down electrons.

    Spin-block mode ordering: within one species, nearest-neighbour modes are adjacent
    and the other species sits wholly outside the pair, so hopping has no JW sign.
    """
    def states(k):
        return [sum(1 << i for i in occ) for occ in itertools.combinations(range(L), k)]

    ups, dns = states(n_up), states(n_dn)
    up_ix = {s: a for a, s in enumerate(ups)}
    dn_ix = {s: b for b, s in enumerate(dns)}
    dim = len(ups) * len(dns)
    H = np.zeros((dim, dim))
    for a, su in enumerate(ups):
        for b, sd in enumerate(dns):
            n = a * len(dns) + b
            H[n, n] += U * bin(su & sd).count("1") - mu * (n_up + n_dn)
            for i in range(L - 1):
                pair = (1 << i) | (1 << (i + 1))
                if bin(su & pair).count("1") == 1:      # an up electron can hop i <-> i+1
                    H[up_ix[su ^ pair] * len(dns) + b, n] += -t
                if bin(sd & pair).count("1") == 1:
                    H[a * len(dns) + dn_ix[sd ^ pair], n] += -t
    return float(np.linalg.eigvalsh(H)[0])


def _nn(L, J, delta=1.0):
    return [(i, i + 1, J, delta) for i in range(L - 1)]


def _nnn(L, J, delta=1.0):
    return [(i, i + 2, J, delta) for i in range(L - 2)]


# ── The ED gate (native engine) ───────────────────────────────────────────────

@pytest.mark.parametrize("L, J, delta, sz", [
    (10, 1.0, 0.5, 0.0),     # easy-plane
    (11, 1.0, 1.0, 0.5),     # odd chain: the C API used to fail on every odd L
    (10, 1.0, 2.0, 1.0),     # easy-axis, magnetised sector
    (9, -1.0, 1.0, 0.5),     # ferromagnet
])
def test_xxz_matches_ed(engine, L, J, delta, sz):
    ed = _spin_sector_ed(L, 1, round(2 * sz), _nn(L, J, delta))
    r = _gs(XXZChain(L=L, J=J, delta=delta), engine, sz=sz)
    assert r.sector == {"sz": sz}
    assert abs(r.energy - ed) < TOL, (r.energy, ed)


@pytest.mark.parametrize("L, J2, delta, sz", [
    (12, 0.3, 1.0, 0.0),
    (10, 0.4, 0.7, 1.0),
    (11, 0.6, 1.0, 0.5),
])
def test_j1j2_matches_ed(engine, L, J2, delta, sz):
    ed = _spin_sector_ed(L, 1, round(2 * sz), _nn(L, 1.0, delta) + _nnn(L, J2, delta))
    r = _gs(J1J2Chain(L=L, J1=1.0, J2=J2, delta=delta), engine, sz=sz)
    assert abs(r.energy - ed) < TOL, (r.energy, ed)


def test_j1j2_majumdar_ghosh_is_exact(engine):
    # At J2 = J1/2 an even open chain's ground state is a product of singlets.
    L = 16
    r = _gs(J1J2Chain(L=L, J1=1.0, J2=0.5), engine, sz=0)
    assert abs(r.energy - (-3 * L / 8)) < TOL, r.energy


@pytest.mark.parametrize("L, J, sz", [(8, 1.0, 0), (7, 1.0, 1), (6, -0.7, 2)])
def test_spin1_matches_ed(engine, L, J, sz):
    ed = _spin_sector_ed(L, 2, 2 * sz, _nn(L, J))
    r = _gs(Spin1Chain(L=L, J=J), engine, sz=sz)
    assert abs(r.energy - ed) < TOL, (r.energy, ed)


@pytest.mark.parametrize("L, t, U, mu, n_up, n_dn", [
    (6, 1.0, 4.0, 0.0, 3, 3),     # half filling
    (5, 1.0, 2.0, 0.0, 3, 2),     # odd chain
    (7, 1.0, 4.0, 0.3, 4, 3),
    (6, 0.8, 8.0, 0.0, 2, 2),     # doped, strong coupling
    (5, 1.0, 3.0, 0.0, 4, 3),     # above half filling: doubly occupied start sites
])
def test_hubbard_matches_ed(engine, L, t, U, mu, n_up, n_dn):
    ed = _hubbard_sector_ed(L, t, U, mu, n_up, n_dn)
    r = _gs(HubbardChain(L=L, t=t, U=U, mu=mu), engine, n_up=n_up, n_dn=n_dn)
    assert r.sector == {"n_up": n_up, "n_dn": n_dn}
    assert r.method == "u1xu1"
    assert abs(r.energy - ed) < TOL, (r.energy, ed)


def _free_fermion_energy(L, t, n_up, n_dn):
    # U = 0: fill the lowest single-particle levels -2t cos(k pi / (L+1)) per species.
    eps = sorted(-2 * t * math.cos(k * math.pi / (L + 1)) for k in range(1, L + 1))
    return sum(eps[:n_up]) + sum(eps[:n_dn])


def test_hubbard_free_fermions_are_exact(engine):
    # L = 8: max_bond 256 = 4^4 holds every state at the middle bond, so no truncation.
    L, n_up, n_dn = 8, 4, 3
    r = _gs(HubbardChain(L=L, U=0.0), engine, n_up=n_up, n_dn=n_dn)
    exact = _free_fermion_energy(L, 1.0, n_up, n_dn)
    assert abs(r.energy - exact) < TOL, (r.energy, exact)


def test_truncated_result_stays_variational(engine):
    # L = 12 needs up to 4^6 states at the middle bond; 256 kept means real truncation.
    # DMRG is variational, so the energy must stay above the exact one, and the
    # discarded weight must be reported rather than hidden.
    L, n_up, n_dn = 12, 5, 4
    r = _gs(HubbardChain(L=L, U=0.0), engine, n_up=n_up, n_dn=n_dn)
    exact = _free_fermion_energy(L, 1.0, n_up, n_dn)
    assert exact - 1e-12 <= r.energy < exact + 1e-6, (r.energy, exact)
    assert r.truncation_error > 0


def test_default_sectors(engine):
    assert _gs(XXZChain(L=7), engine).sector == {"sz": 0.5}
    assert _gs(Spin1Chain(L=5), engine).sector == {"sz": 0.0}
    assert _gs(HubbardChain(L=5, U=1.0), engine).sector == {"n_up": 3, "n_dn": 2}


def test_imbalanced_hubbard_sector_runs_at_moderate_bond(engine):
    # The start state used to alternate spins "while both remain", piling a spin
    # imbalance's excess at the right end: a path the engine's bond-space guess pruned, so
    # this sector failed below max_bond 91.
    from quantx_engine.dmrg import ground_state
    r = ground_state(HubbardChain(L=32, U=4.0), n_up=23, n_dn=9, max_bond=64,
                     n_sweeps=2, tol=1e-6, engine=engine)
    assert r.sector == {"n_up": 23, "n_dn": 9} and math.isfinite(r.energy)


# ── Validation (pure Python, no native library needed) ───────────────────────

def test_numpy_integers_are_accepted():
    # np.arange loops hand over numpy integers; they must work, and reach the engine's
    # JSON as plain numbers (json.dumps rejects numpy scalars).
    import json
    m = XXZChain(L=np.int64(8))
    assert type(m.L) is int and m.problem()["n_sites"] == 8
    assert HubbardChain(L=np.int32(6)).sector(np.int64(3), np.int64(2)) == {"n_up": 3, "n_dn": 2}
    json.dumps(dict(HubbardChain(L=np.int64(6), U=np.float32(4.0)).problem(),
                    sector=HubbardChain(L=6).sector(np.int64(3), np.int64(3))))


def test_spin_convention_is_s_not_pauli():
    # The engine's spin-1/2 builders take Pauli couplings: J sigma.sigma = 4 J S.S.
    assert XXZChain(L=4, J=2.0).problem()["J"] == 0.5
    p = J1J2Chain(L=4, J1=1.0, J2=0.5).problem()
    assert (p["J1"], p["J2"]) == (0.25, 0.125)
    assert Spin1Chain(L=4, J=2.0).problem()["J"] == 2.0


@pytest.mark.parametrize("bad", [2, 201, 3.0, True])
def test_chain_length_is_checked(bad):
    with pytest.raises(ModelError):
        XXZChain(L=bad)


@pytest.mark.parametrize("model, sz", [
    (XXZChain(L=8), 0.5),      # parity: an even spin-1/2 chain has integer Sz
    (XXZChain(L=8), 4.5),      # beyond full polarisation
    (Spin1Chain(L=5), 0.5),    # spin-1: integer Sz only
    (XXZChain(L=8), 0.25),     # not a multiple of 1/2
])
def test_unreachable_spin_sector_is_refused(model, sz):
    with pytest.raises(ModelError):
        model.sector(sz)


@pytest.mark.parametrize("n_up, n_dn", [(-1, 2), (5, 7), (2.0, 2)])
def test_unreachable_hubbard_sector_is_refused(n_up, n_dn):
    with pytest.raises(ModelError):
        HubbardChain(L=6).sector(n_up, n_dn)


def test_sector_arguments_must_fit_the_model():
    from quantx_engine.dmrg import DMRGError, ground_state
    with pytest.raises(DMRGError):
        ground_state(HubbardChain(L=6), sz=0.0)
    with pytest.raises(DMRGError):
        ground_state(XXZChain(L=6), n_up=3)
