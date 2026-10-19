"""M11 gate: matrix-product states that stay in the engine, against exact diagonalisation.

Each check chains operations on states the engine keeps (a DMRG ground state, one-site
operators, TDVP in real and imaginary time, overlaps and matrix elements) and compares the
result with an exact diagonalisation written here, independently of the engine. The chains
are small enough for the bond dimension to hold the whole sector, so the ground states are
exact to DMRG's convergence: 1e-14 in the energy, and about 1e-8 in observables, which a
state's error enters linearly rather than quadratically. Two-site TDVP at full bond
dimension adds no step error: the dynamics agree with the exact evolution to the same 1e-9
at dt = 0.04, 0.02 and 0.01. So the gate is 1e-10 for a ground state's own energy, and
for energy conservation, and 1e-7 for the rest.
"""
from __future__ import annotations

import gc
import itertools
import os

import numpy as np
import pytest

from quantx_engine import mps
from quantx_engine.models import HubbardChain, XXZChain

TOL = 1e-7          # observables: linear in the state's error
TOL_E = 1e-10       # a ground state's own energy: quadratic in it
EXACT = dict(max_bond=128, cutoff=1e-14, n_sweeps=40, tol=1e-13)


@pytest.fixture(scope="module")
def engine():
    unavailable = pytest.fail if os.environ.get("QX_REQUIRE_NATIVE") == "1" else pytest.skip
    try:
        from quantx_engine.engine import Engine
        eng = Engine()
    except Exception as exc:  # QuantXNativeUnavailable, or a missing dependency
        unavailable(f"native library unavailable: {exc}")
    if not hasattr(eng._lib, "qx_mps"):
        unavailable("native library has no qx_mps")
    return eng


# ── exact diagonalisation, independent of the engine ──────────────────────────────────

def xxz_ed(L, J, delta, sz2=0):
    """H = J sum (Sx Sx + Sy Sy + delta Sz Sz), S = sigma/2, at 2 Sz_tot = sz2.

    Returns H on the sector's product basis (local state 0 = up, 1 = down, site 0 first)
    and each basis state's S^z on every site, shape (dim, L)."""
    basis = [c for c in itertools.product((0, 1), repeat=L) if sum(1 - 2 * b for b in c) == sz2]
    index = {c: n for n, c in enumerate(basis)}
    H = np.zeros((len(basis),) * 2)
    for n, c in enumerate(basis):
        for i in range(L - 1):
            H[n, n] += J * delta * (0.5 - c[i]) * (0.5 - c[i + 1])
            if c[i] != c[i + 1]:
                d = list(c)
                d[i], d[i + 1] = c[i + 1], c[i]
                H[index[tuple(d)], n] += 0.5 * J
    return H, np.array([[0.5 - b for b in c] for c in basis])


def hubbard_ed(L, t, U, n_up, n_dn):
    """The open Hubbard chain at (n_up, n_dn), modes ordered spin-block-wise (all up, then
    all down), where nearest-neighbour hopping carries no Jordan-Wigner sign.

    Returns H and each basis state's (n_up, n_dn) on every site, shape (dim, L) each."""
    def states(k):
        return [sum(1 << i for i in occ) for occ in itertools.combinations(range(L), k)]
    ups, dns = states(n_up), states(n_dn)
    up_ix, dn_ix = {s: a for a, s in enumerate(ups)}, {s: b for b, s in enumerate(dns)}
    H = np.zeros((len(ups) * len(dns),) * 2)
    nu, nd = [], []
    for a, su in enumerate(ups):
        for b, sd in enumerate(dns):
            n = a * len(dns) + b
            H[n, n] += U * bin(su & sd).count("1")
            for i in range(L - 1):
                pair = (1 << i) | (1 << (i + 1))
                if bin(su & pair).count("1") == 1:
                    H[up_ix[su ^ pair] * len(dns) + b, n] += -t
                if bin(sd & pair).count("1") == 1:
                    H[a * len(dns) + dn_ix[sd ^ pair], n] += -t
            nu.append([(su >> i) & 1 for i in range(L)])
            nd.append([(sd >> i) & 1 for i in range(L)])
    return H, np.array(nu, dtype=float), np.array(nd, dtype=float)


def ground(H):
    E, V = np.linalg.eigh(H)
    return E, V


# ── the gate ──────────────────────────────────────────────────────────────────────────

def test_ground_state_and_energy_match_ed(engine):
    H, sz = xxz_ed(10, 1.0, 0.5)
    E0 = ground(H)[0][0]
    gs = mps.ground_state(XXZChain(L=10, delta=0.5), engine=engine, **EXACT)
    assert abs(gs.dmrg.energy - E0) < TOL_E and abs(gs.energy() - E0) < TOL_E
    assert abs(gs.norm - 1) < 1e-12 and np.abs(gs.expect("Sz")).max() < TOL
    assert gs.info()["charge"] == 0 and len(gs.bonds) == 9


def test_static_correlations_match_ed(engine):
    L, c = 10, 4
    H, sz = xxz_ed(L, 1.0, 0.5)
    E, V = ground(H)
    g = V[:, 0]
    gs = mps.ground_state(XXZChain(L=L, delta=0.5), engine=engine, **EXACT)
    phi = gs.copy().apply("Sz", c)                      # S^z_c |GS>
    want = (g ** 2) @ (sz * sz[:, [c]])                 # <S^z_j S^z_c> for every j
    assert np.allclose(gs.matrix_elements("Sz", phi), want, atol=TOL, rtol=0)
    assert abs(phi.overlap(phi) - want[c]) < TOL        # |S^z_c GS|^2 = <(S^z_c)^2> = 1/4


@pytest.mark.parametrize("delta", [0.5, 2.0])
def test_dynamical_correlation_matches_the_exact_evolution(engine, delta):
    """<GS| e^{iHt} S^z_j e^{-iHt} S^z_c |GS>: both states evolved, then a matrix element."""
    L, c, dt, steps = 8, 3, 0.05, 16
    H, sz = xxz_ed(L, 1.0, delta)
    E, V = ground(H)
    g, t = V[:, 0], dt * steps
    chain = XXZChain(L=L, delta=delta)
    gs = mps.ground_state(chain, engine=engine, **EXACT)
    phi = gs.copy().apply("Sz", c)
    gs.evolve(dt=dt, steps=steps, max_bond=64, cutoff=1e-14)
    phi.evolve(dt=dt, steps=steps, max_bond=64, cutoff=1e-14)
    psi_t = V @ (np.exp(-1j * E * t) * (V.T @ (sz[:, c] * g)))     # e^{-iHt} S^z_c |GS>
    want = np.exp(1j * E[0] * t) * (g * psi_t) @ sz
    assert np.allclose(gs.matrix_elements("Sz", phi), want, atol=TOL, rtol=0)
    assert abs(gs.norm - 1) < TOL and abs(phi.norm - 0.5) < TOL    # real time keeps the norm


def test_a_non_unitary_operator_leaves_a_state_to_evolve(engine):
    """D = diag(1, 0.3) on one site is no unitary times a number, so applied anywhere but
    at the centre it would leave a site that is no isometry. The energy, the norm and the
    evolution of D|GS> must all still be exact."""
    L, c, dt, steps = 8, 5, 0.05, 12
    H, sz = xxz_ed(L, 1.0, 0.5)
    E, V = ground(H)
    g = V[:, 0]
    d = np.where(sz[:, c] > 0, 1.0, 0.3)                 # D_c on the basis
    v = d * g                                             # D_c |GS>
    chain = XXZChain(L=L, delta=0.5)
    gs = mps.ground_state(chain, engine=engine, **EXACT)
    phi = gs.copy().apply([1.0, 0.3], c)
    assert abs(phi.norm ** 2 - v @ v) < TOL
    assert abs(phi.energy() - (v @ H @ v) / (v @ v)) < TOL     # no eigenstate: linear
    assert np.allclose(phi.expect("Sz"), (v ** 2) @ sz / (v @ v), atol=TOL, rtol=0)
    phi.evolve(dt=dt, steps=steps, max_bond=64, cutoff=1e-14)
    gs.evolve(dt=dt, steps=steps, max_bond=64, cutoff=1e-14)
    t = dt * steps
    v_t = V @ (np.exp(-1j * E * t) * (V.T @ v))
    want = np.exp(1j * E[0] * t) * (g * v_t) @ sz
    assert np.allclose(gs.matrix_elements("Sz", phi), want, atol=TOL, rtol=0)


def test_a_quench_conserves_the_new_energy(engine):
    """The delta = 0.5 ground state evolved under delta = 2: <H_2> stays put, and the
    Loschmidt amplitude <psi0|psi(t)> follows the exact one."""
    L, dt, steps = 8, 0.05, 20
    H2, _ = xxz_ed(L, 1.0, 2.0)
    E2, V2 = ground(H2)
    g = ground(xxz_ed(L, 1.0, 0.5)[0])[1][:, 0]
    psi0 = mps.ground_state(XXZChain(L=L, delta=0.5), engine=engine, **EXACT)
    psi = psi0.copy()
    after = XXZChain(L=L, delta=2.0)
    e_start = psi.energy(after)
    psi.evolve(after, dt=dt, steps=steps, max_bond=64, cutoff=1e-14)
    assert abs(e_start - g @ H2 @ g) < TOL                       # no eigenstate of H_2
    assert abs(psi.energy(after) - e_start) < TOL_E              # TDVP conserves it
    amp = g @ (V2 @ (np.exp(-1j * E2 * dt * steps) * (V2.T @ g)))
    assert abs(psi0.overlap(psi) - amp) < TOL


def test_imaginary_time_finds_the_ground_state(engine):
    L = 8
    E0 = ground(xxz_ed(L, 1.0, 0.5)[0])[0][0]
    chain = XXZChain(L=L, delta=0.5)
    neel = mps.product(chain, ["up", "dn"] * (L // 2), engine=engine)
    assert abs(neel.energy() - (L - 1) * 0.5 * -0.25) < 1e-12      # delta <Sz Sz> = -1/4 J delta
    neel.evolve(dt=0.1, steps=200, imaginary=True, max_bond=64)
    assert abs(neel.norm - 1) < 1e-12 and 0 <= neel.energy() - E0 < 1e-6


def test_hubbard_densities_and_double_occupancy_match_ed(engine):
    L, U, c = 6, 4.0, 2
    H, nu, nd = hubbard_ed(L, 1.0, U, 3, 3)
    g = ground(H)[1][:, 0]
    model = HubbardChain(L=L, U=U)
    gs = mps.ground_state(model, n_up=3, n_dn=3, engine=engine, **EXACT)
    n = nu + nd
    assert np.allclose(gs.expect("D"), (g ** 2) @ (nu * nd), atol=TOL, rtol=0)
    assert np.allclose(gs.expect("N"), (g ** 2) @ n, atol=TOL, rtol=0)
    assert abs(gs.expect("Sz").sum()) < TOL
    phi = gs.copy().apply("N", c)                       # n_c |GS>
    assert np.allclose(gs.matrix_elements("N", phi), (g ** 2) @ (n * n[:, [c]]), atol=TOL, rtol=0)


def test_operators_by_name_and_by_diagonal_agree(engine):
    gs = mps.ground_state(XXZChain(L=6), engine=engine, **EXACT)
    assert np.allclose(gs.expect("Sz"), gs.expect([0.5, -0.5]), atol=1e-14)
    a, b = gs.copy().apply("Sz", 2), gs.copy().apply([0.5, -0.5], 2)
    assert abs(a.overlap(b) - 0.25) < TOL


def test_states_are_freed(engine):
    lib = mps._lib(engine)
    gc.collect()                                        # earlier tests' states first
    live = mps._call(lib, {"op": "live"})["handles"]
    gs = mps.ground_state(XXZChain(L=6), engine=engine, max_bond=16)
    copies = [gs.copy() for _ in range(3)]
    assert mps._call(lib, {"op": "live"})["handles"] == live + 4
    del copies
    gc.collect()
    assert mps._call(lib, {"op": "live"})["handles"] == live + 1
    gs.free()
    gs.free()                                           # a second free does nothing
    assert mps._call(lib, {"op": "live"})["handles"] == live
    with pytest.raises(mps.MPSError, match="freed"):
        gs.energy()
    with mps.product(XXZChain(L=4), [0, 1, 0, 1], engine=engine):
        assert mps._call(lib, {"op": "live"})["handles"] == live + 1
    assert mps._call(lib, {"op": "live"})["handles"] == live


def test_bad_requests_are_refused(engine):
    chain = XXZChain(L=6)
    gs = mps.ground_state(chain, engine=engine, max_bond=16)
    with pytest.raises(mps.MPSError, match="no operator 'N'"):
        gs.apply("N", 0)
    with pytest.raises(mps.MPSError, match="2 entries"):
        gs.apply([1, 2, 3], 0)
    with pytest.raises(mps.MPSError, match="site 6 is not in"):
        gs.apply("Sz", 6)
    with pytest.raises(mps.MPSError, match="annihilates"):
        mps.product(chain, ["up"] * 6, engine=engine).apply([0, 1], 0)
    with pytest.raises(mps.MPSError, match="8-site"):
        gs.evolve(XXZChain(L=8), dt=0.1)
    with pytest.raises(mps.MPSError, match="hubbard"):
        gs.energy(HubbardChain(L=6))
    with pytest.raises(mps.MPSError, match="dt must be > 0"):
        gs.evolve(dt=0.0)
    with pytest.raises(mps.MPSError, match="no spin-1/2 local state"):
        mps.product(chain, ["up", "ud"] * 3, engine=engine)
    with pytest.raises(mps.MPSError, match="sector is sz"):
        mps.ground_state(chain, n_up=3, engine=engine)
    lib = mps._lib(engine)
    with pytest.raises(mps.MPSError, match="no live state"):
        mps._call(lib, {"op": "info", "handle": 10 ** 12})
    with pytest.raises(mps.MPSError, match="unknown key"):
        mps._call(lib, {"op": "info", "handle": gs._h(), "extra": 1})
    with pytest.raises(mps.MPSError, match="unknown op"):
        mps._call(lib, {"op": "bogus"})
