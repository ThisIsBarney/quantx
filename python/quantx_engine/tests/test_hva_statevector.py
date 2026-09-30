"""M5 gate: HVA circuits on the native engine against an independent state vector.

Every circuit (chain, ladder, strip, cylinder; product and U = 0 starts) runs on the
block-sparse engine and on a Fock-space simulator written here, independently of the
engine. From product starts the two must agree to 1e-10 on the energy and on the
entanglement entropies. From U = 0 starts, the prepared energy must match the exact
U = 0 energy to 1e-10, and the circuit to 1e-6 (see FREE_TOL).

The simulator uses the canonical site-major mode order, mode = 2*site + spin (0 up,
1 down). A basis state is the ascending product of creation operators on the vacuum,
so the modes of the sites left of a cut come first, and a cut between sites is a plain
tensor-product cut: the amplitude matrix over (left modes, right modes) gives the
fermionic Schmidt values. The bond groups come from the engine (hva.layout), since
groups do not commute and the order is part of the circuit. The physics is independent:
each group's Hamiltonian is built here from explicit fermion signs, and the groups are
checked separately against the lattice's bonds.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pytest

from quantx_engine.models import HubbardChain, HubbardLattice, ModelError, XXZChain

TOL = 1e-10
# A free start is a DMRG ground state: its energy is exact to TOL (quadratic in the state
# error), but the state itself carries the local eigensolver's residual, which a circuit
# then propagates. Measured: 1e-8 to 7e-8 in the circuit energy on these systems. So the
# circuit is gated at TOL from product starts, and at FREE_TOL from free ones.
FREE_TOL = 1e-6


def _native_or_skip():
    unavailable = pytest.fail if os.environ.get("QX_REQUIRE_NATIVE") == "1" else pytest.skip
    try:
        from quantx_engine.engine import Engine
        engine = Engine()
    except Exception as exc:  # QuantXNativeUnavailable, or a missing dependency
        unavailable(f"native library unavailable: {exc}")
    if not hasattr(engine._lib, "qx_hva_circuit"):
        unavailable("native library has no qx_hva_circuit")
    return engine


@pytest.fixture(scope="module")
def engine():
    return _native_or_skip()


# ── Independent Fock-space simulator ──────────────────────────────────────────

class Fock:
    """The (n_up, n_dn) sector of n sites, in canonical site-major mode order."""

    def __init__(self, n, n_up, n_dn):
        self.n = n
        self.states = []
        for ups in itertools.combinations(range(n), n_up):
            for dns in itertools.combinations(range(n), n_dn):
                self.states.append(sum(1 << (2 * i) for i in ups) +
                                   sum(1 << (2 * i + 1) for i in dns))
        self.index = {s: k for k, s in enumerate(self.states)}

    def hop(self, bonds, t):
        """-t sum_{(i,j)} sum_spin (c+_i c_j + h.c.)."""
        H = np.zeros((len(self.states),) * 2)
        for k, s in enumerate(self.states):
            for i, j in bonds:
                for spin in (0, 1):
                    for a, b in ((2 * i + spin, 2 * j + spin), (2 * j + spin, 2 * i + spin)):
                        if not (s >> b) & 1 or (s >> a) & 1:
                            continue
                        # c_b passes the occupied modes below b; c+_a those below a.
                        s1 = s & ~(1 << b)
                        sign = (-1) ** (bin(s & ((1 << b) - 1)).count("1") +
                                        bin(s1 & ((1 << a) - 1)).count("1"))
                        H[self.index[s1 | (1 << a)], k] += -t * sign
        return H

    def onsite(self, U, mu):
        """Diagonal of U sum_i n_i,up n_i,dn - mu N."""
        return np.array([U * sum((s >> (2 * i)) & (s >> (2 * i + 1)) & 1 for i in range(self.n))
                         - mu * bin(s).count("1") for s in self.states], dtype=float)

    def product(self, config):
        """The basis state of a local configuration (0 empty, 1 up, 2 down, 3 double)."""
        s = sum((1 << (2 * i)) * (c in (1, 3)) + (1 << (2 * i + 1)) * (c in (2, 3))
                for i, c in enumerate(config))
        v = np.zeros(len(self.states), dtype=complex)
        v[self.index[s]] = 1.0
        return v

    def entropies(self, v):
        """Entanglement entropy (natural log) of every cut between sites."""
        out = []
        for k in range(1, self.n):
            rows, cols, entries = {}, {}, []
            for amp, s in zip(v, self.states):
                if abs(amp) > 1e-15:
                    left, right = s & ((1 << (2 * k)) - 1), s >> (2 * k)
                    entries.append((rows.setdefault(left, len(rows)),
                                    cols.setdefault(right, len(cols)), amp))
            M = np.zeros((len(rows), len(cols)), dtype=complex)
            for i, j, a in entries:
                M[i, j] += a
            p = np.linalg.svd(M, compute_uv=False) ** 2
            p = p[p > 1e-16]
            out.append(float(-(p * np.log(p)).sum()))
        return out


def _lattice_bonds(Lx, W, y_periodic):
    bonds = set()
    for x in range(Lx):
        for y in range(W - 1):
            bonds.add((W * x + y, W * x + y + 1))
        if y_periodic and W > 2:
            bonds.add((W * x, W * x + W - 1))
    for x in range(Lx - 1):
        for y in range(W):
            bonds.add((W * x + y, W * (x + 1) + y))
    return bonds


def _reference(lat, groups, angles, n_up, n_dn, init, config=None):
    """Energy and cut entropies of the circuit's output, by state vector."""
    fock = Fock(lat.L, n_up, n_dn)
    Hg = [fock.hop(g, lat.t) for g in groups]
    diag = fock.onsite(lat.U, lat.mu)
    if init == "product":
        v = fock.product(config)
    else:  # the U = 0 ground state of the sector (the engine refuses a degenerate one)
        w, V = np.linalg.eigh(sum(Hg))
        assert w[1] - w[0] > 1e-8, "reference: degenerate U = 0 ground state"
        v = V[:, 0].astype(complex)
    eig = [np.linalg.eigh(h) for h in Hg]
    for layer in angles:
        for (w, V), theta in zip(eig, layer[:-1]):
            v = V @ (np.exp(-1j * theta * w) * (V.conj().T @ v))
        v = np.exp(-1j * layer[-1] * diag) * v
    H = sum(Hg) + np.diag(diag)
    energy = float(np.real(v.conj() @ H @ v) / np.real(v.conj() @ v))
    return energy, fock.entropies(v / np.linalg.norm(v))


def _angles(n_layers, per_layer, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.6, 0.6, size=(n_layers, per_layer)).tolist()


# ── The state-vector gate (native engine) ─────────────────────────────────────

CASES = [
    # model, (n_up, n_dn), init, layers
    (HubbardLattice(Lx=6, W=1, U=4.0), (3, 3), "product", 3),                  # chain
    # Spin-imbalanced. With n_up = n_dn, start-state labels that disagreed with the
    # engine's charge convention would still land in the right sector; here they cannot.
    # (A consistent global up/down swap would still pass: H is spin-flip symmetric, and
    # energies and entropies cannot tell a state from its spin flip.)
    (HubbardLattice(Lx=3, W=2, U=4.0), (4, 2), "product", 3),                  # ladder, 4 up 2 dn
    (HubbardLattice(Lx=3, W=2, U=4.0), (3, 3), "product", 3),                  # ladder
    (HubbardLattice(Lx=2, W=3, U=2.0), (3, 3), "product", 2),                  # open strip
    (HubbardLattice(Lx=2, W=3, U=4.0, y_periodic=True), (3, 3), "product", 3), # cylinder
    (HubbardChain(L=5, U=2.0, mu=0.3), (3, 2), "free", 2),                     # odd chain
    (HubbardLattice(Lx=3, W=2, U=4.0), (3, 3), "free", 3),                     # ladder
    (HubbardLattice(Lx=2, W=3, U=4.0, y_periodic=True), (2, 2), "free", 3),    # cylinder
]


@pytest.mark.parametrize("model, sector, init, layers", CASES,
                         ids=["chain", "ladder-imbalanced", "ladder", "strip", "cylinder",
                              "free-odd-chain", "free-ladder", "free-cylinder"])
def test_circuit_matches_statevector(engine, model, sector, init, layers):
    from quantx_engine import hva
    lay = hva.layout(model, engine=engine)
    angles = _angles(layers, lay.angles_per_layer, seed=sum(sector) + layers)
    if init == "free":
        # A zero angle skips its group: the gauge bug the C++ gate was widened for hid
        # exactly there (circuits that never skip could not see it).
        angles[-1][len(angles[-1]) // 2] = 0.0
    r = hva.circuit(model, angles, n_up=sector[0], n_dn=sector[1], init=init,
                    max_bond=256, cutoff=1e-14, engine=engine)
    lat = model if isinstance(model, HubbardLattice) else HubbardLattice(
        Lx=model.L, W=1, t=model.t, U=model.U, mu=model.mu)
    energy, cuts = _reference(lat, lay.groups, angles, *sector, init, r.init_config)
    assert r.truncation_error < 1e-12
    tol = TOL
    if init == "free":
        fock = Fock(lat.L, *sector)
        H0 = fock.hop(sorted(_lattice_bonds(lat.Lx, lat.W, lat.y_periodic)), lat.t)
        e0 = float(np.linalg.eigvalsh(H0)[0])   # prepared at U = 0 AND mu = 0
        assert abs(r.init_energy - e0) < TOL, (r.init_energy, e0)
        assert abs(r.init_energy_exact - e0) < TOL, (r.init_energy_exact, e0)
        assert abs(r.init_energy_error - (r.init_energy - r.init_energy_exact)) < 1e-12
        tol = FREE_TOL
    assert abs(r.energy - energy) < tol, (r.energy, energy)
    assert abs(r.entropy_max - max(cuts)) < tol, (r.entropy_max, cuts)
    if lat.L % 2 == 0:
        assert abs(r.entropy_mid - cuts[lat.L // 2 - 1]) < tol, (r.entropy_mid, cuts)
    else:
        assert r.entropy_mid is None
    assert r.sector == {"n_up": sector[0], "n_dn": sector[1]}


@pytest.mark.parametrize("model", [HubbardLattice(Lx=6, W=1), HubbardLattice(Lx=3, W=2),
                                   HubbardLattice(Lx=3, W=3),
                                   HubbardLattice(Lx=3, W=4, y_periodic=True)],
                         ids=["chain", "ladder", "strip", "cylinder"])
def test_layout_partitions_the_lattice_bonds(engine, model):
    from quantx_engine import hva
    lay = hva.layout(model, engine=engine)
    assert lay.n_sites == model.L
    flat = [tuple(sorted(b)) for g in lay.groups for b in g]
    assert sorted(flat) == sorted(_lattice_bonds(model.Lx, model.W, model.y_periodic))
    for g in lay.groups:   # a group's bonds share no site, so its exponential factorises
        sites = [s for b in g for s in b]
        assert len(sites) == len(set(sites)), g


def test_degenerate_free_start_is_refused(engine):
    # A W = 3 cylinder's transverse levels are -2, 1, 1: at half filling the Fermi level
    # sits inside the degenerate pair.
    from quantx_engine import hva
    cyl = HubbardLattice(Lx=2, W=3, U=4.0, y_periodic=True)
    angles = [[0.1] * hva.layout(cyl, engine=engine).angles_per_layer]
    with pytest.raises(hva.HVAError, match="degenerate"):
        hva.circuit(cyl, angles, n_up=3, n_dn=3, init="free", engine=engine)
    r = hva.circuit(cyl, angles, n_up=3, n_dn=3, init="product", engine=engine)
    assert np.isfinite(r.energy)


def test_rank_deficient_moves_do_not_crash(engine):
    # A review's reproducer: routing the centre moves through Eigen 3.4.0's BDCSVD hit an
    # out-of-bounds read on their exactly rank-deficient blocks and killed the process
    # on about 1 in 20-40 cylinder circuits. BDCSVD engages only for blocks of 16+
    # columns, which is why the six-site gates above could not see it.
    from quantx_engine import hva
    cyl = HubbardLattice(Lx=4, W=3, U=4.0, y_periodic=True)
    lay = hva.layout(cyl, engine=engine)
    angles = np.random.default_rng(11).uniform(-0.6, 0.6, size=(3, lay.angles_per_layer)).tolist()
    r = hva.circuit(cyl, angles, init="product", max_bond=512, engine=engine)
    assert np.isfinite(r.energy) and r.max_bond <= 512


def test_truncation_is_reported(engine):
    from quantx_engine import hva
    chain = HubbardLattice(Lx=12, W=1, U=4.0)
    r = hva.circuit(chain, _angles(4, 3, seed=5), init="product", max_bond=4, engine=engine)
    assert r.max_bond <= 4
    assert r.truncation_error > 1e-8


@pytest.mark.parametrize("model, sector", [
    (HubbardLattice(Lx=3, W=2, U=4.0), (3, 3)),
    (HubbardLattice(Lx=2, W=3, U=4.0, y_periodic=True), (3, 3)),
    (HubbardLattice(Lx=2, W=3, U=6.0, mu=0.5), (2, 3)),
], ids=["ladder", "cylinder", "strip-doped"])
def test_lattice_ground_state_matches_ed(engine, model, sector):
    from quantx_engine.dmrg import ground_state
    fock = Fock(model.L, *sector)
    H = fock.hop(sorted(_lattice_bonds(model.Lx, model.W, model.y_periodic)), model.t)
    ed = float(np.linalg.eigvalsh(H + np.diag(fock.onsite(model.U, model.mu)))[0])
    r = ground_state(model, n_up=sector[0], n_dn=sector[1], max_bond=256, cutoff=1e-14,
                     n_sweeps=40, tol=1e-13, engine=engine)
    assert abs(r.energy - ed) < TOL, (r.energy, ed)


# ── Validation (pure Python, no native library needed) ───────────────────────

@pytest.mark.parametrize("kwargs", [dict(Lx=3, W=2, y_periodic=True), dict(Lx=1, W=2),
                                    dict(Lx=0, W=3), dict(Lx=2, W=2.0)])
def test_lattice_is_checked(kwargs):
    with pytest.raises(ModelError):
        HubbardLattice(**kwargs)


def test_lattice_indexing():
    lat = HubbardLattice(Lx=4, W=3)
    assert lat.L == 12 and lat.site(2, 1) == 7
    assert lat.problem()["type"] == "hubbard_lattice"


def test_circuit_arguments_are_checked():
    from quantx_engine import hva
    with pytest.raises(hva.HVAError):
        hva.circuit(XXZChain(L=4), [[0.1, 0.1, 0.1]])
    with pytest.raises(hva.HVAError):
        hva.circuit(HubbardChain(L=4), [[0.1, 0.1, 0.1]], init="random")
    with pytest.raises(ModelError):
        hva.circuit(HubbardChain(L=4), [[0.1, 0.1, 0.1]], n_up=5)
