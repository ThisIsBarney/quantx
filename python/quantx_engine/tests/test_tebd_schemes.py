"""The real-time TEBD schemes (M12) through the Python wrappers, against exact
diagonalisation: the default left-right-left sweep and the opt-in even/odd scheme
(TeNPy's splitting on a B-form MPS). Both are second-order Trotter, so each matches
the exact evolution to O(dt^2), and the two differ from each other at that order."""
from __future__ import annotations

import numpy as np
import pytest

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.array([[1, 0], [0, -1]], complex)
SP = np.array([[0, 1], [0, 0]], complex)
SM = np.array([[0, 0], [1, 0]], complex)


def _native(module, cls):
    try:
        from quantx_engine import Engine
        mod = __import__(f"quantx_engine.{module}", fromlist=[cls])
        s = getattr(mod, cls)(Engine())
    except Exception as e:
        pytest.skip(f"{cls}/native unavailable: {e}")
    if s._lib is None:
        pytest.skip("native library not built")
    return s


def _emb(P, i, n):
    m = np.array([[1]], complex)
    for k in range(n):
        m = np.kron(m, P if k == i else np.eye(2, dtype=complex))
    return m


def _xxz(n, delta=1.0):
    H = np.zeros((2 ** n,) * 2, complex)
    for i in range(n - 1):
        H += (_emb(SX, i, n) @ _emb(SX, i + 1, n) + _emb(SY, i, n) @ _emb(SY, i + 1, n)
              + delta * _emb(SZ, i, n) @ _emb(SZ, i + 1, n))
    return H


SCHEMES = ["sweep", "even_odd"]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_neel_quench_vs_ed(scheme):
    s = _native("tdvp", "TDVPSolver")
    n = 6
    ev, V = np.linalg.eigh(_xxz(n))
    psi0 = np.zeros(2 ** n, complex)
    psi0[int("".join("0" if i % 2 == 0 else "1" for i in range(n)), 2)] = 1
    Z = [_emb(SZ, i, n) for i in range(n)]

    def stag(t):
        ps = V @ ((V.conj().T @ psi0) * np.exp(-1j * ev * t))
        return sum((-1) ** i * (ps.conj() @ (Z[i] @ ps)).real for i in range(n)) / n

    r = s.evolve(n, dt=0.02, n_steps=25, chi_max=16, scheme=scheme)
    assert max(abs(m - stag(t)) for m, t in zip(r.staggered_mag, r.times)) < 1e-3
    assert abs(r.energy - (-(n - 1))) < 1e-3          # the Neel energy, conserved


@pytest.mark.parametrize("scheme", SCHEMES)
def test_dynamic_correlation_vs_ed(scheme):
    s = _native("structure_factor", "StructureFactorSolver")
    n = 6
    ev, V = np.linalg.eigh(_xxz(n))
    psi0, E0 = V[:, 0], ev[0]
    r = s.correlations(n, dt=0.05, n_steps=20, chi_max=32, cutoff=1e-12, scheme=scheme)
    Z = [_emb(SZ, i, n) for i in range(n)]
    phi = Z[r.l0] @ psi0
    worst = 0.0
    for ti, t in enumerate(r.times):
        phit = V @ ((V.conj().T @ phi) * np.exp(-1j * (ev - E0) * t))
        Cex = np.array([psi0.conj() @ (Z[j] @ phit) for j in range(n)])
        worst = max(worst, np.max(np.abs(r.C[ti] - Cex)))
    assert worst < 8e-3


@pytest.mark.parametrize("scheme", SCHEMES)
def test_loschmidt_vs_ed(scheme):
    s = _native("tdvp", "LoschmidtSolver")
    n, di, df = 6, 0.5, 1.5
    psi0 = np.linalg.eigh(_xxz(n, di))[1][:, 0]
    evf, Vf = np.linalg.eigh(_xxz(n, df))
    r = s.echo(n, df, delta=di, dt=0.05, n_steps=30, chi_max=32, cutoff=1e-12, scheme=scheme)
    for L, t in zip(r.L, r.times):
        ps = Vf @ ((Vf.conj().T @ psi0) * np.exp(-1j * evf * t))
        assert abs(L - abs(psi0.conj() @ ps) ** 2) < 5e-3


@pytest.mark.parametrize("scheme", SCHEMES)
def test_conductivity_vs_ed(scheme):
    s = _native("structure_factor", "ConductivitySolver")
    n = 6
    ev, V = np.linalg.eigh(_xxz(n))
    psi0, E0 = V[:, 0], ev[0]
    Jop = sum(1j * (_emb(SP, i, n) @ _emb(SM, i + 1, n) - _emb(SM, i, n) @ _emb(SP, i + 1, n))
              for i in range(n - 1))
    c = V.conj().T @ (Jop @ psi0)
    r = s.correlator(n, dt=0.0125, n_steps=120, chi_max=32, cutoff=1e-12, scheme=scheme)
    for C, t in zip(r.C, r.times):
        assert abs(C - np.sum(np.abs(c) ** 2 * np.exp(-1j * (ev - E0) * t))) < 5e-3


def test_the_schemes_differ_at_trotter_order():
    # Different second-order splittings: their results differ, by O(dt^2).
    s = _native("tdvp", "TDVPSolver")
    a = s.evolve(8, dt=0.05, n_steps=20, chi_max=64, scheme="sweep")
    b = s.evolve(8, dt=0.05, n_steps=20, chi_max=64, scheme="even_odd")
    d = max(abs(x - y) for x, y in zip(a.staggered_mag, b.staggered_mag))
    assert 1e-6 < d < 5e-2


def test_refusals():
    from quantx_engine.tdvp import TDVPError
    s = _native("tdvp", "TDVPSolver")
    with pytest.raises(TDVPError, match="scheme"):
        s.evolve(6, n_steps=2, scheme="bogus")
    with pytest.raises(TDVPError, match="real-time only"):          # needs unitary gates
        s.evolve(6, n_steps=2, imag=True, scheme="even_odd")
    # Wherever the block TEBD does not run, the scheme is refused rather than ignored.
    for kw in ({"engine": "tdvp2"}, {"block": False}, {"two_site": False},
               {"initial": "up"}):
        with pytest.raises(TDVPError, match="block TEBD only"):
            s.evolve(6, n_steps=2, scheme="even_odd", **kw)
    with pytest.raises(TDVPError, match="block TEBD only"):
        s.evolve(2, n_steps=2, scheme="even_odd")
