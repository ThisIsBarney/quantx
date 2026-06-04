"""Tests for TDVP time evolution (native C++ backend; skipped without it)."""
from __future__ import annotations
import numpy as np
import pytest


@pytest.fixture()
def solver():
    try:
        from quantx_engine import Engine
        from quantx_engine.tdvp import TDVPSolver
        s = TDVPSolver(Engine())
    except Exception as e:
        pytest.skip(f"TDVP/native unavailable: {e}")
    if s._lib is None:
        pytest.skip("native library not built")
    return s


def _dense_xxz(n, delta=1.0):
    sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]])
    sz=np.array([[1,0],[0,-1]],complex); I=np.eye(2,dtype=complex)
    def emb(P,i):
        m=np.array([[1]],complex)
        for k in range(n): m=np.kron(m, P if k==i else I)
        return m
    H=np.zeros((2**n,)*2,complex)
    for i in range(n-1):
        H += emb(sx,i)@emb(sx,i+1) + emb(sy,i)@emb(sy,i+1) + delta*(emb(sz,i)@emb(sz,i+1))
    return H


def _dense_heisenberg(n):
    return _dense_xxz(n, 1.0)


@pytest.mark.parametrize("delta", [0.5, 1.5])
def test_xxz_imag_time_ground_vs_ed(solver, delta):
    # XXZ (delta != 1) cooled by imaginary-time TEBD -> ground energy vs ED.
    n=6
    e0=np.linalg.eigvalsh(_dense_xxz(n, delta))[0]
    g=solver.evolve(n, delta=delta, dt=0.05, n_steps=400, imag=True, chi_max=32)
    assert abs(g.energy - e0) < 2e-3


@pytest.mark.parametrize("delta", [0.5, 1.5])
def test_xxz_real_time_energy_conserved(solver, delta):
    # Neel energy under XXZ is -delta*(n-1); TEBD conserves it (O(dt^2) Trotter).
    n=8
    r=solver.evolve(n, delta=delta, dt=0.02, n_steps=25, chi_max=16)
    assert abs(r.energy - (-delta*(n-1))) < 1e-2


def test_loschmidt_echo_vs_ed():
    # Quench XXZ(0.5) -> XXZ(2.0): return probability L(t) vs exact diagonalization.
    try:
        from quantx_engine import Engine
        from quantx_engine.tdvp import LoschmidtSolver
        s = LoschmidtSolver(Engine())
    except Exception as e:
        pytest.skip(f"Loschmidt/native unavailable: {e}")
    if s._lib is None:
        pytest.skip("native library not built")
    n, di, df = 8, 0.5, 2.0
    H = lambda delta: _dense_xxz(n, delta)
    evi, Vi = np.linalg.eigh(H(di)); psi0 = Vi[:, 0]
    evf, Vf = np.linalg.eigh(H(df))
    def Lex(t):
        c = (Vf.conj().T @ psi0) * np.exp(-1j * evf * t)
        return abs(psi0.conj() @ (Vf @ c)) ** 2
    r = s.echo(n, df, delta=di, dt=0.05, n_steps=40, chi_max=64)
    assert r.L[0] == pytest.approx(1.0, abs=1e-3)
    assert max(abs(r.L[k] - Lex(t)) for k, t in enumerate(r.times)) < 5e-3


def test_imag_time_ground(solver):
    n=6
    H=_dense_heisenberg(n); e0=np.linalg.eigvalsh(H)[0]
    # engine="tdvp2": Krylov-exact local step (no Trotter error), unlike the
    # default TEBD engine, so energy converges to machine precision.
    g=solver.evolve(n, dt=0.1, n_steps=200, imag=True, two_site=True, chi_max=16,
                    engine="tdvp2")
    assert abs(g.energy-e0) < 1e-6


def test_real_time_quench_vs_exact(solver):
    n=6
    H=_dense_heisenberg(n); ev,V=np.linalg.eigh(H)
    sz=np.array([[1,0],[0,-1]],complex); I=np.eye(2,dtype=complex)
    def emb(P,i):
        m=np.array([[1]],complex)
        for k in range(n): m=np.kron(m, P if k==i else I)
        return m
    psi0=np.zeros(2**n,complex)
    idx=0
    for i in range(n): idx=idx*2+(0 if i%2==0 else 1)
    psi0[idx]=1
    def stag_exact(t):
        c=V.conj().T@psi0; c=c*np.exp(-1j*ev*t); ps=V@c
        return sum(((-1)**i)*(ps.conj()@(emb(sz,i)@ps)).real for i in range(n))/n
    r=solver.evolve(n, dt=0.02, n_steps=25, chi_max=16, two_site=True)
    errs=[abs(r.staggered_mag[k]-stag_exact(r.times[k])) for k in range(len(r.times))]
    assert max(errs) < 1e-3


def test_real_time_energy_conserved(solver):
    n=8
    # engine="tdvp2": exact local step conserves energy to machine precision;
    # the default TEBD has O(dt^2) Trotter drift (tested separately below).
    r=solver.evolve(n, dt=0.05, n_steps=40, chi_max=32, two_site=True, engine="tdvp2")
    # Neel energy under Heisenberg sigma.sigma: each bond <sigma.sigma> on
    # alternating spins contributes the ZZ part = -1; (n-1) bonds.
    assert abs(r.energy - (-(n-1))) < 1e-6


def test_tebd_default_energy_conserved_trotter(solver):
    # The default engine is TEBD: energy is conserved only to O(dt^2) Trotter.
    n=8
    r=solver.evolve(n, dt=0.05, n_steps=40, chi_max=32)
    assert r.method == "tebd2-block"
    assert abs(r.energy - (-(n-1))) < 1e-3
