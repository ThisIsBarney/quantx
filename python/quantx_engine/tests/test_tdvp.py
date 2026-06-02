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


def _dense_heisenberg(n):
    sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]])
    sz=np.array([[1,0],[0,-1]],complex); I=np.eye(2,dtype=complex)
    def emb(P,i):
        m=np.array([[1]],complex)
        for k in range(n): m=np.kron(m, P if k==i else I)
        return m
    H=np.zeros((2**n,)*2,complex)
    for i in range(n-1):
        for P in (sx,sy,sz): H+=emb(P,i)@emb(P,i+1)
    return H


def test_imag_time_ground(solver):
    n=6
    H=_dense_heisenberg(n); e0=np.linalg.eigvalsh(H)[0]
    g=solver.evolve(n, dt=0.1, n_steps=200, imag=True, two_site=True, chi_max=16)
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
    r=solver.evolve(n, dt=0.05, n_steps=40, chi_max=32, two_site=True)
    # Neel energy under Heisenberg sigma.sigma: each bond <sigma.sigma> on
    # alternating spins contributes the ZZ part = -1; (n-1) bonds.
    assert abs(r.energy - (-(n-1))) < 1e-6
