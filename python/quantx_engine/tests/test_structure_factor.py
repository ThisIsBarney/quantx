"""Tests for the dynamical structure factor S(k,omega) (native backend)."""
from __future__ import annotations
import numpy as np
import pytest


@pytest.fixture()
def solver():
    try:
        from quantx_engine import Engine
        from quantx_engine.structure_factor import StructureFactorSolver
        s = StructureFactorSolver(Engine())
    except Exception as e:
        pytest.skip(f"structure factor/native unavailable: {e}")
    if s._lib is None:
        pytest.skip("native library not built")
    return s


def _ed(n):
    sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]])
    sz=np.array([[1,0],[0,-1]],complex); I=np.eye(2,dtype=complex)
    def emb(P,i):
        m=np.array([[1]],complex)
        for k in range(n): m=np.kron(m, P if k==i else I)
        return m
    H=np.zeros((2**n,)*2,complex)
    for i in range(n-1):
        for P in (sx,sy,sz): H+=emb(P,i)@emb(P,i+1)
    ev,V=np.linalg.eigh(H)
    Zs=[emb(sz,i) for i in range(n)]
    return ev,V,Zs


def test_static_structure_factor_vs_ed(solver):
    n = 8
    r = solver.correlations(n, dt=0.05, n_steps=10, chi_max=32, n_sweeps=40)
    # static S(k) = sum_j e^{-ik(j-l0)} <sz_j sz_l0>  from C(t=0) vs exact
    ev,V,Zs = _ed(n)
    psi0 = V[:,0]
    l0 = r.l0
    C0_exact = np.array([np.real(psi0.conj() @ (Zs[j] @ (Zs[l0] @ psi0))) for j in range(n)])
    C0_block = np.real(r.C[0])
    assert np.max(np.abs(C0_block - C0_exact)) < 1e-6
    # k-resolved static factor agrees too
    ks = 2*np.pi*np.arange(n)/n
    rr = np.arange(n) - l0
    Sk_b = np.real(np.exp(-1j*np.outer(ks,rr)) @ C0_block)
    Sk_e = np.real(np.exp(-1j*np.outer(ks,rr)) @ C0_exact)
    assert np.max(np.abs(Sk_b - Sk_e)) < 1e-6


def test_dynamic_correlation_vs_ed(solver):
    n = 8
    r = solver.correlations(n, dt=0.05, n_steps=20, chi_max=32, n_sweeps=40)
    ev,V,Zs = _ed(n); psi0=V[:,0]; E0=ev[0]; l0=r.l0
    phi0 = Zs[l0] @ psi0
    for ti,t in enumerate(r.times):
        c = V.conj().T @ phi0
        c = c * np.exp(-1j*(ev-E0)*t)
        phit = V @ c
        Cex = np.array([psi0.conj() @ (Zs[j] @ phit) for j in range(n)])
        # evolution is TEBD (O(dt^2) Trotter error), so accuracy is ~1e-2 not 1e-5
        assert np.max(np.abs(r.C[ti] - Cex)) < 2e-2


def test_structure_factor_shape_and_peak(solver):
    n = 16
    r = solver.correlations(n, dt=0.05, n_steps=200, chi_max=64)
    ks, w, S = solver.structure_factor(r, n_omega=300, omega_max=np.pi, eta=0.12)
    assert S.shape == (n, 300)
    # AFM Heisenberg: spectral weight concentrates near k=pi (the ordering vector)
    weight = np.trapezoid(np.maximum(S, 0), w, axis=1)
    ik_pi = np.argmin(np.abs(ks - np.pi))
    assert weight[ik_pi] == pytest.approx(weight.max(), rel=0.2)
