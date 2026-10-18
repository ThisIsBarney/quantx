"""Real-time benchmark: QuantX's TDVPSolver vs TeNPy's TwoSiteTDVPEngine.

Superseded as a speed claim. The engine now runs this nearest-neighbour quench with block
TEBD (the result's method is "tebd2-block"), so the QuantX side is no longer TDVP2, and
TeNPy's fast method here is its TEBD, not its TDVP: see the correction at the top of
docs/dmrg_tdvp_report.md and docs/time_evolution_status.md. Kept as an accuracy check
against exact diagonalisation.

Real-time quench of the open spin-1/2 Heisenberg chain from the Neel state,
H = J sum_i sigma_i . sigma_{i+1} (Pauli convention, J=1). We evolve and record
the staggered magnetization m(t) = (1/N) sum_i (-1)^i <sigma^z_i>, comparing
both engines' accuracy (vs exact diagonalization for small N) and wall time.

TeNPy uses spin-1/2 operators S = sigma/2, so couplings are set Jx=Jy=Jz=4 and
its <Sz> is doubled to sigma^z units.

Thread control: BENCH_THREADS (default 1), pinned before numpy loads.

Run:  python benchmarks/tdvp_vs_tenpy.py
"""
from __future__ import annotations

import os
_THREADS = os.environ.get("BENCH_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import json
import time
import numpy as np


def exact_stag(n, dt, n_steps):
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
    psi0=np.zeros(2**n,complex)
    idx=0
    for i in range(n): idx=idx*2+(0 if i%2==0 else 1)
    psi0[idx]=1
    Zs=[emb(sz,i) for i in range(n)]
    out=[]
    for step in range(n_steps+1):
        t=step*dt
        c=V.conj().T@psi0; c=c*np.exp(-1j*ev*t); ps=V@c
        m=sum(((-1)**i)*(ps.conj()@(Zs[i]@ps)).real for i in range(n))/n
        out.append(m)
    return out


def quantx_run(n, dt, n_steps, chi):
    from quantx_engine.engine import Engine
    from quantx_engine.tdvp import TDVPSolver
    s = TDVPSolver(Engine())
    r = s.evolve(n, dt=dt, n_steps=n_steps, chi_max=chi, two_site=True, initial="neel")
    return r.staggered_mag, r.wall_time_s, r.max_bond


def tenpy_run(n, dt, n_steps, chi):
    from tenpy.models.spins import SpinChain
    from tenpy.networks.mps import MPS
    from tenpy.algorithms.tdvp import TwoSiteTDVPEngine
    M = SpinChain({"S":0.5,"L":n,"Jx":4,"Jy":4,"Jz":4,"hz":0,
                   "bc_MPS":"finite","conserve":None})
    psi = MPS.from_product_state(M.lat.mps_sites(), ["up","down"]*(n//2), "finite")
    eng = TwoSiteTDVPEngine(psi, M, {"dt":dt, "N_steps":1,
                                     "trunc_params":{"chi_max":chi,"svd_min":1e-12}})
    def stag():
        sz = psi.expectation_value("Sz")        # spin-1/2 units
        return float(sum(((-1)**i)*2.0*sz[i] for i in range(n))/n)  # -> sigma^z units
    series=[stag()]
    t0=time.perf_counter()
    for _ in range(n_steps):
        eng.run()
        series.append(stag())
    return series, time.perf_counter()-t0, max(psi.chi)


def main():
    print(f"threads={_THREADS}\n")
    dt=0.05; T=2.0; n_steps=int(round(T/dt)); chi=64
    print(f"Heisenberg Neel quench, dt={dt}, T={T}, chi_max={chi}\n")
    results={"threads":_THREADS,"runs":[]}
    for n in [10, 16, 20]:
        qx_m, qx_t, qx_b = quantx_run(n, dt, n_steps, chi)
        tp_m, tp_t, tp_b = tenpy_run(n, dt, n_steps, chi)
        line=f"N={n:<3} QuantX {qx_t:7.3f}s (bond {qx_b})   TeNPy {tp_t:7.3f}s (bond {tp_b})   speedup {tp_t/qx_t:5.1f}x"
        if n <= 12:
            ex=exact_stag(n, dt, n_steps)                  # t = 0, dt, ..., T
            off=len(ex)-len(qx_m)                         # QuantX's series leaves out t = 0
            eq=max(abs(qx_m[k]-ex[k+off]) for k in range(len(qx_m)))
            et=max(abs(tp_m[k]-ex[k]) for k in range(len(ex)))
            line+=f"   err(QuantX)={eq:.1e} err(TeNPy)={et:.1e}"
        print(line)
        results["runs"].append({"N":n,"qx_time":qx_t,"tp_time":tp_t,
                                "qx_bond":qx_b,"tp_bond":tp_b})
    out=os.path.join(os.path.dirname(__file__), "results_tdvp_vs_tenpy.json")
    with open(out,"w") as f: json.dump(results,f,indent=2)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
