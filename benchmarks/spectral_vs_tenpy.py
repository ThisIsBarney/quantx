"""Dynamical structure factor benchmark: QuantX vs TeNPy.

Both compute the time-dependent ground-state correlations
C_{j,l0}(t) = <psi0| sz_j e^{-i(H-E0)t} sz_l0 |psi0> for a central source l0 of an
open spin-1/2 Heisenberg chain (H = J sum sigma.sigma), then S(k,omega) is a cheap
FFT. The expensive part is the real-time evolution + per-step measurement.

QuantX: DMRG ground state + block-sparse (Sz) real-time TDVP2 + block two-MPS
overlap. TeNPy: DMRG ground state + TEBD real-time evolution + MPSEnvironment
measurement. We report the wall time of the full C_{j,l0}(t) computation and
cross-check the correlations agree.

Thread control: BENCH_THREADS (default 1).

Run:  python benchmarks/spectral_vs_tenpy.py
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


def quantx_corr(n, dt, n_steps, chi):
    from quantx_engine.engine import Engine
    from quantx_engine.structure_factor import StructureFactorSolver
    s = StructureFactorSolver(Engine())
    r = s.correlations(n, dt=dt, n_steps=n_steps, chi_max=chi, n_sweeps=30)
    return r.C, r.times, r.l0, r.wall_time_s, r.max_bond


def tenpy_corr(n, dt, n_steps, chi):
    from tenpy.models.spins import SpinChain
    from tenpy.networks.mps import MPS, MPSEnvironment
    from tenpy.algorithms import dmrg, tebd
    M = SpinChain({"S":0.5,"L":n,"Jx":4,"Jy":4,"Jz":4,"hz":0,
                   "bc_MPS":"finite","conserve":"Sz"})
    t0 = time.perf_counter()
    # ground state
    psi0 = MPS.from_lat_product_state(M.lat, [["up"],["down"]])
    dmrg.TwoSiteDMRGEngine(psi0, M, {"trunc_params":{"chi_max":chi,"svd_min":1e-10},
                                     "max_sweeps":30, "mixer":True,
                                     "combine":True}).run()
    l0 = n // 2
    # TeNPy SpinSite uses Sz = sigma^z / 2, so C(sigma) = 4 * C(Sz). Evolve
    # phi = Sz_l0 |psi0> and rescale the measured correlations by 4 at the end.
    phi = psi0.copy()
    phi.apply_local_op(l0, "Sz", unitary=False)
    eng = tebd.TEBDEngine(phi, M, {"trunc_params":{"chi_max":chi,"svd_min":1e-10},
                                   "dt":dt, "order":2, "N_steps":1})
    n_t = n_steps + 1
    C = np.zeros((n_t, n), dtype=complex)
    times = np.zeros(n_t)
    E0 = M.H_MPO.expectation_value(psi0)
    for ti in range(n_t):
        t = ti * dt
        env = MPSEnvironment(psi0, phi)
        sz = env.expectation_value("Sz")              # <psi0| Sz_j |phi(t)>
        C[ti, :] = 4.0 * np.exp(1j * E0 * t) * sz     # -> sigma^z convention
        times[ti] = t
        if ti < n_steps:
            eng.run_evolution(1, dt)                    # one real-time step
    bond = max(phi.chi)
    return C, times, l0, time.perf_counter() - t0, bond


def main():
    print(f"threads={_THREADS}\n")
    hdr = f"{'N':>3} {'steps':>5} {'chi':>4} {'method':<14} {'time(s)':>9} {'bond':>5}"
    print(hdr); print("-"*len(hdr))
    results = {"threads":_THREADS, "runs":[]}
    dt, n_steps = 0.05, 150
    for n, chi in [(16, 64), (24, 96), (32, 128), (48, 128)]:
        row = {}
        for name, fn in [("quantx-tdvp2", quantx_corr), ("tenpy-tebd", tenpy_corr)]:
            try:
                C, times, l0, wt, bond = fn(n, dt, n_steps, chi)
                row[name] = (C, wt, bond)
                print(f"{n:>3} {n_steps:>5} {chi:>4} {name:<14} {wt:>9.2f} {bond:>5}")
                results["runs"].append({"N":n,"steps":n_steps,"chi":chi,"method":name,
                                        "time_s":wt,"bond":bond})
            except Exception as e:
                print(f"{n:>3} {n_steps:>5} {chi:>4} {name:<14}  FAILED: {e}")
                row[name] = None
        if row.get("quantx-tdvp2") and row.get("tenpy-tebd"):
            qt, tt = row["quantx-tdvp2"][1], row["tenpy-tebd"][1]
            # agreement on the correlations (max abs diff over j,t)
            Cd = np.max(np.abs(row["quantx-tdvp2"][0] - row["tenpy-tebd"][0]))
            speed = tt / qt if qt > 0 else float("nan")
            print(f"{'':>3} {'':>5} {'':>4} {'-> speedup':<14} {speed:>8.1f}x  (C agree |d|={Cd:.2e})")
        print()
    out = os.path.join(os.path.dirname(__file__), "results_spectral_vs_tenpy.json")
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
