"""Three-way DMRG benchmark: QuantX U(1) block-sparse vs QuantX SU(2) vs TeNPy.

Ground state of the open Heisenberg chain  H = J * sum_i sigma_i . sigma_{i+1}
(Pauli convention, J=1). TeNPy uses spin-1/2 operators S=sigma/2, so its
couplings are set to Jx=Jy=Jz=4 to reproduce the same H.

For each chain length we run a bond-dimension ladder per method, recording the
converged energy and wall time, and the energy error against the best (lowest)
energy found. This gives an accuracy-vs-time comparison: the method that reaches
a target accuracy in the least time wins.

Thread control: set BENCH_THREADS (default 1) -- BLAS / OpenMP threads are
pinned before numpy loads so the comparison is per-core unless overridden.

Run:  python benchmarks/dmrg_su2_vs_tenpy.py [--quick]
Saves benchmarks/results_su2_vs_tenpy.json
"""
from __future__ import annotations

import os
_THREADS = os.environ.get("BENCH_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import argparse
import json
import time

import numpy as np


# ── TeNPy runner ──────────────────────────────────────────────────────────────
def tenpy_run(N: int, chi: int, max_sweeps: int = 30):
    from tenpy.models.spins import SpinChain
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import dmrg
    M = SpinChain({"S": 0.5, "L": N, "Jx": 4, "Jy": 4, "Jz": 4, "hz": 0,
                   "bc_MPS": "finite", "conserve": "Sz"})
    psi = MPS.from_product_state(M.lat.mps_sites(), ["up", "down"] * (N // 2), "finite")
    eng = dmrg.TwoSiteDMRGEngine(psi, M, {
        "trunc_params": {"chi_max": chi, "svd_min": 1e-12},
        "max_sweeps": max_sweeps, "min_sweeps": 2,
        "mixer": True, "combine": True,
    })
    t0 = time.perf_counter()
    E, _ = eng.run()
    return float(E), time.perf_counter() - t0


# ── QuantX runner ─────────────────────────────────────────────────────────────
def quantx_run(solver, N: int, param: int, method: str, n_sweeps: int = 10):
    r = solver.solve(N, hamiltonian_type="heisenberg", J=1.0,
                     max_bond=param, n_sweeps=n_sweeps, method=method)
    return float(r.energy), float(r.wall_time_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grid")
    args = ap.parse_args()

    from quantx_engine.engine import Engine
    from quantx_engine.dmrg import DMRGSolver
    solver = DMRGSolver(Engine())

    if args.quick:
        Ns = [32]
        chis = [16, 32, 64]
        mults = [8, 16, 32]
        ref_chi = 96
    else:
        Ns = [32, 48, 64]
        chis = [16, 32, 64, 128]
        mults = [8, 16, 32, 64]
        ref_chi = 200

    print(f"threads={_THREADS}  (BLAS/OpenMP pinned)\n")
    results = {"threads": _THREADS, "runs": []}

    for N in Ns:
        # reference: best energy from a high-chi TeNPy run
        E_ref, _ = tenpy_run(N, ref_chi, max_sweeps=40)
        print(f"=== N={N}  (E_ref={E_ref:.8f}, E_ref/N={E_ref/N:.6f}) ===")
        print(f"{'method':<10}{'param':>6}{'energy':>16}{'err':>12}{'time(s)':>10}")

        rows = []
        for chi in chis:
            E, t = tenpy_run(N, chi)
            rows.append(("tenpy", chi, E, t))
        for chi in chis:
            E, t = quantx_run(solver, N, chi, "u1")
            rows.append(("quantx-u1", chi, E, t))
        for m in mults:
            E, t = quantx_run(solver, N, m, "su2")
            rows.append(("quantx-su2", m, E, t))

        for name, p, E, t in rows:
            err = abs(E - E_ref)
            print(f"{name:<10}{p:>6}{E:>16.8f}{err:>12.2e}{t:>10.3f}")
            results["runs"].append({"N": N, "method": name, "param": p,
                                    "energy": E, "err": err, "time_s": t,
                                    "E_ref": E_ref})
        print()

    out = os.path.join(os.path.dirname(__file__), "results_su2_vs_tenpy.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
