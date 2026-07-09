"""Two-leg Hubbard LADDER ground-state DMRG: QuantX (rung-fused d=16) vs TeNPy, best-vs-best.

Same problem: H = -t (leg hops) - t_perp (rung hops) + U sum n_up n_dn, t=t_perp=1,
U=8, OPEN 2xLx ladder, half filling, N+Sz conserved, fixed chi (svd_min 1e-12),
converge to dE < 1e-10 per sweep. Final energies must agree to ~1e-7.

The two tools use their own natural formulations of the SAME Hamiltonian:
  * TeNPy: "Ladder" lattice of Electron sites (2Lx MPS sites, its MPO handles the
    range-2 leg couplings natively), single-thread AND all-cores timed, faster taken.
  * QuantX: rung-fused chain (Lx MPS sites of d=16, JW strings inside the operators)
    -- the formulation the universal-MPO engine runs with zero engine changes.
Chi caps are matched in the PHYSICAL sense (same bond of the same ladder cut carries
at most chi states in both formulations -- the fused chain's bonds ARE the rung cuts).

Run:  python benchmarks/hubbard_ladder_vs_tenpy.py
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
_NCORE = str(os.cpu_count() or 8)

import json, re, subprocess, sys, time

U        = 8.0
LX       = int(os.environ.get("BENCH_LX", "16"))
REPS     = int(os.environ.get("BENCH_REPS", "2"))
COOLDOWN = float(os.environ.get("BENCH_COOLDOWN", "4"))
PLAN     = [256]
if os.environ.get("BENCH_PLAN"):
    PLAN = [int(c) for c in os.environ["BENCH_PLAN"].split(",")]

_here = os.path.dirname(__file__)
QX_BENCH = os.environ.get("QX_HUB_SHOWCASE",
                          os.path.join(_here, "..", "build-cuda-blas", "Release", "qx_hubbard_showcase.exe"))
QX_DLL_DIR = os.environ.get("QX_DLL_DIR", r"C:\ProgramData\miniforge3\envs\qxblas\Library\bin")


def _min_of(fn, reps):
    best, e = float("inf"), None
    for r in range(reps):
        w, en = fn()
        if w < best: best, e = w, en
        if r < reps - 1: time.sleep(COOLDOWN)
    return best, e


def tenpy_ladder(chi, nthreads):
    code = f"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="{nthreads}"
import time
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
M = FermiHubbardModel({{"lattice": "Ladder", "L": {LX}, "t": 1.0, "U": {U},
                        "mu": 0, "V": 0, "bc_MPS": "finite"}})
psi = MPS.from_product_state(M.lat.mps_sites(), ["up", "down"] * {LX})
t0 = time.perf_counter()
info = dmrg.run(psi, M, {{"trunc_params": {{"chi_max": {chi}, "svd_min": 1e-12}},
                          "max_E_err": 1e-10, "max_sweeps": 40}})
w = time.perf_counter() - t0
print("RESULT", w, info["E"])
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
    m = re.search(r"RESULT ([\d.eE+-]+) ([\d.eE+-]+)", out)
    if not m:
        raise RuntimeError(f"tenpy ladder parse fail (chi={chi}, thr={nthreads}):\n{out[-2000:]}")
    return float(m.group(1)), float(m.group(2))


def quantx(chi):
    env = dict(os.environ)
    env["PATH"] = QX_DLL_DIR + os.pathsep + env.get("PATH", "")
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[_v] = _NCORE
    # site-split (period-2 d=4) formulation by default -- the performance mapping
    # (same manifold as TeNPy's site chain); QX_FUSED=1 falls back to rung fusion.
    split = "0" if os.environ.get("QX_FUSED") == "1" else "1"
    env["QX_LADDER"] = f"{LX} {chi} {U} 1.0 {split}"
    out = subprocess.run([QX_BENCH], capture_output=True, text=True, env=env).stdout
    m = re.search(r"E0=([\d.eE+-]+) \(([\d.]+)s\)", out)
    if not m:
        raise RuntimeError(f"qx ladder parse fail (chi={chi}):\n{out}")
    return float(m.group(2)), float(m.group(1))


def main():
    for p, what in ((QX_BENCH, "qx_hubbard_showcase"),):
        if not os.path.exists(p):
            sys.exit(f"{what} not found at {p}")
    import tenpy
    print(f"2-leg Hubbard LADDER GS DMRG  Lx={LX} (2x{LX} sites) U={U:g} half filling "
          f"reps={REPS}(min) | tenpy {tenpy.__version__} (Ladder lattice, best of "
          f"1/{_NCORE} thr, N+Sz) | QuantX (site-split period-2 d=4, MT, N+Sz packed)\n", flush=True)
    hdr = f"{'chi':>5} {'TeNPy(s)':>10} {'thr':>4} {'QX(s)':>9} {'QX<TeNPy':>9}  {'|dE|':>8}"
    print(hdr, flush=True); print("-" * len(hdr), flush=True)
    results = {"Lx": LX, "U": U, "reps": REPS, "runs": []}
    for chi in PLAN:
        t1, e1 = _min_of(lambda: tenpy_ladder(chi, 1), REPS)
        tm, em = _min_of(lambda: tenpy_ladder(chi, int(_NCORE)), REPS)
        tt, te, thr = (t1, e1, 1) if t1 <= tm else (tm, em, int(_NCORE))
        qw, qe = _min_of(lambda: quantx(chi), REPS)
        dE = abs(qe - te)
        print(f"{chi:>5} {tt:>10.2f} {thr:>4} {qw:>9.2f} {tt/qw:>8.2f}x  {dE:>8.1e}", flush=True)
        results["runs"].append({"chi": chi, "tenpy": tt, "tenpy_threads": thr,
                                "qx": qw, "ratio": tt/qw, "E_tenpy": te, "E_qx": qe})
        time.sleep(COOLDOWN)
    out = os.path.join(_here, "results_hubbard_ladder_vs_tenpy.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
