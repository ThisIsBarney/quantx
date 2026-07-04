"""Fermionic Hubbard GROUND-STATE DMRG: QuantX vs TeNPy, best-vs-best.

Same problem on both sides: H = -t sum(c^dag c + h.c.) + U sum n_up n_dn, t=1, U=4,
half filling, OPEN chain, fixed chi_max (svd_min=1e-12), sweep until the energy
change per sweep < ~1e-10. Both conserve N AND Sz (QuantX via the packed two-U(1)
charge). The converged energies must agree to ~1e-8 -- one computation timed twice.

Each tool at its own optimum:
  * TeNPy two-site DMRG. Threads: BOTH single-thread and all-cores are timed and the
    faster is reported (DMRG's larger blocks can benefit from BLAS threads, unlike
    its TEBD) -- we do not guess its optimum, we measure it.
  * QuantX run_dmrg_blocks (via qx_hubbard_showcase QX_HUB_BENCH mode): native
    multi-thread, packed two-U(1), hybrid two-site -> single-site sweeps.

Run (qxbench env):  python benchmarks/hubbard_dmrg_vs_tenpy.py
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
_NCORE = str(os.cpu_count() or 8)

import json, re, subprocess, sys, time

U        = 4.0
N        = int(os.environ.get("BENCH_N", "64"))
REPS     = int(os.environ.get("BENCH_REPS", "2"))
COOLDOWN = float(os.environ.get("BENCH_COOLDOWN", "4"))
PLAN     = [128, 256]
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


def tenpy_dmrg(chi, nthreads):
    # run in a subprocess so the thread env takes effect per-run
    code = f"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="{nthreads}"
import time
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
M = FermiHubbardChain({{"L": {N}, "t": 1.0, "U": {U}, "mu": 0, "V": 0, "bc_MPS": "finite"}})
psi = MPS.from_lat_product_state(M.lat, [["up"], ["down"]])
t0 = time.perf_counter()
info = dmrg.run(psi, M, {{"trunc_params": {{"chi_max": {chi}, "svd_min": 1e-12}},
                          "max_E_err": 1e-10, "max_sweeps": 40}})
w = time.perf_counter() - t0
print("RESULT", w, info["E"])
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
    m = re.search(r"RESULT ([\d.eE+-]+) ([\d.eE+-]+)", out)
    if not m:
        raise RuntimeError(f"tenpy dmrg parse fail (chi={chi}, thr={nthreads}):\n{out[-2000:]}")
    return float(m.group(1)), float(m.group(2))


def quantx(chi):
    env = dict(os.environ)
    env["PATH"] = QX_DLL_DIR + os.pathsep + env.get("PATH", "")
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[_v] = _NCORE
    env["QX_HUB_BENCH"] = f"{N} {chi}"; env["QX_HUB_U"] = str(U)
    out = subprocess.run([QX_BENCH], capture_output=True, text=True, env=env).stdout
    m = re.search(r"E=([\d.eE+-]+) wall=\s*([\d.]+)", out)
    if not m:
        raise RuntimeError(f"qx bench parse fail (chi={chi}):\n{out}")
    return float(m.group(2)), float(m.group(1))


def main():
    if not os.path.exists(QX_BENCH):
        sys.exit(f"qx_hubbard_showcase not found at {QX_BENCH}")
    import tenpy
    print(f"Hubbard GS DMRG  N={N} U={U:g} half-filling reps={REPS}(min) | "
          f"tenpy {tenpy.__version__} (best of 1/{_NCORE} threads, N+Sz) | "
          f"QuantX (multi-thread, N+Sz packed)\n", flush=True)
    hdr = f"{'chi':>5} {'TeNPy(s)':>10} {'thr':>4} {'QX(s)':>9} {'QX<TeNPy':>9}  {'|dE|':>8}"
    print(hdr, flush=True); print("-" * len(hdr), flush=True)
    results = {"N": N, "U": U, "reps": REPS, "runs": []}
    for chi in PLAN:
        t1, e1 = _min_of(lambda: tenpy_dmrg(chi, 1), REPS)
        tm, em = _min_of(lambda: tenpy_dmrg(chi, int(_NCORE)), REPS)
        tt, te, thr = (t1, e1, 1) if t1 <= tm else (tm, em, int(_NCORE))
        qw, qe = _min_of(lambda: quantx(chi), REPS)
        dE = abs(qe - te)
        print(f"{chi:>5} {tt:>10.2f} {thr:>4} {qw:>9.2f} {tt/qw:>8.2f}x  {dE:>8.1e}", flush=True)
        results["runs"].append({"chi": chi, "tenpy": tt, "tenpy_threads": thr,
                                "qx": qw, "ratio": tt/qw, "E_tenpy": te, "E_qx": qe})
        time.sleep(COOLDOWN)
    out = os.path.join(_here, "results_hubbard_dmrg_vs_tenpy.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
