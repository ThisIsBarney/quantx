"""Width-4 Hubbard CYLINDER ground-state DMRG: QuantX vs TeNPy, best-vs-best.

The minimal stage of 2D Hubbard physics (stripes live on width-4+ cylinders):
H = -t (all NN bonds of an Lx x 4 square lattice, periodic around y) + U sum n_up n_dn,
t=1, U=8, half filling, N+Sz conserved, fixed chi (svd_min 1e-12). Width-4 cylinders
at practical chi are TRUNCATION-limited (per-sweep dE plateaus above any tol), so the
protocol is a FIXED SWEEP BUDGET (12 sweeps, both tools): report wall time and the
energy reached -- lower E at equal budget = better convergence per sweep.
No small-size ED exists at this width -- the CROSS-ENGINE energy agreement (~1e-6 at
fixed chi) is itself the correctness check, on top of the W=2/W=3 ED-exact validation
of the same builder.

  * TeNPy: Square lattice Lx x 4, bc_y="cylinder" (its native 2D path; MPO handles the
    long-range snake couplings), 1 and 20 threads timed, faster taken.
  * QuantX: build_hubbard_lattice_split_mpo (period-4 MPO, range-4 F-carry chains),
    column-major -- the same snake ordering as TeNPy's default, so chi caps match cut
    for cut.

Run:  python benchmarks/hubbard_cylinder_vs_tenpy.py
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
_NCORE = str(os.cpu_count() or 8)

import json, re, subprocess, sys, time

U        = 8.0
LX       = int(os.environ.get("BENCH_LX", "8"))
WY       = 4
REPS     = int(os.environ.get("BENCH_REPS", "2"))
COOLDOWN = float(os.environ.get("BENCH_COOLDOWN", "4"))
SWEEPS   = int(os.environ.get("BENCH_SWEEPS", "12"))
PLAN     = [256, 512]
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


def tenpy_cyl(chi, nthreads):
    code = f"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="{nthreads}"
import time
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
M = FermiHubbardModel({{"lattice": "Square", "Lx": {LX}, "Ly": {WY}, "t": 1.0, "U": {U},
                        "mu": 0, "V": 0, "bc_MPS": "finite", "bc_y": "cylinder"}})
psi = MPS.from_product_state(M.lat.mps_sites(), ["up", "down"] * ({LX} * {WY} // 2))
t0 = time.perf_counter()
info = dmrg.run(psi, M, {{"trunc_params": {{"chi_max": {chi}, "svd_min": 1e-12}},
                          "max_E_err": 1e-12, "max_sweeps": {SWEEPS}, "min_sweeps": {SWEEPS},
                          "max_trunc_err": 1.0}})   # disable the safety guard: fixed-chi
                                                    # cylinder runs are truncation-limited by design
w = time.perf_counter() - t0
print("RESULT", w, info["E"])
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
    m = re.search(r"RESULT ([\d.eE+-]+) ([\d.eE+-]+)", out)
    if not m:
        raise RuntimeError(f"tenpy cylinder parse fail (chi={chi}, thr={nthreads}):\n{out[-2000:]}")
    return float(m.group(1)), float(m.group(2))


def quantx(chi):
    env = dict(os.environ)
    env["PATH"] = QX_DLL_DIR + os.pathsep + env.get("PATH", "")
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[_v] = _NCORE
    env["QX_CYL"] = f"{LX} {WY} {chi} {U} 1 {SWEEPS}"
    out = subprocess.run([QX_BENCH], capture_output=True, text=True, env=env).stdout
    m = re.search(r"E0=([\d.eE+-]+) \(([\d.]+)s\)", out)
    if not m:
        raise RuntimeError(f"qx cylinder parse fail (chi={chi}):\n{out}")
    return float(m.group(2)), float(m.group(1))


def main():
    if not os.path.exists(QX_BENCH):
        sys.exit(f"qx_hubbard_showcase not found at {QX_BENCH}")
    import tenpy
    print(f"Width-4 Hubbard CYLINDER GS DMRG  {LX}x{WY} (N={LX*WY} sites) U={U:g} "
          f"half filling reps={REPS}(min) | tenpy {tenpy.__version__} (Square/cylinder, "
          f"best of 1/{_NCORE} thr, N+Sz) | QuantX (period-4 site-split, MT, N+Sz packed)\n",
          flush=True)
    hdr = f"{'chi':>5} {'TeNPy(s)':>10} {'thr':>4} {'QX(s)':>9} {'QX<TeNPy':>9}  {'|dE|':>8}"
    print(hdr, flush=True); print("-" * len(hdr), flush=True)
    results = {"Lx": LX, "Wy": WY, "U": U, "reps": REPS, "runs": []}
    for chi in PLAN:
        t1, e1 = _min_of(lambda: tenpy_cyl(chi, 1), REPS)
        tm, em = _min_of(lambda: tenpy_cyl(chi, int(_NCORE)), REPS)
        tt, te, thr = (t1, e1, 1) if t1 <= tm else (tm, em, int(_NCORE))
        qw, qe = _min_of(lambda: quantx(chi), REPS)
        dE = abs(qe - te)
        print(f"{chi:>5} {tt:>10.2f} {thr:>4} {qw:>9.2f} {tt/qw:>8.2f}x  {dE:>8.1e}", flush=True)
        results["runs"].append({"chi": chi, "tenpy": tt, "tenpy_threads": thr,
                                "qx": qw, "ratio": tt/qw, "E_tenpy": te, "E_qx": qe})
        time.sleep(COOLDOWN)
    out = os.path.join(_here, "results_hubbard_cylinder_vs_tenpy.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
