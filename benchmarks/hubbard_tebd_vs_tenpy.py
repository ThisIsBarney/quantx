"""Fermionic Hubbard (d=4) real-time TEBD: QuantX vs TeNPy, the "are we leading?" benchmark.

Same physics on both sides:
  H = -t sum_{i,s}(c^dag_{i s} c_{i+1 s} + h.c.) + U sum_i n_up n_dn   (t=1, U=4)
  |psi(0)> = CDW |0, updn, 0, updn, ...>  (half-filling),  real-time evolve dt,
  2nd-order Trotter, truncate to chi_max (svd_min = 1e-12).

Each tool runs AT ITS OWN OPTIMUM on the same box (the fair "best vs best"):
  * TeNPy FermiHubbardChain, SINGLE-thread (its TEBD does not parallelize; more threads
    hurt it), conserving BOTH U(1)s (N and Sz) -- its default best.
  * QuantX qx_tebd_bench (d=4): native MULTI-thread OpenMP block-sparse SVDs, ALSO
    conserving both U(1)s via the packed charge q = N*K + 2Sz (QX_HUB_2U1=0 falls back
    to single-U(1) N -- the 5-6x-slower baseline that motivated the packing). QX-GPU
    additionally routes the few dominant SVD blocks per bond to the 4090.

The Jordan-Wigner statistics are handled inside each tool (QuantX: in the MPO;
TeNPy: in its fermionic site/coupling machinery) -- the timed computation is the
same fermionic quench.

Run (qxbench env, for tenpy):
  conda activate qxbench
  python benchmarks/hubbard_tebd_vs_tenpy.py
"""
from __future__ import annotations
import os
# TeNPy (in-process) single-thread -- its optimum. Set before numpy/tenpy import.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
_NCORE = str(os.cpu_count() or 8)

import json, re, subprocess, sys, time

DT       = float(os.environ.get("BENCH_DT", "0.05"))
REPS     = int(os.environ.get("BENCH_REPS", "2"))
COOLDOWN = float(os.environ.get("BENCH_COOLDOWN", "4"))
N        = int(os.environ.get("BENCH_N", "32"))
U        = 4.0                      # matches qx_tebd_bench's d=4 hard-wired U
CUTOFF   = 1e-12
PLAN = [(256, 40), (512, 40)]
if os.environ.get("BENCH_PLAN"):   # e.g. "256:40,512:40"
    PLAN = [(int(c), int(s)) for c, s in (p.split(":") for p in os.environ["BENCH_PLAN"].split(","))]

_here = os.path.dirname(__file__)
QX_BENCH = os.environ.get("QX_TEBD_BENCH",
                          os.path.join(_here, "..", "build-cuda-blas", "Release", "qx_tebd_bench.exe"))
QX_DLL_DIR = os.environ.get("QX_DLL_DIR", r"C:\ProgramData\miniforge3\envs\qxblas\Library\bin")


def _min_of(fn, reps):
    best = float("inf")
    for r in range(reps):
        best = min(best, fn())
        if r < reps - 1:
            time.sleep(COOLDOWN)
    return best


def tenpy_hubbard(chi, steps):
    from tenpy.models.hubbard import FermiHubbardChain
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import tebd
    M = FermiHubbardChain({"L": N, "t": 1.0, "U": U, "mu": 0, "V": 0,
                           "bc_MPS": "finite"})   # default conserves N AND Sz (its best)
    psi = MPS.from_lat_product_state(M.lat, [["empty"], ["full"]])   # CDW half-filling
    eng = tebd.TEBDEngine(psi, M, {"trunc_params": {"chi_max": chi, "svd_min": CUTOFF}, "order": 2})
    # ONE run_evolution call: TeNPy merges the interior 2nd-order Trotter half-layers
    # across steps (2N+1 layers instead of 3N) -- its legitimate optimum, ~1.5x faster
    # than a loop of 1-step calls. Nothing is measured mid-run on either side.
    t0 = time.perf_counter()
    eng.run_evolution(steps, DT)
    return time.perf_counter() - t0


def quantx(chi, steps, gpu):
    env = dict(os.environ)
    env["PATH"] = QX_DLL_DIR + os.pathsep + env.get("PATH", "")
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[_v] = _NCORE          # QuantX native multi-thread (its optimum)
    if gpu:
        env.pop("QX_GPU_MIN_BLK", None); env["QX_GPU_THETA_MINWORK"] = "999999999999"
    else:
        env["QX_GPU_MIN_BLK"] = "999999"; env["QX_GPU_THETA_MINWORK"] = "999999999999"
    out = subprocess.run([QX_BENCH, str(N), str(chi), str(steps), str(DT), "4"],
                         capture_output=True, text=True, env=env).stdout
    m = re.search(r"wall=\s*([\d.]+)\s*s", out)
    if not m:
        raise RuntimeError(f"qx_tebd_bench parse fail (chi={chi}):\n{out}")
    return float(m.group(1))


def main():
    if not os.path.exists(QX_BENCH):
        sys.exit(f"qx_tebd_bench not found at {QX_BENCH}")
    import tenpy
    print(f"Hubbard (d=4, U={U:g}) TEBD  N={N} dt={DT} cutoff={CUTOFF} reps={REPS}(min) | "
          f"tenpy {tenpy.__version__} (single-thread, N+Sz) | QuantX (multi-thread, N+Sz packed)\n",
          flush=True)
    hdr = f"{'chi':>5} {'steps':>6} {'TeNPy(s)':>10} {'QX-CPU(s)':>10} {'QX-GPU(s)':>10} {'CPU<TeNPy':>10}"
    print(hdr, flush=True); print("-" * len(hdr), flush=True)
    results = {"N": N, "dt": DT, "U": U, "reps": REPS, "runs": []}
    for chi, steps in PLAN:
        tt = _min_of(lambda: tenpy_hubbard(chi, steps), REPS)
        qc = _min_of(lambda: quantx(chi, steps, False), REPS)
        qg = _min_of(lambda: quantx(chi, steps, True), REPS)
        print(f"{chi:>5} {steps:>6} {tt:>10.2f} {qc:>10.2f} {qg:>10.2f} {tt/qc:>9.2f}x", flush=True)
        results["runs"].append({"chi": chi, "steps": steps, "tenpy": tt,
                                "qx_cpu": qc, "qx_gpu": qg, "cpu_vs_tenpy": tt/qc})
        time.sleep(COOLDOWN)
    out = os.path.join(_here, "results_hubbard_vs_tenpy.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
