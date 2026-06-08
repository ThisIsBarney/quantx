"""QuantX-GPU vs TeNPy-CPU: real-time Neel-quench TEBD, margin vs bond dimension chi.

The plan's headline deliverable (docs/gpu_acceleration_plan.md §7): show the QuantX
advantage *grows with chi* against the real opponent. QuantX side is timed via the
native qx_tebd_bench.exe (CUDA + OpenBLAS build; no Python/DLL plumbing); TeNPy side
is timed in-process. Both run the *same* physics:

  H = sum_i sigma_i . sigma_{i+1}   (Heisenberg, Pauli convention)
  |psi(0)> = Neel,  real-time evolve dt, n_steps, 2nd-order Trotter, truncate to chi.

TeNPy's SpinSite uses S = sigma/2, so Jx=Jy=Jz=4 reproduces sigma.sigma. Single-thread
CPU for TeNPy (its optimal config; here backed by Intel MKL). QuantX uses the GPU for
the large per-Sz-sector SVD + theta build, and OpenBLAS (single-thread) for the small
sub-threshold blocks.

Reproducibility on a consumer box (the absolute times — and the single-thread-CPU vs
GPU ratio — are thermally sensitive without controls):
  * lock GPU clocks first:   nvidia-smi -lgc 2100,2100   (reset after: nvidia-smi -rgc)
  * min-of-N trials per cell (least-throttled = the hardware's true capability)
  * cooldown between cells

Run (qxbench env, activated so MKL/OpenBLAS DLLs load):
  conda activate qxbench
  python benchmarks/gpu_vs_tenpy_quench.py
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"   # single-thread CPU baseline (assigned, not setdefault)

import json
import re
import subprocess
import sys
import time

DT       = float(os.environ.get("BENCH_DT", "0.05"))
STEPS    = int(os.environ.get("BENCH_STEPS", "50"))
REPS     = int(os.environ.get("BENCH_REPS", "3"))
COOLDOWN = float(os.environ.get("BENCH_COOLDOWN", "5"))
CUTOFF   = 1e-12
# per-N chi plan; TeNPy single-thread at large chi for N=64 is the long pole, so cap it.
PLAN = {32: [128, 256, 512, 1024], 64: [128, 256, 512]}
if os.environ.get("BENCH_PLAN"):   # e.g. "32:256,512,1024;64:512"
    PLAN = {int(n): [int(c) for c in cs.split(",")]
            for n, cs in (p.split(":") for p in os.environ["BENCH_PLAN"].split(";"))}

_here = os.path.dirname(__file__)
QX_BENCH = os.environ.get("QX_TEBD_BENCH",
                          os.path.join(_here, "..", "build-cuda-blas", "Release", "qx_tebd_bench.exe"))
QX_DLL_DIR = os.environ.get("QX_DLL_DIR",
                            r"C:\ProgramData\miniforge3\envs\qxblas\Library\bin")  # openblas.dll + deps


def _min_of(fn, reps):
    best, meta = float("inf"), None
    for r in range(reps):
        t, m = fn()
        if t < best:
            best, meta = t, m
        if r < reps - 1:
            time.sleep(COOLDOWN)
    return best, meta


def tenpy_quench(n, dt, n_steps, chi):
    from tenpy.models.spins import SpinChain
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import tebd
    M = SpinChain({"S": 0.5, "L": n, "Jx": 4, "Jy": 4, "Jz": 4, "hz": 0,
                   "bc_MPS": "finite", "conserve": "Sz"})
    psi = MPS.from_lat_product_state(M.lat, [["up"], ["down"]])
    eng = tebd.TEBDEngine(psi, M, {"trunc_params": {"chi_max": chi, "svd_min": CUTOFF},
                                   "order": 2})
    t0 = time.perf_counter()
    for _ in range(n_steps):
        eng.run_evolution(1, dt)
    return time.perf_counter() - t0, max(psi.chi)


def quantx_gpu(n, dt, n_steps, chi, fp32=False):
    env = dict(os.environ)
    env["PATH"] = QX_DLL_DIR + os.pathsep + env.get("PATH", "")   # openblas.dll + CUDA on PATH
    env["QX_GPU_MIN_BLK"] = "128"
    env["QX_GPU_THETA_MINWORK"] = "2097152"
    env["QX_GPU_FP32"] = "1" if fp32 else "0"
    env.pop("QX_TEBD_PROFILE", None)
    out = subprocess.run([QX_BENCH, str(n), str(chi), str(n_steps), str(dt)],
                         capture_output=True, text=True, env=env).stdout
    m = re.search(r"wall=\s*([\d.]+)\s*s", out)
    b = re.search(r"max_bond=(\d+)", out)
    if not m:
        raise RuntimeError(f"qx_tebd_bench parse fail (n={n} chi={chi}):\n{out}")
    return float(m.group(1)), (int(b.group(1)) if b else 0)


def main():
    if not os.path.exists(QX_BENCH):
        sys.exit(f"qx_tebd_bench not found at {QX_BENCH} (set QX_TEBD_BENCH or build build-cuda-blas)")
    import tenpy
    print(f"dt={DT} steps={STEPS} cutoff={CUTOFF} reps={REPS}(min-of-N) cooldown={COOLDOWN}s | "
          f"tenpy {tenpy.__version__} | CPU threads=1 | GPU boost (min-of-N picks best trial)\n")
    results = {"dt": DT, "steps": STEPS, "reps": REPS, "runs": []}
    for n in sorted(PLAN):
        hdr = f"N={n}: {'chi':>5} {'TeNPy(s)':>10} {'QX-GPU(s)':>10} {'QX-FP32(s)':>11} {'GPU/TeNPy':>10} {'FP32/TeNPy':>11}"
        print(hdr); print("-" * len(hdr))
        for chi in PLAN[n]:
            tt, _ = _min_of(lambda: tenpy_quench(n, DT, STEPS, chi), REPS)
            qg, _ = _min_of(lambda: quantx_gpu(n, DT, STEPS, chi, False), REPS)
            qf, _ = _min_of(lambda: quantx_gpu(n, DT, STEPS, chi, True), REPS)
            print(f"     {chi:>5} {tt:>10.2f} {qg:>10.2f} {qf:>11.2f} {tt/qg:>9.2f}x {tt/qf:>10.2f}x")
            results["runs"].append({"N": n, "chi": chi, "tenpy": tt, "qx_gpu": qg,
                                    "qx_fp32": qf, "gpu_vs_tenpy": tt/qg, "fp32_vs_tenpy": tt/qf})
            time.sleep(COOLDOWN)
        print()
    out = os.path.join(_here, "results_gpu_vs_tenpy.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
