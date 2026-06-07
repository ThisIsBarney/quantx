"""QuantX-GPU vs TeNPy-CPU: real-time Neel-quench TEBD, margin vs bond dimension chi.

The plan's headline deliverable (docs/gpu_acceleration_plan.md): show the QuantX
advantage *grows with chi* against the real opponent. QuantX side is timed via the
native qx_tebd_bench.exe (CUDA build; no Python/DLL plumbing); TeNPy side is timed
in-process here. Both run the *same* physics:

  H = sum_i sigma_i . sigma_{i+1}   (Heisenberg, Pauli convention)
  |psi(0)> = Neel,  real-time evolve dt, n_steps, 2nd-order Trotter, truncate to chi.

TeNPy's SpinSite uses S = sigma/2, so Jx=Jy=Jz=4 reproduces sigma.sigma (matches the
existing benchmarks). Single-thread CPU for TeNPy (its optimal config); GPU for QuantX.

Run (qxbench env):
  C:\\ProgramData\\miniforge3\\envs\\qxbench\\python.exe benchmarks/gpu_vs_tenpy_quench.py
"""
from __future__ import annotations
import os
# pin single-thread BEFORE numpy/tenpy import (fair: TeNPy's optimal config).
# Assign unconditionally -- setdefault would leave a caller's pre-set OMP_NUM_THREADS
# in force while we still report threads=1, mislabeling the baseline.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import re
import subprocess
import sys
import time

N       = int(os.environ.get("BENCH_N", "32"))
DT      = float(os.environ.get("BENCH_DT", "0.05"))
STEPS   = int(os.environ.get("BENCH_STEPS", "60"))
CHIS    = [int(x) for x in os.environ.get("BENCH_CHIS", "256,512,1024").split(",")]
CUTOFF  = 1e-12
QX_BENCH = os.environ.get(
    "QX_TEBD_BENCH",
    os.path.join(os.path.dirname(__file__), "..", "build-cuda", "Release", "qx_tebd_bench.exe"))


def tenpy_quench(n, dt, n_steps, chi):
    import numpy as np  # noqa
    from tenpy.models.spins import SpinChain
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import tebd
    M = SpinChain({"S": 0.5, "L": n, "Jx": 4, "Jy": 4, "Jz": 4, "hz": 0,
                   "bc_MPS": "finite", "conserve": "Sz"})
    psi = MPS.from_lat_product_state(M.lat, [["up"], ["down"]])   # Neel
    eng = tebd.TEBDEngine(psi, M, {"trunc_params": {"chi_max": chi, "svd_min": CUTOFF},
                                   "order": 2})
    t0 = time.perf_counter()
    for _ in range(n_steps):
        eng.run_evolution(1, dt)
    return time.perf_counter() - t0, max(psi.chi)


def quantx_gpu(n, dt, n_steps, chi, fp32=False):
    env = dict(os.environ)
    env["QX_GPU_MIN_BLK"] = "128"
    env["QX_GPU_THETA_MINWORK"] = "2097152"
    env["QX_GPU_FP32"] = "1" if fp32 else "0"
    env.pop("QX_TEBD_PROFILE", None)
    out = subprocess.run([QX_BENCH, str(n), str(chi), str(n_steps), str(dt)],
                         capture_output=True, text=True, env=env).stdout
    m = re.search(r"wall=\s*([\d.]+)\s*s", out)
    b = re.search(r"max_bond=(\d+)", out)
    if not m:
        raise RuntimeError(f"could not parse qx_tebd_bench output:\n{out}")
    return float(m.group(1)), (int(b.group(1)) if b else 0)


def main():
    if not os.path.exists(QX_BENCH):
        sys.exit(f"qx_tebd_bench not found at {QX_BENCH} (set QX_TEBD_BENCH or build it)")
    import tenpy
    print(f"N={N} dt={DT} steps={STEPS} cutoff={CUTOFF} | tenpy {tenpy.__version__} | threads=1\n")
    hdr = f"{'chi':>5} {'TeNPy-CPU(s)':>13} {'QX-GPU(s)':>10} {'QX-GPU-FP32(s)':>15} {'GPU/TeNPy':>10} {'FP32/TeNPy':>11}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for chi in CHIS:
        qg, qb = quantx_gpu(N, DT, STEPS, chi, fp32=False)
        qf, _  = quantx_gpu(N, DT, STEPS, chi, fp32=True)
        tt, tb = tenpy_quench(N, DT, STEPS, chi)
        print(f"{chi:>5} {tt:>13.2f} {qg:>10.2f} {qf:>15.2f} {tt/qg:>9.2f}x {tt/qf:>10.2f}x")
        rows.append((chi, tt, qg, qf, qb, tb))
    print("\n(QX-GPU = Phase A+B FP64; FP32 = opt-in Phase C. bonds: "
          + ", ".join(f"chi={r[0]}:QX{r[4]}/TeNPy{r[5]}" for r in rows) + ")")


if __name__ == "__main__":
    main()
