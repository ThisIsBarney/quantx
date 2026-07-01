"""Spin-1 (d=3) real-time TEBD: QuantX vs TeNPy, the "are we leading?" benchmark.

Same physics on both sides:
  H = J sum_i S_i . S_{i+1}   (spin-1, physical S operators, J=1)
  |psi(0)> = Neel |+1 -1 ...>,  real-time evolve dt, n_steps, 2nd-order Trotter,
  truncate to chi_max (svd_min = 1e-12).  Both conserve Sz (block-sparse).

Each tool is run AT ITS OWN OPTIMUM on the same box (the fair "best vs best"):
  * TeNPy SpinChain S=1, Jx=Jy=Jz=1 == sum S.S. SINGLE-thread CPU -- its optimum, since
    TeNPy's TEBD does not parallelize (MKL multi-thread *hurts* it ~3x: oversubscription
    on the many small per-Sz-sector SVDs).
  * QuantX qx_tebd_bench (d=3): native MULTI-thread -- its OpenMP block-sparse SVD
    parallelism is exactly what scales (QX-CPU). QX-GPU additionally routes the FEW
    dominant SVD blocks per bond to the 4090 (a d-aware crossover, 256 for spin-1) so it
    OVERLAPS the CPU swarm -- the big win at large chi. (An earlier version flooded the
    device at min_blk=64 and lost; only the dominant-block routing wins for the
    fragmented spin-1 bond.)

Honest finding (this box, N=32): CPU-vs-CPU QuantX leads TeNPy's best by 1.30-1.46x;
with the GPU (a capability TeNPy's TEBD lacks) ~2.0x at chi=512, widening with chi. The
conserved energy matches to 9 digits across all three, so it is one computation timed
three ways. (Give TeNPy all 20 cores and it degrades ~3x -- meaningless, so we don't.)

Run (qxbench env, for tenpy + MKL):
  conda activate qxbench
  python benchmarks/spin1_tebd_vs_tenpy.py
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
CUTOFF   = 1e-12
# (chi, steps): steps chosen so the real-time bond grows to fill chi_max.
PLAN = [(256, 40), (512, 50)]
if os.environ.get("BENCH_PLAN"):   # e.g. "256:40,512:50,768:60"
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


def tenpy_spin1(chi, steps):
    from tenpy.models.spins import SpinChain
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import tebd
    M = SpinChain({"S": 1.0, "L": N, "Jx": 1, "Jy": 1, "Jz": 1, "hz": 0,
                   "bc_MPS": "finite", "conserve": "Sz"})
    psi = MPS.from_lat_product_state(M.lat, [["up"], ["down"]])
    eng = tebd.TEBDEngine(psi, M, {"trunc_params": {"chi_max": chi, "svd_min": CUTOFF}, "order": 2})
    t0 = time.perf_counter()
    for _ in range(steps):
        eng.run_evolution(1, DT)
    return time.perf_counter() - t0


def quantx(chi, steps, gpu):
    env = dict(os.environ)
    env["PATH"] = QX_DLL_DIR + os.pathsep + env.get("PATH", "")
    # QuantX runs in its native MULTI-thread config (its optimum) -- undo the
    # single-thread caps the parent set for TeNPy.
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[_v] = _NCORE
    if gpu:
        # Route only the FEW dominant SVD blocks per bond to the GPU (the d-aware
        # default crossover, 256 for d=3), so the 4090 overlaps the CPU swarm instead
        # of serializing many medium blocks on one device. Theta stays on the CPU
        # (its GPU crossover is higher for the fragmented spin-1 bond).
        env.pop("QX_GPU_MIN_BLK", None); env["QX_GPU_THETA_MINWORK"] = "999999999999"
    else:
        env["QX_GPU_MIN_BLK"] = "999999"; env["QX_GPU_THETA_MINWORK"] = "999999999999"
    out = subprocess.run([QX_BENCH, str(N), str(chi), str(steps), str(DT), "3"],
                         capture_output=True, text=True, env=env).stdout
    m = re.search(r"wall=\s*([\d.]+)\s*s", out)
    if not m:
        raise RuntimeError(f"qx_tebd_bench parse fail (chi={chi}):\n{out}")
    return float(m.group(1))


def main():
    if not os.path.exists(QX_BENCH):
        sys.exit(f"qx_tebd_bench not found at {QX_BENCH}")
    import tenpy
    print(f"spin-1 (d=3) TEBD  N={N} dt={DT} cutoff={CUTOFF} reps={REPS}(min) | "
          f"tenpy {tenpy.__version__} | CPU threads=1\n")
    hdr = f"{'chi':>5} {'steps':>6} {'TeNPy(s)':>10} {'QX-CPU(s)':>10} {'QX-GPU(s)':>10} {'CPU<TeNPy':>10}"
    print(hdr); print("-" * len(hdr))
    results = {"N": N, "dt": DT, "reps": REPS, "runs": []}
    for chi, steps in PLAN:
        tt = _min_of(lambda: tenpy_spin1(chi, steps), REPS)
        qc = _min_of(lambda: quantx(chi, steps, False), REPS)
        qg = _min_of(lambda: quantx(chi, steps, True), REPS)
        print(f"{chi:>5} {steps:>6} {tt:>10.2f} {qc:>10.2f} {qg:>10.2f} {tt/qc:>9.2f}x")
        results["runs"].append({"chi": chi, "steps": steps, "tenpy": tt,
                                "qx_cpu": qc, "qx_gpu": qg, "cpu_vs_tenpy": tt/qc})
        time.sleep(COOLDOWN)
    out = os.path.join(_here, "results_spin1_vs_tenpy.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
