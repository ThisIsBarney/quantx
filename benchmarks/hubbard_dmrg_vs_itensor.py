"""Fermionic Hubbard GROUND-STATE DMRG: QuantX vs ITensor (Julia), best-vs-best.

Same problem as hubbard_dmrg_vs_tenpy.py: H = -t sum(c^dag c + h.c.) + U sum n_up n_dn,
t=1, U=4, OPEN chain, half filling, fixed chi_max (svd cutoff 1e-12), sweep until the
energy change per sweep < ~1e-10. Both conserve N AND Sz (ITensor "Electron" sites with
conserve_qns=true; QuantX packed two-U(1)). Converged energies must agree to ~1e-8.

Each tool at its own optimum:
  * ITensor (ITensorMPS.dmrg): JIT excluded via an in-process warm-up run (the Julia
    analogue of excluding Python import time). BLAS threads: BOTH 1 and all-cores are
    timed, the faster is reported -- measured, not guessed.
  * QuantX run_dmrg_blocks (qx_hubbard_showcase QX_HUB_BENCH mode): native multi-thread.

Run (needs julia + ITensors/ITensorMPS installed):
  python benchmarks/hubbard_dmrg_vs_itensor.py
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time
import _native                  # benchmarks/_native.py

U        = 4.0
N        = int(os.environ.get("BENCH_N", "64"))
REPS     = int(os.environ.get("BENCH_REPS", "2"))
COOLDOWN = float(os.environ.get("BENCH_COOLDOWN", "4"))
PLAN     = [128, 256]
if os.environ.get("BENCH_PLAN"):
    PLAN = [int(c) for c in os.environ["BENCH_PLAN"].split(",")]
_NCORE = str(os.cpu_count() or 8)

_here = os.path.dirname(__file__)
JULIA = os.environ.get("JULIA_EXE",
                       os.path.expandvars(r"%LOCALAPPDATA%\Programs\Julia-1.12.6\bin\julia.exe"))
JL_SCRIPT = os.path.join(_here, "itensor_hubbard_dmrg.jl")
QX_BENCH = None                 # found in main(): benchmarks/_native.py


def _min_of(fn, reps):
    best, e = float("inf"), None
    for r in range(reps):
        w, en = fn()
        if w < best: best, e = w, en
        if r < reps - 1: time.sleep(COOLDOWN)
    return best, e


def itensor(chi, nthreads):
    out = subprocess.run([JULIA, "--startup-file=no", JL_SCRIPT,
                          str(N), str(U), str(chi), str(nthreads)],
                         capture_output=True, text=True).stdout
    m = re.search(r"RESULT ([\d.eE+-]+) ([\d.eE+-]+) (\d+)", out)
    if not m:
        raise RuntimeError(f"itensor parse fail (chi={chi}, thr={nthreads}):\n{out[-2000:]}")
    return float(m.group(1)), float(m.group(2))


def quantx(chi):
    env = _native.env()             # QX_DLL_DIR first on PATH, if set
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[_v] = _NCORE
    env["QX_HUB_BENCH"] = f"{N} {chi}"; env["QX_HUB_U"] = str(U)
    out = subprocess.run([QX_BENCH], capture_output=True, text=True, env=env).stdout
    m = re.search(r"E=([\d.eE+-]+) wall=\s*([\d.]+)", out)
    if not m:
        raise RuntimeError(f"qx bench parse fail (chi={chi}):\n{out}")
    return float(m.group(2)), float(m.group(1))


def main():
    global QX_BENCH
    QX_BENCH = _native.program("qx_hubbard_showcase", "QX_HUB_SHOWCASE")
    for p, what in ((JULIA, "julia"),):
        if not os.path.exists(p):
            sys.exit(f"{what} not found at {p}")
    print(f"Hubbard GS DMRG  N={N} U={U:g} half-filling reps={REPS}(min) | "
          f"ITensor (JIT excluded, best of 1/{_NCORE} BLAS threads, N+Sz) | "
          f"QuantX (multi-thread, N+Sz packed)\n", flush=True)
    hdr = f"{'chi':>5} {'ITensor(s)':>11} {'thr':>4} {'QX(s)':>9} {'QX<IT':>8}  {'|dE|':>8}"
    print(hdr, flush=True); print("-" * len(hdr), flush=True)
    results = {"N": N, "U": U, "reps": REPS, "runs": []}
    for chi in PLAN:
        t1, e1 = _min_of(lambda: itensor(chi, 1), REPS)
        tm, em = _min_of(lambda: itensor(chi, int(_NCORE)), REPS)
        tt, te, thr = (t1, e1, 1) if t1 <= tm else (tm, em, int(_NCORE))
        qw, qe = _min_of(lambda: quantx(chi), REPS)
        dE = abs(qe - te)
        print(f"{chi:>5} {tt:>11.2f} {thr:>4} {qw:>9.2f} {tt/qw:>7.2f}x  {dE:>8.1e}", flush=True)
        results["runs"].append({"chi": chi, "itensor": tt, "itensor_blas_threads": thr,
                                "qx": qw, "ratio": tt/qw, "E_itensor": te, "E_qx": qe})
        time.sleep(COOLDOWN)
    out = os.path.join(_here, "results_hubbard_dmrg_vs_itensor.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
