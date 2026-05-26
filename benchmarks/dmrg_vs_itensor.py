"""M178 — DMRG benchmark: QuantX vs tenpy vs ITensor.

Measures wall time, peak memory, and energy error vs Bethe ansatz for the
1D Heisenberg chain at various system sizes (n) and bond dimensions (chi).

Backends
--------
* QuantX   — always run (native C++ when available, pure-Python fallback)
* tenpy    — run if `import tenpy` succeeds and ``--no-tenpy`` not given
* ITensor  -- run via Julia subprocess if ``julia`` is on PATH and a
              Project.toml with ITensors.jl is found (or can be instantiated)

Bethe-ansatz reference
----------------------
  E_∞ / n = J * (1/4 - ln 2) ≈ -0.4431 J   (n → ∞ limit)

Output
------
* benchmarks/results/dmrg_scaling.json  — machine-readable full results
* benchmarks/results/dmrg_scaling.png   — wall-time + energy-error plots

Usage
-----
  # Quick smoke (CI, n=8 and n=10, QuantX only)
  python benchmarks/dmrg_vs_itensor.py --sizes 8 10 --chi 16 32 --no-tenpy

  # Full benchmark (workflow_dispatch or local)
  python benchmarks/dmrg_vs_itensor.py --sizes 20 50 100 --chi 50 100 200
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

# Bethe ansatz for the SPIN-1/2 convention: E/n = J(1/4 - ln2) ≈ -0.4431 J
# QuantX uses FULL Pauli operators (X,Y,Z with eigenvalues ±1), which are
# twice the spin-1/2 operators, so the Hamiltonian is 4× larger:
#   H_QuantX = Σ_i (Xi Xi+1 + Yi Yi+1 + Zi Zi+1) = 4 * H_spin½
# Reference energy per site in the Pauli convention:
BETHE_E_PER_SITE = 4.0 * (0.25 - math.log(2))   # ≈ -1.7726 (J=1, Pauli)

RESULTS_DIR = Path(__file__).parent / "results"

# ── helpers ──────────────────────────────────────────────────────────────────

def _mem_mib(peak_bytes: int) -> float:
    return round(peak_bytes / (1024 ** 2), 3)


def _energy_error(energy: float, n: int) -> float:
    """Relative error vs Bethe ansatz (per site)."""
    ref = BETHE_E_PER_SITE * n
    return abs(energy - ref) / abs(ref)


def _row(backend: str, n: int, chi: int, energy: float,
         wall_s: float, peak_mib: float, note: str = "") -> dict:
    return {
        "backend":    backend,
        "n":          n,
        "chi":        chi,
        "energy":     round(energy, 10),
        "wall_s":     round(wall_s, 4),
        "peak_mib":   peak_mib,
        "bethe_ref":  round(BETHE_E_PER_SITE * n, 10),
        "energy_err": round(_energy_error(energy, n), 6),
        "note":       note,
    }


# ── QuantX backend ────────────────────────────────────────────────────────────

class _NoNativeEngine:
    """Minimal stub that makes DMRGSolver use its pure-Python fallback."""
    _lib = None


def run_quantx(n: int, chi: int, n_sweeps: int = 20) -> dict:
    from quantx_engine.cached_dmrg import CachedDMRG

    # Use the native Engine when the shared library is available; otherwise
    # fall back to the pure-Python exact-diag path (viable for n ≤ 20).
    try:
        from quantx_engine import Engine
        engine = Engine()
    except Exception:
        engine = _NoNativeEngine()  # type: ignore[assignment]

    dmrg = CachedDMRG(engine)

    tracemalloc.start()
    t0 = time.perf_counter()
    result = dmrg.solve(
        n_qubits=n,
        hamiltonian_type="heisenberg",
        J=1.0,
        max_bond=chi,
        n_sweeps=n_sweeps,
        tol=1e-10,
    )
    wall_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    backend_tag = f"quantx/{result.backend}"
    return _row(backend_tag, n, chi, result.energy, wall_s, _mem_mib(peak))


# ── tenpy backend ─────────────────────────────────────────────────────────────

def run_tenpy(n: int, chi: int, n_sweeps: int = 20) -> dict:
    import tenpy
    from tenpy.networks.mps import MPS
    from tenpy.models.xxz_chain import XXZChain2
    from tenpy.algorithms import dmrg as tenpy_dmrg

    # XXZChain2 uses spin-1/2 operators (Sx=σx/2, etc.), so the coupling must
    # be scaled by 4 to match QuantX's full-Pauli convention
    # (Xi Xi+1 + Yi Yi+1 + Zi Zi+1 where X,Y,Z have eigenvalues ±1).
    # Without the factor, H_tenpy(J=1) = H_QuantX(J=1) / 4.
    J_tenpy = 4.0  # = J_pauli (1.0) * 4
    model_params = {
        "L":        n,
        "Jxx":      J_tenpy,
        "Jz":       J_tenpy,
        "hz":       0.0,
        "bc_MPS":   "finite",
        "conserve": "Sz",
        "sort_charge": True,
    }
    model = XXZChain2(model_params)

    product_state = ["up", "down"] * (n // 2)
    psi = MPS.from_lat_product_state(model.lat, product_state)

    dmrg_params = {
        "trunc_params": {"chi_max": chi, "svd_min": 1e-10},
        "N_sweeps_check": 1,
        "mixer":     None,
        "active_sites": 2,
    }

    tracemalloc.start()
    t0 = time.perf_counter()
    eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    # run up to n_sweeps
    for _ in range(n_sweeps):
        if eng.sweep():
            break
    energy = model.bond_energies(psi).sum()
    wall_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return _row("tenpy", n, chi, float(energy), wall_s, _mem_mib(peak))


# ── ITensor (Julia) backend ───────────────────────────────────────────────────

_JULIA_SCRIPT = """\
using ITensors, ITensors.HDF5
using LinearAlgebra, Printf, JSON

function heisenberg_dmrg(n::Int, chi::Int, nsweeps::Int=20)
    sites = siteinds("S=1/2", n)

    # Build Heisenberg MPO (full Pauli, not spin-1/2 normalisation)
    os = OpSum()
    for j in 1:n-1
        os += 1.0,"Sx",j,"Sx",j+1
        os += 1.0,"Sy",j,"Sy",j+1
        os += 1.0,"Sz",j,"Sz",j+1
    end
    H = MPO(4 .* os, sites)   # factor 4: ITensor uses S=1/2, we want Pauli

    # Initial state: Neel
    state = [isodd(i) ? "Up" : "Dn" for i in 1:n]
    psi0 = MPS(sites, state)

    sweeps = Sweeps(nsweeps)
    maxdim!(sweeps, chi)
    cutoff!(sweeps, 1e-10)

    energy, psi = dmrg(H, psi0, sweeps; outputlevel=0)
    return energy
end

n   = parse(Int, ARGS[1])
chi = parse(Int, ARGS[2])
nsw = parse(Int, ARGS[3])

t0  = time()
E   = heisenberg_dmrg(n, chi, nsw)
dt  = time() - t0

println(JSON.json(Dict("energy" => E, "wall_s" => dt)))
"""


def run_itensor(n: int, chi: int, n_sweeps: int = 20) -> dict:
    """Run ITensor DMRG via Julia subprocess."""
    import shutil
    import tempfile

    julia = shutil.which("julia")
    if julia is None:
        raise RuntimeError("julia not on PATH")

    with tempfile.NamedTemporaryFile(suffix=".jl", mode="w",
                                     delete=False) as fh:
        fh.write(_JULIA_SCRIPT)
        script = fh.name

    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [julia, "--project=@.", script, str(n), str(chi), str(n_sweeps)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        wall_s = time.perf_counter() - t0

        if proc.returncode != 0:
            raise RuntimeError(
                f"Julia exited {proc.returncode}:\n{proc.stderr[-2000:]}"
            )

        # Last non-empty line is the JSON payload
        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        data = json.loads(lines[-1])
        energy = float(data["energy"])
        julia_wall = float(data.get("wall_s", wall_s))

    finally:
        os.unlink(script)

    return _row("itensor", n, chi, energy, julia_wall, 0.0,
                note="memory not measured for Julia subprocess")


# ── plotting ──────────────────────────────────────────────────────────────────

def make_plots(rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed — skipping plot")
        return

    backends = sorted({r["backend"] for r in rows})
    chi_vals = sorted({r["chi"] for r in rows})
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    bc_map = {b: colors[i % len(colors)] for i, b in enumerate(backends)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- wall time vs n ---
    ax = axes[0]
    for chi in chi_vals:
        for backend in backends:
            pts = [(r["n"], r["wall_s"]) for r in rows
                   if r["backend"] == backend and r["chi"] == chi]
            if pts:
                pts.sort()
                xs, ys = zip(*pts)
                lbl = f"{backend} χ={chi}"
                ax.plot(xs, ys, "o-", color=bc_map[backend],
                        linestyle="--" if chi == chi_vals[-1] else "-",
                        label=lbl)
    ax.set_xlabel("n (qubits)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("DMRG wall time vs system size")
    ax.legend(fontsize=7, ncol=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # --- energy error vs n ---
    ax = axes[1]
    for chi in chi_vals:
        for backend in backends:
            pts = [(r["n"], r["energy_err"]) for r in rows
                   if r["backend"] == backend and r["chi"] == chi]
            if pts:
                pts.sort()
                xs, ys = zip(*pts)
                lbl = f"{backend} χ={chi}"
                ax.plot(xs, ys, "s-", color=bc_map[backend],
                        linestyle="--" if chi == chi_vals[-1] else "-",
                        label=lbl)
    ax.set_xlabel("n (qubits)")
    ax.set_ylabel("|E - E_Bethe| / |E_Bethe|")
    ax.set_title("Energy error vs Bethe ansatz")
    ax.legend(fontsize=7, ncol=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes",    nargs="+", type=int,
                   default=[20, 50, 100],
                   help="System sizes (n_qubits) to benchmark")
    p.add_argument("--chi",      nargs="+", type=int,
                   default=[50, 100, 200],
                   help="Bond dimensions to sweep")
    p.add_argument("--n-sweeps", type=int, default=20,
                   help="Number of DMRG sweeps per run")
    p.add_argument("--no-tenpy", action="store_true",
                   help="Skip tenpy even if installed")
    p.add_argument("--no-itensor", action="store_true",
                   help="Skip ITensor even if julia is available")
    p.add_argument("--out-dir",  type=Path,
                   default=RESULTS_DIR,
                   help="Directory for JSON + PNG output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    errors: list[str] = []

    # Determine which optional backends are available
    have_tenpy = False
    if not args.no_tenpy:
        try:
            import tenpy  # noqa: F401
            have_tenpy = True
            print("[info] tenpy available")
        except ImportError:
            print("[info] tenpy not installed — skipping")

    import shutil
    have_itensor = (not args.no_itensor) and (shutil.which("julia") is not None)
    if have_itensor:
        print("[info] julia found — ITensor backend enabled")
    elif not args.no_itensor:
        print("[info] julia not on PATH — skipping ITensor")

    header = (
        f"{'backend':<20} {'n':>5} {'chi':>6} "
        f"{'energy':>14} {'wall_s':>10} {'peak_MiB':>10} {'err%':>8}"
    )
    print(header)
    print("-" * len(header))

    for n in args.sizes:
        for chi in args.chi:
            # QuantX
            try:
                r = run_quantx(n, chi, args.n_sweeps)
                rows.append(r)
                print(
                    f"{r['backend']:<20} {n:>5} {chi:>6} "
                    f"{r['energy']:>14.6f} {r['wall_s']:>10.3f} "
                    f"{r['peak_mib']:>10.1f} {r['energy_err']*100:>7.3f}%"
                )
            except Exception as exc:
                msg = f"[error] quantx n={n} chi={chi}: {exc}"
                print(msg)
                errors.append(msg)

            # tenpy
            if have_tenpy:
                try:
                    r = run_tenpy(n, chi, args.n_sweeps)
                    rows.append(r)
                    print(
                        f"{r['backend']:<20} {n:>5} {chi:>6} "
                        f"{r['energy']:>14.6f} {r['wall_s']:>10.3f} "
                        f"{r['peak_mib']:>10.1f} {r['energy_err']*100:>7.3f}%"
                    )
                except Exception as exc:
                    msg = f"[error] tenpy n={n} chi={chi}: {exc}"
                    print(msg)
                    errors.append(msg)

            # ITensor
            if have_itensor:
                try:
                    r = run_itensor(n, chi, args.n_sweeps)
                    rows.append(r)
                    print(
                        f"{r['backend']:<20} {n:>5} {chi:>6} "
                        f"{r['energy']:>14.6f} {r['wall_s']:>10.3f} "
                        f"{'n/a':>10} {r['energy_err']*100:>7.3f}%"
                    )
                except Exception as exc:
                    msg = f"[error] itensor n={n} chi={chi}: {exc}"
                    print(msg)
                    errors.append(msg)

    # Write JSON
    json_path = args.out_dir / "dmrg_scaling.json"
    payload = {
        "meta": {
            "date":       time.strftime("%Y-%m-%d"),
            "bethe_ref":  BETHE_E_PER_SITE,
            "sizes":      args.sizes,
            "chi":        args.chi,
            "n_sweeps":   args.n_sweeps,
        },
        "results": rows,
        "errors":  errors,
    }
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[out] {json_path}")

    # Write PNG
    png_path = args.out_dir / "dmrg_scaling.png"
    if rows:
        make_plots(rows, png_path)

    if errors:
        print(f"\n[warn] {len(errors)} error(s) during benchmark — check JSON.")

    # Exit non-zero only if ALL backends failed for ALL configs
    if rows:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
