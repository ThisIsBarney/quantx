"""M175 — Lindblad benchmark: QuantX sparse solver vs QuTiP.

Measures wall time and peak memory for Lindblad time evolution of a
1D Heisenberg chain with boundary amplitude damping:

    H = J Σ_{i} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    L_k = √γ σ⁻_k   for k = 0, n-1   (boundary dissipation)

Problem sizes: n = 6, 8, 10, 12, 14 qubits.

Outputs
-------
benchmarks/results/lindblad_scaling.json — raw timing/memory data
benchmarks/results/lindblad_scaling.png  — wall-time scaling plot

Usage
-----
# Full benchmark (requires QuTiP installed):
    python benchmarks/lindblad_vs_qutip.py

# QuantX only (no QuTiP required):
    python benchmarks/lindblad_vs_qutip.py --no-qutip

# Quick smoke test (n=6,8 only):
    python benchmarks/lindblad_vs_qutip.py --quick

References
----------
.. [1] Johansson et al. (2012). QuTiP. Comput. Phys. Commun. 183, 1760.
       DOI: 10.1016/j.cpc.2012.02.021
.. [2] Lindblad (1976). Commun. Math. Phys. 48, 119.
       DOI: 10.1007/BF01608499
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

# Ensure repo packages are importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Benchmark parameters ─────────────────────────────────────────────────────

J     = 1.0    # Heisenberg coupling
GAMMA = 0.1    # boundary damping rate
DT    = 0.05   # RK4 time step
T_TOT = 2.0    # total evolution time (short — we measure speed, not physics)

SIZES_FULL  = [6, 8, 10, 12, 14]
SIZES_QUICK = [6, 8]


# ── QuantX helpers ───────────────────────────────────────────────────────────

def build_heisenberg_ham(n: int) -> list[tuple[float, str]]:
    """Build Heisenberg XXX Hamiltonian as (coeff, indexed_pauli_label) list.

    Uses indexed format ("X0X1", "Y2Y3", ...) so both the dense C++ backend
    (which parses indexed labels) and the sparse backend (which converts them
    via _parse_pauli_label) receive the correct operator.

    Convention: full Pauli matrices (not spin-1/2 operators), matching QuTiP
    path which uses sigmax() = [[0,1],[1,0]] without the 1/2 factor.
    """
    terms = []
    for i in range(n - 1):
        for gate in ("X", "Y", "Z"):
            label = f"{gate}{i}{gate}{i+1}"   # e.g. "X0X1", "Y2Y3"
            terms.append((J, label))
    return terms


def build_boundary_jumps(n: int):
    from quantx_engine.contrib.lindblad import JumpOperator
    return [
        JumpOperator("amplitude_damping", qubit=0,     rate=GAMMA),
        JumpOperator("amplitude_damping", qubit=n - 1, rate=GAMMA),
    ]


def run_quantx(n: int) -> dict[str, Any]:
    """Run QuantX LindbladSolver and return timing + memory."""
    from quantx_engine import Engine
    from quantx_engine.contrib.lindblad import LindbladSolver

    ham  = build_heisenberg_ham(n)
    jops = build_boundary_jumps(n)

    tracemalloc.start()
    t0 = time.perf_counter()

    try:
        solver = LindbladSolver(Engine())
        result = solver.solve(
            n_qubits=n,
            hamiltonian_terms=ham,
            jump_operators=jops,
            total_time=T_TOT,
            dt=DT,
        )
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "n_qubits":    n,
            "tool":        "quantx",
            "wall_time_s": elapsed,
            "peak_mem_mb": peak / 1e6,
            "final_purity": result.final_purity,
            "n_steps":     len(result.frames) - 1,
            "error":       None,
        }
    except Exception as e:
        tracemalloc.stop()
        return {
            "n_qubits": n, "tool": "quantx",
            "wall_time_s": None, "peak_mem_mb": None,
            "final_purity": None, "n_steps": None,
            "error": str(e),
        }


# ── QuTiP helpers ────────────────────────────────────────────────────────────

def run_qutip(n: int) -> dict[str, Any]:
    """Run QuTiP mesolve and return timing + memory."""
    try:
        import qutip as qt
    except ImportError:
        return {
            "n_qubits": n, "tool": "qutip",
            "wall_time_s": None, "peak_mem_mb": None,
            "final_purity": None, "n_steps": None,
            "error": "QuTiP not installed",
        }

    try:
        d = 2 ** n
        sx = qt.sigmax()
        sy = qt.sigmay()
        sz = qt.sigmaz()
        sm = qt.sigmam()
        Id = qt.qeye(2)

        def embed(op, i, n):
            ops = [Id] * n
            ops[i] = op
            return qt.tensor(ops)

        # Hamiltonian
        H = sum(
            J * (embed(sx, i, n) * embed(sx, i + 1, n) +
                 embed(sy, i, n) * embed(sy, i + 1, n) +
                 embed(sz, i, n) * embed(sz, i + 1, n))
            for i in range(n - 1)
        )

        # Jump operators
        c_ops = [
            (GAMMA ** 0.5) * embed(sm, 0,     n),
            (GAMMA ** 0.5) * embed(sm, n - 1, n),
        ]

        # Initial state |0…0⟩
        psi0 = qt.tensor([qt.basis(2, 0)] * n)
        rho0 = psi0 * psi0.dag()

        times = [k * DT for k in range(int(T_TOT / DT) + 1)]

        tracemalloc.start()
        t0 = time.perf_counter()

        output = qt.mesolve(H, rho0, times, c_ops, [])

        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rho_final = output.states[-1]
        purity = float((rho_final * rho_final).tr().real)

        return {
            "n_qubits":    n,
            "tool":        "qutip",
            "wall_time_s": elapsed,
            "peak_mem_mb": peak / 1e6,
            "final_purity": purity,
            "n_steps":     len(times) - 1,
            "error":       None,
        }
    except Exception as e:
        tracemalloc.stop()
        return {
            "n_qubits": n, "tool": "qutip",
            "wall_time_s": None, "peak_mem_mb": None,
            "final_purity": None, "n_steps": None,
            "error": str(e),
        }


# ── Report ───────────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    header = f"{'n':>4}  {'tool':>10}  {'time (s)':>10}  {'mem (MB)':>10}  {'purity':>8}  {'note':}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        t  = f"{r['wall_time_s']:.3f}" if r["wall_time_s"] is not None else "OOM/ERR"
        m  = f"{r['peak_mem_mb']:.1f}"  if r["peak_mem_mb"] is not None else "—"
        p  = f"{r['final_purity']:.4f}" if r["final_purity"] is not None else "—"
        note = r["error"] or ""
        print(f"{r['n_qubits']:>4}  {r['tool']:>10}  {t:>10}  {m:>10}  {p:>8}  {note}")


def save_results(results: list[dict]) -> None:
    out = RESULTS_DIR / "lindblad_scaling.json"
    with open(out, "w") as f:
        json.dump({"benchmark": "lindblad_vs_qutip", "results": results}, f, indent=2)
    print(f"\nResults saved → {out}")


def plot_results(results: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Lindblad time evolution: QuantX sparse vs QuTiP\n"
                 f"1D Heisenberg + boundary damping, T={T_TOT}, dt={DT}")

    for tool, color, marker in [("quantx", "#1f77b4", "o"), ("qutip", "#ff7f0e", "s")]:
        rows = [r for r in results if r["tool"] == tool and r["wall_time_s"] is not None]
        if not rows:
            continue
        ns = [r["n_qubits"] for r in rows]
        ts = [r["wall_time_s"] for r in rows]
        ms = [r["peak_mem_mb"] for r in rows]
        axes[0].plot(ns, ts, marker=marker, color=color, label=tool, linewidth=2)
        axes[1].plot(ns, ms, marker=marker, color=color, label=tool, linewidth=2)

    axes[0].set_xlabel("n_qubits")
    axes[0].set_ylabel("Wall time (s)")
    axes[0].set_title("Wall time scaling")
    axes[0].legend()
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("n_qubits")
    axes[1].set_ylabel("Peak memory (MB)")
    axes[1].set_title("Memory scaling")
    axes[1].legend()
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    out = RESULTS_DIR / "lindblad_scaling.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Plot saved    → {out}")


# ── Speedup summary ──────────────────────────────────────────────────────────

def print_speedup(results: list[dict]) -> None:
    qx_map  = {r["n_qubits"]: r for r in results if r["tool"] == "quantx"}
    qt_map  = {r["n_qubits"]: r for r in results if r["tool"] == "qutip"}
    common  = sorted(set(qx_map) & set(qt_map))
    if not common:
        return
    print("\nSpeedup (QuTiP / QuantX wall time):")
    for n in common:
        qx_t = qx_map[n]["wall_time_s"]
        qt_t = qt_map[n]["wall_time_s"]
        if qx_t and qt_t:
            speedup = qt_t / qx_t
            print(f"  n={n:2d}: {speedup:6.1f}x  (QuantX {qx_t:.3f}s vs QuTiP {qt_t:.3f}s)")
        elif qx_t and not qt_t:
            print(f"  n={n:2d}: QuTiP OOM/ERR — QuantX {qx_t:.3f}s")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Lindblad benchmark: QuantX vs QuTiP")
    parser.add_argument("--no-qutip", action="store_true",
                        help="Skip QuTiP benchmark (QuantX only)")
    parser.add_argument("--quick",    action="store_true",
                        help=f"Run small sizes only: {SIZES_QUICK}")
    parser.add_argument("--sizes",    nargs="+", type=int,
                        help="Custom qubit sizes to benchmark")
    args = parser.parse_args()

    sizes = args.sizes or (SIZES_QUICK if args.quick else SIZES_FULL)
    run_qutip_flag = not args.no_qutip

    print("=" * 60)
    print("M175 Benchmark: Lindblad QuantX vs QuTiP")
    print(f"Problem: 1D Heisenberg + boundary damping, T={T_TOT}s, dt={DT}")
    print(f"Sizes:   {sizes}")
    print("=" * 60)

    results: list[dict] = []

    for n in sizes:
        print(f"\n[n={n}] Running QuantX ...", end=" ", flush=True)
        r = run_quantx(n)
        results.append(r)
        if r["error"]:
            print(f"ERROR: {r['error']}")
        else:
            print(f"{r['wall_time_s']:.3f}s  {r['peak_mem_mb']:.1f} MB")

        if run_qutip_flag:
            print(f"[n={n}] Running QuTiP  ...", end=" ", flush=True)
            r = run_qutip(n)
            results.append(r)
            if r["error"]:
                print(f"ERROR: {r['error']}")
            else:
                print(f"{r['wall_time_s']:.3f}s  {r['peak_mem_mb']:.1f} MB")

    print_table(results)
    print_speedup(results)
    save_results(results)
    plot_results(results)


if __name__ == "__main__":
    main()
