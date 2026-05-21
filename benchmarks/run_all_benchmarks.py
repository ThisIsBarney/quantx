"""Master benchmark runner — runs all 5 QuantX benchmark suites and produces
a summary in Markdown (BENCHMARKS.md) and a combined JSON report.

Usage:
    python benchmarks/run_all_benchmarks.py                 # all benchmarks
    python benchmarks/run_all_benchmarks.py --suite lanczos # single suite
    python benchmarks/run_all_benchmarks.py --quick         # reduced sizes

The runner is intentionally stand-alone so it can be invoked from CI
without installing any build tools.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.dirname(__file__))

from bench_utils import BenchResult, hardware_info, print_table, save_results


# ---------------------------------------------------------------------------
# Per-suite entry points
# ---------------------------------------------------------------------------

def run_lanczos(quick: bool) -> list[BenchResult]:
    import bench_lanczos
    sizes = [4, 6] if quick else bench_lanczos.SIZES
    return bench_lanczos.run(sizes)


def run_trotter(quick: bool) -> list[BenchResult]:
    import bench_trotter
    sizes = [4, 6] if quick else bench_trotter.SIZES
    return bench_trotter.run(sizes)


def run_vqe(quick: bool) -> list[BenchResult]:
    import bench_vqe
    return bench_vqe.run()


def run_qec(quick: bool) -> list[BenchResult]:
    import bench_qec
    distances = [3, 5] if quick else bench_qec.DISTANCES
    p_physical = [0.001, 0.05] if quick else bench_qec.P_PHYSICAL
    n_shots = 100 if quick else bench_qec.N_SHOTS
    return bench_qec.run(distances, p_physical, n_shots)


def run_mps(quick: bool) -> list[BenchResult]:
    import bench_mps
    sizes = [4, 8, 12] if quick else bench_mps.SIZES
    return bench_mps.run(sizes)


SUITES = {
    "lanczos": ("Benchmark 1: Heisenberg Lanczos eigensolver",  run_lanczos),
    "trotter": ("Benchmark 2: Ising Trotter time evolution",    run_trotter),
    "vqe":     ("Benchmark 3: VQE variational optimisation",    run_vqe),
    "qec":     ("Benchmark 4: QEC logical error rate",          run_qec),
    "mps":     ("Benchmark 5: MPS GHZ fidelity",               run_mps),
}


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def _result_rows(results: list[BenchResult]) -> list[str]:
    rows = []
    for r in results:
        params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
        if r.skipped:
            status = f"SKIP ({r.skip_reason})"
            row = f"| {r.benchmark} | {r.backend} | {params_str} | — | {status} |"
        elif r.error:
            row = f"| {r.benchmark} | {r.backend} | {params_str} | ERR | {r.error[:60]} |"
        else:
            val = f"{r.result_value:.4f}" if isinstance(r.result_value, float) else str(r.result_value)
            row = f"| {r.benchmark} | {r.backend} | {params_str} | {r.wall_time_s:.3f} | {val} |"
        rows.append(row)
    return rows


def generate_markdown(
    all_results: dict[str, list[BenchResult]],
    hw: dict,
    output_path: str,
) -> None:
    lines: list[str] = []
    lines.append("# QuantX Engine — Benchmark Report\n")
    lines.append(f"_Generated on {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}_\n")

    lines.append("## Hardware / Software Environment\n")
    lines.append("| Key | Value |")
    lines.append("|-----|-------|")
    for k, v in hw.items():
        lines.append(f"| {k} | {v or '—'} |")
    lines.append("")

    lines.append("## Methodology\n")
    lines.append(
        "Wall-clock time is measured with `time.perf_counter()` around the **hot path\n"
        "only** — matrix construction, I/O, and library imports are excluded.\n"
        "Each benchmark runs a single trial (no warm-up) to reflect cold-start latency\n"
        "typical in scientific workflows.  Competitor libraries (SciPy, QuTiP,\n"
        "PennyLane, Stim, TeNPy) are imported at runtime; if a library is not installed\n"
        "the row is marked **SKIP** and QuantX timings are still reported.\n"
        "\n"
        "To reproduce:\n"
        "```bash\n"
        "# full suite\n"
        "python benchmarks/run_all_benchmarks.py\n"
        "\n"
        "# single suite\n"
        "python benchmarks/run_all_benchmarks.py --suite lanczos\n"
        "\n"
        "# smoke test (small sizes)\n"
        "python benchmarks/run_all_benchmarks.py --quick\n"
        "```\n"
    )

    # Per-suite descriptions
    suite_notes = {
        "lanczos": (
            "Ground-state energy of a 1D Heisenberg chain computed with\n"
            "Lanczos (iterative sparse diagonalisation, k=1 eigenvalue).\n"
            "Result column shows E₀ (lower is more negative for antiferromagnet).\n"
        ),
        "trotter": (
            "Norm-preserving second-order Trotter evolution of a transverse-field\n"
            "Ising chain from t=0 to t=1 with n\\_steps=100.  Result column shows\n"
            "‖ψ(t=1)‖, which must equal 1.0 (unitarity check).\n"
        ),
        "vqe": (
            "Variational ground-state optimisation on a minimal H₂ Hamiltonian.\n"
            "OO-VQE uses an orbital-rotation outer loop; Noise-aware VQE adds a\n"
            "depolarising noise model.  Result column shows final energy (Hartree).\n"
        ),
        "qec": (
            "Monte-Carlo simulation of syndrome extraction for a repetition code\n"
            "at distances d = 3, 5, 7, 9 and three physical error rates.\n"
            "Result column shows the logical error rate at the **highest** distance.\n"
        ),
        "mps": (
            "GHZ state fidelity computed via MPS with bond dimension 2 (exact\n"
            "for GHZ).  Tests linear-in-n scaling of the MPS contraction engine.\n"
            "Result column shows ⟨ψ_exact|ψ_MPS⟩² (should be 1.0 for all sizes).\n"
        ),
    }

    for suite_key, results in all_results.items():
        title, _ = SUITES[suite_key]
        lines.append(f"## {title}\n")
        if suite_key in suite_notes:
            lines.append(suite_notes[suite_key])
        lines.append("| Benchmark | Backend | Params | Time (s) | Result |")
        lines.append("|-----------|---------|--------|----------|--------|")
        lines.extend(_result_rows(results))
        lines.append("")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Markdown report written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run QuantX benchmark suites")
    parser.add_argument(
        "--suite",
        choices=list(SUITES),
        default=None,
        help="Run a single named suite (default: all suites)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced problem sizes for rapid smoke-testing",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks",
        help="Directory for JSON result files (default: benchmarks/)",
    )
    args = parser.parse_args(argv)

    suites_to_run = [args.suite] if args.suite else list(SUITES)
    hw = hardware_info()
    all_results: dict[str, list[BenchResult]] = {}

    for key in suites_to_run:
        title, runner = SUITES[key]
        print(f"\n{'='*60}")
        print(title)
        print('='*60)
        results = runner(args.quick)
        print_table(results)
        save_results(results, os.path.join(args.output_dir, f"results_{key}.json"))
        all_results[key] = results

    # Combined JSON
    combined_path = os.path.join(args.output_dir, "results_all.json")
    combined = {
        "hardware": hw,
        "results": {k: [r.to_dict() for r in v] for k, v in all_results.items()},
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(combined_path, "w") as fh:
        json.dump(combined, fh, indent=2, default=str)
    print(f"\nCombined results written to {combined_path}")

    # Markdown report
    md_path = os.path.join(args.output_dir, "BENCHMARKS.md")
    generate_markdown(all_results, hw, md_path)


if __name__ == "__main__":
    main()
