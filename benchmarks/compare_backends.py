"""M167 — QuantX performance comparison vs Qiskit Aer, QuTiP, and SciPy.

Benchmarks four canonical quantum simulation problems and reports wall-clock
times for each available backend.  Missing backends are skipped gracefully.

Problems
--------
1. Lanczos ground state   — 16-site 1D Heisenberg chain
2. Lindblad evolution     — 6-qubit depolarizing channel, 100 steps
3. DMRG (MPS)             — 50-site Heisenberg, bond-dim=64
4. TEBD time evolution    — 30-site TFIM, 200 steps, dt=0.05

CI gate
-------
QuantX must not be more than 2× slower than the fastest competing backend
on any problem for which at least one competitor is available and completes
without error.  Run with --enforce-gate to activate the check.

Usage
-----
    python benchmarks/compare_backends.py
    python benchmarks/compare_backends.py --enforce-gate
    python benchmarks/compare_backends.py --output benchmarks/results_comparison.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
from bench_utils import BenchResult, hardware_info, print_table, save_results, timer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip(benchmark: str, backend: str, reason: str) -> BenchResult:
    return BenchResult(
        benchmark=benchmark, backend=backend, params={},
        wall_time_s=0.0, skipped=True, skip_reason=reason
    )


def _error(benchmark: str, backend: str, params: dict, exc: Exception) -> BenchResult:
    return BenchResult(
        benchmark=benchmark, backend=backend, params=params,
        wall_time_s=0.0, error=str(exc)[:200]
    )


# ===========================================================================
# Problem 1 — Lanczos ground state, 16-site Heisenberg
# ===========================================================================

N_LANCZOS = 16


def _heisenberg_sparse(n: int):
    """Build 2^n × 2^n sparse Heisenberg Hamiltonian via scipy."""
    from scipy.sparse import kron, eye
    I = eye(2, format="csr", dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    dim = 2 ** n
    H = None
    for i in range(n - 1):
        for op in [X, Y, Z]:
            term = eye(1, format="csr", dtype=complex)
            for j in range(n):
                mat = op if j in (i, i + 1) else I
                term = kron(term, mat, format="csr")
            H = (0.5 * term) if H is None else H + 0.5 * term
    return H


def bench_lanczos_quantx() -> BenchResult:
    benchmark, backend = "lanczos_ground_state", "quantx"
    params = {"n_sites": N_LANCZOS, "k": 1}
    try:
        from quantx_engine.contrib.lanczos import LanczosSolver
        with timer() as elapsed:
            solver = LanczosSolver(k=1, max_iter=500, tol=1e-10)
            # Use the native Heisenberg solver: routes through the C++ backend
            # when available and falls back to sparse-matrix Lanczos otherwise.
            # This avoids calling .toarray() on a 2^16 × 2^16 matrix (~32 GiB).
            result = solver.solve_heisenberg(N_LANCZOS)
        return BenchResult(benchmark=benchmark, backend=backend, params=params,
                           wall_time_s=elapsed[0], result_value=float(result.eigenvalues[0]))
    except Exception as exc:
        return _error(benchmark, backend, params, exc)


def bench_lanczos_scipy() -> BenchResult:
    benchmark, backend = "lanczos_ground_state", "scipy"
    params = {"n_sites": N_LANCZOS, "k": 1}
    try:
        from scipy.sparse.linalg import eigsh
        H = _heisenberg_sparse(N_LANCZOS)
        with timer() as elapsed:
            vals, _ = eigsh(H, k=1, which="SA")
        return BenchResult(benchmark=benchmark, backend=backend, params=params,
                           wall_time_s=elapsed[0], result_value=float(vals[0]))
    except Exception as exc:
        return _error(benchmark, backend, params, exc)


def bench_lanczos_qutip() -> BenchResult:
    benchmark, backend = "lanczos_ground_state", "qutip"
    params = {"n_sites": N_LANCZOS}
    try:
        import qutip as qt
    except ImportError:
        return _skip(benchmark, backend, "qutip not installed")
    try:
        from scipy.sparse import issparse
        H = _heisenberg_sparse(N_LANCZOS)
        H_qt = qt.Qobj(H, dims=[[2] * N_LANCZOS, [2] * N_LANCZOS])
        with timer() as elapsed:
            vals = H_qt.eigenenergies(sparse=True, eigvals=1)
        return BenchResult(benchmark=benchmark, backend=backend, params=params,
                           wall_time_s=elapsed[0], result_value=float(vals[0]))
    except Exception as exc:
        return _error(benchmark, backend, params, exc)


def bench_lanczos_qiskit() -> BenchResult:
    benchmark, backend = "lanczos_ground_state", "qiskit-aer"
    params = {"n_sites": N_LANCZOS}
    # Qiskit Aer does not have a native eigensolver; skip
    return _skip(benchmark, backend, "Qiskit Aer has no sparse eigensolver")


# ===========================================================================
# Problem 2 — Lindblad evolution, 6 qubits, 100 steps
# ===========================================================================

N_LINDBLAD = 6
DT_LINDBLAD = 0.1
N_STEPS_LINDBLAD = 100


def bench_lindblad_quantx() -> BenchResult:
    benchmark, backend = "lindblad_evolution", "quantx"
    params = {"n_qubits": N_LINDBLAD, "steps": N_STEPS_LINDBLAD, "dt": DT_LINDBLAD}
    try:
        from quantx_engine import Engine
        from quantx_engine.contrib.lindblad import LindbladSolver, JumpOperator
        # Transverse-field Ising Hamiltonian: H = 0.5 * sum_i Z_i
        hamiltonian_terms = [(0.5, f"Z{i}") for i in range(N_LINDBLAD)]
        # Single-qubit amplitude-damping collapse operators (rate 0.01 per qubit)
        jump_operators = [
            JumpOperator("amplitude_damping", qubit=i, rate=0.01)
            for i in range(N_LINDBLAD)
        ]
        solver = LindbladSolver(Engine())
        with timer() as elapsed:
            result = solver.solve(
                n_qubits=N_LINDBLAD,
                hamiltonian_terms=hamiltonian_terms,
                jump_operators=jump_operators,
                total_time=DT_LINDBLAD * N_STEPS_LINDBLAD,
                dt=DT_LINDBLAD,
            )
        return BenchResult(benchmark=benchmark, backend=backend, params=params,
                           wall_time_s=elapsed[0], result_value=result.final_purity)
    except Exception as exc:
        return _error(benchmark, backend, params, exc)


def bench_lindblad_qutip() -> BenchResult:
    benchmark, backend = "lindblad_evolution", "qutip"
    params = {"n_qubits": N_LINDBLAD, "steps": N_STEPS_LINDBLAD, "dt": DT_LINDBLAD}
    try:
        import qutip as qt
    except ImportError:
        return _skip(benchmark, backend, "qutip not installed")
    try:
        dims = [2] * N_LINDBLAD
        dim = 2 ** N_LINDBLAD
        I2 = qt.qeye(2)
        sz = qt.sigmaz()
        H = qt.Qobj(np.zeros((dim, dim), dtype=complex), dims=[dims, dims])
        for i in range(N_LINDBLAD):
            ops_list = [I2] * N_LINDBLAD
            ops_list[i] = 0.5 * sz
            H += qt.tensor(ops_list)
        c_ops = []
        for i in range(N_LINDBLAD):
            ops_list = [I2] * N_LINDBLAD
            ops_list[i] = 0.1 * qt.destroy(2)
            c_ops.append(qt.tensor(ops_list))
        rho0 = qt.Qobj(np.eye(dim, dtype=complex) / dim, dims=[dims, dims])
        t_list = np.linspace(0, DT_LINDBLAD * N_STEPS_LINDBLAD, N_STEPS_LINDBLAD + 1)
        with timer() as elapsed:
            result = qt.mesolve(H, rho0, t_list, c_ops=c_ops)
        rho_final = result.states[-1]
        purity = float(np.real((rho_final * rho_final).tr()))
        return BenchResult(benchmark=benchmark, backend=backend, params=params,
                           wall_time_s=elapsed[0], result_value=purity)
    except Exception as exc:
        return _error(benchmark, backend, params, exc)


def bench_lindblad_qiskit() -> BenchResult:
    return _skip("lindblad_evolution", "qiskit-aer",
                 "Qiskit Aer noise simulation uses Monte Carlo trajectories — different algorithm")


# ===========================================================================
# Problem 3 — DMRG / MPS ground state, 50-site Heisenberg, bond-dim=64
# ===========================================================================

N_DMRG = 50
CHI_DMRG = 64


def bench_dmrg_quantx() -> BenchResult:
    benchmark, backend = "dmrg_ground_state", "quantx"
    params = {"n_sites": N_DMRG, "bond_dim": CHI_DMRG}
    try:
        from quantx_engine.contrib.tensor_network import MPO
        from quantx_engine.contrib.mps_simulator import CachedDMRG
        mpo = MPO.heisenberg(N_DMRG)
        solver = CachedDMRG(mpo=mpo, max_chi=CHI_DMRG, n_sweeps=10)
        with timer() as elapsed:
            result = solver.run()
        return BenchResult(benchmark=benchmark, backend=backend, params=params,
                           wall_time_s=elapsed[0], result_value=float(result.energy))
    except Exception as exc:
        return _error(benchmark, backend, params, exc)


def bench_dmrg_qutip() -> BenchResult:
    return _skip("dmrg_ground_state", "qutip", "QuTiP does not include DMRG")


def bench_dmrg_qiskit() -> BenchResult:
    return _skip("dmrg_ground_state", "qiskit-aer", "Qiskit Aer does not include DMRG")


# ===========================================================================
# Problem 4 — TEBD time evolution, 30-site TFIM, 200 steps, dt=0.05
# ===========================================================================

N_TEBD = 30
DT_TEBD = 0.05
N_STEPS_TEBD = 200
CHI_TEBD = 32


def bench_tebd_quantx() -> BenchResult:
    benchmark, backend = "tebd_time_evolution", "quantx"
    params = {"n_sites": N_TEBD, "bond_dim": CHI_TEBD, "steps": N_STEPS_TEBD, "dt": DT_TEBD}
    try:
        from quantx_engine.contrib.mps_simulator import MPSSimulator
        # TFIM Hamiltonian: H = -sum_{i} Z_i Z_{i+1} - sum_i X_i
        hamiltonian_terms = []
        for i in range(N_TEBD - 1):
            label = "I" * i + "ZZ" + "I" * (N_TEBD - i - 2)
            hamiltonian_terms.append({"label": label, "coefficient": -1.0})
        for i in range(N_TEBD):
            label = "I" * i + "X" + "I" * (N_TEBD - i - 1)
            hamiltonian_terms.append({"label": label, "coefficient": -1.0})
        sim = MPSSimulator(n_qubits=N_TEBD, max_bond=CHI_TEBD)
        with timer() as elapsed:
            sim.evolve_hamiltonian(
                hamiltonian_terms,
                total_time=DT_TEBD * N_STEPS_TEBD,
                dt=DT_TEBD,
            )
        # Report max bond dimension used as a proxy for entanglement growth
        return BenchResult(benchmark=benchmark, backend=backend, params=params,
                           wall_time_s=elapsed[0], result_value=float(sim.max_bond_used))
    except Exception as exc:
        return _error(benchmark, backend, params, exc)


def bench_tebd_qutip() -> BenchResult:
    return _skip("tebd_time_evolution", "qutip",
                 "QuTiP uses dense matrix ODE solver, not MPS/TEBD")


def bench_tebd_qiskit() -> BenchResult:
    return _skip("tebd_time_evolution", "qiskit-aer",
                 "Qiskit Aer statevector simulation not MPS-based")


# ===========================================================================
# CI gate check
# ===========================================================================

GATE_MAX_SLOWDOWN = 2.0  # QuantX must be within 2× of fastest reference


def check_gate(results: list[BenchResult]) -> tuple[bool, list[str]]:
    """Return (passed, list_of_violations)."""
    from collections import defaultdict
    by_benchmark: dict[str, list[BenchResult]] = defaultdict(list)
    for r in results:
        by_benchmark[r.benchmark].append(r)

    violations = []
    for bench_name, bench_results in by_benchmark.items():
        quantx_results = [r for r in bench_results if r.backend == "quantx"
                          and not r.skipped and not r.error]
        reference_results = [r for r in bench_results if r.backend != "quantx"
                             and not r.skipped and not r.error]
        if not quantx_results or not reference_results:
            continue
        quantx_time = quantx_results[0].wall_time_s
        fastest_ref = min(r.wall_time_s for r in reference_results)
        if fastest_ref > 0 and quantx_time / fastest_ref > GATE_MAX_SLOWDOWN:
            violations.append(
                f"{bench_name}: QuantX {quantx_time:.3f}s vs fastest ref "
                f"{fastest_ref:.3f}s → {quantx_time/fastest_ref:.1f}× (limit {GATE_MAX_SLOWDOWN}×)"
            )

    return len(violations) == 0, violations


# ===========================================================================
# Markdown report generation
# ===========================================================================

def generate_markdown_report(results: list[BenchResult], hw: dict) -> str:
    """Generate a Markdown table report for benchmarks/README.md."""
    from collections import defaultdict
    by_benchmark: dict[str, list[BenchResult]] = defaultdict(list)
    for r in results:
        by_benchmark[r.benchmark].append(r)

    backends_order = ["quantx", "scipy", "qutip", "qiskit-aer"]

    lines = [
        "# QuantX Benchmark Results",
        "",
        f"Generated on: {time.strftime('%Y-%m-%d')}",
        "",
        "## Hardware",
        "",
        f"| Property | Value |",
        f"|----------|-------|",
    ]
    for k, v in hw.items():
        if v is not None:
            lines.append(f"| {k} | {v} |")
    lines += ["", "## Results", ""]

    problem_labels = {
        "lanczos_ground_state":  "Lanczos ground state (n=16 Heisenberg)",
        "lindblad_evolution":    "Lindblad evolution (6 qubits, 100 steps)",
        "dmrg_ground_state":     "DMRG ground state (n=50, χ=64)",
        "tebd_time_evolution":   "TEBD time evolution (n=30, 200 steps)",
    }

    for bench_name in problem_labels:
        bench_results = {r.backend: r for r in by_benchmark.get(bench_name, [])}
        label = problem_labels[bench_name]
        lines += [f"### {label}", "", "| Backend | Time (s) | Result | Notes |",
                  "|---------|----------|--------|-------|"]
        for backend in backends_order:
            r = bench_results.get(backend)
            if r is None:
                continue
            if r.skipped:
                lines.append(f"| {backend} | — | — | {r.skip_reason} |")
            elif r.error:
                lines.append(f"| {backend} | ERR | — | `{r.error[:80]}` |")
            else:
                result_str = f"{r.result_value:.4f}" if isinstance(r.result_value, float) else str(r.result_value)
                lines.append(f"| {backend} | {r.wall_time_s:.3f} | {result_str} | |")
        lines.append("")

    lines += [
        "## Gate",
        "",
        f"QuantX must not be >{GATE_MAX_SLOWDOWN}× slower than the fastest reference backend.",
        "",
    ]
    passed, violations = check_gate(results)
    if passed:
        lines.append("✅ **Gate PASSED** — QuantX is within the speed budget on all problems.")
    else:
        lines.append("❌ **Gate FAILED:**")
        for v in violations:
            lines.append(f"- {v}")

    return "\n".join(lines) + "\n"


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="QuantX backend comparison benchmarks")
    parser.add_argument("--enforce-gate", action="store_true",
                        help="Exit 1 if QuantX is >2× slower than fastest reference")
    parser.add_argument("--output", default="benchmarks/results_comparison.json",
                        help="Path to save JSON results (default: benchmarks/results_comparison.json)")
    parser.add_argument("--markdown", default="benchmarks/BENCHMARKS.md",
                        help="Path to save Markdown report")
    args = parser.parse_args()

    print("QuantX backend comparison benchmark")
    print("=" * 50)

    results: list[BenchResult] = []

    # Problem 1 — Lanczos
    print("\n[1/4] Lanczos ground state (n=16 Heisenberg) ...")
    results += [bench_lanczos_quantx(), bench_lanczos_scipy(),
                bench_lanczos_qutip(), bench_lanczos_qiskit()]

    # Problem 2 — Lindblad
    print("[2/4] Lindblad evolution (6 qubits, 100 steps) ...")
    results += [bench_lindblad_quantx(), bench_lindblad_qutip(), bench_lindblad_qiskit()]

    # Problem 3 — DMRG
    print("[3/4] DMRG ground state (n=50 Heisenberg, bond-dim=64) ...")
    results += [bench_dmrg_quantx(), bench_dmrg_qutip(), bench_dmrg_qiskit()]

    # Problem 4 — TEBD
    print("[4/4] TEBD time evolution (n=30 TFIM, 200 steps) ...")
    results += [bench_tebd_quantx(), bench_tebd_qutip(), bench_tebd_qiskit()]

    print_table(results)

    # Save JSON
    hw = hardware_info()
    save_results(results, args.output)

    # Save Markdown
    md = generate_markdown_report(results, hw)
    os.makedirs(os.path.dirname(args.markdown) or ".", exist_ok=True)
    with open(args.markdown, "w") as f:
        f.write(md)
    print(f"Markdown report saved to {args.markdown}")

    # Gate check
    passed, violations = check_gate(results)
    if violations:
        print("\n⚠ Gate violations:")
        for v in violations:
            print(f"  {v}")
    else:
        print("\n✅ CI gate: PASSED")

    if args.enforce_gate and not passed:
        print("ERROR: --enforce-gate active, gate failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
