"""Benchmark 1: Lanczos eigensolver — QuantX vs. QuTiP vs. SciPy sparse.

Scenario: Compute the ground-state energy of a 1D Heisenberg chain of L sites
          using Lanczos (iterative sparse eigensolver).

Backends compared:
  - quantx     : LanczosSolver (native Python, k=1 eigenvalue)
  - scipy      : scipy.sparse.linalg.eigsh (always available with scipy)
  - qutip      : qutip.eigenstates (if installed)
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
from bench_utils import BenchResult, print_table, save_results, timer

SIZES = [4, 6, 8, 10, 12]    # number of Heisenberg spins


def build_heisenberg_matrix(n: int) -> np.ndarray:
    """Build dense n-site 1D Heisenberg Hamiltonian matrix."""
    I  = np.eye(2)
    X  = np.array([[0,1],[1,0]], dtype=float)
    Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z  = np.array([[1,0],[0,-1]], dtype=float)
    ops = [X, Y, Z]

    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(n - 1):
        for op in ops:
            kron = np.array([1.0], dtype=complex)
            for j in range(n):
                if j == i or j == i + 1:
                    kron = np.kron(kron, op)
                else:
                    kron = np.kron(kron, I)
            H += 0.5 * kron

    return H.real


def bench_quantx(n: int) -> BenchResult:
    try:
        from quantx_engine.lanczos import LanczosSolver
        H = build_heisenberg_matrix(n)
        solver = LanczosSolver(k=1, max_iter=200, tol=1e-10)
        with timer() as elapsed:
            result = solver.solve(H)
        return BenchResult(
            benchmark="heisenberg_lanczos",
            backend="quantx",
            params={"n_sites": n},
            wall_time_s=elapsed[0],
            result_value=float(result.eigenvalues[0]),
        )
    except Exception as exc:
        return BenchResult("heisenberg_lanczos", "quantx", {"n_sites": n}, 0.0, error=str(exc))


def bench_scipy(n: int) -> BenchResult:
    try:
        import scipy.sparse
        import scipy.sparse.linalg
        H = build_heisenberg_matrix(n)
        H_sp = scipy.sparse.csr_matrix(H)
        with timer() as elapsed:
            vals, _ = scipy.sparse.linalg.eigsh(H_sp, k=1, which="SA")
        return BenchResult(
            benchmark="heisenberg_lanczos",
            backend="scipy",
            params={"n_sites": n},
            wall_time_s=elapsed[0],
            result_value=float(vals[0]),
        )
    except ImportError:
        return BenchResult("heisenberg_lanczos", "scipy", {"n_sites": n}, 0.0,
                           skipped=True, skip_reason="scipy not installed")
    except Exception as exc:
        return BenchResult("heisenberg_lanczos", "scipy", {"n_sites": n}, 0.0, error=str(exc))


def bench_qutip(n: int) -> BenchResult:
    try:
        import qutip
        H = build_heisenberg_matrix(n)
        Hq = qutip.Qobj(H)
        with timer() as elapsed:
            energies, _ = Hq.eigenstates(sparse=True, sort="low", eigvals=1)
        return BenchResult(
            benchmark="heisenberg_lanczos",
            backend="qutip",
            params={"n_sites": n},
            wall_time_s=elapsed[0],
            result_value=float(energies[0]),
        )
    except ImportError:
        return BenchResult("heisenberg_lanczos", "qutip", {"n_sites": n}, 0.0,
                           skipped=True, skip_reason="qutip not installed")
    except Exception as exc:
        return BenchResult("heisenberg_lanczos", "qutip", {"n_sites": n}, 0.0, error=str(exc))


def run(sizes: list[int] = SIZES) -> list[BenchResult]:
    results = []
    for n in sizes:
        print(f"  n_sites={n}", flush=True)
        results.append(bench_quantx(n))
        results.append(bench_scipy(n))
        results.append(bench_qutip(n))
    return results


if __name__ == "__main__":
    print("Benchmark 1: Heisenberg Lanczos eigensolver")
    results = run()
    print_table(results)
    save_results(results, "benchmarks/results_lanczos.json")
