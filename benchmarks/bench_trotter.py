"""Benchmark 2: Trotter time evolution — QuantX vs. QuTiP vs. SciPy expm.

Scenario: Evolve a 1D Ising model (n qubits) from t=0 to t=1 using second-order
          Trotter decomposition with n_steps=100 steps.

Backends compared:
  - quantx     : TrotterSimulator (ising_trotter helper)
  - scipy_expm : exact matrix exponentiation (scipy.linalg.expm) — reference
  - qutip      : qutip.sesolve (Schrödinger equation solver)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
from bench_utils import BenchResult, print_table, save_results, timer

SIZES = [4, 6, 8, 10]


def bench_quantx(n: int) -> BenchResult:
    try:
        from quantx_engine.contrib.trotter import ising_trotter
        ts = ising_trotter(n=n, J=1.0, h=0.5)
        with timer() as elapsed:
            result = ts.run(t=1.0, n_steps=100, order=2)
        final_norm = float(np.linalg.norm(result.state))
        return BenchResult(
            benchmark="ising_trotter",
            backend="quantx",
            params={"n_qubits": n, "n_steps": 100},
            wall_time_s=elapsed[0],
            result_value=final_norm,
        )
    except Exception as exc:
        return BenchResult("ising_trotter", "quantx", {"n_qubits": n}, 0.0, error=str(exc))


def bench_scipy_expm(n: int) -> BenchResult:
    try:
        import scipy.linalg
        # Build TFIM Hamiltonian
        dim = 2 ** n
        I  = np.eye(2)
        Z  = np.array([[1,0],[0,-1]], dtype=float)
        X  = np.array([[0,1],[1,0]], dtype=float)
        H = np.zeros((dim, dim), dtype=float)
        for i in range(n - 1):
            ZZ = np.array([1.0])
            for j in range(n):
                ZZ = np.kron(ZZ, Z if j in (i, i+1) else I)
            H -= ZZ
        for i in range(n):
            XI = np.array([1.0])
            for j in range(n):
                XI = np.kron(XI, X if j == i else I)
            H -= 0.5 * XI
        psi0 = np.ones(dim) / np.sqrt(dim)
        with timer() as elapsed:
            U = scipy.linalg.expm(-1j * H * 1.0)
            psi_f = U @ psi0
        final_norm = float(np.abs(np.linalg.norm(psi_f)))
        return BenchResult(
            benchmark="ising_trotter",
            backend="scipy_expm",
            params={"n_qubits": n, "exact": True},
            wall_time_s=elapsed[0],
            result_value=final_norm,
        )
    except ImportError:
        return BenchResult("ising_trotter", "scipy_expm", {"n_qubits": n}, 0.0,
                           skipped=True, skip_reason="scipy not installed")
    except Exception as exc:
        return BenchResult("ising_trotter", "scipy_expm", {"n_qubits": n}, 0.0, error=str(exc))


def bench_qutip(n: int) -> BenchResult:
    try:
        import qutip
        import scipy.linalg
        # Build Hamiltonian via scipy then wrap as Qobj
        dim = 2 ** n
        I  = np.eye(2)
        Z  = np.array([[1,0],[0,-1]], dtype=float)
        X  = np.array([[0,1],[1,0]], dtype=float)
        H = np.zeros((dim, dim), dtype=float)
        for i in range(n - 1):
            ZZ = np.array([1.0])
            for j in range(n):
                ZZ = np.kron(ZZ, Z if j in (i, i+1) else I)
            H -= ZZ
        for i in range(n):
            XI = np.array([1.0])
            for j in range(n):
                XI = np.kron(XI, X if j == i else I)
            H -= 0.5 * XI
        Hq = qutip.Qobj(H)
        psi0 = qutip.Qobj(np.ones(dim) / np.sqrt(dim))
        times = np.linspace(0, 1, 11)
        with timer() as elapsed:
            output = qutip.sesolve(Hq, psi0, times)
        final_norm = float(output.states[-1].norm())
        return BenchResult(
            benchmark="ising_trotter",
            backend="qutip",
            params={"n_qubits": n},
            wall_time_s=elapsed[0],
            result_value=final_norm,
        )
    except ImportError:
        return BenchResult("ising_trotter", "qutip", {"n_qubits": n}, 0.0,
                           skipped=True, skip_reason="qutip not installed")
    except Exception as exc:
        return BenchResult("ising_trotter", "qutip", {"n_qubits": n}, 0.0, error=str(exc))


def run(sizes: list[int] = SIZES) -> list[BenchResult]:
    results = []
    for n in sizes:
        print(f"  n_qubits={n}", flush=True)
        results.append(bench_quantx(n))
        results.append(bench_scipy_expm(n))
        results.append(bench_qutip(n))
    return results


if __name__ == "__main__":
    print("Benchmark 2: Ising Trotter time evolution")
    results = run()
    print_table(results)
    save_results(results, "benchmarks/results_trotter.json")
