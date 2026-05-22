"""Benchmark 5: MPS simulation — GHZ fidelity vs. system size.

Scenario: Compute the fidelity of the MPS representation of an n-qubit GHZ
          state against the exact state vector for increasing n.
          Bond dimension max_bond=2 is exact for GHZ.

Backends compared:
  - quantx : ghz_fidelity (native MPS simulator)
  - tenpy  : TeNPy MPSEnvironment (if installed)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
from bench_utils import BenchResult, print_table, save_results, timer

SIZES     = [4, 8, 12, 16, 20, 32, 64]
MAX_BOND  = 2      # exact for GHZ; higher bond tests compression overhead


def bench_quantx(n: int, max_bond: int = MAX_BOND) -> BenchResult:
    try:
        from quantx_engine.mps_simulator import ghz_fidelity
        with timer() as elapsed:
            fidelity = ghz_fidelity(n_qubits=n, max_bond=max_bond)
        return BenchResult(
            benchmark="mps_ghz_fidelity",
            backend="quantx",
            params={"n_qubits": n, "max_bond": max_bond},
            wall_time_s=elapsed[0],
            result_value=float(fidelity),
        )
    except Exception as exc:
        return BenchResult(
            "mps_ghz_fidelity", "quantx",
            {"n_qubits": n, "max_bond": max_bond},
            0.0, error=str(exc),
        )


def bench_tenpy(n: int, max_bond: int = MAX_BOND) -> BenchResult:
    """TeNPy MPS GHZ state construction (optional competitor)."""
    try:
        import tenpy
        import tenpy.networks.mps as tenpy_mps
        import tenpy.networks.site as tenpy_site

        site = tenpy_site.SpinHalfSite(conserve="None")
        sites = [site] * n

        with timer() as elapsed:
            # Build GHZ = (|0...0> + |1...1>) / sqrt(2) as MPS
            # TeNPy allows this via from_Bflat
            B_list = []
            for i in range(n):
                if i == 0:
                    B = np.zeros((1, 2, 2), dtype=complex)
                    B[0, 0, 0] = 1.0   # |0> branch
                    B[0, 1, 1] = 1.0   # |1> branch
                elif i == n - 1:
                    B = np.zeros((2, 2, 1), dtype=complex)
                    B[0, 0, 0] = 1.0 / np.sqrt(2)
                    B[1, 1, 0] = 1.0 / np.sqrt(2)
                else:
                    B = np.zeros((2, 2, 2), dtype=complex)
                    B[0, 0, 0] = 1.0
                    B[1, 1, 1] = 1.0
                B_list.append(B)
            SVs = [np.ones(min(max_bond, 2 ** min(i, n - i))) for i in range(n + 1)]
            mps_state = tenpy_mps.MPS.from_Bflat(sites, B_list, SVs=SVs, bc="finite")
            # Compute true GHZ fidelity: |<GHZ_exact | psi_MPS>|^2
            # Convert MPS to dense statevector and overlap with exact GHZ vector
            psi_mps = mps_state.get_theta(0, n).to_ndarray().reshape(-1)
            psi_ghz = np.zeros(2 ** n, dtype=complex)
            psi_ghz[0] = 1.0 / np.sqrt(2)           # |00...0>
            psi_ghz[-1] = 1.0 / np.sqrt(2)           # |11...1>
            fidelity = float(abs(np.dot(psi_ghz.conj(), psi_mps)) ** 2)

        return BenchResult(
            benchmark="mps_ghz_fidelity",
            backend="tenpy",
            params={"n_qubits": n, "max_bond": max_bond},
            wall_time_s=elapsed[0],
            result_value=fidelity,
        )
    except ImportError:
        return BenchResult(
            "mps_ghz_fidelity", "tenpy",
            {"n_qubits": n, "max_bond": max_bond},
            0.0, skipped=True, skip_reason="tenpy not installed",
        )
    except Exception as exc:
        return BenchResult(
            "mps_ghz_fidelity", "tenpy",
            {"n_qubits": n, "max_bond": max_bond},
            0.0, error=str(exc),
        )


def run(sizes: list[int] = SIZES, max_bond: int = MAX_BOND) -> list[BenchResult]:
    results = []
    for n in sizes:
        print(f"  n_qubits={n}", flush=True)
        results.append(bench_quantx(n, max_bond))
        results.append(bench_tenpy(n, max_bond))
    return results


if __name__ == "__main__":
    print("Benchmark 5: MPS GHZ fidelity")
    results = run()
    print_table(results)
    save_results(results, "benchmarks/results_mps.json")
