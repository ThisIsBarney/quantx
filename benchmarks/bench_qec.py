"""Benchmark 4: Quantum error correction — logical error rate vs. code distance.

Scenario: Measure wall-clock time to simulate n_shots rounds of syndrome
          extraction for a repetition code at code distances d = 3, 5, 7, 9.
          Physical error rate is varied: p_low (below threshold) and p_high
          (above threshold).

Backends compared:
  - quantx : logical_error_rate_vs_distance (native Python)
  (PyMatching / Stim would be added here when installed)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
from bench_utils import BenchResult, print_table, save_results, timer

DISTANCES  = [3, 5, 7, 9]
P_PHYSICAL = [0.001, 0.05, 0.10]   # below threshold, near, above
N_SHOTS    = 500


def bench_quantx(distances: list[int], p_phys: float, n_shots: int) -> BenchResult:
    try:
        from quantx_engine.contrib.fault_tolerant import logical_error_rate_vs_distance
        with timer() as elapsed:
            rates = logical_error_rate_vs_distance(
                distances, p_phys, n_shots=n_shots, rng_seed=0
            )
        # Encode the full result dict as a compact string for display
        summary = {f"d={d}": float(f"{r:.5f}") for d, r in rates.items()}
        # Primary figure of merit: logical rate at highest distance
        best_rate = float(rates[max(distances)])
        return BenchResult(
            benchmark="qec_logical_rate",
            backend="quantx",
            params={"distances": distances, "p_phys": p_phys, "n_shots": n_shots},
            wall_time_s=elapsed[0],
            result_value=best_rate,
        )
    except Exception as exc:
        return BenchResult(
            "qec_logical_rate", "quantx",
            {"distances": distances, "p_phys": p_phys},
            0.0, error=str(exc),
        )


def bench_stim(distances: list[int], p_phys: float, n_shots: int) -> BenchResult:
    """Stim fast stabiliser simulator (optional competitor)."""
    try:
        import stim
        results_per_d = {}
        with timer() as elapsed:
            for d in distances:
                circuit = stim.Circuit.generated(
                    "repetition_code:memory",
                    rounds=d,
                    distance=d,
                    before_round_data_depolarization=p_phys,
                )
                sampler = circuit.compile_detector_sampler()
                # separate_observables=True returns (det_samples, obs_samples);
                # obs_samples contains the actual logical observable outcomes.
                _, obs_samples = sampler.sample(shots=n_shots, separate_observables=True)
                # Logical error rate = fraction of shots with a logical flip
                n_fail = int(np.sum(obs_samples.flatten()))
                results_per_d[d] = n_fail / n_shots
        best_rate = float(results_per_d[max(distances)])
        return BenchResult(
            benchmark="qec_logical_rate",
            backend="stim",
            params={"distances": distances, "p_phys": p_phys, "n_shots": n_shots},
            wall_time_s=elapsed[0],
            result_value=best_rate,
        )
    except ImportError:
        return BenchResult(
            "qec_logical_rate", "stim",
            {"distances": distances, "p_phys": p_phys},
            0.0, skipped=True, skip_reason="stim not installed",
        )
    except Exception as exc:
        return BenchResult(
            "qec_logical_rate", "stim",
            {"distances": distances, "p_phys": p_phys},
            0.0, error=str(exc),
        )


def run(
    distances: list[int] = DISTANCES,
    p_physical: list[float] = P_PHYSICAL,
    n_shots: int = N_SHOTS,
) -> list[BenchResult]:
    results = []
    for p in p_physical:
        print(f"  p_phys={p}", flush=True)
        results.append(bench_quantx(distances, p, n_shots))
        results.append(bench_stim(distances, p, n_shots))
    return results


if __name__ == "__main__":
    print("Benchmark 4: QEC logical error rate vs. code distance")
    results = run()
    print_table(results)
    save_results(results, "benchmarks/results_qec.json")
