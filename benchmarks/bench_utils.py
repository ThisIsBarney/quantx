"""Shared utilities for QuantX benchmark suite."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BenchResult:
    """Result of a single benchmark run."""
    benchmark: str
    backend: str
    params: dict
    wall_time_s: float
    result_value: Any = None  # e.g. energy, fidelity, error rate
    error: str | None = None
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@contextmanager
def timer():
    """Context manager that yields elapsed wall-clock time."""
    start = time.perf_counter()
    elapsed = [0.0]
    try:
        yield elapsed
    finally:
        elapsed[0] = time.perf_counter() - start


def hardware_info() -> dict:
    """Collect basic hardware/software info for reproducibility."""
    import numpy as np
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "quantx_version": _try_import_version("quantx_engine"),
        "qiskit_version": _try_import_version("qiskit"),
        "pennylane_version": _try_import_version("pennylane"),
        "qutip_version": _try_import_version("qutip"),
    }


def _try_import_version(package: str) -> str | None:
    try:
        mod = __import__(package)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None


def save_results(results: list[BenchResult], path: str = "benchmarks/results.json") -> None:
    """Save benchmark results to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "hardware": hardware_info(),
        "results": [r.to_dict() for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Results saved to {path}")


def print_table(results: list[BenchResult]) -> None:
    """Print benchmark results as a text table."""
    header = f"{'Benchmark':<35} {'Backend':<14} {'Params':<22} {'Time (s)':>10} {'Result':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())[:20]
        if r.skipped:
            row = f"{'  ' + r.benchmark:<35} {'[SKIP]':<14} {params_str:<22} {'n/a':>10} {'':>12}"
        elif r.error:
            row = f"{'  ' + r.benchmark:<35} {r.backend:<14} {params_str:<22} {'ERR':>10} {'':>12}"
        else:
            result_str = f"{r.result_value:.4f}" if isinstance(r.result_value, float) else str(r.result_value or "")
            row = f"{'  ' + r.benchmark:<35} {r.backend:<14} {params_str:<22} {r.wall_time_s:>10.3f} {result_str:>12}"
        print(row)
    print("=" * len(header))
