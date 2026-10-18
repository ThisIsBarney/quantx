"""Parallel parameter sweep utilities for QuantX.

Run any callable (solver, circuit, QML training loop) over a grid of parameters
using Python's concurrent.futures for CPU-bound parallelism.

Usage::

    from quantx_engine.contrib.sweep import ParameterSweep, SweepResult, grid_sweep

    # Simple grid sweep over two parameters
    def my_solver(J, h):
        # ... run simulation ...
        return {"energy": -J - h}

    sweep = ParameterSweep(fn=my_solver, n_workers=4)
    result = sweep.run({"J": [0.5, 1.0, 2.0], "h": [0.0, 0.5, 1.0]})
    df = result.to_dataframe()   # pandas if available, else list of dicts

References
----------
.. [1] Peruzzo, A. et al. (2014). A variational eigenvalue solver on a
       photonic chip. *Nat. Commun.* 5, 4213.
       https://doi.org/10.1038/ncomms5213
       (VQE parameter sweeps as canonical example)

.. [2] McClean, J. R. et al. (2018). Barren plateaus in quantum neural
       network training landscapes.
       *Nat. Commun.* 9, 4812. https://doi.org/10.1038/s41467-018-07090-4
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


class SweepError(RuntimeError):
    """Raised for invalid sweep configuration or unrecoverable job failures."""


# ---------------------------------------------------------------------------
# Module-level worker functions — required for ProcessPoolExecutor pickling
# ---------------------------------------------------------------------------

def _grid_worker(args):
    """Picklable worker for grid combos: args = (fn, keys, combo, catch_errors)."""
    fn, keys, combo, catch_errors = args
    kwargs = dict(zip(keys, combo))
    try:
        result = fn(**kwargs)
        if not isinstance(result, dict):
            result = {"result": result}
        return {**kwargs, **result}
    except Exception as e:
        if catch_errors:
            return {**kwargs, "__error__": str(e), "__error_type__": type(e).__name__}
        raise


def _list_worker(args):
    """Picklable worker for list items: args = (fn, kwargs, catch_errors)."""
    fn, kwargs, catch_errors = args
    try:
        result = fn(**kwargs)
        if not isinstance(result, dict):
            result = {"result": result}
        return {**kwargs, **result}
    except Exception as e:
        if catch_errors:
            return {**kwargs, "__error__": str(e), "__error_type__": type(e).__name__}
        raise



@dataclass
class SweepResult:
    records: list[dict]       # each record = params + result dict merged
    n_total: int
    n_failed: int
    elapsed_seconds: float

    def successes(self) -> list[dict]:
        """Return records that completed without error."""
        return [r for r in self.records if "__error__" not in r]

    def failures(self) -> list[dict]:
        """Return records that raised an exception."""
        return [r for r in self.records if "__error__" in r]

    def to_dataframe(self):
        """Return pandas DataFrame if available, else list of dicts."""
        try:
            import pandas as pd
            return pd.DataFrame(self.successes())
        except ImportError:
            return self.successes()

    def best(self, key: str, minimize: bool = True) -> dict:
        """Return the record with the best (min or max) value for key."""
        s = self.successes()
        if not s:
            raise SweepError("No successful records to find best from")
        if not all(key in r for r in s):
            raise SweepError(f"Key {key!r} not present in all successful records")
        return min(s, key=lambda r: r[key]) if minimize else max(s, key=lambda r: r[key])

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_failed": self.n_failed,
            "n_success": self.n_total - self.n_failed,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }


@dataclass
class ParameterSweep:
    fn: Callable                      # function to sweep; must accept keyword args matching param names
    n_workers: int = 1                # 1 = sequential, >1 = ThreadPoolExecutor (or ProcessPoolExecutor)
    executor_type: str = "thread"     # "thread" or "process"
    timeout: Optional[float] = None   # per-job timeout in seconds (None = no limit)
    catch_errors: bool = True         # if True, capture exceptions; if False, re-raise

    def __post_init__(self):
        if self.n_workers < 1:
            raise SweepError(f"n_workers must be >= 1, got {self.n_workers}")
        if self.executor_type not in {"thread", "process"}:
            raise SweepError(
                f"executor_type must be 'thread' or 'process', got {self.executor_type!r}"
            )

    def run(self, param_grid: dict[str, list]) -> SweepResult:
        """Run fn over Cartesian product of param_grid values.

        param_grid: {"param1": [v1, v2, ...], "param2": [u1, u2, ...]}
        Each combination is passed as fn(param1=v1, param2=u1).
        Result of fn must be a dict (or will be wrapped as {"result": fn(...)}).
        """
        import itertools
        import time
        from concurrent.futures import (
            ThreadPoolExecutor,
            ProcessPoolExecutor,
            TimeoutError as FuturesTimeout,
        )

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))

        records = []
        start = time.perf_counter()

        if self.n_workers == 1:
            for combo in combos:
                records.append(_grid_worker((self.fn, keys, combo, self.catch_errors)))
        else:
            ExecutorClass = (
                ThreadPoolExecutor if self.executor_type == "thread" else ProcessPoolExecutor
            )
            executor = ExecutorClass(max_workers=self.n_workers)
            try:
                futures = {
                    executor.submit(_grid_worker, (self.fn, keys, combo, self.catch_errors)): combo
                    for combo in combos
                }
                timed_out = False
                for future, combo in futures.items():
                    try:
                        records.append(future.result(timeout=self.timeout))
                    except FuturesTimeout:
                        timed_out = True
                        kwargs = dict(zip(keys, combo))
                        if self.catch_errors:
                            records.append(
                                {**kwargs, "__error__": "timeout", "__error_type__": "TimeoutError"}
                            )
                        else:
                            raise SweepError(f"Job timed out for {kwargs}")
                    except Exception as e:
                        kwargs = dict(zip(keys, combo))
                        if self.catch_errors:
                            records.append(
                                {**kwargs, "__error__": str(e), "__error_type__": type(e).__name__}
                            )
                        else:
                            raise
            finally:
                # Cancel pending futures and shut down without blocking on timed-out tasks
                _wait = not timed_out
                try:
                    executor.shutdown(wait=_wait, cancel_futures=True)
                except TypeError:  # Python < 3.9 does not have cancel_futures
                    executor.shutdown(wait=False)

        elapsed = time.perf_counter() - start
        n_failed = sum(1 for r in records if "__error__" in r)
        return SweepResult(
            records=records,
            n_total=len(records),
            n_failed=n_failed,
            elapsed_seconds=elapsed,
        )

    def run_list(self, param_list: list[dict]) -> SweepResult:
        """Run fn over an explicit list of parameter dicts (not Cartesian product).

        param_list: [{"J": 1.0, "h": 0.5}, {"J": 2.0, "h": 1.0}, ...]
        """
        import time
        from concurrent.futures import (
            ThreadPoolExecutor,
            ProcessPoolExecutor,
            TimeoutError as FuturesTimeout,
        )

        records = []
        start = time.perf_counter()

        if self.n_workers == 1:
            for kwargs in param_list:
                records.append(_list_worker((self.fn, kwargs, self.catch_errors)))
        else:
            ExecutorClass = (
                ThreadPoolExecutor if self.executor_type == "thread" else ProcessPoolExecutor
            )
            executor = ExecutorClass(max_workers=self.n_workers)
            timed_out = False
            try:
                futures = {
                    executor.submit(_list_worker, (self.fn, kw, self.catch_errors)): kw
                    for kw in param_list
                }
                for future, kw in futures.items():
                    try:
                        records.append(future.result(timeout=self.timeout))
                    except FuturesTimeout:
                        timed_out = True
                        if self.catch_errors:
                            records.append({**kw, "__error__": "timeout", "__error_type__": "TimeoutError"})
                        else:
                            raise SweepError(f"Job timed out for {kw}")
                    except Exception as e:
                        if self.catch_errors:
                            records.append({**kw, "__error__": str(e), "__error_type__": type(e).__name__})
                        else:
                            raise
            finally:
                _wait = not timed_out
                try:
                    executor.shutdown(wait=_wait, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=False)

        elapsed = time.perf_counter() - start
        n_failed = sum(1 for r in records if "__error__" in r)
        return SweepResult(
            records=records,
            n_total=len(records),
            n_failed=n_failed,
            elapsed_seconds=elapsed,
        )

    def to_dict(self) -> dict:
        return {"n_workers": self.n_workers, "executor_type": self.executor_type}


def grid_sweep(
    fn,
    param_grid: dict[str, list],
    n_workers: int = 1,
    executor_type: str = "thread",
    catch_errors: bool = True,
) -> SweepResult:
    """Convenience wrapper: run fn over Cartesian product of param_grid."""
    sweep = ParameterSweep(
        fn=fn, n_workers=n_workers, executor_type=executor_type, catch_errors=catch_errors
    )
    return sweep.run(param_grid)


def list_sweep(
    fn,
    param_list: list[dict],
    n_workers: int = 1,
    executor_type: str = "thread",
    catch_errors: bool = True,
) -> SweepResult:
    """Convenience wrapper: run fn over explicit list of parameter dicts."""
    sweep = ParameterSweep(
        fn=fn, n_workers=n_workers, executor_type=executor_type, catch_errors=catch_errors
    )
    return sweep.run_list(param_list)
