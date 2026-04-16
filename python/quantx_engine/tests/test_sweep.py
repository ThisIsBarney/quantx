"""Tests for parallel parameter sweep."""
from __future__ import annotations
import math, time, pytest

from quantx_engine.sweep import (
    ParameterSweep, SweepResult, SweepError, grid_sweep, list_sweep
)


# Simple fn for testing
def _add(a, b):
    return {"sum": a + b}

def _square(x):
    return {"sq": x * x}

def _slow(x, delay=0.01):
    time.sleep(delay)
    return {"x": x}

def _fail_on_zero(x):
    if x == 0:
        raise ValueError("zero not allowed")
    return {"inv": 1.0 / x}


def test_grid_sweep_basic():
    result = grid_sweep(_add, {"a": [1, 2], "b": [10, 20]})
    assert result.n_total == 4
    assert result.n_failed == 0
    sums = sorted(r["sum"] for r in result.records)
    assert sums == [11, 21, 12, 22] or sorted(sums) == sorted([11, 21, 12, 22])


def test_grid_sweep_result_count():
    result = grid_sweep(_square, {"x": [1, 2, 3, 4, 5]})
    assert result.n_total == 5
    assert all("sq" in r for r in result.records)


def test_grid_sweep_values_correct():
    result = grid_sweep(_square, {"x": [2, 3]})
    squares = {r["x"]: r["sq"] for r in result.records}
    assert squares[2] == 4
    assert squares[3] == 9


def test_list_sweep_basic():
    params = [{"a": 1, "b": 2}, {"a": 10, "b": 20}]
    result = list_sweep(_add, params)
    assert result.n_total == 2
    sums = sorted(r["sum"] for r in result.records)
    assert sums == [3, 30]


def test_catch_errors_true():
    result = grid_sweep(_fail_on_zero, {"x": [0, 1, 2]}, catch_errors=True)
    assert result.n_failed == 1
    assert result.n_total == 3
    assert len(result.failures()) == 1
    assert "__error__" in result.failures()[0]
    assert len(result.successes()) == 2


def test_catch_errors_false_raises():
    sweep = ParameterSweep(fn=_fail_on_zero, catch_errors=False)
    with pytest.raises(ValueError, match="zero"):
        sweep.run({"x": [0, 1]})


def test_best_minimize():
    result = grid_sweep(_square, {"x": [1, 2, 3]})
    best = result.best("sq", minimize=True)
    assert best["sq"] == 1


def test_best_maximize():
    result = grid_sweep(_square, {"x": [1, 2, 3]})
    best = result.best("sq", minimize=False)
    assert best["sq"] == 9


def test_best_missing_key_raises():
    result = grid_sweep(_square, {"x": [1, 2]})
    with pytest.raises(SweepError):
        result.best("nonexistent_key")


def test_to_dict():
    result = grid_sweep(_add, {"a": [1], "b": [2]})
    d = result.to_dict()
    assert "n_total" in d and "n_failed" in d and "elapsed_seconds" in d


def test_sweep_to_dict():
    sweep = ParameterSweep(fn=_add, n_workers=2)
    d = sweep.to_dict()
    assert d["n_workers"] == 2


def test_invalid_n_workers():
    with pytest.raises(SweepError):
        ParameterSweep(fn=_add, n_workers=0)


def test_invalid_executor_type():
    with pytest.raises(SweepError):
        ParameterSweep(fn=_add, executor_type="gpu")


def test_parallel_thread_sweep():
    result = grid_sweep(_square, {"x": list(range(10))}, n_workers=4)
    assert result.n_total == 10
    assert result.n_failed == 0


def test_parallel_produces_same_results_as_sequential():
    params = {"x": list(range(8))}
    seq = grid_sweep(_square, params, n_workers=1)
    par = grid_sweep(_square, params, n_workers=4)
    assert sorted(r["sq"] for r in seq.records) == sorted(r["sq"] for r in par.records)


def test_result_scalar_wrapped():
    """Function returning a scalar is wrapped as {"result": value}."""
    result = grid_sweep(lambda x: x * 2, {"x": [5]})
    assert result.records[0]["result"] == 10


def test_empty_grid():
    result = grid_sweep(_add, {"a": [], "b": [1, 2]})
    assert result.n_total == 0


def test_single_param():
    result = grid_sweep(_square, {"x": [7]})
    assert result.n_total == 1
    assert result.records[0]["sq"] == 49


def test_sweep_result_elapsed_positive():
    result = grid_sweep(_square, {"x": [1, 2, 3]})
    assert result.elapsed_seconds >= 0
