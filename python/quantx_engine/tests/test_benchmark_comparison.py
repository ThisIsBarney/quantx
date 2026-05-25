"""Smoke tests for benchmarks/compare_backends.py (M167).

These tests verify the benchmark functions import correctly and produce
well-formed BenchResult objects — they do NOT require QuTiP or Qiskit.
"""
from __future__ import annotations

import sys
import os

# Make benchmarks/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "benchmarks"))

import pytest


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------

def test_compare_backends_imports():
    """compare_backends.py must be importable without errors."""
    import compare_backends  # noqa: F401


# ---------------------------------------------------------------------------
# BenchResult shape
# ---------------------------------------------------------------------------

def test_bench_result_structure():
    """BenchResult objects from QuantX functions are well-formed."""
    from compare_backends import bench_lanczos_quantx, bench_lanczos_scipy
    from bench_utils import BenchResult

    r = bench_lanczos_quantx()
    assert isinstance(r, BenchResult)
    assert r.benchmark == "lanczos_ground_state"
    assert r.backend == "quantx"
    # Either ran successfully or got a graceful error
    assert r.error is None or isinstance(r.error, str)
    assert isinstance(r.wall_time_s, float)

    r2 = bench_lanczos_scipy()
    assert isinstance(r2, BenchResult)
    assert r2.backend == "scipy"


def test_skip_results_for_missing_backends():
    """Backends that require optional libraries return skipped results, not exceptions."""
    from compare_backends import bench_lanczos_qiskit, bench_dmrg_qutip, bench_dmrg_qiskit

    r_qiskit = bench_lanczos_qiskit()
    assert r_qiskit.skipped is True
    assert r_qiskit.skip_reason != ""

    r_qutip = bench_dmrg_qutip()
    assert r_qutip.skipped is True

    r_qiskit2 = bench_dmrg_qiskit()
    assert r_qiskit2.skipped is True


def test_lindblad_skip_results():
    from compare_backends import bench_lindblad_qiskit
    r = bench_lindblad_qiskit()
    assert r.skipped is True


def test_tebd_skip_results():
    from compare_backends import bench_tebd_qutip, bench_tebd_qiskit
    for fn in [bench_tebd_qutip, bench_tebd_qiskit]:
        r = fn()
        assert r.skipped is True


# ---------------------------------------------------------------------------
# Gate check logic
# ---------------------------------------------------------------------------

def test_gate_check_no_violations():
    """Gate passes when QuantX is within 2× of reference."""
    from compare_backends import check_gate
    from bench_utils import BenchResult

    results = [
        BenchResult("lanczos", "quantx", {}, wall_time_s=1.0, result_value=-5.0),
        BenchResult("lanczos", "scipy",  {}, wall_time_s=0.8, result_value=-5.0),
    ]
    passed, violations = check_gate(results)
    # 1.0 / 0.8 = 1.25 < 2.0 → pass
    assert passed is True
    assert violations == []


def test_gate_check_violation():
    """Gate fails when QuantX is >2× slower."""
    from compare_backends import check_gate
    from bench_utils import BenchResult

    results = [
        BenchResult("problem_x", "quantx", {}, wall_time_s=5.0, result_value=1.0),
        BenchResult("problem_x", "scipy",  {}, wall_time_s=1.0, result_value=1.0),
    ]
    passed, violations = check_gate(results)
    # 5.0 / 1.0 = 5.0 > 2.0 → fail
    assert passed is False
    assert len(violations) == 1
    assert "problem_x" in violations[0]


def test_gate_check_skipped_reference_ignored():
    """Skipped reference results should not trigger a gate violation."""
    from compare_backends import check_gate
    from bench_utils import BenchResult

    results = [
        BenchResult("prob", "quantx", {}, wall_time_s=10.0, result_value=1.0),
        BenchResult("prob", "qutip",  {}, wall_time_s=0.0,
                    skipped=True, skip_reason="not installed"),
    ]
    passed, violations = check_gate(results)
    # No valid reference → gate passes by default
    assert passed is True


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def test_markdown_report_generation():
    """generate_markdown_report returns a non-empty string with expected headers."""
    from compare_backends import generate_markdown_report
    from bench_utils import BenchResult

    results = [
        BenchResult("lanczos_ground_state", "quantx", {"n_sites": 16},
                    wall_time_s=0.38, result_value=-11.228),
        BenchResult("lanczos_ground_state", "scipy", {"n_sites": 16},
                    wall_time_s=0.41, result_value=-11.228),
    ]
    hw = {"platform": "test", "python": "3.11", "numpy": "1.24", "cpu_count": 4,
          "quantx_version": "0.4.0"}

    md = generate_markdown_report(results, hw)
    assert isinstance(md, str)
    assert "# QuantX Benchmark Results" in md
    assert "lanczos_ground_state" in md.lower() or "lanczos" in md.lower()
    assert "Gate" in md
