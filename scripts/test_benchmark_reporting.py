"""Smoke tests for benchmark reports, gate policy, and trend labeling."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks import check_report_thresholds as crt
from benchmarks import compare_latest_reports as clr
from benchmarks import report_utils as ru
from benchmarks import run_problem_benchmark as rpb
from benchmarks import summarize_reports as sr


def assert_true(condition: bool, message: str) -> None:
  """Fail fast with a clear message when a benchmark assertion does not hold."""
  if not condition:
    raise AssertionError(message)


def test_platform_profile_detection() -> None:
  """Verify platform profiles are detected and loaded consistently."""
  local = ru.detect_platform_profile({"PATH": "x"})
  ci = ru.detect_platform_profile({"CI": "true"})
  assert_true(local == "local", "expected local profile outside CI")
  assert_true(ci == "github-windows", "expected github-windows profile in CI")
  policy = ru.suite_policy("continuous2d_smoke", "local")
  assert_true(policy["profile"] == "local", "suite policy should carry the selected profile")
  assert_true(policy["perf_warnings_blocking"] is False, "continuous smoke should not block on perf warnings")


def test_continuous_metadata() -> None:
  """Verify continuous-space reports carry explainable performance metadata."""
  problem = {
      "kind": "continuous_space_2d",
      "continuous_space_2d": {
          "nx": 4,
          "ny": 5,
          "dt": 0.25,
          "steps": 10,
      },
  }
  result = {
      "backend": "continuous2d",
      "engine_version": "test-1",
      "elapsed_ms": 20.0,
      "observables": {
          "fft_provider_eigen": 1.0,
          "norm": 1.0,
      },
      "continuous_density_frames": [
          {"density_xy_flat": [0.1] * 20},
          {"density_xy_flat": [0.2] * 20},
      ],
  }
  aggregate = {
      "elapsed_ms_mean": 20.0,
      "core_elapsed_ms_mean": 10.0,
  }
  perf = rpb.summarize_continuous_performance(problem, result, aggregate)
  assert_true(perf["grid"] == "4x5", "grid label should reflect the continuous grid shape")
  assert_true(perf["grid_points"] == 20, "grid point count should be derived from nx * ny")
  assert_true(perf["export_frame_count"] == 2, "expected two exported frames")
  assert_true(perf["export_volume"] == 40, "expected dense export volume to be counted")
  assert_true(perf["fft_provider"] == "eigen", "expected FFT provider marker to be inferred")
  assert_true(abs(float(perf["avg_step_ms_mean"]) - 1.0) < 1e-12, "avg step cost should derive from elapsed / steps")


def test_gate_classification() -> None:
  """Verify performance gates become warnings while accuracy gates remain blocking."""
  case = {
      "id": "continuous",
      "ok_rate": 1.0,
      "aggregate": {
          "core_elapsed_ms_mean": 5000.0,
          "norm_abs_error_mean": 0.0,
          "sigma_x_abs_error_mean": 0.0,
          "sigma_z_abs_error_mean": 0.0,
      },
  }
  gates = {
      "ok_rate_min": 1.0,
      "core_elapsed_ms_mean_max": 3.0,
      "norm_abs_error_mean_max": 0.05,
      "sigma_x_abs_error_mean_max": 0.08,
      "sigma_z_abs_error_mean_max": 0.08,
  }
  warning_gate = crt.check_case(case, gates, perf_warnings_blocking=False)
  strict_gate = crt.check_case(case, gates, perf_warnings_blocking=True)
  assert_true(len(warning_gate[0]) == 0, "performance threshold should not fail in warning mode")
  assert_true(len(warning_gate[1]) == 1, "performance threshold should emit one warning")
  assert_true(len(strict_gate[0]) == 1, "performance threshold should fail in strict mode")


def test_summary_and_trend_labels() -> None:
  """Verify report summaries and trends keep the new metadata visible."""
  report = {
      "_file": "continuous2d_smoke_000.json",
      "suite": "continuous2d_smoke",
      "generated_at": "2026-03-20T00:00:00+00:00",
      "platform_profile": "github-windows",
      "gate": {"perf_warnings_blocking": False},
      "cases": [
          {
              "aggregate": {
                  "core_elapsed_ms_mean": 10.0,
                  "mean_abs_error_mean": 0.01,
                  "max_abs_error_mean": 0.02,
              },
              "performance": {
                  "grid": "4x5",
                  "export_volume": 40,
                  "fft_provider": "eigen",
                  "avg_step_ms_mean": 1.0,
              },
          }
      ],
  }
  summary = sr.summarize(report)
  assert_true(summary["platform_profile"] == "github-windows", "summary should keep profile")
  assert_true(summary["perf_warning_policy"] == "non-blocking", "summary should label perf policy")
  assert_true(summary["grid"] == "4x5", "summary should expose the grid label")
  assert_true(summary["export_volume"] == 40, "summary should expose export volume")
  assert_true(summary["fft_provider"] == "eigen", "summary should expose FFT provider")

  prev = {
      "cases": [
          {
              "id": "continuous",
              "aggregate": {"core_elapsed_ms_mean": 12.0, "mean_abs_error_mean": 0.0, "max_abs_error_mean": 0.0},
          }
      ]
  }
  curr = {
      "cases": [
          {
              "id": "continuous",
              "aggregate": {"core_elapsed_ms_mean": 10.0, "mean_abs_error_mean": 0.0, "max_abs_error_mean": 0.0},
              "performance": {"grid": "4x5", "export_volume": 40, "fft_provider": "eigen", "avg_step_ms_mean": 1.0},
          }
      ]
  }
  case_map = clr.case_map(curr)
  assert_true("continuous" in case_map, "trend case map should include the case id")
  assert_true(case_map["continuous"]["performance"]["grid"] == "4x5", "trend case map should keep performance metadata")


def test_backwards_compatible_frame_extraction() -> None:
  """Verify old export payload shapes still normalize correctly."""
  old_style = {"frames": [{"density": [1.0, 2.0, 3.0]}]}
  nested = {"continuous_export": {"frames": [{"density_xy_flat": [0.1, 0.2]}]}}
  assert_true(len(ru.extract_continuous_frames(old_style)) == 1, "legacy frame payload should still be accepted")
  assert_true(len(ru.extract_continuous_frames(nested)) == 1, "nested export payload should still be accepted")


def main() -> None:
  """Run benchmark reporting smoke tests."""
  test_platform_profile_detection()
  test_continuous_metadata()
  test_gate_classification()
  test_summary_and_trend_labels()
  test_backwards_compatible_frame_extraction()
  print("benchmark reporting smoke passed")


if __name__ == "__main__":
  main()
