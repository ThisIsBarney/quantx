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
from benchmarks import run_circuit_benchmark as rcb
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
      "continuous_export": {
          "schema_version": 1,
          "export_config": {
              "frame_stride": 4,
              "density_floor_ratio": 0.05,
              "points_max": 128,
          },
          "frames": [
              {"step": 0, "density_xy_flat": [0.1] * 20},
              {"step": 4, "density_xy_flat": [0.2] * 20},
          ],
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

  diag = ru.continuous_export_diagnostics(result, steps=10)
  assert_true(diag["schema_version"] == 1, "schema_version should be surfaced when present")
  assert_true("frame_stride=4" in diag["export_config_summary"], "export_config summary should include frame stride")
  assert_true("density_floor_ratio=0.05" in diag["export_config_summary"], "export_config summary should include floor ratio")
  assert_true("explicit sampling" in diag["frame_sampling"], "sampling note should prefer explicit configuration")
  assert_true(diag["diagnostic_tag"] == "ok", "complete continuous export should not emit warnings")


def test_continuous_export_missing_metadata() -> None:
  """Verify missing continuous-export metadata becomes a warning-only diagnosis."""
  diag = ru.continuous_export_diagnostics(
      {
          "backend": "continuous2d",
          "continuous_export": {
              "frames": [{"step": 0, "density_xy_flat": [0.1, 0.2]}],
          },
      },
      steps=8,
  )
  assert_true(diag["schema_version"] == "n/a", "missing schema version should be surfaced as n/a")
  assert_true("missing schema_version" in diag["diagnostic_tag"], "missing schema version should be warned")
  assert_true("missing export_config" in diag["diagnostic_tag"], "missing export config should be warned")
  assert_true("observed 1 frame(s)" in diag["frame_sampling"], "sampling note should fall back to an observed cadence")


def test_circuit_capability_summary() -> None:
  """Verify circuit capability metadata is summarized for reports."""
  capabilities = {
      "backend": "statevector",
      "supported_gates": ["H", "CNOT", "CX", "SWAP"],
      "supported_noise_models": ["none", "depolarizing"],
      "max_qubits": 25,
      "max_shots": 1000,
      "supports_qiskit_like": True,
      "supports_cirq_like": False,
  }
  diag = ru.circuit_capability_diagnostics(capabilities)
  assert_true(diag["status"] == "ok", "complete capability payload should not warn")
  assert_true("backend=statevector" in diag["capability_summary"], "capability summary should include backend")
  assert_true("gates=[H, CNOT, CX, SWAP]" in diag["capability_summary"], "capability summary should include gates")
  assert_true("noise=[none, depolarizing]" in diag["capability_summary"], "capability summary should include noise models")
  assert_true("adapters={qiskit_like=true, cirq_like=false}" in diag["capability_summary"], "capability summary should include adapter hints")
  assert_true(diag["supported_gate_count"] == 4, "capability summary should count supported gates")
  assert_true(diag["supported_noise_model_count"] == 2, "capability summary should count supported noise models")

  report = {
      "suite": "circuit_accuracy_smoke",
      "generated_at": "2026-03-20T00:00:00+00:00",
      "repeats": 3,
      "circuit_capability_summary": diag["capability_summary"],
      "circuit_capability_diag": diag,
      "cases": [
          {
              "id": "bell_clean",
              "aggregate": {
                  "elapsed_ms_mean": 1.0,
                  "core_elapsed_ms_mean": 0.9,
                  "max_abs_error_mean": 0.01,
                  "mean_abs_error_mean": 0.005,
              },
              "gate": {"status": "PASS"},
          }
      ],
      "gate": {"status": "PASS", "perf_warnings_blocking": False},
  }
  out_dir = ROOT / "out" / "benchmark_reporting_smoke"
  out_dir.mkdir(parents=True, exist_ok=True)
  path = out_dir / "circuit_summary.md"
  try:
    rcb.write_markdown_summary(report, path)
    content = path.read_text(encoding="utf-8")
    assert_true("circuit_capability_summary" in content, "markdown summary should include capability summary")
    assert_true("circuit_capability_diag" in content, "markdown summary should include capability diagnostics")
  finally:
    if path.exists():
      path.unlink()


def test_circuit_capability_missing_fields_warning_only() -> None:
  """Verify missing circuit capability fields become warnings, not failures."""
  diag = ru.circuit_capability_diagnostics({"backend": "statevector"})
  assert_true(diag["status"] == "warn", "missing capability fields should only warn")
  assert_true("missing supported_gates" in diag["diagnostic_tag"], "missing gates should be warned")
  assert_true("missing supported_noise_models" in diag["diagnostic_tag"], "missing noise models should be warned")
  assert_true("missing max_qubits" in diag["diagnostic_tag"], "missing max_qubits should be warned")
  assert_true("missing max_shots" in diag["diagnostic_tag"], "missing max_shots should be warned")
  assert_true("backend=statevector" in diag["capability_summary"], "summary should preserve the available backend")


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
                  "kind": "continuous_space_2d",
                  "grid": "4x5",
                  "export_volume": 40,
                  "fft_provider": "eigen",
                  "avg_step_ms_mean": 1.0,
                  "steps": 10,
              },
              "runs": [
                  {
                      "result": {
                          "continuous_export": {
                              "schema_version": 2,
                              "export_config": {
                                  "frame_stride": 2,
                                  "density_floor_ratio": 0.02,
                              },
                              "frames": [
                                  {"step": 0, "density_xy_flat": [0.1] * 20},
                                  {"step": 2, "density_xy_flat": [0.2] * 20},
                              ],
                          }
                      }
                  }
              ],
          },
          {
              "aggregate": {
                  "core_elapsed_ms_mean": 11.0,
                  "mean_abs_error_mean": 0.02,
                  "max_abs_error_mean": 0.03,
              },
              "performance": {
                  "kind": "continuous_space_2d",
                  "grid": "4x5",
                  "export_volume": 40,
                  "fft_provider": "eigen",
                  "avg_step_ms_mean": 1.1,
                  "steps": 10,
              },
              "runs": [
                  {
                      "result": {
                          "continuous_export": {
                              "frames": [
                                  {"step": 0, "density_xy_flat": [0.1] * 20},
                              ],
                          }
                      }
                  }
              ],
          }
      ],
  }
  summary = sr.summarize(report)
  assert_true(summary["platform_profile"] == "github-windows", "summary should keep profile")
  assert_true(summary["perf_warning_policy"] == "non-blocking", "summary should label perf policy")
  assert_true(summary["grid"] == "4x5", "summary should expose the grid label")
  assert_true(summary["export_volume"] == 80, "summary should expose aggregated export volume")
  assert_true(summary["fft_provider"] == "eigen", "summary should expose FFT provider")
  assert_true(summary["continuous_export_schema"] == "mixed: 2, n/a", "summary should expose continuous export schema versions")
  assert_true("frame_stride=2" in summary["continuous_export_config"], "summary should expose export config summary")
  assert_true("explicit sampling" in summary["continuous_export_sampling"], "summary should explain the sampling rate")
  assert_true("warn:" in summary["continuous_export_diag"], "summary should retain warning tags for incomplete export metadata")

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
  test_circuit_capability_summary()
  test_circuit_capability_missing_fields_warning_only()
  print("benchmark reporting smoke passed")


if __name__ == "__main__":
  main()
