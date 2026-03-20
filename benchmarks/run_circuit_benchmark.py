import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks.report_utils import detect_platform_profile, suite_policy


def load_json(path: Path) -> dict:
  """Load a JSON file from disk."""
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def run_case(engine: "Engine", root: Path, case: dict, options: dict, repeats: int) -> dict:
  """Execute one circuit benchmark case repeatedly and aggregate the results."""
  circuit_path = root / case["circuit_file"]
  circuit = load_json(circuit_path)
  runs = []

  for i in range(repeats):
    start = time.perf_counter()
    result = engine.run_circuit(circuit, options)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    runs.append({
        "run_index": i,
        "elapsed_ms": elapsed_ms,
        "result": result,
    })

  max_abs_errors = []
  mean_abs_errors = []
  core_elapsed = []
  for run in runs:
    summary = run["result"].get("error_summary", {})
    if "max_abs_error" in summary:
      max_abs_errors.append(float(summary["max_abs_error"]))
    if "mean_abs_error" in summary:
      mean_abs_errors.append(float(summary["mean_abs_error"]))
    if "elapsed_ms" in run["result"]:
      core_elapsed.append(float(run["result"]["elapsed_ms"]))

  elapsed = [r["elapsed_ms"] for r in runs]
  aggregate = {
      "elapsed_ms_mean": statistics.fmean(elapsed) if elapsed else 0.0,
      "elapsed_ms_min": min(elapsed) if elapsed else 0.0,
      "elapsed_ms_max": max(elapsed) if elapsed else 0.0,
      "core_elapsed_ms_mean": statistics.fmean(core_elapsed) if core_elapsed else None,
      "max_abs_error_mean": statistics.fmean(max_abs_errors) if max_abs_errors else None,
      "mean_abs_error_mean": statistics.fmean(mean_abs_errors) if mean_abs_errors else None,
  }
  engine_version = None
  if runs:
    engine_version = runs[0]["result"].get("engine_version")

  return {
      "id": case["id"],
      "circuit_file": case["circuit_file"],
      "gates": dict(case.get("gates", {})),
      "engine_version": engine_version,
      "aggregate": aggregate,
      "runs": runs,
  }


def write_markdown_summary(report: dict, output_path: Path) -> None:
  """Write a compact human-readable markdown summary for circuit benchmarks."""
  lines = []
  lines.append(f"# Benchmark Summary: {report['suite']}")
  lines.append("")
  lines.append(f"- generated_at: {report['generated_at']}")
  lines.append(f"- repeats: {report['repeats']}")
  if report.get("platform_profile"):
    lines.append(f"- platform_profile: {report['platform_profile']}")
  if report.get("engine_version"):
    lines.append(f"- engine_version: {report['engine_version']}")
  gate_status = report.get("gate", {}).get("status")
  if gate_status is not None:
    lines.append(f"- gate_status: {gate_status}")
  gate_policy = report.get("gate", {}).get("perf_warnings_blocking")
  if gate_policy is not None:
    label = "blocking" if gate_policy else "non-blocking"
    lines.append(f"- perf_warning_policy: {label}")
  lines.append("")
  lines.append("| case | status | elapsed_ms_mean | core_elapsed_ms_mean | max_abs_error_mean | mean_abs_error_mean |")
  lines.append("|---|---|---:|---:|---:|---:|")
  for case in report["cases"]:
    agg = case["aggregate"]
    status = case.get("gate", {}).get("status", "N/A")
    max_err = "n/a" if agg["max_abs_error_mean"] is None else f"{agg['max_abs_error_mean']:.6f}"
    mean_err = "n/a" if agg["mean_abs_error_mean"] is None else f"{agg['mean_abs_error_mean']:.6f}"
    core_elapsed = "n/a" if agg["core_elapsed_ms_mean"] is None else f"{agg['core_elapsed_ms_mean']:.3f}"
    lines.append(
        f"| {case['id']} | {status} | {agg['elapsed_ms_mean']:.3f} | "
        f"{core_elapsed} | {max_err} | {mean_err} |"
    )
  output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_gates(report_cases: list[dict], gates: dict, perf_warnings_blocking: bool) -> dict:
  """Evaluate circuit benchmark gates with performance warnings optionally non-blocking."""
  if not gates:
    for case in report_cases:
      case["gate"] = {"status": "SKIPPED", "reasons": []}
    return {"status": "SKIPPED", "failures": [], "warnings": [], "perf_warnings_blocking": perf_warnings_blocking}

  failures = []
  warnings = []
  for case in report_cases:
    case_failures = []
    case_warnings = []
    agg = case.get("aggregate", {})
    effective_gates = dict(gates)
    effective_gates.update(case.get("gates", {}))

    def check_le(metric: str, threshold_key: str) -> None:
      threshold = effective_gates.get(threshold_key)
      if threshold is None:
        return
      value = agg.get(metric)
      if value is None:
        if metric == "core_elapsed_ms_mean" and not perf_warnings_blocking:
          case_warnings.append(f"missing metric {metric}")
        else:
          case_failures.append(f"missing metric {metric}")
        return
      if float(value) > float(threshold):
        msg = f"{metric}={value:.6f} > {threshold_key}={threshold:.6f}"
        if metric == "core_elapsed_ms_mean" and not perf_warnings_blocking:
          case_warnings.append(msg)
        else:
          case_failures.append(msg)

    check_le("max_abs_error_mean", "max_abs_error_mean_max")
    check_le("mean_abs_error_mean", "mean_abs_error_mean_max")
    check_le("core_elapsed_ms_mean", "core_elapsed_ms_mean_max")

    if case_failures:
      case["gate"] = {"status": "FAIL", "reasons": case_failures}
      failures.append({"case": case["id"], "reasons": case_failures})
    else:
      case["gate"] = {"status": "PASS", "reasons": []}
    if case_warnings:
      warnings.append({"case": case["id"], "reasons": case_warnings})

  return {
      "status": "FAIL" if failures else "PASS",
      "failures": failures,
      "warnings": warnings,
      "perf_warnings_blocking": perf_warnings_blocking,
  }


def main() -> None:
  """Run the circuit benchmark suite and emit JSON/Markdown reports."""
  parser = argparse.ArgumentParser(description="Run QuantX circuit benchmark suite.")
  parser.add_argument(
      "--suite",
      type=str,
      default="benchmarks/suites/circuit_accuracy_smoke.json",
      help="Path to suite JSON, relative to repo root.",
  )
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(root / "python"))
  from quantx_engine import Engine

  suite_path = root / args.suite
  suite = load_json(suite_path)
  profile = detect_platform_profile()
  policy = suite_policy(suite.get("name", "unnamed_suite"), profile, suite, root)

  engine = Engine(str(root / "build-vs" / "Debug" / "quantx.dll"))

  repeats = int(suite.get("repeats", 1))
  options = dict(suite.get("options", {}))
  gates = dict(suite.get("gates", {}))
  report_cases = []
  for case in suite.get("cases", []):
    report_cases.append(run_case(engine, root, case, options, repeats))
  gate = evaluate_gates(report_cases, gates, perf_warnings_blocking=policy["perf_warnings_blocking"])
  case_versions = {c.get("engine_version") for c in report_cases if c.get("engine_version")}
  report_engine_version = sorted(case_versions)[0] if case_versions else None

  report = {
      "suite": suite.get("name", "unnamed_suite"),
      "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
      "repeats": repeats,
      "engine_version": report_engine_version,
      "platform_profile": policy["profile"],
      "platform_profile_label": policy["profile_label"],
      "suite_policy": policy,
      "cases": report_cases,
      "gate": gate,
  }

  out_dir = root / "benchmarks" / "reports"
  out_dir.mkdir(parents=True, exist_ok=True)
  ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
  out_file = out_dir / f"{suite.get('name','suite')}_{ts}.json"
  with out_file.open("w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
  print(f"wrote benchmark report: {out_file}")
  summary_file = out_dir / f"{suite.get('name','suite')}_{ts}.md"
  write_markdown_summary(report, summary_file)
  print(f"wrote benchmark summary: {summary_file}")

  for warning in gate.get("warnings", []):
    print(f"gate warning: {warning['case']}")
    for reason in warning.get("reasons", []):
      print(f"- WARN: {reason}")


if __name__ == "__main__":
  main()
