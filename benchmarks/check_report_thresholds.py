import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks.report_utils import detect_platform_profile, suite_policy


def load_json(path: Path) -> dict:
  """Load a JSON file from disk."""
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def find_latest_report(report_dir: Path, suite_name: str) -> Path | None:
  """Return the newest report JSON for a suite, or ``None`` if missing."""
  candidates = sorted(report_dir.glob(f"{suite_name}_*.json"))
  return candidates[-1] if candidates else None


def merged_case_gates(global_gates: dict, case: dict) -> dict:
  """Merge suite-level gates with case-level overrides."""
  case_gates = case.get("gates", {})
  if not isinstance(case_gates, dict):
    case_gates = {}
  out = dict(global_gates)
  out.update(case_gates)
  return out


def check_case(case: dict, gates: dict, perf_warnings_blocking: bool) -> tuple[list[str], list[str]]:
  """Check one benchmark case and split failures from warnings."""
  agg = case.get("aggregate", {})
  failures: list[str] = []
  warnings: list[str] = []

  ok_rate_min = gates.get("ok_rate_min")
  if ok_rate_min is not None:
    ok_rate = case.get("ok_rate")
    if ok_rate is None:
      failures.append(f"{case['id']} missing metric ok_rate")
    elif float(ok_rate) < float(ok_rate_min):
      failures.append(f"{case['id']} ok_rate={ok_rate:.6f} < ok_rate_min={ok_rate_min:.6f}")

  def check_le(metric: str, threshold_key: str, is_performance: bool = False) -> None:
    threshold = gates.get(threshold_key)
    if threshold is None:
      return
    value = agg.get(metric)
    if value is None:
      target = failures if (not is_performance or perf_warnings_blocking) else warnings
      target.append(f"{case['id']} missing metric {metric}")
      return
    if float(value) > float(threshold):
      target = failures if (not is_performance or perf_warnings_blocking) else warnings
      target.append(f"{case['id']} {metric}={value:.6f} > {threshold_key}={threshold:.6f}")

  check_le("max_abs_error_mean", "max_abs_error_mean_max")
  check_le("mean_abs_error_mean", "mean_abs_error_mean_max")
  check_le("core_elapsed_ms_mean", "core_elapsed_ms_mean_max", is_performance=True)
  check_le("norm_abs_error_mean", "norm_abs_error_mean_max")
  check_le("sigma_x_abs_error_mean", "sigma_x_abs_error_mean_max")
  check_le("sigma_z_abs_error_mean", "sigma_z_abs_error_mean_max")
  return failures, warnings


def main() -> None:
  """Check the latest benchmark report for the requested suite."""
  parser = argparse.ArgumentParser(description="Check latest benchmark report against suite gates.")
  parser.add_argument("--suite", required=True, help="Suite JSON path relative to repo root.")
  parser.add_argument(
    "--report",
    required=False,
    default="",
    help="Optional report JSON path relative to repo root. If omitted, use latest suite report.",
  )
  parser.add_argument(
      "--perf-warn-only",
      action="store_true",
      help="Treat performance threshold violations as warnings (non-blocking).",
  )
  parser.add_argument(
      "--platform-profile",
      default="",
      help="Optional platform profile name (local, github-windows, ...). Defaults to environment detection.",
  )
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[1]
  suite = load_json(root / args.suite)
  suite_name = suite.get("name", "")
  gates = suite.get("gates", {})
  profile_name = args.platform_profile or detect_platform_profile()
  policy = suite_policy(suite_name, profile_name, suite, root)
  if not suite_name:
    print("suite.name is required for gating", file=sys.stderr)
    sys.exit(2)
  if not gates:
    print(f"suite {suite_name} has no gates configured, treat as pass")
    sys.exit(0)

  report_dir = root / "benchmarks" / "reports"
  if args.report:
    latest = root / args.report
    if not latest.exists():
      print(f"report not found: {latest}", file=sys.stderr)
      sys.exit(2)
  else:
    latest = find_latest_report(report_dir, suite_name)
    if latest is None:
      print(f"no report found for suite {suite_name} in {report_dir}", file=sys.stderr)
      sys.exit(2)

  report = load_json(latest)
  failures: list[str] = []
  warnings: list[str] = []
  perf_warnings_blocking = bool(policy["perf_warnings_blocking"]) and not args.perf_warn_only
  for case in report.get("cases", []):
    effective_gates = merged_case_gates(gates, case)
    case_failures, case_warnings = check_case(case, effective_gates, perf_warnings_blocking=perf_warnings_blocking)
    failures.extend(case_failures)
    warnings.extend(case_warnings)

  if warnings:
    print(f"gate warnings for report: {latest}")
    for w in warnings:
      print(f"- WARN: {w}")

  if failures:
    print(f"gate failed for report: {latest}")
    for f in failures:
      print(f"- {f}")
    sys.exit(1)

  print(f"gate passed for report: {latest}")


if __name__ == "__main__":
  main()
