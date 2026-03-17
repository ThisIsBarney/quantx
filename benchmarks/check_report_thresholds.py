import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def find_latest_report(report_dir: Path, suite_name: str) -> Path | None:
  candidates = sorted(report_dir.glob(f"{suite_name}_*.json"))
  return candidates[-1] if candidates else None


def merged_case_gates(global_gates: dict, case: dict) -> dict:
  case_gates = case.get("gates", {})
  if not isinstance(case_gates, dict):
    case_gates = {}
  out = dict(global_gates)
  out.update(case_gates)
  return out


def check_case(case: dict, gates: dict) -> list[str]:
  agg = case.get("aggregate", {})
  failures = []

  def check_le(metric: str, threshold_key: str) -> None:
    threshold = gates.get(threshold_key)
    if threshold is None:
      return
    value = agg.get(metric)
    if value is None:
      failures.append(f"{case['id']} missing metric {metric}")
      return
    if float(value) > float(threshold):
      failures.append(f"{case['id']} {metric}={value:.6f} > {threshold_key}={threshold:.6f}")

  check_le("max_abs_error_mean", "max_abs_error_mean_max")
  check_le("mean_abs_error_mean", "mean_abs_error_mean_max")
  check_le("core_elapsed_ms_mean", "core_elapsed_ms_mean_max")
  return failures


def main() -> None:
  parser = argparse.ArgumentParser(description="Check latest benchmark report against suite gates.")
  parser.add_argument("--suite", required=True, help="Suite JSON path relative to repo root.")
  parser.add_argument(
      "--report",
      required=False,
      default="",
      help="Optional report JSON path relative to repo root. If omitted, use latest suite report.",
  )
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[1]
  suite = load_json(root / args.suite)
  suite_name = suite.get("name", "")
  gates = suite.get("gates", {})
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
  for case in report.get("cases", []):
    effective_gates = merged_case_gates(gates, case)
    failures.extend(check_case(case, effective_gates))

  if failures:
    print(f"gate failed for report: {latest}")
    for f in failures:
      print(f"- {f}")
    sys.exit(1)

  print(f"gate passed for report: {latest}")


if __name__ == "__main__":
  main()
