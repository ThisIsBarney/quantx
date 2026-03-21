import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks import summarize_reports as sr


def load_json(path: Path) -> dict:
  """Load a JSON report from disk."""
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def latest_two(report_dir: Path, suite_name: str) -> list[Path]:
  """Return the two newest reports for one suite."""
  files = sorted(report_dir.glob(f"{suite_name}_*.json"))
  return files[-2:] if len(files) >= 2 else files


def case_map(report: dict) -> dict:
  """Convert a report into a per-case lookup table."""
  out = {}
  for case in report.get("cases", []):
    out[case["id"]] = {
        "aggregate": case.get("aggregate", {}),
        "performance": case.get("performance", {}),
  }
  return out


def build_trend_lines(prev: dict, curr: dict, prev_name: str, curr_name: str, suite_name: str) -> list[str]:
  """Build the markdown lines for a trend report."""
  prev_cases = case_map(prev)
  curr_cases = case_map(curr)
  ids = sorted(set(prev_cases.keys()) & set(curr_cases.keys()))

  lines = []
  lines.append(f"# Trend: {suite_name}")
  lines.append("")
  lines.append(f"- previous: {prev_name}")
  lines.append(f"- current: {curr_name}")
  if curr.get("platform_profile"):
    lines.append(f"- platform_profile: {curr.get('platform_profile')}")
  gate_policy = curr.get("gate", {}).get("perf_warnings_blocking")
  if gate_policy is not None:
    label = "blocking" if gate_policy else "non-blocking"
    lines.append(f"- perf_warning_policy: {label}")
    if not gate_policy:
      lines.append("- suite_label: perf warnings non-blocking")
  capability_summary = curr.get("circuit_capability_summary")
  capability_diag = curr.get("circuit_capability_diag", {})
  if capability_summary:
    lines.append(f"- circuit_capability_summary: {capability_summary}")
  if isinstance(capability_diag, dict) and capability_diag.get("diagnostic_tag"):
    lines.append(f"- circuit_capability_diag: {capability_diag['diagnostic_tag']}")
    if capability_diag.get("status") != "ok":
      lines.append("- note: capability metadata is advisory; missing fields are warnings and do not change accuracy gate outcomes.")
  lines.extend(f"- {line}" for line in sr.trend_legend_lines())
  lines.append("")
  lines.append(
      "| case | grid | export_volume | fft_provider | avg_step_ms_mean | metric | prev | curr | delta | status | class |"
  )
  lines.append("|---|---|---|---:|---|---|---:|---:|---:|---|---|")

  metrics = ["core_elapsed_ms_mean", "mean_abs_error_mean", "max_abs_error_mean"]
  for cid in ids:
    perf = curr_cases[cid].get("performance", {})
    grid = perf.get("grid", "n/a")
    export_volume = perf.get("export_volume", "n/a")
    fft_provider = perf.get("fft_provider", "n/a")
    avg_step_ms = perf.get("avg_step_ms_mean", None)
    for metric in metrics:
      pv = prev_cases[cid].get("aggregate", {}).get(metric)
      cv = curr_cases[cid].get("aggregate", {}).get(metric)
      dv = delta(cv, pv)
      status = sign_label(dv, lower_is_better=True)
      issue_class = sr.trend_issue_classification(metric, pv, cv, perf)
      lines.append(
          f"| {cid} | {grid} | {export_volume} | {fft_provider} | {fmt(avg_step_ms, 3)} | "
          f"{metric} | {fmt(pv)} | {fmt(cv)} | {fmt(dv)} | {status} | {issue_class} |"
      )

  return lines


def fmt(v: float | None, digits: int = 6) -> str:
  """Format an optional float for trend tables."""
  if v is None:
    return "n/a"
  return f"{v:.{digits}f}"


def delta(curr: float | None, prev: float | None) -> float | None:
  """Compute a signed delta, preserving missing values."""
  if curr is None or prev is None:
    return None
  return curr - prev


def sign_label(v: float | None, lower_is_better: bool = True) -> str:
  """Convert a delta into a human-readable trend label."""
  if v is None:
    return "N/A"
  if abs(v) < 1e-12:
    return "SAME"
  if lower_is_better:
    return "IMPROVED" if v < 0 else "REGRESSED"
  return "IMPROVED" if v > 0 else "REGRESSED"


def main() -> None:
  """Generate a Markdown trend report from the newest two suite reports."""
  parser = argparse.ArgumentParser(description="Compare latest two reports for a suite.")
  parser.add_argument("--suite", required=True, help="Suite name (not path), e.g. circuit_accuracy_smoke")
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[1]
  report_dir = root / "benchmarks" / "reports"
  pair = latest_two(report_dir, args.suite)
  if len(pair) < 2:
    print(f"need at least two reports for suite {args.suite}")
    return

  prev_path, curr_path = pair[0], pair[1]
  prev = load_json(prev_path)
  curr = load_json(curr_path)
  lines = build_trend_lines(prev, curr, prev_path.name, curr_path.name, args.suite)

  out = report_dir / f"{args.suite}_TREND.md"
  out.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"wrote trend report: {out}")


if __name__ == "__main__":
  main()
