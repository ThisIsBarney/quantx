import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks import report_utils as ru


def load_reports(report_dir: Path) -> list[dict]:
  """Load every JSON report in a directory, skipping unreadable files."""
  reports = []
  for path in sorted(report_dir.glob("*.json")):
    try:
      data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
      continue
    data["_file"] = path.name
    reports.append(data)
  return reports


def first_case_result(case: dict) -> dict:
  """Return the first run result attached to a benchmark case."""
  for run in case.get("runs", []):
    result = run.get("result")
    if isinstance(result, dict):
      return result
  return {}


def trend_legend_lines() -> list[str]:
  """Return the canonical legend text used by trend reports."""
  return [
      "legend: accuracy regression = a precision metric worsened compared with the previous report.",
      "legend: perf warning = a performance metric worsened; the suite gate may still block it depending on policy.",
      "legend: metadata warning = the comparison is incomplete because a report field is missing or undecodable.",
  ]


def _trend_missing_context(perf: dict) -> bool:
  """Return True when trend context is incomplete enough to warrant a metadata warning."""
  for key in ("grid", "export_volume", "fft_provider", "avg_step_ms_mean"):
    value = perf.get(key)
    if value is None:
      return True
    if isinstance(value, str) and value.strip().lower() == "n/a":
      return True
  return False


def trend_issue_classification(metric: str, prev: float | None, curr: float | None, perf: dict) -> str:
  """Classify one trend row into an explainable issue bucket."""
  if prev is None or curr is None:
    return "metadata warning"
  if _trend_missing_context(perf):
    return "metadata warning"

  delta = curr - prev
  if abs(delta) < 1e-12:
    return "ok"
  if metric in {"mean_abs_error_mean", "max_abs_error_mean"} and delta > 0:
    return "accuracy regression"
  if metric == "core_elapsed_ms_mean" and delta > 0:
    return "perf warning"
  return "ok"


def summarize(report: dict) -> dict:
  """Collapse one report into a table-friendly summary row."""
  case_count = 0
  mean_elapsed = []
  mean_err = []
  max_err = []
  grid_labels = []
  export_volume = []
  fft_providers = []
  avg_step_ms = []
  continuous_export_schema = []
  continuous_export_config = []
  continuous_export_sampling = []
  continuous_export_diag = []
  for case in report.get("cases", []):
    case_count += 1
    agg = case.get("aggregate", {})
    perf = case.get("performance", {})
    if agg.get("core_elapsed_ms_mean") is not None:
      mean_elapsed.append(float(agg["core_elapsed_ms_mean"]))
    if agg.get("mean_abs_error_mean") is not None:
      mean_err.append(float(agg["mean_abs_error_mean"]))
    if agg.get("max_abs_error_mean") is not None:
      max_err.append(float(agg["max_abs_error_mean"]))
    if perf.get("grid") is not None:
      grid_labels.append(str(perf["grid"]))
    if perf.get("export_volume") is not None:
      export_volume.append(int(perf["export_volume"]))
    if perf.get("fft_provider") is not None:
      fft_providers.append(str(perf["fft_provider"]))
    if perf.get("avg_step_ms_mean") is not None:
      avg_step_ms.append(float(perf["avg_step_ms_mean"]))
    diag = ru.continuous_export_diagnostics(first_case_result(case), int(perf["steps"]) if perf.get("steps") is not None else None)
    if diag["has_continuous_export"] or str(perf.get("kind", "")).startswith("continuous"):
      continuous_export_schema.append(str(diag["schema_version"]))
      continuous_export_config.append(str(diag["export_config_summary"]))
      continuous_export_sampling.append(str(diag["frame_sampling"]))
      continuous_export_diag.append(str(diag["diagnostic_tag"]))

  def avg(xs: list[float]) -> float | None:
    if not xs:
      return None
    return sum(xs) / len(xs)

  def joined_unique(xs: list[str]) -> str:
    """Join unique string values into a stable human-readable token."""
    uniq = sorted({x for x in xs if x})
    if not uniq:
      return "n/a"
    if len(uniq) == 1:
      return uniq[0]
    return "mixed: " + ", ".join(uniq)

  return {
      "file": report.get("_file", "unknown"),
      "suite": report.get("suite", "unknown"),
      "generated_at": report.get("generated_at", "unknown"),
      "platform_profile": report.get("platform_profile", "unknown"),
      "perf_warning_policy": "blocking" if report.get("gate", {}).get("perf_warnings_blocking") else "non-blocking",
      "cases": case_count,
      "core_elapsed_ms_mean": avg(mean_elapsed),
      "mean_abs_error_mean": avg(mean_err),
      "max_abs_error_mean": avg(max_err),
      "grid": joined_unique(grid_labels),
      "export_volume": sum(export_volume) if export_volume else None,
      "fft_provider": joined_unique(fft_providers),
      "avg_step_ms_mean": avg(avg_step_ms),
      "continuous_export_schema": joined_unique(continuous_export_schema),
      "continuous_export_config": joined_unique(continuous_export_config),
      "continuous_export_sampling": joined_unique(continuous_export_sampling),
      "continuous_export_diag": joined_unique(continuous_export_diag),
  }


def fmt(v: float | None, ndigits: int = 6) -> str:
  """Format an optional float for Markdown tables."""
  if v is None:
    return "n/a"
  return f"{v:.{ndigits}f}"


def main() -> None:
  """Build the benchmark report index from the latest JSON outputs."""
  root = Path(__file__).resolve().parents[1]
  report_dir = root / "benchmarks" / "reports"
  report_dir.mkdir(parents=True, exist_ok=True)

  reports = load_reports(report_dir)
  rows = [summarize(r) for r in reports]

  lines = []
  lines.append("# Benchmark Report Index")
  lines.append("")
  lines.append("- note: a passing gate means the configured thresholds passed for this run; it does not mean the latest result has no historical trend regression.")
  lines.append("- note: continuous_export diagnostics are advisory labels. `warn` indicates missing metadata, not a failed benchmark gate.")
  lines.append("")
  lines.append(
      "| file | suite | profile | perf_policy | generated_at | cases | core_elapsed_ms_mean | mean_abs_error_mean | "
      "max_abs_error_mean | grid | export_volume | fft_provider | avg_step_ms_mean | ce_schema | ce_config | "
      "ce_sampling | ce_diag |"
  )
  lines.append("|---|---|---|---|---|---:|---:|---:|---:|---|---:|---|---:|---|---|---|---|")
  for row in rows:
    lines.append(
        f"| {row['file']} | {row['suite']} | {row['platform_profile']} | {row['perf_warning_policy']} | "
        f"{row['generated_at']} | {row['cases']} | {fmt(row['core_elapsed_ms_mean'], 3)} | "
        f"{fmt(row['mean_abs_error_mean'])} | {fmt(row['max_abs_error_mean'])} | {row['grid']} | "
        f"{fmt(row['export_volume'], 0)} | {row['fft_provider']} | {fmt(row['avg_step_ms_mean'], 3)} | "
        f"{row['continuous_export_schema']} | {row['continuous_export_config']} | {row['continuous_export_sampling']} | "
        f"{row['continuous_export_diag']} |"
    )

  out = report_dir / "INDEX.md"
  out.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"wrote report index: {out}")


if __name__ == "__main__":
  main()
