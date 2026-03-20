import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))


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

  def avg(xs: list[float]) -> float | None:
    if not xs:
      return None
    return sum(xs) / len(xs)

  def joined_unique(xs: list[str]) -> str:
    if not xs:
      return "n/a"
    uniq = sorted(set(xs))
    return ", ".join(uniq)

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
      "fft_provider": joined_unique(fft_providers) if fft_providers else "n/a",
      "avg_step_ms_mean": avg(avg_step_ms),
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
  lines.append(
      "| file | suite | profile | perf_policy | generated_at | cases | core_elapsed_ms_mean | mean_abs_error_mean | "
      "max_abs_error_mean | grid | export_volume | fft_provider | avg_step_ms_mean |"
  )
  lines.append("|---|---|---|---|---|---:|---:|---:|---:|---|---:|---|---:|")
  for row in rows:
    lines.append(
        f"| {row['file']} | {row['suite']} | {row['platform_profile']} | {row['perf_warning_policy']} | "
        f"{row['generated_at']} | {row['cases']} | {fmt(row['core_elapsed_ms_mean'], 3)} | "
        f"{fmt(row['mean_abs_error_mean'])} | {fmt(row['max_abs_error_mean'])} | {row['grid']} | "
        f"{fmt(row['export_volume'], 0)} | {row['fft_provider']} | {fmt(row['avg_step_ms_mean'], 3)} |"
    )

  out = report_dir / "INDEX.md"
  out.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"wrote report index: {out}")


if __name__ == "__main__":
  main()
