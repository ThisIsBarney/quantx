import json
from pathlib import Path


def load_reports(report_dir: Path) -> list[dict]:
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
  case_count = 0
  mean_elapsed = []
  mean_err = []
  max_err = []
  for case in report.get("cases", []):
    case_count += 1
    agg = case.get("aggregate", {})
    if agg.get("core_elapsed_ms_mean") is not None:
      mean_elapsed.append(float(agg["core_elapsed_ms_mean"]))
    if agg.get("mean_abs_error_mean") is not None:
      mean_err.append(float(agg["mean_abs_error_mean"]))
    if agg.get("max_abs_error_mean") is not None:
      max_err.append(float(agg["max_abs_error_mean"]))

  def avg(xs: list[float]) -> float | None:
    if not xs:
      return None
    return sum(xs) / len(xs)

  return {
      "file": report.get("_file", "unknown"),
      "suite": report.get("suite", "unknown"),
      "generated_at": report.get("generated_at", "unknown"),
      "cases": case_count,
      "core_elapsed_ms_mean": avg(mean_elapsed),
      "mean_abs_error_mean": avg(mean_err),
      "max_abs_error_mean": avg(max_err),
  }


def fmt(v: float | None, ndigits: int = 6) -> str:
  if v is None:
    return "n/a"
  return f"{v:.{ndigits}f}"


def main() -> None:
  root = Path(__file__).resolve().parents[1]
  report_dir = root / "benchmarks" / "reports"
  report_dir.mkdir(parents=True, exist_ok=True)

  reports = load_reports(report_dir)
  rows = [summarize(r) for r in reports]

  lines = []
  lines.append("# Benchmark Report Index")
  lines.append("")
  lines.append("| file | suite | generated_at | cases | core_elapsed_ms_mean | mean_abs_error_mean | max_abs_error_mean |")
  lines.append("|---|---|---|---:|---:|---:|---:|")
  for row in rows:
    lines.append(
        f"| {row['file']} | {row['suite']} | {row['generated_at']} | {row['cases']} | "
        f"{fmt(row['core_elapsed_ms_mean'], 3)} | {fmt(row['mean_abs_error_mean'])} | {fmt(row['max_abs_error_mean'])} |"
    )

  out = report_dir / "INDEX.md"
  out.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"wrote report index: {out}")


if __name__ == "__main__":
  main()
