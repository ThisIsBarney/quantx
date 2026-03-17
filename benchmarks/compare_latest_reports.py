import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def latest_two(report_dir: Path, suite_name: str) -> list[Path]:
  files = sorted(report_dir.glob(f"{suite_name}_*.json"))
  return files[-2:] if len(files) >= 2 else files


def case_map(report: dict) -> dict:
  out = {}
  for case in report.get("cases", []):
    out[case["id"]] = case.get("aggregate", {})
  return out


def fmt(v: float | None, digits: int = 6) -> str:
  if v is None:
    return "n/a"
  return f"{v:.{digits}f}"


def delta(curr: float | None, prev: float | None) -> float | None:
  if curr is None or prev is None:
    return None
  return curr - prev


def sign_label(v: float | None, lower_is_better: bool = True) -> str:
  if v is None:
    return "N/A"
  if abs(v) < 1e-12:
    return "SAME"
  if lower_is_better:
    return "IMPROVED" if v < 0 else "REGRESSED"
  return "IMPROVED" if v > 0 else "REGRESSED"


def main() -> None:
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
  prev_cases = case_map(prev)
  curr_cases = case_map(curr)
  ids = sorted(set(prev_cases.keys()) & set(curr_cases.keys()))

  lines = []
  lines.append(f"# Trend: {args.suite}")
  lines.append("")
  lines.append(f"- previous: {prev_path.name}")
  lines.append(f"- current: {curr_path.name}")
  lines.append("")
  lines.append("| case | metric | prev | curr | delta | status |")
  lines.append("|---|---|---:|---:|---:|---|")

  metrics = ["core_elapsed_ms_mean", "mean_abs_error_mean", "max_abs_error_mean"]
  for cid in ids:
    for m in metrics:
      pv = prev_cases[cid].get(m)
      cv = curr_cases[cid].get(m)
      dv = delta(cv, pv)
      status = sign_label(dv, lower_is_better=True)
      lines.append(f"| {cid} | {m} | {fmt(pv)} | {fmt(cv)} | {fmt(dv)} | {status} |")

  out = report_dir / f"{args.suite}_TREND.md"
  out.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"wrote trend report: {out}")


if __name__ == "__main__":
  main()
