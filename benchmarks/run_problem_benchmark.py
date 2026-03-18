import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def load_json(path: Path) -> dict:
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def run_case(engine: "Engine", root: Path, case: dict, options: dict, repeats: int) -> dict:
  problem_path = root / case["problem_file"]
  problem = load_json(problem_path)
  runs = []

  for i in range(repeats):
    start = time.perf_counter()
    result = engine.run(problem, options)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    runs.append({
        "run_index": i,
        "elapsed_ms": elapsed_ms,
        "result": result,
    })

  elapsed = [r["elapsed_ms"] for r in runs]
  core_elapsed = []
  norm_abs_error = []
  sigma_x = []
  sigma_y = []

  for run in runs:
    result = run["result"]
    if "elapsed_ms" in result:
      core_elapsed.append(float(result["elapsed_ms"]))
    obs = result.get("observables", {})
    if "norm" in obs:
      norm_abs_error.append(abs(float(obs["norm"]) - 1.0))
    if "sigma_x" in obs:
      sigma_x.append(float(obs["sigma_x"]))
    if "sigma_y" in obs:
      sigma_y.append(float(obs["sigma_y"]))

  aggregate = {
      "elapsed_ms_mean": statistics.fmean(elapsed) if elapsed else 0.0,
      "elapsed_ms_min": min(elapsed) if elapsed else 0.0,
      "elapsed_ms_max": max(elapsed) if elapsed else 0.0,
      "core_elapsed_ms_mean": statistics.fmean(core_elapsed) if core_elapsed else None,
      "norm_abs_error_mean": statistics.fmean(norm_abs_error) if norm_abs_error else None,
      "sigma_x_mean": statistics.fmean(sigma_x) if sigma_x else None,
      "sigma_y_mean": statistics.fmean(sigma_y) if sigma_y else None,
  }

  engine_version = None
  backend = None
  ok_rate = 0.0
  if runs:
    engine_version = runs[0]["result"].get("engine_version")
    backend = runs[0]["result"].get("backend")
    ok_count = sum(1 for r in runs if bool(r["result"].get("ok", False)))
    ok_rate = float(ok_count) / float(len(runs))

  return {
      "id": case["id"],
      "problem_file": case["problem_file"],
      "engine_version": engine_version,
      "backend": backend,
      "ok_rate": ok_rate,
      "aggregate": aggregate,
      "runs": runs,
  }


def write_markdown_summary(report: dict, output_path: Path) -> None:
  lines = []
  lines.append(f"# Problem Benchmark Summary: {report['suite']}")
  lines.append("")
  lines.append(f"- generated_at: {report['generated_at']}")
  lines.append(f"- repeats: {report['repeats']}")
  if report.get("engine_version"):
    lines.append(f"- engine_version: {report['engine_version']}")
  gate_status = report.get("gate", {}).get("status")
  if gate_status is not None:
    lines.append(f"- gate_status: {gate_status}")
  lines.append("")
  lines.append("| case | backend | ok_rate | elapsed_ms_mean | core_elapsed_ms_mean | norm_abs_error_mean | sigma_x_mean | sigma_y_mean |")
  lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

  for case in report["cases"]:
    agg = case["aggregate"]

    def fmt(v, digits=6):
      return "n/a" if v is None else f"{v:.{digits}f}"

    lines.append(
        f"| {case['id']} | {case.get('backend','n/a')} | {case.get('ok_rate',0.0):.2f} | "
        f"{fmt(agg.get('elapsed_ms_mean'), 3)} | {fmt(agg.get('core_elapsed_ms_mean'), 3)} | "
        f"{fmt(agg.get('norm_abs_error_mean'), 6)} | {fmt(agg.get('sigma_x_mean'), 6)} | {fmt(agg.get('sigma_y_mean'), 6)} |"
    )

  output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_gates(report_cases: list[dict], gates: dict) -> dict:
  if not gates:
    return {"status": "SKIPPED", "failures": []}

  failures = []
  for case in report_cases:
    agg = case.get("aggregate", {})
    case_failures = []

    ok_rate_min = gates.get("ok_rate_min")
    if ok_rate_min is not None and float(case.get("ok_rate", 0.0)) < float(ok_rate_min):
      case_failures.append(f"ok_rate={case.get('ok_rate',0.0):.3f} < ok_rate_min={float(ok_rate_min):.3f}")

    norm_err_max = gates.get("norm_abs_error_mean_max")
    if norm_err_max is not None:
      value = agg.get("norm_abs_error_mean")
      if value is None:
        case_failures.append("missing metric norm_abs_error_mean")
      elif float(value) > float(norm_err_max):
        case_failures.append(f"norm_abs_error_mean={value:.6f} > norm_abs_error_mean_max={float(norm_err_max):.6f}")

    core_elapsed_max = gates.get("core_elapsed_ms_mean_max")
    if core_elapsed_max is not None:
      value = agg.get("core_elapsed_ms_mean")
      if value is None:
        case_failures.append("missing metric core_elapsed_ms_mean")
      elif float(value) > float(core_elapsed_max):
        case_failures.append(f"core_elapsed_ms_mean={value:.6f} > core_elapsed_ms_mean_max={float(core_elapsed_max):.6f}")

    if case_failures:
      failures.append({"case": case["id"], "reasons": case_failures})

  return {"status": "FAIL" if failures else "PASS", "failures": failures}


def main() -> None:
  parser = argparse.ArgumentParser(description="Run QuantX problem benchmark suite.")
  parser.add_argument(
      "--suite",
      type=str,
      default="benchmarks/suites/continuous2d_smoke.json",
      help="Path to suite JSON, relative to repo root.",
  )
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(root / "python"))
  from quantx_engine import Engine

  suite_path = root / args.suite
  suite = load_json(suite_path)

  engine = Engine(str(root / "build-vs" / "Debug" / "quantx.dll"))

  repeats = int(suite.get("repeats", 1))
  options = dict(suite.get("options", {}))
  gates = dict(suite.get("gates", {}))

  report_cases = []
  for case in suite.get("cases", []):
    report_cases.append(run_case(engine, root, case, options, repeats))

  gate = evaluate_gates(report_cases, gates)
  case_versions = {c.get("engine_version") for c in report_cases if c.get("engine_version")}
  report_engine_version = sorted(case_versions)[0] if case_versions else None

  report = {
      "suite": suite.get("name", "unnamed_suite"),
      "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
      "repeats": repeats,
      "engine_version": report_engine_version,
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


if __name__ == "__main__":
  main()
