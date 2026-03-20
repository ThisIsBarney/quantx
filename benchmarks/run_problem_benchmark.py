import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks.report_utils import (
    continuous_export_volume,
    detect_platform_profile,
    extract_continuous_frames,
    get_continuous_config,
    infer_fft_provider,
    suite_policy,
)


def expected_sigma(case: dict, problem: dict) -> float | None:
  """Compute the analytic sigma for free Gaussian smoke cases when available."""
  analytic = case.get("analytic", {})
  if analytic.get("model") != "free_gaussian_sigma":
    return None

  c2d = problem.get("continuous_space_2d", {})
  c3d = problem.get("continuous_space_3d", {})
  cfg = c2d if c2d else c3d
  sigma0 = float(analytic.get("sigma0", 0.0))
  steps = int(cfg.get("steps", 0))
  dt = float(cfg.get("dt", 0.0))
  if sigma0 <= 0.0 or steps <= 0 or dt <= 0.0:
    return None

  t = float(steps) * dt
  return (sigma0 * sigma0 + (t * t) / (4.0 * sigma0 * sigma0)) ** 0.5


def load_json(path: Path) -> dict:
  """Load a JSON file from disk."""
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def summarize_continuous_performance(problem: dict, result: dict, aggregate: dict) -> dict:
  """Derive explainable performance metadata for a continuous-space case."""
  cfg = get_continuous_config(problem)
  if cfg is None:
    return {}

  frames = extract_continuous_frames(result)
  frame_count = len(frames)
  export_volume = continuous_export_volume(frames)
  steps = int(cfg.get("steps", 0) or 0)
  avg_step_ms = None
  core_elapsed = aggregate.get("core_elapsed_ms_mean")
  elapsed = aggregate.get("elapsed_ms_mean")
  if steps > 0:
    if core_elapsed is not None:
      avg_step_ms = float(core_elapsed) / float(steps)
    elif elapsed is not None:
      avg_step_ms = float(elapsed) / float(steps)

  grid = cfg.get("grid", [])
  grid_label = "x".join(str(int(v)) for v in grid) if grid else "n/a"
  return {
      "kind": cfg.get("kind", "n/a"),
      "grid": grid_label,
      "grid_points": cfg.get("grid_points"),
      "steps": steps,
      "dt": cfg.get("dt"),
      "export_frame_count": frame_count,
      "export_volume": export_volume,
      "fft_provider": infer_fft_provider(result),
      "avg_step_ms_mean": avg_step_ms,
  }


def run_case(engine: "Engine", root: Path, case: dict, options: dict, repeats: int) -> dict:
  """Execute one benchmark case repeatedly and aggregate the results."""
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
  sigma_z = []
  sigma_x_abs_error = []
  sigma_z_abs_error = []

  sigma_expected = expected_sigma(case, problem)

  for run in runs:
    result = run["result"]
    if "elapsed_ms" in result:
      core_elapsed.append(float(result["elapsed_ms"]))
    obs = result.get("observables", {})
    if "norm" in obs:
      norm_abs_error.append(abs(float(obs["norm"]) - 1.0))
    if "sigma_x" in obs:
      sx = float(obs["sigma_x"])
      sigma_x.append(sx)
      if sigma_expected is not None:
        sigma_x_abs_error.append(abs(sx - sigma_expected))
    if "sigma_y" in obs:
      sigma_y.append(float(obs["sigma_y"]))
    if "sigma_z" in obs:
      sz = float(obs["sigma_z"])
      sigma_z.append(sz)
      if sigma_expected is not None:
        sigma_z_abs_error.append(abs(sz - sigma_expected))

  aggregate = {
      "elapsed_ms_mean": statistics.fmean(elapsed) if elapsed else 0.0,
      "elapsed_ms_min": min(elapsed) if elapsed else 0.0,
      "elapsed_ms_max": max(elapsed) if elapsed else 0.0,
      "core_elapsed_ms_mean": statistics.fmean(core_elapsed) if core_elapsed else None,
      "norm_abs_error_mean": statistics.fmean(norm_abs_error) if norm_abs_error else None,
      "sigma_x_mean": statistics.fmean(sigma_x) if sigma_x else None,
      "sigma_y_mean": statistics.fmean(sigma_y) if sigma_y else None,
      "sigma_z_mean": statistics.fmean(sigma_z) if sigma_z else None,
      "sigma_x_abs_error_mean": statistics.fmean(sigma_x_abs_error) if sigma_x_abs_error else None,
      "sigma_z_abs_error_mean": statistics.fmean(sigma_z_abs_error) if sigma_z_abs_error else None,
      "sigma_x_expected": sigma_expected,
  }

  engine_version = None
  backend = None
  result0 = None
  ok_rate = 0.0
  if runs:
    result0 = runs[0]["result"]
    engine_version = result0.get("engine_version")
    backend = result0.get("backend")
    ok_count = sum(1 for r in runs if bool(r["result"].get("ok", False)))
    ok_rate = float(ok_count) / float(len(runs))

  performance = summarize_continuous_performance(problem, result0 or {}, aggregate) if result0 else {}

  return {
      "id": case["id"],
      "problem_file": case["problem_file"],
      "engine_version": engine_version,
      "backend": backend,
      "ok_rate": ok_rate,
      "aggregate": aggregate,
      "performance": performance,
      "runs": runs,
  }


def write_markdown_summary(report: dict, output_path: Path) -> None:
  """Write a compact human-readable markdown summary for one benchmark run."""
  lines = []
  lines.append(f"# Problem Benchmark Summary: {report['suite']}")
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
  lines.append(
      "| case | backend | ok_rate | grid | export_volume | fft_provider | avg_step_ms_mean | "
      "elapsed_ms_mean | core_elapsed_ms_mean | norm_abs_error_mean | sigma_x_mean | sigma_y_mean | sigma_z_mean | "
      "sigma_x_expected | sigma_x_abs_error_mean | sigma_z_abs_error_mean |"
  )
  lines.append("|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

  for case in report["cases"]:
    agg = case["aggregate"]
    perf = case.get("performance", {})

    def fmt(v, digits=6):
      return "n/a" if v is None else f"{v:.{digits}f}"

    lines.append(
        f"| {case['id']} | {case.get('backend','n/a')} | {case.get('ok_rate',0.0):.2f} | {perf.get('grid','n/a')} | "
        f"{perf.get('export_volume', 'n/a')} | {perf.get('fft_provider', 'n/a')} | {fmt(perf.get('avg_step_ms_mean'), 3)} | "
        f"{fmt(agg.get('elapsed_ms_mean'), 3)} | {fmt(agg.get('core_elapsed_ms_mean'), 3)} | "
        f"{fmt(agg.get('norm_abs_error_mean'), 6)} | {fmt(agg.get('sigma_x_mean'), 6)} | {fmt(agg.get('sigma_y_mean'), 6)} | "
        f"{fmt(agg.get('sigma_z_mean'), 6)} | {fmt(agg.get('sigma_x_expected'), 6)} | "
        f"{fmt(agg.get('sigma_x_abs_error_mean'), 6)} | {fmt(agg.get('sigma_z_abs_error_mean'), 6)} |"
    )

  output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_gates(report_cases: list[dict], gates: dict, perf_warnings_blocking: bool) -> dict:
  """Evaluate suite gates and split failures from performance-only warnings."""
  if not gates:
    return {"status": "SKIPPED", "failures": [], "warnings": [], "perf_warnings_blocking": perf_warnings_blocking}

  failures = []
  warnings = []
  for case in report_cases:
    agg = case.get("aggregate", {})
    case_failures = []
    case_warnings = []

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

    sigma_err_max = gates.get("sigma_x_abs_error_mean_max")
    if sigma_err_max is not None:
      value = agg.get("sigma_x_abs_error_mean")
      if value is None:
        case_failures.append("missing metric sigma_x_abs_error_mean")
      elif float(value) > float(sigma_err_max):
        case_failures.append(
            f"sigma_x_abs_error_mean={value:.6f} > sigma_x_abs_error_mean_max={float(sigma_err_max):.6f}"
        )

    sigma_z_err_max = gates.get("sigma_z_abs_error_mean_max")
    if sigma_z_err_max is not None:
      value = agg.get("sigma_z_abs_error_mean")
      if value is None:
        case_failures.append("missing metric sigma_z_abs_error_mean")
      elif float(value) > float(sigma_z_err_max):
        case_failures.append(
            f"sigma_z_abs_error_mean={value:.6f} > sigma_z_abs_error_mean_max={float(sigma_z_err_max):.6f}"
        )

    core_elapsed_max = gates.get("core_elapsed_ms_mean_max")
    if core_elapsed_max is not None:
      value = agg.get("core_elapsed_ms_mean")
      message = None
      if value is None:
        message = "missing metric core_elapsed_ms_mean"
      elif float(value) > float(core_elapsed_max):
        message = f"core_elapsed_ms_mean={value:.6f} > core_elapsed_ms_mean_max={float(core_elapsed_max):.6f}"
      if message is not None:
        if perf_warnings_blocking:
          case_failures.append(message)
        else:
          case_warnings.append(message)

    if case_failures:
      failures.append({"case": case["id"], "reasons": case_failures})
    if case_warnings:
      warnings.append({"case": case["id"], "reasons": case_warnings})

  return {
      "status": "FAIL" if failures else "PASS",
      "failures": failures,
      "warnings": warnings,
      "perf_warnings_blocking": perf_warnings_blocking,
  }


def main() -> None:
  """Run a benchmark suite and write JSON/Markdown reports."""
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
