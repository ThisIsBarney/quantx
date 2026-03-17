#!/usr/bin/env python3
"""Generate lightweight HTML visualizations for QuantX simulations."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _svg_line_chart(title: str, x_label: str, y_label: str, series: list[tuple[str, list[tuple[float, float]], str]]) -> str:
  width = 900
  height = 420
  left = 70
  right = 20
  top = 50
  bottom = 60
  plot_w = width - left - right
  plot_h = height - top - bottom

  xs = [x for _, pts, _ in series for x, _ in pts]
  ys = [y for _, pts, _ in series for _, y in pts]
  x_min = min(xs) if xs else 0.0
  x_max = max(xs) if xs else 1.0
  y_min = min(ys) if ys else 0.0
  y_max = max(ys) if ys else 1.0
  if abs(x_max - x_min) < 1e-12:
    x_max = x_min + 1.0
  if abs(y_max - y_min) < 1e-12:
    y_max = y_min + 1.0

  def sx(x: float) -> float:
    return left + (x - x_min) / (x_max - x_min) * plot_w

  def sy(y: float) -> float:
    return top + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

  lines = [
      f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
      '<style>text{font-family:Segoe UI,Arial,sans-serif;font-size:12px;} .grid{stroke:#ddd;stroke-width:1} .axis{stroke:#333;stroke-width:1.5}</style>',
      f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="18">{title}</text>',
      f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>',
      f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>',
  ]

  for i in range(6):
    gx = left + plot_w * i / 5.0
    gy = top + plot_h * i / 5.0
    xv = x_min + (x_max - x_min) * i / 5.0
    yv = y_max - (y_max - y_min) * i / 5.0
    lines.append(f'<line class="grid" x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top+plot_h}"/>')
    lines.append(f'<line class="grid" x1="{left}" y1="{gy:.1f}" x2="{left+plot_w}" y2="{gy:.1f}"/>')
    lines.append(f'<text x="{gx:.1f}" y="{top+plot_h+18}" text-anchor="middle">{xv:.3g}</text>')
    lines.append(f'<text x="{left-8}" y="{gy+4:.1f}" text-anchor="end">{yv:.3g}</text>')

  lines.append(f'<text x="{left + plot_w/2:.1f}" y="{height-18}" text-anchor="middle">{x_label}</text>')
  lines.append(f'<text x="18" y="{top + plot_h/2:.1f}" transform="rotate(-90 18 {top + plot_h/2:.1f})" text-anchor="middle">{y_label}</text>')

  legend_x = left + 10
  legend_y = top + 10
  for idx, (name, pts, color) in enumerate(series):
    poly = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in pts)
    lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{poly}"/>')
    ly = legend_y + idx * 18
    lines.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+20}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
    lines.append(f'<text x="{legend_x+26}" y="{ly+4}">{name}</text>')

  lines.append("</svg>")
  return "\n".join(lines)


def _html_page(title: str, body_blocks: list[str]) -> str:
  blocks = "\n<hr/>\n".join(body_blocks)
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
</head>
<body>
  <h1>{title}</h1>
  {blocks}
</body>
</html>
"""


def _make_engine(root: Path):
  sys.path.insert(0, str(root / "python"))
  from quantx_engine import Engine

  return Engine(str(root / "build-vs" / "Debug" / "quantx.dll"))


def _rabi_data(engine, samples: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
  pts_sim = []
  pts_ref = []
  for i in range(samples):
    t = math.pi * i / max(1, samples - 1)
    dt = 0.01 if t > 0 else 0.01
    problem = {
        "qubits": 1,
        "equation": "Schrodinger",
        "total_time": t,
        "dt": dt,
        "method": "krylov",
        "krylov_dim": 8,
        "hamiltonian_terms": [{"label": "X0", "coefficient": 1.0}],
    }
    result = engine.run(problem, {"enable_checkpoint": False})
    sim = float(result.get("observables", {}).get("population_1", 0.0))
    ref = math.sin(t) ** 2
    pts_sim.append((t, sim))
    pts_ref.append((t, ref))
  return pts_sim, pts_ref


def _mps_data(engine, samples: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
  bond = []
  trunc = []
  for i in range(samples):
    t = 2.0 * i / max(1, samples - 1)
    problem = {
        "qubits": 64,
        "equation": "Schrodinger",
        "total_time": t,
        "dt": 0.05,
        "method": "tebd",
        "mps_max_bond": 32,
        "mps_trunc_tol": 1e-6,
        "expected_circuit_depth": 200,
        "hamiltonian_terms": [{"label": "X0X1", "coefficient": 1.0}, {"label": "Z0", "coefficient": 0.2}],
    }
    result = engine.run(problem, {"enable_checkpoint": False})
    obs = result.get("observables", {})
    bond.append((t, float(obs.get("bond_dim", 0.0))))
    trunc.append((t, float(obs.get("truncation_error", 0.0))))
  return bond, trunc


def main() -> int:
  parser = argparse.ArgumentParser(description="Generate QuantX visualization HTML.")
  parser.add_argument("--scenario", choices=["rabi", "mps", "all"], default="all")
  parser.add_argument("--out-dir", default="out/visualizations")
  parser.add_argument("--samples-rabi", type=int, default=80)
  parser.add_argument("--samples-mps", type=int, default=60)
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[1]
  out_dir = root / args.out_dir
  out_dir.mkdir(parents=True, exist_ok=True)
  engine = _make_engine(root)

  generated = []

  if args.scenario in {"rabi", "all"}:
    sim, ref = _rabi_data(engine, args.samples_rabi)
    chart = _svg_line_chart(
        "Rabi Oscillation: Population(|1>) vs Time",
        "time",
        "population_1",
        [("simulation", sim, "#0b84f3"), ("analytic sin^2(t)", ref, "#f39c12")],
    )
    html = _html_page("QuantX Visualization - Rabi", [chart])
    path = out_dir / "rabi_population.html"
    path.write_text(html, encoding="utf-8")
    generated.append(path)

  if args.scenario in {"mps", "all"}:
    bond, trunc = _mps_data(engine, args.samples_mps)
    chart1 = _svg_line_chart("MPS Bond Dimension vs Time", "time", "bond_dim", [("bond_dim", bond, "#2e8b57")])
    chart2 = _svg_line_chart(
        "MPS Truncation Error vs Time", "time", "truncation_error", [("truncation_error", trunc, "#c0392b")]
    )
    html = _html_page("QuantX Visualization - MPS", [chart1, chart2])
    path = out_dir / "mps_diagnostics.html"
    path.write_text(html, encoding="utf-8")
    generated.append(path)

  for p in generated:
    print(f"wrote visualization: {p}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
