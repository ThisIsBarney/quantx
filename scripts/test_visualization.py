#!/usr/bin/env python3
"""Smoke-test visualization generation."""

from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path


def _write_continuous_export_fixture(path: Path) -> None:
  """Write a tiny engine-export-shaped continuous 2D payload for smoke testing."""
  payload = {
      "continuous_export": {
          "kind": "continuous_space_2d",
          "source": "engine-export",
          "solver_mode": "engine",
          "x_min": -1.0,
          "x_max": 1.0,
          "y_min": -1.0,
          "y_max": 1.0,
          "dens_max": 1.0,
          "frames": [
              {
                  "t": 0.0,
                  "density": [[0.1, 0.2], [0.3, 0.4]],
                  "exp_x": 0.0,
                  "exp_y": 0.0,
              },
              {
                  "t": 0.1,
                  "density": [[0.2, 0.3], [0.4, 0.5]],
                  "exp_x": 0.1,
                  "exp_y": -0.1,
              },
          ],
      }
  }
  path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
  """Run visualization smoke checks across fallback and exported-data paths."""
  root = Path(__file__).resolve().parents[1]
  out_dir = root / "out" / "visualizations_test"
  out_dir.mkdir(parents=True, exist_ok=True)

  cmd = [
      sys.executable,
      str(root / "scripts" / "visualize_simulation.py"),
      "--scenario",
      "all",
      "--out-dir",
      str(out_dir),
      "--samples-rabi",
      "20",
      "--samples-mps",
      "20",
      "--samples-spatial-t",
      "40",
      "--samples-spatial-x",
      "120",
      "--samples-spatial2d-t",
      "24",
      "--samples-spatial2d-x",
      "40",
      "--samples-spatial2d-y",
      "40",
      "--samples-spatial3d-t",
      "16",
      "--samples-spatial3d-x",
      "20",
      "--samples-spatial3d-y",
      "20",
      "--samples-spatial3d-z",
      "20",
      "--spatial-potential",
      "barrier",
      "--spatial2d-potential",
      "barrier",
      "--spatial3d-potential",
      "barrier",
      "--spatial3d-boundary",
      "absorbing",
      "--spatial3d-export-json",
  ]
  subprocess.run(cmd, check=True)

  python_only_dir = out_dir / "python_mode"
  subprocess.run(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--scenario",
          "spatial2d",
          "--solver-mode",
          "python",
          "--out-dir",
          str(python_only_dir),
          "--samples-spatial2d-t",
          "8",
          "--samples-spatial2d-x",
          "16",
          "--samples-spatial2d-y",
          "16",
      ],
      check=True,
  )

  export_fixture = out_dir / "continuous2d_export_fixture.json"
  _write_continuous_export_fixture(export_fixture)
  engine_export_dir = out_dir / "engine_export_mode"
  subprocess.run(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--scenario",
          "spatial2d",
          "--solver-mode",
          "engine",
          "--continuous-export-json",
          str(export_fixture),
          "--out-dir",
          str(engine_export_dir),
      ],
      check=True,
  )

  files = [
      out_dir / "rabi_population.html",
      out_dir / "mps_diagnostics.html",
      out_dir / "quantum_spatial_wavepacket_barrier.html",
      out_dir / "quantum_spatial2d_wavepacket_barrier.html",
      out_dir / "quantum_spatial3d_wavepacket_barrier_absorbing.html",
      out_dir / "quantum_spatial3d_wavepacket_barrier_absorbing.json",
      python_only_dir / "quantum_spatial2d_wavepacket_barrier.html",
      engine_export_dir / "quantum_spatial2d_wavepacket_barrier.html",
  ]
  for path in files:
    if not path.exists():
      raise RuntimeError(f"missing visualization file: {path}")
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
      if '"frames"' not in content:
        raise RuntimeError(f"spatial3d json missing frames payload: {path}")
      data = json.loads(content)
      frames = data.get("frames", [])
      if not frames or not isinstance(frames[0].get("points"), list):
        raise RuntimeError(f"spatial3d json missing volumetric point payload: {path}")
      continue
    if ("createSvgEl" not in content) and ("Spatial Wavepacket" not in content):
      raise RuntimeError(f"visualization file missing svg renderer script: {path}")
    if "Play" not in content or ("range" not in content):
      raise RuntimeError(f"visualization file missing animation controls: {path}")
    if path == engine_export_dir / "quantum_spatial2d_wavepacket_barrier.html" and "engine-export" not in content:
      raise RuntimeError(f"engine-export visualization should mention its data source: {path}")

  print("visualization smoke passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
