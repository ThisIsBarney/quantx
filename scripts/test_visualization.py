#!/usr/bin/env python3
"""Smoke-test visualization generation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
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

  files = [
      out_dir / "rabi_population.html",
      out_dir / "mps_diagnostics.html",
      out_dir / "quantum_spatial_wavepacket_barrier.html",
      out_dir / "quantum_spatial2d_wavepacket_barrier.html",
      out_dir / "quantum_spatial3d_wavepacket_barrier_absorbing.html",
      out_dir / "quantum_spatial3d_wavepacket_barrier_absorbing.json",
  ]
  for path in files:
    if not path.exists():
      raise RuntimeError(f"missing visualization file: {path}")
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
      if '"frames"' not in content:
        raise RuntimeError(f"spatial3d json missing frames payload: {path}")
      continue
    if ("createSvgEl" not in content) and ("Spatial Wavepacket" not in content):
      raise RuntimeError(f"visualization file missing svg renderer script: {path}")
    if "Play" not in content or ("range" not in content):
      raise RuntimeError(f"visualization file missing animation controls: {path}")

  print("visualization smoke passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
