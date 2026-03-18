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
  ]
  subprocess.run(cmd, check=True)

  files = [
      out_dir / "rabi_population.html",
      out_dir / "mps_diagnostics.html",
      out_dir / "quantum_spatial_wavepacket.html",
  ]
  for path in files:
    if not path.exists():
      raise RuntimeError(f"missing visualization file: {path}")
    content = path.read_text(encoding="utf-8")
    if ("createSvgEl" not in content) and ("Spatial Wavepacket" not in content):
      raise RuntimeError(f"visualization file missing svg renderer script: {path}")
    if "Play" not in content or ("range" not in content):
      raise RuntimeError(f"visualization file missing animation controls: {path}")

  print("visualization smoke passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
