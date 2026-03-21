#!/usr/bin/env python3
"""Smoke-test visualization generation."""

from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path


def _write_continuous_export_fixture(path: Path, schema_version: int | None = 1) -> None:
  """Write a tiny engine-export-shaped continuous 2D payload for smoke testing."""
  payload = {
      "continuous_export": {
          **({"schema_version": schema_version} if schema_version is not None else {}),
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


def _write_continuous_export_fixture_3d(path: Path, schema_version: int | None = 1) -> None:
  """Write a tiny engine-export-shaped continuous 3D payload for smoke testing."""
  payload = {
      "continuous_export": {
          **({"schema_version": schema_version} if schema_version is not None else {}),
          "kind": "continuous_space_3d",
          "source": "engine-export",
          "solver_mode": "engine",
          "boundary": "periodic",
          "potential_type": "barrier",
          "x_min": -1.0,
          "x_max": 1.0,
          "y_min": -1.0,
          "y_max": 1.0,
          "z_min": -1.0,
          "z_max": 1.0,
          "dens_max": 0.4,
          "frames": [
              {
                  "t": 0.0,
                  "nx": 2,
                  "ny": 2,
                  "nz": 2,
                  "norm": 1.0,
                  "exp_x": 0.0,
                  "exp_y": 0.1,
                  "exp_z": -0.1,
                  "density_xy_flat": [0.1, 0.2, 0.3, 0.4],
                  "density_xz_flat": [0.1, 0.2, 0.3, 0.4],
                  "density_yz_flat": [0.1, 0.2, 0.3, 0.4],
                  "points_xyzd_flat": [0.0, 0.0, 0.0, 0.1, 0.5, 0.0, 0.0, 0.2],
              },
              {
                  "t": 0.1,
                  "nx": 2,
                  "ny": 2,
                  "nz": 2,
                  "norm": 0.95,
                  "exp_x": 0.1,
                  "exp_y": 0.0,
                  "exp_z": 0.2,
                  "density_xy_flat": [0.2, 0.3, 0.2, 0.1],
                  "density_xz_flat": [0.1, 0.1, 0.2, 0.2],
                  "density_yz_flat": [0.05, 0.1, 0.15, 0.2],
                  "points_xyzd_flat": [0.0, 0.0, 0.0, 0.2, -0.5, 0.5, 0.5, 0.15],
              },
          ],
      }
  }
  path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_corrupt_continuous_export_fixture_3d(path: Path, schema_version: int | None = 1) -> None:
  """Write a malformed engine-export-shaped 3D payload for negative smoke coverage."""
  payload = {
      "continuous_export": {
          **({"schema_version": schema_version} if schema_version is not None else {}),
          "kind": "continuous_space_3d",
          "source": "engine-export",
          "solver_mode": "engine",
          "frames": [
              {
                  "t": 0.0,
                  "nx": 2,
                  "ny": 2,
                  "nz": 2,
                  "density_xy_flat": [0.1, 0.2, 0.3, 0.4],
                  "density_xz_flat": [0.1, 0.2, 0.3, 0.4],
                  "density_yz_flat": [0.1, 0.2, 0.3, 0.4],
              }
          ],
      }
  }
  path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_expected_failure(cmd: list[str], needle: str) -> None:
  """Run a command that should fail and assert on the console diagnostics."""
  proc = subprocess.run(cmd, capture_output=True, text=True)
  if proc.returncode == 0:
    raise RuntimeError(f"expected command to fail, but it succeeded: {' '.join(cmd)}")
  stderr = (proc.stderr or "") + (proc.stdout or "")
  if needle not in stderr:
    raise RuntimeError(f"missing expected diagnostic '{needle}' in output:\n{stderr}")


def _run_expected_success_with_warning(cmd: list[str], needle: str) -> str:
  """Run a command that should succeed and assert on the warning diagnostics."""
  proc = subprocess.run(cmd, capture_output=True, text=True)
  if proc.returncode != 0:
    raise RuntimeError(f"expected command to succeed, but it failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
  stderr = (proc.stderr or "") + (proc.stdout or "")
  if needle not in stderr:
    raise RuntimeError(f"missing expected warning '{needle}' in output:\n{stderr}")
  return stderr


def _ensure_native_build(root: Path) -> None:
  """Confirm a native engine DLL is available in the current worktree or an ancestor."""
  for candidate_root in [root, *root.parents]:
    dll = candidate_root / "build-vs" / "Debug" / "quantx.dll"
    if dll.exists():
      return
  raise RuntimeError("native engine DLL not found in this worktree or any ancestor build-vs directory")


def main() -> int:
  """Run visualization smoke checks across fallback and exported-data paths."""
  root = Path(__file__).resolve().parents[1]
  out_dir = root / "out" / "visualizations_test"
  out_dir.mkdir(parents=True, exist_ok=True)

  python_only_dir = out_dir / "python_mode"
  subprocess.run(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--scenario",
          "spatial",
          "--solver-mode",
          "python",
          "--out-dir",
          str(python_only_dir),
          "--samples-spatial-t",
          "8",
          "--samples-spatial-x",
          "16",
          "--spatial-potential",
          "barrier",
      ],
      check=True,
  )

  python_only_2d_dir = out_dir / "python_mode_2d"
  subprocess.run(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--scenario",
          "spatial2d",
          "--solver-mode",
          "python",
          "--out-dir",
          str(python_only_2d_dir),
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
          "--strict-export-schema",
          "--out-dir",
          str(engine_export_dir),
      ],
      check=True,
  )

  future_export_fixture = out_dir / "continuous2d_export_future_fixture.json"
  _write_continuous_export_fixture(future_export_fixture, schema_version=2)
  engine_export_future_dir = out_dir / "engine_export_future_mode"
  _run_expected_success_with_warning(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--scenario",
          "spatial2d",
          "--solver-mode",
          "engine",
          "--continuous-export-json",
          str(future_export_fixture),
          "--out-dir",
          str(engine_export_future_dir),
      ],
      "schema_version=2 is newer than supported version 1",
  )
  _run_expected_failure(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--scenario",
          "spatial2d",
          "--solver-mode",
          "engine",
          "--continuous-export-json",
          str(future_export_fixture),
          "--strict-export-schema",
          "--out-dir",
          str(out_dir / "engine_export_future_strict_fail"),
      ],
      "schema_version=2 is newer than supported version 1",
  )

  invalid_export_fixture = out_dir / "continuous2d_export_invalid_fixture.json"
  _write_continuous_export_fixture(invalid_export_fixture, schema_version=None)
  python_fallback_dir = out_dir / "python_fallback_mode"
  _run_expected_success_with_warning(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--scenario",
          "spatial2d",
          "--solver-mode",
          "python",
          "--continuous-export-json",
          str(invalid_export_fixture),
          "--out-dir",
          str(python_fallback_dir),
      ],
      "continuous_export.schema_version is missing or below 1",
  )

  _run_expected_failure(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--scenario",
          "spatial2d",
          "--solver-mode",
          "python",
          "--continuous-export-json",
          str(invalid_export_fixture),
          "--strict-export-schema",
          "--out-dir",
          str(out_dir / "python_fallback_strict_fail"),
      ],
      "continuous_export.schema_version is missing or below 1",
  )

  wrapper_export_dir = root / "out" / "visualizations"
  wrapper_engine_export = wrapper_export_dir / "quantum_spatial2d_wavepacket_barrier.html"
  if wrapper_engine_export.exists():
    wrapper_engine_export.unlink()
  subprocess.run(
      [
          "powershell",
          "-File",
          str(root / "scripts" / "run_visualization.ps1"),
          "-Scenario",
          "spatial2d",
          "-SolverMode",
          "engine",
          "-ContinuousExportJson",
          str(export_fixture),
          "-StrictExportSchema",
      ],
      check=True,
  )

  export_fixture_3d = out_dir / "continuous3d_export_fixture.json"
  _write_continuous_export_fixture_3d(export_fixture_3d)
  engine_export_3d_dir = out_dir / "engine_export_mode_3d"
  subprocess.run(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--scenario",
          "spatial3d",
          "--solver-mode",
          "engine",
          "--continuous-export-json",
          str(export_fixture_3d),
          "--strict-export-schema",
          "--spatial3d-export-json",
          "--out-dir",
          str(engine_export_3d_dir),
      ],
      check=True,
  )

  corrupt_export_fixture_3d = out_dir / "continuous3d_export_corrupt_fixture.json"
  _write_corrupt_continuous_export_fixture_3d(corrupt_export_fixture_3d)
  _run_expected_failure(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--scenario",
          "spatial3d",
          "--solver-mode",
          "engine",
          "--continuous-export-json",
          str(corrupt_export_fixture_3d),
          "--strict-export-schema",
          "--out-dir",
          str(out_dir / "engine_export_bad"),
      ],
      "points_xyzd_flat",
  )

  auto_fallback_dir = out_dir / "engine_export_auto_fallback"
  _run_expected_success_with_warning(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--scenario",
          "spatial3d",
          "--solver-mode",
          "auto",
          "--continuous-export-json",
          str(corrupt_export_fixture_3d),
          "--out-dir",
          str(auto_fallback_dir),
      ],
      "falling back to live engine execution or Python reference solver",
  )

  js_export_dir = out_dir / "js_export_bundle"
  subprocess.run(
      [
          "powershell",
          "-Command",
          f'.\\scripts\\run_spatial3d_js_export.ps1 -SolverMode python -OutDir "{js_export_dir}"',
      ],
      check=True,
  )

  files = [
      python_only_dir / "quantum_spatial_wavepacket_barrier.html",
      python_only_2d_dir / "quantum_spatial2d_wavepacket_barrier.html",
      engine_export_future_dir / "quantum_spatial2d_wavepacket_barrier.html",
      python_fallback_dir / "quantum_spatial2d_wavepacket_barrier.html",
      wrapper_engine_export,
      engine_export_dir / "quantum_spatial2d_wavepacket_barrier.html",
      engine_export_3d_dir / "quantum_spatial3d_wavepacket_barrier_periodic.html",
      engine_export_3d_dir / "quantum_spatial3d_wavepacket_barrier_periodic.json",
      auto_fallback_dir / "quantum_spatial3d_wavepacket_barrier_periodic.html",
      js_export_dir / "spatial3d_player.html",
      js_export_dir / "quantum_spatial3d_wavepacket_barrier_periodic.json",
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
      if path == engine_export_3d_dir / "quantum_spatial3d_wavepacket_barrier_periodic.json":
        frame0 = frames[0]
        if not isinstance(frame0.get("xy"), list) or not isinstance(frame0.get("xz"), list) or not isinstance(frame0.get("yz"), list):
          raise RuntimeError(f"engine-export 3d json missing normalized slice payloads: {path}")
        if not isinstance(frame0.get("points"), list) or not frame0["points"]:
          raise RuntimeError(f"engine-export 3d json missing normalized point payloads: {path}")
        if "point_vmax" not in frame0 or "point_count" not in frame0:
          raise RuntimeError(f"engine-export 3d json missing point diagnostics: {path}")
      continue
    if path.name == "spatial3d_player.html":
      if "QuantX Spatial3D World Viewer" not in content:
        raise RuntimeError(f"3d player bundle missing title marker: {path}")
      if "THREE.WebGLRenderer" not in content or "THREE.OrbitControls" not in content:
        raise RuntimeError(f"3d player bundle missing WebGL controls: {path}")
      if "./vendor/three.min.js" not in content or "./vendor/OrbitControls.js" not in content:
        raise RuntimeError(f"3d player bundle missing local vendor references: {path}")
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
