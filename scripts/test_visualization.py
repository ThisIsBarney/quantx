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


def _write_continuous_export_fixture_2d_with_phase(path: Path, schema_version: int | None = 1) -> None:
  """Write a 2D engine-export payload with phase, dirac_s0, and dirac_s1 fields for M14 smoke testing."""
  import math
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
          "has_phase": True,
          "has_dirac": True,
          "frames": [
              {
                  "t": 0.0,
                  "density": [[0.1, 0.2], [0.3, 0.4]],
                  "phase_xy_flat": [0.0, math.pi / 4, math.pi / 2, math.pi],
                  "dirac_s0_xy_flat": [0.1, 0.2, 0.3, 0.4],
                  "dirac_s1_xy_flat": [0.05, 0.1, 0.15, 0.2],
                  "exp_x": 0.0,
                  "exp_y": 0.0,
              },
              {
                  "t": 0.1,
                  "density": [[0.2, 0.3], [0.4, 0.5]],
                  "phase_xy_flat": [0.1, math.pi / 3, math.pi, -math.pi / 2],
                  "dirac_s0_xy_flat": [0.2, 0.3, 0.4, 0.5],
                  "dirac_s1_xy_flat": [0.1, 0.12, 0.18, 0.22],
                  "exp_x": 0.1,
                  "exp_y": -0.1,
              },
          ],
      }
  }
  path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_continuous_export_fixture_3d_with_phase_and_volume(path: Path, schema_version: int | None = 1) -> None:
  """Write a 3D engine-export payload with phase slices and full volume grid for M14 smoke testing."""
  import math
  # 2×2×2 grid → 4 elements per slice, 8 elements for full volume
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
          "has_phase": True,
          "has_volume_grid": True,
          "nx": 2,
          "ny": 2,
          "nz": 2,
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
                  "phase_xy_flat": [0.0, math.pi / 4, math.pi / 2, math.pi],
                  "phase_xz_flat": [0.1, 0.3, 0.6, 1.0],
                  "phase_yz_flat": [-0.5, 0.0, 0.5, 1.2],
                  "density_xyz_flat": [0.1, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4, 0.38],
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
                  "phase_xy_flat": [0.2, math.pi / 3, -0.8, 2.0],
                  "phase_xz_flat": [0.0, 0.4, 0.9, -1.1],
                  "phase_yz_flat": [-1.0, -0.5, 0.2, 0.8],
                  "density_xyz_flat": [0.2, 0.25, 0.18, 0.12, 0.3, 0.28, 0.22, 0.15],
                  "points_xyzd_flat": [0.0, 0.0, 0.0, 0.2, -0.5, 0.5, 0.5, 0.15],
              },
          ],
      }
  }
  path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_expected_failure(cmd: list[str], needle: str) -> None:
  """Run a command that should fail and assert on the console diagnostics."""
  proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
  if proc.returncode == 0:
    raise RuntimeError(f"expected command to fail, but it succeeded: {' '.join(cmd)}")
  stderr = (proc.stderr or "") + (proc.stdout or "")
  if needle not in stderr:
    raise RuntimeError(f"missing expected diagnostic '{needle}' in output:\n{stderr}")


def _run_expected_success_with_warning(cmd: list[str], needle: str) -> str:
  """Run a command that should succeed and assert on the warning diagnostics."""
  proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
  if proc.returncode != 0:
    raise RuntimeError(f"expected command to succeed, but it failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
  stderr = (proc.stderr or "") + (proc.stdout or "")
  if needle not in stderr:
    raise RuntimeError(f"missing expected warning '{needle}' in output:\n{stderr}")
  return stderr


def _run_expected_success_with_output(cmd: list[str], *needles: str) -> str:
  """Run a command that should succeed and assert that all snippets appear in output."""
  proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
  if proc.returncode != 0:
    raise RuntimeError(f"expected command to succeed, but it failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
  output = (proc.stdout or "") + (proc.stderr or "")
  for needle in needles:
    if needle not in output:
      raise RuntimeError(f"missing expected text '{needle}' in output:\n{output}")
  return output


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

  _run_expected_success_with_output(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--help",
      ],
      "Examples:",
      "--continuous-export-json",
      "--strict-export-schema",
  )

  _run_expected_success_with_output(
      [
          "powershell",
          "-Command",
          f'$OutputEncoding=[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); Get-Help "{root / "scripts" / "run_visualization.ps1"}" -Full | Out-String -Width 200',
      ],
      "-Scenario",
      "-StrictExportSchema",
      "EXAMPLE 1",
  )

  missing_export_path = out_dir / "missing-export.json"
  if missing_export_path.exists():
    missing_export_path.unlink()
  _run_expected_failure(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--scenario",
          "spatial2d",
          "--solver-mode",
          "python",
          "--continuous-export-json",
          str(missing_export_path),
          "--out-dir",
          str(out_dir / "missing_export_fail"),
      ],
      "continuous export json file not found",
  )
  _run_expected_failure(
      [
          "powershell",
          "-File",
          str(root / "scripts" / "run_visualization.ps1"),
          "-Scenario",
          "spatial2d",
          "-SolverMode",
          "python",
          "-ContinuousExportJson",
          str(missing_export_path),
      ],
      "continuous export json file not found",
  )

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

  wrapper_spatial3d_dir = root / "out" / "visualizations"
  wrapper_spatial3d_html = wrapper_spatial3d_dir / "quantum_spatial3d_wavepacket_barrier_periodic.html"
  if wrapper_spatial3d_html.exists():
    wrapper_spatial3d_html.unlink()
  subprocess.run(
      [
          "powershell",
          "-File",
          str(root / "scripts" / "run_visualization.ps1"),
          "-Scenario",
          "spatial3d",
          "-SolverMode",
          "python",
      ],
      check=True,
  )

  # --- M14: dirac2d scenario (Python reference solver) ---
  dirac2d_dir = out_dir / "python_dirac2d"
  subprocess.run(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--scenario",
          "dirac2d",
          "--solver-mode",
          "python",
          "--out-dir",
          str(dirac2d_dir),
          "--samples-spatial2d-t",
          "6",
          "--samples-spatial2d-x",
          "16",
          "--samples-spatial2d-y",
          "16",
      ],
      check=True,
  )
  dirac2d_html = dirac2d_dir / "quantum_dirac2d_barrier.html"
  if not dirac2d_html.exists():
    raise RuntimeError(f"missing Dirac spinor visualization: {dirac2d_html}")
  dirac2d_content = dirac2d_html.read_text(encoding="utf-8")
  for needle in ('"has_phase":true', '"has_dirac":true', "Play", "range"):
    # allow for spaces around colon in JSON
    if needle not in dirac2d_content and needle.replace(":", ": ") not in dirac2d_content:
      raise RuntimeError(f"dirac2d HTML missing expected content '{needle}': {dirac2d_html}")

  # --- M14: 2D engine-export with phase + dirac fields ---
  export_fixture_2d_phase = out_dir / "continuous2d_export_phase_fixture.json"
  _write_continuous_export_fixture_2d_with_phase(export_fixture_2d_phase)
  engine_export_2d_phase_dir = out_dir / "engine_export_mode_2d_phase"
  subprocess.run(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--scenario",
          "spatial2d",
          "--solver-mode",
          "engine",
          "--continuous-export-json",
          str(export_fixture_2d_phase),
          "--strict-export-schema",
          "--out-dir",
          str(engine_export_2d_phase_dir),
      ],
      check=True,
  )
  engine_export_2d_phase_html = engine_export_2d_phase_dir / "quantum_spatial2d_wavepacket_barrier.html"
  if not engine_export_2d_phase_html.exists():
    raise RuntimeError(f"missing 2D phase visualization: {engine_export_2d_phase_html}")
  phase_html_content = engine_export_2d_phase_html.read_text(encoding="utf-8")
  for needle in ('"has_phase"', '"has_dirac"'):
    if needle not in phase_html_content:
      raise RuntimeError(f"2D phase engine-export HTML missing '{needle}': {engine_export_2d_phase_html}")

  # --- M14: 3D engine-export with phase slices + volume grid ---
  export_fixture_3d_phase = out_dir / "continuous3d_export_phase_volume_fixture.json"
  _write_continuous_export_fixture_3d_with_phase_and_volume(export_fixture_3d_phase)
  engine_export_3d_phase_dir = out_dir / "engine_export_mode_3d_phase"
  subprocess.run(
      [
          sys.executable,
          str(root / "scripts" / "visualize_simulation.py"),
          "--scenario",
          "spatial3d",
          "--solver-mode",
          "engine",
          "--continuous-export-json",
          str(export_fixture_3d_phase),
          "--strict-export-schema",
          "--spatial3d-export-json",
          "--out-dir",
          str(engine_export_3d_phase_dir),
      ],
      check=True,
  )
  engine_export_3d_phase_json = engine_export_3d_phase_dir / "quantum_spatial3d_wavepacket_barrier_periodic.json"
  if not engine_export_3d_phase_json.exists():
    raise RuntimeError(f"missing 3D phase JSON export: {engine_export_3d_phase_json}")
  phase3d_data = json.loads(engine_export_3d_phase_json.read_text(encoding="utf-8"))
  phase3d_frame0 = phase3d_data.get("frames", [{}])[0]
  for key in ("phase_xy", "phase_xz", "phase_yz"):
    if not isinstance(phase3d_frame0.get(key), list):
      raise RuntimeError(f"3D phase JSON export missing normalized '{key}' slice: {engine_export_3d_phase_json}")
  if not isinstance(phase3d_frame0.get("density_xyz"), list):
    raise RuntimeError(f"3D phase JSON export missing 'density_xyz' volume grid: {engine_export_3d_phase_json}")
  if not phase3d_data.get("has_phase"):
    raise RuntimeError(f"3D phase JSON export missing 'has_phase' flag: {engine_export_3d_phase_json}")
  if not phase3d_data.get("has_volume_grid"):
    raise RuntimeError(f"3D phase JSON export missing 'has_volume_grid' flag: {engine_export_3d_phase_json}")

  # --- M14: 2D spatial page phase toggle present in Python-mode output ---
  phase_toggle_content = python_only_2d_dir / "quantum_spatial2d_wavepacket_barrier.html"
  if phase_toggle_content.exists():
    page2d_text = phase_toggle_content.read_text(encoding="utf-8")
    if "colorMapPhase" not in page2d_text:
      raise RuntimeError(f"2D spatial page missing phase colormap function: {phase_toggle_content}")
    if "colorMode" not in page2d_text:
      raise RuntimeError(f"2D spatial page missing color mode selector: {phase_toggle_content}")

  files = [
      python_only_dir / "quantum_spatial_wavepacket_barrier.html",
      python_only_2d_dir / "quantum_spatial2d_wavepacket_barrier.html",
      engine_export_future_dir / "quantum_spatial2d_wavepacket_barrier.html",
      python_fallback_dir / "quantum_spatial2d_wavepacket_barrier.html",
      wrapper_engine_export,
      wrapper_spatial3d_html,
      engine_export_dir / "quantum_spatial2d_wavepacket_barrier.html",
      engine_export_3d_dir / "quantum_spatial3d_wavepacket_barrier_periodic.html",
      engine_export_3d_dir / "quantum_spatial3d_wavepacket_barrier_periodic.json",
      auto_fallback_dir / "quantum_spatial3d_wavepacket_barrier_periodic.html",
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
    if ("createSvgEl" not in content) and ("Spatial Wavepacket" not in content):
      raise RuntimeError(f"visualization file missing svg renderer script: {path}")
    if "Play" not in content or ("range" not in content):
      raise RuntimeError(f"visualization file missing animation controls: {path}")
    if path == engine_export_dir / "quantum_spatial2d_wavepacket_barrier.html" and "engine-export" not in content:
      raise RuntimeError(f"engine-export visualization should mention its data source: {path}")

  # Convenience aliases used from M15 onward.
  py = sys.executable
  vis_script = str(root / "scripts" / "visualize_simulation.py")
  tmp = out_dir

  # ── M15: driven-potential scenarios ─────────────────────────────────────────
  driven_dir = tmp / "driven"
  driven_dir.mkdir(parents=True, exist_ok=True)

  # driven_barrier scenario
  _run_expected_success_with_output(
      [py, vis_script, "--scenario", "driven_barrier",
       "--samples-spatial2d-t", "8", "--samples-spatial2d-x", "16", "--samples-spatial2d-y", "16",
       "--out-dir", str(driven_dir)],
      "quantum_driven_barrier.html",
  )
  driven_barrier_html = driven_dir / "quantum_driven_barrier.html"
  if not driven_barrier_html.exists():
    raise RuntimeError(f"driven_barrier html not generated: {driven_barrier_html}")
  content = driven_barrier_html.read_text(encoding="utf-8")
  if "drv_chart" not in content:
    raise RuntimeError("driven_barrier html missing drv_chart canvas")
  if "drawDrivingChart" not in content:
    raise RuntimeError("driven_barrier html missing drawDrivingChart function")
  if "driving_amplitude" not in content:
    raise RuntimeError("driven_barrier html missing driving_amplitude data")

  # bloch scenario
  _run_expected_success_with_output(
      [py, vis_script, "--scenario", "bloch",
       "--samples-spatial2d-t", "8", "--samples-spatial2d-x", "16", "--samples-spatial2d-y", "16",
       "--out-dir", str(driven_dir)],
      "quantum_bloch_oscillation.html",
  )
  bloch_html = driven_dir / "quantum_bloch_oscillation.html"
  if not bloch_html.exists():
    raise RuntimeError(f"bloch html not generated: {bloch_html}")
  bloch_content = bloch_html.read_text(encoding="utf-8")
  if "dipole_x" not in bloch_content:
    raise RuntimeError("bloch html missing dipole_x spatial profile in payload")

  # gaussian_pulse scenario
  _run_expected_success_with_output(
      [py, vis_script, "--scenario", "gaussian_pulse",
       "--samples-spatial2d-t", "8", "--samples-spatial2d-x", "16", "--samples-spatial2d-y", "16",
       "--out-dir", str(driven_dir)],
      "quantum_gaussian_pulse.html",
  )
  gp_html = driven_dir / "quantum_gaussian_pulse.html"
  if not gp_html.exists():
    raise RuntimeError(f"gaussian_pulse html not generated: {gp_html}")
  gp_content = gp_html.read_text(encoding="utf-8")
  if "gaussian_pulse" not in gp_content:
    raise RuntimeError("gaussian_pulse html missing gaussian_pulse type in payload")

  # 2D engine-export with driving_amplitude field in frames
  driven_export_dir = tmp / "driven_export"
  driven_export_dir.mkdir(parents=True, exist_ok=True)
  driven_fixture = driven_export_dir / "driven_export.json"
  _write_continuous_export_fixture_2d_with_driving(driven_fixture)
  _run_expected_success_with_output(
      [py, vis_script, "--scenario", "spatial2d", "--solver-mode", "engine",
       "--continuous-export-json", str(driven_fixture), "--strict-export-schema",
       "--out-dir", str(driven_export_dir)],
      "quantum_spatial2d_wavepacket_barrier.html",
  )
  driven_exp_html = driven_export_dir / "quantum_spatial2d_wavepacket_barrier.html"
  if not driven_exp_html.exists():
    raise RuntimeError(f"driven export html not generated: {driven_exp_html}")
  driven_exp_content = driven_exp_html.read_text(encoding="utf-8")
  if "has_driving" not in driven_exp_content:
    raise RuntimeError("driven engine-export html missing has_driving flag")

  # ── M16: Lindblad open quantum systems ──────────────────────────────────────
  lindblad_dir = tmp / "lindblad"
  lindblad_dir.mkdir(parents=True, exist_ok=True)

  _run_expected_success_with_output(
      [py, vis_script, "--scenario", "lindblad",
       "--lindblad-scenario", "amplitude_damping",
       "--samples-lindblad-t", "10",
       "--out-dir", str(lindblad_dir)],
      "quantum_lindblad_amplitude_damping.html",
  )
  lindblad_html = lindblad_dir / "quantum_lindblad_amplitude_damping.html"
  if not lindblad_html.exists():
    raise RuntimeError(f"lindblad html not generated: {lindblad_html}")
  lb_content = lindblad_html.read_text(encoding="utf-8")
  if "purity" not in lb_content:
    raise RuntimeError("lindblad html missing purity data")
  if "entropy" not in lb_content:
    raise RuntimeError("lindblad html missing entropy data")
  if "cv_pop" not in lb_content:
    raise RuntimeError("lindblad html missing population canvas")

  # ── M17: Wigner quasi-probability and transport observables ─────────────────
  wigner_dir = tmp / "wigner"
  wigner_dir.mkdir(parents=True, exist_ok=True)

  for scenario_key in ("wigner_free", "wigner_barrier"):
    _run_expected_success_with_output(
        [py, vis_script, "--scenario", scenario_key,
         "--samples-wigner-t", "6",
         "--samples-spatial2d-x", "32",
         "--samples-spatial2d-y", "32",
         "--out-dir", str(wigner_dir)],
        f"quantum_wigner_{scenario_key}.html",
    )
    wigner_html = wigner_dir / f"quantum_wigner_{scenario_key}.html"
    if not wigner_html.exists():
      raise RuntimeError(f"wigner html not generated: {wigner_html}")
    wc = wigner_html.read_text(encoding="utf-8")
    if "wigner_flat" not in wc:
      raise RuntimeError(f"{scenario_key}: html missing wigner_flat data")
    if "cv_wigner" not in wc:
      raise RuntimeError(f"{scenario_key}: html missing Wigner canvas")
    if "cv_jx" not in wc:
      raise RuntimeError(f"{scenario_key}: html missing current canvas")
    if "transmission" not in wc:
      raise RuntimeError(f"{scenario_key}: html missing transmission data")

  print("visualization smoke passed")
  return 0


def _write_continuous_export_fixture_2d_with_driving(path: Path, schema_version: int | None = 1) -> None:
  """Write a 2D engine-export payload with driving_amplitude fields for M15 smoke testing."""
  import math
  payload = {
      "continuous_export": {
          **({"schema_version": schema_version} if schema_version is not None else {}),
          "kind": "continuous_space_2d",
          "source": "engine-export",
          "solver_mode": "engine",
          "potential_type": "barrier",
          "x_min": -1.0,
          "x_max": 1.0,
          "y_min": -1.0,
          "y_max": 1.0,
          "dens_max": 1.0,
          "has_driving": True,
          "driving": {"type": "sine", "amplitude": 0.4, "frequency": 1.8, "spatial_profile": "uniform"},
          "frames": [
              {
                  "t": 0.0,
                  "density_xy_flat": [0.1, 0.2, 0.3, 0.4],
                  "nx": 2,
                  "ny": 2,
                  "norm": 1.0,
                  "exp_x": 0.0,
                  "exp_y": 0.0,
                  "driving_amplitude": 0.4,
              },
              {
                  "t": 0.1,
                  "density_xy_flat": [0.2, 0.3, 0.4, 0.5],
                  "nx": 2,
                  "ny": 2,
                  "norm": 0.98,
                  "exp_x": 0.1,
                  "exp_y": -0.05,
                  "driving_amplitude": 0.28,
              },
          ],
      }
  }
  path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
  raise SystemExit(main())
