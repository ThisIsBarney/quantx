#!/usr/bin/env python3
"""Generate animated HTML visualizations for QuantX simulations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path



def _user_cli_error(message: str, hint: str | None = None, example: str | None = None) -> RuntimeError:
  """Build a user-facing CLI error with next-step guidance."""
  parts = [message]
  if hint:
    parts.append(f"hint: {hint}")
  if example:
    parts.append(f"example: {example}")
  return RuntimeError("\n".join(parts))


def _animated_page(title: str, charts: list[dict]) -> str:
  """Render a lightweight SVG animation page for 1D time-series charts."""
  charts_json = json.dumps(charts, ensure_ascii=False)
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; }}
    .chart {{ margin: 24px 0; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 8px; }}
    button {{ padding: 4px 10px; }}
    .legend {{ font-size: 12px; margin-top: 6px; }}
    .legend span {{ margin-right: 16px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Use Play/Pause or drag the slider to inspect time evolution.</p>
  <div id="charts"></div>
  <script>
  const CHARTS = {charts_json};

  function createSvgEl(tag, attrs) {{
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs || {{}})) el.setAttribute(k, v);
    return el;
  }}

  function buildChart(container, cfg, idx) {{
    const width = 900, height = 420;
    const left = 70, right = 20, top = 50, bottom = 60;
    const plotW = width - left - right, plotH = height - top - bottom;
    const allX = cfg.series.flatMap(s => s.points.map(p => p[0]));
    const allY = cfg.series.flatMap(s => s.points.map(p => p[1]));
    let xMin = Math.min(...allX), xMax = Math.max(...allX);
    let yMin = Math.min(...allY), yMax = Math.max(...allY);
    if (Math.abs(xMax - xMin) < 1e-12) xMax = xMin + 1;
    if (Math.abs(yMax - yMin) < 1e-12) yMax = yMin + 1;

    const sx = x => left + (x - xMin) / (xMax - xMin) * plotW;
    const sy = y => top + (1 - (y - yMin) / (yMax - yMin)) * plotH;

    const box = document.createElement("div");
    box.className = "chart";
    const h2 = document.createElement("h2");
    h2.textContent = cfg.title;
    box.appendChild(h2);

    const svg = createSvgEl("svg", {{ viewBox: `0 0 ${{width}} ${{height}}`, width, height }});
    box.appendChild(svg);

    svg.appendChild(createSvgEl("line", {{ x1: left, y1: top + plotH, x2: left + plotW, y2: top + plotH, stroke: "#333", "stroke-width": "1.5" }}));
    svg.appendChild(createSvgEl("line", {{ x1: left, y1: top, x2: left, y2: top + plotH, stroke: "#333", "stroke-width": "1.5" }}));

    for (let i = 0; i <= 5; i++) {{
      const gx = left + plotW * i / 5;
      const gy = top + plotH * i / 5;
      const xv = xMin + (xMax - xMin) * i / 5;
      const yv = yMax - (yMax - yMin) * i / 5;
      svg.appendChild(createSvgEl("line", {{ x1: gx, y1: top, x2: gx, y2: top + plotH, stroke: "#ddd", "stroke-width": "1" }}));
      svg.appendChild(createSvgEl("line", {{ x1: left, y1: gy, x2: left + plotW, y2: gy, stroke: "#ddd", "stroke-width": "1" }}));
      const tx = createSvgEl("text", {{ x: gx, y: top + plotH + 18, "text-anchor": "middle", "font-size": "12" }});
      tx.textContent = xv.toPrecision(3);
      svg.appendChild(tx);
      const ty = createSvgEl("text", {{ x: left - 8, y: gy + 4, "text-anchor": "end", "font-size": "12" }});
      ty.textContent = yv.toPrecision(3);
      svg.appendChild(ty);
    }}

    const xLabel = createSvgEl("text", {{ x: left + plotW / 2, y: height - 18, "text-anchor": "middle", "font-size": "12" }});
    xLabel.textContent = cfg.x_label;
    svg.appendChild(xLabel);
    const yLabel = createSvgEl("text", {{ x: 18, y: top + plotH / 2, transform: `rotate(-90 18 ${{top + plotH / 2}})`, "text-anchor": "middle", "font-size": "12" }});
    yLabel.textContent = cfg.y_label;
    svg.appendChild(yLabel);

    const paths = cfg.series.map(s => {{
      const p = createSvgEl("path", {{ fill: "none", stroke: s.color, "stroke-width": "2" }});
      svg.appendChild(p);
      return p;
    }});

    const controls = document.createElement("div");
    controls.className = "controls";
    const playBtn = document.createElement("button");
    playBtn.textContent = "Play";
    const pauseBtn = document.createElement("button");
    pauseBtn.textContent = "Pause";
    const slider = document.createElement("input");
    slider.type = "range";
    const maxFrames = Math.max(...cfg.series.map(s => s.points.length));
    slider.min = "1";
    slider.max = String(maxFrames);
    slider.value = "1";
    slider.style.width = "420px";
    const frameLabel = document.createElement("span");
    frameLabel.textContent = `frame 1/${{maxFrames}}`;
    controls.append(playBtn, pauseBtn, slider, frameLabel);
    box.appendChild(controls);

    const legend = document.createElement("div");
    legend.className = "legend";
    cfg.series.forEach(s => {{
      const item = document.createElement("span");
      item.innerHTML = `<b style="color:${{s.color}}">${{s.name}}</b>`;
      legend.appendChild(item);
    }});
    box.appendChild(legend);

    container.appendChild(box);

    let timer = null;
    function render(frame) {{
      for (let si = 0; si < cfg.series.length; si++) {{
        const pts = cfg.series[si].points.slice(0, Math.min(frame, cfg.series[si].points.length));
        const d = pts.map((p, i) => `${{i === 0 ? "M" : "L"}}${{sx(p[0]).toFixed(2)}} ${{sy(p[1]).toFixed(2)}}`).join(" ");
        paths[si].setAttribute("d", d);
      }}
      frameLabel.textContent = `frame ${{frame}}/${{maxFrames}}`;
      slider.value = String(frame);
    }}
    function play() {{
      if (timer) return;
      timer = setInterval(() => {{
        let f = Number(slider.value);
        f = f >= maxFrames ? 1 : f + 1;
        render(f);
      }}, 90);
    }}
    function pause() {{
      if (!timer) return;
      clearInterval(timer);
      timer = null;
    }}
    playBtn.onclick = play;
    pauseBtn.onclick = pause;
    slider.oninput = () => render(Number(slider.value));
    render(1);
  }}

  const container = document.getElementById("charts");
  CHARTS.forEach((cfg, idx) => buildChart(container, cfg, idx));
  </script>
</body>
</html>
"""


def _make_engine(root: Path):
  """Construct the native Python SDK wrapper when the DLL is available."""
  sys.path.insert(0, str(root / "python"))
  from quantx_engine import Engine

  search_roots = [root, *root.parents]
  dll = None
  for candidate_root in search_roots:
    candidate = candidate_root / "build-vs" / "Debug" / "quantx.dll"
    if candidate.exists():
      dll = candidate
      break
  if dll is None:
    return None
  try:
    return Engine(str(dll))
  except OSError:
    return None


def _continuous_export_schema_version(payload: dict) -> int | None:
  """Return the declared continuous-export schema version, if any."""
  raw = payload.get("schema_version")
  if raw is None:
    return None
  try:
    return int(raw)
  except (TypeError, ValueError):
    return None


def _prepare_continuous_export_payload(
    payload: dict,
    source: str,
    strict_export_schema: bool,
    fallback_hint: str,
) -> dict | None:
  """Validate the export schema version and normalize supported payloads.

  Version 1 is accepted as-is, versions above 1 emit a warning but continue in
  best-effort mode, and missing or version<1 payloads either raise immediately
  in strict mode or return ``None`` so the caller can fall back.
  """
  version = _continuous_export_schema_version(payload)
  if version is None or version < 1:
    message = f"{source}: continuous_export.schema_version is missing or below 1; expected version 1"
    hint = "re-export the payload with continuous_export.schema_version=1, or omit --strict-export-schema to allow fallback"
    if strict_export_schema:
      raise _user_cli_error(message, hint=hint)
    print(f"warning: {message}; {fallback_hint}; {hint}", file=sys.stderr)
    return None
  if version > 1:
    message = (
        f"{source}: continuous_export.schema_version={version} is newer than supported version 1; "
        "best-effort rendering will continue"
    )
    hint = "re-export the payload as schema_version=1 or update this visualizer to the newer export contract"
    if strict_export_schema:
      raise _user_cli_error(message, hint=hint)
    print(f"warning: {message}; {hint}", file=sys.stderr)
  return _normalize_engine_export_payload(payload)


def _load_continuous_export_file(path: Path, strict_export_schema: bool) -> dict | None:
  """Load a continuous-space export payload from a JSON file."""
  if not path.exists() or not path.is_file():
    raise _user_cli_error(
        f"continuous export json file not found: {path}",
        hint="pass an existing file path, or omit --continuous-export-json to let QuantX solve live",
        example="python scripts/visualize_simulation.py --scenario spatial2d --continuous-export-json out/export.json",
    )
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError as ex:
    raise _user_cli_error(
        f"continuous export json is not valid JSON: {path}",
        hint="re-run the exporter and make sure the file is a JSON object with a continuous_export.frames array",
        example="python scripts/visualize_simulation.py --scenario spatial3d --continuous-export-json out/export.json",
    ) from ex
  if isinstance(data, dict):
    if isinstance(data.get("continuous_export"), dict):
      return _prepare_continuous_export_payload(
          data["continuous_export"],
          source=f"engine-export file {path}",
          strict_export_schema=strict_export_schema,
          fallback_hint="falling back to live engine execution or Python reference solver",
      )
    export = data.get("export")
    if isinstance(export, dict):
      for key in ("continuous_export", "continuous", "frames"):
        value = export.get(key)
        if isinstance(value, dict):
          return _prepare_continuous_export_payload(
              value,
              source=f"engine-export file {path}",
              strict_export_schema=strict_export_schema,
              fallback_hint="falling back to live engine execution or Python reference solver",
          )
        if isinstance(value, list):
          return _prepare_continuous_export_payload(
              {"frames": value},
              source=f"engine-export file {path}",
              strict_export_schema=strict_export_schema,
              fallback_hint="falling back to live engine execution or Python reference solver",
          )
    if isinstance(data.get("frames"), list):
      return _prepare_continuous_export_payload(
          data,
          source=f"engine-export file {path}",
          strict_export_schema=strict_export_schema,
          fallback_hint="falling back to live engine execution or Python reference solver",
      )
  raise _user_cli_error(
      f"continuous export json must contain a frames payload: {path}",
      hint="use an engine export file produced by QuantX, or omit --continuous-export-json to solve live",
      example="python scripts/visualize_simulation.py --scenario spatial2d --solver-mode python",
  )


def _extract_continuous_export_from_result(result: dict) -> dict | None:
  """Normalize any continuous export payload returned by the native runtime."""
  if not isinstance(result, dict):
    return None

  candidates: list[dict] = []
  for key in ("continuous_export", "continuous_frames", "continuous"):
    value = result.get(key)
    if isinstance(value, dict):
      candidates.append(value)
    elif isinstance(value, list):
      candidates.append({"frames": value})

  export = result.get("export")
  if isinstance(export, dict):
    for key in ("continuous_export", "continuous", "frames"):
      value = export.get(key)
      if isinstance(value, dict):
        candidates.append(value)
      elif isinstance(value, list):
        candidates.append({"frames": value})

  frames = result.get("frames")
  if isinstance(frames, list):
    candidates.append({"frames": frames})

  for candidate in candidates:
    if isinstance(candidate, dict) and isinstance(candidate.get("frames"), list) and candidate["frames"]:
      return candidate
  return None


def _attach_continuous_metadata(data: dict, solver_mode: str, source: str) -> dict:
  """Clone a payload and annotate the source used to produce it."""
  payload = dict(data)
  payload.setdefault("frames", [])
  payload["solver_mode"] = solver_mode
  payload["source"] = source
  return payload


def _require_list(value: object, label: str) -> list:
  """Return a list value or raise a diagnostic error."""
  if not isinstance(value, list):
    raise RuntimeError(f"engine-export payload field '{label}' must be a list")
  return value


def _reshape_flat_to_rows(flat: list[float], rows: int, cols: int, label: str) -> list[list[float]]:
  """Reshape a flat row-major list into nested lists and validate its size."""
  if rows <= 0 or cols <= 0:
    raise RuntimeError(f"engine-export payload field '{label}' requires positive row and column sizes")
  if len(flat) != rows * cols:
    raise RuntimeError(
        f"engine-export payload field '{label}' has {len(flat)} entries but expected {rows * cols} for {rows}x{cols}"
    )
  return [flat[r * cols:(r + 1) * cols] for r in range(rows)]


def _reshape_flat_points(flat: list[float], label: str) -> list[list[float]]:
  """Reshape a flat [x,y,z,density] list into point rows and validate its size."""
  if len(flat) % 4 != 0:
    raise RuntimeError(
        f"engine-export payload field '{label}' must contain a multiple of 4 values for [x,y,z,density] tuples"
    )
  return [flat[i:i + 4] for i in range(0, len(flat), 4)]


def _normalize_engine_export_payload(payload: dict) -> dict:
  """Convert native engine export payload to visualizer-friendly frame shapes."""
  if not isinstance(payload, dict):
    raise RuntimeError("engine-export payload must be a JSON object")
  frames = payload.get("frames")
  if not isinstance(frames, list) or not frames:
    raise RuntimeError("engine-export payload must contain a non-empty 'frames' array")

  normalized_frames = []
  dens_max = float(payload.get("dens_max", 0.0) or 0.0)
  base = frames[0] if isinstance(frames[0], dict) else {}
  for frame_idx, raw_frame in enumerate(frames):
    if not isinstance(raw_frame, dict):
      raise RuntimeError(f"engine-export frame {frame_idx} must be a JSON object")

    frame = dict(raw_frame)
    nx = int(frame.get("nx", base.get("nx", 0)) or 0)
    ny = int(frame.get("ny", base.get("ny", 0)) or 0)
    nz = int(frame.get("nz", base.get("nz", 0)) or 0)

    if nz <= 0:
      if "density" in frame and isinstance(frame["density"], list):
        density = _require_list(frame["density"], f"frames[{frame_idx}].density")
      else:
        density_flat = _require_list(frame.get("density_xy_flat"), f"frames[{frame_idx}].density_xy_flat")
        if nx <= 0 or ny <= 0:
          raise RuntimeError(f"engine-export frame {frame_idx} requires nx and ny for 2D density normalization")
        density = _reshape_flat_to_rows([float(v) for v in density_flat], ny, nx, f"frames[{frame_idx}].density_xy_flat")
      frame["density"] = density
      frame["exp_x"] = float(frame.get("exp_x", frame.get("x_mean", 0.0)) or 0.0)
      frame["exp_y"] = float(frame.get("exp_y", frame.get("y_mean", 0.0)) or 0.0)
      frame["t"] = float(frame.get("t", frame.get("time", 0.0)) or 0.0)
      dens_max = max(dens_max, max((max(row) for row in density), default=0.0))
      # Optional phase data (arg(ψ₀))
      if "phase_xy_flat" in frame and nx > 0 and ny > 0:
        frame["phase"] = _reshape_flat_to_rows(
            [float(v) for v in frame["phase_xy_flat"]], ny, nx, f"frames[{frame_idx}].phase_xy_flat"
        )
      # Optional Dirac spinor component densities
      for comp, key in (("dirac_s0_xy_flat", "dirac_s0"), ("dirac_s1_xy_flat", "dirac_s1")):
        if comp in frame and nx > 0 and ny > 0:
          frame[key] = _reshape_flat_to_rows(
              [float(v) for v in frame[comp]], ny, nx, f"frames[{frame_idx}].{comp}"
          )
    else:
      if nx <= 0 or ny <= 0 or nz <= 0:
        raise RuntimeError(f"engine-export frame {frame_idx} requires nx, ny and nz for 3D density normalization")
      xy_flat = _require_list(frame.get("density_xy_flat"), f"frames[{frame_idx}].density_xy_flat")
      xz_flat = _require_list(frame.get("density_xz_flat"), f"frames[{frame_idx}].density_xz_flat")
      yz_flat = _require_list(frame.get("density_yz_flat"), f"frames[{frame_idx}].density_yz_flat")
      points_flat = _require_list(frame.get("points_xyzd_flat"), f"frames[{frame_idx}].points_xyzd_flat")
      frame["xy"] = _reshape_flat_to_rows([float(v) for v in xy_flat], ny, nx, f"frames[{frame_idx}].density_xy_flat")
      frame["xz"] = _reshape_flat_to_rows([float(v) for v in xz_flat], nz, nx, f"frames[{frame_idx}].density_xz_flat")
      frame["yz"] = _reshape_flat_to_rows([float(v) for v in yz_flat], nz, ny, f"frames[{frame_idx}].density_yz_flat")
      frame["points"] = _reshape_flat_points([float(v) for v in points_flat], f"frames[{frame_idx}].points_xyzd_flat")
      frame["t"] = float(frame.get("t", frame.get("time", 0.0)) or 0.0)
      frame["point_count"] = len(frame["points"])
      frame["point_vmax"] = max((float(p[3]) for p in frame["points"]), default=0.0)
      dens_max = max(
          dens_max,
          max((max(row) for row in frame["xy"]), default=0.0),
          max((max(row) for row in frame["xz"]), default=0.0),
          max((max(row) for row in frame["yz"]), default=0.0),
      )
      # Optional phase slices
      if "phase_xy_flat" in frame:
        frame["phase_xy"] = _reshape_flat_to_rows(
            [float(v) for v in frame["phase_xy_flat"]], ny, nx, f"frames[{frame_idx}].phase_xy_flat"
        )
      if "phase_xz_flat" in frame:
        frame["phase_xz"] = _reshape_flat_to_rows(
            [float(v) for v in frame["phase_xz_flat"]], nz, nx, f"frames[{frame_idx}].phase_xz_flat"
        )
      if "phase_yz_flat" in frame:
        frame["phase_yz"] = _reshape_flat_to_rows(
            [float(v) for v in frame["phase_yz_flat"]], nz, ny, f"frames[{frame_idx}].phase_yz_flat"
        )
      # Optional full voxel grid for marching cubes (kept as flat list)
      if "density_xyz_flat" in frame:
        frame["density_xyz"] = [float(v) for v in frame["density_xyz_flat"]]

    normalized_frames.append(frame)

  out = dict(payload)
  out["frames"] = normalized_frames
  out["dens_max"] = max(dens_max, float(out.get("dens_max", 0.0) or 0.0))
  first = normalized_frames[0]
  out.setdefault("x_min", float(first.get("x_min", -1.0)))
  out.setdefault("x_max", float(first.get("x_max", 1.0)))
  out.setdefault("y_min", float(first.get("y_min", -1.0)))
  out.setdefault("y_max", float(first.get("y_max", 1.0)))
  out.setdefault("z_min", float(first.get("z_min", -1.0)))
  out.setdefault("z_max", float(first.get("z_max", 1.0)))
  out.setdefault("boundary", first.get("boundary", out.get("boundary", "periodic")))
  out.setdefault("potential_type", first.get("potential_kind", out.get("potential_type", "none")))
  # Propagate feature flags derived from the normalized frames
  out["has_phase"] = bool(out.get("has_phase")) or any("phase" in f or "phase_xy" in f for f in normalized_frames)
  out["has_dirac"] = bool(out.get("has_dirac")) or any("dirac_s0" in f for f in normalized_frames)
  out["has_volume_grid"] = bool(out.get("has_volume_grid")) or any("density_xyz" in f for f in normalized_frames)
  out["has_driving"] = bool(out.get("has_driving")) or any("driving_amplitude" in f for f in normalized_frames)
  return out


def _continuous_problem_2d_request(samples_t: int, nx: int, ny: int, potential_type: str) -> dict:
  """Build the native 2D problem payload used when asking the engine for frames."""
  nx = max(32, nx)
  ny = max(32, ny)
  steps = max(1, samples_t - 1)
  return {
      "kind": "continuous_space_2d",
      "equation": "Schrodinger",
      "observables": ["norm", "x_mean", "y_mean", "sigma_x", "sigma_y"],
      "continuous_space_2d": {
          "nx": nx,
          "ny": ny,
          "dx": 16.0 / max(1, nx - 1),
          "dy": 16.0 / max(1, ny - 1),
          "dt": 3.0 / steps,
          "steps": steps,
          "boundary": "periodic",
          "potential_kind": potential_type,
          "potential_params": {
              "height": 0.9,
              "omega": 0.28,
              "wall": 5.0,
              "sigma_x": 0.6,
              "sigma_y": 0.6,
              "x_half_width": 6.0,
              "y_half_width": 6.0,
          },
          "initial_state": {
              "kind": "gaussian",
              "params": {
                  "psi_x0": -4.5,
                  "psi_y0": -2.0,
                  "psi_sigma": 0.7,
                  "psi_kx": 2.2,
                  "psi_ky": 1.1,
              },
          },
      },
  }


def _continuous_problem_2d_driven_request(
    samples_t: int, nx: int, ny: int, potential_type: str, driving: dict
) -> dict:
  """Build the native 2D problem payload with a driving field for engine execution."""
  base = _continuous_problem_2d_request(samples_t, nx, ny, potential_type)
  base["continuous_space_2d"]["driving"] = driving
  return base


def _continuous_problem_3d_request(samples_t: int, nx: int, ny: int, nz: int, potential_type: str, boundary: str) -> dict:
  """Build the native 3D problem payload used when asking the engine for frames."""
  nx = max(16, nx)
  ny = max(16, ny)
  nz = max(16, nz)
  steps = max(1, samples_t - 1)
  return {
      "kind": "continuous_space_3d",
      "equation": "Schrodinger",
      "observables": ["norm", "x_mean", "y_mean", "z_mean", "sigma_x", "sigma_y", "sigma_z"],
      "continuous_space_3d": {
          "nx": nx,
          "ny": ny,
          "nz": nz,
          "dx": 12.0 / max(1, nx - 1),
          "dy": 12.0 / max(1, ny - 1),
          "dz": 12.0 / max(1, nz - 1),
          "dt": 2.5 / steps,
          "steps": steps,
          "boundary": boundary,
          "potential_kind": potential_type,
          "potential_params": {
              "height": 0.95,
              "omega": 0.25,
              "wall": 5.5,
              "sigma_x": 0.55,
              "x_half_width": 4.8,
              "y_half_width": 4.8,
              "z_half_width": 4.8,
          },
          "initial_state": {
              "kind": "gaussian",
              "params": {
                  "psi_x0": -2.0,
                  "psi_y0": -1.2,
                  "psi_z0": -0.8,
                  "psi_sigma": 0.7,
                  "psi_kx": 1.8,
                  "psi_ky": 1.0,
                  "psi_kz": 0.8,
              },
          },
      },
  }


def _resolve_continuous_payload(
    engine,
    problem: dict,
    solver_mode: str,
    reference_builder,
    export_json_path: Path | None,
    strict_export_schema: bool,
) -> dict:
  """Prefer engine-exported frames and fall back to the Python reference solver."""
  if export_json_path is not None:
    if not export_json_path.exists() or not export_json_path.is_file():
      raise _user_cli_error(
          f"continuous export json file not found: {export_json_path}",
          hint="pass an existing file path, or omit --continuous-export-json to let QuantX solve live",
          example="python scripts/visualize_simulation.py --scenario spatial2d --continuous-export-json out/export.json",
      )
    try:
      export = _load_continuous_export_file(export_json_path, strict_export_schema)
    except Exception as ex:
      if strict_export_schema:
        raise _user_cli_error(
            f"continuous visualization received invalid engine export file: {ex}",
            hint="fix the export file or remove --strict-export-schema to allow fallback to live solving",
            example="python scripts/visualize_simulation.py --scenario spatial2d --continuous-export-json out/export.json",
        ) from ex
      print(
          f"warning: {ex}; falling back to live engine execution or Python reference solver",
          file=sys.stderr,
      )
    else:
      if export is None:
        if strict_export_schema:
          raise _user_cli_error(
              "continuous visualization requested strict export-schema validation but the export schema is incompatible",
              hint="re-export the payload with schema_version=1 or remove --strict-export-schema to continue best-effort rendering",
              example="python scripts/visualize_simulation.py --scenario spatial2d --continuous-export-json out/export.json",
          )
      else:
        return _attach_continuous_metadata(export, solver_mode, "engine-export-file")

  if solver_mode in {"auto", "engine"} and engine is not None:
    result = engine.run(problem, {"enable_checkpoint": False, "export_continuous_frames": True})
    export = None
    extractor = getattr(engine, "extract_continuous_export", None)
    if callable(extractor):
      export = extractor(result)
    if export is None:
      export = _extract_continuous_export_from_result(result)
    if export is not None:
      try:
        normalized = _prepare_continuous_export_payload(
            export,
            source="native engine result",
            strict_export_schema=strict_export_schema,
            fallback_hint="falling back to Python reference solver",
        )
      except Exception as ex:
        if strict_export_schema:
          raise _user_cli_error(
              f"continuous visualization received invalid engine export payload: {ex}",
              hint="check the native exporter contract or rerun without --strict-export-schema to allow a Python fallback",
              example="python scripts/visualize_simulation.py --scenario spatial2d --solver-mode engine",
          ) from ex
        print(f"warning: {ex}; falling back to Python reference solver", file=sys.stderr)
      else:
        if normalized is not None:
          return _attach_continuous_metadata(normalized, solver_mode, "engine-export")
    if solver_mode == "engine":
      raise RuntimeError("continuous visualization requested solver-mode=engine but the native result did not include a frames export")

  if solver_mode == "engine":
    raise _user_cli_error(
        "continuous visualization requested solver-mode=engine but the native engine is unavailable",
        hint="build the native engine with `powershell -Command .\\scripts\\build.ps1`, or provide --continuous-export-json to render a precomputed export",
        example="python scripts/visualize_simulation.py --scenario spatial2d --solver-mode python",
    )

  return _attach_continuous_metadata(reference_builder(), solver_mode, "python-reference")


