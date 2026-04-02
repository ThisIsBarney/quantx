#!/usr/bin/env python3
"""Generate animated HTML visualizations for QuantX simulations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


CLI_EXAMPLES = """Examples:
  python scripts/visualize_simulation.py --scenario spatial2d --solver-mode python
  python scripts/visualize_simulation.py --scenario spatial2d --solver-mode engine --continuous-export-json out/export.json --strict-export-schema
  python scripts/visualize_simulation.py --scenario spatial3d --continuous-export-json out/export.json --spatial3d-export-json
"""


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


def _spatial_wavepacket_frames(
    samples_t: int, samples_x: int, potential_type: str, source: str = "python-reference"
) -> dict:
  """Build a 1D reference solution for the spatial wavepacket animation."""
  # Finite-difference + Crank-Nicolson for 1D Schrodinger equation (hbar=m=1).
  x_min, x_max = -12.0, 12.0
  t_min, t_max = 0.0, 4.0
  n = max(80, samples_x)
  nt = max(20, samples_t)
  xs = [x_min + (x_max - x_min) * i / (n - 1) for i in range(n)]
  dx = (x_max - x_min) / (n - 1)
  dt = (t_max - t_min) / (nt - 1)

  # Initial right-moving Gaussian wavepacket.
  x0, sigma0, k0 = -6.0, 0.7, 2.8
  psi = []
  for x in xs:
    amp = math.exp(-((x - x0) ** 2) / (4.0 * sigma0 * sigma0))
    phase = complex(math.cos(k0 * x), math.sin(k0 * x))
    psi.append(amp * phase)

  # Potential profiles for different spatial scenarios.
  if potential_type == "free":
    potential = [0.0 for _ in xs]
  elif potential_type == "harmonic":
    omega = 0.35
    potential = [0.5 * (omega * omega) * (x * x) for x in xs]
  elif potential_type == "well":
    wall = 5.0
    potential = [0.0 if abs(x) <= 6.0 else wall for x in xs]
  else:  # barrier
    v0, vb_w = 0.8, 0.7
    potential = [v0 * math.exp(-(x * x) / (2.0 * vb_w * vb_w)) for x in xs]

  # Tridiagonal coefficients for A psi_{n+1} = d (interior points only).
  off_a = complex(0.0, -dt / (4.0 * dx * dx))
  off_b = complex(0.0, dt / (4.0 * dx * dx))
  diag_a = []
  diag_b = []
  for j in range(1, n - 1):
    h_diag = (1.0 / (dx * dx)) + potential[j]
    diag_a.append(complex(1.0, 0.5 * dt * h_diag))
    diag_b.append(complex(1.0, -0.5 * dt * h_diag))

  def solve_tridiagonal(a: complex, b_diag: list[complex], c: complex, d: list[complex]) -> list[complex]:
    m = len(d)
    cp = [0j] * m
    dp = [0j] * m
    cp[0] = c / b_diag[0]
    dp[0] = d[0] / b_diag[0]
    for i in range(1, m):
      denom = b_diag[i] - a * cp[i - 1]
      cp[i] = (c / denom) if i < m - 1 else 0j
      dp[i] = (d[i] - a * dp[i - 1]) / denom
    x = [0j] * m
    x[-1] = dp[-1]
    for i in range(m - 2, -1, -1):
      x[i] = dp[i] - cp[i] * x[i + 1]
    return x

  def normalize(arr: list[complex]) -> None:
    s = sum((z.real * z.real + z.imag * z.imag) for z in arr) * dx
    if s > 0.0:
      inv = 1.0 / math.sqrt(s)
      for i in range(len(arr)):
        arr[i] *= inv

  normalize(psi)
  frames = []
  n_particles = 56

  for it in range(nt):
    t = t_min + (t_max - t_min) * it / max(1, nt - 1)
    density = [(z.real * z.real + z.imag * z.imag) for z in psi]
    phase = [math.atan2(z.imag, z.real) for z in psi]
    z_norm = sum(density) * dx
    if z_norm > 0.0:
      density = [p / z_norm for p in density]

    exp_x = sum(x * p for x, p in zip(xs, density)) * dx
    cdf, acc = [], 0.0
    for p in density:
      acc += p * dx
      cdf.append(acc)
    cloud = []
    j = 0
    for i in range(n_particles):
      q = (i + 0.5) / n_particles
      while j < len(cdf) - 1 and cdf[j] < q:
        j += 1
      cloud.append(xs[j])
    frames.append({"t": t, "density": density, "phase": phase, "exp_x": exp_x, "cloud": cloud})

    if it == nt - 1:
      break

    # Build RHS d = B * psi_n (Dirichlet boundary psi[0]=psi[-1]=0).
    rhs = []
    for k in range(n - 2):
      j = k + 1
      val = diag_b[k] * psi[j] + off_b * psi[j - 1] + off_b * psi[j + 1]
      rhs.append(val)
    interior = solve_tridiagonal(off_a, diag_a, off_a, rhs)
    psi = [0j] + interior + [0j]
    normalize(psi)

  return {
      "xs": xs,
      "frames": frames,
      "x_min": x_min,
      "x_max": x_max,
      "potential": potential,
      "potential_type": potential_type,
      "source": source,
  }


def _spatial_wavepacket_page(data: dict) -> str:
  """Render the 1D spatial wavepacket animation page."""
  payload = json.dumps(data, ensure_ascii=False)
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>QuantX Animation - Spatial Wavepacket ({data.get("potential_type","barrier")})</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 8px; }}
    button {{ padding: 4px 10px; }}
  </style>
</head>
<body>
  <h1>QuantX Animation - Spatial Wavepacket ({data.get("potential_type","barrier")})</h1>
  <p>1D Schrodinger solver (finite difference + Crank-Nicolson): showing spatial probability density |psi(x,t)|^2, phase color, potential profile, sampled particle cloud, and <b>data source:</b> {data.get("source","python-reference")}.</p>
  <svg id="svg" viewBox="0 0 1000 520" width="1000" height="520" xmlns="http://www.w3.org/2000/svg"></svg>
  <div class="controls">
    <button id="play">Play</button>
    <button id="pause">Pause</button>
    <input id="slider" type="range" min="1" step="1" style="width:520px"/>
    <span id="label"></span>
  </div>
  <script>
  const DATA = {payload};
  const svg = document.getElementById("svg");
  const play = document.getElementById("play");
  const pause = document.getElementById("pause");
  const slider = document.getElementById("slider");
  const label = document.getElementById("label");

  const W=1000,H=520,L=80,R=30,T=40,B=120;
  const PW=W-L-R, PH=H-T-B;
  const xMin=DATA.x_min, xMax=DATA.x_max;
  const yMax=Math.max(...DATA.frames.flatMap(f=>f.density)) * 1.1;
  const vMax=Math.max(1e-12, ...DATA.potential.map(v=>Math.abs(v)));

  const sx=x=>L+(x-xMin)/(xMax-xMin)*PW;
  const sy=y=>T+(1-y/yMax)*PH;

  function el(tag, attrs) {{
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k,v] of Object.entries(attrs||{{}})) e.setAttribute(k,v);
    return e;
  }}
  function phaseColor(phi) {{
    const h = ((phi + Math.PI) / (2*Math.PI)) * 360;
    return `hsl(${{h}},85%,50%)`;
  }}

  svg.appendChild(el("line",{{x1:L,y1:T+PH,x2:L+PW,y2:T+PH,stroke:"#333","stroke-width":"1.5"}}));
  svg.appendChild(el("line",{{x1:L,y1:T,x2:L,y2:T+PH,stroke:"#333","stroke-width":"1.5"}}));
  for(let i=0;i<=5;i++) {{
    const gx=L+PW*i/5, gy=T+PH*i/5;
    svg.appendChild(el("line",{{x1:gx,y1:T,x2:gx,y2:T+PH,stroke:"#ddd"}}));
    svg.appendChild(el("line",{{x1:L,y1:gy,x2:L+PW,y2:gy,stroke:"#ddd"}}));
    const tx=el("text",{{x:gx,y:T+PH+20,"text-anchor":"middle","font-size":"12"}}); tx.textContent=(xMin+(xMax-xMin)*i/5).toPrecision(3); svg.appendChild(tx);
    const ty=el("text",{{x:L-8,y:gy+4,"text-anchor":"end","font-size":"12"}}); ty.textContent=(yMax-yMax*i/5).toPrecision(3); svg.appendChild(ty);
  }}
  const xl=el("text",{{x:L+PW/2,y:H-26,"text-anchor":"middle","font-size":"12"}}); xl.textContent="position x"; svg.appendChild(xl);
  const yl=el("text",{{x:20,y:T+PH/2,transform:`rotate(-90 20 ${{T+PH/2}})`,"text-anchor":"middle","font-size":"12"}}); yl.textContent="|psi(x,t)|^2"; svg.appendChild(yl);

  const probPath = el("path",{{fill:"none",stroke:"#0b84f3","stroke-width":"2.5"}}); svg.appendChild(probPath);
  const potPath = el("path",{{fill:"none",stroke:"#8e44ad","stroke-width":"2","stroke-dasharray":"5 4"}}); svg.appendChild(potPath);
  const expLine = el("line",{{x1:L,y1:T,x2:L,y2:T+PH,stroke:"#f39c12","stroke-width":"2","stroke-dasharray":"6 4"}}); svg.appendChild(expLine);
  const phaseDots = [];
  for(let i=0;i<DATA.xs.length;i+=4) {{
    const c = el("circle",{{cx:L,cy:T+PH,r:2.1,fill:"#999"}});
    phaseDots.push(c); svg.appendChild(c);
  }}
  const cloudDots = [];
  for(let i=0;i<56;i++) {{
    const c = el("circle",{{cx:L,cy:T+PH+55,r:4,fill:"#2e8b57",opacity:"0.78"}});
    cloudDots.push(c); svg.appendChild(c);
  }}
  const cloudAxis = el("line",{{x1:L,y1:T+PH+55,x2:L+PW,y2:T+PH+55,stroke:"#555","stroke-width":"1"}}); svg.appendChild(cloudAxis);
  const cloudLabel = el("text",{{x:L,y:T+PH+84,"font-size":"12"}}); cloudLabel.textContent="sampled position cloud (from |psi|^2)"; svg.appendChild(cloudLabel);
  const potLabel = el("text",{{x:L+PW-6,y:T+16,"text-anchor":"end","font-size":"12",fill:"#8e44ad"}}); potLabel.textContent="potential V(x) (scaled)"; svg.appendChild(potLabel);

  slider.max = String(DATA.frames.length);
  slider.value = "1";

  let timer = null;
  function render(frameIdx) {{
    const f = DATA.frames[frameIdx-1];
    const d = DATA.xs.map((x,i)=>`${{i===0?"M":"L"}}${{sx(x).toFixed(2)}} ${{sy(f.density[i]).toFixed(2)}}`).join(" ");
    probPath.setAttribute("d", d);
    const dp = DATA.xs.map((x,i)=>{{
      const pv = (DATA.potential[i] / vMax) * (0.85 * yMax);
      return `${{i===0?"M":"L"}}${{sx(x).toFixed(2)}} ${{sy(pv).toFixed(2)}}`;
    }}).join(" ");
    potPath.setAttribute("d", dp);
    const ex = sx(f.exp_x);
    expLine.setAttribute("x1", ex); expLine.setAttribute("x2", ex);

    let di = 0;
    for(let i=0;i<DATA.xs.length;i+=4) {{
      phaseDots[di].setAttribute("cx", sx(DATA.xs[i]));
      phaseDots[di].setAttribute("cy", sy(f.density[i]));
      phaseDots[di].setAttribute("fill", phaseColor(f.phase[i]));
      di++;
    }}
    for(let i=0;i<cloudDots.length;i++) {{
      cloudDots[i].setAttribute("cx", sx(f.cloud[i]));
    }}
    label.textContent = `frame ${{frameIdx}}/${{DATA.frames.length}}, t=${{f.t.toFixed(3)}}`;
    slider.value = String(frameIdx);
  }}
  function doPlay() {{
    if (timer) return;
    timer = setInterval(()=>{{
      let f = Number(slider.value);
      f = f >= DATA.frames.length ? 1 : f + 1;
      render(f);
    }}, 85);
  }}
  function doPause() {{
    if(!timer) return;
    clearInterval(timer);
    timer = null;
  }}
  play.onclick = doPlay;
  pause.onclick = doPause;
  slider.oninput = ()=>render(Number(slider.value));
  render(1);
  </script>
</body>
</html>"""


def _spatial_wavepacket_2d_frames(
    samples_t: int,
    nx: int,
    ny: int,
    potential_type: str,
    source: str = "python-reference",
    driving: dict | None = None,
) -> dict:
  """Build a 2D reference solution for the spatial wavepacket animation.

  driving: optional dict with keys type (sine/gaussian_pulse/chirp),
           amplitude, frequency, phase, t_center, t_width, spatial_profile.
  """
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("2D spatial visualization requires numpy") from ex

  x_min, x_max = -8.0, 8.0
  y_min, y_max = -8.0, 8.0
  t_min, t_max = 0.0, 3.0
  nx = max(32, nx)
  ny = max(32, ny)
  nt = max(16, samples_t)

  xs = np.linspace(x_min, x_max, nx)
  ys = np.linspace(y_min, y_max, ny)
  dx = float(xs[1] - xs[0])
  dy = float(ys[1] - ys[0])
  dt = (t_max - t_min) / (nt - 1)

  X, Y = np.meshgrid(xs, ys, indexing="xy")

  # Initial 2D Gaussian packet with momentum.
  x0, y0 = -4.5, -2.0
  sigma = 0.7
  kx0, ky0 = 2.2, 1.1
  envelope = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (4.0 * sigma * sigma))
  phase = np.exp(1j * (kx0 * X + ky0 * Y))
  psi = envelope * phase

  if potential_type == "free":
    V_static = np.zeros_like(X)
  elif potential_type == "harmonic":
    omega = 0.28
    V_static = 0.5 * (omega * omega) * (X * X + Y * Y)
  elif potential_type == "well":
    wall = 5.0
    V_static = np.where((np.abs(X) <= 6.0) & (np.abs(Y) <= 6.0), 0.0, wall)
  else:  # barrier
    V_static = 0.9 * np.exp(-(X * X) / (2.0 * 0.6 * 0.6))

  # Driving field spatial profile.
  has_driving = driving is not None and driving.get("type", "none") != "none" and driving.get("amplitude", 0.0) != 0.0
  if has_driving:
    sp = driving.get("spatial_profile", "uniform")
    if sp == "dipole_x":
      V_drive_spatial = X.copy()
    elif sp == "dipole_y":
      V_drive_spatial = Y.copy()
    else:
      V_drive_spatial = np.ones_like(X)

    def drive_envelope(t: float) -> float:
      A = driving.get("amplitude", 0.0)
      f = driving.get("frequency", 1.0)
      ph = driving.get("phase", 0.0)
      dtype = driving.get("type", "sine")
      if dtype == "sine":
        return A * math.cos(f * t + ph)
      if dtype == "gaussian_pulse":
        tc = driving.get("t_center", 0.0)
        tw = max(1e-30, driving.get("t_width", 1.0))
        return A * math.exp(-0.5 * ((t - tc) / tw) ** 2) * math.cos(f * t + ph)
      if dtype == "chirp":
        tw = driving.get("t_width", 0.0)  # chirp rate α
        return A * math.cos((f + 0.5 * tw * t) * t + ph)
      return 0.0
  else:
    def drive_envelope(t: float) -> float:
      return 0.0
    V_drive_spatial = np.zeros_like(X)

  kx_arr = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
  ky_arr = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
  KX, KY = np.meshgrid(kx_arr, ky_arr, indexing="xy")
  kinetic_phase = np.exp(-1j * 0.5 * (KX * KX + KY * KY) * dt)

  def normalize(a):
    nrm = float(np.sum(np.abs(a) ** 2) * dx * dy)
    if nrm > 0:
      a /= math.sqrt(nrm)
    return a

  psi = normalize(psi)
  frames = []
  dens_max = 1e-12

  for it in range(nt):
    t = t_min + (t_max - t_min) * it / max(1, nt - 1)
    dens = np.abs(psi) ** 2
    dens_max = max(dens_max, float(np.max(dens)))
    exp_x = float(np.sum(X * dens) * dx * dy)
    exp_y = float(np.sum(Y * dens) * dx * dy)
    drv_amp = drive_envelope(t)
    frame: dict = {
        "t": t,
        "density": dens.tolist(),
        "phase": np.angle(psi).tolist(),
        "exp_x": exp_x,
        "exp_y": exp_y,
    }
    if has_driving:
      frame["driving_amplitude"] = drv_amp
    frames.append(frame)
    if it == nt - 1:
      break

    # Strang splitting: V/2 -> T -> V/2 (time-centered potential at t+dt/2)
    t_mid = t + 0.5 * dt
    drv_mid = drive_envelope(t_mid)
    V_total = V_static + drv_mid * V_drive_spatial
    potential_half = np.exp(-1j * V_total * dt * 0.5)
    psi = potential_half * psi
    psi_k = np.fft.fft2(psi)
    psi = np.fft.ifft2(kinetic_phase * psi_k)
    psi = potential_half * psi
    psi = normalize(psi)

  out: dict = {
      "x_min": x_min,
      "x_max": x_max,
      "y_min": y_min,
      "y_max": y_max,
      "potential_type": potential_type,
      "frames": frames,
      "dens_max": dens_max,
      "has_phase": True,
      "source": source,
  }
  if has_driving:
    out["has_driving"] = True
    out["driving"] = driving
  return out


def _spatial_wavepacket_2d_page(data: dict) -> str:
  """Render the 2D spatial wavepacket heatmap animation page with phase toggle and colorbar."""
  payload = json.dumps(data, ensure_ascii=False)
  has_phase = bool(data.get("has_phase"))
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>QuantX Animation - 2D Spatial Wavepacket ({data.get("potential_type","barrier")})</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; background: #f8f8f8; }}
    h1 {{ font-size: 18px; margin-bottom: 4px; }}
    .row {{ display: flex; gap: 12px; align-items: flex-start; margin-top: 8px; }}
    .cv-wrap {{ position: relative; }}
    canvas#cv {{ border: 1px solid #aaa; display: block; }}
    canvas#colorbar {{ border: 1px solid #aaa; display: block; cursor: default; }}
    .cb-labels {{ display: flex; flex-direction: column; justify-content: space-between; font-size: 11px; color: #555; height: 400px; padding: 2px 0; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 8px; flex-wrap: wrap; }}
    button {{ padding: 4px 10px; }}
    select {{ padding: 3px 6px; }}
    label {{ font-size: 13px; }}
  </style>
</head>
<body>
  <h1>QuantX Animation - 2D Spatial Wavepacket ({data.get("potential_type","barrier")})</h1>
  <p style="font-size:13px;color:#555">Split-step Fourier: |ψ(x,y,t)|² heatmap. Source: <b>{data.get("source","python-reference")}</b>.</p>
  <div class="controls">
    <button id="play">Play</button>
    <button id="pause">Pause</button>
    <input id="slider" type="range" min="1" step="1" style="width:400px"/>
    <span id="label" style="font-size:13px"></span>
    {"<label>Color: <select id='colorMode'><option value='density'>Density |ψ|²</option><option value='phase'>Phase arg(ψ)</option></select></label>" if has_phase else ""}
  </div>
  <div class="row">
    <div class="cv-wrap">
      <canvas id="cv" width="512" height="512"></canvas>
    </div>
    <div>
      <canvas id="colorbar" width="28" height="400"></canvas>
    </div>
    <div class="cb-labels" id="cbLabels">
      <span>max</span><span></span><span>0</span>
    </div>
  </div>
  <script>
  const DATA = {payload};
  const cv = document.getElementById("cv");
  const ctx = cv.getContext("2d");
  const slider = document.getElementById("slider");
  const label = document.getElementById("label");
  const cbCanvas = document.getElementById("colorbar");
  const cbCtx = cbCanvas.getContext("2d");
  const cbLabels = document.getElementById("cbLabels");
  slider.max = String(DATA.frames.length);
  slider.value = "1";

  function colorMapDensity(v) {{
    const t = Math.max(0, Math.min(1, v));
    return [Math.floor(255*Math.sqrt(t)), Math.floor(255*t*t), Math.floor(255*(1-t)), 255];
  }}
  function colorMapPhase(phi) {{
    const h = ((phi + Math.PI) / (2*Math.PI)) * 360;
    const s = 0.88, l = 0.52, a = s * Math.min(l, 1-l);
    const f = n => Math.round(255*(l - a*Math.max(-1, Math.min((n+h/30)%12-3, 9-(n+h/30)%12, 1))));
    return [f(0), f(8), f(4), 255];
  }}

  function drawColorbar(mode) {{
    const h = cbCanvas.height;
    for (let y = 0; y < h; y++) {{
      const t = 1 - y/h;
      const rgba = (mode === "phase") ? colorMapPhase((t*2-1)*Math.PI) : colorMapDensity(t);
      cbCtx.fillStyle = `rgb(${{rgba[0]}},${{rgba[1]}},${{rgba[2]}})`;
      cbCtx.fillRect(0, y, cbCanvas.width, 1);
    }}
    if (mode === "phase") {{
      cbLabels.innerHTML = "<span>+π</span><span>0</span><span>−π</span>";
    }} else {{
      const densMax = Math.max(1e-12, DATA.dens_max||0);
      cbLabels.innerHTML = `<span>${{densMax.toExponential(1)}}</span><span></span><span>0</span>`;
    }}
  }}

  function render(frameIdx) {{
    const f = DATA.frames[frameIdx - 1];
    const mode = document.getElementById("colorMode")?.value || "density";
    const densMax = Math.max(1e-12, DATA.dens_max || 0);
    const ny = f.density.length;
    const nx = f.density[0].length;
    const img = ctx.createImageData(nx, ny);
    let p = 0;
    if (mode === "phase" && f.phase && DATA.has_phase) {{
      for (let j = 0; j < ny; j++) {{
        for (let i = 0; i < nx; i++) {{
          const c = colorMapPhase(f.phase[j][i]);
          img.data[p++]=c[0]; img.data[p++]=c[1]; img.data[p++]=c[2]; img.data[p++]=c[3];
        }}
      }}
    }} else {{
      for (let j = 0; j < ny; j++) {{
        for (let i = 0; i < nx; i++) {{
          const c = colorMapDensity(f.density[j][i] / densMax);
          img.data[p++]=c[0]; img.data[p++]=c[1]; img.data[p++]=c[2]; img.data[p++]=c[3];
        }}
      }}
    }}
    const tmp = document.createElement("canvas");
    tmp.width = nx; tmp.height = ny;
    tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(tmp, 0, 0, cv.width, cv.height);

    // Center-of-mass marker
    const sx = x => (x - DATA.x_min) / (DATA.x_max - DATA.x_min) * cv.width;
    const sy = y => cv.height - (y - DATA.y_min) / (DATA.y_max - DATA.y_min) * cv.height;
    ctx.beginPath();
    ctx.arc(sx(f.exp_x), sy(f.exp_y), 6, 0, Math.PI*2);
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();

    // Axis tick labels
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.font = "11px sans-serif";
    ctx.fillText(DATA.x_min.toFixed(1), 3, cv.height-4);
    ctx.fillText(DATA.x_max.toFixed(1), cv.width-28, cv.height-4);
    ctx.save(); ctx.translate(12, cv.height/2); ctx.rotate(-Math.PI/2);
    ctx.fillText(`y:[${DATA.y_min.toFixed(1)},${DATA.y_max.toFixed(1)}]`, -40, 0);
    ctx.restore();

    label.textContent = `frame ${{frameIdx}}/${{DATA.frames.length}}, t=${{f.t.toFixed(3)}}`;
    slider.value = String(frameIdx);
    drawColorbar(mode);
  }}

  const colorModeEl = document.getElementById("colorMode");
  if (colorModeEl) colorModeEl.onchange = () => render(Number(slider.value));

  let timer = null;
  function doPlay() {{
    if (timer) return;
    timer = setInterval(() => {{
      let f = Number(slider.value);
      f = f >= DATA.frames.length ? 1 : f + 1;
      render(f);
    }}, 90);
  }}
  function doPause() {{
    if (!timer) return;
    clearInterval(timer);
    timer = null;
  }}
  document.getElementById("play").onclick = doPlay;
  document.getElementById("pause").onclick = doPause;
  slider.oninput = () => render(Number(slider.value));
  render(1);
  </script>
</body>
</html>"""


def _spatial_wavepacket_2d_dirac_frames(
    samples_t: int,
    nx: int,
    ny: int,
    potential_type: str,
    mass: float = 1.0,
    source: str = "python-reference",
) -> dict:
  """Build a 2D (2+1D) Dirac spinor reference solution using split-step Fourier.

  Two-component spinor: (ψ₀ up, ψ₁ down).  Algorithm matches continuous2d_backend.cpp
  evolve_dirac(): V/2 → T_Dirac(k-space) → V/2.
  """
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("Dirac 2D visualization requires numpy") from ex

  x_min, x_max = -8.0, 8.0
  y_min, y_max = -8.0, 8.0
  t_min, t_max = 0.0, 3.0
  nx = max(32, nx)
  ny = max(32, ny)
  nt = max(16, samples_t)

  xs = np.linspace(x_min, x_max, nx)
  ys = np.linspace(y_min, y_max, ny)
  dx = float(xs[1] - xs[0])
  dy = float(ys[1] - ys[0])
  dt = (t_max - t_min) / max(1, nt - 1)

  X, Y = np.meshgrid(xs, ys, indexing="xy")  # shape: (ny, nx)

  # Initial state: Gaussian wavepacket in ψ₀, zero in ψ₁
  x0, y0, sigma = -4.5, -2.0, 0.7
  kx0, ky0 = 2.2, 1.1
  envelope = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (4.0 * sigma ** 2))
  phase_init = np.exp(1j * (kx0 * X + ky0 * Y))
  psi0 = (envelope * phase_init).astype(complex)
  psi1 = np.zeros_like(psi0)

  nrm = float(np.sum(np.abs(psi0) ** 2) * dx * dy)
  if nrm > 0:
    psi0 /= math.sqrt(nrm)

  # Potential (same profiles as _spatial_wavepacket_2d_frames)
  if potential_type == "free":
    V = np.zeros_like(X)
  elif potential_type == "harmonic":
    V = 0.5 * (0.28 ** 2) * (X * X + Y * Y)
  elif potential_type == "well":
    V = np.where((np.abs(X) <= 6.0) & (np.abs(Y) <= 6.0), 0.0, 5.0)
  else:  # barrier
    V = 0.9 * np.exp(-(X * X) / (2.0 * 0.6 ** 2))

  potential_half = np.exp(-1j * V * (dt * 0.5))

  # Kinetic step: precompute Dirac unitary matrices in k-space
  kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
  ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
  KX, KY = np.meshgrid(kx, ky, indexing="xy")  # shape: (ny, nx)
  omega = np.sqrt(KX ** 2 + KY ** 2 + mass ** 2)
  cos_wdt = np.cos(omega * dt)
  sin_wdt_ov_w = np.where(omega > 1e-14, np.sin(omega * dt) / omega, dt * np.ones_like(omega))
  # U₀₀ = cos - i·sin/ω·m,  U₀₁ = -sin/ω·(ky + i·kx)
  # U₁₀ = -sin/ω·(-ky + i·kx), U₁₁ = cos + i·sin/ω·m
  # Matches continuous2d_backend.cpp apply_full_kinetic_dirac_2d() matrix elements
  s = sin_wdt_ov_w
  u00 = cos_wdt - 1j * s * mass
  u01 = -s * KY - 1j * s * KX
  u10 = s * KY - 1j * s * KX
  u11 = cos_wdt + 1j * s * mass

  def apply_kinetic_dirac(p0: np.ndarray, p1: np.ndarray):
    p0k = np.fft.fft2(p0)
    p1k = np.fft.fft2(p1)
    return np.fft.ifft2(u00 * p0k + u01 * p1k), np.fft.ifft2(u10 * p0k + u11 * p1k)

  frames = []
  dens_max = 1e-12

  for it in range(nt):
    t = t_min + (t_max - t_min) * it / max(1, nt - 1)
    dens0 = np.abs(psi0) ** 2
    dens1 = np.abs(psi1) ** 2
    density = dens0 + dens1
    dens_max = max(dens_max, float(np.max(density)))
    exp_x = float(np.sum(X * density) * dx * dy)
    exp_y = float(np.sum(Y * density) * dx * dy)
    frames.append({
        "t": t,
        "density": density.tolist(),
        "phase": np.angle(psi0).tolist(),
        "dirac_s0": dens0.tolist(),
        "dirac_s1": dens1.tolist(),
        "spin_x": (2.0 * np.real(np.conj(psi0) * psi1)).tolist(),
        "spin_y": (2.0 * np.imag(np.conj(psi0) * psi1)).tolist(),
        "spin_z": (dens0 - dens1).tolist(),
        "exp_x": exp_x,
        "exp_y": exp_y,
    })
    if it == nt - 1:
      break

    # Strang splitting: V/2 → T_Dirac → V/2
    psi0 = potential_half * psi0
    psi1 = potential_half * psi1
    psi0, psi1 = apply_kinetic_dirac(psi0, psi1)
    psi0 = potential_half * psi0
    psi1 = potential_half * psi1
    # Renormalise total spinor
    nrm = float(np.sum(np.abs(psi0) ** 2 + np.abs(psi1) ** 2) * dx * dy)
    if nrm > 0:
      inv_sqrt = 1.0 / math.sqrt(nrm)
      psi0 *= inv_sqrt
      psi1 *= inv_sqrt

  return {
      "x_min": x_min,
      "x_max": x_max,
      "y_min": y_min,
      "y_max": y_max,
      "potential_type": potential_type,
      "mass": mass,
      "frames": frames,
      "dens_max": dens_max,
      "has_phase": True,
      "has_dirac": True,
      "source": source,
  }


def _driven_wavepacket_2d_page(data: dict) -> str:
  """Render a driven-potential 2D animation with a driving field overlay panel."""
  payload = json.dumps(data, ensure_ascii=False)
  drv = data.get("driving", {})
  drv_type = drv.get("type", "sine")
  drv_info = f"type={drv_type}, A={drv.get('amplitude',0):.3g}, ω={drv.get('frequency',1):.3g}"
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>QuantX Animation - Driven 2D ({data.get("potential_type","barrier")})</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; background: #f4f6f8; }}
    h1 {{ font-size: 18px; margin-bottom: 4px; }}
    .layout {{ display: flex; gap: 14px; align-items: flex-start; max-width: 880px; }}
    .colorbar-col {{ display: flex; flex-direction: column; align-items: center; gap: 2px; margin-top: 4px; }}
    canvas#colorbar {{ border: 1px solid #aaa; display: block; cursor: default; }}
    .cb-labels {{ display: flex; flex-direction: column; justify-content: space-between;
                  height: 400px; font-size: 11px; color: #555; margin-left: 2px; }}
    .main-col {{ flex: 1; }}
    .row-ctrl {{ display: flex; gap: 10px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 10px; flex-wrap: wrap; }}
    button {{ padding: 4px 10px; }}
    .driving-panel {{ margin-top: 10px; background: #fff; border: 1px solid #ccc; border-radius: 8px;
                      padding: 8px; max-width: 540px; }}
    .driving-panel h3 {{ margin: 0 0 4px; font-size: 13px; color: #333; }}
    canvas#drv_chart {{ display: block; border: 1px solid #bbb; background: #fafafa; }}
    .meta {{ margin-top: 8px; font-size: 12px; color: #444; }}
  </style>
</head>
<body>
  <h1>QuantX Animation – Driven 2D Wavepacket ({data.get("potential_type","barrier")})</h1>
  <p style="font-size:13px;color:#555">Driving field: <b>{drv_info}</b>. Source: <b>{data.get("source","python-reference")}</b>.</p>
  <div class="controls">
    <button id="play">Play</button>
    <button id="pause">Pause</button>
    <input id="slider" type="range" min="1" step="1" style="width:380px"/>
    <span id="label" style="font-size:13px"></span>
    <select id="colorMode" style="font-size:12px;padding:3px 6px">
      <option value="density">Density</option>
      <option value="phase">Phase HSL</option>
    </select>
  </div>
  <div class="layout" style="margin-top:10px">
    <div class="main-col">
      <canvas id="main_canvas" width="540" height="540" style="display:block;border:1px solid #bbb"></canvas>
    </div>
    <div class="colorbar-col">
      <canvas id="colorbar" width="28" height="400"></canvas>
      <div class="cb-labels" id="cbLabels">
        <span id="cb_top"></span><span id="cb_mid"></span><span id="cb_bot"></span>
      </div>
    </div>
  </div>
  <div class="driving-panel">
    <h3>Driving field V_drive(t) = A·envelope(t)·spatial(r)</h3>
    <canvas id="drv_chart" width="540" height="80"></canvas>
  </div>
  <div id="meta" class="meta"></div>
  <script>
  const DATA = {payload};
  const slider = document.getElementById("slider");
  const label  = document.getElementById("label");
  const meta   = document.getElementById("meta");
  const colorModeCtl = document.getElementById("colorMode");
  slider.max = String(DATA.frames.length);
  slider.value = "1";

  function colorMapDensity(v) {{
    const t = Math.max(0, Math.min(1, v));
    return [Math.floor(255*Math.sqrt(t)), Math.floor(255*t*t), Math.floor(255*(1-t)), 255];
  }}
  function colorMapPhase(phi) {{
    const h = ((phi + Math.PI) / (2*Math.PI)) * 360;
    const s = 0.88, l = 0.52, a = s * Math.min(l, 1-l);
    const f = n => Math.round(255*(l - a*Math.max(-1, Math.min((n+h/30)%12-3, 9-(n+h/30)%12, 1))));
    return [f(0), f(8), f(4), 255];
  }}

  function drawColorbar(mode) {{
    const cv = document.getElementById("colorbar");
    const ctx = cv.getContext("2d");
    const h = cv.height;
    for (let y = 0; y < h; y++) {{
      const t = 1 - y / (h - 1);
      let c;
      if (mode === "phase") {{
        c = colorMapPhase((t * 2 - 1) * Math.PI);
      }} else {{
        c = colorMapDensity(t);
      }}
      ctx.fillStyle = `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;
      ctx.fillRect(0, y, cv.width, 1);
    }}
    if (mode === "phase") {{
      document.getElementById("cb_top").textContent = "+π";
      document.getElementById("cb_mid").textContent = "0";
      document.getElementById("cb_bot").textContent = "−π";
    }} else {{
      const mx = DATA.dens_max || 1;
      document.getElementById("cb_top").textContent = mx.toExponential(1);
      document.getElementById("cb_mid").textContent = (mx/2).toExponential(1);
      document.getElementById("cb_bot").textContent = "0";
    }}
  }}

  function drawFrame(fi) {{
    const f = DATA.frames[fi];
    const mode = colorModeCtl.value;
    const cv = document.getElementById("main_canvas");
    const ctx = cv.getContext("2d");
    const densArr = f.density, phaseArr = f.phase;
    const ny = densArr.length, nx = densArr[0].length;
    const dmax = DATA.dens_max || 1;
    // Render at grid resolution into an offscreen canvas, then scale to fill the viewport.
    const off = document.createElement("canvas");
    off.width = nx; off.height = ny;
    const offCtx = off.getContext("2d");
    const img = offCtx.createImageData(nx, ny);
    let p = 0;
    for (let j = 0; j < ny; j++) {{
      for (let i = 0; i < nx; i++) {{
        const c = mode === "phase" ? colorMapPhase(phaseArr[j][i]) : colorMapDensity(densArr[j][i] / dmax);
        img.data[p++]=c[0]; img.data[p++]=c[1]; img.data[p++]=c[2]; img.data[p++]=c[3];
      }}
    }}
    offCtx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, cv.width, cv.height);
    label.textContent = `t = ${{f.t.toFixed(3)}}  ⟨x⟩ = ${{f.exp_x.toFixed(2)}}  ⟨y⟩ = ${{f.exp_y.toFixed(2)}}`;
    if (f.driving_amplitude !== undefined) {{
      meta.textContent = `Drive A(t) = ${{f.driving_amplitude.toExponential(3)}}`;
    }}
    drawDrivingChart(fi);
  }}

  function drawDrivingChart(currentFi) {{
    const cv = document.getElementById("drv_chart");
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const amps = DATA.frames.map(f => f.driving_amplitude !== undefined ? f.driving_amplitude : 0);
    const maxA = Math.max(...amps.map(Math.abs), 1e-12);
    const n = amps.length;
    // Axis
    ctx.strokeStyle = "#ccc"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, H/2); ctx.lineTo(W, H/2); ctx.stroke();
    // Driving curve
    ctx.strokeStyle = "#1565c0"; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {{
      const x = (i / (n-1)) * W;
      const y = H/2 - (amps[i] / maxA) * (H/2 - 4);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }}
    ctx.stroke();
    // Current time cursor
    const cx = (currentFi / (n-1)) * W;
    ctx.strokeStyle = "#e53935"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
    // Labels
    ctx.fillStyle = "#888"; ctx.font = "10px sans-serif";
    ctx.fillText(`t=${{DATA.frames[0].t.toFixed(1)}}`, 2, H-2);
    ctx.fillText(`t=${{DATA.frames[n-1].t.toFixed(1)}}`, W-28, H-2);
    ctx.fillText(`+${{maxA.toExponential(2)}}`, 2, 11);
    ctx.fillText(`-${{maxA.toExponential(2)}}`, 2, H-12);
  }}

  function render(n) {{
    const fi = Math.max(0, Math.min(n - 1, parseInt(slider.value, 10) - 1));
    drawFrame(fi);
    drawColorbar(colorModeCtl.value);
  }}

  slider.addEventListener("input", () => render(DATA.frames.length));
  colorModeCtl.addEventListener("change", () => render(DATA.frames.length));

  let playing = false, timer = null;
  document.getElementById("play").onclick = () => {{
    if (playing) return;
    playing = true;
    timer = setInterval(() => {{
      let v = parseInt(slider.value, 10) + 1;
      if (v > DATA.frames.length) v = 1;
      slider.value = String(v);
      render(DATA.frames.length);
    }}, 80);
  }};
  document.getElementById("pause").onclick = () => {{ playing = false; clearInterval(timer); }};

  render(DATA.frames.length);
  </script>
</body>
</html>"""


def _spatial_wavepacket_3d_frames(
    samples_t: int,
    nx: int,
    ny: int,
    nz: int,
    potential_type: str,
    boundary: str,
    source: str = "python-reference",
) -> dict:
  """Build a 3D reference solution for the spatial wavepacket animation."""
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("3D spatial visualization requires numpy") from ex

  x_min, x_max = -6.0, 6.0
  y_min, y_max = -6.0, 6.0
  z_min, z_max = -6.0, 6.0
  t_min, t_max = 0.0, 2.5
  nx = max(16, nx)
  ny = max(16, ny)
  nz = max(16, nz)
  nt = max(10, samples_t)

  xs = np.linspace(x_min, x_max, nx)
  ys = np.linspace(y_min, y_max, ny)
  zs = np.linspace(z_min, z_max, nz)
  dx = float(xs[1] - xs[0])
  dy = float(ys[1] - ys[0])
  dz = float(zs[1] - zs[0])
  dt = (t_max - t_min) / (nt - 1)

  X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

  x0, y0, z0 = -2.0, -1.2, -0.8
  sigma = 0.7
  kx0, ky0, kz0 = 1.8, 1.0, 0.8
  envelope = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2) / (4.0 * sigma * sigma))
  phase = np.exp(1j * (kx0 * X + ky0 * Y + kz0 * Z))
  psi = envelope * phase

  if potential_type == "free":
    V = np.zeros_like(X)
  elif potential_type == "harmonic":
    omega = 0.25
    V = 0.5 * (omega * omega) * (X * X + Y * Y + Z * Z)
  elif potential_type == "well":
    wall = 5.5
    V = np.where((np.abs(X) <= 4.8) & (np.abs(Y) <= 4.8) & (np.abs(Z) <= 4.8), 0.0, wall)
  else:  # barrier
    V = 0.95 * np.exp(-(X * X) / (2.0 * 0.55 * 0.55))

  kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
  ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
  kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
  KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
  kinetic_phase = np.exp(-1j * 0.5 * (KX * KX + KY * KY + KZ * KZ) * dt)
  potential_half = np.exp(-1j * V * dt * 0.5)

  absorb_mask = np.ones_like(V, dtype=float)
  if boundary == "absorbing":
    width = 4.0
    strength = 4.0
    ix = np.minimum(np.arange(nx), np.arange(nx)[::-1]).reshape(nx, 1, 1)
    iy = np.minimum(np.arange(ny), np.arange(ny)[::-1]).reshape(1, ny, 1)
    iz = np.minimum(np.arange(nz), np.arange(nz)[::-1]).reshape(1, 1, nz)
    edge = np.minimum(np.minimum(ix, iy), iz).astype(float)
    t = np.clip((width - edge) / width, 0.0, 1.0)
    absorb_mask = np.exp(-strength * t * t * dt)

  def normalize(a):
    nrm = float(np.sum(np.abs(a) ** 2) * dx * dy * dz)
    if nrm > 0:
      a /= math.sqrt(nrm)
    return a

  def apply_dirichlet(a):
    a[0, :, :] = 0
    a[-1, :, :] = 0
    a[:, 0, :] = 0
    a[:, -1, :] = 0
    a[:, :, 0] = 0
    a[:, :, -1] = 0
    return a

  psi = normalize(psi)
  if boundary == "dirichlet":
    psi = apply_dirichlet(psi)
    psi = normalize(psi)

  frames = []
  dens_max = 1e-12
  z_mid = nz // 2
  y_mid = ny // 2
  x_mid = nx // 2
  points_max_per_frame = 1400
  points_density_floor_ratio = 0.08

  def sample_points(dens: np.ndarray) -> tuple[list[list[float]], float]:
    vmax = float(np.max(dens))
    if vmax <= 0.0:
      return [], 0.0

    floor = vmax * points_density_floor_ratio
    coords = np.argwhere(dens >= floor)
    if coords.shape[0] == 0:
      coords = np.argwhere(dens > 0.0)
    if coords.shape[0] == 0:
      return [], vmax

    values = dens[coords[:, 0], coords[:, 1], coords[:, 2]]
    if coords.shape[0] > points_max_per_frame:
      keep_idx = np.argpartition(values, -points_max_per_frame)[-points_max_per_frame:]
      coords = coords[keep_idx]
      values = values[keep_idx]

    points = []
    for n in range(coords.shape[0]):
      ix, iy, iz = coords[n]
      points.append(
          [
              round(float(xs[ix]), 4),
              round(float(ys[iy]), 4),
              round(float(zs[iz]), 4),
              round(float(values[n]), 8),
          ]
      )
    return points, vmax

  for it in range(nt):
    t = t_min + (t_max - t_min) * it / max(1, nt - 1)
    dens = np.abs(psi) ** 2
    points, point_vmax = sample_points(dens)
    dens_max = max(dens_max, float(np.max(dens)))
    norm = float(np.sum(dens) * dx * dy * dz)
    exp_x = float(np.sum(X * dens) * dx * dy * dz)
    exp_y = float(np.sum(Y * dens) * dx * dy * dz)
    exp_z = float(np.sum(Z * dens) * dx * dy * dz)
    psi_phase = np.angle(psi)
    frame_entry: dict = {
            "t": t,
            "xy": dens[:, :, z_mid].tolist(),
            "xz": dens[:, y_mid, :].tolist(),
            "yz": dens[x_mid, :, :].tolist(),
            "phase_xy": psi_phase[:, :, z_mid].tolist(),
            "phase_xz": psi_phase[:, y_mid, :].tolist(),
            "phase_yz": psi_phase[x_mid, :, :].tolist(),
            "exp_x": exp_x,
            "exp_y": exp_y,
            "exp_z": exp_z,
            "norm": norm,
            "points": points,
            "point_vmax": point_vmax,
            "point_count": len(points),
    }
    if nx * ny * nz <= 131072:
      frame_entry["density_xyz"] = dens.ravel(order="C").tolist()
    frames.append(frame_entry)
    if it == nt - 1:
      break

    psi = potential_half * psi
    psi_k = np.fft.fftn(psi)
    psi = np.fft.ifftn(kinetic_phase * psi_k)
    psi = potential_half * psi
    if boundary == "absorbing":
      psi = psi * absorb_mask
    if boundary == "dirichlet":
      psi = apply_dirichlet(psi)
    if boundary != "absorbing":
      psi = normalize(psi)

  return {
      "x_min": x_min,
      "x_max": x_max,
      "y_min": y_min,
      "y_max": y_max,
      "z_min": z_min,
      "z_max": z_max,
      "nx": nx,
      "ny": ny,
      "nz": nz,
      "potential_type": potential_type,
      "boundary": boundary,
      "frames": frames,
      "dens_max": dens_max,
      "has_phase": True,
      "has_volume_grid": nx * ny * nz <= 131072,
      "volume_point_fields": ["x", "y", "z", "density"],
      "volume_points_max_per_frame": points_max_per_frame,
      "volume_points_density_floor_ratio": points_density_floor_ratio,
      "source": source,
  }


def _spatial_wavepacket_3d_page(data: dict) -> str:
  """Render the 3D slice-based spatial wavepacket animation page with phase toggle and colorbar."""
  payload = json.dumps(data, ensure_ascii=False)
  has_phase = bool(data.get("has_phase"))
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>QuantX Animation - 3D Spatial Wavepacket ({data.get("potential_type","barrier")}, {data.get("boundary","periodic")})</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; background: #f8f8f8; }}
    h1 {{ font-size: 18px; margin-bottom: 4px; }}
    .row {{ display: flex; gap: 10px; align-items: flex-start; margin-top: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; }}
    .panel {{ border: 1px solid #ccc; border-radius: 8px; padding: 8px; background: #fff; }}
    .panel h3 {{ margin: 0 0 6px; font-size: 13px; }}
    canvas.slice {{ width: 100%; aspect-ratio: 1/1; border: 1px solid #bbb; display: block; }}
    canvas#colorbar {{ border: 1px solid #aaa; display: block; }}
    .cb-labels {{ display: flex; flex-direction: column; justify-content: space-between; font-size: 11px; color: #555; height: 280px; padding: 2px 0; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 8px; flex-wrap: wrap; }}
    button {{ padding: 4px 10px; }}
    select {{ padding: 3px 6px; }}
    .meta {{ margin-top: 8px; font-size: 12px; color: #444; }}
  </style>
</head>
<body>
  <h1>QuantX Animation - 3D Spatial Wavepacket</h1>
  <p style="font-size:13px;color:#555">Split-step Fourier. Orthogonal midplane slices of |ψ|². Source: <b>{data.get("source","python-reference")}</b>.</p>
  <div class="controls">
    <button id="play">Play</button>
    <button id="pause">Pause</button>
    <input id="slider" type="range" min="1" step="1" style="width:400px"/>
    <span id="label" style="font-size:13px"></span>
    {"<label>Color: <select id='colorMode'><option value='density'>Density |ψ|²</option><option value='phase'>Phase arg(ψ)</option></select></label>" if has_phase else ""}
  </div>
  <div class="row">
    <div class="grid">
      <div class="panel"><h3>XY (z-mid)</h3><canvas class="slice" id="cv_xy" width="240" height="240"></canvas></div>
      <div class="panel"><h3>XZ (y-mid)</h3><canvas class="slice" id="cv_xz" width="240" height="240"></canvas></div>
      <div class="panel"><h3>YZ (x-mid)</h3><canvas class="slice" id="cv_yz" width="240" height="240"></canvas></div>
    </div>
    <div>
      <canvas id="colorbar" width="26" height="280"></canvas>
    </div>
    <div class="cb-labels" id="cbLabels">
      <span>max</span><span></span><span>0</span>
    </div>
  </div>
  <div id="meta" class="meta"></div>
  <script>
  const DATA = {payload};
  const slider = document.getElementById("slider");
  const label = document.getElementById("label");
  const meta = document.getElementById("meta");
  const cbCanvas = document.getElementById("colorbar");
  const cbCtx = cbCanvas.getContext("2d");
  const cbLabels = document.getElementById("cbLabels");
  const ctxXY = document.getElementById("cv_xy").getContext("2d");
  const ctxXZ = document.getElementById("cv_xz").getContext("2d");
  const ctxYZ = document.getElementById("cv_yz").getContext("2d");
  slider.max = String(DATA.frames.length);
  slider.value = "1";

  function colorMapDensity(v) {{
    const t = Math.max(0, Math.min(1, v));
    return [Math.floor(255*Math.pow(t,0.55)), Math.floor(255*t*t), Math.floor(255*(1-t)), 255];
  }}
  function colorMapPhase(phi) {{
    const h = ((phi + Math.PI) / (2*Math.PI)) * 360;
    const s = 0.88, l = 0.52, a = s * Math.min(l, 1-l);
    const f = n => Math.round(255*(l - a*Math.max(-1, Math.min((n+h/30)%12-3, 9-(n+h/30)%12, 1))));
    return [f(0), f(8), f(4), 255];
  }}

  function drawColorbar(mode) {{
    const h = cbCanvas.height;
    for (let y = 0; y < h; y++) {{
      const t = 1 - y/h;
      const rgba = (mode === "phase") ? colorMapPhase((t*2-1)*Math.PI) : colorMapDensity(t);
      cbCtx.fillStyle = `rgb(${{rgba[0]}},${{rgba[1]}},${{rgba[2]}})`;
      cbCtx.fillRect(0, y, cbCanvas.width, 1);
    }}
    if (mode === "phase") {{
      cbLabels.innerHTML = "<span>+π</span><span>0</span><span>−π</span>";
    }} else {{
      cbLabels.innerHTML = `<span>${{(DATA.dens_max||0).toExponential(1)}}</span><span></span><span>0</span>`;
    }}
  }}

  function drawSlice(ctx, densArr, phaseArr, mode) {{
    const rows = densArr.length;
    const cols = densArr[0].length;
    const img = ctx.createImageData(cols, rows);
    const densMax = Math.max(1e-12, DATA.dens_max || 0);
    let p = 0;
    if (mode === "phase" && phaseArr && DATA.has_phase) {{
      for (let j = 0; j < rows; j++) {{
        for (let i = 0; i < cols; i++) {{
          const c = colorMapPhase(phaseArr[j][i]);
          img.data[p++]=c[0]; img.data[p++]=c[1]; img.data[p++]=c[2]; img.data[p++]=c[3];
        }}
      }}
    }} else {{
      for (let j = 0; j < rows; j++) {{
        for (let i = 0; i < cols; i++) {{
          const c = colorMapDensity(densArr[j][i] / densMax);
          img.data[p++]=c[0]; img.data[p++]=c[1]; img.data[p++]=c[2]; img.data[p++]=c[3];
        }}
      }}
    }}
    const tmp = document.createElement("canvas");
    tmp.width = cols; tmp.height = rows;
    tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(tmp, 0, 0, ctx.canvas.width, ctx.canvas.height);
  }}

  function render(frameIdx) {{
    const f = DATA.frames[frameIdx - 1];
    const mode = document.getElementById("colorMode")?.value || "density";
    drawSlice(ctxXY, f.xy, f.phase_xy, mode);
    drawSlice(ctxXZ, f.xz, f.phase_xz, mode);
    drawSlice(ctxYZ, f.yz, f.phase_yz, mode);
    label.textContent = `frame ${{frameIdx}}/${{DATA.frames.length}}, t=${{f.t.toFixed(3)}}`;
    meta.textContent = `boundary=${{DATA.boundary}}, potential=${{DATA.potential_type}}, norm=${{(f.norm||0).toFixed(6)}}, <x,y,z>=(${{(f.exp_x||0).toFixed(3)}}, ${{(f.exp_y||0).toFixed(3)}}, ${{(f.exp_z||0).toFixed(3)}})`;
    slider.value = String(frameIdx);
    drawColorbar(mode);
  }}

  const colorModeEl = document.getElementById("colorMode");
  if (colorModeEl) colorModeEl.onchange = () => render(Number(slider.value));

  let timer = null;
  function doPlay() {{
    if (timer) return;
    timer = setInterval(() => {{
      let f = Number(slider.value);
      f = f >= DATA.frames.length ? 1 : f + 1;
      render(f);
    }}, 90);
  }}
  function doPause() {{
    if (!timer) return;
    clearInterval(timer);
    timer = null;
  }}
  document.getElementById("play").onclick = doPlay;
  document.getElementById("pause").onclick = doPause;
  slider.oninput = () => render(Number(slider.value));
  render(1);
  </script>
</body>
</html>"""


def _dirac_spinor_page(data: dict) -> str:
  """Render the Dirac spinor 2D visualization: 4-component panels + spin current arrows."""
  payload = json.dumps(data, ensure_ascii=False)
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>QuantX Animation - 2D Dirac Spinor ({data.get("potential_type","barrier")})</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; background: #f4f6f8; }}
    h1 {{ font-size: 18px; margin-bottom: 4px; }}
    .grid2x2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; max-width: 1100px; }}
    .panel {{ background: #fff; border: 1px solid #ccc; border-radius: 8px; padding: 8px; }}
    .panel h3 {{ margin: 0 0 5px; font-size: 13px; color: #333; }}
    .cv-wrap {{ position: relative; display: inline-block; }}
    canvas.hm {{ display: block; border: 1px solid #bbb; }}
    svg.arrows {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
    .row-ctrl {{ display: flex; gap: 10px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }}
    .colorbar-row {{ display: flex; gap: 6px; align-items: center; margin-top: 4px; font-size: 11px; color: #555; }}
    canvas.cb {{ border: 1px solid #aaa; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 10px; flex-wrap: wrap; }}
    button {{ padding: 4px 10px; }}
    select {{ padding: 3px 6px; font-size: 12px; }}
    .meta {{ margin-top: 8px; font-size: 12px; color: #444; }}
  </style>
</head>
<body>
  <h1>QuantX Animation - 2D Dirac Spinor ({data.get("potential_type","barrier")})</h1>
  <p style="font-size:13px;color:#555">Split-step Fourier Dirac solver (m={data.get("mass",1.0):.2f}). Source: <b>{data.get("source","python-reference")}</b>.</p>
  <div class="controls">
    <button id="play">Play</button>
    <button id="pause">Pause</button>
    <input id="slider" type="range" min="1" step="1" style="width:380px"/>
    <span id="label" style="font-size:13px"></span>
  </div>
  <div class="grid2x2" style="margin-top:10px">
    <div class="panel">
      <h3>ψ₀ density |ψ₀|²  (up spinor)</h3>
      <div class="cv-wrap"><canvas class="hm" id="cv_s0" width="340" height="340"></canvas></div>
      <div class="colorbar-row"><canvas class="cb" id="cb_s0" width="200" height="12"></canvas><span id="cbl_s0"></span></div>
    </div>
    <div class="panel">
      <h3>ψ₁ density |ψ₁|²  (down spinor)</h3>
      <div class="cv-wrap"><canvas class="hm" id="cv_s1" width="340" height="340"></canvas></div>
      <div class="colorbar-row"><canvas class="cb" id="cb_s1" width="200" height="12"></canvas><span id="cbl_s1"></span></div>
    </div>
    <div class="panel">
      <h3>Total density + phase arg(ψ₀)  <select id="totalMode"><option value="density">Density</option><option value="phase">Phase HSL</option></select></h3>
      <div class="cv-wrap"><canvas class="hm" id="cv_tot" width="340" height="340"></canvas></div>
      <div class="colorbar-row"><canvas class="cb" id="cb_tot" width="200" height="12"></canvas><span id="cbl_tot"></span></div>
    </div>
    <div class="panel">
      <h3>Spin-z density + current arrows</h3>
      <div class="cv-wrap">
        <canvas class="hm" id="cv_spin" width="340" height="340"></canvas>
        <svg class="arrows" id="svg_arrows" width="340" height="340"></svg>
      </div>
      <div class="colorbar-row"><canvas class="cb" id="cb_spin" width="200" height="12"></canvas><span id="cbl_spin"></span></div>
      <div class="colorbar-row"><span style="color:#2166ac">■ spin↓</span>&nbsp;&nbsp;<span style="color:#d6604d">■ spin↑</span>&nbsp;arrows = spin current (J<sub>x</sub>,J<sub>y</sub>)</div>
    </div>
  </div>
  <div id="meta" class="meta"></div>
  <script>
  const DATA = {payload};
  const slider = document.getElementById("slider");
  const label = document.getElementById("label");
  const meta = document.getElementById("meta");
  slider.max = String(DATA.frames.length);
  slider.value = "1";

  function colorMapDensity(v) {{
    const t = Math.max(0, Math.min(1, v));
    return [Math.floor(255*Math.sqrt(t)), Math.floor(255*t*t), Math.floor(255*(1-t)), 255];
  }}
  function colorMapPhase(phi) {{
    const h = ((phi + Math.PI) / (2*Math.PI)) * 360;
    const s = 0.88, l = 0.52, a = s * Math.min(l, 1-l);
    const f = n => Math.round(255*(l - a*Math.max(-1, Math.min((n+h/30)%12-3, 9-(n+h/30)%12, 1))));
    return [f(0), f(8), f(4), 255];
  }}
  // Blue (-1) to white (0) to red (+1) for spin-z
  function colorMapSpinZ(v) {{
    const t = Math.max(-1, Math.min(1, v));
    if (t < 0) {{
      const s = -t;
      return [Math.round(255*(1-s*0.66)), Math.round(255*(1-s*0.58)), 255, 255];
    }}
    return [255, Math.round(255*(1-t*0.62)), Math.round(255*(1-t*0.69)), 255];
  }}

  function drawHeatmap(canvasId, arr2d, colorFn) {{
    const cv = document.getElementById(canvasId);
    const ctx = cv.getContext("2d");
    const ny = arr2d.length, nx = arr2d[0].length;
    const img = ctx.createImageData(nx, ny);
    let p = 0;
    for (let j = 0; j < ny; j++) {{
      for (let i = 0; i < nx; i++) {{
        const c = colorFn(arr2d[j][i]);
        img.data[p++]=c[0]; img.data[p++]=c[1]; img.data[p++]=c[2]; img.data[p++]=c[3];
      }}
    }}
    const tmp = document.createElement("canvas");
    tmp.width = nx; tmp.height = ny;
    tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(tmp, 0, 0, cv.width, cv.height);
  }}

  function drawColorbarH(canvasId, labelId, mode, maxVal) {{
    const cv = document.getElementById(canvasId);
    const ctx = cv.getContext("2d");
    const w = cv.width;
    for (let x = 0; x < w; x++) {{
      const t = x / w;
      let rgba;
      if (mode === "phase") rgba = colorMapPhase((t*2-1)*Math.PI);
      else if (mode === "spinz") rgba = colorMapSpinZ(t*2-1);
      else rgba = colorMapDensity(t);
      ctx.fillStyle = `rgb(${{rgba[0]}},${{rgba[1]}},${{rgba[2]}})`;
      ctx.fillRect(x, 0, 1, cv.height);
    }}
    const lbl = document.getElementById(labelId);
    if (mode === "phase") lbl.textContent = " −π → +π";
    else if (mode === "spinz") lbl.textContent = " −1 → 0 → +1";
    else lbl.textContent = ` 0 → ${{(maxVal||0).toExponential(2)}}`;
  }}

  function drawSpinArrows(f) {{
    const svg = document.getElementById("svg_arrows");
    const W = 340, H = 340;
    svg.setAttribute("width", W); svg.setAttribute("height", H);
    svg.innerHTML = "";
    const ny = f.spin_x.length, nx = f.spin_x[0].length;
    const step = Math.max(4, Math.floor(Math.min(nx, ny) / 18));
    const scaleX = W / nx, scaleY = H / ny;
    const arrowLen = Math.min(scaleX, scaleY) * step * 0.7;
    for (let j = step; j < ny - step; j += step) {{
      for (let i = step; i < nx - step; i += step) {{
        const jx = f.spin_x[j][i], jy = f.spin_y[j][i], sz = f.spin_z[j][i];
        const mag = Math.sqrt(jx*jx + jy*jy);
        if (mag < 1e-10) continue;
        const px = (i + 0.5) * scaleX, py = (j + 0.5) * scaleY;
        const dx = (jx/mag) * arrowLen * Math.min(1, mag*3);
        const dy = (jy/mag) * arrowLen * Math.min(1, mag*3);
        const col = sz > 0 ? `hsl(0,80%,${{40+Math.abs(sz)*20|0}}%)` : `hsl(220,80%,${{40+Math.abs(sz)*20|0}}%)`;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", px); line.setAttribute("y1", py);
        line.setAttribute("x2", px+dx); line.setAttribute("y2", py+dy);
        line.setAttribute("stroke", col);
        line.setAttribute("stroke-width", "1.5");
        line.setAttribute("stroke-linecap", "round");
        svg.appendChild(line);
      }}
    }}
  }}

  function render(frameIdx) {{
    const f = DATA.frames[frameIdx - 1];
    const densMax = Math.max(1e-12, DATA.dens_max||0);
    const totalMode = document.getElementById("totalMode").value;

    // Panel 1: |ψ₀|²
    drawHeatmap("cv_s0", f.dirac_s0, v => colorMapDensity(v/densMax));
    drawColorbarH("cb_s0", "cbl_s0", "density", densMax);

    // Panel 2: |ψ₁|²
    drawHeatmap("cv_s1", f.dirac_s1, v => colorMapDensity(v/densMax));
    drawColorbarH("cb_s1", "cbl_s1", "density", densMax);

    // Panel 3: total density or phase
    if (totalMode === "phase") {{
      drawHeatmap("cv_tot", f.phase, colorMapPhase);
      drawColorbarH("cb_tot", "cbl_tot", "phase", 0);
    }} else {{
      drawHeatmap("cv_tot", f.density, v => colorMapDensity(v/densMax));
      drawColorbarH("cb_tot", "cbl_tot", "density", densMax);
    }}

    // Panel 4: spin-z + arrows
    drawHeatmap("cv_spin", f.spin_z, colorMapSpinZ);
    drawSpinArrows(f);
    drawColorbarH("cb_spin", "cbl_spin", "spinz", 0);

    label.textContent = `frame ${{frameIdx}}/${{DATA.frames.length}}, t=${{f.t.toFixed(3)}}`;
    meta.textContent = `potential=${{DATA.potential_type}}, mass=${{DATA.mass||1}}, <x,y>=(${{f.exp_x.toFixed(3)}},${{f.exp_y.toFixed(3)}})`;
    slider.value = String(frameIdx);
  }}

  document.getElementById("totalMode").onchange = () => render(Number(slider.value));
  let timer = null;
  function doPlay() {{
    if (timer) return;
    timer = setInterval(() => {{
      let f = Number(slider.value);
      f = f >= DATA.frames.length ? 1 : f + 1;
      render(f);
    }}, 90);
  }}
  function doPause() {{ if (!timer) return; clearInterval(timer); timer = null; }}
  document.getElementById("play").onclick = doPlay;
  document.getElementById("pause").onclick = doPause;
  slider.oninput = () => render(Number(slider.value));
  render(1);
  </script>
</body>
</html>"""


def _lindblad_frames(
    n_qubits: int,
    hamiltonian_terms: list[dict],
    collapse_ops: list[dict],
    total_time: float = 2.0,
    dt: float = 0.01,
    samples: int = 80,
    source: str = "python-reference",
) -> dict:
    """Simulate Lindblad master equation and return animation frame data.

    hamiltonian_terms: list of {"label": "X0Z1", "coefficient": 1.0}
    collapse_ops: list of {"label": "AmplitudeDamping0", "coefficient": gamma}
    Returns a dict with keys: n_qubits, frames, source, total_time
    """
    try:
      import numpy as np
    except ImportError as ex:
      raise RuntimeError("Lindblad visualization requires numpy") from ex

    dim = 1 << n_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0

    # Build Hamiltonian matrix
    H = np.zeros((dim, dim), dtype=complex)
    import re
    token_re = re.compile(r'([XYZxyz])(\d+)')

    def _pauli_x():
        return np.array([[0, 1], [1, 0]], dtype=complex)
    def _pauli_y():
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    def _pauli_z():
        return np.array([[1, 0], [0, -1]], dtype=complex)
    def _sigma_minus():
        return np.array([[0, 1], [0, 0]], dtype=complex)

    def _embed(op2x2, q, n):
        """Embed 2x2 op at qubit q (qubit 0 = LSB) in 2^n space."""
        full_dim = 1 << n
        q_mask = 1 << q
        result = np.zeros((full_dim, full_dim), dtype=complex)
        for col in range(full_dim):
            c_q = (col >> q) & 1
            col_rest = col & ~q_mask
            for row in range(full_dim):
                if (row & ~q_mask) == col_rest:
                    r_q = (row >> q) & 1
                    result[row, col] = op2x2[r_q, c_q]
        return result

    def _build_H(terms, n):
        full_dim = 1 << n
        H = np.zeros((full_dim, full_dim), dtype=complex)
        for t in terms:
            label = t.get("label", "")
            coeff = t.get("coefficient", 0.0)
            if any(label.startswith(p) for p in ("AmplitudeDamping", "Dephasing", "Depolarizing")):
                continue
            if not label:
                continue
            flip_mask = 0
            yz_mask = 0
            y_count = 0
            valid = True
            for m in token_re.finditer(label):
                p = m.group(1).upper()
                q = int(m.group(2))
                if q >= n or q >= 63:
                    valid = False; break
                bit = 1 << q
                if p == 'X':
                    flip_mask |= bit
                elif p == 'Y':
                    flip_mask |= bit; yz_mask |= bit; y_count += 1
                elif p == 'Z':
                    yz_mask |= bit
                else:
                    valid = False; break
            if not valid:
                continue
            phases = [1+0j, 0+1j, -1+0j, 0-1j]
            global_phase = phases[y_count % 4]
            for src in range(full_dim):
                dst = src ^ flip_mask
                sign_bits = bin(src & yz_mask).count('1')
                sign = -1.0 if sign_bits % 2 else 1.0
                H[dst, src] += coeff * global_phase * sign
        return H

    def _build_collapse_ops(ops_spec, n):
        Lk = []
        for op in ops_spec:
            label = op.get("label", "")
            gamma = max(0.0, op.get("coefficient", 0.0))
            sqrt_gamma = math.sqrt(gamma)
            if label.startswith("AmplitudeDamping"):
                q = int(label[len("AmplitudeDamping"):])
                if q < n:
                    L = sqrt_gamma * _embed(_sigma_minus(), q, n)
                    Lk.append(L)
            elif label.startswith("Dephasing"):
                q = int(label[len("Dephasing"):])
                if q < n:
                    L = (sqrt_gamma * 0.5) * _embed(_pauli_z(), q, n)
                    Lk.append(L)
            elif label.startswith("Depolarizing"):
                q = int(label[len("Depolarizing"):])
                if q < n:
                    sqrt_g3 = math.sqrt(gamma / 3.0) if gamma > 0 else 0.0
                    for op2x2 in [_pauli_x(), _pauli_y(), _pauli_z()]:
                        Lk.append(sqrt_g3 * _embed(op2x2, q, n))
        return Lk

    H_mat = _build_H(hamiltonian_terms + collapse_ops, n_qubits)
    Lk_list = _build_collapse_ops(hamiltonian_terms + collapse_ops, n_qubits)
    LdagL_list = [L.conj().T @ L for L in Lk_list]

    def lindblad_rhs(rho):
        drho = -1j * (H_mat @ rho - rho @ H_mat)
        for L, LdagL in zip(Lk_list, LdagL_list):
            drho += L @ rho @ L.conj().T - 0.5 * (LdagL @ rho + rho @ LdagL)
        return drho

    def rk4_step(rho, dt):
        k1 = lindblad_rhs(rho)
        k2 = lindblad_rhs(rho + 0.5 * dt * k1)
        k3 = lindblad_rhs(rho + 0.5 * dt * k2)
        k4 = lindblad_rhs(rho + dt * k3)
        rho_new = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return 0.5 * (rho_new + rho_new.conj().T)  # symmetrise

    def record_frame(rho, step, t):
        pops = [max(0.0, rho[i, i].real) for i in range(dim)]
        purity = max(0.0, (rho @ rho).trace().real)
        # entropy from eigenvalues
        evals = np.maximum(0.0, np.linalg.eigvalsh(rho))
        entropy = float(-np.sum(evals * np.log(np.where(evals > 1e-15, evals, 1.0)) * (evals > 1e-15)))
        off_diag = [abs(rho[i, j]) for i in range(dim) for j in range(dim) if i != j]
        coherence = max(off_diag) if off_diag else 0.0
        return {
            "step": step,
            "t": round(t, 10),
            "purity": round(purity, 10),
            "entropy": round(entropy, 10),
            "populations": [round(p, 10) for p in pops],
            "coherence": round(coherence, 10),
        }

    # Compute step stride for output frames
    total_steps = max(1, round(total_time / dt))
    stride = max(1, total_steps // max(1, samples - 1))

    frames = [record_frame(rho, 0, 0.0)]
    for s in range(total_steps):
        rho = rk4_step(rho, dt)
        t = (s + 1) * dt
        if (s + 1) % stride == 0 or s + 1 == total_steps:
            frames.append(record_frame(rho, s + 1, t))

    return {
        "n_qubits": n_qubits,
        "dim": dim,
        "source": source,
        "total_time": total_time,
        "frames": frames,
        "hamiltonian_terms": hamiltonian_terms,
        "collapse_ops": collapse_ops,
    }


def _lindblad_page(data: dict) -> str:
    """Render an animated HTML page for Lindblad density-matrix evolution."""
    data_json = json.dumps(data, ensure_ascii=False)
    n_qubits = data.get("n_qubits", 1)
    dim = data.get("dim", 1 << n_qubits)
    state_labels = [f"|{i:0{n_qubits}b}\u27e9" for i in range(dim)]
    labels_json = json.dumps(state_labels)

    # Generate a distinct color for each basis state
    colors = ["#0b84f3", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#34495e"]
    colors_json = json.dumps([colors[i % len(colors)] for i in range(dim)])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>QuantX \u2014 Lindblad Open Quantum Systems</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; background: #0f1117; color: #e0e0e0;
            margin: 0; padding: 16px; }}
    h1 {{ font-size: 1.4em; margin-bottom: 4px; color: #90caf9; }}
    .subtitle {{ color: #888; font-size: 0.9em; margin-bottom: 16px; }}
    .panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; max-width: 1100px; }}
    .panel {{ background: #1a1d27; border-radius: 8px; padding: 12px; }}
    .panel h2 {{ font-size: 1em; margin: 0 0 8px; color: #b0c4de; }}
    canvas {{ width: 100%; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin: 12px 0; }}
    button {{ padding: 4px 14px; background: #263040; border: 1px solid #4a90d9;
             color: #e0e0e0; border-radius: 4px; cursor: pointer; }}
    button:hover {{ background: #2d4060; }}
    input[type=range] {{ flex: 1; accent-color: #4a90d9; }}
    .info {{ font-size: 0.82em; color: #888; margin-top: 4px; }}
    .legend {{ font-size: 0.82em; margin-top: 6px; }}
    .legend span {{ margin-right: 12px; }}
  </style>
</head>
<body>
  <h1>Lindblad \u2014 Open Quantum System Evolution</h1>
  <p class="subtitle">source: {data.get("source","python-reference")} &nbsp;|&nbsp;
     qubits: {n_qubits} &nbsp;|&nbsp;
     dim: {dim}\xd7{dim} density matrix</p>
  <div class="controls">
    <button id="play">\u25b6 Play</button>
    <button id="pause">\u23f8 Pause</button>
    <input type="range" id="slider" min="1" max="1" value="1"/>
    <span id="label" class="info">frame 1</span>
  </div>
  <div class="panels">
    <div class="panel">
      <h2>Populations \u03c1\u1d62\u1d62</h2>
      <canvas id="cv_pop" height="240"></canvas>
      <div id="leg_pop" class="legend"></div>
    </div>
    <div class="panel">
      <h2>Purity Tr(\u03c1\u00b2) &amp; Coherence max|\u03c1\u1d62\u2c7c|</h2>
      <canvas id="cv_pur" height="240"></canvas>
    </div>
    <div class="panel">
      <h2>Von Neumann Entropy \u2212Tr(\u03c1 log \u03c1)</h2>
      <canvas id="cv_ent" height="200"></canvas>
    </div>
    <div class="panel">
      <h2>Density Matrix \u03c1 (real part at current frame)</h2>
      <canvas id="cv_rho" height="200"></canvas>
    </div>
  </div>
  <script>
  const DATA = {data_json};
  const STATE_LABELS = {labels_json};
  const COLORS = {colors_json};
  const DIM = DATA.dim;
  const frames = DATA.frames;

  const slider = document.getElementById("slider");
  slider.max = String(frames.length);

  // Build population legend
  const legPop = document.getElementById("leg_pop");
  COLORS.forEach((c, i) => {{
    const sp = document.createElement("span");
    sp.innerHTML = `<span style="color:${{c}}">&#9632;</span> ${{STATE_LABELS[i]}}`;
    legPop.appendChild(sp);
  }});

  // \u2500 Generic line chart helper \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  function drawLineChart(canvasId, series, currentFrame) {{
    const cv = document.getElementById(canvasId);
    const W = cv.offsetWidth || 500, H = cv.height;
    cv.width = W;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#1a1d27";
    ctx.fillRect(0, 0, W, H);

    const L = 48, R = 12, T = 8, B = 32;
    const plotW = W - L - R, plotH = H - T - B;

    // Find ranges
    let yMin = Infinity, yMax = -Infinity;
    for (const s of series) {{
      for (const [, y] of s.points) {{
        yMin = Math.min(yMin, y); yMax = Math.max(yMax, y);
      }}
    }}
    if (!isFinite(yMin)) yMin = 0;
    if (!isFinite(yMax)) yMax = 1;
    if (Math.abs(yMax - yMin) < 1e-10) {{ yMin -= 0.1; yMax += 0.1; }}
    const xMin = frames[0].t, xMax = frames[frames.length-1].t;
    const sx = x => L + (x - xMin) / (xMax - xMin) * plotW;
    const sy = y => T + (1 - (y - yMin) / (yMax - yMin)) * plotH;

    // Axes
    ctx.strokeStyle = "#444";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, T + plotH); ctx.lineTo(L + plotW, T + plotH); ctx.stroke();
    ctx.fillStyle = "#888"; ctx.font = "10px sans-serif"; ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {{
      const v = yMin + (yMax - yMin) * i / 4;
      const py = sy(v);
      ctx.fillText(v.toFixed(2), L - 4, py + 3);
      ctx.strokeStyle = "#2a2a3a"; ctx.beginPath(); ctx.moveTo(L, py); ctx.lineTo(L + plotW, py); ctx.stroke();
    }}
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {{
      const v = xMin + (xMax - xMin) * i / 4;
      ctx.fillText(v.toFixed(2), sx(v), T + plotH + 16);
    }}

    // Series
    for (const s of series) {{
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      s.points.forEach(([x, y], i) => {{
        if (i === 0) ctx.moveTo(sx(x), sy(y));
        else ctx.lineTo(sx(x), sy(y));
      }});
      ctx.stroke();
    }}

    // Current time marker
    const t_cur = frames[currentFrame - 1].t;
    ctx.strokeStyle = "#ffffff44";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(sx(t_cur), T); ctx.lineTo(sx(t_cur), T + plotH); ctx.stroke();
    ctx.setLineDash([]);
  }}

  // \u2500 Density matrix heatmap \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  function drawRhoHeatmap(frameIdx) {{
    const cv = document.getElementById("cv_rho");
    const W = cv.offsetWidth || 300;
    cv.width = W;
    const H = cv.height;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#1a1d27";
    ctx.fillRect(0, 0, W, H);

    const f = frames[frameIdx - 1];
    const pops = f.populations;
    if (!pops) return;
    const cellW = Math.floor(W / DIM), cellH = Math.floor(H / DIM);

    // We only have populations (diagonal). Show diagonal bar chart as heatmap substitute.
    const maxPop = Math.max(...pops, 1e-10);
    for (let i = 0; i < DIM; ++i) {{
      const v = pops[i] / maxPop;
      const g = Math.round(v * 200);
      ctx.fillStyle = `rgb(${{Math.round(v*80)}},${{Math.round(v*160)}},255)`;
      const x = i * cellW + 2;
      const barH = Math.round(v * (H - 20));
      ctx.fillRect(x, H - barH - 10, Math.max(1, cellW - 4), barH);
      ctx.fillStyle = "#888";
      ctx.font = "9px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(STATE_LABELS[i], x + (cellW - 4)/2, H - 2);
    }}
  }}

  function render(frameIdx) {{
    const popSeries = COLORS.map((c, i) => ({{
      color: c,
      points: frames.map(f => [f.t, f.populations[i] || 0]),
    }}));
    drawLineChart("cv_pop", popSeries, frameIdx);
    drawLineChart("cv_pur", [
      {{ color: "#4fc3f7", points: frames.map(f => [f.t, f.purity]) }},
      {{ color: "#ffb74d", points: frames.map(f => [f.t, f.coherence]) }},
    ], frameIdx);
    drawLineChart("cv_ent", [
      {{ color: "#a5d6a7", points: frames.map(f => [f.t, f.entropy]) }},
    ], frameIdx);
    drawRhoHeatmap(frameIdx);

    const f = frames[frameIdx - 1];
    document.getElementById("label").textContent =
      `frame ${{frameIdx}}/${{frames.length}}, t=${{f.t.toFixed(4)}}, ` +
      `purity=${{f.purity.toFixed(4)}}, entropy=${{f.entropy.toFixed(4)}}`;
    slider.value = String(frameIdx);
  }}

  let timer = null;
  function doPlay() {{
    if (timer) return;
    timer = setInterval(() => {{
      let v = Number(slider.value);
      v = v >= frames.length ? 1 : v + 1;
      render(v);
    }}, 60);
  }}
  function doPause() {{ if (!timer) return; clearInterval(timer); timer = null; }}
  document.getElementById("play").onclick = doPlay;
  document.getElementById("pause").onclick = doPause;
  slider.oninput = () => render(Number(slider.value));
  render(1);
  </script>
</body>
</html>"""


def _mps_frames(
    n_sites: int,
    hamiltonian_terms: list[dict],
    total_time: float = 6.0,
    dt: float = 0.05,
    max_bond: int = 32,
    trunc_tol: float = 1e-8,
    samples: int = 60,
    source: str = "python-reference",
) -> dict:
    """Numpy TEBD reference solver for 1D nearest-neighbour Hamiltonians.

    Supports Pauli-string terms with |i-j|=1 (NN two-body) and single-site.
    Returns bond entropies, bond dimensions, and site magnetisation over time.
    """
    try:
        import numpy as np
    except ImportError as ex:
        raise RuntimeError("MPS visualization requires numpy") from ex

    import re
    d = 2  # qubit physical dimension
    tok = re.compile(r'([XYZxyz])(\d+)')

    def _px(): return np.array([[0,1],[1,0]], dtype=complex)
    def _py(): return np.array([[0,-1j],[1j,0]], dtype=complex)
    def _pz(): return np.array([[1,0],[0,-1]], dtype=complex)
    def _pi(): return np.eye(2, dtype=complex)
    def _p(c):
        c = c.upper()
        return _px() if c=='X' else _py() if c=='Y' else _pz() if c=='Z' else _pi()

    def kron2(A, B): return np.kron(A, B)

    # Parse Hamiltonian into NN 2-body and single-site terms.
    two_body: dict[int, np.ndarray] = {}   # bond b → 4×4 Hermitian
    single_body: dict[int, np.ndarray] = {}  # site → 2×2 Hermitian
    for ht in hamiltonian_terms:
        label, coeff = ht.get("label",""), ht.get("coefficient", 0.0)
        ops = [(m.group(1).upper(), int(m.group(2))) for m in tok.finditer(label)]
        if not ops: continue
        if len(ops) == 1:
            s, q = ops[0]
            if 0 <= q < n_sites:
                single_body[q] = single_body.get(q, np.zeros((2,2), complex)) + coeff * _p(s)
        elif len(ops) == 2:
            (s1,q1),(s2,q2) = ops
            if abs(q1-q2) == 1 and 0 <= q1 < n_sites and 0 <= q2 < n_sites:
                b = min(q1,q2)
                op = kron2(_p(s1 if q1<q2 else s2), _p(s2 if q1<q2 else s1))
                two_body[b] = two_body.get(b, np.zeros((4,4), complex)) + coeff * op

    # Build gates for 2nd-order Suzuki-Trotter:
    # even bonds use half-steps at start/end, odd bonds use full step in middle.
    def _expm_small(H):
        vals, vecs = np.linalg.eigh(H)
        return (vecs * np.exp(-1j * vals)) @ vecs.conj().T

    even_half_gates = {b: _expm_small(0.5 * dt * H) for b, H in two_body.items() if b % 2 == 0}
    odd_full_gates = {b: _expm_small(dt * H) for b, H in two_body.items() if b % 2 == 1}
    # Single-site half-step gates for 2nd-order ST (½ dt each end of sweep).
    single_half = {s: _expm_small(0.5 * dt * H) for s, H in single_body.items()}

    # MPS state: list of site tensors A[i] shape (chi_L*d, chi_R).
    tensors = [np.zeros((d, 1), dtype=complex) for _ in range(n_sites)]
    for t in tensors: t[0, 0] = 1.0  # |0⟩
    schmidt = [np.array([1.0], dtype=float) for _ in range(max(0, n_sites - 1))]

    def bond_dim(i): return tensors[i].shape[1]

    def apply_1site(s, U2):
        A = tensors[s]
        chi_L = A.shape[0] // d
        chi_R = A.shape[1]
        for il in range(chi_L):
            for ir in range(chi_R):
                a0, a1 = A[il*d, ir], A[il*d+1, ir]
                A[il*d,   ir] = U2[0,0]*a0 + U2[0,1]*a1
                A[il*d+1, ir] = U2[1,0]*a0 + U2[1,1]*a1

    def apply_2site(b, U4):
        Ai, Aip1 = tensors[b], tensors[b+1]
        chi_L = Ai.shape[0] // d
        chi_M = Ai.shape[1]
        chi_R = Aip1.shape[1]
        # Reshape Aip1: (chi_M*d, chi_R) → (chi_M, d*chi_R)
        Aip1_r = np.zeros((chi_M, d*chi_R), complex)
        for im in range(chi_M):
            for s in range(d):
                Aip1_r[im, s*chi_R:(s+1)*chi_R] = Aip1[im*d+s, :]
        Theta = Ai @ Aip1_r  # (chi_L*d, d*chi_R)
        # Apply gate.
        Tp = np.zeros_like(Theta)
        for il in range(chi_L):
            for ir in range(chi_R):
                for s1n in range(d):
                    for s2n in range(d):
                        v = complex(0)
                        for s1o in range(d):
                            for s2o in range(d):
                                v += U4[s1n*d+s2n, s1o*d+s2o] * Theta[il*d+s1o, s2o*chi_R+ir]
                        Tp[il*d+s1n, s2n*chi_R+ir] = v
        # SVD and truncate.
        U, sv, Vh = np.linalg.svd(Tp, full_matrices=False)
        abs_tol = trunc_tol * sv[0] if sv.size > 0 else 1e-30
        chi_new = max(1, min(max_bond, int(np.sum(sv > abs_tol))))
        schmidt[b] = sv[:chi_new].copy()
        tensors[b]   = U[:, :chi_new]
        SV = np.diag(sv[:chi_new]) @ Vh[:chi_new, :]  # (chi_new, d*chi_R)
        Aip1_new = np.zeros((chi_new*d, chi_R), complex)
        for im in range(chi_new):
            for s in range(d):
                Aip1_new[im*d+s, :] = SV[im, s*chi_R:(s+1)*chi_R]
        tensors[b+1] = Aip1_new

    def tebd_step():
        for s, U in single_half.items(): apply_1site(s, U)
        for b in range(0, n_sites-1, 2):
            if b in even_half_gates: apply_2site(b, even_half_gates[b])
        for b in range(1, n_sites-1, 2):
            if b in odd_full_gates: apply_2site(b, odd_full_gates[b])
        for b in range(0, n_sites-1, 2):
            if b in even_half_gates: apply_2site(b, even_half_gates[b])
        for s, U in single_half.items(): apply_1site(s, U)

    def bond_entropy(b):
        sv = schmidt[b]
        norm2 = np.sum(sv**2)
        if norm2 < 1e-30: return 0.0
        p = sv**2 / norm2
        p = p[p > 1e-15]
        return float(-np.sum(p * np.log(p)))

    def magnetization_z(s):
        A = tensors[s]
        chi_L = A.shape[0] // d
        chi_R = A.shape[1]
        val = 0.0
        for il in range(chi_L):
            for ir in range(chi_R):
                val += abs(A[il*d, ir])**2 - abs(A[il*d+1, ir])**2
        return float(val)

    def collect():
        bds = [int(bond_dim(i)) for i in range(n_sites-1)]
        bes = [round(bond_entropy(b), 10) for b in range(n_sites-1)]
        mag = [round(magnetization_z(s), 10) for s in range(n_sites)]
        return {"bond_dims": bds, "bond_entropies": bes, "magnetization": mag,
                "max_bond_dim": max(bds) if bds else 1}

    total_steps = max(1, round(total_time / dt))
    stride = max(1, total_steps // max(1, samples - 1))

    frames = [{"step": 0, "t": 0.0, **collect()}]
    for s in range(total_steps):
        tebd_step()
        if (s+1) % stride == 0 or s+1 == total_steps:
            frames.append({"step": s+1, "t": round((s+1)*dt, 10), **collect()})

    return {
        "n_sites": n_sites, "max_bond": max_bond, "total_time": total_time,
        "source": source, "frames": frames,
        "hamiltonian_terms": hamiltonian_terms,
    }


def _mps_page(data: dict) -> str:
    """Render an animated HTML page for MPS/TEBD bond entropies and magnetization."""
    data_json = json.dumps(data, ensure_ascii=False)
    n_sites = data.get("n_sites", 4)
    title = f"MPS TEBD — {n_sites}-site chain"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif;padding:20px}}
h1{{font-size:1.1rem;font-weight:600;margin-bottom:12px;color:#58a6ff}}
.meta{{font-size:0.78rem;color:#8b949e;margin-bottom:16px}}
.controls{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}}
button{{background:#21262d;border:1px solid #30363d;color:#e6edf3;
        padding:6px 14px;border-radius:6px;cursor:pointer;font-size:0.82rem}}
button:hover{{background:#30363d}}
input[type=range]{{width:220px;accent-color:#58a6ff}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:980px}}
.panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}}
.panel h2{{font-size:0.85rem;font-weight:600;color:#8b949e;margin-bottom:10px}}
canvas{{display:block}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">TEBD 2nd-order Suzuki-Trotter &nbsp;|&nbsp; {n_sites} sites</div>
<div class="controls">
  <button id="btnPlay">&#9654; Play</button>
  <label style="font-size:.8rem;color:#8b949e">Step: <input type="range" id="slider" min="1" max="1" value="1"></label>
  <span id="tLabel" style="font-size:.8rem;color:#58a6ff"></span>
</div>
<div class="grid">
  <div class="panel">
    <h2>Bond Entanglement Entropy S(bond j)</h2>
    <canvas id="cvEntropy" width="440" height="200"></canvas>
  </div>
  <div class="panel">
    <h2>Site Magnetisation ⟨Z_i⟩</h2>
    <canvas id="cvMag" width="440" height="200"></canvas>
  </div>
  <div class="panel">
    <h2>Bond Entropy vs Time (heatmap)</h2>
    <canvas id="cvHeatEntropy" width="440" height="200"></canvas>
  </div>
  <div class="panel">
    <h2>Magnetisation vs Time (heatmap)</h2>
    <canvas id="cvHeatMag" width="440" height="200"></canvas>
  </div>
</div>
<script>
const DATA = {data_json};
const frames = DATA.frames;
const N = frames.length;
const nSites = DATA.n_sites;
const nBonds = nSites - 1;

const slider = document.getElementById("slider");
slider.max = N;

// ── Colour helpers ──────────────────────────────────────────────────────────
function lerp(a,b,t){{return a+(b-a)*t;}}
function heatColour(v, vMin, vMax, palette){{
  const t=vMax>vMin?(v-vMin)/(vMax-vMin):0.5;
  const ti=Math.min(palette.length-2,Math.floor(t*(palette.length-1)));
  const tf=t*(palette.length-1)-ti;
  const c0=palette[ti],c1=palette[ti+1];
  return `rgb(${{Math.round(lerp(c0[0],c1[0],tf))}},${{Math.round(lerp(c0[1],c1[1],tf))}},${{Math.round(lerp(c0[2],c1[2],tf))}})`;
}}
const HOT  =[[13,17,23],[30,80,160],[88,166,255],[63,185,80],[255,220,80],[255,120,40],[255,40,40]];
const COOL =[[13,17,23],[30,30,120],[60,60,200],[80,150,220],[200,200,255]];
const RWB  =[[0,80,200],[255,255,255],[200,0,0]];

// ── Static heatmaps ─────────────────────────────────────────────────────────
function drawHeatmap(canvasId, nCols, seriesKey, vMin, vMax, palette){{
  const cv=document.getElementById(canvasId);
  const W=cv.width, H=cv.height;
  const ctx=cv.getContext("2d");
  ctx.clearRect(0,0,W,H);
  const cellW=W/N, cellH=H/nCols;
  frames.forEach((f,fi)=>{{
    const vals=f[seriesKey]||[];
    for(let j=0;j<nCols;j++){{
      ctx.fillStyle=heatColour(vals[j]??0,vMin,vMax,palette);
      ctx.fillRect(fi*cellW,j*cellH,Math.ceil(cellW),Math.ceil(cellH));
    }}
  }});
}}

const maxEnt=Math.max(...frames.flatMap(f=>f.bond_entropies||[0]),0.01);
const maxMag=1.0;
drawHeatmap("cvHeatEntropy", nBonds, "bond_entropies", 0, maxEnt, HOT);
drawHeatmap("cvHeatMag",    nSites, "magnetization",  -1, 1,      RWB);

// ── Bar charts ──────────────────────────────────────────────────────────────
function drawBars(canvasId, vals, vMin, vMax, colour, label){{
  const cv=document.getElementById(canvasId);
  const W=cv.width,H=cv.height;
  const ctx=cv.getContext("2d");
  const pad={{l:36,r:8,t:8,b:28}};
  const w=W-pad.l-pad.r, h=H-pad.t-pad.b;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle="#30363d"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+h); ctx.lineTo(pad.l+w,pad.t+h); ctx.stroke();
  const range=vMax-vMin||1;
  ctx.fillStyle="#8b949e"; ctx.font="9px system-ui"; ctx.textAlign="right";
  for(let i=0;i<=4;i++){{
    const v=vMin+range*i/4;
    const y=pad.t+h-(v-vMin)/range*h;
    ctx.fillText(v.toFixed(2),pad.l-2,y+3);
    ctx.strokeStyle="#21262d"; ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+w,y); ctx.stroke();
  }}
  if(!vals||!vals.length)return;
  const bw=w/vals.length*0.8;
  const gap=w/vals.length;
  vals.forEach((v,i)=>{{
    const x=pad.l+i*gap+(gap-bw)/2;
    const y0=pad.t+h-(0-vMin)/range*h;
    const y=pad.t+h-(v-vMin)/range*h;
    ctx.fillStyle=colour;
    ctx.fillRect(x,Math.min(y,y0),bw,Math.abs(y-y0)+0.5);
  }});
  ctx.fillStyle=colour; ctx.font="9px system-ui"; ctx.textAlign="left";
  ctx.fillText(label, pad.l+2, pad.t+12);
}}

// Cursor overlay on heatmap.
function drawCursor(canvasId, fi){{
  const cv=document.getElementById(canvasId);
  const ctx=cv.getContext("2d");
  const x=cv.width*fi/N;
  ctx.strokeStyle="rgba(255,255,255,0.5)"; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,cv.height); ctx.stroke();
  ctx.setLineDash([]);
}}

function render(idx){{
  const f=frames[idx-1];
  document.getElementById("tLabel").textContent=
    `t=${{f.t?.toFixed(3)??0}}  χ_max=${{f.max_bond_dim??1}}`;
  drawBars("cvEntropy", f.bond_entropies||[], 0, maxEnt, "#58a6ff", "S (nats)");
  drawBars("cvMag",     f.magnetization||[], -1, 1,      "#f0883e", "⟨Z_i⟩");
  // Redraw heatmaps and cursor.
  drawHeatmap("cvHeatEntropy", nBonds, "bond_entropies", 0, maxEnt, HOT);
  drawHeatmap("cvHeatMag",    nSites, "magnetization",  -1, 1,      RWB);
  drawCursor("cvHeatEntropy", idx-1);
  drawCursor("cvHeatMag",     idx-1);
}}
render(1);
slider.addEventListener("input",()=>render(+slider.value));

let timer=null;
document.getElementById("btnPlay").addEventListener("click",function(){{
  if(timer){{clearInterval(timer);timer=null;this.textContent="▶ Play";return;}}
  this.textContent="⏸ Pause";
  let idx=+slider.value;
  timer=setInterval(()=>{{
    idx=idx>=N?1:idx+1; slider.value=idx; render(idx);
    if(idx>=N){{clearInterval(timer);timer=null;document.getElementById("btnPlay").textContent="▶ Play";}}
  }},100);
}});
</script>
</body>
</html>"""


def _entanglement_frames(
    n_qubits: int,
    hamiltonian_terms: list[dict],
    collapse_ops: list[dict],
    n_a: int,
    total_time: float = 4.0,
    dt: float = 0.02,
    samples: int = 80,
    source: str = "python-reference",
) -> dict:
    """Compute bipartite entanglement measures via numpy.

    For pure-state time evolution (no collapse ops), uses exact matrix
    exponentiation and Schmidt decomposition via SVD.  For mixed-state runs
    (collapse_ops non-empty) uses Lindblad RK4 then partial-trace + SVD.

    Returns a dict with keys: n_qubits, n_a, n_b, frames, source, total_time.
    Each frame: step, t, entropy_a, mutual_info, concurrence, log_negativity,
                schmidt_values.
    """
    try:
      import numpy as np
    except ImportError as ex:
      raise RuntimeError("Entanglement visualization requires numpy") from ex

    import re
    token_re = re.compile(r'([XYZxyz])(\d+)')

    dim = 1 << n_qubits
    n_b = n_qubits - n_a
    da = 1 << n_a
    db = 1 << n_b

    # ── Hamiltonian builder (replicates _lindblad_frames logic) ─────────────
    def _pauli_x():
        return np.array([[0, 1], [1, 0]], dtype=complex)
    def _pauli_y():
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    def _pauli_z():
        return np.array([[1, 0], [0, -1]], dtype=complex)
    def _sigma_minus():
        return np.array([[0, 1], [0, 0]], dtype=complex)

    def _embed(op2x2, q, n):
        full_dim = 1 << n
        q_mask = 1 << q
        result = np.zeros((full_dim, full_dim), dtype=complex)
        for col in range(full_dim):
            c_q = (col >> q) & 1
            col_rest = col & ~q_mask
            for row in range(full_dim):
                if (row & ~q_mask) == col_rest:
                    r_q = (row >> q) & 1
                    result[row, col] = op2x2[r_q, c_q]
        return result

    def _build_H(terms, n):
        H = np.zeros((1 << n, 1 << n), dtype=complex)
        for t in terms:
            label = t.get("label", "")
            coeff = t.get("coefficient", 0.0)
            if any(label.startswith(p) for p in ("AmplitudeDamping", "Dephasing", "Depolarizing")):
                continue
            if not label:
                continue
            flip_mask = yz_mask = y_count = 0
            valid = True
            for m in token_re.finditer(label):
                p = m.group(1).upper()
                q = int(m.group(2))
                if q >= n or q >= 63:
                    valid = False; break
                bit = 1 << q
                if p == 'X':
                    flip_mask |= bit
                elif p == 'Y':
                    flip_mask |= bit; yz_mask |= bit; y_count += 1
                elif p == 'Z':
                    yz_mask |= bit
                else:
                    valid = False; break
            if not valid:
                continue
            global_phase = [1+0j, 0+1j, -1+0j, 0-1j][y_count % 4]
            for src in range(1 << n):
                dst = src ^ flip_mask
                sign = -1.0 if bin(src & yz_mask).count('1') % 2 else 1.0
                H[dst, src] += coeff * global_phase * sign
        return H

    def _build_collapse_ops(ops_spec, n):
        Lk = []
        for op in ops_spec:
            label = op.get("label", "")
            gamma = max(0.0, op.get("coefficient", 0.0))
            sg = math.sqrt(gamma)
            if label.startswith("AmplitudeDamping"):
                q = int(label[len("AmplitudeDamping"):])
                if q < n:
                    Lk.append(sg * _embed(_sigma_minus(), q, n))
            elif label.startswith("Dephasing"):
                q = int(label[len("Dephasing"):])
                if q < n:
                    Lk.append((sg * 0.5) * _embed(_pauli_z(), q, n))
            elif label.startswith("Depolarizing"):
                q = int(label[len("Depolarizing"):])
                if q < n:
                    sg3 = math.sqrt(gamma / 3.0) if gamma > 0 else 0.0
                    for op2x2 in [_pauli_x(), _pauli_y(), _pauli_z()]:
                        Lk.append(sg3 * _embed(op2x2, q, n))
        return Lk

    all_terms = hamiltonian_terms + collapse_ops
    H_mat = _build_H(all_terms, n_qubits)
    Lk_list = _build_collapse_ops(all_terms, n_qubits)

    # ── Entanglement measures ────────────────────────────────────────────────
    def _entropy(rho_sub):
        evals = np.maximum(0.0, np.linalg.eigvalsh(rho_sub))
        return float(-np.sum(evals * np.log(np.where(evals > 1e-15, evals, 1.0)) * (evals > 1e-15)))

    def _partial_trace_b(rho):
        """Trace over B: ρ_A[ia, ia'] = Σ_ib ρ[ia + ib*da, ia' + ib*da]."""
        rho_a = np.zeros((da, da), dtype=complex)
        for ib in range(db):
            rho_a += rho[ib*da:(ib+1)*da, ib*da:(ib+1)*da]
        return rho_a

    def _partial_trace_a(rho):
        """Trace over A: ρ_B[ib, ib'] = Σ_ia ρ[ia + ib*da, ia + ib'*da]."""
        rho_b = np.zeros((db, db), dtype=complex)
        for ia in range(da):
            rows = np.arange(db) * da + ia
            rho_b += rho[np.ix_(rows, rows)]
        return rho_b

    def _partial_transpose_a(rho):
        """Partial transpose w.r.t. A."""
        out = np.zeros((dim, dim), dtype=complex)
        for ib in range(db):
            for ibp in range(db):
                block = rho[ib*da:(ib+1)*da, ibp*da:(ibp+1)*da]
                out[ib*da:(ib+1)*da, ibp*da:(ibp+1)*da] = block.T
        return out

    def _wootters_concurrence(rho):
        """Wootters concurrence for 2-qubit density matrix."""
        sy_sy = np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]], dtype=complex)
        rho_tilde = sy_sy @ rho.conj() @ sy_sy
        R = rho @ rho_tilde
        evals = np.linalg.eigvals(R)
        lambdas = np.sort(np.sqrt(np.maximum(0.0, evals.real)))[::-1]
        return float(max(0.0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]))

    def record_frame_pure(psi, step, t):
        """Compute entanglement from pure statevector via SVD."""
        M = psi.reshape(db, da)  # psi[ib*da + ia] → M[ib, ia]
        sv = np.linalg.svd(M, compute_uv=False)  # descending order
        probs = sv**2
        probs = probs[probs > 1e-15]
        entropy_a = float(-np.sum(probs * np.log(probs)))
        mutual_info = 2.0 * entropy_a  # pure state: S_A = S_B
        concurrence = float(2.0 * sv[0] * sv[1]) if (n_a == 1 and n_b == 1 and len(sv) >= 2) else -1.0
        sv_sum = float(np.sum(sv))
        log_neg = float(np.log2(max(1.0, sv_sum)))
        return {
            "step": step, "t": round(t, 10),
            "entropy_a": round(entropy_a, 10),
            "mutual_info": round(mutual_info, 10),
            "concurrence": round(concurrence, 10),
            "log_negativity": round(log_neg, 10),
            "schmidt_values": [round(float(s), 10) for s in sv],
        }

    def record_frame_mixed(rho, step, t):
        """Compute entanglement from density matrix via partial trace."""
        rho_a = _partial_trace_b(rho)
        rho_b = _partial_trace_a(rho)
        entropy_a = _entropy(rho_a)
        entropy_b = _entropy(rho_b)
        entropy_total = _entropy(rho)
        mutual_info = entropy_a + entropy_b - entropy_total
        rho_pt = _partial_transpose_a(rho)
        pt_evals = np.linalg.eigvalsh(rho_pt)
        trace_norm = float(np.sum(np.abs(pt_evals)))
        log_neg = float(np.log2(max(1.0, trace_norm)))
        sv_a = np.sqrt(np.maximum(0.0, np.linalg.eigvalsh(rho_a)))[::-1]
        concurrence = _wootters_concurrence(rho) if n_qubits == 2 else -1.0
        return {
            "step": step, "t": round(t, 10),
            "entropy_a": round(entropy_a, 10),
            "mutual_info": round(mutual_info, 10),
            "concurrence": round(float(concurrence), 10),
            "log_negativity": round(log_neg, 10),
            "schmidt_values": [round(float(s), 10) for s in sv_a],
        }

    total_steps = max(1, round(total_time / dt))
    stride = max(1, total_steps // max(1, samples - 1))

    is_mixed = len(Lk_list) > 0

    if is_mixed:
        # Mixed: Lindblad RK4
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0
        LdagL_list = [L.conj().T @ L for L in Lk_list]

        def lindblad_rhs(r):
            dr = -1j * (H_mat @ r - r @ H_mat)
            for L, LdagL in zip(Lk_list, LdagL_list):
                dr += L @ r @ L.conj().T - 0.5 * (LdagL @ r + r @ LdagL)
            return dr

        def rk4_step(r):
            k1 = lindblad_rhs(r)
            k2 = lindblad_rhs(r + 0.5 * dt * k1)
            k3 = lindblad_rhs(r + 0.5 * dt * k2)
            k4 = lindblad_rhs(r + dt * k3)
            r_new = r + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            return 0.5 * (r_new + r_new.conj().T)

        frames = [record_frame_mixed(rho, 0, 0.0)]
        for s in range(total_steps):
            rho = rk4_step(rho)
            t_s = (s + 1) * dt
            if (s + 1) % stride == 0 or s + 1 == total_steps:
                frames.append(record_frame_mixed(rho, s + 1, t_s))
    else:
        # Pure: exact matrix exponential
        evals, evecs = np.linalg.eigh(H_mat)
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1.0

        frames = [record_frame_pure(psi, 0, 0.0)]
        U_dt = evecs @ np.diag(np.exp(-1j * evals * dt)) @ evecs.conj().T
        for s in range(total_steps):
            psi = U_dt @ psi
            t_s = (s + 1) * dt
            if (s + 1) % stride == 0 or s + 1 == total_steps:
                frames.append(record_frame_pure(psi, s + 1, t_s))

    return {
        "n_qubits": n_qubits,
        "n_a": n_a,
        "n_b": n_b,
        "is_mixed": is_mixed,
        "source": source,
        "total_time": total_time,
        "frames": frames,
        "hamiltonian_terms": hamiltonian_terms,
        "collapse_ops": collapse_ops,
    }


def _entanglement_page(data: dict) -> str:
    """Render an animated HTML page for bipartite entanglement measures."""
    data_json = json.dumps(data, ensure_ascii=False)
    n_a = data.get("n_a", 1)
    n_b = data.get("n_b", 1)
    n_qubits = data.get("n_qubits", n_a + n_b)
    is_mixed = data.get("is_mixed", False)
    title = f"Entanglement: {n_qubits}-qubit ({n_a}|{n_b} cut)"
    mixed_label = "Mixed (Lindblad)" if is_mixed else "Pure (Schrödinger)"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif;padding:20px}}
h1{{font-size:1.1rem;font-weight:600;margin-bottom:12px;color:#58a6ff}}
.meta{{font-size:0.78rem;color:#8b949e;margin-bottom:16px}}
.controls{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}}
.controls label{{font-size:0.8rem;color:#8b949e}}
button{{background:#21262d;border:1px solid #30363d;color:#e6edf3;
        padding:6px 14px;border-radius:6px;cursor:pointer;font-size:0.82rem}}
button:hover{{background:#30363d}}
input[type=range]{{width:200px;accent-color:#58a6ff}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:960px}}
.panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}}
.panel h2{{font-size:0.85rem;font-weight:600;color:#8b949e;margin-bottom:10px}}
canvas{{display:block;width:100%;height:200px}}
.tick{{font-size:9px;fill:#8b949e}}
.schmidt-bar{{display:flex;gap:3px;align-items:flex-end;height:80px;margin-top:8px}}
.sbar{{background:#58a6ff;border-radius:2px 2px 0 0;min-width:8px;transition:height 0.15s}}
.info{{font-size:0.75rem;color:#8b949e;margin-top:8px}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">Type: {mixed_label} &nbsp;|&nbsp; Subsystem A: qubits [0,{n_a}) &nbsp;|&nbsp; Subsystem B: qubits [{n_a},{n_qubits})</div>
<div class="controls">
  <button id="btnPlay">&#9654; Play</button>
  <label>Step: <input type="range" id="slider" min="1" max="1" value="1"></label>
  <span id="timeLabel" style="font-size:0.8rem;color:#58a6ff"></span>
</div>
<div class="grid">
  <div class="panel">
    <h2>S(ρ_A) — Entanglement Entropy</h2>
    <canvas id="cvEntropy"></canvas>
  </div>
  <div class="panel">
    <h2>Mutual Information I(A:B)</h2>
    <canvas id="cvMutual"></canvas>
  </div>
  <div class="panel">
    <h2>Concurrence{"" if n_qubits == 2 else " (N/A — not 2-qubit)"} &amp; Log-Negativity</h2>
    <canvas id="cvConc"></canvas>
  </div>
  <div class="panel">
    <h2>Schmidt Spectrum</h2>
    <div class="schmidt-bar" id="schmidtBars"></div>
    <div class="info" id="schmidtInfo"></div>
  </div>
</div>
<script>
const DATA = {data_json};
const frames = DATA.frames;
const N = frames.length;

const slider = document.getElementById("slider");
const timeLabel = document.getElementById("timeLabel");
slider.max = N;

// ── Canvas chart helper ──────────────────────────────────────────────────────
function initCanvas(id) {{
  const cv = document.getElementById(id);
  cv.width = cv.offsetWidth || 440;
  cv.height = 200;
  return cv.getContext("2d");
}}

function drawChart(ctx, series, color, label, yMin, yMax) {{
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const pad = {{l:40,r:12,t:10,b:30}};
  const w = W - pad.l - pad.r, h = H - pad.t - pad.b;
  ctx.clearRect(0, 0, W, H);
  // axes
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+h); ctx.lineTo(pad.l+w,pad.t+h); ctx.stroke();
  // tick labels
  ctx.fillStyle = "#8b949e"; ctx.font = "9px system-ui"; ctx.textAlign = "right";
  const range = yMax - yMin || 1;
  for (let i = 0; i <= 4; i++) {{
    const v = yMin + (range * i / 4);
    const y = pad.t + h - (v - yMin) / range * h;
    ctx.fillText(v.toFixed(2), pad.l - 3, y + 3);
    ctx.strokeStyle = "#21262d"; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l+w, y); ctx.stroke();
  }}
  // line
  if (series.length < 2) return;
  ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.beginPath();
  series.forEach((v, i) => {{
    const x = pad.l + i / (series.length - 1) * w;
    const y = pad.t + h - (v - yMin) / range * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.stroke();
  // label
  ctx.fillStyle = color; ctx.font = "9px system-ui"; ctx.textAlign = "left";
  ctx.fillText(label, pad.l + 2, pad.t + 12);
}}

const entropy_series = frames.map(f => f.entropy_a);
const mutual_series = frames.map(f => f.mutual_info);
const conc_series = frames.map(f => f.concurrence);
const lneg_series = frames.map(f => f.log_negativity);

const e_max = Math.max(...entropy_series, 0.01);
const m_max = Math.max(...mutual_series, 0.01);
const c_max = Math.max(...conc_series.filter(v=>v>=0), ...lneg_series, 0.01);

const ctxE = initCanvas("cvEntropy");
const ctxM = initCanvas("cvMutual");
const ctxC = initCanvas("cvConc");

function drawAll() {{
  drawChart(ctxE, entropy_series, "#58a6ff", "S(ρ_A) [nats]", 0, e_max);
  drawChart(ctxM, mutual_series, "#3fb950", "I(A:B) [nats]", 0, m_max);
  const conc_valid = conc_series[0] >= 0;
  if (conc_valid) {{
    drawChart(ctxC, conc_series, "#f0883e", "Concurrence", 0, c_max);
  }} else {{
    drawChart(ctxC, lneg_series, "#bc8cff", "Log-Neg", 0, c_max);
  }}
}}
drawAll();

// ── Cursor lines ─────────────────────────────────────────────────────────────
function drawCursor(ctx, idx) {{
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const pad = {{l:40,r:12,t:10,b:30}};
  const w = W - pad.l - pad.r;
  const x = pad.l + idx / (N - 1) * w;
  ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.lineWidth = 1;
  ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + H - pad.t - pad.b); ctx.stroke();
  ctx.setLineDash([]);
}}

// ── Schmidt bars ──────────────────────────────────────────────────────────────
function updateSchmidt(f) {{
  const container = document.getElementById("schmidtBars");
  const sv = f.schmidt_values || [];
  const maxSv = Math.max(...sv, 0.01);
  container.innerHTML = sv.map((s, i) => {{
    const h = Math.round(s / maxSv * 76);
    const col = `hsl(${{200 + i*30}},70%,55%)`;
    return `<div class="sbar" style="height:${{h}}px;background:${{col}}" title="σ${{i}}=${{s.toFixed(4)}}"></div>`;
  }}).join("");
  const rank = sv.filter(s => s > 1e-4).length;
  document.getElementById("schmidtInfo").textContent =
    `Schmidt rank ≈ ${{rank}} · max σ = ${{sv[0]?.toFixed(4) ?? "—"}}`;
}}

// ── Render ────────────────────────────────────────────────────────────────────
function render(idx) {{
  const f = frames[idx - 1];
  timeLabel.textContent = `t = ${{f.t.toFixed(3)}}  S = ${{f.entropy_a.toFixed(4)}}  I = ${{f.mutual_info.toFixed(4)}}`;
  drawAll();
  [ctxE, ctxM, ctxC].forEach(c => drawCursor(c, idx - 1));
  updateSchmidt(f);
}}
render(1);
slider.addEventListener("input", () => render(+slider.value));

// ── Play/pause ────────────────────────────────────────────────────────────────
let timer = null;
document.getElementById("btnPlay").addEventListener("click", function() {{
  if (timer) {{ clearInterval(timer); timer = null; this.textContent = "▶ Play"; return; }}
  this.textContent = "⏸ Pause";
  let idx = +slider.value;
  timer = setInterval(() => {{
    idx = idx >= N ? 1 : idx + 1;
    slider.value = idx;
    render(idx);
    if (idx >= N) {{ clearInterval(timer); timer = null; document.getElementById("btnPlay").textContent = "▶ Play"; }}
  }}, 80);
}});
</script>
</body>
</html>"""


def _wigner_frames(
    nx: int,
    ny: int,
    total_time: float,
    dt: float,
    potential_type: str,
    samples: int,
) -> dict:
  """Python reference solver: Wigner quasi-probability and transport observables.

  Evolves a 2D Gaussian wavepacket with split-step FFT and, at each sample step,
  computes W(x, p_x) along y=y_mid, probability current J_x / J_y, and momentum
  expectations.  Returns a dict consumable by _wigner_page().
  """
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("Wigner visualization requires numpy") from ex

  pi = math.pi
  dx = 0.25
  dy = 0.25
  xs = np.array([(j - nx / 2.0) * dx for j in range(nx)])
  ys = np.array([(j - ny / 2.0) * dy for j in range(ny)])
  X, Y = np.meshgrid(xs, ys, indexing="ij")   # shape (nx, ny)

  # Initial Gaussian wavepacket: centred at x0 with initial momentum kx0.
  sigma = 1.5
  kx0 = 1.8
  x0, y0 = -4.0, 0.0
  psi: "np.ndarray" = (
      np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (4.0 * sigma * sigma))
      * np.exp(1j * kx0 * X)
  )
  nrm = float(np.sqrt(np.sum(np.abs(psi) ** 2) * dx * dy))
  psi = psi / nrm

  # Static potential.
  if potential_type == "barrier":
    V = np.where(np.abs(X) < 0.4, 2.5, 0.0).astype(float)
  elif potential_type == "harmonic":
    V = 0.05 * (X * X + Y * Y)
  else:
    V = np.zeros_like(X, dtype=float)

  # Spectral propagators (mass=1, hbar=1).
  kx_arr = 2.0 * pi * np.fft.fftfreq(nx, d=dx)
  ky_arr = 2.0 * pi * np.fft.fftfreq(ny, d=dy)
  KX, KY = np.meshgrid(kx_arr, ky_arr, indexing="ij")
  kinetic_half = np.exp(-1j * 0.5 * dt * (KX * KX + KY * KY))
  potential_full = np.exp(-1j * dt * V)

  n_steps = max(1, int(round(total_time / dt)))
  stride = max(1, n_steps // samples)
  ny_mid = ny // 2
  N = nx  # Wigner grid is square: N x N

  def _record(step: int, t: float, psi_now: "np.ndarray") -> dict:
    psi_row = psi_now[:, ny_mid]  # shape (nx,)

    # Wigner function W(x_j, p_k).
    ls = np.arange(N)
    signs = np.where(ls % 2 == 0, 1.0, -1.0)
    W = np.empty((N, N), dtype=float)
    for j in range(N):
      idx_plus = (j + ls) % N
      idx_minus = (j - ls) % N
      a_prime = psi_row[idx_plus] * np.conj(psi_row[idx_minus]) * signs
      w_fft = np.fft.fft(a_prime)
      W[j, :] = (dx / pi) * np.real(w_fft)

    # Probability current J_x, J_y via spectral differentiation.
    psi_k = np.fft.fft2(psi_now)
    Jx = np.imag(np.conj(psi_now) * np.fft.ifft2(1j * KX * psi_k))  # shape (nx, ny)
    Jy = np.imag(np.conj(psi_now) * np.fft.ifft2(1j * KY * psi_k))

    # Transport: integrate J_x at x=1/4*L and x=3/4*L.
    x_left = N // 4
    x_right = (3 * N) // 4
    transmission = float(np.sum(Jx[x_right, :]) * dy)
    reflection = float(-np.sum(Jx[x_left, :]) * dy)

    # Momentum-space expectation values.
    dens_k = np.abs(psi_k) ** 2
    norm_k = float(np.sum(dens_k))
    inv_nk = 1.0 / norm_k if norm_k > 1e-30 else 0.0
    px_mean = float(np.sum(KX * dens_k)) * inv_nk
    py_mean = float(np.sum(KY * dens_k)) * inv_nk
    p2 = float(np.sum((KX * KX + KY * KY) * dens_k)) * inv_nk
    ke = p2 / 2.0  # mass=1

    # p_min / p_max for Wigner axis (k=0 is leftmost momentum bin).
    p_min = -0.5 * pi / dx
    p_max = (0.5 * N - 1.0) * pi / (N * dx)

    return {
        "step": step,
        "time": float(t),
        "nx": N,
        "np": N,
        "x_min": float(xs[0]),
        "x_max": float(xs[-1]),
        "p_min": p_min,
        "p_max": p_max,
        # W[j,k] row-major → wigner_flat[j*np+k]
        "wigner_flat": W.flatten().tolist(),
        # current_x/y: C++ layout [y*nx + x] = Jx.T.flatten()
        "current_x_flat": Jx.T.flatten().tolist(),
        "current_y_flat": Jy.T.flatten().tolist(),
        "transmission": transmission,
        "reflection": reflection,
        "px_mean": px_mean,
        "py_mean": py_mean,
        "kinetic_energy": ke,
    }

  frames = [_record(0, 0.0, psi)]
  for s in range(1, n_steps + 1):
    # Strang splitting: T/2 → V → T/2.
    psi_k2 = np.fft.fft2(psi) * kinetic_half
    psi = np.fft.ifft2(psi_k2) * potential_full
    psi_k2 = np.fft.fft2(psi) * kinetic_half
    psi = np.fft.ifft2(psi_k2)
    # Normalize.
    nrm2 = float(np.sqrt(np.sum(np.abs(psi) ** 2) * dx * dy))
    if nrm2 > 1e-30:
      psi = psi / nrm2
    if s % stride == 0 or s == n_steps:
      frames.append(_record(s, s * dt, psi))

  return {
      "frames": frames,
      "potential_type": potential_type,
      "nx": nx,
      "ny": ny,
      "x_min": float(xs[0]),
      "x_max": float(xs[-1]),
  }


def _wigner_page(data: dict) -> str:
  """Generate a self-contained HTML page visualising the Wigner function and transport."""
  frames = data["frames"]
  if not frames:
    return "<html><body><p>No Wigner frames to display.</p></body></html>"

  potential_type = data.get("potential_type", "unknown")
  frames_json = json.dumps(frames, separators=(",", ":"))

  # Build per-frame scalar time series for the chart panels.
  times_json = json.dumps([f["time"] for f in frames], separators=(",", ":"))
  px_json    = json.dumps([f["px_mean"] for f in frames], separators=(",", ":"))
  ke_json    = json.dumps([f["kinetic_energy"] for f in frames], separators=(",", ":"))
  trans_json = json.dumps([f["transmission"] for f in frames], separators=(",", ":"))
  refl_json  = json.dumps([f["reflection"] for f in frames], separators=(",", ":"))

  nx_val = frames[0].get("nx", 64)
  np_val = frames[0].get("np", 64)
  x_min  = frames[0].get("x_min", -8.0)
  x_max  = frames[0].get("x_max",  8.0)
  p_min  = frames[0].get("p_min", -6.3)
  p_max  = frames[0].get("p_max",  6.3)

  return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>QuantX – Wigner Function ({potential_type})</title>
<style>
  body{{margin:0;padding:12px 16px;font-family:monospace;font-size:12px;background:#111;color:#ddd}}
  h2{{margin:4px 0 10px;color:#7cf;font-size:14px}}
  .row{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:12px}}
  .panel{{display:flex;flex-direction:column;align-items:center}}
  .panel label{{font-size:11px;color:#aaa;margin-bottom:3px}}
  canvas{{background:#000;border:1px solid #333;display:block}}
  .slider-row{{display:flex;align-items:center;gap:10px;margin-bottom:10px}}
  input[type=range]{{width:380px;accent-color:#7cf}}
  #frame-info{{color:#7cf;min-width:120px}}
  svg{{overflow:visible}}
</style>
</head>
<body>
<h2>Wigner Quasi-Probability  ·  {potential_type} potential</h2>

<div class="slider-row">
  <label>frame:</label>
  <input type="range" id="sl" min="0" max="{len(frames)-1}" value="0" step="1">
  <span id="frame-info">t = 0.000</span>
</div>

<div class="row">
  <div class="panel">
    <label>W(x, p&#x2093;) – Wigner quasi-probability  (red&gt;0 / blue&lt;0)</label>
    <canvas id="cv_wigner" width="{nx_val * 4}" height="{np_val * 4}"></canvas>
  </div>
  <div class="panel">
    <label>J&#x2093;(x, y) – x-probability current  (red&gt;0 / blue&lt;0)</label>
    <canvas id="cv_jx" width="{nx_val * 4}" height="{nx_val * 4}"></canvas>
  </div>
</div>

<div class="row">
  <div class="panel">
    <label>&#x27E8;p&#x2093;&#x27E9;(t) and kinetic energy K(t)</label>
    <svg id="svg_mom" width="340" height="150"></svg>
  </div>
  <div class="panel">
    <label>Transmission T(t) and Reflection R(t)</label>
    <svg id="svg_tr" width="340" height="150"></svg>
  </div>
</div>

<div style="font-size:10px;color:#555;margin-top:4px">
  grid {nx_val}×{np_val} &nbsp;|&nbsp;
  x ∈ [{x_min:.2f}, {x_max:.2f}] &nbsp;|&nbsp;
  p ∈ [{p_min:.2f}, {p_max:.2f}]
</div>

<script>
const FRAMES = {frames_json};
const TIMES  = {times_json};
const PX_MEAN = {px_json};
const KE     = {ke_json};
const TRANS  = {trans_json};
const REFL   = {refl_json};

const cvW  = document.getElementById("cv_wigner");
const ctxW = cvW.getContext("2d");
const cvJ  = document.getElementById("cv_jx");
const ctxJ = cvJ.getContext("2d");

function divergingColor(val, maxAbs) {{
  const t = Math.min(1.0, Math.abs(val) / (maxAbs + 1e-30));
  const bg = 255;
  if (val >= 0) {{
    return [Math.round(bg * (1-t) + 220 * t),
            Math.round(bg * (1-t) + 30  * t),
            Math.round(bg * (1-t) + 30  * t)];
  }} else {{
    return [Math.round(bg * (1-t) + 30  * t),
            Math.round(bg * (1-t) + 30  * t),
            Math.round(bg * (1-t) + 220 * t)];
  }}
}}

function drawWigner(frame) {{
  const nx = frame.nx, np = frame.np;
  const W  = frame.wigner_flat;
  let maxW = 0;
  for (let v of W) maxW = Math.max(maxW, Math.abs(v));
  const off = document.createElement("canvas");
  off.width = nx; off.height = np;
  const offCtx = off.getContext("2d");
  const img = offCtx.createImageData(nx, np);
  for (let iy = 0; iy < np; iy++) {{
    const k = np - 1 - iy;   // flip: p_max at top
    for (let ix = 0; ix < nx; ix++) {{
      const w = W[ix * np + k];
      const [r,g,b] = divergingColor(w, maxW);
      const pi = 4 * (iy * nx + ix);
      img.data[pi]=r; img.data[pi+1]=g; img.data[pi+2]=b; img.data[pi+3]=255;
    }}
  }}
  offCtx.putImageData(img, 0, 0);
  ctxW.imageSmoothingEnabled = false;
  ctxW.drawImage(off, 0, 0, cvW.width, cvW.height);
}}

function drawCurrentX(frame) {{
  const nx = frame.nx, ny_size = frame.current_x_flat.length / nx;
  const J  = frame.current_x_flat;
  let maxJ = 0;
  for (let v of J) maxJ = Math.max(maxJ, Math.abs(v));
  const off = document.createElement("canvas");
  off.width = nx; off.height = ny_size;
  const offCtx = off.getContext("2d");
  const img = offCtx.createImageData(nx, ny_size);
  for (let iy = 0; iy < ny_size; iy++) {{
    const iy_flip = ny_size - 1 - iy;
    for (let ix = 0; ix < nx; ix++) {{
      const j_val = J[iy_flip * nx + ix];  // layout: [y*nx + x]
      const [r,g,b] = divergingColor(j_val, maxJ);
      const pi = 4 * (iy * nx + ix);
      img.data[pi]=r; img.data[pi+1]=g; img.data[pi+2]=b; img.data[pi+3]=255;
    }}
  }}
  offCtx.putImageData(img, 0, 0);
  ctxJ.imageSmoothingEnabled = false;
  ctxJ.drawImage(off, 0, 0, cvJ.width, cvJ.height);
}}

// ── SVG line charts ────────────────────────────────────────────────────────
function makeSVGLine(svgId, series, yLabel) {{
  const svg = document.getElementById(svgId);
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height");
  const mg = {{l:44,r:8,t:8,b:26}};
  const pw = W - mg.l - mg.r, ph = H - mg.t - mg.b;

  // Compute ranges.
  const allT = TIMES;
  const tMin = allT[0], tMax = allT[allT.length-1];
  let yMin = Infinity, yMax = -Infinity;
  for (const s of series) for (const v of s.vals) {{ yMin=Math.min(yMin,v); yMax=Math.max(yMax,v); }}
  const yPad = (yMax - yMin) * 0.1 || 0.1;
  yMin -= yPad; yMax += yPad;
  const tRange = tMax - tMin || 1, yRange = yMax - yMin;

  function tx(t){{ return mg.l + (t - tMin) / tRange * pw; }}
  function ty(y){{ return mg.t + (1 - (y - yMin) / yRange) * ph; }}

  let html = `<rect x="${{mg.l}}" y="${{mg.t}}" width="${{pw}}" height="${{ph}}" fill="#1a1a2e" stroke="#333"/>`;

  // Grid lines.
  for (let i=0;i<=4;i++) {{
    const yv = yMin + i * yRange/4;
    const yp = ty(yv);
    html += `<line x1="${{mg.l}}" x2="${{mg.l+pw}}" y1="${{yp}}" y2="${{yp}}" stroke="#333" stroke-dasharray="3,3"/>`;
    html += `<text x="${{mg.l-4}}" y="${{yp+4}}" text-anchor="end" fill="#888" font-size="9">${{yv.toFixed(2)}}</text>`;
  }}

  // Zero line if zero is in range.
  if (yMin < 0 && yMax > 0) {{
    html += `<line x1="${{mg.l}}" x2="${{mg.l+pw}}" y1="${{ty(0)}}" y2="${{ty(0)}}" stroke="#555" stroke-width="1"/>`;
  }}

  // Series.
  const colors = ["#f55","#55f","#5f5","#fa0"];
  for (let si=0; si<series.length; si++) {{
    const s = series[si];
    const pts = TIMES.map((t,i)=>`${{tx(t).toFixed(1)}},${{ty(s.vals[i]).toFixed(1)}}`).join(" ");
    html += `<polyline points="${{pts}}" fill="none" stroke="${{colors[si]}}" stroke-width="1.5"/>`;
    html += `<text x="${{mg.l+pw-4}}" y="${{mg.t+14+si*14}}" text-anchor="end" fill="${{colors[si]}}" font-size="9">${{s.name}}</text>`;
  }}

  // Current time marker (updated in render()).
  html += `<line id="${{svgId}}_tm" x1="${{mg.l}}" x2="${{mg.l}}" y1="${{mg.t}}" y2="${{mg.t+ph}}" stroke="#7cf" stroke-width="1" stroke-dasharray="4,3"/>`;

  // Axes.
  html += `<line x1="${{mg.l}}" x2="${{mg.l+pw}}" y1="${{mg.t+ph}}" y2="${{mg.t+ph}}" stroke="#666"/>`;
  html += `<line x1="${{mg.l}}" x2="${{mg.l}}" y1="${{mg.t}}" y2="${{mg.t+ph}}" stroke="#666"/>`;
  html += `<text x="${{mg.l+pw/2}}" y="${{H-4}}" text-anchor="middle" fill="#888" font-size="9">t</text>`;
  html += `<text x="10" y="${{mg.t+ph/2}}" text-anchor="middle" transform="rotate(-90,10,${{mg.t+ph/2}})" fill="#888" font-size="9">${{yLabel}}</text>`;

  svg.innerHTML = html;
}}

makeSVGLine("svg_mom", [{{name:"⟨px⟩", vals:PX_MEAN}}, {{name:"K.E.", vals:KE}}], "energy");
makeSVGLine("svg_tr",  [{{name:"T", vals:TRANS}}, {{name:"R", vals:REFL}}], "flux");

function updateTimeLine(svgId, t) {{
  const svg = document.getElementById(svgId);
  const W = +svg.getAttribute("width");
  const mg = {{l:44,r:8,t:8,b:26}};
  const pw = W - mg.l - mg.r;
  const tMin = TIMES[0], tMax = TIMES[TIMES.length-1], tRange = tMax-tMin||1;
  const xp = (mg.l + (t-tMin)/tRange*pw).toFixed(1);
  const el = document.getElementById(svgId+"_tm");
  if (el) {{ el.setAttribute("x1",xp); el.setAttribute("x2",xp); }}
}}

function render(idx) {{
  const frame = FRAMES[idx];
  document.getElementById("frame-info").textContent = `t = ${{frame.time.toFixed(3)}}`;
  drawWigner(frame);
  drawCurrentX(frame);
  updateTimeLine("svg_mom", frame.time);
  updateTimeLine("svg_tr",  frame.time);
}}

document.getElementById("sl").addEventListener("input", function() {{
  render(+this.value);
}});

render(0);
</script>
</body>
</html>
"""


def _rabi_data(engine, samples: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
  """Collect the simulation and analytic traces for the Rabi demo."""
  pts_sim = []
  pts_ref = []
  for i in range(samples):
    t = math.pi * i / max(1, samples - 1)
    problem = {
        "qubits": 1,
        "equation": "Schrodinger",
        "total_time": t,
        "dt": 0.01,
        "method": "krylov",
        "krylov_dim": 8,
        "hamiltonian_terms": [{"label": "X0", "coefficient": 1.0}],
    }
    result = engine.run(problem, {"enable_checkpoint": False})
    pts_sim.append((t, float(result.get("observables", {}).get("population_1", 0.0))))
    pts_ref.append((t, math.sin(t) ** 2))
  return pts_sim, pts_ref


def _mps_data(engine, samples: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
  """Collect the bond-dimension and truncation traces for the MPS demo."""
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


# ── M22: Quantum Error Correction ─────────────────────────────────────────────

def _qec_frames(
    code: str = "bit_flip_3",
    physical_error_rate: float = 0.05,
    error_type: str = "X",
    n_trials: int = 2000,
    n_detail_trials: int = 64,
    sweep_error_rates: list[float] | None = None,
    seed: int = 42,
) -> dict:
  """Monte Carlo QEC reference simulator.

  Three codes supported:
    bit_flip_3   — 3-qubit repetition code, corrects any single X error.
                   Syndromes: Z₀Z₁ (s0), Z₁Z₂ (s1).
    phase_flip_3 — 3-qubit phase-flip code, corrects any single Z error.
                   Same syndrome table as bit_flip_3 but for Z errors.
    shor_9       — 9-qubit Shor code, corrects any single-qubit Pauli error.
                   Syndromes: 6 Z-type (bit-flip per group) + 2 X-type (phase across groups).

  Theoretical logical error rates:
    Repetition-3: p_log = 3p²(1−p) + p³
    Shor-9 (approx composite): p_log ≈ theory_rep3(theory_rep3(p))

  Returns dict with keys:
    code, n_physical, n_syndrome, physical_error_rate, error_type, n_trials,
    logical_error_rate, logical_error_rate_theory, frames, syndrome_histogram,
    threshold_curve.
  """
  import random as _rnd

  rng = _rnd.Random(seed)

  CODE_PARAMS = {
      "bit_flip_3":   {"n_physical": 3, "n_syndrome": 2},
      "phase_flip_3": {"n_physical": 3, "n_syndrome": 2},
      "shor_9":       {"n_physical": 9, "n_syndrome": 8},
  }
  if code not in CODE_PARAMS:
    raise ValueError(f"Unknown QEC code: {code!r}. Choose bit_flip_3 | phase_flip_3 | shor_9")
  n_physical = CODE_PARAMS[code]["n_physical"]
  n_syndrome = CODE_PARAMS[code]["n_syndrome"]

  # ── Theoretical formulas ────────────────────────────────────────────────────
  def _theory_rep3(p: float) -> float:
    return 3 * p * p * (1 - p) + p * p * p

  def _theory(p: float) -> float:
    if code == "shor_9":
      # Effective per-qubit error rates for each Pauli component.
      p_x = (2 * p / 3) if error_type == "depolarizing" else (p if error_type in ("X",) else 0.0)
      p_z = (2 * p / 3) if error_type == "depolarizing" else (p if error_type in ("Z",) else 0.0)
      # Bit-flip logical: at least one group fails its 3-qubit bit-flip code.
      p_bf = 1.0 - (1.0 - _theory_rep3(p_x)) ** 3 if p_x > 0 else 0.0
      # Phase-flip logical: outer 3-qubit code fails; group effective Z = odd-parity count.
      if p_z > 0:
        p_z_odd = 3 * p_z * (1 - p_z) ** 2 + p_z ** 3
        p_pf = _theory_rep3(p_z_odd)
      else:
        p_pf = 0.0
      return p_bf + p_pf - p_bf * p_pf
    return _theory_rep3(p)

  # ── Single-trial helpers ─────────────────────────────────────────────────────
  def _lookup3(e0: bool, e1: bool, e2: bool) -> tuple[int, int, int]:
    """Z0Z1 / Z1Z2 parity syndrome + correction qubit for repetition codes."""
    s0 = e0 ^ e1
    s1 = e1 ^ e2
    c = {(False, False): -1, (True, False): 0, (True, True): 1, (False, True): 2}[(s0, s1)]
    return int(s0), int(s1), c

  def _trial_bit_flip(p: float) -> tuple[str, str, str, bool]:
    e = [rng.random() < p for _ in range(3)]
    s0, s1, c = _lookup3(e[0], e[1], e[2])
    res = e[:]
    if c >= 0:
      res[c] = not res[c]
    ok = sum(res) < 2
    err = "".join(f"X{q}" for q in range(3) if e[q]) or "I"
    cor = f"X{c}" if c >= 0 else "I"
    return err, f"{s0}{s1}", cor, ok

  def _trial_phase_flip(p: float) -> tuple[str, str, str, bool]:
    e = [rng.random() < p for _ in range(3)]
    s0, s1, c = _lookup3(e[0], e[1], e[2])
    res = e[:]
    if c >= 0:
      res[c] = not res[c]
    ok = sum(res) < 2
    err = "".join(f"Z{q}" for q in range(3) if e[q]) or "I"
    cor = f"Z{c}" if c >= 0 else "I"
    return err, f"{s0}{s1}", cor, ok

  def _trial_shor(p: float) -> tuple[str, str, str, bool]:
    # Sample errors
    eX = [False] * 9
    eZ = [False] * 9
    if error_type == "X":
      eX = [rng.random() < p for _ in range(9)]
    elif error_type == "Z":
      eZ = [rng.random() < p for _ in range(9)]
    else:  # depolarizing
      for q in range(9):
        r = rng.random()
        if r < p / 3:
          eX[q] = True
        elif r < 2 * p / 3:
          eZ[q] = True
        elif r < p:
          eX[q] = True
          eZ[q] = True

    # Bit-flip syndromes (Z-stabilizers, detect X per group)
    bf_syn = []
    for g in range(3):
      b = 3 * g
      bf_syn.append(eX[b] ^ eX[b + 1])
      bf_syn.append(eX[b + 1] ^ eX[b + 2])

    # Phase-flip syndromes (X-stabilizers, detect Z across groups)
    gzp = [sum(int(eZ[3 * g + k]) for k in range(3)) % 2 for g in range(3)]
    ps0 = gzp[0] ^ gzp[1]
    ps1 = gzp[1] ^ gzp[2]
    syndrome_str = "".join(map(str, bf_syn + [ps0, ps1]))

    # Bit-flip correction per group
    resX = eX[:]
    for g in range(3):
      b = 3 * g
      _, _, c = _lookup3(eX[b], eX[b + 1], eX[b + 2])
      if c >= 0:
        resX[b + c] = not resX[b + c]

    # Phase-flip correction (Z on first qubit of identified group)
    resZ = eZ[:]
    pf_lookup = {(False, False): -1, (True, False): 0, (True, True): 1, (False, True): 2}
    cg = pf_lookup[(bool(ps0), bool(ps1))]
    if cg >= 0:
      resZ[3 * cg] = not resZ[3 * cg]

    # Logical error check
    bf_err = any(sum(int(resX[3 * g + k]) for k in range(3)) >= 2 for g in range(3))
    pf_err = sum(sum(int(resZ[3 * g + k]) for k in range(3)) % 2 for g in range(3)) >= 2
    ok = not bf_err and not pf_err

    parts_x = [f"X{q}" for q in range(9) if eX[q]]
    parts_z = [f"Z{q}" for q in range(9) if eZ[q]]
    err = " ".join(parts_x + parts_z) or "I"
    return err, syndrome_str, "decoded", ok

  # ── Monte Carlo runner ───────────────────────────────────────────────────────
  def _run_mc(p_phys: float, n: int, collect_detail: bool) -> tuple[float, list, dict]:
    successes = 0
    frames: list[dict] = []
    syn_hist: dict[str, dict] = {}

    for t in range(n):
      if code == "bit_flip_3":
        err_s, syn_s, cor_s, ok = _trial_bit_flip(p_phys)
      elif code == "phase_flip_3":
        err_s, syn_s, cor_s, ok = _trial_phase_flip(p_phys)
      else:
        err_s, syn_s, cor_s, ok = _trial_shor(p_phys)

      if ok:
        successes += 1
      if collect_detail and t < n_detail_trials:
        frames.append({"trial": t, "error": err_s, "syndrome": syn_s,
                        "correction": cor_s, "succeeded": ok})
      if syn_s not in syn_hist:
        syn_hist[syn_s] = {"count": 0, "successes": 0}
      syn_hist[syn_s]["count"] += 1
      if ok:
        syn_hist[syn_s]["successes"] += 1

    p_log = 1.0 - successes / n if n > 0 else 0.0
    return p_log, frames, syn_hist

  p_log, frames, syn_hist = _run_mc(physical_error_rate, n_trials, collect_detail=True)

  # ── Threshold sweep ─────────────────────────────────────────────────────────
  threshold_curve: list[dict] = []
  if sweep_error_rates:
    sweep_n = max(500, n_trials // 4)
    for p_sw in sweep_error_rates:
      p_log_mc, _, _ = _run_mc(p_sw, sweep_n, collect_detail=False)
      threshold_curve.append({
          "p_physical": p_sw,
          "p_logical_mc": p_log_mc,
          "p_logical_theory": _theory(p_sw),
      })

  return {
      "code": code,
      "n_physical": n_physical,
      "n_syndrome": n_syndrome,
      "physical_error_rate": physical_error_rate,
      "error_type": error_type,
      "n_trials": n_trials,
      "n_detail_trials": min(n_detail_trials, n_trials),
      "logical_error_rate": p_log,
      "logical_error_rate_theory": _theory(physical_error_rate),
      "frames": frames,
      "syndrome_histogram": syn_hist,
      "threshold_curve": threshold_curve,
  }


def _qec_page(data: dict, title: str = "QEC") -> str:
  """Generate a self-contained HTML page for QEC Monte Carlo results."""
  import json as _json

  data_js = _json.dumps(data, ensure_ascii=False)
  code_label = {
      "bit_flip_3":   "3-Qubit Bit-Flip Code",
      "phase_flip_3": "3-Qubit Phase-Flip Code",
      "shor_9":       "9-Qubit Shor Code",
  }.get(data.get("code", ""), data.get("code", ""))

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --accent:#58a6ff;
           --green:#3fb950; --red:#f85149; --orange:#d29922; --muted:#8b949e;
           --text:#e6edf3; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
          min-height:100vh; padding:24px; }}
  h1 {{ font-size:1.4rem; font-weight:600; color:var(--accent); margin-bottom:4px; }}
  .subtitle {{ color:var(--muted); font-size:.85rem; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; max-width:1140px; margin:0 auto; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px;
            padding:16px; }}
  .panel.wide {{ grid-column:span 2; }}
  .panel h2 {{ font-size:.95rem; color:var(--muted); margin-bottom:12px; font-weight:500; }}
  canvas {{ display:block; width:100%; }}
  .stat-row {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:18px;
               max-width:1140px; margin-left:auto; margin-right:auto; margin-bottom:18px; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:8px;
           padding:8px 14px; min-width:120px; }}
  .stat-label {{ font-size:.75rem; color:var(--muted); margin-bottom:2px; }}
  .stat-value {{ font-size:1.1rem; font-weight:600; color:var(--accent); }}
  .stat-value.good {{ color:var(--green); }}
  .stat-value.warn {{ color:var(--orange); }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">{code_label} — Monte Carlo reference simulation</p>

<div class="stat-row" id="stats"></div>

<div class="grid">
  <div class="panel">
    <h2>Trial Grid (first {min(data.get('n_detail_trials', 64), len(data.get('frames', [])))} trials)</h2>
    <canvas id="cvGrid" height="200"></canvas>
  </div>
  <div class="panel">
    <h2>Syndrome Success Rate</h2>
    <canvas id="cvSyndrome" height="200"></canvas>
  </div>
  <div class="panel wide" id="thresholdPanel" style="display:none">
    <h2>Logical Error Rate vs Physical Error Rate (Threshold Curve)</h2>
    <canvas id="cvThreshold" height="240"></canvas>
  </div>
</div>

<script>
const DATA = {data_js};

// ── Stats ─────────────────────────────────────────────────────────────────────
(function() {{
  const s = document.getElementById('stats');
  function stat(label, val, cls) {{
    const d = document.createElement('div'); d.className = 'stat';
    d.innerHTML = `<div class="stat-label">${{label}}</div>
                   <div class="stat-value ${{cls||''}}">${{val}}</div>`;
    s.appendChild(d);
  }}
  stat('Code', DATA.code.replace(/_/g,' '));
  stat('n_physical', DATA.n_physical);
  stat('n_syndrome', DATA.n_syndrome);
  stat('p_physical', DATA.physical_error_rate.toFixed(3));
  stat('Error type', DATA.error_type);
  stat('n_trials', DATA.n_trials);
  const ler = DATA.logical_error_rate;
  const cls = ler < DATA.physical_error_rate ? 'good' : 'warn';
  stat('p_logical (MC)', ler.toFixed(4), cls);
  stat('p_logical (theory)', DATA.logical_error_rate_theory.toFixed(4));
  const gain = DATA.physical_error_rate > 0
    ? (DATA.physical_error_rate / Math.max(ler, 1e-9)).toFixed(1) + '×'
    : '—';
  stat('Error reduction', gain, ler < DATA.physical_error_rate ? 'good' : 'warn');
}})();

// ── Trial grid ─────────────────────────────────────────────────────────────────
function drawGrid() {{
  const cv = document.getElementById('cvGrid');
  const dpr = window.devicePixelRatio||1;
  cv.width  = (cv.offsetWidth||500)*dpr;
  cv.height = (cv.offsetHeight||200)*dpr;
  const ctx = cv.getContext('2d'); ctx.scale(dpr,dpr);
  const W = cv.offsetWidth||500, H = cv.offsetHeight||200;
  ctx.fillStyle='#161b22'; ctx.fillRect(0,0,W,H);

  const frames = DATA.frames||[];
  if (!frames.length) return;
  const n = frames.length;
  const cols = Math.ceil(Math.sqrt(n * W / H));
  const rows = Math.ceil(n / cols);
  const cellW = (W-4) / cols, cellH = (H-4) / rows;
  const pad = 1;

  for (let i=0; i<n; i++) {{
    const f = frames[i];
    const col = i % cols, row = Math.floor(i/cols);
    const x = 2 + col*cellW, y = 2 + row*cellH;
    const hasErr = f.error !== 'I';
    const ok = f.succeeded;
    ctx.fillStyle = ok ? (hasErr ? '#2ea04326' : '#3fb95033') : '#f8514966';
    if (!ok) ctx.fillStyle = '#f85149cc';
    else if (hasErr) ctx.fillStyle = '#3fb95088';
    else ctx.fillStyle = '#30363d';
    ctx.fillRect(x+pad, y+pad, cellW-2*pad, cellH-2*pad);
  }}

  // Legend
  ctx.font = '10px system-ui'; ctx.textAlign='left';
  const ly = H-6;
  const legend = [['#3fb95088','Corrected'], ['#f85149cc','Logical error'], ['#30363d','No error']];
  let lx = 4;
  for (const [c, lab] of legend) {{
    ctx.fillStyle=c; ctx.fillRect(lx, ly-8, 10, 10);
    ctx.fillStyle='#8b949e'; ctx.fillText(' '+lab, lx+12, ly);
    lx += ctx.measureText(' '+lab).width + 20;
  }}
}}

// ── Syndrome success-rate bar chart ────────────────────────────────────────────
function drawSyndrome() {{
  const cv = document.getElementById('cvSyndrome');
  const dpr = window.devicePixelRatio||1;
  cv.width  = (cv.offsetWidth||500)*dpr;
  cv.height = (cv.offsetHeight||200)*dpr;
  const ctx = cv.getContext('2d'); ctx.scale(dpr,dpr);
  const W = cv.offsetWidth||500, H = cv.offsetHeight||200;
  ctx.fillStyle='#161b22'; ctx.fillRect(0,0,W,H);

  const hist = DATA.syndrome_histogram||{{}};
  const entries = Object.entries(hist).sort((a,b)=>b[1].count-a[1].count);
  if (!entries.length) return;

  const pad = {{l:40,r:10,t:12,b:36}};
  const cw = W-pad.l-pad.r, ch = H-pad.t-pad.b;
  const n = entries.length;
  const barW = Math.max(1, cw/n - 2);

  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch);
  ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();

  // Y gridlines at 25%, 50%, 75%, 100%
  ctx.fillStyle='#8b949e'; ctx.font='9px system-ui'; ctx.textAlign='right';
  for (let t=0; t<=4; t++) {{
    const y = pad.t + ch - (t/4)*ch;
    ctx.fillText((t*25)+'%', pad.l-3, y+3);
    ctx.strokeStyle='#21262d'; ctx.lineWidth=0.5;
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+cw,y); ctx.stroke();
  }}

  for (let i=0; i<n; i++) {{
    const [syn, stats] = entries[i];
    const rate = stats.count>0 ? stats.successes/stats.count : 0;
    const x = pad.l + i*(cw/n);
    const bh = rate*ch;
    ctx.fillStyle = rate > 0.9 ? '#3fb950' : rate > 0.5 ? '#d29922' : '#f85149';
    ctx.fillRect(x+1, pad.t+ch-bh, barW, bh);
    // Label
    ctx.fillStyle='#8b949e'; ctx.font='8px system-ui'; ctx.textAlign='center';
    if (barW > 12) ctx.fillText(syn, x+barW/2+1, pad.t+ch+12);
  }}
  ctx.fillStyle='#8b949e'; ctx.font='10px system-ui'; ctx.textAlign='center';
  ctx.fillText('Syndrome pattern', pad.l+cw/2, H-2);
}}

// ── Threshold curve ────────────────────────────────────────────────────────────
function drawThreshold() {{
  const tc = DATA.threshold_curve||[];
  if (!tc.length) return;
  document.getElementById('thresholdPanel').style.display='';
  const cv = document.getElementById('cvThreshold');
  const dpr = window.devicePixelRatio||1;
  cv.width  = (cv.offsetWidth||900)*dpr;
  cv.height = (cv.offsetHeight||240)*dpr;
  const ctx = cv.getContext('2d'); ctx.scale(dpr,dpr);
  const W = cv.offsetWidth||900, H = cv.offsetHeight||240;
  ctx.fillStyle='#161b22'; ctx.fillRect(0,0,W,H);

  const pad = {{l:52,r:20,t:14,b:40}};
  const cw = W-pad.l-pad.r, ch = H-pad.t-pad.b;

  const pMax = Math.max(...tc.map(p=>p.p_physical), 0.5);
  function xOf(p) {{ return pad.l + (p/pMax)*cw; }}
  function yOf(p) {{ return pad.t + ch - (p/pMax)*ch; }}

  // Axes
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch);
  ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();

  // Grid + ticks
  ctx.fillStyle='#8b949e'; ctx.font='10px system-ui';
  for (let t=0; t<=4; t++) {{
    const p = t/4*pMax;
    const y = yOf(p), x = xOf(p);
    ctx.textAlign='right'; ctx.fillText(p.toFixed(2), pad.l-4, y+3);
    ctx.strokeStyle='#21262d'; ctx.lineWidth=0.5;
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+cw,y); ctx.stroke();
    ctx.textAlign='center'; ctx.fillText(p.toFixed(2), x, pad.t+ch+14);
    ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,pad.t+ch); ctx.stroke();
  }}

  // Diagonal p_log = p_phys (no improvement)
  ctx.strokeStyle='#30363d66'; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(xOf(0),yOf(0)); ctx.lineTo(xOf(pMax),yOf(pMax)); ctx.stroke();
  ctx.setLineDash([]);

  // Theory curve
  ctx.strokeStyle='#58a6ff'; ctx.lineWidth=2;
  ctx.beginPath();
  for (let i=0; i<tc.length; i++) {{
    const px = xOf(tc[i].p_physical), py = yOf(tc[i].p_logical_theory);
    if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();

  // MC dots
  ctx.fillStyle='#d29922';
  for (const pt of tc) {{
    ctx.beginPath(); ctx.arc(xOf(pt.p_physical), yOf(pt.p_logical_mc), 4, 0, 2*Math.PI); ctx.fill();
  }}
  // MC line
  ctx.strokeStyle='#d29922'; ctx.lineWidth=1.5;
  ctx.beginPath();
  for (let i=0; i<tc.length; i++) {{
    const px = xOf(tc[i].p_physical), py = yOf(tc[i].p_logical_mc);
    if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();

  // Labels
  ctx.fillStyle='#8b949e'; ctx.font='10px system-ui'; ctx.textAlign='center';
  ctx.fillText('Physical error rate p', pad.l+cw/2, H-4);
  ctx.save(); ctx.translate(13,pad.t+ch/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign='center'; ctx.fillText('Logical error rate', 0, 0); ctx.restore();

  // Legend
  ctx.textAlign='left'; ctx.font='11px system-ui';
  ctx.strokeStyle='#58a6ff'; ctx.lineWidth=2; ctx.beginPath();
  ctx.moveTo(pad.l+4, pad.t+10); ctx.lineTo(pad.l+22, pad.t+10); ctx.stroke();
  ctx.fillStyle='#e6edf3'; ctx.fillText(' Theory', pad.l+24, pad.t+14);
  ctx.fillStyle='#d29922'; ctx.beginPath(); ctx.arc(pad.l+4+100+9, pad.t+10, 4, 0, 2*Math.PI); ctx.fill();
  ctx.fillStyle='#e6edf3'; ctx.fillText(' Monte Carlo', pad.l+4+100+15, pad.t+14);
  ctx.strokeStyle='#30363d66'; ctx.lineWidth=1; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(pad.l+4+230, pad.t+10); ctx.lineTo(pad.l+22+230, pad.t+10); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#e6edf3'; ctx.fillText(' No correction', pad.l+24+230, pad.t+14);
}}

window.addEventListener('load', () => {{ drawGrid(); drawSyndrome(); drawThreshold(); }});
window.addEventListener('resize', () => {{ drawGrid(); drawSyndrome(); drawThreshold(); }});
</script>
</body>
</html>"""
  return html


# ── M24: φ⁴ Scalar Quantum Field Theory ───────────────────────────────────────

def _phi4_dynamics_frames(
    n_sites: int = 4,
    d_local: int = 6,
    lam: float = 1.0,
    kappa: float = 1.0,
    m2_initial: float = 1.0,
    m2_final: float = -1.5,
    n_steps: int = 80,
    dt: float = 0.05,
    epsilon: float = 0.0,
) -> dict:
  """Compute φ⁴ quantum quench dynamics via exact time evolution (NumPy/SciPy reference)."""
  import numpy as np

  d = d_local
  N = n_sites

  # Local operators (same as in _phi4_frames)
  a_loc  = np.diag(np.sqrt(np.arange(1, d, dtype=float)), 1)
  adag   = a_loc.T
  phi_loc  = (a_loc + adag) / np.sqrt(2.0)
  pi2_loc  = np.real(-0.5 * (a_loc - adag) @ (a_loc - adag))
  phi2_loc = phi_loc @ phi_loc
  phi4_loc = phi2_loc @ phi2_loc
  n_op_loc = adag @ a_loc  # occupation number operator

  def kron_single(op, site):
    """Embed a d×d operator at `site` into the full d^N Hilbert space."""
    ops = [np.eye(d)] * N
    ops[site] = op
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  def kron_pair(op1, s1, op2, s2):
    """Embed op1⊗op2 at sites s1,s2 into the full space."""
    ops = [np.eye(d)] * N
    ops[s1] = op1
    ops[s2] = op2
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  def build_H(m2_val: float) -> "np.ndarray":
    dim = d ** N
    H = np.zeros((dim, dim), dtype=float)
    for i in range(N):
      H += 0.5 * kron_single(pi2_loc,  i)
      H += (0.5 * m2_val) * kron_single(phi2_loc, i)
      H += (0.25 * lam)   * kron_single(phi4_loc, i)
      H += epsilon        * kron_single(phi_loc,  i)
    for i in range(N - 1):
      H -= kappa * kron_pair(phi_loc, i, phi_loc, i + 1)
    return H

  def site_expect(psi, op, site):
    """Compute Re<psi|O_site|psi> without building the full kron."""
    # Reshape psi as d^site × d × d^(N-site-1)
    left  = d ** site
    right = d ** (N - site - 1)
    v = psi.reshape(left, d, right)
    Ov = np.tensordot(op, v, axes=([1], [1])).transpose(1, 0, 2)
    return float(np.real(np.vdot(psi, Ov.reshape(-1))))

  H_init  = build_H(m2_initial)
  H_final = build_H(m2_final)

  # Ground state of pre-quench Hamiltonian
  try:
    import numpy.linalg as nla
    vals_i, vecs_i = nla.eigh(H_init)
    psi0 = vecs_i[:, 0].copy()
  except Exception:
    psi0 = np.ones(d**N, dtype=float) / np.sqrt(d**N)

  # Post-quench gap for reference
  try:
    vals_f, _ = np.linalg.eigh(H_final)
    delta_E_final = float(vals_f[1] - vals_f[0]) if len(vals_f) > 1 else 0.0
    E0_initial = float(vals_i[0])
  except Exception:
    delta_E_final = 0.0
    E0_initial = 0.0

  # Time-evolution operator (try scipy, fall back to Taylor Krylov)
  try:
    from scipy.linalg import expm as _scipy_expm
    U_step = _scipy_expm(-1j * dt * H_final)
    def evolve(psi_cx):
      return U_step @ psi_cx
  except ImportError:
    # Fallback: small-m Arnoldi Krylov (m=15)
    def evolve(psi_cx):
      m = min(15, len(psi_cx))
      V = []
      b = float(np.linalg.norm(psi_cx))
      v = psi_cx / b
      H_m = np.zeros((m, m), dtype=complex)
      for j in range(m):
        V.append(v)
        w = H_final @ v
        for i in range(j + 1):
          H_m[i, j] = np.vdot(V[i], w)
          w = w - H_m[i, j] * V[i]
        nrm = float(np.linalg.norm(w))
        if nrm < 1e-14:
          m = j + 1
          break
        if j < m - 1:
          H_m[j + 1, j] = nrm
          v = w / nrm
      Vm = np.column_stack(V[:m])
      exp_Hm = np.array([[np.exp(-1j * dt * ev) for ev in np.linalg.eigvalsh(H_m[:m, :m])]])
      evals, evecs = np.linalg.eigh(H_m[:m, :m])
      exp_Hm_mat = evecs @ np.diag(np.exp(-1j * dt * evals)) @ evecs.conj().T
      e1 = np.zeros(m, dtype=complex); e1[0] = b
      phi_m = exp_Hm_mat @ e1
      return Vm @ phi_m

  def entropy_from_psi(psi_cx):
    dim_A = d ** (N // 2)
    dim_B = d ** (N - N // 2)
    M = psi_cx.reshape(dim_A, dim_B)
    sv = np.linalg.svd(M, compute_uv=False)
    probs = sv ** 2
    probs = probs[probs > 1e-14]
    return float(-np.sum(probs * np.log(probs))) if len(probs) > 0 else 0.0

  psi = psi0.astype(complex).copy()
  frames = []
  for step in range(n_steps + 1):
    t = step * dt
    phi_exp  = [site_expect(psi, phi_loc,  i) for i in range(N)]
    n_exp    = [site_expect(psi, n_op_loc, i) for i in range(N)]
    phi2_exp = [site_expect(psi, phi2_loc, i) for i in range(N)]
    order    = float(np.sqrt(max(0.0, sum(phi2_exp) / N)))
    S_ent    = entropy_from_psi(psi)
    Hpsi     = H_final @ psi
    energy   = float(np.real(np.vdot(psi, Hpsi)))
    frames.append({
        "time":        round(t, 6),
        "phi_expect":  [round(float(np.real(x)), 8) for x in phi_exp],
        "occupation":  [round(float(np.real(x)), 8) for x in n_exp],
        "order_param": round(order, 8),
        "entropy":     round(S_ent, 8),
        "energy":      round(energy, 8),
    })
    if step < n_steps:
      psi = evolve(psi)
      nrm = float(np.linalg.norm(psi))
      if nrm > 1e-14:
        psi /= nrm

  return {
      "n_sites":       N,
      "d_local":       d,
      "m2_initial":    m2_initial,
      "m2_final":      m2_final,
      "lam":           lam,
      "kappa":         kappa,
      "dt":            dt,
      "n_steps":       n_steps,
      "E0_initial":    round(E0_initial, 8),
      "delta_E_final": round(delta_E_final, 8),
      "frames":        frames,
  }


def _phi4_frames(
    n_sites: int = 4,
    m2: float = 1.0,
    lam: float = 1.0,
    kappa: float = 1.0,
    d_local: int = 10,
    scenario: str = "symmetric",
    n_sweep: int = 25,
    m2_min: float = -3.0,
    m2_max: float = 2.5,
    epsilon: float = 0.05,
    seed: int = 42,
) -> dict:
  """φ⁴ scalar QFT on a 1D lattice via truncated Fock space exact diagonalization.

  Hamiltonian (natural units, lattice spacing a=1):

    H = Σᵢ [½πᵢ² + ½m²φᵢ² + (λ/4)φᵢ⁴] − κ Σ⟨ij⟩ φᵢφⱼ

  Local operators in truncated oscillator basis (d_local levels):
    φᵢ = (âᵢ + â†ᵢ)/√2,   πᵢ = i(â†ᵢ − âᵢ)/√2

  A small explicit symmetry-breaking field ε·Σφᵢ is added to select a vacuum
  in the broken phase (m² < 0), making ⟨φ⟩ non-zero there.

  Scenarios:
    "symmetric"  — m² > 0 ground state + spectral gap
    "broken"     — m² < 0 ground state showing SSB
    "sweep"      — scan m² from m2_max → m2_min tracking order parameter + gap
  """
  try:
    import numpy as np
    from numpy.linalg import eigh
  except ImportError as ex:
    raise RuntimeError("phi4 visualization requires numpy") from ex

  N = n_sites
  d = d_local

  # ── Local Fock-space operators ─────────────────────────────────────────────
  # â: lowering, (â)_{n-1,n} = √n
  a_loc = np.diag(np.sqrt(np.arange(1, d, dtype=float)), 1)   # d×d, upper triangular
  adag  = a_loc.T
  phi_loc  = (a_loc + adag) / np.sqrt(2.0)                   # position-like field
  pi_loc   = 1j * (adag - a_loc) / np.sqrt(2.0)              # momentum-like field
  pi2_loc  = np.real(-0.5 * (a_loc - adag) @ (a_loc - adag)) # ½π² per site
  phi2_loc = phi_loc @ phi_loc
  phi4_loc = phi2_loc @ phi2_loc
  n_op_loc = np.diag(np.arange(d, dtype=float))               # number operator N̂

  I_loc = np.eye(d, dtype=float)

  # ── Tensor product helpers ─────────────────────────────────────────────────
  def kron_single(op: "np.ndarray", site: int) -> "np.ndarray":
    """I ⊗ ... ⊗ op (at site) ⊗ ... ⊗ I, total N sites."""
    ops = [I_loc] * N
    ops[site] = op
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  def kron_pair(op1: "np.ndarray", s1: int, op2: "np.ndarray", s2: int) -> "np.ndarray":
    ops = [I_loc] * N
    ops[s1] = op1
    ops[s2] = op2
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  # ── Hamiltonian builder ────────────────────────────────────────────────────
  def build_H(m2_val: float, eps: float = epsilon) -> "np.ndarray":
    dim = d ** N
    H = np.zeros((dim, dim), dtype=float)
    for i in range(N):
      H += 0.5 * kron_single(pi2_loc,      i)         # ½πᵢ²
      H += (0.5 * m2_val) * kron_single(phi2_loc,  i)   # ½m²φᵢ²
      H += (0.25 * lam)   * kron_single(phi4_loc,  i)   # λ/4 φᵢ⁴
      H += eps             * kron_single(phi_loc,   i)   # explicit Z₂ breaking
    for i in range(N - 1):
      H -= kappa * kron_pair(phi_loc, i, phi_loc, i + 1)  # −κ φᵢφᵢ₊₁
    return H

  # ── Solve: get lowest 3 eigenpairs ────────────────────────────────────────
  def solve(m2_val: float, eps: float = epsilon) -> "tuple":
    H = build_H(m2_val, eps)
    # eigh returns eigenvalues in ascending order.
    vals, vecs = eigh(H)
    return vals[:3], vecs[:, :3]   # (energies, states)

  # ── Observables from a state vector ───────────────────────────────────────
  def site_expect(psi: "np.ndarray", op_loc: "np.ndarray", site: int) -> float:
    """⟨ψ|op_site|ψ⟩ using tensor structure (avoid full kron for speed)."""
    d_n = d ** N
    d_before = d ** site
    d_after  = d ** (N - site - 1)
    psi_r = psi.reshape(d_before, d, d_after)
    # Contract op over the site index.
    psi_op = np.einsum("ij,ajb->aib", op_loc, psi_r)
    return float(np.real(np.dot(psi.ravel(), psi_op.ravel())))

  def all_site_obs(psi: "np.ndarray") -> dict:
    phi_exp  = [site_expect(psi, phi_loc,  i) for i in range(N)]
    phi2_exp = [site_expect(psi, phi2_loc, i) for i in range(N)]
    n_exp    = [site_expect(psi, n_op_loc, i) for i in range(N)]
    op_val   = float(np.sqrt(max(0.0, sum(phi2_exp) / N)))
    return {
        "phi_expectation":  phi_exp,
        "phi2_expectation": phi2_exp,
        "occupation":       n_exp,
        "order_parameter":  op_val,
        "phi_mean":         float(np.mean(phi_exp)),
    }

  # ── Marginal field distribution p(φ₀) at site 0 ───────────────────────────
  def field_distribution(psi: "np.ndarray", n_bins: int = 60) -> "tuple":
    """Return (phi_vals, probs) for the marginal distribution at site 0."""
    # psi has shape (d^N,). Reshape to (d, d^{N-1}).
    d_rest = d ** (N - 1)
    psi_r = psi.reshape(d, d_rest)  # index 0 = site 0 Fock level
    # Marginal amplitudes in the φ basis: ψ_φ(φ₀) = Σ_{n₀} ψ_{n₀} φ_{n₀}(φ₀).
    # Use a discrete grid for φ and evaluate in the harmonic oscillator basis.
    phi_max = 4.0 * np.sqrt(float(d))
    phi_vals = np.linspace(-phi_max, phi_max, n_bins)
    # Harmonic oscillator wave functions ψ_n(φ) = H_n(φ) e^{-φ²/2} / normalization.
    # We use the recurrence relation for Hermite polynomials.
    # ψ_0(φ) = π^{-1/4} exp(-φ²/2)
    # ψ_n(φ) = (a†)^n / √(n!) ψ_0
    # Numerically stable via recurrence: ψ_n = (√2 φ ψ_{n-1} - √(n-1) ψ_{n-2}) / √n
    psi_phi = np.zeros((d, n_bins), dtype=float)
    psi_phi[0] = np.pi ** (-0.25) * np.exp(-0.5 * phi_vals ** 2)
    if d > 1:
      psi_phi[1] = np.sqrt(2.0) * phi_vals * psi_phi[0]
    for n in range(2, d):
      psi_phi[n] = (np.sqrt(2.0) * phi_vals * psi_phi[n-1] -
                    np.sqrt(float(n-1)) * psi_phi[n-2]) / np.sqrt(float(n))
    # Marginal density: ρ(φ₀) = Σ_{rest} |Σ_n0 ψ_{n₀,rest} ψ_{HO,n₀}(φ₀)|²
    # = |Σ_n0 c_{n0} ψ_n(φ₀)|² where c_{n₀} = Σ_{rest} ψ_{n₀,rest}
    # Reduce over rest sites: c_{n₀} = √(sum_{rest} |ψ_{n₀,rest}|²) × phase
    # For probability: ρ(φ₀) = Σ_n₀ Σ_m₀ (ρ_{n₀m₀}) ψ_n(φ₀) ψ_m(φ₀)
    # where ρ_{n₀m₀} = Σ_{rest} ψ*_{n₀,rest} ψ_{m₀,rest}  (reduced density matrix)
    rho_local = psi_r @ psi_r.T   # (d, d) reduced density matrix (real for real psi)
    prob = np.einsum("ij,ix,jx->x", rho_local, psi_phi, psi_phi)
    prob = np.real(prob)
    prob = np.clip(prob, 0.0, None)
    # Normalise (approximate: discrete grid, may not integrate to 1 perfectly).
    dv = phi_vals[1] - phi_vals[0]
    prob /= max(prob.sum() * dv, 1e-12)
    return phi_vals.tolist(), prob.tolist()

  # ── Main computation ───────────────────────────────────────────────────────
  vals, vecs = solve(m2)
  E0, E1, E2 = float(vals[0]), float(vals[1]), float(vals[2])
  psi0 = vecs[:, 0]
  obs  = all_site_obs(psi0)
  phi_vals, probs = field_distribution(psi0)

  # ── Phase transition sweep ─────────────────────────────────────────────────
  sweep = []
  if scenario == "sweep" and n_sweep > 0:
    for i in range(n_sweep):
      m2_i = m2_max - (m2_max - m2_min) * i / max(1, n_sweep - 1)
      v_i, vec_i = solve(m2_i)
      psi_i = vec_i[:, 0]
      obs_i = all_site_obs(psi_i)
      sweep.append({
          "m2":             m2_i,
          "ground_energy":  float(v_i[0]),
          "order_parameter": obs_i["order_parameter"],
          "energy_gap":     float(v_i[1] - v_i[0]),
          "phi_mean":       obs_i["phi_mean"],
      })

  # ── Classical potential for visualisation ─────────────────────────────────
  phi_cl = np.linspace(-4.0, 4.0, 200)
  V_cl   = (0.5 * m2 * phi_cl ** 2 + 0.25 * lam * phi_cl ** 4).tolist()
  phi_cl = phi_cl.tolist()

  return {
      "n_sites":           N,
      "d_local":           d,
      "m2":                m2,
      "lam":               lam,
      "kappa":             kappa,
      "scenario":          scenario,
      "ground_energy":     E0,
      "first_excited":     E1,
      "second_excited":    E2,
      "energy_gap":        E1 - E0,
      "order_parameter":   obs["order_parameter"],
      "phi_expectation":   obs["phi_expectation"],
      "phi2_expectation":  obs["phi2_expectation"],
      "occupation":        obs["occupation"],
      "phi_mean":          obs["phi_mean"],
      "field_phi_vals":    phi_vals,
      "field_probs":       probs,
      "classical_phi":     phi_cl,
      "classical_V":       V_cl,
      "sweep":             sweep,
  }


def _phi4_page(data: dict, title: str = "φ⁴ QFT") -> str:
  """Generate a self-contained HTML page for the φ⁴ QFT results."""
  import json as _json

  data_js = _json.dumps(data, ensure_ascii=False)
  m2_val  = data.get("m2", 1.0)
  scenario = data.get("scenario", "symmetric")

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --accent:#58a6ff;
           --green:#3fb950; --orange:#d29922; --red:#f85149; --purple:#bc8cff;
           --yellow:#e3b341; --muted:#8b949e; --text:#e6edf3; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
          min-height:100vh; padding:24px; }}
  h1 {{ font-size:1.4rem; font-weight:600; color:var(--accent); margin-bottom:4px; }}
  .subtitle {{ color:var(--muted); font-size:.85rem; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; max-width:1100px; margin:0 auto; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; }}
  .panel.wide {{ grid-column:span 2; }}
  .panel h2 {{ font-size:.95rem; color:var(--muted); margin-bottom:12px; font-weight:500; }}
  canvas {{ display:block; width:100%; }}
  .stat-row {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px; max-width:1100px;
               margin-left:auto; margin-right:auto; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:8px 14px; }}
  .stat-label {{ font-size:.75rem; color:var(--muted); margin-bottom:2px; }}
  .stat-value {{ font-size:1.05rem; font-weight:600; color:var(--accent); }}
  .tag {{ display:inline-block; font-size:.75rem; padding:2px 8px; border-radius:4px;
          background:#21262d; border:1px solid var(--border); color:var(--muted);
          margin-right:6px; margin-bottom:12px; }}
  .phase-badge {{ display:inline-block; font-size:.8rem; padding:3px 10px; border-radius:12px;
                  font-weight:600; margin-bottom:16px; }}
  .phase-sym {{ background:rgba(88,166,255,.15); border:1px solid #58a6ff; color:#58a6ff; }}
  .phase-brk {{ background:rgba(240,81,73,.15); border:1px solid #f85149; color:#f85149; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">φ⁴ scalar field theory on a 1D lattice · truncated Fock space · exact diagonalization</p>
<div>
  <span class="tag">N={data.get('n_sites',4)} sites</span>
  <span class="tag">d={data.get('d_local',10)} levels</span>
  <span class="tag">m²={data.get('m2',1.0)}</span>
  <span class="tag">λ={data.get('lam',1.0)}</span>
  <span class="tag">κ={data.get('kappa',1.0)}</span>
</div>
<div id="phaseBadge"></div>

<div class="stat-row" id="stats"></div>

<div class="grid">
  <div class="panel">
    <h2>Classical Potential V(φ) = ½m²φ² + ¼λφ⁴</h2>
    <canvas id="cvPot" height="230"></canvas>
  </div>
  <div class="panel">
    <h2>Ground State Field Distribution p(φ₀)</h2>
    <canvas id="cvDist" height="230"></canvas>
  </div>
  <div class="panel">
    <h2>Site Profile: ⟨φᵢ⟩ and √⟨φᵢ²⟩ per Lattice Site</h2>
    <canvas id="cvSite" height="230"></canvas>
  </div>
  <div class="panel">
    <h2>Mean Occupation ⟨nᵢ⟩ per Site (quantum fluctuations)</h2>
    <canvas id="cvOcc" height="230"></canvas>
  </div>
  <div class="panel wide" id="sweepPanel" style="display:none">
    <h2>Phase Transition: Order Parameter √⟨φ²⟩ and Energy Gap Δ vs m²</h2>
    <canvas id="cvSweep" height="240"></canvas>
  </div>
</div>

<script>
const DATA = {data_js};

// ── Stats ─────────────────────────────────────────────────────────────────────
(function() {{
  const s = document.getElementById('stats');
  function stat(label, val, color) {{
    const d = document.createElement('div');
    d.className = 'stat';
    d.innerHTML = `<div class="stat-label">${{label}}</div>
      <div class="stat-value" style="${{color ? 'color:'+color : ''}}">${{val}}</div>`;
    s.appendChild(d);
  }}
  stat('Ground energy E₀', DATA.ground_energy.toFixed(5));
  stat('First excited E₁', DATA.first_excited.toFixed(5));
  stat('Energy gap Δ', DATA.energy_gap.toFixed(5),
       DATA.energy_gap < 0.05 ? '#f85149' : '#3fb950');
  stat('Order parameter √⟨φ²⟩', DATA.order_parameter.toFixed(5),
       DATA.order_parameter > 0.3 ? '#f85149' : '#58a6ff');
  stat('⟨φ⟩ (mean)', DATA.phi_mean.toFixed(5));

  // Phase badge.
  const badge = document.getElementById('phaseBadge');
  const broken = DATA.order_parameter > 0.3 && DATA.m2 < 0;
  badge.innerHTML = broken
    ? '<span class="phase-badge phase-brk">⚠ Broken Phase — Z₂ symmetry spontaneously broken</span>'
    : '<span class="phase-badge phase-sym">✓ Symmetric Phase — Z₂ preserved</span>';
}})();

// ── Canvas helpers ─────────────────────────────────────────────────────────────
function setup(id, h) {{
  const cv = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  cv.width  = (cv.offsetWidth  || 500) * dpr;
  cv.height = (cv.offsetHeight || h  ) * dpr;
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.fillStyle = '#161b22'; ctx.fillRect(0, 0, cv.offsetWidth||500, cv.offsetHeight||h);
  return {{ctx, W: cv.offsetWidth||500, H: cv.offsetHeight||h}};
}}

function axes(ctx, pad, W, H) {{
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H-pad.b);
  ctx.lineTo(W-pad.r, H-pad.b); ctx.stroke();
}}

function grid(ctx, pad, W, H) {{
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const y = pad.t + (H-pad.t-pad.b) * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W-pad.r, y); ctx.stroke();
  }}
}}

// ── Classical potential ────────────────────────────────────────────────────────
function drawPot() {{
  const {{ctx, W, H}} = setup('cvPot', 230);
  const pad = {{l:52, r:12, t:16, b:28}};
  const xs  = DATA.classical_phi;
  const ys  = DATA.classical_V;
  if (!xs || !xs.length) return;
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  // Clip to show interesting part of potential.
  const ymax_vis = Math.min(Math.max(...ys), 8.0);
  const ymin_vis = Math.min(...ys.filter(y=>y<=ymax_vis), 0);
  function xOf(v) {{ return pad.l + (v - xs[0]) / (xs[xs.length-1]-xs[0]) * cw; }}
  function yOf(v) {{ return H - pad.b - (Math.min(v,ymax_vis) - ymin_vis) / (ymax_vis - ymin_vis + 1e-9) * ch; }}

  // Fill under curve.
  ctx.fillStyle = 'rgba(88,166,255,0.07)';
  ctx.beginPath();
  ctx.moveTo(xOf(xs[0]), H - pad.b);
  for (let i = 0; i < xs.length; i++) ctx.lineTo(xOf(xs[i]), yOf(ys[i]));
  ctx.lineTo(xOf(xs[xs.length-1]), H - pad.b); ctx.closePath(); ctx.fill();

  // Curve.
  ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 2; ctx.beginPath();
  let first = true;
  for (let i = 0; i < xs.length; i++) {{
    const y = ys[i];
    if (y > ymax_vis + 0.5) {{ first = true; continue; }}
    if (first) {{ ctx.moveTo(xOf(xs[i]), yOf(y)); first = false; }}
    else ctx.lineTo(xOf(xs[i]), yOf(y));
  }}
  ctx.stroke();

  // Zero line.
  const y0 = yOf(0);
  if (y0 >= pad.t && y0 <= H - pad.b) {{
    ctx.strokeStyle = '#30363d'; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(pad.l, y0); ctx.lineTo(W-pad.r, y0); ctx.stroke();
  }}

  // Minima markers.
  const m2 = DATA.m2, lam = DATA.lam;
  if (m2 < 0 && lam > 0) {{
    const phi_min = Math.sqrt(-m2 / (2*lam));  // classical minimum ±φ* = ±√(-m²/2λ)
    [-phi_min, phi_min].forEach(pm => {{
      if (Math.abs(pm) <= Math.abs(xs[xs.length-1])) {{
        const V_min = 0.5*m2*pm*pm + 0.25*lam*pm*pm*pm*pm;
        ctx.fillStyle = '#f85149';
        ctx.beginPath(); ctx.arc(xOf(pm), yOf(V_min), 5, 0, 2*Math.PI); ctx.fill();
      }}
    }});
    ctx.fillStyle = '#f85149'; ctx.font = '10px system-ui'; ctx.textAlign = 'left';
    ctx.fillText('±φ* = ±' + phi_min.toFixed(2), pad.l+4, pad.t+12);
  }}

  axes(ctx, pad, W, H);
  // x axis labels.
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  [-3,-2,-1,0,1,2,3].forEach(v => {{
    if (v >= xs[0] && v <= xs[xs.length-1])
      ctx.fillText(v, xOf(v), H-pad.b+14);
  }});
  ctx.fillText('φ', pad.l+cw/2, H-4);
  // y labels.
  ctx.textAlign = 'right';
  [ymin_vis, (ymin_vis+ymax_vis)/2, ymax_vis].forEach(v => {{
    ctx.fillText(v.toFixed(2), pad.l-4, yOf(v)+3);
  }});
}}

// ── Field distribution p(φ₀) ──────────────────────────────────────────────────
function drawDist() {{
  const {{ctx, W, H}} = setup('cvDist', 230);
  const pad = {{l:52, r:12, t:16, b:28}};
  const xs = DATA.field_phi_vals;
  const ps = DATA.field_probs;
  if (!xs || !xs.length) return;
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  const maxP = Math.max(...ps, 1e-9);
  function xOf(v) {{ return pad.l + (v-xs[0])/(xs[xs.length-1]-xs[0])*cw; }}
  function yOf(p) {{ return H - pad.b - (p/maxP)*ch; }}

  // Fill.
  ctx.fillStyle = 'rgba(188,140,255,0.15)';
  ctx.beginPath(); ctx.moveTo(xOf(xs[0]), H-pad.b);
  for (let i=0;i<xs.length;i++) ctx.lineTo(xOf(xs[i]), yOf(ps[i]));
  ctx.lineTo(xOf(xs[xs.length-1]), H-pad.b); ctx.closePath(); ctx.fill();

  // Line.
  ctx.strokeStyle = '#bc8cff'; ctx.lineWidth = 2; ctx.beginPath();
  for (let i=0;i<xs.length;i++) {{
    if(i===0) ctx.moveTo(xOf(xs[i]),yOf(ps[i]));
    else ctx.lineTo(xOf(xs[i]),yOf(ps[i]));
  }}
  ctx.stroke();

  // Classical minima.
  const m2=DATA.m2, lam=DATA.lam;
  if (m2<0 && lam>0) {{
    const pm=Math.sqrt(-m2/(2*lam));
    [pm,-pm].forEach(v => {{
      if (v>=xs[0]&&v<=xs[xs.length-1]) {{
        ctx.strokeStyle='#f85149'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
        ctx.beginPath(); ctx.moveTo(xOf(v),pad.t); ctx.lineTo(xOf(v),H-pad.b); ctx.stroke();
        ctx.setLineDash([]);
      }}
    }});
  }}

  axes(ctx, pad, W, H);
  ctx.fillStyle='#8b949e'; ctx.font='10px system-ui'; ctx.textAlign='center';
  [-3,-2,-1,0,1,2,3].forEach(v => {{
    if(v>=xs[0]&&v<=xs[xs.length-1]) ctx.fillText(v, xOf(v), H-pad.b+14);
  }});
  ctx.fillText('φ₀', pad.l+cw/2, H-4);
  ctx.textAlign='right';
  [0, maxP/2, maxP].forEach(v => ctx.fillText(v.toFixed(3), pad.l-4, yOf(v)+3));
  ctx.fillStyle='#bc8cff'; ctx.textAlign='left';
  ctx.fillText('p(φ₀) = ⟨φ₀|ρ_local|φ₀⟩', pad.l+4, pad.t+12);
}}

// ── Site profile ───────────────────────────────────────────────────────────────
function drawSite() {{
  const {{ctx, W, H}} = setup('cvSite', 230);
  const pad = {{l:52, r:12, t:16, b:28}};
  const phi  = DATA.phi_expectation  || [];
  const phi2 = DATA.phi2_expectation || [];
  const N = phi.length;
  if (!N) return;
  const cw = W-pad.l-pad.r, ch = H-pad.t-pad.b;
  const maxV = Math.max(...phi.map(Math.abs), ...phi2.map(Math.sqrt), 0.05);
  function xOf(i) {{ return pad.l + (i+0.5)*(cw/N); }}
  function yOf(v) {{ return H-pad.b-(v+maxV)/(2*maxV)*ch; }}

  // Zero line.
  ctx.strokeStyle='#30363d'; ctx.lineWidth=0.8;
  ctx.beginPath(); ctx.moveTo(pad.l,yOf(0)); ctx.lineTo(W-pad.r,yOf(0)); ctx.stroke();

  axes(ctx, pad, W, H); grid(ctx, pad, W, H);

  const bw = Math.max(4, cw/N*0.35);
  for (let i=0;i<N;i++) {{
    const x=xOf(i);
    // √⟨φ²⟩ bar (background).
    const sq=Math.sqrt(phi2[i]||0);
    ctx.fillStyle='rgba(88,166,255,0.3)';
    const bh_sq=sq/(2*maxV)*ch;
    ctx.fillRect(x-bw*0.8, H-pad.b-bh_sq, bw*1.6, bh_sq);
    // ⟨φ⟩ bar (foreground).
    const v=phi[i]||0;
    ctx.fillStyle=v>=0?'#3fb950':'#f85149';
    const bh=Math.abs(v)/(2*maxV)*ch;
    ctx.fillRect(x-bw/2, v>=0?H-pad.b-bh:H-pad.b, bw, bh);
    // Label.
    ctx.fillStyle='#8b949e'; ctx.font='9px system-ui'; ctx.textAlign='center';
    ctx.fillText(i, x, H-pad.b+12);
  }}
  ctx.textAlign='right'; ctx.fillStyle='#8b949e'; ctx.font='10px system-ui';
  [-maxV,0,maxV].forEach(v => ctx.fillText(v.toFixed(2), pad.l-4, yOf(v)+3));
  ctx.textAlign='left';
  ctx.fillStyle='rgba(88,166,255,0.6)'; ctx.fillRect(pad.l+4, pad.t+2, 10, 8);
  ctx.fillStyle='#8b949e'; ctx.fillText(' √⟨φ²⟩', pad.l+16, pad.t+10);
  ctx.fillStyle='#3fb950'; ctx.fillRect(pad.l+4, pad.t+14, 10, 8);
  ctx.fillText(' ⟨φ⟩', pad.l+16, pad.t+22);
}}

// ── Occupation ─────────────────────────────────────────────────────────────────
function drawOcc() {{
  const {{ctx, W, H}} = setup('cvOcc', 230);
  const pad = {{l:52, r:12, t:16, b:28}};
  const occ = DATA.occupation || [];
  const N = occ.length;
  if (!N) return;
  const cw = W-pad.l-pad.r, ch = H-pad.t-pad.b;
  const maxV = Math.max(...occ, 0.1);
  function xOf(i) {{ return pad.l+(i+0.5)*(cw/N); }}
  function yOf(v) {{ return H-pad.b-v/maxV*ch; }}
  axes(ctx, pad, W, H); grid(ctx, pad, W, H);
  const bw = Math.max(4, cw/N*0.5);
  for (let i=0;i<N;i++) {{
    const x=xOf(i);
    const bh=(occ[i]||0)/maxV*ch;
    ctx.fillStyle='#d29922';
    ctx.fillRect(x-bw/2, H-pad.b-bh, bw, bh);
    ctx.fillStyle='#8b949e'; ctx.font='9px system-ui'; ctx.textAlign='center';
    ctx.fillText(i, x, H-pad.b+12);
    ctx.fillStyle='#e6edf3'; ctx.font='9px system-ui';
    ctx.fillText((occ[i]||0).toFixed(2), x, H-pad.b-bh-4);
  }}
  ctx.textAlign='right'; ctx.fillStyle='#8b949e'; ctx.font='10px system-ui';
  [0,maxV/2,maxV].forEach(v=>ctx.fillText(v.toFixed(2),pad.l-4,yOf(v)+3));
  ctx.fillStyle='#d29922'; ctx.textAlign='left'; ctx.font='10px system-ui';
  ctx.fillText('⟨n̂ᵢ⟩ (quanta per site)', pad.l+4, pad.t+12);
}}

// ── Phase transition sweep ─────────────────────────────────────────────────────
function drawSweep() {{
  const sweep = DATA.sweep || [];
  if (!sweep.length) return;
  document.getElementById('sweepPanel').style.display='';
  const {{ctx, W, H}} = setup('cvSweep', 240);
  const pad = {{l:56, r:60, t:20, b:36}};
  const m2s = sweep.map(s=>s.m2);
  const ops  = sweep.map(s=>s.order_parameter);
  const gaps = sweep.map(s=>s.energy_gap);
  const minM=Math.min(...m2s), maxM=Math.max(...m2s);
  const maxOP=Math.max(...ops, 0.01), maxGap=Math.max(...gaps, 0.01);
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  function xOf(m) {{ return pad.l+(m-minM)/(maxM-minM+1e-9)*cw; }}
  function yOfOP(v) {{ return H-pad.b-v/maxOP*ch; }}
  function yOfG(v)  {{ return H-pad.b-v/maxGap*ch; }}

  axes(ctx, pad, W, H); grid(ctx, pad, W, H);

  // Vertical line at m²=0.
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(xOf(0),pad.t); ctx.lineTo(xOf(0),H-pad.b); ctx.stroke();

  // Order parameter (left axis, blue).
  ctx.strokeStyle='#58a6ff'; ctx.lineWidth=2.5; ctx.beginPath();
  for (let i=0;i<m2s.length;i++) {{
    if(i===0) ctx.moveTo(xOf(m2s[i]),yOfOP(ops[i]));
    else ctx.lineTo(xOf(m2s[i]),yOfOP(ops[i]));
  }}
  ctx.stroke();
  // Dots.
  ops.forEach((op,i) => {{
    ctx.fillStyle='#58a6ff';
    ctx.beginPath(); ctx.arc(xOf(m2s[i]),yOfOP(op),3,0,2*Math.PI); ctx.fill();
  }});

  // Energy gap (right axis, orange).
  ctx.strokeStyle='#d29922'; ctx.lineWidth=2; ctx.beginPath();
  for (let i=0;i<m2s.length;i++) {{
    if(i===0) ctx.moveTo(xOf(m2s[i]),yOfG(gaps[i]));
    else ctx.lineTo(xOf(m2s[i]),yOfG(gaps[i]));
  }}
  ctx.stroke();

  // x axis labels.
  ctx.fillStyle='#8b949e'; ctx.font='10px system-ui'; ctx.textAlign='center';
  [-3,-2,-1,0,1,2].forEach(v => {{
    if(v>=minM&&v<=maxM) ctx.fillText(v.toFixed(0),xOf(v),H-pad.b+14);
  }});
  ctx.fillText('bare mass squared m²', pad.l+cw/2, H-4);

  // Left y labels (order parameter).
  ctx.fillStyle='#58a6ff'; ctx.textAlign='right';
  [0,maxOP/2,maxOP].forEach(v=>ctx.fillText(v.toFixed(2),pad.l-4,yOfOP(v)+3));
  ctx.save(); ctx.translate(14,pad.t+ch/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign='center'; ctx.fillText('√⟨φ²⟩ (order param)', 0, 0); ctx.restore();

  // Right y labels (gap).
  ctx.fillStyle='#d29922'; ctx.textAlign='left';
  [0,maxGap/2,maxGap].forEach(v=>ctx.fillText(v.toFixed(2),W-pad.r+4,yOfG(v)+3));
  ctx.save(); ctx.translate(W-12,pad.t+ch/2); ctx.rotate(Math.PI/2);
  ctx.textAlign='center'; ctx.fillText('gap Δ=E₁−E₀', 0, 0); ctx.restore();

  // Legend.
  ctx.textAlign='left'; ctx.font='11px system-ui';
  ctx.fillStyle='#58a6ff'; ctx.fillText('order param √⟨φ²⟩', pad.l+8, pad.t+14);
  ctx.fillStyle='#d29922'; ctx.fillText('energy gap Δ', pad.l+8, pad.t+28);
  ctx.fillStyle='#8b949e'; ctx.font='9px system-ui';
  ctx.fillText('← symmetric', xOf(0)+4, pad.t+14);
  ctx.textAlign='right'; ctx.fillText('broken →', xOf(0)-4, pad.t+14);
}}

window.addEventListener('load', () => {{
  drawPot(); drawDist(); drawSite(); drawOcc(); drawSweep();
}});
window.addEventListener('resize', () => {{
  drawPot(); drawDist(); drawSite(); drawOcc(); drawSweep();
}});
</script>
</body>
</html>"""
  return html


def _phi4_dynamics_page(data: dict, title: str = "φ⁴ QFT — Quantum Quench Dynamics") -> str:
  """Generate a self-contained HTML page for the φ⁴ quantum quench dynamics."""
  import json as _json

  data_js = _json.dumps(data, ensure_ascii=False)
  N        = data.get("n_sites", 4)
  d        = data.get("d_local", 6)
  m2i      = data.get("m2_initial", 1.0)
  m2f      = data.get("m2_final", -1.5)
  n_steps  = data.get("n_steps", 80)
  dt_val   = data.get("dt", 0.05)
  frames   = data.get("frames", [])
  final_op = frames[-1]["order_param"] if frames else 0.0
  final_S  = frames[-1]["entropy"]     if frames else 0.0

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --accent:#36cfc9;
           --green:#3fb950; --orange:#d29922; --red:#f85149; --muted:#8b949e; --text:#e6edf3; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
          min-height:100vh; padding:24px; }}
  h1 {{ font-size:1.4rem; font-weight:600; color:var(--accent); margin-bottom:4px; }}
  .subtitle {{ color:var(--muted); font-size:.85rem; margin-bottom:16px; }}
  .tags {{ margin-bottom:16px; }}
  .tag {{ display:inline-block; font-size:.75rem; padding:2px 8px; border-radius:4px;
          background:#21262d; border:1px solid var(--border); color:var(--muted);
          margin-right:6px; margin-bottom:4px; }}
  .stat-row {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; max-width:1100px;
               margin-left:auto; margin-right:auto; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:8px 14px; }}
  .stat-label {{ font-size:.75rem; color:var(--muted); margin-bottom:2px; }}
  .stat-value {{ font-size:1.05rem; font-weight:600; color:var(--accent); }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; max-width:1100px; margin:0 auto; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; }}
  .panel h2 {{ font-size:.92rem; color:var(--muted); margin-bottom:10px; font-weight:500; }}
  canvas {{ display:block; width:100%; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">φ⁴ scalar QFT — quantum quench dynamics · exact time evolution · truncated Fock space</p>
<div class="tags">
  <span class="tag">N={N} sites</span>
  <span class="tag">d={d} levels</span>
  <span class="tag">m²_initial={m2i}</span>
  <span class="tag">m²_final={m2f}</span>
  <span class="tag">n_steps={n_steps}</span>
  <span class="tag">dt={dt_val}</span>
</div>
<div class="stat-row">
  <div class="stat"><div class="stat-label">final order param √⟨φ²⟩</div>
    <div class="stat-value">{final_op:.6f}</div></div>
  <div class="stat"><div class="stat-label">final entanglement S</div>
    <div class="stat-value">{final_S:.6f}</div></div>
  <div class="stat"><div class="stat-label">quench direction</div>
    <div class="stat-value">{"sym→broken" if m2f < 0 else "broken→sym"}</div></div>
</div>
<div class="grid">
  <div class="panel"><h2>Order Parameter √⟨φ²⟩(t)</h2><canvas id="cvOrder" height="220"></canvas></div>
  <div class="panel"><h2>Entanglement Entropy S(t)</h2><canvas id="cvEntropy" height="220"></canvas></div>
  <div class="panel"><h2>⟨φᵢ(t)⟩ Heatmap (blue=−, white=0, red=+)</h2><canvas id="cvPhiHeat" height="220"></canvas></div>
  <div class="panel"><h2>⟨nᵢ(t)⟩ Heatmap (white=0, orange=high)</h2><canvas id="cvOccHeat" height="220"></canvas></div>
</div>
<script>
const DATA = {data_js};

function setup(id, h) {{
  const cv = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const W = cv.parentElement.clientWidth - 32;
  const H = h;
  cv.width  = W * dpr;
  cv.height = H * dpr;
  cv.style.width  = W + 'px';
  cv.style.height = H + 'px';
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  return {{ctx, W, H}};
}}

function axes(ctx, pad, W, H) {{
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H-pad.b);
  ctx.lineTo(W-pad.r, H-pad.b);
  ctx.stroke();
}}

function grid(ctx, pad, W, H, nv, nh) {{
  ctx.strokeStyle='#21262d'; ctx.lineWidth=0.5;
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  for (let i=1; i<=(nh||4); i++) {{
    const y=pad.t+i*ch/(nh||4);
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.stroke();
  }}
  for (let i=1; i<=(nv||5); i++) {{
    const x=pad.l+i*cw/(nv||5);
    ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,H-pad.b); ctx.stroke();
  }}
}}

function drawLineChart(id, h, ys, color, labelY, labelX) {{
  const {{ctx, W, H}} = setup(id, h);
  const pad = {{l:52, r:12, t:16, b:32}};
  if (!ys.length) return;
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const n=ys.length;
  const minY=Math.min(...ys), maxY=Math.max(...ys);
  const rng=Math.max(maxY-minY, 1e-9);
  const pad_y=rng*0.08;
  const lo=minY-pad_y, hi=maxY+pad_y;
  function xOf(i) {{ return pad.l+i/(n-1||1)*cw; }}
  function yOf(v) {{ return H-pad.b-(v-lo)/(hi-lo)*ch; }}

  axes(ctx, pad, W, H); grid(ctx, pad, W, H, 5, 4);

  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
  for (let i=0;i<n;i++) {{
    if(i===0) ctx.moveTo(xOf(i),yOf(ys[i]));
    else ctx.lineTo(xOf(i),yOf(ys[i]));
  }}
  ctx.stroke();

  // Axis labels
  ctx.fillStyle='#8b949e'; ctx.font='10px system-ui'; ctx.textAlign='center';
  const frames=DATA.frames||[];
  const tMax=frames.length?frames[frames.length-1].time:n*DATA.dt;
  [0,0.25,0.5,0.75,1.0].forEach(f=>{{
    const t=(tMax*f).toFixed(1);
    ctx.fillText(t, pad.l+f*cw, H-pad.b+14);
  }});
  ctx.fillText(labelX||'time', pad.l+cw/2, H-2);
  ctx.textAlign='right'; ctx.font='9px system-ui';
  [lo,(lo+hi)/2,hi].forEach(v=>ctx.fillText(v.toFixed(3),pad.l-4,yOf(v)+3));
  ctx.save(); ctx.translate(12,pad.t+ch/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign='center'; ctx.fillStyle=color; ctx.font='10px system-ui';
  ctx.fillText(labelY||'', 0, 0); ctx.restore();
}}

function lerpColor(t, c0, c1) {{
  // t in [0,1], c0/c1 as [r,g,b]
  return [
    Math.round(c0[0]+(c1[0]-c0[0])*t),
    Math.round(c0[1]+(c1[1]-c0[1])*t),
    Math.round(c0[2]+(c1[2]-c0[2])*t),
  ];
}}

function drawHeatmap(id, h, getData, colorFn) {{
  const {{ctx, W, H}} = setup(id, h);
  const pad = {{l:36, r:12, t:16, b:28}};
  const frames=DATA.frames||[];
  const N=DATA.n_sites||4;
  if (!frames.length) return;
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const nT=frames.length, nS=N;
  const cellW=cw/nT, cellH=ch/nS;
  for (let ti=0;ti<nT;ti++) {{
    const row=getData(frames[ti]);
    for (let si=0;si<nS;si++) {{
      const v=row[si]||0;
      ctx.fillStyle=colorFn(v);
      ctx.fillRect(pad.l+ti*cellW, pad.t+si*cellH, Math.ceil(cellW)+1, Math.ceil(cellH)+1);
    }}
  }}
  // Axes
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath();
  ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.lineTo(W-pad.r,H-pad.b);
  ctx.stroke();
  // Labels
  ctx.fillStyle='#8b949e'; ctx.font='9px system-ui'; ctx.textAlign='center';
  const tMax=frames[frames.length-1].time;
  [0,0.5,1.0].forEach(f=>{{
    ctx.fillText((tMax*f).toFixed(1), pad.l+f*cw, H-pad.b+12);
  }});
  ctx.fillText('time', pad.l+cw/2, H-2);
  ctx.textAlign='right';
  for (let si=0;si<nS;si++) {{
    ctx.fillText(si, pad.l-3, pad.t+(si+0.6)*cellH);
  }}
}}

function phiColor(v) {{
  // blue (#1e90ff) → white → red (#ff4500)
  const mx=0.5;
  const t=Math.min(Math.abs(v)/mx, 1.0);
  if (v<0) {{
    const c=lerpColor(t,[255,255,255],[30,144,255]);
    return `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;
  }} else {{
    const c=lerpColor(t,[255,255,255],[255,69,0]);
    return `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;
  }}
}}

function occColor(v) {{
  // white → orange (#ff8c00)
  const mx=3.0;
  const t=Math.min(Math.max(v,0)/mx, 1.0);
  const c=lerpColor(t,[255,255,255],[255,140,0]);
  return `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;
}}

window.addEventListener('load', () => {{
  const frames=DATA.frames||[];
  const orders  = frames.map(f=>f.order_param);
  const entropies = frames.map(f=>f.entropy);
  drawLineChart('cvOrder',  220, orders,   '#36cfc9', '√⟨φ²⟩', 'time');
  drawLineChart('cvEntropy',220, entropies,'#3fb950', 'S_ent',  'time');
  drawHeatmap('cvPhiHeat', 220, f=>f.phi_expect, phiColor);
  drawHeatmap('cvOccHeat', 220, f=>f.occupation, occColor);
}});
window.addEventListener('resize', () => {{
  const frames=DATA.frames||[];
  const orders    = frames.map(f=>f.order_param);
  const entropies = frames.map(f=>f.entropy);
  drawLineChart('cvOrder',  220, orders,   '#36cfc9', '√⟨φ²⟩', 'time');
  drawLineChart('cvEntropy',220, entropies,'#3fb950', 'S_ent',  'time');
  drawHeatmap('cvPhiHeat', 220, f=>f.phi_expect, phiColor);
  drawHeatmap('cvOccHeat', 220, f=>f.occupation, occColor);
}});
</script>
</body>
</html>"""
  return html


# ── M23: Schwinger Model (1+1D QED) Lattice Quantum Field Theory ──────────────

def _schwinger_frames(
    n_sites: int = 8,
    mass: float = 0.5,
    coupling: float = 0.3,
    background: float = 0.0,
    scenario: str = "vacuum",
    dt: float = 0.08,
    n_steps: int = 60,
    n_mass_sweep: int = 30,
    seed: int = 42,
) -> dict:
  """Schwinger model (1+1D QED) via exact diagonalization.

  Kogut-Susskind staggered Hamiltonian (gauge field integrated out, open BCs):

    H = -½ Σ_n (σ+_n σ-_{n+1} + h.c.)        [hopping / kinetic]
      + (m/2) Σ_n (-1)^n σz_n                 [staggered mass]
      + (g²/2) Σ_{l=0}^{N-2} L_l²             [electric field energy]

  where L_l = background + Σ_{k=0}^{l} (σz_k + (-1)^k)/2  via Gauss's law.

  Scenarios:
    "vacuum"    — ground state observables + mass sweep (chiral condensate)
    "string"    — string tension: E(R) vs quark separation R
    "dynamics"  — time evolution from a pair-production initial state under
                  sudden quench of background electric field
  """
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("Schwinger model visualization requires numpy") from ex

  rng = np.random.default_rng(seed)
  N = n_sites
  dim = 1 << N

  # ── Gate primitives ──────────────────────────────────────────────────────
  sz = np.array([[1, 0], [0, -1]], dtype=complex)
  sp = np.array([[0, 1], [0, 0]], dtype=complex)
  sm = np.array([[0, 0], [1, 0]], dtype=complex)
  I2 = np.eye(2, dtype=complex)

  def kron_op(op: "np.ndarray", site: int) -> "np.ndarray":
    ops = [I2] * N
    ops[site] = op
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  def kron_2site(op1: "np.ndarray", op2: "np.ndarray", s1: int, s2: int) -> "np.ndarray":
    ops = [I2] * N
    ops[s1] = op1
    ops[s2] = op2
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  # ── Hamiltonian builder ───────────────────────────────────────────────────
  def build_H(m: float, g2: float, alpha: float) -> "np.ndarray":
    H = np.zeros((dim, dim), dtype=complex)
    # Hopping: -½ Σ (σ+_n σ-_{n+1} + h.c.)
    for n in range(N - 1):
      H -= 0.5 * (kron_2site(sp, sm, n, n + 1) + kron_2site(sm, sp, n, n + 1))
    # Staggered mass: (m/2) Σ (-1)^n σz_n
    for n in range(N):
      H += (m / 2.0) * ((-1) ** n) * kron_op(sz, n)
    # Electric field energy: (g²/2) Σ_{l} L_l²
    # L_l = alpha*I + Σ_{k≤l} (σz_k + (-1)^k) / 2
    for l in range(N - 1):
      # Constant part of L_l.
      c_l = alpha + sum(0.5 * ((-1) ** k) for k in range(l + 1))
      # Operator part of L_l.
      L_op = c_l * np.eye(dim, dtype=complex)
      for k in range(l + 1):
        L_op += 0.5 * kron_op(sz, k)
      H += (g2 / 2.0) * (L_op @ L_op)
    return H

  # ── Ground state via numpy eigh (exact for small N) ──────────────────────
  def ground_state(H: "np.ndarray") -> "tuple[float, np.ndarray]":
    # For dim ≤ 4096 numpy eigh is fast enough.
    vals, vecs = np.linalg.eigh(H)
    return float(np.real(vals[0])), vecs[:, 0]

  # ── Observable extraction ─────────────────────────────────────────────────
  def observables(psi: "np.ndarray", alpha: float) -> dict:
    prob = np.abs(psi) ** 2
    # <σz_n> per site.
    sz_exp = np.zeros(N)
    for n in range(N):
      Sz_n = kron_op(sz, n)
      sz_exp[n] = float(np.real(psi.conj() @ Sz_n @ psi))
    # Charge density Q_n = (σz_n + (-1)^n) / 2.
    charge = np.array([(sz_exp[n] + (-1) ** n) / 2.0 for n in range(N)])
    # Electric field per link: <L_l> = alpha + Σ_{k≤l} <Q_k>.
    elec = np.zeros(N - 1)
    cum = 0.0
    for l in range(N - 1):
      cum += charge[l]
      elec[l] = alpha + cum
    # Chiral condensate: Σ = -(1/N) Σ (-1)^n <σz_n>.
    chiral = -float(np.sum([(-1) ** n * sz_exp[n] for n in range(N)])) / N
    particle_number = float(np.sum(np.maximum(charge, 0.0)))
    return {
        "charge_density": charge.tolist(),
        "electric_field": elec.tolist(),
        "chiral_condensate": chiral,
        "particle_number": particle_number,
    }

  # ── Main scenario dispatch ────────────────────────────────────────────────
  # In dynamics mode, prepare |psi0> in a true pre-quench vacuum (alpha=0)
  # and then evolve under H(alpha_quench). Non-dynamics scenarios continue to
  # use the user-provided background directly.
  alpha0 = 0.0 if scenario == "dynamics" else background
  H0 = build_H(mass, coupling, alpha0)
  E0, psi0 = ground_state(H0)
  obs0 = observables(psi0, alpha0)

  # Mass sweep: chiral condensate and ground energy vs m ∈ [-2, 2].
  mass_sweep = []
  if n_mass_sweep > 0:
    m_vals = np.linspace(-2.0, 2.0, n_mass_sweep)
    for m_val in m_vals:
      Hi = build_H(float(m_val), coupling, background)
      Ei, psi_i = ground_state(Hi)
      obs_i = observables(psi_i, background)
      mass_sweep.append({
          "mass": float(m_val),
          "ground_energy": float(Ei),
          "chiral_condensate": obs_i["chiral_condensate"],
      })

  # String tension: energy vs quark–antiquark separation R.
  string_data = []
  if scenario == "string":
    # Create string state: insert a quark–antiquark pair by setting occupation
    # pattern and measure energy relative to vacuum as a function of R.
    # Physically: place a static charge +1 at site 0, -1 at site R.
    # Achieved by applying σ+_0 σ-_R |vacuum⟩ and measuring <H> - E0.
    # For the string scenario we vary R and compute E_string(R) = <psi_string|H|psi_string> - E0.
    for R in range(1, N // 2 + 1):
      # String state: apply σ+_0 σ-_R to vacuum.
      op_string = kron_2site(sp, sm, 0, R) if R < N else kron_op(sp, 0)
      psi_str = op_string @ psi0
      norm = np.linalg.norm(psi_str)
      if norm < 1e-10:
        # Vacuum already has this structure — use σ-_0 σ+_R instead.
        op_string = kron_2site(sm, sp, 0, R) if R < N else kron_op(sm, 0)
        psi_str = op_string @ psi0
        norm = np.linalg.norm(psi_str)
      if norm < 1e-10:
        continue
      psi_str /= norm
      E_str = float(np.real(psi_str.conj() @ H0 @ psi_str))
      string_data.append({"separation": R, "string_energy": E_str - E0})

  # Dynamics: time evolution from vacuum under sudden background quench.
  frames = []
  if scenario == "dynamics":
    alpha_quench = background if abs(background) > 1e-9 else 0.5
    H_quench = build_H(mass, coupling, alpha_quench)
    # Time evolution via matrix exponential (exact for small N).
    psi = psi0.copy()
    U_step = None  # Lazy: compute once.
    for step in range(n_steps + 1):
      t = step * dt
      obs = observables(psi, alpha_quench)
      E = float(np.real(psi.conj() @ H_quench @ psi))
      frames.append({
          "time": round(t, 6),
          "energy": E,
          **obs,
      })
      if step < n_steps:
        if U_step is None:
          from scipy.linalg import expm  # type: ignore
          U_step = expm(-1j * dt * H_quench)
        psi = U_step @ psi

  return {
      "n_sites": N,
      "mass": mass,
      "coupling": coupling,
      "background": background,
      "scenario": scenario,
      "ground_energy": E0,
      "ground_chiral_condensate": obs0["chiral_condensate"],
      "ground_charge_density": obs0["charge_density"],
      "ground_electric_field": obs0["electric_field"],
      "mass_sweep": mass_sweep,
      "string_data": string_data,
      "frames": frames,
  }


def _schwinger_fallback_dynamics(
    n_sites: int,
    mass: float,
    coupling: float,
    background_quench: float,
    dt: float,
    n_steps: int,
) -> "list[dict]":
  """Trotter-based fallback for dynamics when scipy is unavailable."""
  try:
    import numpy as np
  except ImportError:
    return []

  N = n_sites
  I2 = np.eye(2, dtype=complex)
  sz = np.array([[1, 0], [0, -1]], dtype=complex)
  sp = np.array([[0, 1], [0, 0]], dtype=complex)
  sm = np.array([[0, 0], [1, 0]], dtype=complex)

  def kron_op(op, site):
    ops = [I2] * N
    ops[site] = op
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  def kron_2site(op1, op2, s1, s2):
    ops = [I2] * N
    ops[s1] = op1
    ops[s2] = op2
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  # Build Hamiltonian terms as a list for first-order Trotter.
  # Hopping pairs.
  hop_terms = []
  for n in range(N - 1):
    H_hop = -0.5 * (kron_2site(sp, sm, n, n + 1) + kron_2site(sm, sp, n, n + 1))
    hop_terms.append(H_hop)
  # Diagonal terms (mass + electric).
  diag_terms = []
  for n in range(N):
    diag_terms.append((mass / 2.0) * ((-1) ** n) * kron_op(sz, n))
  for l in range(N - 1):
    c_l = background_quench + sum(0.5 * ((-1) ** k) for k in range(l + 1))
    L_op = c_l * np.eye(1 << N, dtype=complex)
    for k in range(l + 1):
      L_op += 0.5 * kron_op(sz, k)
    diag_terms.append((coupling / 2.0) * (L_op @ L_op))

  # Combine all terms.
  H_full = sum(hop_terms) + sum(diag_terms)
  # Approximate Trotter: exp(-iHdt) ≈ exp(-iH_hop dt/2) exp(-iH_diag dt) exp(-iH_hop dt/2).
  # For simplicity just use full matrix: eigendecompose.
  vals, vecs = np.linalg.eigh(H_full)
  # exp(-iHt) psi = vecs @ diag(exp(-i*vals*t)) @ vecs^† @ psi.
  # Get ground state.
  psi = vecs[:, 0].copy()
  dim = 1 << N

  frames = []
  for step in range(n_steps + 1):
    t = step * dt
    coeffs = np.exp(-1j * vals * t)
    psi_t = vecs @ (coeffs * (vecs.conj().T @ psi))
    charge = np.array(
        [(float(np.real(psi_t.conj() @ kron_op(sz, n) @ psi_t)) + (-1) ** n) / 2.0
         for n in range(N)]
    )
    elec = np.zeros(N - 1)
    cum = 0.0
    for l in range(N - 1):
      cum += charge[l]
      elec[l] = background_quench + cum
    chiral = -float(np.sum([(-1) ** n *
        float(np.real(psi_t.conj() @ kron_op(sz, n) @ psi_t))
        for n in range(N)])) / N
    E = float(np.real(psi_t.conj() @ H_full @ psi_t))
    frames.append({
        "time": round(t, 6),
        "energy": E,
        "charge_density": charge.tolist(),
        "electric_field": elec.tolist(),
        "chiral_condensate": chiral,
        "particle_number": float(np.sum(np.maximum(charge, 0.0))),
    })
  return frames


def _schwinger_page(data: dict, title: str = "Schwinger Model") -> str:
  """Generate a self-contained HTML page for Schwinger model results."""
  import json as _json

  data_js = _json.dumps(data, ensure_ascii=False)
  scenario = data.get("scenario", "vacuum")

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --accent:#58a6ff;
           --green:#3fb950; --orange:#d29922; --red:#f85149; --purple:#bc8cff;
           --muted:#8b949e; --text:#e6edf3; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
          min-height:100vh; padding:24px; }}
  h1 {{ font-size:1.4rem; font-weight:600; color:var(--accent); margin-bottom:4px; }}
  .subtitle {{ color:var(--muted); font-size:.85rem; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; max-width:1100px; margin:0 auto; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; }}
  .panel.wide {{ grid-column:span 2; }}
  .panel h2 {{ font-size:.95rem; color:var(--muted); margin-bottom:12px; font-weight:500; }}
  canvas {{ display:block; width:100%; }}
  .stat-row {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px; max-width:1100px; margin-left:auto; margin-right:auto; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:8px 14px; }}
  .stat-label {{ font-size:.75rem; color:var(--muted); margin-bottom:2px; }}
  .stat-value {{ font-size:1.05rem; font-weight:600; color:var(--accent); }}
  .tag {{ display:inline-block; font-size:.75rem; padding:2px 8px; border-radius:4px;
          background:#21262d; border:1px solid var(--border); color:var(--muted);
          margin-right:6px; margin-bottom:12px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">Kogut-Susskind staggered Hamiltonian · 1+1D QED · exact diagonalization</p>
<div>
  <span class="tag">N={data.get('n_sites',8)} sites</span>
  <span class="tag">m={data.get('mass',0.5)}</span>
  <span class="tag">g²={data.get('coupling',0.3)}</span>
  <span class="tag">α={data.get('background',0.0)}</span>
  <span class="tag">scenario={data.get('scenario','vacuum')}</span>
</div>

<div class="stat-row" id="stats"></div>

<div class="grid">
  <div class="panel">
    <h2>Ground State Charge Density Q_n = ⟨(σz_n + (−1)ⁿ)/2⟩</h2>
    <canvas id="cvCharge" height="220"></canvas>
  </div>
  <div class="panel">
    <h2>Ground State Electric Field ⟨L_l⟩ per Link</h2>
    <canvas id="cvElec" height="220"></canvas>
  </div>
  <div class="panel wide" id="sweepPanel">
    <h2>Chiral Condensate ⟨ψ̄ψ⟩ vs Bare Mass m  (order parameter)</h2>
    <canvas id="cvSweep" height="220"></canvas>
  </div>
  <div class="panel wide" id="stringPanel" style="display:none">
    <h2>String Energy E(R) − E_vac vs Quark Separation R (confinement)</h2>
    <canvas id="cvString" height="220"></canvas>
  </div>
  <div class="panel wide" id="dynPanel" style="display:none">
    <h2>Pair Production Dynamics: Particle Number and Chiral Condensate vs Time</h2>
    <canvas id="cvDyn" height="240"></canvas>
  </div>
  <div class="panel wide" id="heatPanel" style="display:none">
    <h2>Charge Density Heatmap: Q_n(t) vs Site n and Time t</h2>
    <canvas id="cvHeat" height="240"></canvas>
  </div>
</div>

<script>
const DATA = {data_js};

// ── Stats ─────────────────────────────────────────────────────────────────────
(function() {{
  const s = document.getElementById('stats');
  function stat(label, val) {{
    const d = document.createElement('div');
    d.className = 'stat';
    d.innerHTML = `<div class="stat-label">${{label}}</div><div class="stat-value">${{val}}</div>`;
    s.appendChild(d);
  }}
  stat('Ground energy', DATA.ground_energy.toFixed(5));
  stat('Chiral condensate', DATA.ground_chiral_condensate.toFixed(5));
  const ne = DATA.ground_charge_density.filter(q=>q>0.01).length;
  stat('Occupied sites', ne);
  if (DATA.string_data && DATA.string_data.length) {{
    const slope = DATA.string_data.length > 1
      ? (DATA.string_data[DATA.string_data.length-1].string_energy - DATA.string_data[0].string_energy)
        / (DATA.string_data[DATA.string_data.length-1].separation - DATA.string_data[0].separation)
      : DATA.string_data[0].string_energy;
    stat('String tension σ', slope.toFixed(4));
  }}
  if (DATA.frames && DATA.frames.length > 1) {{
    const last = DATA.frames[DATA.frames.length-1];
    stat('Final ⟨n_pairs⟩', last.particle_number.toFixed(3));
  }}
}})();

// ── Drawing helpers ────────────────────────────────────────────────────────────
function setupCanvas(id, h) {{
  const cv = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  cv.width  = (cv.offsetWidth  || 500) * dpr;
  cv.height = (cv.offsetHeight || h  ) * dpr;
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.fillStyle = '#161b22'; ctx.fillRect(0,0,cv.offsetWidth||500, cv.offsetHeight||h);
  return {{ctx, W:cv.offsetWidth||500, H:cv.offsetHeight||h}};
}}

function drawAxes(ctx, pad, W, H) {{
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H-pad.b);
  ctx.lineTo(W-pad.r, H-pad.b); ctx.stroke();
}}

function yTicks(ctx, pad, W, H, minV, maxV, nTick) {{
  const ch = H - pad.t - pad.b;
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
  for (let i = 0; i <= nTick; i++) {{
    const v = minV + (maxV - minV) * i / nTick;
    const y = H - pad.b - (v - minV) / (maxV - minV + 1e-14) * ch;
    ctx.fillText(v.toFixed(3), pad.l - 4, y + 3);
    ctx.strokeStyle = '#21262d'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W-pad.r, y); ctx.stroke();
  }}
}}

// ── Charge density bar chart ───────────────────────────────────────────────────
function drawCharge() {{
  const {{ctx, W, H}} = setupCanvas('cvCharge', 220);
  const pad = {{l:52, r:10, t:12, b:28}};
  const data = DATA.ground_charge_density;
  const N = data.length;
  const maxV = Math.max(...data.map(Math.abs), 0.05);
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  const zero_y = H - pad.b - (0 - (-maxV)) / (2*maxV) * ch;
  drawAxes(ctx, pad, W, H);
  // Zero line.
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, zero_y); ctx.lineTo(W-pad.r, zero_y); ctx.stroke();
  const bw = Math.max(2, cw / N - 1);
  for (let i = 0; i < N; i++) {{
    const x = pad.l + i * (cw / N);
    const bh = Math.abs(data[i]) / (2*maxV) * ch;
    ctx.fillStyle = data[i] >= 0 ? '#3fb950' : '#f85149';
    const y = data[i] >= 0 ? zero_y - bh : zero_y;
    ctx.fillRect(x + 1, y, bw, bh);
    // Site label.
    ctx.fillStyle = '#8b949e'; ctx.font = '9px system-ui'; ctx.textAlign = 'center';
    ctx.fillText(i, x + bw/2, H - pad.b + 12);
  }}
  // Y ticks.
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
  [-maxV, 0, maxV].forEach(v => {{
    const y = H - pad.b - (v - (-maxV)) / (2*maxV) * ch;
    ctx.fillText(v.toFixed(2), pad.l - 4, y + 3);
  }});
  // Legend.
  ctx.fillStyle = '#3fb950'; ctx.font = '10px system-ui'; ctx.textAlign = 'left';
  ctx.fillText('particle (Q>0)', pad.l + 4, pad.t + 12);
  ctx.fillStyle = '#f85149';
  ctx.fillText('antiparticle (Q<0)', pad.l + 4, pad.t + 24);
}}

// ── Electric field per link ────────────────────────────────────────────────────
function drawElec() {{
  const {{ctx, W, H}} = setupCanvas('cvElec', 220);
  const pad = {{l:52, r:10, t:12, b:28}};
  const data = DATA.ground_electric_field;
  const Nl = data.length;
  const minV = Math.min(...data, -0.05), maxV = Math.max(...data, 0.05);
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  drawAxes(ctx, pad, W, H);
  const bw = Math.max(2, cw / Nl - 2);
  for (let i = 0; i < Nl; i++) {{
    const x = pad.l + i * (cw / Nl);
    const bh = (data[i] - minV) / (maxV - minV + 1e-14) * ch;
    ctx.fillStyle = '#58a6ff';
    ctx.fillRect(x + 1, H - pad.b - bh, bw, bh);
    ctx.fillStyle = '#8b949e'; ctx.font = '9px system-ui'; ctx.textAlign = 'center';
    ctx.fillText(i + '–' + (i+1), x + bw/2, H - pad.b + 12);
  }}
  yTicks(ctx, pad, W, H, minV, maxV, 4);
}}

// ── Chiral condensate vs mass sweep ───────────────────────────────────────────
function drawSweep() {{
  const sweep = DATA.mass_sweep || [];
  if (!sweep.length) return;
  const {{ctx, W, H}} = setupCanvas('cvSweep', 220);
  const pad = {{l:56, r:16, t:16, b:32}};
  const ms = sweep.map(s=>s.mass);
  const cs = sweep.map(s=>s.chiral_condensate);
  const minC = Math.min(...cs), maxC = Math.max(...cs);
  const minM = Math.min(...ms), maxM = Math.max(...ms);
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  drawAxes(ctx, pad, W, H);
  function xOf(m) {{ return pad.l + (m - minM) / (maxM - minM) * cw; }}
  function yOf(c) {{ return H - pad.b - (c - minC) / (maxC - minC + 1e-14) * ch; }}
  // Fill area under curve.
  ctx.fillStyle = 'rgba(88,166,255,0.12)';
  ctx.beginPath(); ctx.moveTo(xOf(ms[0]), H - pad.b);
  for (let i = 0; i < ms.length; i++) ctx.lineTo(xOf(ms[i]), yOf(cs[i]));
  ctx.lineTo(xOf(ms[ms.length-1]), H - pad.b); ctx.closePath(); ctx.fill();
  // Line.
  ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i < ms.length; i++) {{
    if (i === 0) ctx.moveTo(xOf(ms[i]), yOf(cs[i]));
    else ctx.lineTo(xOf(ms[i]), yOf(cs[i]));
  }}
  ctx.stroke();
  // Current mass marker.
  const mCur = DATA.mass;
  if (mCur >= minM && mCur <= maxM) {{
    ctx.strokeStyle = '#d29922'; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
    const x = xOf(mCur);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#d29922'; ctx.font = '10px system-ui'; ctx.textAlign = 'left';
    ctx.fillText('m=' + mCur, x + 3, pad.t + 12);
  }}
  // Zero condensate line.
  const y0 = yOf(0);
  if (y0 >= pad.t && y0 <= H - pad.b) {{
    ctx.strokeStyle = '#30363d'; ctx.lineWidth = 0.5; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(pad.l, y0); ctx.lineTo(W-pad.r, y0); ctx.stroke();
  }}
  yTicks(ctx, pad, W, H, minC, maxC, 4);
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  [minM, 0, maxM].forEach(v => {{
    ctx.fillText(v.toFixed(1), xOf(v), H - pad.b + 14);
  }});
  ctx.fillText('bare mass m/J', pad.l + cw/2, H - 4);
  ctx.save(); ctx.translate(14, pad.t + ch/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign = 'center'; ctx.fillText('⟨ψ̄ψ⟩', 0, 0); ctx.restore();
}}

// ── String energy vs separation ────────────────────────────────────────────────
function drawString() {{
  const sdata = DATA.string_data || [];
  if (!sdata.length) return;
  document.getElementById('stringPanel').style.display = '';
  const {{ctx, W, H}} = setupCanvas('cvString', 220);
  const pad = {{l:56, r:16, t:16, b:32}};
  const Rs = sdata.map(s=>s.separation);
  const Es = sdata.map(s=>s.string_energy);
  const minR = Math.min(...Rs), maxR = Math.max(...Rs);
  const minE = 0, maxE = Math.max(...Es, 0.01);
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  function xOf(r) {{ return pad.l + (r - minR) / Math.max(maxR - minR, 1) * cw; }}
  function yOf(e) {{ return H - pad.b - (e - minE) / (maxE - minE + 1e-14) * ch; }}
  drawAxes(ctx, pad, W, H);
  // Linear confinement reference: E = σ·R, σ = (g²/2).
  const sigma = DATA.coupling / 2.0;
  ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 1; ctx.setLineDash([4,3]);
  ctx.beginPath();
  ctx.moveTo(xOf(minR), yOf(sigma * minR));
  ctx.lineTo(xOf(maxR), yOf(sigma * maxR));
  ctx.stroke(); ctx.setLineDash([]);
  // Actual string energies.
  ctx.strokeStyle = '#bc8cff'; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i < Rs.length; i++) {{
    if (i === 0) ctx.moveTo(xOf(Rs[i]), yOf(Es[i]));
    else ctx.lineTo(xOf(Rs[i]), yOf(Es[i]));
  }}
  ctx.stroke();
  for (let i = 0; i < Rs.length; i++) {{
    ctx.fillStyle = '#bc8cff';
    ctx.beginPath(); ctx.arc(xOf(Rs[i]), yOf(Es[i]), 4, 0, 2*Math.PI); ctx.fill();
  }}
  yTicks(ctx, pad, W, H, minE, maxE, 4);
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  Rs.forEach(r => ctx.fillText(r, xOf(r), H - pad.b + 14));
  ctx.fillText('separation R (sites)', pad.l + cw/2, H - 4);
  // Legend.
  ctx.textAlign = 'left'; ctx.font = '10px system-ui';
  ctx.fillStyle = '#3fb950'; ctx.fillText('σ·R (linear)', pad.l+8, pad.t+12);
  ctx.fillStyle = '#bc8cff'; ctx.fillText('E_string(R)', pad.l+8, pad.t+24);
}}

// ── Dynamics: particle number + chiral condensate vs time ─────────────────────
function drawDyn() {{
  const frames = DATA.frames || [];
  if (!frames.length) return;
  document.getElementById('dynPanel').style.display = '';
  document.getElementById('heatPanel').style.display = '';
  const {{ctx, W, H}} = setupCanvas('cvDyn', 240);
  const pad = {{l:56, r:60, t:16, b:32}};
  const ts = frames.map(f=>f.time);
  const ns = frames.map(f=>f.particle_number);
  const cs_dyn = frames.map(f=>f.chiral_condensate);
  const minT = Math.min(...ts), maxT = Math.max(...ts);
  const maxN = Math.max(...ns, 0.01);
  const minC = Math.min(...cs_dyn), maxC = Math.max(...cs_dyn);
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  function xOf(t) {{ return pad.l + (t - minT) / (maxT - minT + 1e-14) * cw; }}
  function yOfN(n) {{ return H - pad.b - (n / maxN) * ch; }}
  function yOfC(c) {{ return H - pad.b - (c - minC) / (maxC - minC + 1e-14) * ch; }}
  drawAxes(ctx, pad, W, H);
  // Grid lines.
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 0.5;
  for (let t = 0; t <= 4; t++) {{
    const y = pad.t + ch * t / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W-pad.r, y); ctx.stroke();
  }}
  // Particle number.
  ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i < ts.length; i++) {{
    if (i === 0) ctx.moveTo(xOf(ts[i]), yOfN(ns[i]));
    else ctx.lineTo(xOf(ts[i]), yOfN(ns[i]));
  }}
  ctx.stroke();
  // Chiral condensate (right axis).
  ctx.strokeStyle = '#d29922'; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i < ts.length; i++) {{
    if (i === 0) ctx.moveTo(xOf(ts[i]), yOfC(cs_dyn[i]));
    else ctx.lineTo(xOf(ts[i]), yOfC(cs_dyn[i]));
  }}
  ctx.stroke();
  // Left y-axis labels (particle number).
  ctx.fillStyle = '#3fb950'; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
  for (let t = 0; t <= 4; t++) {{
    const n = maxN * t / 4;
    ctx.fillText(n.toFixed(2), pad.l - 4, H - pad.b - n/maxN * ch + 3);
  }}
  // Right y-axis labels (chiral condensate).
  ctx.fillStyle = '#d29922'; ctx.textAlign = 'left';
  for (let t = 0; t <= 4; t++) {{
    const c = minC + (maxC - minC) * t / 4;
    ctx.fillText(c.toFixed(3), W - pad.r + 4, H - pad.b - (c-minC)/(maxC-minC+1e-14)*ch + 3);
  }}
  // x ticks.
  ctx.fillStyle = '#8b949e'; ctx.textAlign = 'center'; ctx.font = '10px system-ui';
  for (let t = 0; t <= 5; t++) {{
    const tv = minT + (maxT - minT) * t / 5;
    ctx.fillText(tv.toFixed(1), xOf(tv), H - pad.b + 14);
  }}
  ctx.fillText('time t (J·t/ℏ)', pad.l + cw/2, H - 4);
  // Legend.
  ctx.textAlign = 'left'; ctx.font = '11px system-ui';
  ctx.fillStyle = '#3fb950'; ctx.fillText('particle number ⟨N⟩', pad.l + 8, pad.t + 14);
  ctx.fillStyle = '#d29922'; ctx.fillText('chiral condensate ⟨ψ̄ψ⟩', pad.l + 8, pad.t + 28);
}}

// ── Charge density heatmap ─────────────────────────────────────────────────────
function drawHeat() {{
  const frames = DATA.frames || [];
  if (!frames.length) return;
  const {{ctx, W, H}} = setupCanvas('cvHeat', 240);
  const pad = {{l:52, r:16, t:12, b:32}};
  const Nframes = frames.length;
  const Nsites = (frames[0].charge_density || []).length;
  if (!Nsites) return;
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  const cellW = cw / Nframes, cellH = ch / Nsites;
  // Colour scale: negative (red) → zero (dark) → positive (green).
  const allQ = frames.flatMap(f => f.charge_density);
  const qMax = Math.max(...allQ.map(Math.abs), 0.01);
  function qColor(q) {{
    const t = q / qMax;  // [-1, 1]
    if (t >= 0) {{
      const v = Math.round(t * 200);
      return `rgb(0,${{63 + v}},0)`;
    }} else {{
      const v = Math.round(-t * 200);
      return `rgb(${{63 + v}},0,0)`;
    }}
  }}
  for (let fi = 0; fi < Nframes; fi++) {{
    const charges = frames[fi].charge_density || [];
    for (let si = 0; si < Nsites; si++) {{
      const x = pad.l + fi * cellW;
      const y = pad.t + (Nsites - 1 - si) * cellH;  // site 0 at bottom
      ctx.fillStyle = qColor(charges[si] || 0);
      ctx.fillRect(x, y, Math.ceil(cellW), Math.ceil(cellH));
    }}
  }}
  // Axes.
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H-pad.b);
  ctx.lineTo(W-pad.r, H-pad.b); ctx.stroke();
  // x labels.
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  const tStep = Math.max(1, Math.floor(Nframes / 6));
  for (let fi = 0; fi < Nframes; fi += tStep) {{
    ctx.fillText(frames[fi].time.toFixed(1), pad.l + fi * cellW, H - pad.b + 14);
  }}
  ctx.fillText('time →', pad.l + cw/2, H - 4);
  // y labels.
  ctx.textAlign = 'right';
  for (let si = 0; si < Nsites; si += 2) {{
    const y = pad.t + (Nsites - 1 - si) * cellH + cellH/2;
    ctx.fillText(si, pad.l - 4, y + 3);
  }}
  // Colour bar legend.
  ctx.fillStyle = '#f85149'; ctx.fillRect(W-pad.r-40, pad.t, 12, 10);
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'left';
  ctx.fillText('Q<0', W-pad.r-26, pad.t+9);
  ctx.fillStyle = '#3fb950'; ctx.fillRect(W-pad.r-40, pad.t+14, 12, 10);
  ctx.fillText('Q>0', W-pad.r-26, pad.t+23);
}}

window.addEventListener('load', () => {{
  drawCharge();
  drawElec();
  drawSweep();
  drawString();
  drawDyn();
  drawHeat();
}});
window.addEventListener('resize', () => {{
  drawCharge();
  drawElec();
  drawSweep();
  drawString();
  drawDyn();
  drawHeat();
}});
</script>
</body>
</html>"""
  return html


# ── M21: Quantum Phase Estimation ─────────────────────────────────────────────

def _qpe_frames(
    n_ancilla: int,
    unitary: "np.ndarray",        # 2^n_target × 2^n_target unitary matrix
    eigenstate: "np.ndarray",     # 2^n_target initial target state (normalized)
    n_shots: int = 4096,
    true_phase: float | None = None,
    sweep_ancilla: list[int] | None = None,
    seed: int = 42,
) -> dict:
  """Full-matrix QPE reference solver using numpy FFT.

  Algorithm:
    1. Construct the (2^n_ancilla × 2^n_target) state |+⟩^n_anc ⊗ |eigenstate⟩
    2. Apply controlled-U^{2^k} for k = 0..n_ancilla-1 to build the phase-kickback state
    3. Apply inverse QFT on ancilla register via np.fft.fft / sqrt(N) (NOT ifft — sign convention)
    4. Marginalise over target register → phase probability distribution over 2^n_ancilla bins
    5. Sample n_shots from exact probabilities

  Returns dict with keys: n_ancilla, n_target, phase_probs, shot_counts, estimated_phase,
    true_phase, sweep (list of {n_ancilla, estimated_phase, error} for each ancilla in sweep_ancilla).
  """
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("QPE visualization requires numpy") from ex

  rng = np.random.default_rng(seed)

  dim_tgt = unitary.shape[0]
  n_target = int(round(np.log2(dim_tgt)))
  dim_anc = 1 << n_ancilla

  def run_qpe_exact(n_anc: int) -> "np.ndarray":
    """Return exact phase probabilities (length 2^n_anc) for given n_anc."""
    d_anc = 1 << n_anc
    # Build ancilla state: uniform superposition (|+⟩^n_anc) via equal-amplitude vector of length d_anc.
    # Combined state is d_anc × dim_tgt matrix: row a = amplitude for ancilla=a.
    state_mat = np.zeros((d_anc, dim_tgt), dtype=complex)
    # All ancilla basis states get equal amplitude times eigenstate.
    for a in range(d_anc):
      state_mat[a, :] = eigenstate / np.sqrt(d_anc)

    # Apply controlled-U^{2^k}: ancilla qubit k (in binary, MSB = qubit 0) controls U^{2^k}.
    # For each basis state |a⟩ of ancilla (integer a), each ancilla qubit k contributes a phase
    # if bit k of a is 1 (qubit k measured as 1 means U^{2^k} was applied).
    # The ancilla basis state |a⟩ accumulates phase e^{i * phase_angle * a} if eigenvalue is e^{i*phi}
    # and U|ψ⟩ = e^{i*phi}|ψ⟩.
    # Build U^{2^k} powers iteratively.
    u_power = unitary.copy()  # U^{2^0} = U
    for k in range(n_anc):
      # Ancilla qubit k controls U^{2^k}: bit k of integer a determines whether to apply u_power.
      # This gives phase e^{i2πφ·a} for |a⟩ (since Σ_k bit_k(a)*2^k = a).
      for a in range(d_anc):
        if (a >> k) & 1:
          state_mat[a, :] = u_power @ state_mat[a, :]
      u_power = u_power @ u_power  # U^{2^{k+1}}

    # Inverse QFT on ancilla via fft (NOT ifft — QFT uses e^{+i2πjk/N} = fft convention).
    # QFT†|a⟩ = (1/√N) Σ_k e^{-i2πak/N} |k⟩  →  fft along axis 0 divided by √N.
    state_mat = np.fft.fft(state_mat, axis=0) / np.sqrt(d_anc)

    # Marginalise over target register: prob[a] = Σ_t |state_mat[a,t]|^2
    probs = np.sum(np.abs(state_mat) ** 2, axis=1)
    # Normalise (should already be ~1).
    probs = np.real(probs)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs

  # Main QPE run.
  phase_probs = run_qpe_exact(n_ancilla)

  # Shot sampling.
  shot_counts = [0] * dim_anc
  if n_shots > 0:
    samples = rng.choice(dim_anc, size=n_shots, p=phase_probs)
    for s in samples:
      shot_counts[int(s)] += 1

  peak_bin = int(np.argmax(phase_probs))
  estimated_phase = peak_bin / dim_anc

  # Sweep over ancilla counts if requested.
  sweep = []
  if sweep_ancilla:
    for n_anc_s in sweep_ancilla:
      if n_anc_s < 1:
        continue
      probs_s = run_qpe_exact(n_anc_s)
      d_s = 1 << n_anc_s
      peak_s = int(np.argmax(probs_s))
      est_s = peak_s / d_s
      entry: dict = {"n_ancilla": n_anc_s, "estimated_phase": est_s}
      if true_phase is not None:
        err = abs(est_s - true_phase)
        # Handle wrap-around.
        err = min(err, 1.0 - err)
        entry["error"] = err
      sweep.append(entry)

  return {
      "n_ancilla": n_ancilla,
      "n_target": n_target,
      "phase_probs": phase_probs.tolist(),
      "shot_counts": shot_counts,
      "estimated_phase": estimated_phase,
      "true_phase": true_phase,
      "sweep": sweep,
      "n_shots": n_shots,
  }


def _qpe_page(data: dict, title: str = "QPE") -> str:
  """Generate a self-contained HTML page for QPE results."""
  import json as _json

  data_js = _json.dumps(data, ensure_ascii=False)
  true_phase = data.get("true_phase")
  true_phase_js = "null" if true_phase is None else str(true_phase)

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --accent:#58a6ff;
           --green:#3fb950; --orange:#d29922; --muted:#8b949e; --text:#e6edf3; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
          min-height:100vh; padding:24px; }}
  h1 {{ font-size:1.4rem; font-weight:600; color:var(--accent); margin-bottom:4px; }}
  .subtitle {{ color:var(--muted); font-size:.85rem; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; max-width:1100px; margin:0 auto; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px;
            padding:16px; }}
  .panel.wide {{ grid-column:span 2; }}
  .panel h2 {{ font-size:.95rem; color:var(--muted); margin-bottom:12px; font-weight:500; }}
  canvas {{ display:block; width:100%; }}
  .stat-row {{ display:flex; gap:24px; flex-wrap:wrap; margin-bottom:16px; }}
  .stat {{ background:var(--bg); border:1px solid var(--border); border-radius:8px;
           padding:8px 14px; }}
  .stat-label {{ font-size:.75rem; color:var(--muted); margin-bottom:2px; }}
  .stat-value {{ font-size:1.1rem; font-weight:600; color:var(--accent); }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">Quantum Phase Estimation — Python Reference Solver</p>

<div class="stat-row" id="stats"></div>

<div class="grid">
  <div class="panel">
    <h2>Exact Phase Probability Distribution</h2>
    <canvas id="cvPhase" height="260"></canvas>
  </div>
  <div class="panel">
    <h2>Shot Histogram (n={data.get('n_shots', 0)} samples)</h2>
    <canvas id="cvShots" height="260"></canvas>
  </div>
  <div class="panel wide" id="sweepPanel" style="display:none">
    <h2>Estimation Error vs n_ancilla (precision bits)</h2>
    <canvas id="cvPrecision" height="220"></canvas>
  </div>
</div>

<script>
const DATA = {data_js};
const TRUE_PHASE = {true_phase_js};

// ── Stats row ──────────────────────────────────────────────────────────────────
(function() {{
  const s = document.getElementById('stats');
  function stat(label, val) {{
    const d = document.createElement('div');
    d.className = 'stat';
    d.innerHTML = `<div class="stat-label">${{label}}</div><div class="stat-value">${{val}}</div>`;
    s.appendChild(d);
  }}
  stat('n_ancilla', DATA.n_ancilla);
  stat('n_target', DATA.n_target);
  stat('Estimated φ', DATA.estimated_phase.toFixed(6));
  if (TRUE_PHASE !== null) stat('True φ', TRUE_PHASE.toFixed(6));
  if (TRUE_PHASE !== null) {{
    let err = Math.abs(DATA.estimated_phase - TRUE_PHASE);
    err = Math.min(err, 1 - err);
    stat('|Error|', err.toExponential(3));
  }}
  stat('n_shots', DATA.n_shots);
}})();

// ── Helpers ────────────────────────────────────────────────────────────────────
function drawBarChart(canvasId, values, labels, opts) {{
  const cv = document.getElementById(canvasId);
  const dpr = window.devicePixelRatio || 1;
  cv.width  = cv.offsetWidth  * dpr || 500 * dpr;
  cv.height = cv.offsetHeight * dpr || 260 * dpr;
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = cv.offsetWidth || 500, H = cv.offsetHeight || 260;
  ctx.fillStyle = '#161b22'; ctx.fillRect(0, 0, W, H);

  const pad = {{ l:48, r:16, t:12, b:36 }};
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  const n = values.length;
  if (!n) return;
  const maxV = Math.max(...values, 1e-9);
  const barW = Math.max(1, cw / n - 1);

  for (let i = 0; i < n; i++) {{
    const x = pad.l + i * (cw / n);
    const bh = (values[i] / maxV) * ch;
    let color = opts.barColor || '#58a6ff';
    if (opts.highlightEst && i === opts.peakBin) color = '#d29922';
    if (opts.highlightTrue && TRUE_PHASE !== null) {{
      const trueBin = Math.round(TRUE_PHASE * n) % n;
      if (i === trueBin) color = '#3fb950';
    }}
    ctx.fillStyle = color;
    ctx.fillRect(x + 1, pad.t + ch - bh, barW, bh);
  }}

  // Axes.
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + ch);
  ctx.lineTo(pad.l + cw, pad.t + ch); ctx.stroke();

  // Y ticks.
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
  for (let t = 0; t <= 4; t++) {{
    const y = pad.t + ch - (t / 4) * ch;
    ctx.fillText((maxV * t / 4).toFixed(3), pad.l - 4, y + 3);
    ctx.strokeStyle = '#21262d';
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + cw, y); ctx.stroke();
  }}

  // X labels — show every Nth.
  ctx.fillStyle = '#8b949e'; ctx.textAlign = 'center';
  const step = Math.max(1, Math.floor(n / 8));
  for (let i = 0; i < n; i += step) {{
    const x = pad.l + i * (cw / n) + barW / 2;
    ctx.fillText(i, x, pad.t + ch + 14);
  }}

  // Legend.
  if (opts.highlightEst || opts.highlightTrue) {{
    ctx.textAlign = 'left'; ctx.font = '11px system-ui';
    let lx = pad.l + 4, ly = pad.t + 14;
    if (opts.highlightEst) {{
      ctx.fillStyle = '#d29922'; ctx.fillRect(lx, ly - 8, 10, 10);
      ctx.fillStyle = '#e6edf3'; ctx.fillText(' Estimated peak', lx + 12, ly);
      lx += 130;
    }}
    if (opts.highlightTrue && TRUE_PHASE !== null) {{
      ctx.fillStyle = '#3fb950'; ctx.fillRect(lx, ly - 8, 10, 10);
      ctx.fillStyle = '#e6edf3'; ctx.fillText(' True phase bin', lx + 12, ly);
    }}
  }}
}}

// ── Phase probability chart ────────────────────────────────────────────────────
function drawPhase() {{
  const probs = DATA.phase_probs;
  const peak = probs.indexOf(Math.max(...probs));
  drawBarChart('cvPhase', probs, null, {{
    barColor: '#58a6ff',
    highlightEst: true, peakBin: peak,
    highlightTrue: true,
  }});
}}

// ── Shot histogram ─────────────────────────────────────────────────────────────
function drawShots() {{
  const counts = DATA.shot_counts;
  const peak = counts.indexOf(Math.max(...counts));
  drawBarChart('cvShots', counts, null, {{
    barColor: '#58a6ff',
    highlightEst: true, peakBin: peak,
    highlightTrue: true,
  }});
}}

// ── Precision vs n_ancilla sweep ───────────────────────────────────────────────
function drawSweep() {{
  const sweep = DATA.sweep || [];
  if (!sweep.length || sweep[0].error === undefined) return;
  document.getElementById('sweepPanel').style.display = '';
  const cv = document.getElementById('cvPrecision');
  const dpr = window.devicePixelRatio || 1;
  cv.width  = cv.offsetWidth  * dpr || 900 * dpr;
  cv.height = cv.offsetHeight * dpr || 220 * dpr;
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = cv.offsetWidth || 900, H = cv.offsetHeight || 220;
  ctx.fillStyle = '#161b22'; ctx.fillRect(0, 0, W, H);

  const pad = {{ l:60, r:20, t:16, b:36 }};
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;

  // Theoretical curve: error ≤ 1/(2^n_anc).
  const ns = sweep.map(s => s.n_ancilla);
  const errs = sweep.map(s => s.error);
  const maxN = Math.max(...ns), minN = Math.min(...ns);
  const maxErr = Math.max(...errs, 0.01);

  function xOf(n) {{ return pad.l + (n - minN) / Math.max(maxN - minN, 1) * cw; }}
  function yOf(e) {{ return pad.t + ch - Math.log(e / maxErr) / Math.log(0.001 / maxErr) * ch; }}
  // Simple linear-y fallback.
  function yLin(e) {{ return pad.t + ch - (e / maxErr) * ch; }}

  // Theoretical 1/2^n line.
  ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
  ctx.beginPath();
  for (let i = 0; i < ns.length; i++) {{
    const th = 1.0 / (1 << ns[i]);
    const x = xOf(ns[i]), y = yLin(Math.min(th, maxErr));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  ctx.stroke(); ctx.setLineDash([]);

  // Actual errors.
  ctx.strokeStyle = '#d29922'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < ns.length; i++) {{
    const x = xOf(ns[i]), y = yLin(errs[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  ctx.stroke();
  for (let i = 0; i < ns.length; i++) {{
    ctx.fillStyle = '#d29922';
    ctx.beginPath(); ctx.arc(xOf(ns[i]), yLin(errs[i]), 4, 0, 2*Math.PI); ctx.fill();
  }}

  // Axes & labels.
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t+ch);
  ctx.lineTo(pad.l+cw, pad.t+ch); ctx.stroke();
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  for (let i = 0; i < ns.length; i++) {{
    ctx.fillText(ns[i], xOf(ns[i]), pad.t+ch+14);
  }}
  ctx.fillText('n_ancilla (precision bits)', pad.l + cw/2, H-4);
  ctx.textAlign = 'right';
  for (let t = 0; t <= 4; t++) {{
    const e = maxErr * (1 - t/4);
    ctx.fillText(e.toExponential(1), pad.l-4, yLin(e)+3);
  }}
  ctx.save(); ctx.translate(14, pad.t+ch/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign = 'center'; ctx.fillText('|error|', 0, 0); ctx.restore();

  // Legend.
  ctx.textAlign = 'left'; ctx.font = '11px system-ui';
  ctx.fillStyle = '#d29922'; ctx.fillRect(pad.l+8, pad.t+4, 12, 3);
  ctx.fillStyle = '#e6edf3'; ctx.fillText(' Actual error', pad.l+22, pad.t+9);
  ctx.fillStyle = '#3fb950'; ctx.fillRect(pad.l+8+110, pad.t+4, 12, 3);
  ctx.fillStyle = '#e6edf3'; ctx.fillText(' 1/2ⁿ bound', pad.l+22+110, pad.t+9);
}}

window.addEventListener('load', () => {{
  drawPhase();
  drawShots();
  drawSweep();
}});
window.addEventListener('resize', () => {{
  drawPhase();
  drawShots();
  drawSweep();
}});
</script>
</body>
</html>"""
  return html


# ── M20: VQE / QAOA variational optimizer ─────────────────────────────────────

def _vqe_frames(
    n_qubits: int,
    cost_hamiltonian: list[dict],
    ansatz_type: str = "hardware_efficient",
    n_layers: int = 2,
    optimizer: str = "adam",
    max_iterations: int = 120,
    learning_rate: float = 0.1,
    initial_params: list[float] | None = None,
    seed: int = 42,
) -> dict:
  """Numpy VQE reference solver.

  Returns a dict with keys:
    frames        list of {iteration, energy, gradient_norm, params}
    optimal_energy  float
    optimal_params  list[float]
    converged       bool
    n_qubits        int
    ansatz_type     str
    n_layers        int
    n_params        int
  """
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("VQE visualization requires numpy") from ex

  # ── Gate matrices ──────────────────────────────────────────────────────────
  I2 = np.eye(2, dtype=complex)
  X  = np.array([[0, 1], [1, 0]], dtype=complex)
  Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
  Z  = np.array([[1, 0], [0, -1]], dtype=complex)
  H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
  CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex).reshape(4, 4)

  PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}

  def kron_n(ops: list) -> np.ndarray:
    """Tensor-product of a list of 2×2 matrices → 2^n × 2^n."""
    result = ops[0]
    for op in ops[1:]:
      result = np.kron(result, op)
    return result

  def pauli_string_matrix(label: str, n: int) -> np.ndarray:
    """Build 2^n × 2^n matrix for a Pauli string like 'X0Z1'."""
    import re
    ops = ["I"] * n
    for m in re.finditer(r"([XYZI])(\d+)", label.upper()):
      p, q = m.group(1), int(m.group(2))
      if q < n:
        ops[q] = p
    return kron_n([PAULI[p] for p in ops])

  # ── Build cost Hamiltonian matrix ──────────────────────────────────────────
  dim = 1 << n_qubits
  H_cost = np.zeros((dim, dim), dtype=complex)
  for term in cost_hamiltonian:
    label = term.get("label", "")
    coeff = float(term.get("coefficient", 0.0))
    if not label or label.upper() == "I":
      H_cost += coeff * np.eye(dim, dtype=complex)
    else:
      H_cost += coeff * pauli_string_matrix(label, n_qubits)

  # ── Ansatz circuits ────────────────────────────────────────────────────────
  def ry_mat(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

  def rx_mat(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

  def rz_mat(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)

  def single_qubit_op(gate_2x2: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Embed a 2×2 gate on `qubit` into the n-qubit space."""
    ops = [I2] * n
    ops[qubit] = gate_2x2
    return kron_n(ops)

  def two_qubit_cnot(ctrl: int, targ: int, n: int) -> np.ndarray:
    """Exact CNOT(ctrl, targ) in n-qubit space."""
    dim_n = 1 << n
    mat = np.eye(dim_n, dtype=complex)
    for idx in range(dim_n):
      if (idx >> (n - 1 - ctrl)) & 1:
        flipped = idx ^ (1 << (n - 1 - targ))
        mat[:, idx] = 0
        mat[flipped, idx] = 1.0
    return mat

  def build_hea_state(params: np.ndarray) -> np.ndarray:
    """Hardware-efficient ansatz: n_layers × (RY on each qubit + CNOT chain) + final RY."""
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    p = 0
    for _ in range(n_layers):
      for q in range(n_qubits):
        state = single_qubit_op(ry_mat(params[p]), q, n_qubits) @ state
        p += 1
      for q in range(n_qubits - 1):
        state = two_qubit_cnot(q, q + 1, n_qubits) @ state
    for q in range(n_qubits):
      state = single_qubit_op(ry_mat(params[p]), q, n_qubits) @ state
      p += 1
    return state

  def build_qaoa_state(params: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    """QAOA: |+⟩^n + p rounds of cost + mixer."""
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    # Prepare |+⟩^n.
    for q in range(n_qubits):
      state = single_qubit_op(H_gate, q, n_qubits) @ state
    for layer in range(n_layers):
      gamma = params[2 * layer]
      beta  = params[2 * layer + 1]
      # Cost unitary: CNOT·RZ(γ)·CNOT per edge → exp(-iγ/2 Z_iZ_j).
      # This keeps the gate in the standard form exp(-iθG/2) with θ=γ, G=ZiZj,
      # so the parameter-shift rule ∂E/∂γ = (E(γ+π/2)-E(γ-π/2))/2 is valid.
      for (qi, qj) in edges:
        cx = two_qubit_cnot(qi, qj, n_qubits)
        state = cx @ state
        state = single_qubit_op(rz_mat(gamma), qj, n_qubits) @ state
        state = cx @ state
      # Mixer: RX(β) on each qubit → exp(-iβ/2 Xi), standard form with θ=β.
      for q in range(n_qubits):
        state = single_qubit_op(rx_mat(beta), q, n_qubits) @ state
    return state

  # Detect QAOA mode from cost_hamiltonian structure.
  is_qaoa = (ansatz_type == "qaoa")
  # Extract QAOA edges from ZiZj terms in cost_hamiltonian.
  import re as _re
  qaoa_edges: list[tuple[int, int]] = []
  if is_qaoa:
    for term in cost_hamiltonian:
      lbl = term.get("label", "")
      m = _re.match(r"^Z(\d+)Z(\d+)$", lbl.upper())
      if m:
        qaoa_edges.append((int(m.group(1)), int(m.group(2))))

  if is_qaoa:
    n_params = 2 * n_layers
  else:
    n_params = (n_layers + 1) * n_qubits

  def build_state(params: np.ndarray) -> np.ndarray:
    if is_qaoa:
      return build_qaoa_state(params, qaoa_edges)
    return build_hea_state(params)

  def eval_energy(params: np.ndarray) -> float:
    psi = build_state(params)
    return float(np.real(psi.conj() @ H_cost @ psi))

  # ── Gradient ───────────────────────────────────────────────────────────────
  # HEA: parameter-shift rule (each param appears in exactly one gate).
  # QAOA: parameter-shift is invalid for shared parameters (γ/β appear across
  #       all edge/qubit gates in the same round).  Use central finite differences.
  if is_qaoa:
    FD_EPS = 1e-3  # small finite-difference step

    def gradient(params: np.ndarray) -> np.ndarray:
      grad = np.zeros(n_params)
      for k in range(n_params):
        tp = params.copy(); tp[k] += FD_EPS
        tm = params.copy(); tm[k] -= FD_EPS
        grad[k] = (eval_energy(tp) - eval_energy(tm)) / (2.0 * FD_EPS)
      return grad
  else:
    def gradient(params: np.ndarray) -> np.ndarray:
      shift = np.pi / 2
      grad = np.zeros(n_params)
      for k in range(n_params):
        tp = params.copy(); tp[k] += shift
        tm = params.copy(); tm[k] -= shift
        grad[k] = (eval_energy(tp) - eval_energy(tm)) / 2.0
      return grad

  # ── Adam optimizer ─────────────────────────────────────────────────────────
  rng = np.random.default_rng(seed)
  if initial_params is not None:
    params = np.array(initial_params, dtype=float)
  elif is_qaoa:
    # QAOA: initialize γ ∈ [0, π/2] and β ∈ [0, π/4] to avoid barren plateaus.
    # Interleaved layout: [γ₁, β₁, γ₂, β₂, ...]
    params = np.zeros(n_params)
    for layer in range(n_layers):
      params[2 * layer]     = rng.uniform(0.05, np.pi / 2)   # γ
      params[2 * layer + 1] = rng.uniform(0.05, np.pi / 4)   # β
  else:
    params = rng.uniform(0, 2 * np.pi, n_params)

  b1, b2, eps = 0.9, 0.999, 1e-8
  m_adam = np.zeros(n_params)
  v_adam = np.zeros(n_params)
  t_adam = 0

  frames = []
  converged = False
  prev_energy = eval_energy(params)

  for it in range(max_iterations):
    energy = eval_energy(params)
    grad = gradient(params)
    gnorm = float(np.linalg.norm(grad))

    frames.append({
      "iteration": it,
      "energy": float(energy),
      "gradient_norm": gnorm,
      "params": params.tolist(),
    })

    if gnorm < 1e-5 or (it > 0 and abs(energy - prev_energy) < 1e-7):
      converged = True
      break

    if optimizer == "gradient_descent":
      params = params - learning_rate * grad
    else:
      # Adam
      t_adam += 1
      m_adam = b1 * m_adam + (1 - b1) * grad
      v_adam = b2 * v_adam + (1 - b2) * grad ** 2
      m_hat = m_adam / (1 - b1 ** t_adam)
      v_hat = v_adam / (1 - b2 ** t_adam)
      params = params - learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    prev_energy = energy

  # Final energy.
  final_energy = eval_energy(params)
  frames.append({
    "iteration": len(frames),
    "energy": float(final_energy),
    "gradient_norm": 0.0,
    "params": params.tolist(),
  })

  return {
    "frames": frames,
    "optimal_energy": float(final_energy),
    "optimal_params": params.tolist(),
    "converged": converged,
    "n_qubits": n_qubits,
    "ansatz_type": ansatz_type,
    "n_layers": n_layers,
    "n_params": n_params,
  }


def _vqe_page(data: dict) -> str:
  """HTML page with VQE/QAOA optimization visualization:
    - Energy convergence curve
    - Gradient norm convergence curve
    - Parameter trajectory heatmap
    - Final-state probability distribution
  """
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("VQE page requires numpy") from ex

  frames = data["frames"]
  optimal_energy = data["optimal_energy"]
  converged = data["converged"]
  n_params = data["n_params"]
  ansatz_type = data["ansatz_type"]
  n_qubits = data["n_qubits"]
  n_layers = data["n_layers"]

  iterations   = [f["iteration"] for f in frames]
  energies     = [f["energy"]    for f in frames]
  grad_norms   = [f["gradient_norm"] for f in frames]
  params_mat   = [f["params"]    for f in frames]  # list of lists

  # Rebuild final state for probability distribution.
  import re as _re
  I2 = np.eye(2, dtype=complex)
  X  = np.array([[0, 1], [1, 0]], dtype=complex)
  Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
  Z  = np.array([[1, 0], [0, -1]], dtype=complex)
  H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
  PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}
  dim = 1 << n_qubits

  def kron_n(ops):
    r = ops[0]
    for o in ops[1:]: r = np.kron(r, o)
    return r

  def ry_mat(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

  def rx_mat(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

  def rz_mat(theta):
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]).astype(complex)

  def sq(gate, q, n):
    ops = [I2] * n
    ops[q] = gate
    return kron_n(ops)

  def cx_mat(ctrl, targ, n):
    d = 1 << n
    mat = np.eye(d, dtype=complex)
    for idx in range(d):
      if (idx >> (n - 1 - ctrl)) & 1:
        fl = idx ^ (1 << (n - 1 - targ))
        mat[:, idx] = 0
        mat[fl, idx] = 1.0
    return mat

  best_params = np.array(data["optimal_params"])
  is_qaoa = (ansatz_type == "qaoa")

  if is_qaoa:
    # Extract edges from cost_hamiltonian if available (stored in data).
    qaoa_edges = data.get("qaoa_edges", [])
    state = np.zeros(dim, dtype=complex); state[0] = 1.0
    for q in range(n_qubits): state = sq(H_gate, q, n_qubits) @ state
    for layer in range(n_layers):
      gamma = best_params[2 * layer]
      beta  = best_params[2 * layer + 1]
      for (qi, qj) in qaoa_edges:
        cx = cx_mat(qi, qj, n_qubits)
        state = cx @ state
        state = sq(rz_mat(gamma), qj, n_qubits) @ state
        state = cx @ state
      for q in range(n_qubits):
        state = sq(rx_mat(beta), q, n_qubits) @ state
  else:
    state = np.zeros(dim, dtype=complex); state[0] = 1.0
    p = 0
    for _ in range(n_layers):
      for q in range(n_qubits):
        state = sq(ry_mat(best_params[p]), q, n_qubits) @ state; p += 1
      for q in range(n_qubits - 1):
        state = cx_mat(q, q + 1, n_qubits) @ state
    for q in range(n_qubits):
      state = sq(ry_mat(best_params[p]), q, n_qubits) @ state; p += 1

  probs = np.abs(state) ** 2
  bitstrings = [format(i, f"0{n_qubits}b") for i in range(dim)]

  # ── Build JSON data blobs ─────────────────────────────────────────────────
  import json as _json
  energy_series  = _json.dumps(list(zip(iterations, energies)))
  grad_series    = _json.dumps(list(zip(iterations, grad_norms)))
  params_json    = _json.dumps(params_mat)
  probs_json     = _json.dumps(list(zip(bitstrings, [float(p) for p in probs])))
  param_labels   = _json.dumps([f"θ{k}" for k in range(n_params)])
  iter_labels    = _json.dumps(iterations)

  title_suffix = "QAOA" if is_qaoa else "VQE"
  conv_str = "converged" if converged else "max iterations"
  status_line = f"optimal energy = {optimal_energy:.6f} ({conv_str})"

  return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>QuantX {title_suffix} Optimization</title>
<style>
  body{{font-family:sans-serif;background:#0a0a14;color:#e0e0f0;margin:0;padding:20px;}}
  h1{{color:#a0c8ff;text-align:center;font-size:1.4em;margin-bottom:4px;}}
  .subtitle{{text-align:center;color:#7090b0;font-size:0.9em;margin-bottom:20px;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;max-width:1100px;margin:0 auto;}}
  .panel{{background:#111827;border:1px solid #1e3050;border-radius:8px;padding:14px;}}
  canvas{{width:100%;}}
  h2{{color:#60a8e0;font-size:1em;margin:0 0 10px;}}
</style>
</head>
<body>
<h1>QuantX — {title_suffix} Optimization ({n_qubits} qubits, {n_layers} layers)</h1>
<div class="subtitle">{status_line}</div>
<div class="grid">
  <div class="panel">
    <h2>Energy Convergence</h2>
    <canvas id="cvEnergy" height="220"></canvas>
  </div>
  <div class="panel">
    <h2>Gradient Norm</h2>
    <canvas id="cvGrad" height="220"></canvas>
  </div>
  <div class="panel">
    <h2>Parameter Trajectory Heatmap (θ vs iteration)</h2>
    <canvas id="cvParams" height="220"></canvas>
  </div>
  <div class="panel">
    <h2>Final State Probability Distribution</h2>
    <canvas id="cvProbs" height="220"></canvas>
  </div>
</div>
<script>
const energySeries  = {energy_series};
const gradSeries    = {grad_series};
const paramsMat     = {params_json};
const probsData     = {probs_json};
const paramLabels   = {param_labels};
const iterLabels    = {iter_labels};

// ── Utilities ──────────────────────────────────────────────────────────────
function linspace(a, b, n) {{
  if (n < 2) return [a];
  return Array.from({{length: n}}, (_, i) => a + i * (b - a) / (n - 1));
}}

function drawLineChart(cvId, series, xlabel, ylabel, color) {{
  const cv = document.getElementById(cvId);
  const W = cv.clientWidth, H = 220;
  cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d");
  const pad = {{l:50, r:15, t:10, b:36}};
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0, 0, W, H);
  if (!series.length) return;
  const xs = series.map(p => p[0]), ys = series.map(p => p[1]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys) - 0.05 * (Math.max(...ys) - Math.min(...ys) + 1e-9);
  const yMax = Math.max(...ys) + 0.05 * (Math.max(...ys) - Math.min(...ys) + 1e-9);
  const tx = x => pad.l + (x - xMin) / (xMax - xMin + 1e-9) * pw;
  const ty = y => pad.t + ph - (y - yMin) / (yMax - yMin + 1e-9) * ph;
  ctx.strokeStyle = "#1e3050"; ctx.lineWidth = 1;
  [0.0, 0.25, 0.5, 0.75, 1.0].forEach(r => {{
    const x = pad.l + r * pw, y = pad.t + r * ph;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
  }});
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath();
  series.forEach(([x, y], i) => i === 0 ? ctx.moveTo(tx(x), ty(y)) : ctx.lineTo(tx(x), ty(y)));
  ctx.stroke();
  // Axis labels.
  ctx.fillStyle = "#7090b0"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
  ctx.fillText(xlabel, pad.l + pw / 2, H - 4);
  ctx.save(); ctx.translate(12, pad.t + ph / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylabel, 0, 0); ctx.restore();
  ctx.textAlign = "right";
  [yMin, yMax].forEach(v => ctx.fillText(v.toFixed(3), pad.l - 4, ty(v) + 4));
}}

function drawParamHeatmap(cvId) {{
  const cv = document.getElementById(cvId);
  const W = cv.clientWidth, H = 220;
  cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0, 0, W, H);
  const pad = {{l:50, r:15, t:10, b:36}};
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const nIter = paramsMat.length, nP = paramsMat[0].length;
  if (!nIter || !nP) return;
  // Flatten to find global min/max for color scaling.
  const allVals = paramsMat.flat();
  const vMin = Math.min(...allVals), vMax = Math.max(...allVals);
  const cellW = pw / nIter, cellH = ph / nP;
  for (let i = 0; i < nIter; ++i) {{
    for (let p = 0; p < nP; ++p) {{
      const t = (paramsMat[i][p] - vMin) / (vMax - vMin + 1e-9);
      const r = Math.round(20 + t * 180), g = Math.round(60 + t * 80), b = Math.round(180 - t * 80);
      ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
      ctx.fillRect(pad.l + i * cellW, pad.t + p * cellH, cellW + 1, cellH + 1);
    }}
  }}
  ctx.fillStyle = "#7090b0"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("iteration", pad.l + pw / 2, H - 4);
  ctx.save(); ctx.translate(12, pad.t + ph / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText("parameter", 0, 0); ctx.restore();
}}

function drawProbBars(cvId) {{
  const cv = document.getElementById(cvId);
  const W = cv.clientWidth, H = 220;
  cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0, 0, W, H);
  const pad = {{l:30, r:10, t:10, b:42}};
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const n = probsData.length;
  const barW = pw / n - 2;
  const pMax = Math.max(...probsData.map(d => d[1]));
  probsData.forEach(([bs, prob], i) => {{
    const barH = ph * prob / (pMax + 1e-9);
    const x = pad.l + i * (pw / n) + 1;
    const y = pad.t + ph - barH;
    const t = prob / (pMax + 1e-9);
    ctx.fillStyle = `rgb(${{Math.round(30 + t * 100)}}, ${{Math.round(100 + t * 100)}}, ${{Math.round(200 + t * 55)}})`;
    ctx.fillRect(x, y, barW, barH);
    if (n <= 16) {{
      ctx.fillStyle = "#7090b0"; ctx.font = "9px sans-serif";
      ctx.textAlign = "center"; ctx.save();
      ctx.translate(x + barW / 2, pad.t + ph + 2);
      ctx.rotate(-Math.PI / 3);
      ctx.fillText("|" + bs + "⟩", 0, 0);
      ctx.restore();
    }}
  }});
  ctx.fillStyle = "#7090b0"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("basis state", pad.l + pw / 2, H - 2);
}}

window.addEventListener("load", () => {{
  drawLineChart("cvEnergy", energySeries, "iteration", "⟨H⟩", "#4ca3dd");
  drawLineChart("cvGrad",   gradSeries,   "iteration", "|∇E|", "#e87040");
  drawParamHeatmap("cvParams");
  drawProbBars("cvProbs");
}});
</script>
</body>
</html>"""


# ── M25: Unified Dashboard ────────────────────────────────────────────────────

def _dashboard_page(pages: dict) -> str:  # noqa: PLR0912
  """Single-page unified dashboard embedding all QuantX scenario visualizations.

  Args:
      pages: Mapping of scenario-key (str) → full HTML string for that scenario.

  Returns:
      Self-contained HTML string for the dashboard.
  """
  CATEGORY_DEF = [
      ("🧮 量子基础", [
          ("rabi",               "Rabi 振荡"),
          ("entanglement_rabi",  "纠缠演化 (Rabi)"),
          ("entanglement_decay", "纠缠衰减"),
      ]),
      ("🌊 空间波函数", [
          ("spatial",        "1D 波包"),
          ("spatial2d",      "2D 波包"),
          ("spatial3d",      "3D 波包"),
          ("driven_barrier", "驱动势垒"),
          ("bloch",          "Bloch 振荡"),
          ("gaussian_pulse", "高斯脉冲"),
      ]),
      ("⚛️ 相对论性", [
          ("dirac2d", "2D Dirac 旋量"),
      ]),
      ("📊 相空间", [
          ("wigner_free",    "Wigner 自由"),
          ("wigner_barrier", "Wigner 势垒"),
      ]),
      ("🔗 开放量子系统", [
          ("lindblad", "Lindblad 主方程"),
      ]),
      ("🕸️ 张量网络", [
          ("mps",       "MPS 诊断"),
          ("mps_ising", "MPS Ising"),
          ("mps_xx",    "MPS XX"),
      ]),
      ("🔬 量子算法", [
          ("vqe_ising",       "VQE (Ising)"),
          ("vqe_qaoa",        "QAOA MaxCut"),
          ("qpe_phase",       "QPE 相位门"),
          ("qpe_hamiltonian", "QPE 哈密顿量"),
          ("qec_bit_flip",    "QEC 比特翻转"),
          ("qec_phase_flip",  "QEC 相位翻转"),
          ("qec_shor",        "QEC Shor 码"),
      ]),
      ("🌌 量子场论", [
          ("phi4_symmetric",     "φ⁴ 对称相"),
          ("phi4_broken",        "φ⁴ 破缺相"),
          ("phi4_sweep",         "φ⁴ 相变扫描"),
          ("phi4_dynamics",      "φ⁴ 量子淬火动力学"),
          ("schwinger_vacuum",   "Schwinger 真空"),
          ("hubbard",            "Hubbard Mott 绝缘体"),
          ("floquet",            "Floquet 周期驱动"),
          ("heisenberg",         "XXZ Heisenberg 链"),
          ("bose_hubbard",       "Bose-Hubbard 超流/Mott"),
          ("tfi",                "横场 Ising 量子相变"),
          ("ssh",                "SSH 拓扑绝缘体"),
          ("kitaev",             "Kitaev 链 Majorana 模式"),
          ("anderson",           "Anderson 局域化"),
          ("schwinger_string",   "Schwinger 弦张力"),
          ("schwinger_dynamics", "Schwinger 对产生"),
      ]),
  ]

  CAT_ICONS = {
      "🧮 量子基础":    "🧮",
      "🌊 空间波函数":  "🌊",
      "⚛️ 相对论性":    "⚛️",
      "📊 相空间":      "📊",
      "🔗 开放量子系统":"🔗",
      "🕸️ 张量网络":    "🕸️",
      "🔬 量子算法":    "🔬",
      "🌌 量子场论":    "🌌",
  }

  # Build sidebar HTML and lookup maps.
  nav_html = ""
  key_to_label: dict = {}
  key_to_cat: dict = {}
  key_to_icon: dict = {}
  first_key = None
  available_keys: list = []

  for cat_name, items in CATEGORY_DEF:
    icon = CAT_ICONS.get(cat_name, "📊")
    items_html = ""
    for key, label in items:
      if key not in pages:
        continue
      if first_key is None:
        first_key = key
      available_keys.append(key)
      key_to_label[key] = label
      key_to_cat[key] = cat_name
      key_to_icon[key] = icon
      items_html += (
          f'<button class="nav-item" data-key="{key}" '
          f'data-cat="{cat_name}" data-label="{label}" data-icon="{icon}">'
          f'{label}</button>\n'
      )
    if items_html:
      nav_html += (
          f'<div class="nav-group">'
          f'<div class="nav-cat">{cat_name}</div>'
          f'{items_html}</div>\n'
      )

  def _safe_json_for_script(data: object) -> str:
    # NOTE:
    #   This JSON is inlined into an HTML <script> tag below. Raw "</script>"
    #   inside scenario HTML would terminate the script element early during HTML
    #   parsing, even if it appears inside a JavaScript string literal. Escaping
    #   "<" prevents accidental closing tags while preserving JSON semantics.
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")

  total = len(available_keys)
  pages_json  = _safe_json_for_script(pages)
  labels_json = _safe_json_for_script(key_to_label)
  cats_json   = _safe_json_for_script(key_to_cat)
  icons_json  = _safe_json_for_script(key_to_icon)
  first_json  = _safe_json_for_script(first_key or "")

  return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>QuantX Dashboard</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;overflow:hidden;background:#0d1117;color:#e6edf3;
  font-family:'Segoe UI',system-ui,-apple-system,sans-serif;font-size:14px}}
.shell{{display:flex;height:100vh}}
/* ── Sidebar ────────────────────────────────────── */
.sidebar{{
  width:220px;min-width:170px;flex-shrink:0;
  background:#161b22;border-right:1px solid #30363d;
  display:flex;flex-direction:column;overflow:hidden
}}
.sidebar-header{{
  padding:13px 13px 9px;border-bottom:1px solid #30363d;flex-shrink:0
}}
.logo{{font-size:16px;font-weight:700;color:#58a6ff;letter-spacing:.4px;
  display:flex;align-items:center;gap:6px}}
.logo-badge{{
  background:#21262d;border:1px solid #30363d;border-radius:9px;
  padding:1px 7px;font-size:10px;color:#8b949e;font-weight:400
}}
.logo-sub{{font-size:10px;color:#6e7681;margin-top:2px;margin-bottom:9px}}
.search{{
  width:100%;padding:5px 9px;background:#21262d;border:1px solid #30363d;
  border-radius:6px;color:#e6edf3;font-size:12px;outline:none;
  transition:border-color .15s
}}
.search::placeholder{{color:#484f58}}
.search:focus{{border-color:#58a6ff}}
/* ── Nav scroll ──────────────────────────────────── */
.nav-scroll{{flex:1;overflow-y:auto;padding:6px 0}}
.nav-scroll::-webkit-scrollbar{{width:4px}}
.nav-scroll::-webkit-scrollbar-thumb{{background:#30363d;border-radius:2px}}
.nav-group{{margin-bottom:2px}}
.nav-cat{{
  padding:9px 13px 3px;font-size:10px;font-weight:600;
  color:#6e7681;text-transform:uppercase;letter-spacing:.9px;
  user-select:none
}}
.nav-item{{
  display:block;width:100%;padding:5px 13px 5px 19px;
  background:none;border:none;border-left:2px solid transparent;
  color:#c9d1d9;font-size:12.5px;text-align:left;cursor:pointer;
  transition:background .1s,color .1s;line-height:1.45
}}
.nav-item:hover{{background:#21262d;color:#e6edf3}}
.nav-item.active{{
  background:#1c2d3d;color:#58a6ff;
  border-left-color:#58a6ff;font-weight:500
}}
.nav-item.hidden{{display:none!important}}
.nav-group.all-hidden{{display:none}}
/* ── Viewport ────────────────────────────────────── */
.viewport{{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}}
.vbar{{
  height:40px;flex-shrink:0;background:#161b22;border-bottom:1px solid #30363d;
  display:flex;align-items:center;gap:7px;padding:0 13px
}}
.vbar-icon{{font-size:15px;flex-shrink:0;line-height:1}}
.vbar-title{{font-weight:600;font-size:13px;color:#e6edf3;flex-shrink:0}}
.vbar-cat{{
  font-size:10.5px;color:#8b949e;padding:1px 7px;
  background:#21262d;border:1px solid #30363d;border-radius:9px;flex-shrink:0
}}
.vbar-sep{{flex:1}}
.vbar-hint{{font-size:11px;color:#484f58}}
.btn-newtab{{
  background:#21262d;border:1px solid #30363d;border-radius:6px;
  color:#8b949e;cursor:pointer;padding:3px 9px;font-size:11.5px;
  transition:background .1s,color .1s;flex-shrink:0
}}
.btn-newtab:hover{{background:#2d333b;color:#e6edf3}}
/* ── Main frame / empty state ────────────────────── */
.empty{{
  flex:1;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:12px;color:#484f58
}}
.empty-icon{{font-size:52px;opacity:.35}}
.empty-msg{{font-size:13px}}
.empty-hint{{font-size:11px;color:#30363d}}
.main-frame{{
  flex:1;border:none;width:100%;height:100%;background:#0d1117;display:block
}}
.main-frame.off{{display:none}}
</style>
</head>
<body>
<div class="shell">
  <nav class="sidebar">
    <div class="sidebar-header">
      <div class="logo">QuantX <span class="logo-badge">{total}</span></div>
      <div class="logo-sub">Visualization Dashboard</div>
      <input class="search" id="search" type="text"
             placeholder="搜索场景… (按 / 聚焦)" autocomplete="off"/>
    </div>
    <div class="nav-scroll" id="nav-scroll">
{nav_html}
    </div>
  </nav>
  <div class="viewport">
    <div class="vbar">
      <span class="vbar-icon" id="vbar-icon">⚛️</span>
      <span class="vbar-title" id="vbar-title">QuantX Dashboard</span>
      <span class="vbar-cat"  id="vbar-cat" style="display:none"></span>
      <span class="vbar-sep"></span>
      <span class="vbar-hint" id="vbar-hint">{total} 个场景 · ↑↓ 切换 · / 搜索</span>
      <button class="btn-newtab" id="btn-newtab"
              title="在新标签页中打开当前场景" style="display:none">⧉ 新标签页</button>
    </div>
    <div class="empty" id="empty">
      <span class="empty-icon">⚛️</span>
      <span class="empty-msg">从左侧选择一个场景开始可视化</span>
      <span class="empty-hint">↑ ↓ 键盘导航 · / 快速搜索</span>
    </div>
    <iframe class="main-frame off" id="main-frame"
            sandbox="allow-scripts"
            title="场景可视化"></iframe>
  </div>
</div>
<script>
(function(){{
"use strict";
var PAGES  = {pages_json};
var LABELS = {labels_json};
var CATS   = {cats_json};
var ICONS  = {icons_json};
var current = null;

var frame    = document.getElementById("main-frame");
var empty    = document.getElementById("empty");
var vbarIcon = document.getElementById("vbar-icon");
var vbarTitle= document.getElementById("vbar-title");
var vbarCat  = document.getElementById("vbar-cat");
var vbarHint = document.getElementById("vbar-hint");
var btnNew   = document.getElementById("btn-newtab");
var search   = document.getElementById("search");

function allItems() {{
  return Array.from(document.querySelectorAll(".nav-item"));
}}

function switchTo(key) {{
  if (!PAGES[key] || key === current) return;
  current = key;
  frame.srcdoc = PAGES[key];
  frame.classList.remove("off");
  empty.style.display = "none";
  vbarIcon.textContent  = ICONS[key]  || "📊";
  vbarTitle.textContent = LABELS[key] || key;
  var cat = CATS[key] || "";
  if (cat) {{ vbarCat.textContent = cat; vbarCat.style.display = ""; }}
  else      {{ vbarCat.style.display = "none"; }}
  btnNew.style.display = "";
  vbarHint.style.display = "none";
  allItems().forEach(function(b) {{
    b.classList.toggle("active", b.dataset.key === key);
  }});
  var active = document.querySelector(".nav-item.active");
  if (active) active.scrollIntoView({{block:"nearest",behavior:"smooth"}});
}}

// Open current page in a new browser tab via Blob URL.
btnNew.addEventListener("click", function() {{
  if (!current || !PAGES[current]) return;
  var blob = new Blob([PAGES[current]], {{type:"text/html;charset=utf-8"}});
  var url  = URL.createObjectURL(blob);
  var tab  = window.open(url, "_blank");
  if (!tab) alert("请允许弹出窗口以在新标签页中打开场景。");
  setTimeout(function() {{ URL.revokeObjectURL(url); }}, 15000);
}});

// Sidebar click.
document.querySelectorAll(".nav-item").forEach(function(btn) {{
  btn.addEventListener("click", function() {{ switchTo(btn.dataset.key); }});
}});

// Keyboard navigation.
document.addEventListener("keydown", function(e) {{
  if (document.activeElement === search) {{
    if (e.key === "Escape") {{ search.value = ""; filterNav(""); search.blur(); }}
    return;
  }}
  if (e.key === "/" || e.key === "f") {{
    e.preventDefault(); search.focus(); return;
  }}
  var visible = allItems().filter(function(b) {{
    return !b.classList.contains("hidden");
  }});
  var idx = visible.findIndex(function(b) {{ return b.dataset.key === current; }});
  if (e.key === "ArrowDown") {{
    e.preventDefault();
    if (idx < visible.length - 1) switchTo(visible[idx + 1].dataset.key);
  }} else if (e.key === "ArrowUp") {{
    e.preventDefault();
    if (idx > 0) switchTo(visible[idx - 1].dataset.key);
  }}
}});

// Search / filter sidebar items.
function filterNav(q) {{
  q = q.trim().toLowerCase();
  document.querySelectorAll(".nav-group").forEach(function(grp) {{
    var any = false;
    grp.querySelectorAll(".nav-item").forEach(function(btn) {{
      var match = !q ||
        (btn.dataset.label || "").toLowerCase().includes(q) ||
        (btn.dataset.key   || "").toLowerCase().includes(q) ||
        (btn.dataset.cat   || "").toLowerCase().includes(q);
      btn.classList.toggle("hidden", !match);
      if (match) any = true;
    }});
    grp.classList.toggle("all-hidden", !any);
  }});
}}
search.addEventListener("input", function() {{ filterNav(search.value); }});

// Load the first available scenario on start.
switchTo({first_json});
}})();
</script>
</body>
</html>"""


# ── M27-C: 1D Hubbard Model ───────────────────────────────────────────────────

def _hubbard_frames(
    n_sites: int = 6,
    t_hop: float = 1.0,
    U: float = 4.0,
    n_up: int = 0,
    n_dn: int = 0,
    scenario: str = "sweep",
    n_sweep: int = 20,
    U_min: float = 0.0,
    U_max: float = 12.0,
) -> dict:
  """Compute 1D Hubbard model via exact diagonalization (NumPy reference)."""
  import numpy as np
  from itertools import combinations

  N = n_sites
  if n_up == 0: n_up = N // 2
  if n_dn == 0: n_dn = N // 2

  def enum_states(n_bits, n_ones):
    return [sum(1 << b for b in combo) for combo in combinations(range(n_bits), n_ones)]

  def jw_sign(state, i, j):
    lo, hi = min(i, j), max(i, j)
    if hi - lo <= 1: return 1
    bits_between = sum(1 for b in range(lo + 1, hi) if (state >> b) & 1)
    return (-1) ** bits_between

  def build_H(U_val):
    up_st = enum_states(N, n_up)
    dn_st = enum_states(N, n_dn)
    up_map = {s: i for i, s in enumerate(up_st)}
    dn_map = {s: i for i, s in enumerate(dn_st)}
    dim = len(up_st) * len(dn_st)
    H = np.zeros((dim, dim), dtype=float)
    for iu, us in enumerate(up_st):
      for id_, ds in enumerate(dn_st):
        idx = iu * len(dn_st) + id_
        # diagonal: U * double-occupancy
        docc = bin(us & ds).count('1')
        if docc: H[idx, idx] += U_val * docc
        # spin-up hopping
        for site in range(N - 1):
          b0, b1 = 1 << site, 1 << (site + 1)
          if (us & b0) and not (us & b1):
            us2 = (us ^ b0) | b1
            if us2 in up_map:
              iu2 = up_map[us2]; idx2 = iu2 * len(dn_st) + id_
              sg = jw_sign(us, site, site + 1)
              H[idx2, idx] -= t_hop * sg
              H[idx, idx2] -= t_hop * sg
        # spin-down hopping
        for site in range(N - 1):
          b0, b1 = 1 << site, 1 << (site + 1)
          if (ds & b0) and not (ds & b1):
            ds2 = (ds ^ b0) | b1
            if ds2 in dn_map:
              id2 = dn_map[ds2]; idx2 = iu * len(dn_st) + id2
              sg = jw_sign(ds, site, site + 1)
              H[idx2, idx] -= t_hop * sg
              H[idx, idx2] -= t_hop * sg
    return H, up_st, dn_st, up_map, dn_map

  def compute_obs(psi, up_st, dn_st, up_map, dn_map):
    n_dn_loc = len(dn_st)
    docc = 0.0
    sz_corr = np.zeros(N)
    rho = np.zeros((N, N), dtype=float)
    for iu, us in enumerate(up_st):
      for id_, ds in enumerate(dn_st):
        idx = iu * n_dn_loc + id_
        p = psi[idx] ** 2
        docc += p * bin(us & ds).count('1')
        sz0 = 0.5 * (((us >> 0) & 1) - ((ds >> 0) & 1))
        for r in range(N):
          szr = 0.5 * (((us >> r) & 1) - ((ds >> r) & 1))
          sz_corr[r] += p * sz0 * szr
        # one-body density matrix (diagonal part)
        for s in range(N):
          if (us >> s) & 1: rho[s, s] += p
          if (ds >> s) & 1: rho[s, s] += p
    # off-diagonal rho (spin-up)
    for iu, us in enumerate(up_st):
      for id_, ds in enumerate(dn_st):
        idx = iu * n_dn_loc + id_
        pv = psi[idx]
        for s in range(N):
          if not (us >> s) & 1: continue
          for r in range(N):
            if r == s or (us >> r) & 1: continue
            us2 = (us & ~(1 << s)) | (1 << r)
            if us2 in up_map:
              idx2 = up_map[us2] * n_dn_loc + id_
              sg = jw_sign(us, s, r)
              rho[r, s] += psi[idx2] * pv * sg
    # off-diagonal rho (spin-down)
    for iu, us in enumerate(up_st):
      for id_, ds in enumerate(dn_st):
        idx = iu * n_dn_loc + id_
        pv = psi[idx]
        for s in range(N):
          if not (ds >> s) & 1: continue
          for r in range(N):
            if r == s or (ds >> r) & 1: continue
            ds2 = (ds & ~(1 << s)) | (1 << r)
            if ds2 in dn_map:
              idx2 = iu * n_dn_loc + dn_map[ds2]
              sg = jw_sign(ds, s, r)
              rho[r, s] += psi[idx2] * pv * sg
    n_k = np.array([
        sum(np.cos(2 * np.pi * m * (r - s) / N) * rho[r, s]
            for r in range(N) for s in range(N)) / N
        for m in range(N)
    ])
    s_pi = sum(((-1) ** r) * sz_corr[r] for r in range(N)) / N
    return {
        "double_occupancy": float(docc / N),
        "spin_structure_pi": float(s_pi),
        "spin_corr": sz_corr.tolist(),
        "momentum_dist": n_k.tolist(),
    }

  def solve_one(U_val):
    H, up_st, dn_st, up_map, dn_map = build_H(U_val)
    evals, evecs = np.linalg.eigh(H)
    psi0 = evecs[:, 0]
    E0 = float(evals[0])
    E1 = float(evals[1]) if len(evals) > 1 else E0
    obs = compute_obs(psi0, up_st, dn_st, up_map, dn_map)
    return E0, E1, obs

  sweep_pts = []
  if scenario == "sweep":
    for i in range(n_sweep):
      U_val = U_min + (U_max - U_min) * i / max(n_sweep - 1, 1)
      E0, E1, obs = solve_one(U_val)
      sweep_pts.append({
          "U": float(U_val),
          "ground_energy": E0,
          "energy_gap": E1 - E0,
          "double_occupancy": obs["double_occupancy"],
          "spin_structure_pi": obs["spin_structure_pi"],
      })

  E0, E1, obs = solve_one(U)
  return {
      "n_sites": N,
      "n_up": n_up,
      "n_dn": n_dn,
      "t_hop": t_hop,
      "U": U,
      "ground_energy": E0,
      "energy_gap": E1 - E0,
      "double_occupancy": obs["double_occupancy"],
      "spin_structure_pi": obs["spin_structure_pi"],
      "spin_corr": obs["spin_corr"],
      "momentum_dist": obs["momentum_dist"],
      "sweep": sweep_pts,
      "scenario": scenario,
  }


def _hubbard_page(data: dict, title: str = "1D Hubbard Model") -> str:
  """Generate self-contained HTML for the 1D Hubbard model."""
  import json as json_mod
  data_json = json_mod.dumps(data)
  N = data.get("n_sites", 6)

  return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif;padding:20px}}
h2{{font-size:1.2rem;font-weight:600;color:#58a6ff;margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:960px;margin:0 auto}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
canvas{{display:block;width:100%;height:240px}}
.info{{font-size:0.82rem;color:#8b949e;margin-top:8px}}
.param{{display:flex;gap:24px;flex-wrap:wrap;font-size:0.85rem;color:#8b949e;margin-bottom:16px;max-width:960px;margin-left:auto;margin-right:auto}}
.param span{{color:#58a6ff;font-weight:600}}
</style>
</head>
<body>
<h1 style="font-size:1.4rem;color:#58a6ff;margin-bottom:12px">{title}</h1>
<div class="param">
  <div>N = <span id="pN"></span> sites</div>
  <div>N↑ = <span id="pNup"></span>, N↓ = <span id="pNdn"></span></div>
  <div>t = <span id="pt"></span></div>
  <div>U/t = <span id="pUt"></span></div>
  <div>E₀ = <span id="pE0"></span></div>
  <div>Δ = <span id="pGap"></span></div>
  <div>D = <span id="pDocc"></span></div>
  <div>S(π) = <span id="pSpi"></span></div>
</div>
<div class="grid">
  <div class="card"><h2>Spin Correlation ⟨S⁰ᶻ Sʳᶻ⟩</h2><canvas id="cSpinCorr"></canvas>
    <div class="info">Antiferromagnetic: alternating sign pattern at large U/t</div></div>
  <div class="card"><h2>Momentum Distribution n(k)</h2><canvas id="cMomDist"></canvas>
    <div class="info">Fermi step discontinuity at k_F for non-interacting; smeared at large U</div></div>
  <div class="card"><h2>Double Occupancy vs U/t</h2><canvas id="cDocc"></canvas>
    <div class="info">D → 0 as U/t → ∞ (Mott insulator)</div></div>
  <div class="card"><h2>S(π) Spin Structure Factor vs U/t</h2><canvas id="cSpi"></canvas>
    <div class="info">Antiferromagnetic order grows with U/t</div></div>
</div>
<script>
const DATA = {data_json};

// Populate params
document.getElementById("pN").textContent    = DATA.n_sites;
document.getElementById("pNup").textContent  = DATA.n_up;
document.getElementById("pNdn").textContent  = DATA.n_dn;
document.getElementById("pt").textContent    = DATA.t_hop.toFixed(2);
document.getElementById("pUt").textContent   = (DATA.U / DATA.t_hop).toFixed(2);
document.getElementById("pE0").textContent   = DATA.ground_energy.toFixed(4);
document.getElementById("pGap").textContent  = DATA.energy_gap.toFixed(4);
document.getElementById("pDocc").textContent = DATA.double_occupancy.toFixed(4);
document.getElementById("pSpi").textContent  = DATA.spin_structure_pi.toFixed(4);

function barChart(canvasId, labels, values, colorFn, yLabel) {{
  var cv = document.getElementById(canvasId);
  var w = cv.offsetWidth || 400; var h = cv.offsetHeight || 240;
  cv.width = w; cv.height = h;
  var ctx = cv.getContext("2d");
  var pad = 40;
  var bw = Math.max(1, (w - 2*pad) / values.length - 2);
  var vmax = Math.max(...values.map(Math.abs)) * 1.2 || 1;
  var vmin = Math.min(0, ...values) * 1.2;
  var range = vmax - vmin;
  var zero_y = h - pad - (-vmin / range) * (h - 2*pad);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, h-pad); ctx.lineTo(w-pad, h-pad); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad, zero_y); ctx.lineTo(w-pad, zero_y);
  ctx.strokeStyle = "#444c56"; ctx.stroke();
  for (var i = 0; i < values.length; i++) {{
    var x = pad + i * ((w - 2*pad) / values.length) + 1;
    var bh = Math.abs(values[i]) / range * (h - 2*pad);
    var y = values[i] >= 0 ? zero_y - bh : zero_y;
    ctx.fillStyle = colorFn(i, values[i]);
    ctx.fillRect(x, y, bw, bh);
    if (labels && labels[i] !== undefined) {{
      ctx.fillStyle = "#8b949e"; ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(labels[i], x + bw/2, h - pad + 12);
    }}
  }}
  if (yLabel) {{
    ctx.fillStyle = "#8b949e"; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
    ctx.fillText(yLabel, 4, 14);
  }}
}}

// Spin correlation
var scLabels = DATA.spin_corr.map(function(_, i) {{ return "r=" + i; }});
barChart("cSpinCorr", scLabels, DATA.spin_corr,
  function(i, v) {{ return v >= 0 ? "#388bfd" : "#f47067"; }}, "⟨S⁰ᶻSʳᶻ⟩");

// Momentum distribution
var N = DATA.n_sites;
var kLabels = DATA.momentum_dist.map(function(_, m) {{ return (m <= N/2) ? m : m-N; }});
barChart("cMomDist", kLabels.map(function(k) {{ return "k=" + k; }}), DATA.momentum_dist,
  function() {{ return "#57ab5a"; }}, "n(k)");

// Sweep charts
if (DATA.sweep && DATA.sweep.length > 0) {{
  var sweepU   = DATA.sweep.map(function(p) {{ return p.U / DATA.t_hop; }});
  var sweepD   = DATA.sweep.map(function(p) {{ return p.double_occupancy; }});
  var sweepSpi = DATA.sweep.map(function(p) {{ return p.spin_structure_pi; }});

  function lineChart(canvasId, xs, ys, color, yLabel) {{
    var cv = document.getElementById(canvasId);
    var w = cv.offsetWidth || 400; var h = cv.offsetHeight || 240;
    cv.width = w; cv.height = h;
    var ctx = cv.getContext("2d");
    var pad = 44;
    var xmin = Math.min(...xs), xmax = Math.max(...xs);
    var ymin = Math.min(0, ...ys), ymax = Math.max(...ys) * 1.1 || 1;
    var xr = xmax - xmin || 1; var yr = ymax - ymin || 1;
    function px(x) {{ return pad + (x - xmin)/xr * (w - 2*pad); }}
    function py(y) {{ return h - pad - (y - ymin)/yr * (h - 2*pad); }}
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
    // axes
    ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, h-pad);
    ctx.lineTo(w-pad, h-pad); ctx.stroke();
    // grid
    for (var yi = 0; yi <= 4; yi++) {{
      var gy = py(ymin + yr * yi/4);
      ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(w-pad, gy);
      ctx.strokeStyle = "#21262d"; ctx.stroke();
      ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "right";
      ctx.fillText((ymin + yr * yi/4).toFixed(2), pad - 3, gy + 3);
    }}
    // line
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
    for (var i = 0; i < xs.length; i++) {{
      i === 0 ? ctx.moveTo(px(xs[i]), py(ys[i])) : ctx.lineTo(px(xs[i]), py(ys[i]));
    }}
    ctx.stroke();
    // x ticks
    for (var xi = 0; xi <= 4; xi++) {{
      var gx = px(xmin + xr * xi/4);
      ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "center";
      ctx.fillText((xmin + xr * xi/4).toFixed(1), gx, h - pad + 12);
    }}
    ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "left";
    ctx.fillText(yLabel, 4, 14);
    ctx.fillText("U/t", w/2, h - 4);
  }}

  lineChart("cDocc", sweepU, sweepD, "#e3b341", "Double Occupancy D");
  lineChart("cSpi", sweepU, sweepSpi, "#f47067", "S(π)");
}}
</script>
</body>
</html>"""


# ── M27-D: Floquet Kicked-Ising ───────────────────────────────────────────────

def _floquet_frames(
    n_sites: int = 8,
    J: float = 1.0,
    h_z: float = 0.809,
    drive_amp: float = 1.5,
    drive_omega: float = 4.5,
    n_periods: int = 40,
    scenario: str = "dynamics",
    n_scan: int = 20,
    scan_amp_max: float = 3.0,
) -> dict:
  """Compute Floquet kicked-Ising dynamics (NumPy reference)."""
  import numpy as np

  N = n_sites
  dim = 2 ** N
  PI = np.pi
  T = 2.0 * PI / drive_omega
  halfT = T / 2.0

  # Build H₀ diagonal (Z basis)
  h0_diag = np.zeros(dim)
  for s in range(dim):
    e = 0.0
    for i in range(N - 1):
      zi = 1 if (s >> i) & 1 else -1
      zj = 1 if (s >> (i+1)) & 1 else -1
      e -= J * zi * zj
    for i in range(N):
      zi = 1 if (s >> i) & 1 else -1
      e -= h_z * zi
    h0_diag[s] = e

  def apply_exp_h0(psi, tau):
    return psi * np.exp(-1j * tau * h0_diag)

  def apply_kick(psi, hx):
    c = np.cos(hx); ms = -1j * np.sin(hx)
    out = psi.copy()
    for i in range(N):
      bit = 1 << i
      # Pair each |s⟩ with |s ^ bit⟩
      s_arr = np.arange(dim)
      mask = (s_arr & bit) == 0
      lo = s_arr[mask]; hi = lo | bit
      a, b = out[lo].copy(), out[hi].copy()
      out[lo] = c * a + ms * b
      out[hi] = ms * a + c * b
    return out

  def apply_period(psi):
    psi = apply_exp_h0(psi, halfT)
    psi = apply_kick(psi, drive_amp)
    psi = apply_exp_h0(psi, halfT)
    return psi

  def entropy(psi):
    dimA = 2 ** (N // 2); dimB = 2 ** (N - N // 2)
    M = psi.reshape(dimA, dimB)
    sv = np.linalg.svd(M, compute_uv=False)
    lam = sv ** 2
    lam = lam[lam > 1e-14]
    lam /= lam.sum()
    return float(-np.sum(lam * np.log(lam)))

  frames = []
  # Néel state |↑↓↑↓…⟩
  neel = sum(1 << i for i in range(0, N, 2))
  psi = np.zeros(dim, dtype=complex); psi[neel] = 1.0

  def record(period, t, psi_r):
    sz_ops = np.array([1 if ((s >> i) & 1) else -1
                       for i in range(N) for s in range(dim)]).reshape(N, dim)
    prob = np.abs(psi_r) ** 2
    mz = float(np.sum(prob[np.newaxis, :] * sz_ops) / (N * dim))
    mz = float(np.dot(prob, sum(
        np.array([1 if (s >> i) & 1 else -1 for s in range(dim)]) for i in range(N)
    )) / N)
    en = float(np.real(np.dot(np.conj(psi_r), h0_diag * psi_r)))
    s = entropy(psi_r)
    frames.append({"period": int(period), "time": float(t),
                   "magnetization_z": float(mz), "energy": en, "entropy": s})

  record(0, 0.0, psi)
  for p in range(1, n_periods + 1):
    psi = apply_period(psi)
    record(p, p * T, psi)

  # Quasi-energy spectrum
  quasi_energies = []
  mean_gap_ratio = 0.0
  if dim <= 256:
    U_mat = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
      e = np.zeros(dim, dtype=complex); e[col] = 1.0
      e = apply_period(e)
      U_mat[:, col] = e
    evals = np.linalg.eigvals(U_mat)
    qe = sorted(float(np.angle(ev)) / T for ev in evals)
    quasi_energies = qe
    if len(qe) >= 3:
      diffs = np.diff(qe)
      r_vals = [min(diffs[i], diffs[i+1]) / max(diffs[i], diffs[i+1])
                for i in range(len(diffs)-1)
                if diffs[i] > 1e-14 and diffs[i+1] > 1e-14]
      mean_gap_ratio = float(np.mean(r_vals)) if r_vals else 0.0

  # Amplitude scan
  scan_amps = []
  scan_qe = []
  if scenario == "spectrum" and dim <= 64:
    for si in range(n_scan):
      amp = scan_amp_max * si / max(n_scan - 1, 1)
      scan_amps.append(float(amp))
      U_s = np.zeros((dim, dim), dtype=complex)
      for col in range(dim):
        e = np.zeros(dim, dtype=complex); e[col] = 1.0
        e = apply_exp_h0(e, halfT)
        e = apply_kick(e, amp)
        e = apply_exp_h0(e, halfT)
        U_s[:, col] = e
      evs = np.linalg.eigvals(U_s)
      scan_qe.append(sorted(float(np.angle(ev)) / T for ev in evs))

  return {
      "n_sites": N, "J": J, "h_z": h_z,
      "drive_amp": drive_amp, "drive_omega": drive_omega,
      "T": T, "frames": frames,
      "quasi_energies": quasi_energies,
      "mean_gap_ratio": mean_gap_ratio,
      "scan_amps": scan_amps,
      "scan_qe": scan_qe,
  }


def _floquet_page(data: dict, title: str = "Floquet 周期驱动") -> str:
  """Generate self-contained HTML for the Floquet kicked-Ising simulation."""
  import json as json_mod
  data_json = json_mod.dumps(data)

  return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif;padding:20px}}
h2{{font-size:1.1rem;font-weight:600;color:#58a6ff;margin-bottom:10px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:980px;margin:0 auto}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
canvas{{display:block;width:100%;height:220px}}
.info{{font-size:0.8rem;color:#8b949e;margin-top:6px}}
.param{{display:flex;gap:20px;flex-wrap:wrap;font-size:0.84rem;color:#8b949e;margin-bottom:14px;max-width:980px;margin-left:auto;margin-right:auto}}
.param span{{color:#58a6ff;font-weight:600}}
.badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:600}}
.badge-green{{background:#033a16;color:#3fb950}}
.badge-red{{background:#3d1a1a;color:#f47067}}
</style>
</head>
<body>
<h1 style="font-size:1.4rem;color:#58a6ff;margin-bottom:12px">{title}</h1>
<div class="param">
  <div>N = <span id="pN"></span> qubits</div>
  <div>J = <span id="pJ"></span></div>
  <div>h_z = <span id="pHz"></span></div>
  <div>h_x (kick) = <span id="pHx"></span></div>
  <div>ω = <span id="pOmega"></span></div>
  <div>T = <span id="pT"></span></div>
  <div>Gap ratio r = <span id="pGapR"></span></div>
  <div id="phaseBadge"></div>
</div>
<div class="grid">
  <div class="card"><h2>Magnetization ⟨m_z(t)⟩</h2><canvas id="cMz"></canvas>
    <div class="info">Stroboscopic snapshots at integer multiples of period T</div></div>
  <div class="card"><h2>Bipartite Entanglement Entropy</h2><canvas id="cEnt"></canvas>
    <div class="info">Logarithmic growth → thermalization; bounded → MBL / prethermal</div></div>
  <div class="card"><h2>⟨H₀⟩ Energy Absorption</h2><canvas id="cEn"></canvas>
    <div class="info">Heating to infinite temperature in ergodic phase</div></div>
  <div class="card"><h2>Quasi-energy Spectrum (level spacing)</h2><canvas id="cQE"></canvas>
    <div class="info">Poisson ≈ 0.39 (MBL/integrable); GUE ≈ 0.60 (ergodic/chaotic)</div></div>
</div>
<script>
const DATA = {data_json};

document.getElementById("pN").textContent     = DATA.n_sites;
document.getElementById("pJ").textContent     = DATA.J.toFixed(2);
document.getElementById("pHz").textContent    = DATA.h_z.toFixed(3);
document.getElementById("pHx").textContent    = DATA.drive_amp.toFixed(3);
document.getElementById("pOmega").textContent = DATA.drive_omega.toFixed(3);
document.getElementById("pT").textContent     = (DATA.T || 0).toFixed(3);
var gr = DATA.mean_gap_ratio || 0;
document.getElementById("pGapR").textContent  = gr.toFixed(3);
var badge = document.getElementById("phaseBadge");
if (gr > 0) {{
  if (gr > 0.52) {{
    badge.innerHTML = '<span class="badge badge-red">Ergodic / Chaotic</span>';
  }} else {{
    badge.innerHTML = '<span class="badge badge-green">MBL / Prethermal</span>';
  }}
}}

function lineChart(canvasId, xs, ys, color, xLabel, yLabel, ymin_hint) {{
  var cv = document.getElementById(canvasId);
  var w = cv.offsetWidth || 420; var h = cv.offsetHeight || 220;
  cv.width = w; cv.height = h;
  var ctx = cv.getContext("2d");
  var pad = {{l:50, r:12, t:10, b:32}};
  var xmin = Math.min(...xs), xmax = Math.max(...xs);
  var ymin = ymin_hint !== undefined ? ymin_hint : Math.min(...ys);
  var ymax = Math.max(...ys);
  if (ymax === ymin) {{ ymax += 0.001; }}
  var ypad = (ymax - ymin) * 0.08;
  ymin -= ypad; ymax += ypad;
  var xr = xmax - xmin || 1; var yr = ymax - ymin;
  function px(x) {{ return pad.l + (x - xmin)/xr * (w - pad.l - pad.r); }}
  function py(y) {{ return h - pad.b - (y - ymin)/yr * (h - pad.t - pad.b); }}
  ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#21262d"; ctx.lineWidth = 1;
  for (var gi = 0; gi <= 4; gi++) {{
    var gy = py(ymin + yr * gi / 4);
    ctx.beginPath(); ctx.moveTo(pad.l, gy); ctx.lineTo(w - pad.r, gy); ctx.stroke();
    ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "right";
    ctx.fillText((ymin + yr * gi/4).toFixed(2), pad.l - 3, gy + 3);
  }}
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
  for (var i = 0; i < xs.length; i++) {{
    i === 0 ? ctx.moveTo(px(xs[i]), py(ys[i])) : ctx.lineTo(px(xs[i]), py(ys[i]));
  }}
  ctx.stroke();
  // x ticks
  for (var xi = 0; xi <= 5; xi++) {{
    var gx = px(xmin + xr * xi / 5);
    ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "center";
    ctx.fillText((xmin + xr * xi/5).toFixed(1), gx, h - pad.b + 12);
  }}
  ctx.fillStyle = "#6e7681"; ctx.font = "10px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(xLabel, w/2, h - 2);
  ctx.save(); ctx.translate(12, h/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign = "center"; ctx.fillText(yLabel, 0, 0); ctx.restore();
}}

var ts = DATA.frames.map(function(f) {{ return f.period; }});
var mz = DATA.frames.map(function(f) {{ return f.magnetization_z; }});
var en = DATA.frames.map(function(f) {{ return f.energy; }});
var sv = DATA.frames.map(function(f) {{ return f.entropy; }});

lineChart("cMz", ts, mz, "#58a6ff", "Period n", "⟨m_z⟩");
lineChart("cEnt", ts, sv, "#3fb950", "Period n", "S(n)", 0);
lineChart("cEn", ts, en, "#e3b341", "Period n", "⟨H₀⟩");

// Quasi-energy spectrum: plot level spacings as histogram
(function() {{
  var qe = DATA.quasi_energies;
  if (!qe || qe.length < 4) {{
    var cv = document.getElementById("cQE");
    cv.width = cv.offsetWidth || 420; cv.height = cv.offsetHeight || 220;
    var ctx = cv.getContext("2d");
    ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = "#8b949e"; ctx.font = "13px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("Spectrum: N too large for full diag", cv.width/2, cv.height/2);
    return;
  }}
  // Compute level spacings
  var sorted = qe.slice().sort(function(a,b) {{ return a-b; }});
  var spacings = [];
  for (var i = 0; i < sorted.length - 1; i++)
    spacings.push(sorted[i+1] - sorted[i]);
  var mean_s = spacings.reduce(function(a,b){{return a+b;}}, 0) / spacings.length;
  if (mean_s < 1e-14) mean_s = 1;
  var r_norm = spacings.map(function(s) {{ return s / mean_s; }});

  // Histogram
  var nbins = 20; var binmax = 3.0;
  var bins = new Array(nbins).fill(0);
  r_norm.forEach(function(s) {{
    var bi = Math.min(nbins - 1, Math.floor(s / binmax * nbins));
    bins[bi]++;
  }});
  var bmax = Math.max(...bins);

  var cv = document.getElementById("cQE");
  var w = cv.offsetWidth || 420; var h = cv.offsetHeight || 220;
  cv.width = w; cv.height = h;
  var ctx = cv.getContext("2d");
  var pad = 40; var bw = (w - 2*pad) / nbins;
  ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
  for (var i = 0; i < nbins; i++) {{
    var bh = bins[i] / bmax * (h - 2*pad);
    ctx.fillStyle = "#388bfd";
    ctx.fillRect(pad + i*bw + 1, h - pad - bh, bw - 2, bh);
  }}
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, h-pad); ctx.lineTo(w-pad, h-pad); ctx.stroke();
  // Reference lines: Poisson 0.39, GUE 0.60
  var r_poisson = 0.39, r_gue = 0.60;
  function grx(r) {{ return pad + r / binmax * (w - 2*pad); }}
  ctx.strokeStyle = "#3fb950"; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(grx(r_poisson), pad); ctx.lineTo(grx(r_poisson), h-pad); ctx.stroke();
  ctx.strokeStyle = "#f47067";
  ctx.beginPath(); ctx.moveTo(grx(r_gue), pad); ctx.lineTo(grx(r_gue), h-pad); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#3fb950"; ctx.font = "9px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("P", grx(r_poisson), pad - 3);
  ctx.fillStyle = "#f47067";
  ctx.fillText("GUE", grx(r_gue), pad - 3);
  ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "center";
  for (var xi = 0; xi <= 3; xi++) {{
    ctx.fillText((binmax * xi / 3).toFixed(1), pad + (w-2*pad)*xi/3, h-pad+12);
  }}
  ctx.fillText("r = gap ratio", w/2, h-2);
}})();
</script>
</body>
</html>"""


# ── M28-E: XXZ Heisenberg spin chain ─────────────────────────────────────────

def _heisenberg_frames(
    n_sites: int = 8,
    J: float = 1.0,
    Jz: float = 1.0,
    h: float = 0.0,
    scenario: str = "sweep",
    n_sweep: int = 20,
    Jz_min: float = -2.0,
    Jz_max: float = 3.0,
) -> dict:
  """Compute 1D XXZ Heisenberg chain via exact diagonalization (NumPy reference)."""
  import numpy as np
  from itertools import combinations

  N = n_sites
  n_down = N // 2  # Sz = 0 sector

  def spin_states(n_bits, nd):
    return [sum(1 << b for b in combo) for combo in combinations(range(n_bits), nd)]

  def build_H(Jz_val):
    states = spin_states(N, n_down)
    s_map = {s: i for i, s in enumerate(states)}
    dim = len(states)
    H = np.zeros((dim, dim), dtype=float)
    for bi, s in enumerate(states):
      # ZZ diagonal + longitudinal field
      for i in range(N - 1):
        sz_i = 0.5 if not ((s >> i) & 1) else -0.5
        sz_j = 0.5 if not ((s >> (i+1)) & 1) else -0.5
        H[bi, bi] += Jz_val / 4.0 * (2*sz_i) * (2*sz_j)  # Jz/4 * ZiZj
      if h != 0.0:
        for i in range(N):
          sz_i = 0.5 if not ((s >> i) & 1) else -0.5
          H[bi, bi] -= h / 2.0 * (2*sz_i)
      # XX + YY (= S+S- + S-S+) off-diagonal
      for i in range(N - 1):
        # S+_i S-_{i+1}: site i up (0), site i+1 down (1)
        i_up = not ((s >> i) & 1)
        j_down = (s >> (i+1)) & 1
        if i_up and j_down:
          s2 = s | (1 << i)
          s2 = s2 & ~(1 << (i+1))
          if s2 in s_map:
            H[bi, s_map[s2]] += J / 2.0
            H[s_map[s2], bi] += J / 2.0
    return H, states

  def spin_corr(states, psi):
    corr = np.zeros(N)
    s_map = {s: i for i, s in enumerate(states)}
    for bi, s in enumerate(states):
      p = psi[bi] ** 2
      sz0 = 0.5 if not (s & 1) else -0.5
      for r in range(N):
        # ZZ part
        sz_r = 0.5 if not ((s >> r) & 1) else -0.5
        corr[r] += p * sz0 * sz_r
    # XX+YY contributions via off-diagonal
    for bi, s in enumerate(states):
      for r in range(1, N):
        i_up = not (s & 1)
        j_down = (s >> r) & 1
        if i_up and j_down:
          s2 = s | 1
          s2 = s2 & ~(1 << r)
          if s2 in s_map:
            bj = s_map[s2]
            corr[r] += 0.5 * psi[bi] * psi[bj]  # XX part
    return corr

  if scenario == "sweep":
    sweep = []
    for k in range(n_sweep):
      Jz_k = Jz_min + (Jz_max - Jz_min) * k / max(n_sweep - 1, 1)
      H, states = build_H(Jz_k)
      evals, evecs = np.linalg.eigh(H)
      E0, E1 = evals[0], evals[1] if len(evals) > 1 else evals[0]
      psi = evecs[:, 0]
      corr = spin_corr(states, psi)
      spi = sum((-1)**r * corr[r] for r in range(N)) / N
      Sz_total = sum(
          (0.5 if not ((s >> i) & 1) else -0.5) * psi[bi]**2
          for bi, s in enumerate(states)
          for i in range(N)
      )
      sweep.append({
          "Jz": Jz_k,
          "ground_energy": float(E0),
          "energy_gap": float(E1 - E0),
          "structure_pi": float(spi),
          "Sz_total": float(Sz_total),
      })
    # Single-point at requested Jz
    H0, states0 = build_H(Jz)
    evals0, evecs0 = np.linalg.eigh(H0)
    psi0 = evecs0[:, 0]
    corr0 = spin_corr(states0, psi0)
    spi0 = sum((-1)**r * corr0[r] for r in range(N)) / N
    return {
        "n_sites": N, "J": J, "Jz": Jz, "h": h,
        "ground_energy": float(evals0[0]),
        "energy_gap": float(evals0[1] - evals0[0]) if len(evals0) > 1 else 0.0,
        "structure_pi": float(spi0),
        "spin_corr": corr0.tolist(),
        "sweep": sweep,
    }
  else:  # ground_state
    H0, states0 = build_H(Jz)
    evals0, evecs0 = np.linalg.eigh(H0)
    psi0 = evecs0[:, 0]
    corr0 = spin_corr(states0, psi0)
    spi0 = sum((-1)**r * corr0[r] for r in range(N)) / N
    return {
        "n_sites": N, "J": J, "Jz": Jz, "h": h,
        "ground_energy": float(evals0[0]),
        "energy_gap": float(evals0[1] - evals0[0]) if len(evals0) > 1 else 0.0,
        "structure_pi": float(spi0),
        "spin_corr": corr0.tolist(),
        "sweep": [],
    }


def _heisenberg_page(data: dict, title: str = "XXZ Heisenberg Chain") -> str:
  """Generate self-contained HTML for the 1D XXZ Heisenberg model."""
  import json as json_mod
  data_json = json_mod.dumps(data)
  N = data.get("n_sites", 8)

  return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif;padding:20px}}
h2{{font-size:1.2rem;font-weight:600;color:#58a6ff;margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:960px;margin:0 auto}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
canvas{{display:block;width:100%;height:240px}}
.info{{font-size:0.82rem;color:#8b949e;margin-top:8px}}
.param{{display:flex;gap:24px;flex-wrap:wrap;font-size:0.85rem;color:#8b949e;margin-bottom:16px;max-width:960px;margin-left:auto;margin-right:auto}}
.param span{{color:#58a6ff;font-weight:600}}
</style>
</head>
<body>
<h1 style="font-size:1.4rem;color:#58a6ff;margin-bottom:12px">{title}</h1>
<div class="param">
  <div>N = <span id="pN"></span> sites</div>
  <div>J = <span id="pJ"></span></div>
  <div>Jz = <span id="pJz"></span></div>
  <div>E₀ = <span id="pE0"></span></div>
  <div>Δ = <span id="pGap"></span></div>
  <div>S(π) = <span id="pSpi"></span></div>
</div>
<div class="grid">
  <div class="card"><h2>Spin-Spin Correlation ⟨S₀·Sᵣ⟩</h2><canvas id="cSpinCorr"></canvas>
    <div class="info">Antiferromagnetic: alternating for Jz &gt; 0; ferromagnetic for Jz &lt; 0</div></div>
  <div class="card"><h2>S(π) Spin Structure Factor vs Jz</h2><canvas id="cSpi"></canvas>
    <div class="info">Peaks in antiferromagnetic phase (Jz &gt; −J); vanishes in ferromagnetic phase</div></div>
  <div class="card"><h2>Energy Gap Δ vs Jz</h2><canvas id="cGap"></canvas>
    <div class="info">Gapless for |Jz| ≤ J (XXZ critical line); gapped outside</div></div>
  <div class="card"><h2>Ground Energy E₀ vs Jz</h2><canvas id="cEnergy"></canvas>
    <div class="info">Ground energy decreases with Jz in the antiferromagnetic regime</div></div>
</div>
<script>
const DATA = {data_json};

document.getElementById("pN").textContent   = DATA.n_sites;
document.getElementById("pJ").textContent   = DATA.J.toFixed(3);
document.getElementById("pJz").textContent  = DATA.Jz.toFixed(3);
document.getElementById("pE0").textContent  = DATA.ground_energy.toFixed(4);
document.getElementById("pGap").textContent = DATA.energy_gap.toFixed(4);
document.getElementById("pSpi").textContent = DATA.structure_pi.toFixed(4);

function barChart(canvasId, labels, values, colorFn, yLabel) {{
  var cv = document.getElementById(canvasId);
  var w = cv.offsetWidth || 400; var h = cv.offsetHeight || 240;
  cv.width = w; cv.height = h;
  var ctx = cv.getContext("2d");
  var pad = 44;
  var bw = Math.max(1, (w - 2*pad) / values.length - 2);
  var vmax = Math.max(...values.map(Math.abs)) * 1.2 || 1;
  var vmin = Math.min(0, ...values) * 1.2;
  var range = Math.max(vmax - vmin, 1e-9);
  var zero_y = h - pad - (-vmin / range) * (h - 2*pad);
  ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, h-pad); ctx.lineTo(w-pad, h-pad); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad, zero_y); ctx.lineTo(w-pad, zero_y);
  ctx.strokeStyle = "#444c56"; ctx.stroke();
  for (var i = 0; i < values.length; i++) {{
    var x = pad + i * ((w - 2*pad) / values.length) + 1;
    var bh = Math.abs(values[i]) / range * (h - 2*pad);
    var y = values[i] >= 0 ? zero_y - bh : zero_y;
    ctx.fillStyle = colorFn(i, values[i]);
    ctx.fillRect(x, y, bw, bh);
    if (labels && labels[i] !== undefined) {{
      ctx.fillStyle = "#8b949e"; ctx.font = "10px monospace"; ctx.textAlign = "center";
      ctx.fillText(labels[i], x + bw/2, h - pad + 12);
    }}
  }}
  if (yLabel) {{
    ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "left";
    ctx.fillText(yLabel, 4, 14);
  }}
}}

function lineChart(canvasId, xs, ys, color, yLabel, xLabel) {{
  var cv = document.getElementById(canvasId);
  var w = cv.offsetWidth || 400; var h = cv.offsetHeight || 240;
  cv.width = w; cv.height = h;
  var ctx = cv.getContext("2d");
  var pad = 44;
  var xmin = Math.min(...xs), xmax = Math.max(...xs);
  var ymin = Math.min(...ys), ymax = Math.max(...ys);
  var xr = Math.max(xmax - xmin, 1e-9); var yr = Math.max(ymax - ymin, 1e-9) * 1.1;
  ymin -= yr * 0.05; ymax = ymin + yr;
  function px(x) {{ return pad + (x - xmin)/xr * (w - 2*pad); }}
  function py(y) {{ return h - pad - (y - ymin)/yr * (h - 2*pad); }}
  ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, h-pad); ctx.lineTo(w-pad, h-pad); ctx.stroke();
  for (var yi = 0; yi <= 4; yi++) {{
    var gv = ymin + yr * yi/4;
    var gy = py(gv);
    ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(w-pad, gy);
    ctx.strokeStyle = "#21262d"; ctx.stroke();
    ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "right";
    ctx.fillText(gv.toFixed(2), pad - 3, gy + 3);
  }}
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
  for (var i = 0; i < xs.length; i++) {{
    i === 0 ? ctx.moveTo(px(xs[i]), py(ys[i])) : ctx.lineTo(px(xs[i]), py(ys[i]));
  }}
  ctx.stroke();
  for (var xi = 0; xi <= 4; xi++) {{
    var gx = px(xmin + xr * xi/4);
    ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "center";
    ctx.fillText((xmin + xr * xi/4).toFixed(1), gx, h - pad + 12);
  }}
  if (yLabel) {{ ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "left"; ctx.fillText(yLabel, 4, 14); }}
  if (xLabel) {{ ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "center"; ctx.fillText(xLabel, w/2, h - 4); }}
}}

// Spin-spin correlation bar chart
var scLabels = DATA.spin_corr.map(function(_, i) {{ return "r=" + i; }});
barChart("cSpinCorr", scLabels, DATA.spin_corr,
  function(i, v) {{ return v >= 0 ? "#388bfd" : "#f47067"; }}, "⟨S₀·Sᵣ⟩");

// Sweep line charts
if (DATA.sweep && DATA.sweep.length > 0) {{
  var swJz  = DATA.sweep.map(function(p) {{ return p.Jz; }});
  var swSpi = DATA.sweep.map(function(p) {{ return p.structure_pi; }});
  var swGap = DATA.sweep.map(function(p) {{ return p.energy_gap; }});
  var swE0  = DATA.sweep.map(function(p) {{ return p.ground_energy; }});
  lineChart("cSpi",    swJz, swSpi, "#f47067", "S(π)",  "Jz/J");
  lineChart("cGap",    swJz, swGap, "#e3b341", "Gap Δ", "Jz/J");
  lineChart("cEnergy", swJz, swE0,  "#57ab5a", "E₀",    "Jz/J");
}}
</script>
</body>
</html>"""


# ── M28-F: Bose-Hubbard model ─────────────────────────────────────────────────

def _bose_hubbard_frames(
    n_sites: int = 4,
    n_bosons: int = 4,
    n_max: int = 4,
    J: float = 0.1,
    U: float = 1.0,
    mu: float = 0.0,
    scenario: str = "sweep",
    n_sweep: int = 20,
    J_min: float = 0.0,
    J_max: float = 0.5,
) -> dict:
  """Compute Bose-Hubbard model via exact diagonalization (NumPy reference)."""
  import numpy as np

  N = n_sites

  def enum_states(n_sites_loc, n_bosons_loc, n_max_loc):
    result = []
    state = [0] * n_sites_loc
    def dfs(site, remaining):
      if site == n_sites_loc - 1:
        if remaining <= n_max_loc:
          state[site] = remaining
          result.append(tuple(state))
        return
      lo = max(0, remaining - n_max_loc * (n_sites_loc - 1 - site))
      hi = min(remaining, n_max_loc)
      for n in range(lo, hi + 1):
        state[site] = n
        dfs(site + 1, remaining - n)
    dfs(0, n_bosons_loc)
    return sorted(result)

  def build_H(J_val):
    states = enum_states(N, n_bosons, n_max)
    s_map = {s: i for i, s in enumerate(states)}
    dim = len(states)
    H = np.zeros((dim, dim), dtype=float)
    for bi, s in enumerate(states):
      diag = sum(U/2 * s[i]*(s[i]-1) - mu*s[i] for i in range(N))
      H[bi, bi] = diag
      for i in range(N - 1):
        if s[i+1] >= 1:
          s2 = list(s); s2[i] += 1; s2[i+1] -= 1
          t = tuple(s2)
          if t in s_map:
            mel = -J_val * np.sqrt(float(s[i+1]) * float(s[i]+1))
            bj = s_map[t]
            H[bi, bj] += mel
            H[bj, bi] += mel
    return H, states

  def compute_obs(states, psi):
    occ = np.zeros(N)
    double_occ = 0.0
    num_dist = np.zeros(n_max + 1)
    for bi, s in enumerate(states):
      p = psi[bi] ** 2
      for i in range(N):
        occ[i] += p * s[i]
        double_occ += p * s[i] * (s[i]-1)
      if s[0] <= n_max:
        num_dist[s[0]] += p
    double_occ /= N
    return occ, double_occ, num_dist

  def superfluid_order(states, psi):
    s_map = {s: i for i, s in enumerate(states)}
    sf = 0.0
    for site in range(N):
      re_a = 0.0
      for bi, s in enumerate(states):
        if s[site] == 0: continue
        s2 = list(s); mel = np.sqrt(float(s2[site])); s2[site] -= 1; t = tuple(s2)
        if t in s_map:
          re_a += psi[bi] * mel * psi[s_map[t]]
      sf += abs(re_a)
    return sf / N

  if scenario == "sweep":
    sweep = []
    for k in range(n_sweep):
      J_k = J_min + (J_max - J_min) * k / max(n_sweep - 1, 1)
      H, states = build_H(J_k)
      evals, evecs = np.linalg.eigh(H)
      psi = evecs[:, 0]
      _, docc, _ = compute_obs(states, psi)
      sf = superfluid_order(states, psi)
      E1 = evals[1] if len(evals) > 1 else evals[0]
      sweep.append({
          "J": float(J_k),
          "ground_energy": float(evals[0]),
          "energy_gap": float(E1 - evals[0]),
          "superfluid_op": float(sf),
          "double_occupancy": float(docc),
      })
    # Single-point at requested J
    H0, states0 = build_H(J)
    evals0, evecs0 = np.linalg.eigh(H0)
    psi0 = evecs0[:, 0]
    occ0, docc0, ndist0 = compute_obs(states0, psi0)
    sf0 = superfluid_order(states0, psi0)
    return {
        "n_sites": N, "n_bosons": n_bosons, "n_max": n_max,
        "J": J, "U": U, "mu": mu,
        "ground_energy": float(evals0[0]),
        "energy_gap": float(evals0[1] - evals0[0]) if len(evals0) > 1 else 0.0,
        "superfluid_op": float(sf0),
        "double_occupancy": float(docc0),
        "occupation": occ0.tolist(),
        "number_dist": ndist0.tolist(),
        "sweep": sweep,
    }
  else:  # ground_state
    H0, states0 = build_H(J)
    evals0, evecs0 = np.linalg.eigh(H0)
    psi0 = evecs0[:, 0]
    occ0, docc0, ndist0 = compute_obs(states0, psi0)
    sf0 = superfluid_order(states0, psi0)
    return {
        "n_sites": N, "n_bosons": n_bosons, "n_max": n_max,
        "J": J, "U": U, "mu": mu,
        "ground_energy": float(evals0[0]),
        "energy_gap": float(evals0[1] - evals0[0]) if len(evals0) > 1 else 0.0,
        "superfluid_op": float(sf0),
        "double_occupancy": float(docc0),
        "occupation": occ0.tolist(),
        "number_dist": ndist0.tolist(),
        "sweep": [],
    }


def _bose_hubbard_page(data: dict, title: str = "Bose-Hubbard Model") -> str:
  """Generate self-contained HTML for the Bose-Hubbard model."""
  import json as json_mod
  data_json = json_mod.dumps(data)

  return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif;padding:20px}}
h2{{font-size:1.2rem;font-weight:600;color:#58a6ff;margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:960px;margin:0 auto}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
canvas{{display:block;width:100%;height:240px}}
.info{{font-size:0.82rem;color:#8b949e;margin-top:8px}}
.param{{display:flex;gap:24px;flex-wrap:wrap;font-size:0.85rem;color:#8b949e;margin-bottom:16px;max-width:960px;margin-left:auto;margin-right:auto}}
.param span{{color:#58a6ff;font-weight:600}}
</style>
</head>
<body>
<h1 style="font-size:1.4rem;color:#58a6ff;margin-bottom:12px">{title}</h1>
<div class="param">
  <div>N = <span id="pN"></span> sites</div>
  <div>M = <span id="pM"></span> bosons</div>
  <div>n_max = <span id="pNmax"></span></div>
  <div>J = <span id="pJ"></span></div>
  <div>U = <span id="pU"></span></div>
  <div>J/U = <span id="pJU"></span></div>
  <div>E₀ = <span id="pE0"></span></div>
  <div>Δ = <span id="pGap"></span></div>
  <div>|⟨a⟩| = <span id="pSF"></span></div>
  <div>D = <span id="pDocc"></span></div>
</div>
<div class="grid">
  <div class="card"><h2>Site Occupation ⟨nᵢ⟩</h2><canvas id="cOcc"></canvas>
    <div class="info">Uniform in superfluid phase; integer-locked in Mott phase</div></div>
  <div class="card"><h2>Number Distribution P(n) at site 0</h2><canvas id="cNdist"></canvas>
    <div class="info">Sharp Fock peak in Mott phase; broad distribution in superfluid phase</div></div>
  <div class="card"><h2>Superfluid Order Parameter |⟨a⟩| vs J/U</h2><canvas id="cSF"></canvas>
    <div class="info">|⟨a⟩| = 0 in Mott insulator; grows above critical J/U</div></div>
  <div class="card"><h2>Energy Gap Δ vs J/U</h2><canvas id="cGap"></canvas>
    <div class="info">Gap closes at Mott-superfluid transition</div></div>
</div>
<script>
const DATA = {data_json};

document.getElementById("pN").textContent    = DATA.n_sites;
document.getElementById("pM").textContent    = DATA.n_bosons;
document.getElementById("pNmax").textContent = DATA.n_max;
document.getElementById("pJ").textContent    = DATA.J.toFixed(3);
document.getElementById("pU").textContent    = DATA.U.toFixed(3);
document.getElementById("pJU").textContent   = (DATA.U > 0 ? DATA.J / DATA.U : 0).toFixed(3);
document.getElementById("pE0").textContent   = DATA.ground_energy.toFixed(4);
document.getElementById("pGap").textContent  = DATA.energy_gap.toFixed(4);
document.getElementById("pSF").textContent   = DATA.superfluid_op.toFixed(4);
document.getElementById("pDocc").textContent = DATA.double_occupancy.toFixed(4);

function barChart(canvasId, labels, values, colorFn, yLabel) {{
  var cv = document.getElementById(canvasId);
  var w = cv.offsetWidth || 400; var h = cv.offsetHeight || 240;
  cv.width = w; cv.height = h;
  var ctx = cv.getContext("2d");
  var pad = 44;
  var bw = Math.max(1, (w - 2*pad) / Math.max(values.length,1) - 2);
  var vmax = Math.max(...values.map(Math.abs)) * 1.2 || 1;
  var range = Math.max(vmax, 1e-9);
  ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, h-pad); ctx.lineTo(w-pad, h-pad); ctx.stroke();
  for (var i = 0; i < values.length; i++) {{
    var x = pad + i * ((w - 2*pad) / values.length) + 1;
    var bh = values[i] / range * (h - 2*pad);
    ctx.fillStyle = colorFn(i, values[i]);
    ctx.fillRect(x, h - pad - bh, bw, bh);
    if (labels && labels[i] !== undefined) {{
      ctx.fillStyle = "#8b949e"; ctx.font = "10px monospace"; ctx.textAlign = "center";
      ctx.fillText(labels[i], x + bw/2, h - pad + 12);
    }}
  }}
  if (yLabel) {{
    ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "left";
    ctx.fillText(yLabel, 4, 14);
  }}
}}

function lineChart(canvasId, xs, ys, color, yLabel, xLabel) {{
  var cv = document.getElementById(canvasId);
  var w = cv.offsetWidth || 400; var h = cv.offsetHeight || 240;
  cv.width = w; cv.height = h;
  var ctx = cv.getContext("2d");
  var pad = 44;
  var xmin = Math.min(...xs), xmax = Math.max(...xs);
  var ymin = Math.min(0, ...ys), ymax = Math.max(...ys) * 1.1 || 1;
  var xr = Math.max(xmax - xmin, 1e-9); var yr = Math.max(ymax - ymin, 1e-9);
  function px(x) {{ return pad + (x - xmin)/xr * (w - 2*pad); }}
  function py(y) {{ return h - pad - (y - ymin)/yr * (h - 2*pad); }}
  ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, h-pad); ctx.lineTo(w-pad, h-pad); ctx.stroke();
  for (var yi = 0; yi <= 4; yi++) {{
    var gv = ymin + yr * yi/4; var gy = py(gv);
    ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(w-pad, gy);
    ctx.strokeStyle = "#21262d"; ctx.stroke();
    ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "right";
    ctx.fillText(gv.toFixed(3), pad - 3, gy + 3);
  }}
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
  for (var i = 0; i < xs.length; i++) {{
    i === 0 ? ctx.moveTo(px(xs[i]), py(ys[i])) : ctx.lineTo(px(xs[i]), py(ys[i]));
  }}
  ctx.stroke();
  for (var xi = 0; xi <= 4; xi++) {{
    var gx = px(xmin + xr * xi/4);
    ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "center";
    ctx.fillText((xmin + xr * xi/4).toFixed(3), gx, h - pad + 12);
  }}
  if (yLabel) {{ ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "left"; ctx.fillText(yLabel, 4, 14); }}
  if (xLabel) {{ ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "center"; ctx.fillText(xLabel, w/2, h - 4); }}
}}

// Site occupation
var occLabels = DATA.occupation.map(function(_, i) {{ return "i=" + i; }});
barChart("cOcc", occLabels, DATA.occupation, function() {{ return "#57ab5a"; }}, "⟨nᵢ⟩");

// Number distribution
var nLabels = DATA.number_dist.map(function(_, n) {{ return "n=" + n; }});
barChart("cNdist", nLabels, DATA.number_dist, function(i, v) {{ return "#bc8cff"; }}, "P(n)");

// Sweep charts
if (DATA.sweep && DATA.sweep.length > 0) {{
  var swJU = DATA.sweep.map(function(p) {{ return DATA.U > 0 ? p.J / DATA.U : p.J; }});
  var swSF = DATA.sweep.map(function(p) {{ return p.superfluid_op; }});
  var swGap = DATA.sweep.map(function(p) {{ return p.energy_gap; }});
  lineChart("cSF",  swJU, swSF,  "#388bfd", "|⟨a⟩|",  "J/U");
  lineChart("cGap", swJU, swGap, "#e3b341", "Gap Δ",   "J/U");
}}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# M29-G: Transverse-Field Ising  —  Python reference solver + page
# ─────────────────────────────────────────────────────────────────────────────

def _tfi_frames(
    n_sites: int = 10,
    J: float = 1.0,
    h: float = 0.5,
    scenario: str = "sweep",
    n_sweep: int = 30,
    h_min: float = 0.0,
    h_max: float = 2.0,
) -> dict:
    """NumPy exact-diag TFI reference.  Returns dict with sweep + single-point data."""
    import numpy as np  # noqa: PLC0415

    N = n_sites

    def build_h(J_: float, h_: float) -> "np.ndarray":
        dim = 1 << N
        H = np.zeros((dim, dim), dtype=float)
        for s in range(dim):
            # -J Z_i Z_{i+1}
            diag = 0.0
            for i in range(N - 1):
                zi = -1 if (s >> i) & 1 else 1
                zj = -1 if (s >> (i + 1)) & 1 else 1
                diag += -J_ * zi * zj
            H[s, s] = diag
            # -h X_i
            for i in range(N):
                s2 = s ^ (1 << i)
                H[s, s2] -= h_
        return H

    def solve_point(h_: float):
        H = build_h(J, h_)
        evals, evecs = np.linalg.eigh(H)
        psi0 = evecs[:, 0]
        E0, E1 = evals[0], evals[1] if len(evals) > 1 else evals[0]
        # Order parameter |<Z_0>|
        dim = 1 << N
        Z0 = np.array([-1 if s & 1 else 1 for s in range(dim)], dtype=float)
        op = abs(float(psi0 @ (Z0 * psi0)))
        # Spin-spin corr <Z_0 Z_r>
        corr = []
        for r in range(N):
            Zr = np.array([-1 if (s >> r) & 1 else 1 for s in range(dim)], dtype=float)
            corr.append(float(psi0 @ (Z0 * Zr * psi0)))
        # Correlation length
        xi = 0.0
        xs, ys = [], []
        for r in range(1, N // 2):
            c = abs(corr[r])
            if c > 1e-12:
                xs.append(r)
                ys.append(np.log(c))
        if len(xs) >= 2:
            slope, _ = np.polyfit(xs, ys, 1)
            xi = -1.0 / slope if slope < -1e-12 else 0.0
        return E0, E1 - E0, op, xi, corr

    E0, gap, op, xi, spin_corr = solve_point(h)

    sweep = []
    for k in range(n_sweep):
        hk = h_min + (h_max - h_min) * k / (n_sweep - 1)
        e0k, gapk, opk, xik, _ = solve_point(hk)
        sweep.append({"h": hk, "ground_energy": e0k, "energy_gap": gapk,
                      "order_param": opk, "corr_len": xik})

    return {
        "n_sites": N, "J": J, "h": h,
        "ground_energy": E0, "energy_gap": gap,
        "order_param": op, "corr_len": xi,
        "spin_corr": spin_corr,
        "sweep": sweep,
    }


def _tfi_page(data: dict, title: str = "Transverse-Field Ising Model") -> str:
    """4-panel TFI page: spin-spin corr, order param vs h, gap vs h, corr length vs h."""
    import json as _json  # noqa: PLC0415

    jdata = _json.dumps(data)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body{{margin:0;background:#0d1117;color:#e6edf3;font-family:monospace;}}
  h1{{text-align:center;padding:16px;font-size:1.1rem;color:#388bfd;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;}}
  canvas{{width:100%;height:220px;display:block;}}
  .label{{font-size:0.7rem;color:#8b949e;text-align:center;margin-top:4px;}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="grid">
  <div class="panel"><canvas id="cCorr"></canvas><div class="label">Spin-spin correlator ⟨Z₀Zᵣ⟩</div></div>
  <div class="panel"><canvas id="cOP"></canvas><div class="label">Order parameter |⟨Z₀⟩| vs h/J</div></div>
  <div class="panel"><canvas id="cGap"></canvas><div class="label">Energy gap Δ vs h/J (closes at h=J)</div></div>
  <div class="panel"><canvas id="cXi"></canvas><div class="label">Correlation length ξ vs h/J</div></div>
</div>
<script>
var DATA = {jdata};

function lineChart(id, xs, ys, color, ylabel, xlabel) {{
  var cv = document.getElementById(id);
  if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var xmin = Math.min.apply(null,xs), xmax = Math.max.apply(null,xs);
  var ymin = Math.min.apply(null,ys), ymax = Math.max.apply(null,ys);
  var xr = xmax-xmin||1, yr = ymax-ymin||1;
  ctx.strokeStyle = "#21262d"; ctx.lineWidth = 1;
  for (var gi=0; gi<5; gi++) {{
    var gy = pad + (H-2*pad)*gi/4;
    ctx.beginPath(); ctx.moveTo(pad,gy); ctx.lineTo(W-pad,gy); ctx.stroke();
  }}
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
  for (var i=0; i<xs.length; i++) {{
    var px = pad + (xs[i]-xmin)/xr*(W-2*pad);
    var py = H-pad - (ys[i]-ymin)/yr*(H-2*pad);
    if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace";
  ctx.fillText(ylabel,4,14);
  ctx.fillText(xlabel,W-pad,H-4);
  ctx.fillText(ymin.toFixed(3),2,H-pad);
  ctx.fillText(ymax.toFixed(3),2,pad+4);
}}

function barChart(id, labels, vals, colorFn, ylabel) {{
  var cv = document.getElementById(id);
  if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var n = vals.length;
  var ymax = Math.max.apply(null, vals.map(Math.abs));
  if (ymax === 0) ymax = 1;
  var bw = (W - 2*pad) / n;
  for (var i=0; i<n; i++) {{
    var v = vals[i];
    var bh = Math.abs(v)/ymax*(H-2*pad);
    var bx = pad + i*bw + bw*0.1;
    var by = v >= 0 ? H-pad-bh : H-pad;
    ctx.fillStyle = colorFn(i, v);
    ctx.fillRect(bx, by, bw*0.8, bh);
    if (n <= 16) {{
      ctx.fillStyle="#8b949e"; ctx.font="9px monospace";
      ctx.fillText(labels[i], bx, H-pad+12);
    }}
  }}
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace";
  ctx.fillText(ylabel,4,14);
  // Zero line
  ctx.strokeStyle="#444"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
}}

// Panel 1: spin-spin correlator
var rLabels = DATA.spin_corr.map(function(_,r){{ return "r="+r; }});
barChart("cCorr", rLabels, DATA.spin_corr,
  function(i,v) {{ return v >= 0 ? "#57ab5a" : "#e3b341"; }}, "⟨Z₀Zᵣ⟩");

// Panels 2-4: sweeps
if (DATA.sweep && DATA.sweep.length > 0) {{
  var swH  = DATA.sweep.map(function(p) {{ return p.h / DATA.J; }});
  var swOP = DATA.sweep.map(function(p) {{ return p.order_param; }});
  var swGp = DATA.sweep.map(function(p) {{ return p.energy_gap; }});
  var swXi = DATA.sweep.map(function(p) {{ return p.corr_len; }});
  lineChart("cOP",  swH, swOP, "#57ab5a", "|⟨Z₀⟩|", "h/J");
  lineChart("cGap", swH, swGp, "#e3b341", "Gap Δ",   "h/J");
  lineChart("cXi",  swH, swXi, "#bc8cff", "ξ",       "h/J");
}}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# M29-H: SSH Model  —  Python reference solver + page
# ─────────────────────────────────────────────────────────────────────────────

def _ssh_frames(
    n_cells: int = 10,
    t: float = 1.0,
    delta: float = 0.5,
    scenario: str = "sweep",
    n_sweep: int = 30,
    delta_min: float = -0.9,
    delta_max: float = 0.9,
) -> dict:
    """NumPy SSH reference solver.  Returns spectrum + Zak phase + sweep."""
    import numpy as np  # noqa: PLC0415

    N = 2 * n_cells

    def build_h(delta_: float) -> "np.ndarray":
        t1 = t * (1.0 - delta_)
        t2 = t * (1.0 + delta_)
        H = np.zeros((N, N), dtype=float)
        for i in range(N - 1):
            tij = t1 if i % 2 == 0 else t2
            H[i, i + 1] = -tij
            H[i + 1, i] = -tij
        return H

    def zak_phase(delta_: float) -> float:
        """Discrete Wilson loop in k-space for occupied band."""
        nk = n_cells
        t1 = t * (1.0 - delta_)
        t2 = t * (1.0 + delta_)
        W = 1.0 + 0.0j
        u_prev = None
        u_first = None
        for m in range(nk):
            k = 2.0 * np.pi * m / nk
            dk = -t1 + (-t2) * np.exp(1j * k)
            absdk = abs(dk)
            if absdk < 1e-14:
                u = np.array([1.0, 0.0], dtype=complex)
            else:
                # lower eigenvector: [d*, -|d|]/sqrt(2|d|)  (unnormalized d*/|d|, -1)
                phase = dk / absdk
                u = np.array([np.conj(phase), -1.0], dtype=complex)
                u /= np.linalg.norm(u)
            if u_prev is not None:
                ov = np.dot(np.conj(u_prev), u)
                if abs(ov) > 1e-14:
                    W *= ov / abs(ov)
            if m == 0:
                u_first = u
            u_prev = u
        # Close the loop
        if u_first is not None and u_prev is not None:
            ov = np.dot(np.conj(u_prev), u_first)
            if abs(ov) > 1e-14:
                W *= ov / abs(ov)
        zak = -np.angle(W)
        return float(np.pi if abs(zak) > np.pi / 2 else 0.0)

    def solve_point(delta_: float) -> dict:
        H = build_h(delta_)
        evals, evecs = np.linalg.eigh(H)
        # Bulk gap: edge states in the topological phase have |E| exponentially
        # small (~(t2/t1)^{-n_cells}), while bulk bands start at ~min(t1, t2).
        # Use 0.3*min(t1, t2) to separate edge states from bulk.
        t1_abs = abs(t * (1.0 - delta_))
        t2_abs = abs(t * (1.0 + delta_))
        edge_thr = max(0.3 * min(t1_abs, t2_abs), 1e-10)
        neg = evals[evals < -edge_thr]
        pos = evals[evals >  edge_thr]
        gap = float(pos.min() - neg.max()) if len(neg) and len(pos) else 0.0
        # Edge mode: any eigenvalue with |E| < edge_thr is an in-gap (edge) state
        mid_idx = np.where(np.abs(evals) < edge_thr)[0]
        ew = np.zeros(N)
        if len(mid_idx):
            best = mid_idx[np.argmin(np.abs(evals[mid_idx]))]
            ew = evecs[:, best] ** 2
        zp = zak_phase(delta_)
        return {"energies": evals.tolist(), "gap": gap, "zak_phase": zp, "edge_weight": ew.tolist()}

    sol = solve_point(delta)

    sweep = []
    for k in range(n_sweep):
        dk = delta_min + (delta_max - delta_min) * k / (n_sweep - 1)
        if abs(dk) >= 1.0:
            continue
        s = solve_point(dk)
        ew_boundary = s["edge_weight"][0] + s["edge_weight"][1] if len(s["edge_weight"]) >= 2 else 0.0
        sweep.append({"delta": dk, "gap": s["gap"], "zak_phase": s["zak_phase"],
                      "edge_weight": ew_boundary})

    return {
        "n_cells": n_cells, "t": t, "delta": delta,
        "energies": sol["energies"],
        "gap": sol["gap"],
        "zak_phase": sol["zak_phase"],
        "edge_weight": sol["edge_weight"],
        "sweep": sweep,
    }


def _ssh_page(data: dict, title: str = "SSH Topological Insulator") -> str:
    """4-panel SSH page: spectrum, edge weight profile, gap vs δ, Zak phase vs δ."""
    import json as _json  # noqa: PLC0415

    jdata = _json.dumps(data)
    zak_label = "π (topological)" if abs(data.get("zak_phase", 0)) > 1.0 else "0 (trivial)"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body{{margin:0;background:#0d1117;color:#e6edf3;font-family:monospace;}}
  h1{{text-align:center;padding:16px;font-size:1.1rem;color:#388bfd;}}
  .subtitle{{text-align:center;font-size:0.8rem;color:#8b949e;margin-bottom:8px;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;}}
  canvas{{width:100%;height:220px;display:block;}}
  .label{{font-size:0.7rem;color:#8b949e;text-align:center;margin-top:4px;}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:bold;}}
  .topo{{background:#1c3a2c;color:#57ab5a;border:1px solid #57ab5a;}}
  .trivial{{background:#2c1c1c;color:#e3b341;border:1px solid #e3b341;}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">
  δ = {data.get('delta', 0):.3f} &nbsp;|&nbsp;
  t = {data.get('t', 1):.2f} &nbsp;|&nbsp;
  gap = {data.get('gap', 0):.4f} &nbsp;|&nbsp;
  Zak phase = <span class="badge {'topo' if abs(data.get('zak_phase',0)) > 1.0 else 'trivial'}">{zak_label}</span>
</div>
<div class="grid">
  <div class="panel"><canvas id="cSpec"></canvas><div class="label">Single-particle spectrum (all bands)</div></div>
  <div class="panel"><canvas id="cEdge"></canvas><div class="label">Edge mode |ψ|² per site</div></div>
  <div class="panel"><canvas id="cGap"></canvas><div class="label">Band gap vs δ</div></div>
  <div class="panel"><canvas id="cZak"></canvas><div class="label">Zak phase vs δ (0 or π)</div></div>
</div>
<script>
var DATA = {jdata};

function lineChart(id, xs, ys, color, ylabel, xlabel) {{
  var cv = document.getElementById(id);
  if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var xmin = Math.min.apply(null,xs), xmax = Math.max.apply(null,xs);
  var ymin = Math.min.apply(null,ys), ymax = Math.max.apply(null,ys);
  var xr = xmax-xmin||1, yr = ymax-ymin||1;
  ctx.strokeStyle = "#21262d"; ctx.lineWidth = 1;
  for (var gi=0; gi<5; gi++) {{
    var gy = pad + (H-2*pad)*gi/4;
    ctx.beginPath(); ctx.moveTo(pad,gy); ctx.lineTo(W-pad,gy); ctx.stroke();
  }}
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
  for (var i=0; i<xs.length; i++) {{
    var px = pad + (xs[i]-xmin)/xr*(W-2*pad);
    var py = H-pad - (ys[i]-ymin)/yr*(H-2*pad);
    if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace";
  ctx.fillText(ylabel,4,14);
  ctx.fillText(xlabel,W-pad,H-4);
  ctx.fillText(ymin.toFixed(3),2,H-pad);
  ctx.fillText(ymax.toFixed(3),2,pad+4);
}}

function barChart(id, labels, vals, colorFn, ylabel) {{
  var cv = document.getElementById(id);
  if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var n = vals.length;
  var ymax = Math.max.apply(null, vals.map(Math.abs));
  if (ymax === 0) ymax = 1;
  var bw = (W - 2*pad) / n;
  for (var i=0; i<n; i++) {{
    var v = vals[i];
    var bh = Math.abs(v)/ymax*(H-2*pad);
    var bx = pad + i*bw + bw*0.1;
    var by = H-pad-bh;
    ctx.fillStyle = colorFn(i, v);
    ctx.fillRect(bx, by, bw*0.8, bh);
    if (n <= 24) {{
      ctx.fillStyle="#8b949e"; ctx.font="9px monospace";
      ctx.fillText(i, bx, H-pad+12);
    }}
  }}
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace"; ctx.fillText(ylabel,4,14);
}}

// Panel 1: Spectrum as bar chart (index vs energy)
var eIdx = DATA.energies.map(function(_,i){{ return ""+i; }});
barChart("cSpec", eIdx, DATA.energies,
  function(i,v) {{
    var mid = DATA.energies.length/2;
    if (Math.abs(v) < DATA.gap/4) return "#e3b341";  // mid-gap = gold
    return v < 0 ? "#388bfd" : "#bc8cff";
  }}, "E");

// Panel 2: Edge mode |ψ|² per site
var siteLabels = DATA.edge_weight.map(function(_,i){{ return ""+i; }});
barChart("cEdge", siteLabels, DATA.edge_weight,
  function(i,v) {{ return i < 2 || i >= DATA.edge_weight.length-2 ? "#57ab5a" : "#30363d"; }},
  "|ψ|²");

// Panels 3-4: Sweeps
if (DATA.sweep && DATA.sweep.length > 0) {{
  var swD  = DATA.sweep.map(function(p) {{ return p.delta; }});
  var swGp = DATA.sweep.map(function(p) {{ return p.gap; }});
  var swZp = DATA.sweep.map(function(p) {{ return p.zak_phase; }});
  lineChart("cGap", swD, swGp, "#e3b341", "Gap Δ", "δ");
  lineChart("cZak", swD, swZp, "#57ab5a", "θ_Zak", "δ");
}}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# M30-I: Kitaev Chain  —  Python reference solver + page
# ─────────────────────────────────────────────────────────────────────────────

def _kitaev_frames(
    n_sites: int = 20,
    t: float = 1.0,
    delta: float = 1.0,
    mu: float = 0.0,
    scenario: str = "sweep",
    n_sweep: int = 40,
    mu_min: float = -4.0,
    mu_max: float = 4.0,
) -> dict:
    """NumPy BdG reference for Kitaev chain.  Topo phase: |μ| < 2t."""
    import numpy as np  # noqa: PLC0415

    N = n_sites

    def build_bdg(mu_: float) -> "np.ndarray":
        dim = 2 * N
        H = np.zeros((dim, dim), dtype=float)
        # On-site: -μ/2 in (c,c) block, +μ/2 in (c†,c†) block
        for i in range(N):
            H[i, i]         = -mu_ / 2.0
            H[i + N, i + N] =  mu_ / 2.0
        for i in range(N - 1):
            # Hopping
            H[i, i + 1] += -t / 2.0; H[i + 1, i] += -t / 2.0
            H[i + N, i + 1 + N] += t / 2.0; H[i + 1 + N, i + N] += t / 2.0
            # Pairing
            H[i, i + 1 + N]     += +delta / 2.0
            H[i + 1 + N, i]     += +delta / 2.0
            H[i + 1, i + N]     += -delta / 2.0
            H[i + N, i + 1]     += -delta / 2.0
        return H

    def solve_point(mu_: float) -> dict:
        H = build_bdg(mu_)
        evals, evecs = np.linalg.eigh(H)
        # Gap: smallest positive quasi-energy
        pos = evals[evals > 1e-10]
        gap = float(pos.min()) if len(pos) else 0.0
        # Topological invariant from near-zero modes
        thr = max(0.3 * min(abs(t), abs(delta)), 1e-8)
        n_zero = int(np.sum(np.abs(evals) < thr))
        topo_inv = -1.0 if n_zero > 0 else 1.0
        # Majorana weight: collect ALL near-zero mode indices.
        # In the topological phase there are 2 (±E degenerate pair at ~0).
        # A single eigenvector from a degenerate pair has arbitrary mixing
        # and yields all-zero u+v.  Combine both modes into symmetric (+)
        # and antisymmetric (-) Majorana operators and pick the one with
        # higher left-boundary weight (site 0).
        zero_inds = np.where(np.abs(evals) < thr)[0]
        if len(zero_inds) >= 2:
            idx1, idx2 = zero_inds[0], zero_inds[1]
            u1, v1 = evecs[:N, idx1], evecs[N:, idx1]
            u2, v2 = evecs[:N, idx2], evecs[N:, idx2]
            majA = ((u1 + u2) + (v1 + v2)) ** 2
            majB = ((u1 - u2) + (v1 - v2)) ** 2
            nA, nB = majA.sum(), majB.sum()
            if nA > 1e-14:
                majA /= nA
            if nB > 1e-14:
                majB /= nB
            # Pick the combination with higher weight on either boundary (site 0 or N-1)
            bndA = max(majA[0], majA[N - 1])
            bndB = max(majB[0], majB[N - 1])
            majorana = majA if bndA >= bndB else majB
        else:
            zero_idx = int(np.argmin(np.abs(evals)))
            evec = evecs[:, zero_idx]
            u, v = evec[:N], evec[N:]
            majorana = (u + v) ** 2
            norm = majorana.sum()
            if norm > 1e-14:
                majorana /= norm
        return {"gap": gap, "topo_inv": topo_inv, "majorana": majorana.tolist()}

    sol0 = solve_point(mu)
    H0 = build_bdg(mu)
    evals0, _ = np.linalg.eigh(H0)

    sweep = []
    for k in range(n_sweep):
        muk = mu_min + (mu_max - mu_min) * k / (n_sweep - 1)
        s = solve_point(muk)
        # Take the larger of the two boundary weights (left or right end)
        maj = s["majorana"]
        bdy_l = maj[0] + maj[1] if len(maj) >= 2 else 0.0
        bdy_r = maj[-1] + maj[-2] if len(maj) >= 2 else 0.0
        bdy = max(bdy_l, bdy_r)
        sweep.append({"mu": muk, "gap": s["gap"], "topo_inv": s["topo_inv"],
                      "majorana_weight": bdy})

    return {
        "n_sites": N, "t": t, "delta": delta, "mu": mu,
        "energies": evals0.tolist(),
        "gap": sol0["gap"],
        "topo_inv": sol0["topo_inv"],
        "majorana_weight": sol0["majorana"],
        "sweep": sweep,
    }


def _kitaev_page(data: dict, title: str = "Kitaev Chain") -> str:
    """4-panel Kitaev page: BdG spectrum, Majorana weight, gap vs μ, topo_inv vs μ."""
    import json as _json  # noqa: PLC0415

    jdata = _json.dumps(data)
    topo_label = "Topological (Q=-1)" if data.get("topo_inv", 1) < 0 else "Trivial (Q=+1)"
    topo_cls   = "topo" if data.get("topo_inv", 1) < 0 else "trivial"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body{{margin:0;background:#0d1117;color:#e6edf3;font-family:monospace;}}
  h1{{text-align:center;padding:16px;font-size:1.1rem;color:#388bfd;}}
  .subtitle{{text-align:center;font-size:0.8rem;color:#8b949e;margin-bottom:8px;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;}}
  canvas{{width:100%;height:220px;display:block;}}
  .label{{font-size:0.7rem;color:#8b949e;text-align:center;margin-top:4px;}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:bold;}}
  .topo{{background:#1c3a2c;color:#57ab5a;border:1px solid #57ab5a;}}
  .trivial{{background:#2c1c1c;color:#e3b341;border:1px solid #e3b341;}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">
  μ = {data.get('mu', 0):.2f} &nbsp;|&nbsp; t = {data.get('t', 1):.2f} &nbsp;|&nbsp;
  Δ = {data.get('delta', 1):.2f} &nbsp;|&nbsp;
  Phase: <span class="badge {topo_cls}">{topo_label}</span>
  &nbsp;(QPT at |μ| = 2t = {2*data.get('t',1):.2f})
</div>
<div class="grid">
  <div class="panel"><canvas id="cSpec"></canvas><div class="label">BdG quasi-energy spectrum</div></div>
  <div class="panel"><canvas id="cMaj"></canvas><div class="label">Majorana weight |u+v|² per site</div></div>
  <div class="panel"><canvas id="cGap"></canvas><div class="label">Quasiparticle gap vs μ/t</div></div>
  <div class="panel"><canvas id="cTopo"></canvas><div class="label">Topological invariant Q vs μ/t</div></div>
</div>
<script>
var DATA = {jdata};

function lineChart(id, xs, ys, color, ylabel, xlabel) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var xmin=Math.min.apply(null,xs), xmax=Math.max.apply(null,xs);
  var ymin=Math.min.apply(null,ys), ymax=Math.max.apply(null,ys);
  var xr=xmax-xmin||1, yr=ymax-ymin||1;
  ctx.strokeStyle="#21262d"; ctx.lineWidth=1;
  for(var gi=0;gi<5;gi++){{var gy=pad+(H-2*pad)*gi/4;ctx.beginPath();ctx.moveTo(pad,gy);ctx.lineTo(W-pad,gy);ctx.stroke();}}
  // QPT markers at μ=±2t
  var twoT = 2 * DATA.t;
  [[twoT,true],[-twoT,true]].forEach(function(m){{
    var px=pad+(m[0]-xmin)/xr*(W-2*pad);
    ctx.strokeStyle="#444"; ctx.lineWidth=1; ctx.setLineDash([4,3]);
    ctx.beginPath(); ctx.moveTo(px,pad); ctx.lineTo(px,H-pad); ctx.stroke();
    ctx.setLineDash([]);
  }});
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
  for(var i=0;i<xs.length;i++){{
    var px=pad+(xs[i]-xmin)/xr*(W-2*pad), py=H-pad-(ys[i]-ymin)/yr*(H-2*pad);
    if(i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace";
  ctx.fillText(ylabel,4,14); ctx.fillText(xlabel,W-pad,H-4);
  ctx.fillText(ymin.toFixed(3),2,H-pad); ctx.fillText(ymax.toFixed(3),2,pad+4);
}}

function barChart(id, vals, colorFn, ylabel) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var n = vals.length, ymax = Math.max.apply(null, vals.map(Math.abs)); if(ymax===0)ymax=1;
  var bw = (W-2*pad)/n;
  for(var i=0;i<n;i++){{
    var v=vals[i], bh=Math.abs(v)/ymax*(H-2*pad), bx=pad+i*bw+bw*0.1, by=H-pad-bh;
    ctx.fillStyle=colorFn(i,v); ctx.fillRect(bx,by,bw*0.8,bh);
  }}
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace"; ctx.fillText(ylabel,4,14);
  ctx.strokeStyle="#444"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
}}

// Panel 1: BdG spectrum (bar by eigenvalue index, colour by sign)
barChart("cSpec", DATA.energies, function(i,v){{return v>0?"#bc8cff":"#388bfd";}}, "E");

// Panel 2: Majorana weight
barChart("cMaj", DATA.majorana_weight,
  function(i,v){{return i<2||i>=DATA.majorana_weight.length-2?"#57ab5a":"#30363d";}},
  "|u+v|²");

// Panels 3-4: sweep
if(DATA.sweep && DATA.sweep.length>0){{
  var swM  = DATA.sweep.map(function(p){{return p.mu/DATA.t;}});
  var swGp = DATA.sweep.map(function(p){{return p.gap;}});
  var swQ  = DATA.sweep.map(function(p){{return p.topo_inv;}});
  lineChart("cGap",  swM, swGp, "#e3b341", "Gap",  "μ/t");
  lineChart("cTopo", swM, swQ,  "#57ab5a", "Q", "μ/t");
}}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# M30-J: Anderson Localization  —  Python reference solver + page
# ─────────────────────────────────────────────────────────────────────────────

def _anderson_frames(
    n_sites: int = 100,
    t: float = 1.0,
    W: float = 2.0,
    n_samples: int = 50,
    scenario: str = "sweep",
    n_sweep: int = 20,
    W_min: float = 0.1,
    W_max: float = 8.0,
    seed: int = 42,
) -> dict:
    """NumPy Anderson localization reference."""
    import numpy as np  # noqa: PLC0415

    N = n_sites
    rng = np.random.default_rng(seed)

    def build_h(W_: float, seed_: int) -> "np.ndarray":
        r = np.random.default_rng(seed_)
        eps = r.uniform(-W_ / 2.0, W_ / 2.0, N)
        H = np.diag(eps) - t * (np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1))
        return H

    def gap_ratio(evals: "np.ndarray") -> float:
        d = np.diff(evals)
        d = d[d > 1e-14]
        if len(d) < 2:
            return 0.0
        r = np.minimum(d[:-1], d[1:]) / np.maximum(d[:-1], d[1:])
        return float(r.mean())

    def solve_sample(W_: float, seed_: int) -> dict:
        H = build_h(W_, seed_)
        evals, evecs = np.linalg.eigh(H)
        ipr = (evecs ** 4).sum(axis=0)  # IPR per eigenstate
        mean_ipr = float(ipr.mean())
        mean_r = gap_ratio(evals)
        return {"evals": evals.tolist(), "ipr": ipr.tolist(),
                "mean_ipr": mean_ipr, "mean_r": mean_r}

    # Single sample at (W, seed)
    s0 = solve_sample(W, seed)
    loc_len = min(float(1.0 / s0["mean_ipr"]) if s0["mean_ipr"] > 1e-14 else N, N)

    sweep = []
    for k in range(n_sweep):
        Wk = W_min + (W_max - W_min) * k / (n_sweep - 1)
        sum_ipr, sum_r = 0.0, 0.0
        for samp in range(n_samples):
            s = solve_sample(Wk, int(rng.integers(0, 2**31)))
            sum_ipr += s["mean_ipr"]
            sum_r   += s["mean_r"]
        avg_ipr = sum_ipr / n_samples
        avg_r   = sum_r / n_samples
        xi = min(1.0 / avg_ipr if avg_ipr > 1e-14 else N, N)
        sweep.append({"W": Wk, "mean_ipr": avg_ipr, "mean_r": avg_r, "loc_length": xi})

    return {
        "n_sites": N, "t": t, "W": W,
        "energies": s0["evals"],
        "ipr_spectrum": s0["ipr"],
        "mean_ipr": s0["mean_ipr"],
        "mean_r": s0["mean_r"],
        "loc_length": loc_len,
        "sweep": sweep,
    }


def _anderson_page(data: dict, title: str = "Anderson Localization") -> str:
    """4-panel Anderson page: IPR spectrum, IPR vs W, ⟨r⟩ vs W, ξ vs W."""
    import json as _json  # noqa: PLC0415

    jdata = _json.dumps(data)
    r_val = data.get("mean_r", 0)
    r_label = f"⟨r⟩ = {r_val:.3f} ({'localised ~Poisson' if r_val < 0.45 else 'ergodic ~GOE'})"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body{{margin:0;background:#0d1117;color:#e6edf3;font-family:monospace;}}
  h1{{text-align:center;padding:16px;font-size:1.1rem;color:#388bfd;}}
  .subtitle{{text-align:center;font-size:0.8rem;color:#8b949e;margin-bottom:8px;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;}}
  canvas{{width:100%;height:220px;display:block;}}
  .label{{font-size:0.7rem;color:#8b949e;text-align:center;margin-top:4px;}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">
  N={data.get('n_sites',100)}, t={data.get('t',1):.2f}, W={data.get('W',2):.2f} &nbsp;|&nbsp;
  ξ ≈ {data.get('loc_length',0):.1f} &nbsp;|&nbsp; {r_label}
</div>
<div class="grid">
  <div class="panel"><canvas id="cIPRs"></canvas><div class="label">IPR per eigenstate (single sample)</div></div>
  <div class="panel"><canvas id="cIPRw"></canvas><div class="label">⟨IPR⟩ vs disorder W/t</div></div>
  <div class="panel"><canvas id="cR"></canvas><div class="label">Level-spacing ratio ⟨r⟩ vs W/t (Poisson≈0.39, GOE≈0.53)</div></div>
  <div class="panel"><canvas id="cXi"></canvas><div class="label">Localization length ξ vs W/t</div></div>
</div>
<script>
var DATA = {jdata};

function lineChart(id, xs, ys, color, ylabel, xlabel, refs) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var xmin=Math.min.apply(null,xs), xmax=Math.max.apply(null,xs);
  var allY = ys.slice(); if(refs) refs.forEach(function(r){{allY.push(r.v);}});
  var ymin=Math.min.apply(null,allY), ymax=Math.max.apply(null,allY);
  var xr=xmax-xmin||1, yr=ymax-ymin||1;
  ctx.strokeStyle="#21262d"; ctx.lineWidth=1;
  for(var gi=0;gi<5;gi++){{var gy=pad+(H-2*pad)*gi/4;ctx.beginPath();ctx.moveTo(pad,gy);ctx.lineTo(W-pad,gy);ctx.stroke();}}
  if(refs) refs.forEach(function(r){{
    var py=H-pad-(r.v-ymin)/yr*(H-2*pad);
    ctx.strokeStyle=r.c; ctx.lineWidth=1; ctx.setLineDash([5,3]);
    ctx.beginPath(); ctx.moveTo(pad,py); ctx.lineTo(W-pad,py); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=r.c; ctx.font="9px monospace"; ctx.fillText(r.label,W-pad-4,py-3);
  }});
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
  for(var i=0;i<xs.length;i++){{
    var px=pad+(xs[i]-xmin)/xr*(W-2*pad), py=H-pad-(ys[i]-ymin)/yr*(H-2*pad);
    if(i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace";
  ctx.fillText(ylabel,4,14); ctx.fillText(xlabel,W-pad,H-4);
}}

function barChart(id, vals, colorFn, ylabel) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var n = vals.length, ymax = Math.max.apply(null, vals); if(ymax===0)ymax=1;
  var bw = Math.max(1,(W-2*pad)/n);
  for(var i=0;i<n;i++){{
    var bh=vals[i]/ymax*(H-2*pad), bx=pad+i*bw, by=H-pad-bh;
    ctx.fillStyle=colorFn(i,vals[i]); ctx.fillRect(bx,by,Math.max(1,bw-1),bh);
  }}
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace"; ctx.fillText(ylabel,4,14);
}}

// Panel 1: IPR per state (scatter via bars)
barChart("cIPRs", DATA.ipr_spectrum,
  function(i,v){{return v > 1/DATA.n_sites * 5 ? "#e3b341" : "#388bfd";}}, "IPR");

if(DATA.sweep && DATA.sweep.length>0){{
  var swW  = DATA.sweep.map(function(p){{return p.W/DATA.t;}});
  var swI  = DATA.sweep.map(function(p){{return p.mean_ipr;}});
  var swR  = DATA.sweep.map(function(p){{return p.mean_r;}});
  var swXi = DATA.sweep.map(function(p){{return p.loc_length;}});
  lineChart("cIPRw", swW, swI, "#bc8cff", "⟨IPR⟩", "W/t", null);
  lineChart("cR",  swW, swR, "#57ab5a", "⟨r⟩", "W/t",
    [{{v:0.386,c:"#e3b341",label:"Poisson"}},{{v:0.530,c:"#388bfd",label:"GOE"}}]);
  lineChart("cXi", swW, swXi, "#388bfd", "ξ", "W/t", null);
}}
</script>
</body>
</html>"""


def main() -> int:
  """Parse CLI arguments, generate requested visualizations, and write them to disk."""
  parser = argparse.ArgumentParser(
      description="Generate QuantX visualization HTML.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=CLI_EXAMPLES,
  )
  parser.add_argument("--scenario", choices=["rabi", "mps", "mps_ising", "mps_xx", "spatial", "spatial2d", "spatial3d", "dirac2d", "driven_barrier", "bloch", "gaussian_pulse", "lindblad", "wigner_free", "wigner_barrier", "entanglement_rabi", "entanglement_decay", "vqe_ising", "vqe_qaoa", "qpe_phase", "qpe_hamiltonian", "qec_bit_flip", "qec_phase_flip", "qec_shor", "schwinger_vacuum", "schwinger_string", "schwinger_dynamics", "phi4_symmetric", "phi4_broken", "phi4_sweep", "phi4_dynamics", "hubbard", "floquet", "heisenberg", "bose_hubbard", "tfi", "ssh", "kitaev", "anderson", "all"], default="all")
  parser.add_argument(
      "--solver-mode",
      choices=["auto", "engine", "python"],
      default="auto",
      help="How continuous spatial pages should source their frame data.",
  )
  parser.add_argument(
      "--continuous-export-json",
      default="",
      help="Optional engine-export JSON file to render directly instead of solving again.",
  )
  parser.add_argument(
      "--strict-export-schema",
      action="store_true",
      help="Fail immediately when the continuous export schema_version is missing, below 1, or newer than supported.",
  )
  parser.add_argument("--spatial-potential", choices=["barrier", "free", "harmonic", "well"], default="barrier")
  parser.add_argument("--spatial2d-potential", choices=["barrier", "free", "harmonic", "well"], default="barrier")
  parser.add_argument("--spatial3d-potential", choices=["barrier", "free", "harmonic", "well"], default="barrier")
  parser.add_argument("--spatial3d-boundary", choices=["periodic", "absorbing", "dirichlet"], default="periodic")
  parser.add_argument("--spatial3d-export-json", action="store_true")
  parser.add_argument("--spatial3d-json-path", default="")
  parser.add_argument("--dirac2d-potential", choices=["barrier", "free", "harmonic", "well"], default="barrier")
  parser.add_argument("--dirac2d-mass", type=float, default=1.0)
  parser.add_argument("--lindblad-scenario", choices=["amplitude_damping", "dephasing", "depolarizing", "rabi_decay"], default="amplitude_damping")
  parser.add_argument("--lindblad-qubits", type=int, default=1)
  parser.add_argument("--samples-lindblad-t", type=int, default=80)
  parser.add_argument("--samples-wigner-t", type=int, default=50)
  parser.add_argument("--samples-entanglement-t", type=int, default=80)
  parser.add_argument("--samples-mps-t", type=int, default=60)
  parser.add_argument("--mps-sites", type=int, default=12)
  parser.add_argument("--mps-max-bond", type=int, default=16)
  parser.add_argument("--vqe-max-iter", type=int, default=120, help="max VQE/QAOA optimizer iterations")
  parser.add_argument("--vqe-layers", type=int, default=2, help="number of HEA layers / QAOA rounds")
  parser.add_argument("--vqe-qubits", type=int, default=4, help="number of qubits for VQE scenarios")
  parser.add_argument("--qpe-ancilla", type=int, default=6, help="number of ancilla (precision) qubits for QPE")
  parser.add_argument("--qpe-shots", type=int, default=4096, help="number of measurement shots for QPE")
  parser.add_argument("--qpe-phase", type=float, default=0.375, help="true phase φ ∈ [0,1) for qpe_phase scenario (phase_angle = 2π·φ)")
  parser.add_argument("--qpe-sweep-max", type=int, default=10, help="maximum n_ancilla in precision sweep (1..N)")
  parser.add_argument("--qec-error-rate", type=float, default=0.05, help="physical error rate per qubit for QEC scenarios")
  parser.add_argument("--qec-trials", type=int, default=2000, help="number of Monte Carlo trials for QEC scenarios")
  parser.add_argument("--qec-sweep", action="store_true", default=True, help="include threshold curve sweep in QEC output")
  parser.add_argument("--qec-no-sweep", dest="qec_sweep", action="store_false", help="disable QEC threshold sweep")
  parser.add_argument("--schwinger-sites", type=int, default=8, help="number of lattice sites for Schwinger model (even, ≤14)")
  parser.add_argument("--schwinger-mass", type=float, default=0.5, help="bare fermion mass m in units of J")
  parser.add_argument("--schwinger-coupling", type=float, default=0.3, help="gauge coupling g² in units of J")
  parser.add_argument("--schwinger-background", type=float, default=0.5, help="background electric field α for dynamics quench")
  parser.add_argument("--schwinger-steps", type=int, default=60, help="number of time steps for dynamics scenario")
  parser.add_argument("--schwinger-dt", type=float, default=0.08, help="time step dt for Schwinger dynamics")
  parser.add_argument("--schwinger-mass-sweep", type=int, default=30, help="number of mass values in chiral condensate sweep")
  parser.add_argument("--phi4-sites", type=int, default=4, help="number of lattice sites for phi4 (2-5)")
  parser.add_argument("--phi4-m2", type=float, default=1.0, help="bare mass squared m² for phi4 (negative → broken phase)")
  parser.add_argument("--phi4-lambda", type=float, default=1.0, help="quartic coupling λ in λ/4·φ⁴")
  parser.add_argument("--phi4-kappa", type=float, default=1.0, help="nearest-neighbour hopping κ")
  parser.add_argument("--phi4-d-local", type=int, default=10, help="local Fock space truncation (oscillator levels)")
  parser.add_argument("--phi4-sweep-points", type=int, default=25, help="number of m² points in phi4 phase transition sweep")
  parser.add_argument("--phi4-dynamics-m2-initial", type=float, default=1.0, help="pre-quench mass squared for phi4_dynamics")
  parser.add_argument("--phi4-dynamics-m2-final",   type=float, default=-1.5, help="post-quench mass squared for phi4_dynamics")
  parser.add_argument("--phi4-dynamics-steps",      type=int,   default=80,  help="number of time steps for phi4_dynamics")
  parser.add_argument("--phi4-dynamics-dt",         type=float, default=0.05, help="time step dt for phi4_dynamics")
  parser.add_argument("--hubbard-sites",   type=int,   default=6,    help="number of lattice sites for Hubbard (2-12)")
  parser.add_argument("--hubbard-t",       type=float, default=1.0,  help="hopping amplitude t")
  parser.add_argument("--hubbard-U",       type=float, default=4.0,  help="on-site Coulomb repulsion U")
  parser.add_argument("--hubbard-n-up",    type=int,   default=0,    help="spin-up electrons (0 = half-filling)")
  parser.add_argument("--hubbard-n-dn",    type=int,   default=0,    help="spin-down electrons (0 = half-filling)")
  parser.add_argument("--hubbard-sweep-points", type=int, default=20, help="number of U/t points in sweep")
  parser.add_argument("--hubbard-U-max",   type=float, default=12.0, help="maximum U for Hubbard sweep")
  parser.add_argument("--floquet-sites",   type=int,   default=8,    help="number of qubits for Floquet kicked-Ising")
  parser.add_argument("--floquet-J",       type=float, default=1.0,  help="Ising ZZ coupling J")
  parser.add_argument("--floquet-hz",      type=float, default=0.809, help="longitudinal field h_z")
  parser.add_argument("--floquet-amp",     type=float, default=1.5,  help="kick amplitude h_x")
  parser.add_argument("--floquet-omega",   type=float, default=4.5,  help="drive frequency ω")
  parser.add_argument("--floquet-periods", type=int,   default=40,   help="number of Floquet periods to simulate")
  parser.add_argument("--floquet-scenario", choices=["dynamics", "spectrum"], default="dynamics")
  parser.add_argument("--heisenberg-sites",   type=int,   default=8,    help="number of sites for Heisenberg chain (2-12)")
  parser.add_argument("--heisenberg-J",       type=float, default=1.0,  help="transverse exchange coupling J")
  parser.add_argument("--heisenberg-Jz",      type=float, default=1.0,  help="longitudinal exchange Jz")
  parser.add_argument("--heisenberg-h",       type=float, default=0.0,  help="uniform longitudinal field h")
  parser.add_argument("--heisenberg-sweep-points", type=int, default=20, help="number of Jz points in sweep")
  parser.add_argument("--heisenberg-Jz-min",  type=float, default=-2.0, help="minimum Jz for sweep")
  parser.add_argument("--heisenberg-Jz-max",  type=float, default=3.0,  help="maximum Jz for sweep")
  parser.add_argument("--bose-hubbard-sites",   type=int,   default=4,   help="number of sites for Bose-Hubbard (2-8)")
  parser.add_argument("--bose-hubbard-bosons",  type=int,   default=4,   help="total boson number")
  parser.add_argument("--bose-hubbard-nmax",    type=int,   default=4,   help="max occupation per site")
  parser.add_argument("--bose-hubbard-J",       type=float, default=0.1, help="hopping amplitude J")
  parser.add_argument("--bose-hubbard-U",       type=float, default=1.0, help="on-site interaction U")
  parser.add_argument("--bose-hubbard-mu",      type=float, default=0.0, help="chemical potential mu")
  parser.add_argument("--bose-hubbard-sweep-points", type=int, default=20, help="number of J points in sweep")
  parser.add_argument("--bose-hubbard-J-max",   type=float, default=0.5, help="maximum J for Bose-Hubbard sweep")
  parser.add_argument("--tfi-sites",        type=int,   default=10,   help="number of sites for TFI (2-18)")
  parser.add_argument("--tfi-J",            type=float, default=1.0,  help="ZZ coupling J for TFI")
  parser.add_argument("--tfi-h",            type=float, default=0.5,  help="transverse field h (QPT at h=J)")
  parser.add_argument("--tfi-sweep-points", type=int,   default=30,   help="number of h values in TFI sweep")
  parser.add_argument("--tfi-h-min",        type=float, default=0.0,  help="minimum h for TFI sweep")
  parser.add_argument("--tfi-h-max",        type=float, default=2.0,  help="maximum h for TFI sweep")
  parser.add_argument("--ssh-cells",        type=int,   default=10,   help="number of unit cells for SSH (2-200)")
  parser.add_argument("--ssh-t",            type=float, default=1.0,  help="hopping scale t for SSH")
  parser.add_argument("--ssh-delta",        type=float, default=0.5,  help="dimerization δ (trivial if δ<0, topo if δ>0)")
  parser.add_argument("--ssh-sweep-points", type=int,   default=30,   help="number of δ values in SSH sweep")
  parser.add_argument("--ssh-delta-min",    type=float, default=-0.9, help="minimum δ for SSH sweep")
  parser.add_argument("--ssh-delta-max",    type=float, default=0.9,  help="maximum δ for SSH sweep")
  parser.add_argument("--kitaev-sites",    type=int,   default=20,   help="number of sites for Kitaev chain (2-200)")
  parser.add_argument("--kitaev-t",        type=float, default=1.0,  help="hopping t for Kitaev chain")
  parser.add_argument("--kitaev-delta",    type=float, default=1.0,  help="pairing amplitude Δ")
  parser.add_argument("--kitaev-mu",       type=float, default=0.0,  help="chemical potential μ (topo: |μ|<2t)")
  parser.add_argument("--kitaev-sweep-points", type=int, default=40, help="number of μ values in Kitaev sweep")
  parser.add_argument("--kitaev-mu-min",   type=float, default=-4.0, help="minimum μ for Kitaev sweep")
  parser.add_argument("--kitaev-mu-max",   type=float, default=4.0,  help="maximum μ for Kitaev sweep")
  parser.add_argument("--anderson-sites",   type=int,   default=100, help="number of sites for Anderson chain (4-2000)")
  parser.add_argument("--anderson-t",       type=float, default=1.0, help="hopping t for Anderson chain")
  parser.add_argument("--anderson-W",       type=float, default=2.0, help="disorder strength W")
  parser.add_argument("--anderson-samples", type=int,   default=50,  help="disorder realisations to average")
  parser.add_argument("--anderson-sweep-points", type=int, default=20, help="number of W values in Anderson sweep")
  parser.add_argument("--anderson-W-min",   type=float, default=0.1, help="minimum W for Anderson sweep")
  parser.add_argument("--anderson-W-max",   type=float, default=8.0, help="maximum W for Anderson sweep")
  parser.add_argument("--out-dir", default="out/visualizations")
  parser.add_argument(
      "--dashboard", action="store_true", default=True,
      help="(--scenario all) Generate quantum_dashboard.html aggregating all scenarios.",
  )
  parser.add_argument(
      "--no-dashboard", dest="dashboard", action="store_false",
      help="Skip dashboard generation even when --scenario all is used.",
  )
  parser.add_argument("--samples-rabi", type=int, default=80)
  parser.add_argument("--samples-mps", type=int, default=60)
  parser.add_argument("--samples-spatial-t", type=int, default=120)
  parser.add_argument("--samples-spatial-x", type=int, default=220)
  parser.add_argument("--samples-spatial2d-t", type=int, default=90)
  parser.add_argument("--samples-spatial2d-x", type=int, default=72)
  parser.add_argument("--samples-spatial2d-y", type=int, default=72)
  parser.add_argument("--samples-spatial3d-t", type=int, default=48)
  parser.add_argument("--samples-spatial3d-x", type=int, default=32)
  parser.add_argument("--samples-spatial3d-y", type=int, default=32)
  parser.add_argument("--samples-spatial3d-z", type=int, default=32)
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[1]
  out_dir = root / args.out_dir
  out_dir.mkdir(parents=True, exist_ok=True)
  engine = _make_engine(root) if args.scenario in {"rabi", "mps", "spatial", "spatial2d", "spatial3d", "all"} or args.solver_mode not in {"python", None} else None  # lindblad is python-only
  generated = []
  export_json_path = root / args.continuous_export_json if args.continuous_export_json else None

  if args.scenario in {"rabi", "all"}:
    if engine is None:
      raise RuntimeError("rabi visualization requires the native engine library")
    sim, ref = _rabi_data(engine, args.samples_rabi)
    charts = [
        {
            "title": "Rabi Oscillation: Population(|1>) vs Time",
            "x_label": "time",
            "y_label": "population_1",
            "series": [
                {"name": "simulation", "color": "#0b84f3", "points": sim},
                {"name": "analytic sin^2(t)", "color": "#f39c12", "points": ref},
            ],
        }
    ]
    path = out_dir / "rabi_population.html"
    path.write_text(_animated_page("QuantX Animation - Rabi", charts), encoding="utf-8")
    generated.append(path)

  if args.scenario in {"mps", "all"}:
    if engine is None:
      raise RuntimeError("mps visualization requires the native engine library")
    bond, trunc = _mps_data(engine, args.samples_mps)
    charts = [
        {
            "title": "MPS Bond Dimension vs Time",
            "x_label": "time",
            "y_label": "bond_dim",
            "series": [{"name": "bond_dim", "color": "#2e8b57", "points": bond}],
        },
        {
            "title": "MPS Truncation Error vs Time",
            "x_label": "time",
            "y_label": "truncation_error",
            "series": [{"name": "truncation_error", "color": "#c0392b", "points": trunc}],
        },
    ]
    path = out_dir / "mps_diagnostics.html"
    path.write_text(_animated_page("QuantX Animation - MPS", charts), encoding="utf-8")
    generated.append(path)

  if args.scenario in {"spatial", "all"}:
    spatial_data = _spatial_wavepacket_frames(
        args.samples_spatial_t, args.samples_spatial_x, args.spatial_potential, source="python-reference"
    )
    path = out_dir / f"quantum_spatial_wavepacket_{args.spatial_potential}.html"
    path.write_text(_spatial_wavepacket_page(spatial_data), encoding="utf-8")
    generated.append(path)

  if args.scenario in {"spatial2d", "all"}:
    spatial2d_problem = _continuous_problem_2d_request(
        args.samples_spatial2d_t, args.samples_spatial2d_x, args.samples_spatial2d_y, args.spatial2d_potential
    )
    spatial2d = _resolve_continuous_payload(
        engine,
        spatial2d_problem,
        args.solver_mode,
        lambda: _spatial_wavepacket_2d_frames(
            args.samples_spatial2d_t, args.samples_spatial2d_x, args.samples_spatial2d_y, args.spatial2d_potential
        ),
        export_json_path,
        args.strict_export_schema,
    )
    path = out_dir / f"quantum_spatial2d_wavepacket_{args.spatial2d_potential}.html"
    path.write_text(_spatial_wavepacket_2d_page(spatial2d), encoding="utf-8")
    generated.append(path)

  if args.scenario in {"spatial3d", "all"}:
    spatial3d_problem = _continuous_problem_3d_request(
        args.samples_spatial3d_t,
        args.samples_spatial3d_x,
        args.samples_spatial3d_y,
        args.samples_spatial3d_z,
        args.spatial3d_potential,
        args.spatial3d_boundary,
    )
    spatial3d = _resolve_continuous_payload(
        engine,
        spatial3d_problem,
        args.solver_mode,
        lambda: _spatial_wavepacket_3d_frames(
            args.samples_spatial3d_t,
            args.samples_spatial3d_x,
            args.samples_spatial3d_y,
            args.samples_spatial3d_z,
            args.spatial3d_potential,
            args.spatial3d_boundary,
        ),
        export_json_path,
        args.strict_export_schema,
    )
    path = out_dir / f"quantum_spatial3d_wavepacket_{args.spatial3d_potential}_{args.spatial3d_boundary}.html"
    path.write_text(_spatial_wavepacket_3d_page(spatial3d), encoding="utf-8")
    generated.append(path)
    if args.spatial3d_export_json:
      if args.spatial3d_json_path:
        json_path = root / args.spatial3d_json_path
      else:
        json_path = out_dir / f"quantum_spatial3d_wavepacket_{args.spatial3d_potential}_{args.spatial3d_boundary}.json"
      json_path.parent.mkdir(parents=True, exist_ok=True)
      json_path.write_text(json.dumps(spatial3d, ensure_ascii=False), encoding="utf-8")
      generated.append(json_path)

  if args.scenario in {"dirac2d", "all"}:
    dirac2d_data = _spatial_wavepacket_2d_dirac_frames(
        args.samples_spatial2d_t,
        args.samples_spatial2d_x,
        args.samples_spatial2d_y,
        args.dirac2d_potential,
        args.dirac2d_mass,
    )
    path = out_dir / f"quantum_dirac2d_{args.dirac2d_potential}.html"
    path.write_text(_dirac_spinor_page(dirac2d_data), encoding="utf-8")
    generated.append(path)

  _DRIVEN_SCENARIOS: dict[str, tuple[str, str, dict]] = {
      # scenario_key -> (potential_type, output_filename_stem, driving_config)
      "driven_barrier": (
          "barrier",
          "quantum_driven_barrier",
          {"type": "sine", "amplitude": 0.4, "frequency": 1.8, "spatial_profile": "uniform"},
      ),
      "bloch": (
          "free",
          "quantum_bloch_oscillation",
          {"type": "sine", "amplitude": 0.35, "frequency": 2.5, "spatial_profile": "dipole_x"},
      ),
      "gaussian_pulse": (
          "harmonic",
          "quantum_gaussian_pulse",
          {
              "type": "gaussian_pulse",
              "amplitude": 0.6,
              "frequency": 0.55,
              "t_center": 1.2,
              "t_width": 0.4,
              "spatial_profile": "dipole_x",
          },
      ),
  }
  for scenario_key, (pot_type, stem, drv_cfg) in _DRIVEN_SCENARIOS.items():
    if args.scenario not in {scenario_key, "all"}:
      continue
    driven_problem = _continuous_problem_2d_driven_request(
        args.samples_spatial2d_t, args.samples_spatial2d_x, args.samples_spatial2d_y,
        pot_type, drv_cfg,
    )
    driven_data = _resolve_continuous_payload(
        engine,
        driven_problem,
        args.solver_mode,
        lambda pt=pot_type, dc=drv_cfg: _spatial_wavepacket_2d_frames(
            args.samples_spatial2d_t, args.samples_spatial2d_x, args.samples_spatial2d_y,
            pt, driving=dc,
        ),
        export_json_path,
        args.strict_export_schema,
    )
    # Ensure the driving config is preserved in the payload for the page template.
    if "driving" not in driven_data:
      driven_data["driving"] = drv_cfg
    if "has_driving" not in driven_data:
      driven_data["has_driving"] = True
    path = out_dir / f"{stem}.html"
    path.write_text(_driven_wavepacket_2d_page(driven_data), encoding="utf-8")
    generated.append(path)

  if args.scenario in {"lindblad", "all"}:
    if args.lindblad_scenario == "amplitude_damping":
      h_terms = [{"label": "X0", "coefficient": 0.5}]
      c_ops = [{"label": "AmplitudeDamping0", "coefficient": 0.8}]
    elif args.lindblad_scenario == "dephasing":
      h_terms = [{"label": "X0", "coefficient": 1.0}]
      c_ops = [{"label": "Dephasing0", "coefficient": 1.0}]
    elif args.lindblad_scenario == "depolarizing":
      h_terms = [{"label": "Z0", "coefficient": 0.5}]
      c_ops = [{"label": "Depolarizing0", "coefficient": 0.5}]
    else:  # rabi_decay
      h_terms = [{"label": "X0", "coefficient": 1.0}, {"label": "Z0", "coefficient": -0.5}]
      c_ops = [{"label": "AmplitudeDamping0", "coefficient": 0.3}]
    lindblad_data = _lindblad_frames(
        args.lindblad_qubits, h_terms, c_ops,
        total_time=2.0, dt=0.02,
        samples=args.samples_lindblad_t,
    )
    path = out_dir / f"quantum_lindblad_{args.lindblad_scenario}.html"
    path.write_text(_lindblad_page(lindblad_data), encoding="utf-8")
    generated.append(path)

  _WIGNER_SCENARIOS: dict[str, str] = {
      "wigner_free":    "free",
      "wigner_barrier": "barrier",
  }
  for scenario_key, pot_type in _WIGNER_SCENARIOS.items():
    if args.scenario not in {scenario_key, "all"}:
      continue
    nx_w = args.samples_spatial2d_x
    ny_w = args.samples_spatial2d_y
    wigner_data = _wigner_frames(
        nx=nx_w,
        ny=ny_w,
        total_time=6.0,
        dt=0.05,
        potential_type=pot_type,
        samples=args.samples_wigner_t,
    )
    path = out_dir / f"quantum_wigner_{scenario_key}.html"
    path.write_text(_wigner_page(wigner_data), encoding="utf-8")
    generated.append(path)

  # M19 — MPS/TEBD scenarios
  # mps_ising: transverse-field Ising model H = -Σ Z_i Z_{i+1} - g Σ X_i
  if args.scenario in {"mps_ising", "all"}:
    n = args.mps_sites
    ising_terms = (
        [{"label": f"Z{i}Z{i+1}", "coefficient": -1.0} for i in range(n - 1)]
        + [{"label": f"X{i}", "coefficient": -0.5} for i in range(n)]
    )
    mps_ising_data = _mps_frames(
        n_sites=n,
        hamiltonian_terms=ising_terms,
        total_time=4.0, dt=0.05,
        max_bond=args.mps_max_bond,
        samples=args.samples_mps_t,
    )
    path = out_dir / "quantum_mps_ising.html"
    path.write_text(_mps_page(mps_ising_data), encoding="utf-8")
    generated.append(path)

  # mps_xx: XX model H = Σ X_i X_{i+1} — immediately entangles |00…0⟩ (XX|00⟩=|11⟩≠0)
  if args.scenario in {"mps_xx", "all"}:
    n = args.mps_sites
    xx_terms = [{"label": f"X{i}X{i+1}", "coefficient": 1.0} for i in range(n - 1)]
    mps_xx_data = _mps_frames(
        n_sites=n,
        hamiltonian_terms=xx_terms,
        total_time=3.0, dt=0.04,
        max_bond=args.mps_max_bond,
        samples=args.samples_mps_t,
    )
    path = out_dir / "quantum_mps_xx.html"
    path.write_text(_mps_page(mps_xx_data), encoding="utf-8")
    generated.append(path)

  # M18 — Entanglement scenarios
  # entanglement_rabi: transverse-field Ising model — X0+X1 drives, Z0Z1 creates entanglement
  if args.scenario in {"entanglement_rabi", "all"}:
    h_terms_rabi = [
        {"label": "X0", "coefficient": 0.5},
        {"label": "X1", "coefficient": 0.5},
        {"label": "Z0Z1", "coefficient": 1.0},
    ]
    ent_rabi_data = _entanglement_frames(
        n_qubits=2,
        hamiltonian_terms=h_terms_rabi,
        collapse_ops=[],
        n_a=1,
        total_time=math.pi * 2,
        dt=0.02,
        samples=args.samples_entanglement_t,
    )
    path = out_dir / "quantum_entanglement_rabi.html"
    path.write_text(_entanglement_page(ent_rabi_data), encoding="utf-8")
    generated.append(path)

  # entanglement_decay: 2-qubit Heisenberg XX + amplitude damping on qubit 1
  if args.scenario in {"entanglement_decay", "all"}:
    h_terms_decay = [
        {"label": "X0X1", "coefficient": 1.0},
        {"label": "Z0",   "coefficient": 0.2},
        {"label": "Z1",   "coefficient": 0.2},
    ]
    c_ops_decay = [{"label": "AmplitudeDamping1", "coefficient": 0.5}]
    ent_decay_data = _entanglement_frames(
        n_qubits=2,
        hamiltonian_terms=h_terms_decay,
        collapse_ops=c_ops_decay,
        n_a=1,
        total_time=6.0,
        dt=0.02,
        samples=args.samples_entanglement_t,
    )
    path = out_dir / "quantum_entanglement_decay.html"
    path.write_text(_entanglement_page(ent_decay_data), encoding="utf-8")
    generated.append(path)

  # ── M20: VQE / QAOA variational algorithms ───────────────────────────────────

  # vqe_ising: VQE on the transverse-field Ising model
  # H = -Σ Z_iZ_{i+1} - 0.5 Σ X_i  (4 qubits, HEA ansatz)
  if args.scenario in {"vqe_ising", "all"}:
    n = args.vqe_qubits
    ising_ham = (
        [{"label": f"Z{i}Z{i+1}", "coefficient": -1.0} for i in range(n - 1)]
        + [{"label": f"X{i}", "coefficient": -0.5} for i in range(n)]
    )
    vqe_ising_data = _vqe_frames(
        n_qubits=n,
        cost_hamiltonian=ising_ham,
        ansatz_type="hardware_efficient",
        n_layers=args.vqe_layers,
        optimizer="adam",
        max_iterations=args.vqe_max_iter,
        learning_rate=0.12,
    )
    path = out_dir / "quantum_vqe_ising.html"
    path.write_text(_vqe_page(vqe_ising_data), encoding="utf-8")
    generated.append(path)

  # vqe_qaoa: QAOA MaxCut on a 4-node cycle graph
  if args.scenario in {"vqe_qaoa", "all"}:
    n = args.vqe_qubits
    # 4-node cycle: 0-1-2-3-0
    edges = [(i, (i + 1) % n) for i in range(n)]
    # Cost Hamiltonian for MaxCut: minimize Σ Z_iZ_j (equivalent to maximising cut)
    qaoa_ham = [{"label": f"Z{i}Z{j}", "coefficient": 1.0} for (i, j) in edges]
    vqe_qaoa_data = _vqe_frames(
        n_qubits=n,
        cost_hamiltonian=qaoa_ham,
        ansatz_type="qaoa",
        n_layers=args.vqe_layers,
        optimizer="adam",
        max_iterations=args.vqe_max_iter,
        learning_rate=0.15,
    )
    # Attach edges so _vqe_page can rebuild the final state.
    vqe_qaoa_data["qaoa_edges"] = edges
    path = out_dir / "quantum_vqe_qaoa.html"
    path.write_text(_vqe_page(vqe_qaoa_data), encoding="utf-8")
    generated.append(path)

  # ── M21 — QPE scenarios ──────────────────────────────────────────────────────
  # qpe_phase: single-qubit phase gate P(2π·φ) with exact phase φ = 3/8 = 0.375
  if args.scenario in {"qpe_phase", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("qpe_phase skipped: numpy not available", file=sys.stderr)
    else:
      phi = args.qpe_phase  # e.g. 0.375 → 3/8 → exact representation with n_ancilla >= 3
      phase_angle = 2.0 * 3.141592653589793 * phi
      # Single-qubit phase gate U = diag(1, e^{i·phase_angle})
      U_phase = np.array([[1.0, 0.0], [0.0, np.exp(1j * phase_angle)]], dtype=complex)
      # Eigenstate |1⟩ of P(θ) with eigenvalue e^{iθ}
      eigenstate_phase = np.array([0.0, 1.0], dtype=complex)
      sweep_ns = list(range(1, min(args.qpe_sweep_max, 13) + 1))
      qpe_phase_data = _qpe_frames(
          n_ancilla=args.qpe_ancilla,
          unitary=U_phase,
          eigenstate=eigenstate_phase,
          n_shots=args.qpe_shots,
          true_phase=phi,
          sweep_ancilla=sweep_ns,
          seed=42,
      )
      path = out_dir / "quantum_qpe_phase.html"
      path.write_text(_qpe_page(qpe_phase_data, title="QPE — Phase Gate"), encoding="utf-8")
      generated.append(path)

  # qpe_hamiltonian: Heisenberg ZZ Hamiltonian H = Z₀Z₁ → U = exp(-iHt), t = π/8
  # Eigenvalues of Z₀Z₁: ±1 → U eigenvalues e^{∓iπ/8} → phases ∓1/16 mod 1
  if args.scenario in {"qpe_hamiltonian", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("qpe_hamiltonian skipped: numpy not available", file=sys.stderr)
    else:
      t_evo = 3.141592653589793 / 8.0  # π/8
      # H = Z₀ ⊗ Z₁ = diag(1, -1, -1, 1) in computational basis.
      H_zz = np.diag([1.0, -1.0, -1.0, 1.0])
      # U = exp(-i H t) = diag(e^{-it}, e^{it}, e^{it}, e^{-it})
      U_ham = np.diag(np.exp(-1j * t_evo * np.diag(H_zz)))
      # Eigenstate: |01⟩ (index 1) — eigenvalue of H_zz is -1 → phase = t/(2π)
      eigenstate_ham = np.zeros(4, dtype=complex)
      eigenstate_ham[1] = 1.0
      true_phase_ham = t_evo / (2.0 * 3.141592653589793)  # ≈ 0.0625 = 1/16
      sweep_ns = list(range(1, min(args.qpe_sweep_max, 13) + 1))
      qpe_ham_data = _qpe_frames(
          n_ancilla=args.qpe_ancilla,
          unitary=U_ham,
          eigenstate=eigenstate_ham,
          n_shots=args.qpe_shots,
          true_phase=true_phase_ham,
          sweep_ancilla=sweep_ns,
          seed=42,
      )
      path = out_dir / "quantum_qpe_hamiltonian.html"
      path.write_text(_qpe_page(qpe_ham_data, title="QPE — ZZ Hamiltonian"), encoding="utf-8")
      generated.append(path)

  # ── M22 — QEC scenarios ──────────────────────────────────────────────────────
  # Build the threshold sweep points (log-spaced from 0.01 to 0.45)
  _qec_sweep_pts: list[float] | None = None
  if args.qec_sweep:
    _qec_sweep_pts = [round(0.01 + 0.44 * i / 19, 4) for i in range(20)]

  # qec_bit_flip: 3-qubit bit-flip repetition code with X errors
  if args.scenario in {"qec_bit_flip", "all"}:
    qec_bf_data = _qec_frames(
        code="bit_flip_3",
        physical_error_rate=args.qec_error_rate,
        error_type="X",
        n_trials=args.qec_trials,
        n_detail_trials=64,
        sweep_error_rates=_qec_sweep_pts,
        seed=42,
    )
    path = out_dir / "quantum_qec_bit_flip.html"
    path.write_text(_qec_page(qec_bf_data, title="QEC — 3-Qubit Bit-Flip Code"), encoding="utf-8")
    generated.append(path)

  # qec_phase_flip: 3-qubit phase-flip code with Z errors
  if args.scenario in {"qec_phase_flip", "all"}:
    qec_pf_data = _qec_frames(
        code="phase_flip_3",
        physical_error_rate=args.qec_error_rate,
        error_type="Z",
        n_trials=args.qec_trials,
        n_detail_trials=64,
        sweep_error_rates=_qec_sweep_pts,
        seed=42,
    )
    path = out_dir / "quantum_qec_phase_flip.html"
    path.write_text(_qec_page(qec_pf_data, title="QEC — 3-Qubit Phase-Flip Code"), encoding="utf-8")
    generated.append(path)

  # qec_shor: 9-qubit Shor code with depolarizing noise
  if args.scenario in {"qec_shor", "all"}:
    qec_shor_data = _qec_frames(
        code="shor_9",
        physical_error_rate=args.qec_error_rate,
        error_type="depolarizing",
        n_trials=args.qec_trials,
        n_detail_trials=64,
        sweep_error_rates=_qec_sweep_pts,
        seed=42,
    )
    path = out_dir / "quantum_qec_shor.html"
    path.write_text(_qec_page(qec_shor_data, title="QEC — 9-Qubit Shor Code"), encoding="utf-8")
    generated.append(path)

  # ── M23 — Schwinger model scenarios ─────────────────────────────────────────
  # schwinger_vacuum: ground state observables + chiral condensate mass sweep
  if args.scenario in {"schwinger_vacuum", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("schwinger_vacuum skipped: numpy not available", file=sys.stderr)
    else:
      vac_data = _schwinger_frames(
          n_sites=args.schwinger_sites,
          mass=args.schwinger_mass,
          coupling=args.schwinger_coupling,
          background=0.0,
          scenario="vacuum",
          n_mass_sweep=args.schwinger_mass_sweep,
      )
      path = out_dir / "quantum_schwinger_vacuum.html"
      path.write_text(_schwinger_page(vac_data, title="Schwinger Model — Vacuum"), encoding="utf-8")
      generated.append(path)

  # schwinger_string: string tension E(R) vs quark–antiquark separation
  if args.scenario in {"schwinger_string", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("schwinger_string skipped: numpy not available", file=sys.stderr)
    else:
      str_data = _schwinger_frames(
          n_sites=args.schwinger_sites,
          mass=args.schwinger_mass,
          coupling=args.schwinger_coupling,
          background=0.0,
          scenario="string",
          n_mass_sweep=args.schwinger_mass_sweep,
      )
      path = out_dir / "quantum_schwinger_string.html"
      path.write_text(_schwinger_page(str_data, title="Schwinger Model — String Tension"), encoding="utf-8")
      generated.append(path)

  # schwinger_dynamics: pair production under background field quench
  if args.scenario in {"schwinger_dynamics", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
      # Try scipy for exact expm; fall back to Trotter.
      try:
        import scipy.linalg as _scipy_linalg  # noqa: PLC0415
        _has_scipy = True
      except ImportError:
        _has_scipy = False
    except ImportError:
      print("schwinger_dynamics skipped: numpy not available", file=sys.stderr)
    else:
      # Use smaller N for dynamics (exact matrix exp is O(dim³)).
      n_dyn = min(args.schwinger_sites, 6)
      if not _has_scipy:
        # Fallback: eigendecompose once → fast time evolution.
        dyn_frames = _schwinger_fallback_dynamics(
            n_sites=n_dyn,
            mass=args.schwinger_mass,
            coupling=args.schwinger_coupling,
            background_quench=args.schwinger_background,
            dt=args.schwinger_dt,
            n_steps=args.schwinger_steps,
        )
        dyn_data = _schwinger_frames(
            n_sites=n_dyn,
            mass=args.schwinger_mass,
            coupling=args.schwinger_coupling,
            background=0.0,
            scenario="vacuum",
            n_mass_sweep=args.schwinger_mass_sweep,
        )
        dyn_data["scenario"] = "dynamics"
        dyn_data["frames"] = dyn_frames
        dyn_data["background"] = args.schwinger_background
      else:
        dyn_data = _schwinger_frames(
            n_sites=n_dyn,
            mass=args.schwinger_mass,
            coupling=args.schwinger_coupling,
            background=args.schwinger_background,
            scenario="dynamics",
            dt=args.schwinger_dt,
            n_steps=args.schwinger_steps,
            n_mass_sweep=args.schwinger_mass_sweep,
        )
      path = out_dir / "quantum_schwinger_dynamics.html"
      path.write_text(_schwinger_page(dyn_data, title="Schwinger Model — Pair Production Dynamics"), encoding="utf-8")
      generated.append(path)

  # ── M24 — φ⁴ Scalar QFT scenarios ───────────────────────────────────────────
  _phi4_common = dict(
      n_sites=args.phi4_sites,
      lam=args.phi4_lambda,
      kappa=args.phi4_kappa,
      d_local=args.phi4_d_local,
      n_sweep=args.phi4_sweep_points,
  )

  # phi4_symmetric: m² > 0, single-well potential, Z₂ symmetric vacuum
  if args.scenario in {"phi4_symmetric", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("phi4_symmetric skipped: numpy not available", file=sys.stderr)
    else:
      sym_data = _phi4_frames(m2=max(args.phi4_m2, 0.5), scenario="symmetric", **_phi4_common)
      path = out_dir / "quantum_phi4_symmetric.html"
      path.write_text(_phi4_page(sym_data, title="φ⁴ QFT — Symmetric Phase"), encoding="utf-8")
      generated.append(path)

  # phi4_broken: m² < 0, Mexican-hat potential, Z₂ spontaneously broken
  if args.scenario in {"phi4_broken", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("phi4_broken skipped: numpy not available", file=sys.stderr)
    else:
      brk_data = _phi4_frames(m2=min(args.phi4_m2, -1.5), scenario="broken", **_phi4_common)
      path = out_dir / "quantum_phi4_broken.html"
      path.write_text(_phi4_page(brk_data, title="φ⁴ QFT — Broken Phase (SSB)"), encoding="utf-8")
      generated.append(path)

  # phi4_sweep: scan m² across the quantum phase transition, track order parameter + gap
  if args.scenario in {"phi4_sweep", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("phi4_sweep skipped: numpy not available", file=sys.stderr)
    else:
      swp_data = _phi4_frames(m2=args.phi4_m2, scenario="sweep", **_phi4_common)
      path = out_dir / "quantum_phi4_sweep.html"
      path.write_text(_phi4_page(swp_data, title="φ⁴ QFT — Phase Transition Sweep"), encoding="utf-8")
      generated.append(path)

  if args.scenario in {"phi4_dynamics", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("phi4_dynamics skipped: numpy not available", file=sys.stderr)
    else:
      dyn_data = _phi4_dynamics_frames(
          n_sites=args.phi4_sites,
          d_local=args.phi4_d_local,
          lam=args.phi4_lambda,
          kappa=args.phi4_kappa,
          m2_initial=args.phi4_dynamics_m2_initial,
          m2_final=args.phi4_dynamics_m2_final,
          n_steps=args.phi4_dynamics_steps,
          dt=args.phi4_dynamics_dt,
      )
      path = out_dir / "quantum_phi4_dynamics.html"
      path.write_text(_phi4_dynamics_page(dyn_data, title="φ⁴ QFT — Quantum Quench Dynamics"), encoding="utf-8")
      generated.append(path)

  # ── M27-C: Hubbard Model ──────────────────────────────────────────────────
  if args.scenario in {"hubbard", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("hubbard skipped: numpy not available", file=sys.stderr)
    else:
      hub_data = _hubbard_frames(
          n_sites=args.hubbard_sites,
          t_hop=args.hubbard_t,
          U=args.hubbard_U,
          n_up=args.hubbard_n_up,
          n_dn=args.hubbard_n_dn,
          scenario="sweep",
          n_sweep=args.hubbard_sweep_points,
          U_max=args.hubbard_U_max,
      )
      path = out_dir / "quantum_hubbard.html"
      path.write_text(_hubbard_page(hub_data, title="1D Hubbard Model — Mott Insulator"), encoding="utf-8")
      generated.append(path)

  # ── M27-D: Floquet Kicked-Ising ──────────────────────────────────────────
  if args.scenario in {"floquet", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("floquet skipped: numpy not available", file=sys.stderr)
    else:
      flo_data = _floquet_frames(
          n_sites=args.floquet_sites,
          J=args.floquet_J,
          h_z=args.floquet_hz,
          drive_amp=args.floquet_amp,
          drive_omega=args.floquet_omega,
          n_periods=args.floquet_periods,
          scenario=args.floquet_scenario,
      )
      path = out_dir / "quantum_floquet.html"
      path.write_text(_floquet_page(flo_data, title="Floquet 周期驱动 — Kicked-Ising"), encoding="utf-8")
      generated.append(path)

  # ── M28-E: XXZ Heisenberg chain ──────────────────────────────────────────
  if args.scenario in {"heisenberg", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("heisenberg skipped: numpy not available", file=sys.stderr)
    else:
      heis_data = _heisenberg_frames(
          n_sites=args.heisenberg_sites,
          J=args.heisenberg_J,
          Jz=args.heisenberg_Jz,
          h=args.heisenberg_h,
          scenario="sweep",
          n_sweep=args.heisenberg_sweep_points,
          Jz_min=args.heisenberg_Jz_min,
          Jz_max=args.heisenberg_Jz_max,
      )
      path = out_dir / "quantum_heisenberg.html"
      path.write_text(_heisenberg_page(heis_data, title="1D XXZ Heisenberg Chain"), encoding="utf-8")
      generated.append(path)

  # ── M28-F: Bose-Hubbard model ─────────────────────────────────────────────
  if args.scenario in {"bose_hubbard", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("bose_hubbard skipped: numpy not available", file=sys.stderr)
    else:
      bh_data = _bose_hubbard_frames(
          n_sites=args.bose_hubbard_sites,
          n_bosons=args.bose_hubbard_bosons,
          n_max=args.bose_hubbard_nmax,
          J=args.bose_hubbard_J,
          U=args.bose_hubbard_U,
          mu=args.bose_hubbard_mu,
          scenario="sweep",
          n_sweep=args.bose_hubbard_sweep_points,
          J_min=0.0,
          J_max=args.bose_hubbard_J_max,
      )
      path = out_dir / "quantum_bose_hubbard.html"
      path.write_text(_bose_hubbard_page(bh_data, title="Bose-Hubbard Model — Mott/Superfluid"), encoding="utf-8")
      generated.append(path)

  # ── M29-G: Transverse-Field Ising model ──────────────────────────────────
  if args.scenario in {"tfi", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("tfi skipped: numpy not available", file=sys.stderr)
    else:
      tfi_data = _tfi_frames(
          n_sites=args.tfi_sites,
          J=args.tfi_J,
          h=args.tfi_h,
          scenario="sweep",
          n_sweep=args.tfi_sweep_points,
          h_min=args.tfi_h_min,
          h_max=args.tfi_h_max,
      )
      path = out_dir / "quantum_tfi.html"
      path.write_text(_tfi_page(tfi_data, title="Transverse-Field Ising — QPT at h=J"), encoding="utf-8")
      generated.append(path)

  # ── M29-H: SSH model ──────────────────────────────────────────────────────
  if args.scenario in {"ssh", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("ssh skipped: numpy not available", file=sys.stderr)
    else:
      ssh_data = _ssh_frames(
          n_cells=args.ssh_cells,
          t=args.ssh_t,
          delta=args.ssh_delta,
          scenario="sweep",
          n_sweep=args.ssh_sweep_points,
          delta_min=args.ssh_delta_min,
          delta_max=args.ssh_delta_max,
      )
      path = out_dir / "quantum_ssh.html"
      path.write_text(_ssh_page(ssh_data, title="SSH Model — Bulk-Boundary Correspondence"), encoding="utf-8")
      generated.append(path)

  # ── M30-I: Kitaev chain ───────────────────────────────────────────────────
  if args.scenario in {"kitaev", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("kitaev skipped: numpy not available", file=sys.stderr)
    else:
      kitaev_data = _kitaev_frames(
          n_sites=args.kitaev_sites,
          t=args.kitaev_t,
          delta=args.kitaev_delta,
          mu=args.kitaev_mu,
          scenario="sweep",
          n_sweep=args.kitaev_sweep_points,
          mu_min=args.kitaev_mu_min,
          mu_max=args.kitaev_mu_max,
      )
      path = out_dir / "quantum_kitaev.html"
      path.write_text(_kitaev_page(kitaev_data, title="Kitaev Chain — Majorana Zero Modes"), encoding="utf-8")
      generated.append(path)

  # ── M30-J: Anderson localization ──────────────────────────────────────────
  if args.scenario in {"anderson", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("anderson skipped: numpy not available", file=sys.stderr)
    else:
      anderson_data = _anderson_frames(
          n_sites=args.anderson_sites,
          t=args.anderson_t,
          W=args.anderson_W,
          n_samples=args.anderson_samples,
          scenario="sweep",
          n_sweep=args.anderson_sweep_points,
          W_min=args.anderson_W_min,
          W_max=args.anderson_W_max,
      )
      path = out_dir / "quantum_anderson.html"
      path.write_text(_anderson_page(anderson_data, title="Anderson Localization — 1D Disorder"), encoding="utf-8")
      generated.append(path)

  # ── M25: Unified Dashboard ─────────────────────────────────────────────────
  # When all scenarios were generated and --dashboard is active, build a single
  # quantum_dashboard.html that embeds every scenario page via iframe srcdoc.
  if args.scenario == "all" and args.dashboard:
    # Map logical scenario keys → file stems used when writing individual pages.
    _DASH_STEM_MAP = {
        "rabi":               "rabi_population",
        "mps":                "mps_diagnostics",
        "spatial":            f"quantum_spatial_wavepacket_{args.spatial_potential}",
        "spatial2d":          f"quantum_spatial2d_wavepacket_{args.spatial2d_potential}",
        "spatial3d":          f"quantum_spatial3d_wavepacket_{args.spatial3d_potential}_{args.spatial3d_boundary}",
        "driven_barrier":     "quantum_driven_barrier",
        "bloch":              "quantum_bloch_oscillation",
        "gaussian_pulse":     "quantum_gaussian_pulse",
        "dirac2d":            f"quantum_dirac2d_{args.dirac2d_potential}",
        "lindblad":           f"quantum_lindblad_{args.lindblad_scenario}",
        "wigner_free":        "quantum_wigner_wigner_free",
        "wigner_barrier":     "quantum_wigner_wigner_barrier",
        "mps_ising":          "quantum_mps_ising",
        "mps_xx":             "quantum_mps_xx",
        "entanglement_rabi":  "quantum_entanglement_rabi",
        "entanglement_decay": "quantum_entanglement_decay",
        "vqe_ising":          "quantum_vqe_ising",
        "vqe_qaoa":           "quantum_vqe_qaoa",
        "qpe_phase":          "quantum_qpe_phase",
        "qpe_hamiltonian":    "quantum_qpe_hamiltonian",
        "qec_bit_flip":       "quantum_qec_bit_flip",
        "qec_phase_flip":     "quantum_qec_phase_flip",
        "qec_shor":           "quantum_qec_shor",
        "schwinger_vacuum":   "quantum_schwinger_vacuum",
        "schwinger_string":   "quantum_schwinger_string",
        "schwinger_dynamics": "quantum_schwinger_dynamics",
        "phi4_symmetric":     "quantum_phi4_symmetric",
        "phi4_broken":        "quantum_phi4_broken",
        "phi4_sweep":         "quantum_phi4_sweep",
        "phi4_dynamics":      "quantum_phi4_dynamics",
        "hubbard":            "quantum_hubbard",
        "floquet":            "quantum_floquet",
        "heisenberg":         "quantum_heisenberg",
        "bose_hubbard":       "quantum_bose_hubbard",
        "tfi":                "quantum_tfi",
        "ssh":                "quantum_ssh",
        "kitaev":             "quantum_kitaev",
        "anderson":           "quantum_anderson",
    }
    dash_pages: dict = {}
    for key, stem in _DASH_STEM_MAP.items():
      candidate = out_dir / f"{stem}.html"
      if candidate.exists():
        dash_pages[key] = candidate.read_text(encoding="utf-8")
    if dash_pages:
      dash_path = out_dir / "quantum_dashboard.html"
      dash_path.write_text(_dashboard_page(dash_pages), encoding="utf-8")
      generated.append(dash_path)

  for p in generated:
    print(f"wrote visualization: {p}")
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except (FileNotFoundError, RuntimeError) as ex:
    print(f"error: {ex}", file=sys.stderr)
    raise SystemExit(1)
