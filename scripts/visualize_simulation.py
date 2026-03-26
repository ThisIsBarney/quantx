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
    samples_t: int, nx: int, ny: int, potential_type: str, source: str = "python-reference"
) -> dict:
  """Build a 2D reference solution for the spatial wavepacket animation."""
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
    V = np.zeros_like(X)
  elif potential_type == "harmonic":
    omega = 0.28
    V = 0.5 * (omega * omega) * (X * X + Y * Y)
  elif potential_type == "well":
    wall = 5.0
    V = np.where((np.abs(X) <= 6.0) & (np.abs(Y) <= 6.0), 0.0, wall)
  else:  # barrier
    V = 0.9 * np.exp(-(X * X) / (2.0 * 0.6 * 0.6))

  kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
  ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
  KX, KY = np.meshgrid(kx, ky, indexing="xy")
  kinetic_phase = np.exp(-1j * 0.5 * (KX * KX + KY * KY) * dt)
  potential_half = np.exp(-1j * V * dt * 0.5)

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
    frames.append(
        {
            "t": t,
            "density": dens.tolist(),
            "phase": np.angle(psi).tolist(),
            "exp_x": exp_x,
            "exp_y": exp_y,
        }
    )
    if it == nt - 1:
      break

    # Strang splitting: V/2 -> T -> V/2
    psi = potential_half * psi
    psi_k = np.fft.fft2(psi)
    psi = np.fft.ifft2(kinetic_phase * psi_k)
    psi = potential_half * psi
    psi = normalize(psi)

  return {
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


def main() -> int:
  """Parse CLI arguments, generate requested visualizations, and write them to disk."""
  parser = argparse.ArgumentParser(
      description="Generate QuantX visualization HTML.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=CLI_EXAMPLES,
  )
  parser.add_argument("--scenario", choices=["rabi", "mps", "spatial", "spatial2d", "spatial3d", "dirac2d", "all"], default="all")
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
  parser.add_argument("--out-dir", default="out/visualizations")
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
  engine = _make_engine(root) if args.scenario in {"rabi", "mps", "spatial", "spatial2d", "spatial3d", "all"} or args.solver_mode not in {"python", None} else None
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

  for p in generated:
    print(f"wrote visualization: {p}")
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except (FileNotFoundError, RuntimeError) as ex:
    print(f"error: {ex}", file=sys.stderr)
    raise SystemExit(1)
