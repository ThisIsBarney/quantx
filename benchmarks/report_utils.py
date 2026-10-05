"""Shared helpers for benchmark reports, gate policy, and trend labeling."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def repo_root() -> Path:
  """Return the repository root for the current benchmark helper module."""
  return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
  """Load a JSON file into a dictionary."""
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def engine_library_path(root: Path) -> str:
  """Path of the native library the benchmark runners load.

  QUANTX_LIB_PATH when set; otherwise the Release build in build-vs, then the Debug one.
  Both runners used to hard-code build-vs/Debug, so when the benchmark gate moved to a
  Release build every suite failed to load the engine. If neither build exists, the
  Release path is returned and Engine() reports it as missing.
  """
  explicit = os.environ.get("QUANTX_LIB_PATH")
  if explicit:
    return explicit
  for config in ("Release", "Debug"):
    candidate = root / "build-vs" / config / "quantx.dll"
    if candidate.exists():
      return str(candidate)
  return str(root / "build-vs" / "Release" / "quantx.dll")


def detect_platform_profile(env: dict[str, str] | None = None) -> str:
  """Detect the current benchmark platform profile.

  The profile is primarily used for report labeling and suite-level strategy
  selection. Local runs default to ``local`` while CI-style runs default to
  ``github-windows`` unless the caller overrides the environment variable.
  """
  env = env or os.environ  # type: ignore[assignment]
  explicit = str(env.get("QX_BENCHMARK_PLATFORM_PROFILE", "")).strip().lower()
  if explicit:
    return explicit
  if str(env.get("GITHUB_ACTIONS", "")).strip().lower() in {"1", "true", "yes", "on"}:
    return "github-windows"
  if str(env.get("CI", "")).strip().lower() in {"1", "true", "yes", "on"}:
    return "github-windows"
  return "local"


def load_platform_profiles(root: Path | None = None) -> dict[str, Any]:
  """Load the platform profile registry if it exists."""
  root = root or repo_root()
  path = root / "benchmarks" / "platform_profiles.json"
  if not path.exists():
    return {}
  data = load_json(path)
  return data if isinstance(data, dict) else {}


def suite_policy(
    suite_name: str,
    profile_name: str | None = None,
    suite_json: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
  """Resolve the effective gate policy for a suite on a given profile.

  The policy is intentionally small and append-only:
  - performance thresholds may be blocking or warning-only
  - accuracy/stability thresholds remain blocking
  - the returned label is suitable for trend/report headings
  """
  root = root or repo_root()
  profile_name = (profile_name or detect_platform_profile()).strip().lower()
  profiles = load_platform_profiles(root)
  profile = profiles.get(profile_name, {}) if isinstance(profiles, dict) else {}
  if not isinstance(profile, dict):
    profile = {}

  default_blocking = bool(profile.get("default_perf_warnings_blocking", False))
  default_label = str(profile.get("default_label", "perf warnings non-blocking"))
  suite_policies = profile.get("suite_policies", {})
  if not isinstance(suite_policies, dict):
    suite_policies = {}
  selected = suite_policies.get(suite_name, {})
  if not isinstance(selected, dict):
    selected = {}

  perf_blocking = bool(selected.get("perf_warnings_blocking", default_blocking))
  label = str(selected.get("label", default_label if not perf_blocking else "perf warnings blocking"))

  if suite_json:
    override = suite_json.get("benchmark_policy", {})
    if isinstance(override, dict):
      if "perf_warnings_blocking" in override:
        perf_blocking = bool(override["perf_warnings_blocking"])
      if "label" in override:
        label = str(override["label"])

  return {
      "profile": profile_name,
      "profile_label": str(profile.get("display_name", profile_name)),
      "perf_warnings_blocking": perf_blocking,
      "label": label,
  }


def get_continuous_config(problem: dict[str, Any]) -> dict[str, Any] | None:
  """Extract the active continuous-space grid configuration from a problem."""
  for key in ("continuous_space_2d", "continuous_space_3d"):
    cfg = problem.get(key)
    if not isinstance(cfg, dict):
      continue
    if key == "continuous_space_2d":
      nx = int(cfg.get("nx", 0) or 0)
      ny = int(cfg.get("ny", 0) or 0)
      return {
          "kind": key,
          "grid": [nx, ny],
          "grid_points": nx * ny,
          "steps": int(cfg.get("steps", 0) or 0),
          "dt": float(cfg.get("dt", 0.0) or 0.0),
      }
    nx = int(cfg.get("nx", 0) or 0)
    ny = int(cfg.get("ny", 0) or 0)
    nz = int(cfg.get("nz", 0) or 0)
    return {
        "kind": key,
        "grid": [nx, ny, nz],
        "grid_points": nx * ny * nz,
        "steps": int(cfg.get("steps", 0) or 0),
        "dt": float(cfg.get("dt", 0.0) or 0.0),
    }
  return None


def _as_string_list(value: Any) -> list[str]:
  """Normalize a capability field into a compact list of strings."""
  if not isinstance(value, list):
    return []
  out: list[str] = []
  for item in value:
    if isinstance(item, str):
      text = item.strip()
      if text:
        out.append(text)
    elif item is not None:
      out.append(str(item))
  return out


def summarize_circuit_capabilities(capabilities: dict[str, Any] | None) -> str:
  """Summarize circuit capability metadata for reports and trend tables."""
  if not isinstance(capabilities, dict) or not capabilities:
    return "n/a"

  parts: list[str] = []
  backend = capabilities.get("backend")
  if isinstance(backend, str) and backend.strip():
    parts.append(f"backend={backend.strip()}")

  gates = _as_string_list(capabilities.get("supported_gates"))
  if gates:
    preview = ", ".join(gates[:6])
    if len(gates) > 6:
      preview += ", ..."
    parts.append(f"gates=[{preview}]")

  noise = _as_string_list(capabilities.get("supported_noise_models"))
  if noise:
    preview = ", ".join(noise[:6])
    if len(noise) > 6:
      preview += ", ..."
    parts.append(f"noise=[{preview}]")

  max_qubits = capabilities.get("max_qubits")
  if isinstance(max_qubits, (int, float)) and not isinstance(max_qubits, bool):
    parts.append(f"max_qubits={int(max_qubits)}")

  max_shots = capabilities.get("max_shots")
  if isinstance(max_shots, (int, float)) and not isinstance(max_shots, bool):
    parts.append(f"max_shots={int(max_shots)}")

  adapter_flags = []
  for key, label in (("supports_qiskit_like", "qiskit_like"), ("supports_cirq_like", "cirq_like")):
    value = capabilities.get(key)
    if isinstance(value, bool):
      adapter_flags.append(f"{label}={'true' if value else 'false'}")
  if adapter_flags:
    parts.append("adapters={" + ", ".join(adapter_flags) + "}")

  if not parts:
    keys = ", ".join(sorted(capabilities.keys()))
    return f"custom[{keys}]"
  return "; ".join(parts)


def circuit_capability_diagnostics(capabilities: dict[str, Any] | None) -> dict[str, Any]:
  """Summarize circuit capabilities and downgrade missing fields to warnings.

  The diagnostics are advisory only. Missing fields should not fail benchmark
  reporting because the capability object is append-only and may evolve.
  """
  warnings: list[str] = []
  if not isinstance(capabilities, dict) or not capabilities:
    warnings.append("missing circuit capability payload")
    capabilities = {}

  backend = capabilities.get("backend")
  if not (isinstance(backend, str) and backend.strip()):
    warnings.append("missing backend")

  gates = _as_string_list(capabilities.get("supported_gates"))
  if not gates:
    warnings.append("missing supported_gates")

  noise = _as_string_list(capabilities.get("supported_noise_models"))
  if not noise:
    warnings.append("missing supported_noise_models")

  if not isinstance(capabilities.get("max_qubits"), (int, float)) or isinstance(capabilities.get("max_qubits"), bool):
    warnings.append("missing max_qubits")

  if not isinstance(capabilities.get("max_shots"), (int, float)) or isinstance(capabilities.get("max_shots"), bool):
    warnings.append("missing max_shots")

  status = "ok" if not warnings else "warn"
  diagnostic_tag = "ok" if not warnings else f"warn: {'; '.join(warnings)}"
  return {
      "status": status,
      "diagnostic_tag": diagnostic_tag,
      "diagnostic_warnings": warnings,
      "capability_summary": summarize_circuit_capabilities(capabilities),
      "backend": backend if isinstance(backend, str) and backend.strip() else "n/a",
      "supported_gate_count": len(gates),
      "supported_noise_model_count": len(noise),
      "max_qubits": capabilities.get("max_qubits") if isinstance(capabilities.get("max_qubits"), (int, float)) else "n/a",
      "max_shots": capabilities.get("max_shots") if isinstance(capabilities.get("max_shots"), (int, float)) else "n/a",
  }


def extract_continuous_frames(result: dict[str, Any]) -> list[dict[str, Any]]:
  """Return any native continuous export frames embedded in a benchmark result."""
  export_block = extract_continuous_export_block(result)
  if export_block is None:
    return []
  frames = export_block.get("frames")
  if isinstance(frames, list):
    return [frame for frame in frames if isinstance(frame, dict)]
  return []


def extract_continuous_export_block(result: dict[str, Any]) -> dict[str, Any] | None:
  """Return the first structured continuous-export block found in a result.

  The helper is intentionally tolerant of older payload shapes. It accepts the
  native ``continuous_export`` object, legacy list-only encodings, and nested
  ``export`` wrappers so that the reporting layer can keep working as the
  schema evolves append-only.
  """
  if not isinstance(result, dict):
    return None

  candidates: list[dict[str, Any]] = []
  for key in ("continuous_export", "continuous_frames", "continuous", "continuous_density_frames"):
    value = result.get(key)
    if isinstance(value, dict):
      candidates.append(value)
    elif isinstance(value, list):
      candidates.append({"frames": value})

  export_block = result.get("export")
  if isinstance(export_block, dict):
    for key in ("continuous_export", "continuous", "frames"):
      value = export_block.get(key)
      if isinstance(value, dict):
        candidates.append(value)
      elif isinstance(value, list):
        candidates.append({"frames": value})

  if isinstance(result.get("frames"), list):
    candidates.append({"frames": result["frames"]})

  for candidate in candidates:
    if any(key in candidate for key in ("frames", "schema_version", "export_config")):
      return candidate
  return None


def _compact_value(value: Any) -> str:
  """Convert a structured export field into a short human-readable token."""
  if value is None:
    return "n/a"
  if isinstance(value, bool):
    return "true" if value else "false"
  if isinstance(value, (int, float)):
    return str(value)
  if isinstance(value, str):
    text = value.strip()
    return text if text else "n/a"
  if isinstance(value, dict):
    parts = []
    for key in sorted(value.keys()):
      parts.append(f"{key}={_compact_value(value[key])}")
    return "{" + ", ".join(parts) + "}"
  if isinstance(value, list):
    return "[" + ", ".join(_compact_value(item) for item in value[:6]) + ("..." if len(value) > 6 else "") + "]"
  return str(value)


def _first_numeric(export_config: dict[str, Any] | None, keys: tuple[str, ...]) -> float | None:
  """Return the first numeric value found under any of the provided keys."""
  if not isinstance(export_config, dict):
    return None
  for key in keys:
    if key not in export_config:
      continue
    value = export_config.get(key)
    if isinstance(value, bool):
      continue
    if isinstance(value, (int, float)):
      return float(value)
  return None


def summarize_continuous_export_config(export_config: dict[str, Any] | None) -> str:
  """Summarize a continuous-export configuration for compact reports."""
  if not isinstance(export_config, dict) or not export_config:
    return "n/a"

  parts: list[str] = []

  frame_stride = _first_numeric(export_config, ("frame_stride", "export_frame_stride", "stride", "sample_stride"))
  if frame_stride is not None:
    parts.append(f"frame_stride={int(frame_stride) if frame_stride.is_integer() else frame_stride:g}")

  points_max = _first_numeric(export_config, ("points_max", "export_points_max", "max_points"))
  if points_max is not None:
    parts.append(f"points_max={int(points_max) if points_max.is_integer() else points_max:g}")

  density_floor = _first_numeric(
      export_config,
      ("density_floor_ratio", "export_density_floor_ratio", "floor_ratio", "density_floor"),
  )
  if density_floor is not None:
    parts.append(f"density_floor_ratio={density_floor:g}")

  if not parts:
    keys = ", ".join(sorted(export_config.keys()))
    return f"custom[{keys}]"

  return "; ".join(parts)


def explain_continuous_frame_sampling(
    export_config: dict[str, Any] | None,
    frames: list[dict[str, Any]],
    steps: int | None = None,
) -> str:
  """Explain how continuous frames were sampled without over-claiming accuracy.

  The output intentionally distinguishes between an explicit sampling rate and a
  best-effort observed cadence so that benchmark summaries do not imply that a
  successful gate automatically means there was no trend change.
  """
  if isinstance(export_config, dict):
    stride = _first_numeric(export_config, ("frame_stride", "export_frame_stride", "stride", "sample_stride"))
    if stride is not None and stride > 0.0:
      stride_text = int(stride) if stride.is_integer() else stride
      if float(stride_text) == 1.0:
        return "explicit sampling: 1 frame per step"
      return f"explicit sampling: 1 frame every {stride_text} step(s)"

  frame_count = len(frames)
  if frame_count <= 0:
    return "no exported frames"
  if steps is not None and steps > 0:
    return f"observed {frame_count} frame(s) across {steps} step(s); sampling stride not declared"
  return f"observed {frame_count} frame(s); sampling stride not declared"


def continuous_export_diagnostics(result: dict[str, Any], steps: int | None = None) -> dict[str, Any]:
  """Summarize continuous-export metadata and flag missing fields as warnings.

  The return value is intentionally advisory. It is suitable for human-readable
  report columns, but it never changes gate outcomes or hard-fails benchmark
  evaluation.
  """
  export_block = extract_continuous_export_block(result)
  frames = extract_continuous_frames(result)
  export_config = None
  schema_version = None
  if isinstance(export_block, dict):
    schema_version = export_block.get("schema_version")
    export_config = export_block.get("export_config") if isinstance(export_block.get("export_config"), dict) else None

  warnings: list[str] = []
  if export_block is None:
    warnings.append("missing continuous_export payload")
  if schema_version is None:
    warnings.append("missing schema_version")
  if export_config is None:
    warnings.append("missing export_config")
  if export_block is not None and not frames:
    warnings.append("missing frames")

  status = "ok" if not warnings else "warn"
  if warnings:
    diagnostic_tag = f"warn: {'; '.join(warnings)}"
  else:
    diagnostic_tag = "ok"

  return {
      "has_continuous_export": export_block is not None,
      "schema_version": schema_version if schema_version is not None else "n/a",
      "export_config_summary": summarize_continuous_export_config(export_config),
      "frame_sampling": explain_continuous_frame_sampling(export_config, frames, steps),
      "export_frame_count": len(frames),
      "export_volume": continuous_export_volume(frames),
      "diagnostic_status": status,
      "diagnostic_tag": diagnostic_tag,
      "diagnostic_warnings": warnings,
  }


def infer_fft_provider(result: dict[str, Any]) -> str:
  """Infer the active FFT provider from observable flags or backend markers."""
  observables = result.get("observables", {})
  if isinstance(observables, dict):
    if float(observables.get("fft_provider_cufft", 0.0) or 0.0) > 0.5:
      return "cufft"
    if float(observables.get("fft_provider_eigen", 0.0) or 0.0) > 0.5:
      return "eigen"
  backend = str(result.get("backend", "")).lower()
  if "cufft" in backend:
    return "cufft"
  if "eigen" in backend:
    return "eigen"
  return "n/a"


def continuous_export_volume(frames: list[dict[str, Any]]) -> int:
  """Compute a coarse export volume for a frame list.

  The result is the sum of exported sample points across all frames. Dense
  slices count their flattened scalar entries, and point clouds count logical
  [x, y, z, density] points rather than raw floats.
  """
  total = 0
  for frame in frames:
    frame_total = 0
    if "density_xy_flat" in frame and isinstance(frame["density_xy_flat"], list):
      frame_total += len(frame["density_xy_flat"])
    if "density_xz_flat" in frame and isinstance(frame["density_xz_flat"], list):
      frame_total += len(frame["density_xz_flat"])
    if "density_yz_flat" in frame and isinstance(frame["density_yz_flat"], list):
      frame_total += len(frame["density_yz_flat"])
    if "points_xyzd_flat" in frame and isinstance(frame["points_xyzd_flat"], list):
      frame_total += len(frame["points_xyzd_flat"]) // 4
    elif "points" in frame and isinstance(frame["points"], list):
      frame_total += len(frame["points"])
    elif "density" in frame and isinstance(frame["density"], list):
      frame_total += len(frame["density"])
    total += frame_total if frame_total > 0 else 1
  return total
