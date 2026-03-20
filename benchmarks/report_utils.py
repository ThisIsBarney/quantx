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


def extract_continuous_frames(result: dict[str, Any]) -> list[dict[str, Any]]:
  """Return any native continuous export frames embedded in a benchmark result."""
  if not isinstance(result, dict):
    return []

  candidates: list[Any] = []
  for key in ("continuous_density_frames", "continuous_export", "continuous_frames", "continuous"):
    value = result.get(key)
    if isinstance(value, list):
      candidates.append(value)
    elif isinstance(value, dict):
      frames = value.get("frames")
      if isinstance(frames, list):
        candidates.append(frames)

  export_block = result.get("export")
  if isinstance(export_block, dict):
    for key in ("continuous_export", "continuous", "frames"):
      value = export_block.get(key)
      if isinstance(value, list):
        candidates.append(value)
      elif isinstance(value, dict):
        frames = value.get("frames")
        if isinstance(frames, list):
          candidates.append(frames)

  if isinstance(result.get("frames"), list):
    candidates.append(result["frames"])

  for candidate in candidates:
    if isinstance(candidate, list) and candidate:
      first = candidate[0]
      if isinstance(first, dict):
        return [frame for frame in candidate if isinstance(frame, dict)]
  return []


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
