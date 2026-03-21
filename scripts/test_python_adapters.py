#!/usr/bin/env python3
"""Smoke-test Python framework adapter helpers."""

from __future__ import annotations

import sys
from pathlib import Path


def _normalize_caps(capabilities: dict | None) -> dict:
  """Build a stable capability payload for pure-Python tests."""
  return capabilities or {
      "backend": "statevector",
      "supported_gates": ["H", "CNOT", "CX", "SWAP"],
      "supported_noise_models": ["none", "depolarizing", "amplitude_damping"],
      "max_qubits": 25,
      "max_shots": 1000,
      "supports_qiskit_like": True,
      "supports_cirq_like": True,
  }


def test_pure_python_helpers() -> None:
  """Exercise Python-only capability and error mapping helpers."""
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
  from quantx_engine import CircuitError, Engine

  caps = _normalize_caps(None)
  summary = Engine.circuit_capability_summary(caps)
  if "backend=statevector" not in summary:
    raise RuntimeError("capability summary should include backend")
  if not Engine.circuit_supports_gate(caps, "cx"):
    raise RuntimeError("capability helper should normalize CX to CNOT")
  if not Engine.circuit_supports_noise_model(caps, "DEPOLARIZING"):
    raise RuntimeError("capability helper should normalize noise model names")

  ok_payload = {"ok": True, "value": 1}
  if Engine.raise_for_circuit_error(ok_payload) is not ok_payload:
    raise RuntimeError("successful circuit payload should pass through unchanged")

  err_payload = {
      "ok": False,
      "error_code": "QX_UNSUPPORTED_GATE",
      "error_message": "unsupported gate: foo",
  }
  try:
    Engine.raise_for_circuit_error(err_payload)
  except CircuitError as exc:
    if exc.code != "QX_UNSUPPORTED_GATE":
      raise RuntimeError("circuit error should preserve machine-readable code")
    if "unsupported gate" not in str(exc):
      raise RuntimeError("circuit error should preserve human-readable message")
    if exc.payload.get("error_code") != "QX_UNSUPPORTED_GATE":
      raise RuntimeError("circuit error payload should preserve the original response")
  else:
    raise RuntimeError("circuit error helper should raise on ok=false payloads")


def _native_dll_is_compatible(root: Path) -> tuple[bool, str]:
  """Return whether a native DLL exists and is compatible with the current Python runtime."""
  dll = root / "build-vs" / "Debug" / "quantx.dll"
  if not dll.exists():
    return False, "native quantx.dll not found"

  try:
    data = dll.read_bytes()
  except OSError as exc:
    return False, f"unable to read native DLL: {exc}"

  if len(data) < 0x40 or data[0:2] != b"MZ":
    return False, "native quantx.dll is not a PE file"
  pe_offset = int.from_bytes(data[0x3C:0x40], "little")
  if pe_offset < 0 or pe_offset + 6 > len(data):
    return False, "native quantx.dll is truncated"
  machine = int.from_bytes(data[pe_offset + 4:pe_offset + 6], "little")
  if machine != 0x8664:
    return False, f"native quantx.dll is not x64-compatible: {dll}"
  return True, str(dll)


def main() -> int:
  """Run Python adapter tests and skip native-dependent checks when needed."""
  root = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(root / "python"))
  from quantx_engine import Engine

  test_pure_python_helpers()

  compatible, detail = _native_dll_is_compatible(root)
  if not compatible:
    print(f"python adapter smoke skipped native integration: {detail}")
    print("python adapter pure-python checks passed")
    return 0

  eng = Engine(detail)

  caps = eng.circuit_capabilities()
  if caps.get("backend") != "statevector":
    raise RuntimeError("circuit_capabilities backend mismatch")
  supported_gates = set(caps.get("supported_gates", []))
  for gate in {"H", "CNOT", "CX", "SWAP"}:
    if gate not in supported_gates:
      raise RuntimeError(f"missing supported gate: {gate}")
  supported_noise = set(caps.get("supported_noise_models", []))
  for noise in {"none", "depolarizing", "amplitude_damping"}:
    if noise not in supported_noise:
      raise RuntimeError(f"missing supported noise model: {noise}")
  if int(caps.get("max_qubits", 0)) < 25:
    raise RuntimeError("circuit_capabilities max_qubits too small")
  if int(caps.get("max_shots", 0)) < 1000:
    raise RuntimeError("circuit_capabilities max_shots too small")

  qiskit_req = {
      "num_qubits": 2,
      "shots": 1000,
      "observables": ["Z0", "Z1", "Z0Z1"],
      "operations": [
          {"name": "h", "qubits": [0]},
          {"name": "cx", "qubits": [0, 1]},
      ],
      "seed": 1234,
  }
  r1 = eng.run_qiskit_like(qiskit_req, {"enable_checkpoint": False})
  if not r1.get("ok", False):
    raise RuntimeError("run_qiskit_like failed")

  cirq_req = {
      "qubits": 2,
      "repetitions": 1000,
      "observables": ["Z0", "Z1", "Z0Z1"],
      "moments": [
          [{"gate": "H", "targets": [0]}],
          [{"gate": "CNOT", "controls": [0], "targets": [1]}],
      ],
      "seed": 1234,
  }
  r2 = eng.run_cirq_like(cirq_req, {"enable_checkpoint": False})
  if not r2.get("ok", False):
    raise RuntimeError("run_cirq_like failed")
  if r1.get("backend") != r2.get("backend"):
    raise RuntimeError("adapter backends diverged")
  if r1.get("expectation_exact") != r2.get("expectation_exact"):
    raise RuntimeError("adapter exact expectations diverged")
  if r1.get("counts") != r2.get("counts"):
    raise RuntimeError("adapter sampled counts diverged")
  if r1.get("error_summary") != r2.get("error_summary"):
    raise RuntimeError("adapter error summaries diverged")

  bad_gate = eng.run_circuit({
      "qubits": 1,
      "operations": [{"name": "foo", "targets": [0]}],
  }, {"enable_checkpoint": False})
  if bad_gate.get("ok", True):
    raise RuntimeError("unsupported gate should fail")
  if bad_gate.get("error_code") != "QX_UNSUPPORTED_GATE":
    raise RuntimeError("unsupported gate should return QX_UNSUPPORTED_GATE")
  if "unsupported gate" not in bad_gate.get("error_message", ""):
    raise RuntimeError("unsupported gate should include human message")

  bad_noise = eng.validate_circuit({
      "qubits": 1,
      "operations": [{"name": "x", "targets": [0]}],
      "noise": {"kind": "phase_flip"},
  })
  if bad_noise.get("ok", True):
    raise RuntimeError("unsupported noise should fail validation")
  if bad_noise.get("error_code") != "QX_UNSUPPORTED_NOISE":
    raise RuntimeError("unsupported noise should return QX_UNSUPPORTED_NOISE")
  if "unsupported noise model" not in bad_noise.get("error_message", ""):
    raise RuntimeError("unsupported noise should include human message")

  continuous_problem = {
      "kind": "continuous_space_2d",
      "continuous_space_2d": {
          "nx": 64,
          "ny": 64,
          "dx": 0.1,
          "dy": 0.1,
          "dt": 0.001,
          "steps": 20,
          "boundary": "periodic",
          "potential_kind": "none",
          "potential_params": {
              "psi_sigma": 0.2,
              "psi_kx": 0.0,
              "psi_ky": 0.0,
              "psi_x0": 0.0,
              "psi_y0": 0.0,
          },
      },
  }
  v = eng.validate_problem(continuous_problem)
  if not v.get("ok", False):
    raise RuntimeError("validate_problem for continuous2d failed")

  r3 = eng.run(continuous_problem, {"enable_checkpoint": False})
  if not r3.get("ok", False):
    raise RuntimeError("run continuous2d failed")
  if r3.get("backend") != "continuous2d":
    raise RuntimeError("run continuous2d did not select continuous2d backend")
  if "norm" not in r3.get("observables", {}):
    raise RuntimeError("run continuous2d missing norm observable")

  continuous3d_problem = {
      "kind": "continuous_space_3d",
      "continuous_space_3d": {
          "nx": 24,
          "ny": 24,
          "nz": 24,
          "dx": 0.1,
          "dy": 0.1,
          "dz": 0.1,
          "dt": 0.001,
          "steps": 24,
          "boundary": "periodic",
          "potential_kind": "none",
          "potential_params": {
              "psi_sigma": 0.25,
              "psi_kx": 0.0,
              "psi_ky": 0.0,
              "psi_kz": 0.0,
              "psi_x0": 0.0,
              "psi_y0": 0.0,
              "psi_z0": 0.0,
          },
      },
  }

  v3d = eng.validate_problem(continuous3d_problem)
  if not v3d.get("ok", False):
    raise RuntimeError("validate_problem for continuous3d failed")

  e3d = eng.explain(continuous3d_problem, {"enable_checkpoint": False})
  if e3d.get("selected_backend") != "continuous3d":
    raise RuntimeError("explain continuous3d did not select continuous3d backend")

  r4 = eng.run(continuous3d_problem, {"enable_checkpoint": False})
  if not r4.get("ok", False):
    raise RuntimeError("run continuous3d failed")
  if r4.get("backend") != "continuous3d":
    raise RuntimeError("run continuous3d did not select continuous3d backend")
  if "norm" not in r4.get("observables", {}):
    raise RuntimeError("run continuous3d missing norm observable")
  if "sigma_z" not in r4.get("observables", {}):
    raise RuntimeError("run continuous3d missing sigma_z observable")

  print("python adapter smoke passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
