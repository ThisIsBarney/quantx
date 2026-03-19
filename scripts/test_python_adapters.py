#!/usr/bin/env python3
"""Smoke-test Python framework adapter helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
  root = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(root / "python"))
  from quantx_engine import Engine

  dll = root / "build-vs" / "Debug" / "quantx.dll"
  eng = Engine(str(dll))

  qiskit_req = {
      "num_qubits": 2,
      "shots": 1000,
      "observables": ["Z0", "Z1", "Z0Z1"],
      "operations": [
          {"name": "h", "qubits": [0]},
          {"name": "cx", "qubits": [0, 1]},
      ],
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
  }
  r2 = eng.run_cirq_like(cirq_req, {"enable_checkpoint": False})
  if not r2.get("ok", False):
    raise RuntimeError("run_cirq_like failed")

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

  print("python adapter smoke passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
