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

  print("python adapter smoke passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
