import ctypes
import json
import os
from ctypes import c_char_p, c_void_p


class Engine:
  """ctypes wrapper around the native QuantX library."""

  def __init__(self, lib_path=None):
    """Load the native library and bind the exported C ABI entrypoints."""
    self._lib_path = lib_path or self._default_library_path()
    self._lib = ctypes.CDLL(self._lib_path)
    self._lib.qx_run.argtypes = [c_char_p, c_char_p]
    self._lib.qx_run.restype = c_void_p
    self._lib.qx_explain_backend_choice.argtypes = [c_char_p, c_char_p]
    self._lib.qx_explain_backend_choice.restype = c_void_p
    self._lib.qx_run_circuit.argtypes = [c_char_p, c_char_p]
    self._lib.qx_run_circuit.restype = c_void_p
    self._lib.qx_validate_problem.argtypes = [c_char_p]
    self._lib.qx_validate_problem.restype = c_void_p
    self._lib.qx_validate_circuit.argtypes = [c_char_p]
    self._lib.qx_validate_circuit.restype = c_void_p
    self._lib.qx_version.argtypes = []
    self._lib.qx_version.restype = c_char_p
    self._lib.qx_free_string.argtypes = [c_void_p]
    self._lib.qx_free_string.restype = None

  def run(self, problem: dict, options: dict | None = None) -> dict:
    """Run a Hamiltonian simulation request and return the decoded JSON result."""
    options = options or {}
    return self._call(self._lib.qx_run, problem, options)

  def explain(self, problem: dict, options: dict | None = None) -> dict:
    """Ask the scheduler which backend the native runtime would select."""
    options = options or {}
    return self._call(self._lib.qx_explain_backend_choice, problem, options)

  def run_circuit(self, circuit: dict, options: dict | None = None) -> dict:
    """Run a circuit-model request through the native runtime."""
    options = options or {}
    return self._call(self._lib.qx_run_circuit, circuit, options)

  def run_qiskit_like(self, request: dict, options: dict | None = None) -> dict:
    """Execute a minimal Qiskit-like circuit payload.

    Expected shape:
    {
      "num_qubits": 2,                      # optional if "qubits" exists
      "shots": 1000,                        # optional
      "observables": ["Z0", "Z1"],          # optional
      "operations": [
        {"name": "h", "qubits": [0]},
        {"name": "cx", "qubits": [0, 1]},
        {"name": "rx", "qubits": [1], "params": [1.5708]}
      ]
    }
    """
    return self.run_circuit(self._from_qiskit_like(request), options)

  def run_cirq_like(self, request: dict, options: dict | None = None) -> dict:
    """Execute a minimal Cirq-like circuit payload.

    Expected shape:
    {
      "qubits": 2,
      "repetitions": 1000,                  # optional alias of shots
      "moments": [                          # optional, alternative to operations
        [{"gate": "H", "targets": [0]}],
        [{"gate": "CNOT", "controls": [0], "targets": [1]}]
      ],
      "operations": [ ... ]                 # optional direct operation list
    }
    """
    return self.run_circuit(self._from_cirq_like(request), options)

  def validate_problem(self, problem: dict) -> dict:
    """Validate a problem JSON payload against the native schema checker."""
    p = json.dumps(problem).encode("utf-8")
    ptr = self._lib.qx_validate_problem(p)
    if not ptr:
      raise RuntimeError("native call returned null pointer")
    try:
      raw = ctypes.cast(ptr, c_char_p).value.decode("utf-8")
      return json.loads(raw)
    finally:
      self._lib.qx_free_string(ptr)

  def validate_circuit(self, circuit: dict) -> dict:
    """Validate a circuit JSON payload against the native schema checker."""
    c = json.dumps(circuit).encode("utf-8")
    ptr = self._lib.qx_validate_circuit(c)
    if not ptr:
      raise RuntimeError("native call returned null pointer")
    try:
      raw = ctypes.cast(ptr, c_char_p).value.decode("utf-8")
      return json.loads(raw)
    finally:
      self._lib.qx_free_string(ptr)

  def version(self) -> str:
    """Return the native engine version string."""
    return self._lib.qx_version().decode("utf-8")

  @staticmethod
  def extract_continuous_export(result: dict) -> dict | None:
    """Normalize any continuous-space export payload embedded in a native result."""
    if not isinstance(result, dict):
      return None

    candidates = []
    for key in ("continuous_export", "continuous_frames", "continuous"):
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

    frames = result.get("frames")
    if isinstance(frames, list):
      candidates.append({"frames": frames})

    for candidate in candidates:
      if isinstance(candidate, dict) and isinstance(candidate.get("frames"), list) and candidate["frames"]:
        return candidate
    return None

  def _call(self, fn, problem: dict, options: dict) -> dict:
    """Invoke a native JSON API and decode the returned payload."""
    p = json.dumps(problem).encode("utf-8")
    o = json.dumps(options).encode("utf-8")
    ptr = fn(p, o)
    if not ptr:
      raise RuntimeError("native call returned null pointer")
    try:
      raw = ctypes.cast(ptr, c_char_p).value.decode("utf-8")
      return json.loads(raw)
    finally:
      self._lib.qx_free_string(ptr)

  @staticmethod
  def _normalize_gate_name(name: str) -> str:
    """Map framework-specific gate spellings into the native gate vocabulary."""
    n = name.upper()
    mapping = {
        "CX": "CNOT",
        "C-X": "CNOT",
    }
    return mapping.get(n, n)

  def _from_qiskit_like(self, request: dict) -> dict:
    """Convert a compact Qiskit-like request into the native circuit schema."""
    qubits = request.get("num_qubits", request.get("qubits"))
    if qubits is None:
      raise ValueError("qiskit-like request requires num_qubits or qubits")
    ops = []
    for raw in request.get("operations", []):
      name = self._normalize_gate_name(str(raw.get("name", "")))
      q = list(raw.get("qubits", []))
      controls = list(raw.get("controls", []))
      targets = list(raw.get("targets", []))
      if not targets:
        if name in {"CNOT", "CZ"} and len(q) >= 2:
          controls = controls or [q[0]]
          targets = [q[1]]
        else:
          targets = q
      op = {"name": name, "targets": targets}
      if controls:
        op["controls"] = controls
      if "params" in raw:
        op["params"] = list(raw["params"])
      ops.append(op)
    return {
        "qubits": int(qubits),
        "shots": int(request.get("shots", 0)),
        "observables": list(request.get("observables", [])),
        "operations": ops,
    }

  def _from_cirq_like(self, request: dict) -> dict:
    """Convert a compact Cirq-like request into the native circuit schema."""
    qubits = request.get("qubits")
    if qubits is None:
      raise ValueError("cirq-like request requires qubits")

    raw_ops = []
    if "operations" in request:
      raw_ops.extend(request.get("operations", []))
    for moment in request.get("moments", []):
      raw_ops.extend(moment)

    ops = []
    for raw in raw_ops:
      name = self._normalize_gate_name(str(raw.get("gate", raw.get("name", ""))))
      targets = list(raw.get("targets", raw.get("qubits", [])))
      controls = list(raw.get("controls", []))
      if name in {"CNOT", "CZ"} and not controls and len(targets) >= 2:
        controls = [targets[0]]
        targets = [targets[1]]
      op = {"name": name, "targets": targets}
      if controls:
        op["controls"] = controls
      if "params" in raw:
        op["params"] = list(raw["params"])
      ops.append(op)

    shots = request.get("shots", request.get("repetitions", 0))
    return {
        "qubits": int(qubits),
        "shots": int(shots),
        "observables": list(request.get("observables", [])),
        "operations": ops,
    }

  @staticmethod
  def _default_library_path():
    """Pick the most likely native library path for the current platform."""
    cwd = os.getcwd()
    if os.name == "nt":
      candidates = [
          os.path.join(cwd, "build", "Debug", "quantx.dll"),
          os.path.join(cwd, "build-vs", "Debug", "quantx.dll"),
      ]
      for p in candidates:
        if os.path.exists(p):
          return p
      return candidates[0]
    if os.uname().sysname == "Darwin":
      return os.path.join(cwd, "build", "libquantx.dylib")
    return os.path.join(cwd, "build", "libquantx.so")
