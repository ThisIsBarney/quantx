import ctypes
import json
import os
from ctypes import c_char_p, c_void_p


class CircuitError(RuntimeError):
  """Structured circuit execution error carrying a machine-readable code."""

  def __init__(self, code: str, message: str, payload: dict | None = None):
    """Create a circuit error with the native code, human message, and payload."""
    super().__init__(message)
    self._code = str(code)
    self._payload = dict(payload or {})

  @property
  def code(self) -> str:
    """Return the machine-readable error code."""
    return self._code

  @property
  def payload(self) -> dict:
    """Return the raw result payload that triggered this exception."""
    return dict(self._payload)


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
    self._lib.qx_circuit_capabilities.argtypes = []
    self._lib.qx_circuit_capabilities.restype = c_void_p
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

  def circuit_capabilities(self) -> dict:
    """Return the gate-model capability contract exposed by the native runtime."""
    ptr = self._lib.qx_circuit_capabilities()
    if not ptr:
      raise RuntimeError("native call returned null pointer")
    try:
      raw = ctypes.cast(ptr, c_char_p).value.decode("utf-8")
      return json.loads(raw)
    finally:
      self._lib.qx_free_string(ptr)

  def circuit_capability_summary(self) -> str:
    """Return a compact human-readable summary of the native circuit capabilities."""
    return self.summarize_circuit_capabilities(self.circuit_capabilities())

  def supports_circuit_gate(self, gate: str) -> bool:
    """Check whether the native circuit backend advertises a specific gate."""
    return self.circuit_supports_gate(self.circuit_capabilities(), gate)

  def supports_noise_model(self, noise_model: str) -> bool:
    """Check whether the native circuit backend advertises a specific noise model."""
    return self.circuit_supports_noise_model(self.circuit_capabilities(), noise_model)

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

  def run_circuit_checked(self, circuit: dict, options: dict | None = None) -> dict:
    """Run a circuit request and raise CircuitError on an unsupported result."""
    result = self.run_circuit(circuit, options)
    self.raise_for_circuit_error(result)
    return result

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

  @staticmethod
  def circuit_error_code(result: dict) -> str | None:
    """Extract the machine-readable circuit error code from a result payload."""
    if not isinstance(result, dict):
      return None
    if result.get("ok", True):
      return None
    code = result.get("error_code")
    if isinstance(code, str) and code.strip():
      return code.strip()
    code = result.get("code")
    if isinstance(code, str) and code.strip():
      return code.strip()
    return None

  @staticmethod
  def circuit_error_message(result: dict) -> str | None:
    """Extract the human-readable circuit error message from a result payload."""
    if not isinstance(result, dict):
      return None
    if result.get("ok", True):
      return None
    message = result.get("error_message")
    if isinstance(message, str) and message.strip():
      return message.strip()
    message = result.get("error")
    if isinstance(message, str) and message.strip():
      return message.strip()
    return None

  @staticmethod
  def raise_for_circuit_error(result: dict) -> dict:
    """Raise CircuitError when a circuit result reports ok=false.

    The method returns the original payload unchanged when it is successful so
    callers can chain it in checked execution helpers.
    """
    if not isinstance(result, dict):
      raise CircuitError("QX_INVALID_RESPONSE", "circuit result must be a JSON object", {"result": result})
    if result.get("ok", True):
      return result
    code = Engine.circuit_error_code(result) or "QX_UNKNOWN_ERROR"
    message = Engine.circuit_error_message(result) or "circuit execution failed"
    raise CircuitError(code, message, result)

  @staticmethod
  def circuit_capability_summary(capabilities: dict | None) -> str:
    """Return a compact summary string for a circuit capability payload."""
    if not isinstance(capabilities, dict) or not capabilities:
      return "n/a"

    parts = []
    backend = capabilities.get("backend")
    if isinstance(backend, str) and backend.strip():
      parts.append(f"backend={backend.strip()}")

    gates = Engine._normalize_string_list(capabilities.get("supported_gates"))
    if gates:
      preview = ", ".join(gates[:6])
      if len(gates) > 6:
        preview += ", ..."
      parts.append(f"gates=[{preview}]")

    noise = Engine._normalize_string_list(capabilities.get("supported_noise_models"))
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
      return "custom[" + ", ".join(sorted(capabilities.keys())) + "]"
    return "; ".join(parts)

  @staticmethod
  def circuit_supports_gate(capabilities: dict | None, gate: str) -> bool:
    """Check whether a capability payload lists a gate in its supported vocabulary."""
    normalized_gate = Engine._normalize_gate_name(gate)
    gates = {Engine._normalize_gate_name(item) for item in Engine._normalize_string_list(
        capabilities.get("supported_gates") if isinstance(capabilities, dict) else None
    )}
    return normalized_gate in gates

  @staticmethod
  def circuit_supports_noise_model(capabilities: dict | None, noise_model: str) -> bool:
    """Check whether a capability payload lists a supported noise model."""
    noise_models = {item.strip().lower() for item in Engine._normalize_string_list(
        capabilities.get("supported_noise_models") if isinstance(capabilities, dict) else None
    )}
    return noise_model.strip().lower() in noise_models

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

  @staticmethod
  def _normalize_string_list(values) -> list[str]:
    """Normalize a JSON list into a compact list of non-empty strings."""
    if not isinstance(values, list):
      return []
    out = []
    for value in values:
      if isinstance(value, str):
        text = value.strip()
        if text:
          out.append(text)
      elif value is not None:
        out.append(str(value))
    return out

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
    out: dict = {
        "qubits": int(qubits),
        "shots": int(request.get("shots", 0)),
        "observables": list(request.get("observables", [])),
        "operations": ops,
        "seed": int(request.get("seed", 42)),
    }
    if isinstance(request.get("noise"), dict):
      out["noise"] = request["noise"]
    if "depolarizing_prob" in request:
      out["depolarizing_prob"] = float(request["depolarizing_prob"])
    if "amplitude_damping_gamma" in request:
      out["amplitude_damping_gamma"] = float(request["amplitude_damping_gamma"])
    if isinstance(request.get("parameter_map"), dict):
      out["parameter_map"] = {str(k): float(v) for k, v in request["parameter_map"].items()}
    if "clbits" in request:
      out["clbits"] = int(request["clbits"])
    return out

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
    out: dict = {
        "qubits": int(qubits),
        "shots": int(shots),
        "observables": list(request.get("observables", [])),
        "operations": ops,
        "seed": int(request.get("seed", 42)),
    }
    if isinstance(request.get("noise"), dict):
      out["noise"] = request["noise"]
    if "depolarizing_prob" in request:
      out["depolarizing_prob"] = float(request["depolarizing_prob"])
    if "amplitude_damping_gamma" in request:
      out["amplitude_damping_gamma"] = float(request["amplitude_damping_gamma"])
    if isinstance(request.get("parameter_map"), dict):
      out["parameter_map"] = {str(k): float(v) for k, v in request["parameter_map"].items()}
    if "clbits" in request:
      out["clbits"] = int(request["clbits"])
    return out

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
      candidates = [
          os.path.join(cwd, "build", "Debug", "libquantx.dylib"),
          os.path.join(cwd, "build", "libquantx.dylib"),
      ]
      for p in candidates:
        if os.path.exists(p):
          return p
      return candidates[0]
    candidates = [
        os.path.join(cwd, "build", "Debug", "libquantx.so"),
        os.path.join(cwd, "build", "libquantx.so"),
    ]
    for p in candidates:
      if os.path.exists(p):
        return p
    return candidates[0]
