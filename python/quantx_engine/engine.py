"""C++ engine bridge — ctypes / pybind11 interface to the QuantX native library.

Provides the :class:`Engine` facade that dispatches to native C++ solvers
(Lanczos, Trotter, MPS) when the shared library is available, and falls
back transparently to the pure-Python implementations otherwise.

M159 adds a pybind11 fast path: when the ``_quantx_py`` extension module is
importable (built with ``QX_ENABLE_PYBIND=ON``), methods such as
:meth:`Engine.run_statevector` and :meth:`Engine.run_circuit_sv` return NumPy
arrays directly without any JSON round-trip overhead.

The native library is located via the environment variable
``QUANTX_LIB_PATH`` or by searching the standard platform-specific paths
(``quantx_engine.dll`` on Windows, ``libquantx_engine.so`` on Linux,
``libquantx_engine.dylib`` on macOS).

References
----------
.. [1] Python Software Foundation (2023). ctypes — A foreign function library
       for Python. https://doi.org/10.48550/arXiv.1811.04968
       (Official ctypes documentation: https://docs.python.org/3/library/ctypes.html)

.. [2] Beazley, D. (2009). *Python Essential Reference* (4th ed.).
       Addison-Wesley. ISBN 978-0-672-32978-4.

.. [3] Jakob, W., Rhinelander, J., & Moldovan, D. (2017). pybind11 — Seamless
       operability between C++11 and Python. https://github.com/pybind/pybind11
"""
import ctypes
import json
import os
from ctypes import c_char_p, c_void_p
from typing import TYPE_CHECKING

# ── M159: try to import the pybind11 extension once at module load time ───────
# The extension (_quantx_py.so / _quantx_py.pyd) is built with QX_ENABLE_PYBIND=ON
# and copied into the quantx_engine package directory alongside the ctypes shared
# library.  Use a relative import so it is found in the installed wheel's package
# directory rather than the top-level sys.path (which would always miss it).
_PYBIND_MODULE = None
try:
    from . import _quantx_py as _PYBIND_MODULE  # type: ignore[import]
except ImportError:
    _PYBIND_MODULE = None

if TYPE_CHECKING:  # pragma: no cover
    from quantx_engine.lindblad import JumpOperator, LindbladResult


class QuantXNativeUnavailable(RuntimeError):
    """Raised when the native QuantX shared library cannot be loaded.

    This happens when ``quantx-engine`` is installed as a pure-Python wheel
    (no C++ build) or when the library path is incorrect.  All pure-Python
    modules — simulators, chemistry solvers, error-correction codes, etc. —
    continue to work normally; only features that call through to the native
    C++ engine require the library.
    """


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
    """Load the native library and bind the exported C ABI entrypoints.

    Raises
    ------
    QuantXNativeUnavailable
        If the shared library cannot be found or loaded.  Pure-Python
        functionality is unaffected.
    """
    self._lib_path = lib_path or self._default_library_path()
    # On Windows, ctypes.CDLL cannot find DLL dependencies that live in the
    # MSYS2/UCRT64 runtime directory (libgomp-1.dll, libstdc++-6.dll, etc.).
    # Python 3.8+ provides os.add_dll_directory() to register extra search
    # paths before loading. We probe common MSYS2 locations automatically.
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        _msys2_dirs = [
            r"C:\msys64\ucrt64\bin",
            r"C:\msys64\mingw64\bin",
            r"C:\msys64\mingw32\bin",
        ]
        for _d in _msys2_dirs:
            if os.path.isdir(_d):
                try:
                    os.add_dll_directory(_d)
                except OSError:
                    pass
    try:
        self._lib = ctypes.CDLL(self._lib_path)
    except OSError as exc:
        raise QuantXNativeUnavailable(
            f"Cannot load the QuantX native library from '{self._lib_path}'.\n"
            "If you installed quantx-engine as a pure-Python package (no C++ build),\n"
            "the Engine() class is unavailable.  All other quantx_engine modules\n"
            "work without the native library.\n"
            f"  Original error: {exc}"
        ) from exc
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
    self._lib.qx_run_solver.argtypes = [c_char_p, c_char_p]
    self._lib.qx_run_solver.restype = c_void_p
    self._lib.qx_list_solvers.argtypes = []
    self._lib.qx_list_solvers.restype = c_void_p

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

  def capabilities(self) -> dict:
    """Alias for circuit_capabilities() — preferred entry point for M8 adapter integration."""
    return self.circuit_capabilities()

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

  def parse_qasm(self, qasm_source: str, shots: int = 0, seed: int = 42,
                 parameter_map: dict | None = None) -> dict:
    """Parse an OpenQASM 3.0 string and return a QuantX native circuit dict.

    Pure-Python transpiler — no native library call. The returned dict can be
    passed directly to run_circuit().
    """
    from quantx_engine.qasm import parse_qasm_to_circuit
    return parse_qasm_to_circuit(qasm_source, shots=shots, seed=seed,
                                  parameter_map=parameter_map)

  def run_qasm(self, qasm_source: str, options: dict | None = None,
               shots: int = 0, seed: int = 42,
               parameter_map: dict | None = None) -> dict:
    """Parse an OpenQASM 3.0 string and execute it through the native circuit backend.

    Equivalent to engine.run_circuit(engine.parse_qasm(qasm_source, ...), options).
    """
    circuit = self.parse_qasm(qasm_source, shots=shots, seed=seed,
                               parameter_map=parameter_map)
    return self.run_circuit(circuit, options)

  def run_circuit_checked(self, circuit: dict, options: dict | None = None) -> dict:
    """Run a circuit request and raise CircuitError on an unsupported result."""
    result = self.run_circuit(circuit, options)
    self.raise_for_circuit_error(result)
    return result

  # ── M159: pybind11 zero-copy fast paths ─────────────────────────────────────

  @staticmethod
  def pybind11_available() -> bool:
    """Return True when the pybind11 extension module ``_quantx_py`` is loaded.

    When True, :meth:`run_statevector` and :meth:`run_circuit_sv` avoid the
    JSON serialisation overhead and return NumPy arrays via a direct C++→Python
    memory copy.
    """
    return _PYBIND_MODULE is not None

  def run_statevector(
      self,
      problem: dict,
      options: dict | None = None,
  ):
    """Run a Hamiltonian simulation and return the statevector.

    When the pybind11 extension is available (:meth:`pybind11_available` is
    ``True``), this method returns a ``numpy.ndarray`` of dtype ``complex128``
    with shape ``(2**n_qubits,)`` — no JSON serialisation is performed.

    Without pybind11 the result falls back to the ctypes JSON path and the
    statevector is returned as a list of ``[re, im]`` pairs embedded in the
    normal result dict under the ``"state"`` key.

    Parameters
    ----------
    problem : dict
        Problem specification (same schema as :meth:`run`).
    options : dict, optional
        Runtime options.

    Returns
    -------
    numpy.ndarray (complex128) or dict
        NumPy array via pybind11, or full result dict via ctypes fallback.
    """
    options = options or {}
    if _PYBIND_MODULE is not None:
      return _PYBIND_MODULE.run_statevector(
          json.dumps(problem),
          json.dumps(options),
      )
    # ctypes fallback — result["state"] is a list of [re, im] pairs
    return self.run(problem, options)

  def run_circuit_sv(self, circuit: dict):
    """Execute a gate-level circuit and return the output statevector.

    When the pybind11 extension is available, this returns a
    ``numpy.ndarray[complex128]`` directly.  Otherwise it calls the ctypes
    path and returns the full result dict.

    Parameters
    ----------
    circuit : dict
        Circuit payload (same schema as :meth:`run_circuit`).

    Returns
    -------
    numpy.ndarray (complex128) or dict
    """
    if _PYBIND_MODULE is not None:
      return _PYBIND_MODULE.run_circuit_sv(json.dumps(circuit))
    return self.run_circuit(circuit)

  def run_circuit_counts_np(self, circuit: dict) -> dict:
    """Execute a gate-level circuit and return measurement counts.

    When pybind11 is available, the counts dict is populated directly from C++
    (no JSON round-trip).  Otherwise this is equivalent to calling
    ``run_circuit(circuit).get("counts", {})``.

    Parameters
    ----------
    circuit : dict

    Returns
    -------
    dict[str, int]
    """
    if _PYBIND_MODULE is not None:
      return _PYBIND_MODULE.run_circuit_counts(json.dumps(circuit))
    result = self.run_circuit(circuit)
    return result.get("counts", {})

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
        "MEAS": "MEASURE",
        "M": "MEASURE",
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
      clbits_ref = list(raw.get("clbits", []))
      if not targets:
        if name in {"CNOT", "CZ"} and len(q) >= 2:
          controls = controls or [q[0]]
          targets = [q[1]]
        else:
          targets = q
      if name == "MEASURE" and not controls and clbits_ref:
        controls = [clbits_ref[0]]
      op = {"name": name, "targets": targets}
      if controls:
        op["controls"] = controls
      if "params" in raw:
        op["params"] = list(raw["params"])
      if "param_names" in raw:
        op["param_names"] = list(raw["param_names"])
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
      clbits_ref = list(raw.get("clbits", []))
      if name in {"CNOT", "CZ"} and not controls and len(targets) >= 2:
        controls = [targets[0]]
        targets = [targets[1]]
      if name == "MEASURE" and not controls and clbits_ref:
        controls = [clbits_ref[0]]
      op = {"name": name, "targets": targets}
      if controls:
        op["controls"] = controls
      if "params" in raw:
        op["params"] = list(raw["params"])
      if "param_names" in raw:
        op["param_names"] = list(raw["param_names"])
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

  def run_solver(self, solver: str, options: dict | None = None) -> dict:
    """Run a named physics solver by key with optional JSON-serializable options dict."""
    opts = options or {}
    s = solver.encode("utf-8")
    o = json.dumps(opts).encode("utf-8")
    ptr = self._lib.qx_run_solver(s, o)
    if not ptr:
      raise RuntimeError("native call returned null pointer")
    try:
      raw = ctypes.cast(ptr, c_char_p).value.decode("utf-8")
      return json.loads(raw)
    finally:
      self._lib.qx_free_string(ptr)

  def list_solvers(self) -> list[str]:
    """Return the list of all available solver names supported by the native engine."""
    ptr = self._lib.qx_list_solvers()
    if not ptr:
      raise RuntimeError("native call returned null pointer")
    try:
      raw = ctypes.cast(ptr, c_char_p).value.decode("utf-8")
      return json.loads(raw)
    finally:
      self._lib.qx_free_string(ptr)

  async def run_solver_async(self, solver: str, options: dict | None = None) -> dict:
    """Async variant of run_solver — runs in a thread pool to avoid blocking the event loop."""
    import asyncio
    return await asyncio.to_thread(self.run_solver, solver, options)

  # --- Typed convenience methods for all 37 solvers ---

  def run_vqe(self, options: dict | None = None) -> dict:
    """Run variational eigensolver (VQE) / QAOA optimization."""
    return self.run_solver("vqe", options)

  def run_qpe(self, options: dict | None = None) -> dict:
    """Run Quantum Phase Estimation."""
    return self.run_solver("qpe", options)

  def run_qec(self, options: dict | None = None) -> dict:
    """Run Quantum Error Correction Monte Carlo."""
    return self.run_solver("qec", options)

  def run_qfield(self, options: dict | None = None) -> dict:
    """Run Schwinger model (1+1D QED)."""
    return self.run_solver("qfield", options)

  def run_phi4(self, options: dict | None = None) -> dict:
    """Run phi4 scalar quantum field theory."""
    return self.run_solver("phi4", options)

  def run_hubbard(self, options: dict | None = None) -> dict:
    """Run 1D Hubbard model (Mott insulator)."""
    return self.run_solver("hubbard", options)

  def run_floquet(self, options: dict | None = None) -> dict:
    """Run Floquet kicked-Ising model."""
    return self.run_solver("floquet", options)

  def run_heisenberg(self, options: dict | None = None) -> dict:
    """Run 1D XXZ Heisenberg spin chain."""
    return self.run_solver("heisenberg", options)

  def run_bose_hubbard(self, options: dict | None = None) -> dict:
    """Run Bose-Hubbard model."""
    return self.run_solver("bose_hubbard", options)

  def run_tfi(self, options: dict | None = None) -> dict:
    """Run Transverse-Field Ising model."""
    return self.run_solver("tfi", options)

  def run_ssh(self, options: dict | None = None) -> dict:
    """Run SSH topological insulator."""
    return self.run_solver("ssh", options)

  def run_kitaev(self, options: dict | None = None) -> dict:
    """Run Kitaev chain (p-wave topological superconductor)."""
    return self.run_solver("kitaev", options)

  def run_anderson(self, options: dict | None = None) -> dict:
    """Run Anderson localization."""
    return self.run_solver("anderson", options)

  def run_chern(self, options: dict | None = None) -> dict:
    """Run 2D Chern insulator (Qi-Wu-Zhang)."""
    return self.run_solver("chern", options)

  def run_hofstadter(self, options: dict | None = None) -> dict:
    """Run Hofstadter butterfly."""
    return self.run_solver("hofstadter", options)

  def run_mbl(self, options: dict | None = None) -> dict:
    """Run Many-Body Localization."""
    return self.run_solver("mbl", options)

  def run_spectral(self, options: dict | None = None) -> dict:
    """Run spectral function / dynamical structure factor."""
    return self.run_solver("spectral", options)

  def run_tenfold(self, options: dict | None = None) -> dict:
    """Run 10-fold way topological classification."""
    return self.run_solver("tenfold", options)

  def run_second_chern(self, options: dict | None = None) -> dict:
    """Run second Chern number (4D Quantum Hall)."""
    return self.run_solver("second_chern", options)

  def run_wilson_loop(self, options: dict | None = None) -> dict:
    """Run Wilson loop / Wannier Charge Center."""
    return self.run_solver("wilson_loop", options)

  def run_edge_states(self, options: dict | None = None) -> dict:
    """Run edge states / surface spectrum."""
    return self.run_solver("edge_states", options)

  def run_haldane(self, options: dict | None = None) -> dict:
    """Run Haldane model (Quantum Anomalous Hall)."""
    return self.run_solver("haldane", options)

  def run_weyl(self, options: dict | None = None) -> dict:
    """Run Weyl semimetal / Surface Fermi Arcs."""
    return self.run_solver("weyl", options)

  def run_fu_kane(self, options: dict | None = None) -> dict:
    """Run Fu-Kane superconductor."""
    return self.run_solver("fu_kane", options)

  def run_lutchyn_oreg(self, options: dict | None = None) -> dict:
    """Run Lutchyn-Oreg Majorana nanowire."""
    return self.run_solver("lutchyn_oreg", options)

  def run_kitaev_honeycomb(self, options: dict | None = None) -> dict:
    """Run Kitaev honeycomb model."""
    return self.run_solver("kitaev_honeycomb", options)

  def run_bbh(self, options: dict | None = None) -> dict:
    """Run BBH quadrupole insulator."""
    return self.run_solver("bbh", options)

  def run_floquet_topo(self, options: dict | None = None) -> dict:
    """Run Floquet topological insulator."""
    return self.run_solver("floquet_topo", options)

  def run_non_herm(self, options: dict | None = None) -> dict:
    """Run non-Hermitian topology."""
    return self.run_solver("non_herm", options)

  def run_aah(self, options: dict | None = None) -> dict:
    """Run Aubry-Andre-Harper quasicrystal."""
    return self.run_solver("aah", options)

  def run_syk(self, options: dict | None = None) -> dict:
    """Run Sachdev-Ye-Kitaev model."""
    return self.run_solver("syk", options)

  def run_kondo(self, options: dict | None = None) -> dict:
    """Run Kondo/SIAM model."""
    return self.run_solver("kondo", options)

  def run_luttinger(self, options: dict | None = None) -> dict:
    """Run Luttinger liquid (XXZ chain)."""
    return self.run_solver("luttinger", options)

  def run_toric(self, options: dict | None = None) -> dict:
    """Run toric code / surface code."""
    return self.run_solver("toric", options)

  def run_mipt(self, options: dict | None = None) -> dict:
    """Run measurement-induced phase transition."""
    return self.run_solver("mipt", options)

  def run_rg(self, options: dict | None = None) -> dict:
    """Run Richardson-Gaudin/BCS pairing."""
    return self.run_solver("rg", options)

  def run_entanglement(self, options: dict | None = None) -> dict:
    """Run bipartite entanglement entropy."""
    return self.run_solver("entanglement", options)

  # --- Async variants for all 37 solvers ---

  async def run_vqe_async(self, options: dict | None = None) -> dict:
    """Async variant of run_vqe."""
    import asyncio
    return await asyncio.to_thread(self.run_vqe, options)

  async def run_qpe_async(self, options: dict | None = None) -> dict:
    """Async variant of run_qpe."""
    import asyncio
    return await asyncio.to_thread(self.run_qpe, options)

  async def run_qec_async(self, options: dict | None = None) -> dict:
    """Async variant of run_qec."""
    import asyncio
    return await asyncio.to_thread(self.run_qec, options)

  async def run_qfield_async(self, options: dict | None = None) -> dict:
    """Async variant of run_qfield."""
    import asyncio
    return await asyncio.to_thread(self.run_qfield, options)

  async def run_phi4_async(self, options: dict | None = None) -> dict:
    """Async variant of run_phi4."""
    import asyncio
    return await asyncio.to_thread(self.run_phi4, options)

  async def run_hubbard_async(self, options: dict | None = None) -> dict:
    """Async variant of run_hubbard."""
    import asyncio
    return await asyncio.to_thread(self.run_hubbard, options)

  async def run_floquet_async(self, options: dict | None = None) -> dict:
    """Async variant of run_floquet."""
    import asyncio
    return await asyncio.to_thread(self.run_floquet, options)

  async def run_heisenberg_async(self, options: dict | None = None) -> dict:
    """Async variant of run_heisenberg."""
    import asyncio
    return await asyncio.to_thread(self.run_heisenberg, options)

  async def run_bose_hubbard_async(self, options: dict | None = None) -> dict:
    """Async variant of run_bose_hubbard."""
    import asyncio
    return await asyncio.to_thread(self.run_bose_hubbard, options)

  async def run_tfi_async(self, options: dict | None = None) -> dict:
    """Async variant of run_tfi."""
    import asyncio
    return await asyncio.to_thread(self.run_tfi, options)

  async def run_ssh_async(self, options: dict | None = None) -> dict:
    """Async variant of run_ssh."""
    import asyncio
    return await asyncio.to_thread(self.run_ssh, options)

  async def run_kitaev_async(self, options: dict | None = None) -> dict:
    """Async variant of run_kitaev."""
    import asyncio
    return await asyncio.to_thread(self.run_kitaev, options)

  async def run_anderson_async(self, options: dict | None = None) -> dict:
    """Async variant of run_anderson."""
    import asyncio
    return await asyncio.to_thread(self.run_anderson, options)

  async def run_chern_async(self, options: dict | None = None) -> dict:
    """Async variant of run_chern."""
    import asyncio
    return await asyncio.to_thread(self.run_chern, options)

  async def run_hofstadter_async(self, options: dict | None = None) -> dict:
    """Async variant of run_hofstadter."""
    import asyncio
    return await asyncio.to_thread(self.run_hofstadter, options)

  async def run_mbl_async(self, options: dict | None = None) -> dict:
    """Async variant of run_mbl."""
    import asyncio
    return await asyncio.to_thread(self.run_mbl, options)

  async def run_spectral_async(self, options: dict | None = None) -> dict:
    """Async variant of run_spectral."""
    import asyncio
    return await asyncio.to_thread(self.run_spectral, options)

  async def run_tenfold_async(self, options: dict | None = None) -> dict:
    """Async variant of run_tenfold."""
    import asyncio
    return await asyncio.to_thread(self.run_tenfold, options)

  async def run_second_chern_async(self, options: dict | None = None) -> dict:
    """Async variant of run_second_chern."""
    import asyncio
    return await asyncio.to_thread(self.run_second_chern, options)

  async def run_wilson_loop_async(self, options: dict | None = None) -> dict:
    """Async variant of run_wilson_loop."""
    import asyncio
    return await asyncio.to_thread(self.run_wilson_loop, options)

  async def run_edge_states_async(self, options: dict | None = None) -> dict:
    """Async variant of run_edge_states."""
    import asyncio
    return await asyncio.to_thread(self.run_edge_states, options)

  async def run_haldane_async(self, options: dict | None = None) -> dict:
    """Async variant of run_haldane."""
    import asyncio
    return await asyncio.to_thread(self.run_haldane, options)

  async def run_weyl_async(self, options: dict | None = None) -> dict:
    """Async variant of run_weyl."""
    import asyncio
    return await asyncio.to_thread(self.run_weyl, options)

  async def run_fu_kane_async(self, options: dict | None = None) -> dict:
    """Async variant of run_fu_kane."""
    import asyncio
    return await asyncio.to_thread(self.run_fu_kane, options)

  async def run_lutchyn_oreg_async(self, options: dict | None = None) -> dict:
    """Async variant of run_lutchyn_oreg."""
    import asyncio
    return await asyncio.to_thread(self.run_lutchyn_oreg, options)

  async def run_kitaev_honeycomb_async(self, options: dict | None = None) -> dict:
    """Async variant of run_kitaev_honeycomb."""
    import asyncio
    return await asyncio.to_thread(self.run_kitaev_honeycomb, options)

  async def run_bbh_async(self, options: dict | None = None) -> dict:
    """Async variant of run_bbh."""
    import asyncio
    return await asyncio.to_thread(self.run_bbh, options)

  async def run_floquet_topo_async(self, options: dict | None = None) -> dict:
    """Async variant of run_floquet_topo."""
    import asyncio
    return await asyncio.to_thread(self.run_floquet_topo, options)

  async def run_non_herm_async(self, options: dict | None = None) -> dict:
    """Async variant of run_non_herm."""
    import asyncio
    return await asyncio.to_thread(self.run_non_herm, options)

  async def run_aah_async(self, options: dict | None = None) -> dict:
    """Async variant of run_aah."""
    import asyncio
    return await asyncio.to_thread(self.run_aah, options)

  async def run_syk_async(self, options: dict | None = None) -> dict:
    """Async variant of run_syk."""
    import asyncio
    return await asyncio.to_thread(self.run_syk, options)

  async def run_kondo_async(self, options: dict | None = None) -> dict:
    """Async variant of run_kondo."""
    import asyncio
    return await asyncio.to_thread(self.run_kondo, options)

  async def run_luttinger_async(self, options: dict | None = None) -> dict:
    """Async variant of run_luttinger."""
    import asyncio
    return await asyncio.to_thread(self.run_luttinger, options)

  async def run_toric_async(self, options: dict | None = None) -> dict:
    """Async variant of run_toric."""
    import asyncio
    return await asyncio.to_thread(self.run_toric, options)

  async def run_mipt_async(self, options: dict | None = None) -> dict:
    """Async variant of run_mipt."""
    import asyncio
    return await asyncio.to_thread(self.run_mipt, options)

  async def run_rg_async(self, options: dict | None = None) -> dict:
    """Async variant of run_rg."""
    import asyncio
    return await asyncio.to_thread(self.run_rg, options)

  async def run_entanglement_async(self, options: dict | None = None) -> dict:
    """Async variant of run_entanglement."""
    import asyncio
    return await asyncio.to_thread(self.run_entanglement, options)

  # ── Lindblad open-system solver (M71) ────────────────────────────────────

  def run_lindblad(
      self,
      n_qubits: int,
      hamiltonian_terms: list[tuple[float, str]],
      jump_operators: "list[JumpOperator]",
      total_time: float,
      dt: float = 0.01,
      entanglement_cut: int = 0,
  ) -> "LindbladResult":
      """Run the Lindblad master equation and return a :class:`~quantx_engine.lindblad.LindbladResult`.

      Convenience wrapper around :class:`~quantx_engine.lindblad.LindbladSolver`.

      Parameters
      ----------
      n_qubits:
          Number of qubits (max 10).
      hamiltonian_terms:
          Coherent Hamiltonian as ``[(coeff, "Z0Z1"), ...]``.
      jump_operators:
          List of :class:`~quantx_engine.lindblad.JumpOperator`.
      total_time:
          Total evolution time.
      dt:
          RK4 time step (default 0.01).
      entanglement_cut:
          Bipartite cut qubit index for entanglement tracking (0 = disabled).

      Returns
      -------
      LindbladResult
      """
      from quantx_engine.lindblad import LindbladSolver
      return LindbladSolver(self).solve(
          n_qubits=n_qubits,
          hamiltonian_terms=hamiltonian_terms,
          jump_operators=jump_operators,
          total_time=total_time,
          dt=dt,
          entanglement_cut=entanglement_cut,
      )

  async def run_lindblad_async(
      self,
      n_qubits: int,
      hamiltonian_terms: list[tuple[float, str]],
      jump_operators: "list[JumpOperator]",
      total_time: float,
      dt: float = 0.01,
      entanglement_cut: int = 0,
  ) -> "LindbladResult":
      """Async variant of run_lindblad."""
      import asyncio
      return await asyncio.to_thread(
          self.run_lindblad,
          n_qubits, hamiltonian_terms, jump_operators, total_time, dt, entanglement_cut,
      )

  @staticmethod
  def _default_library_path() -> str:
    """Return the best candidate path for the native QuantX shared library.

    Search order (first match wins):

    1. ``QUANTX_LIB_PATH`` environment variable — explicit user / CI override.
    2. Package directory — the wheel-bundled ``.so`` / ``.dll`` / ``.dylib``
       placed alongside ``__init__.py`` when ``pip install quantx-engine``
       installs a native wheel.
    3. Development build directories relative to the current working directory
       (``build/Release/``, ``build/Debug/``, ``build-vs/…``).

    Returns
    -------
    str
        Absolute path to the library (may not exist if none of the above was
        found; the caller handles the ``OSError`` from ``ctypes.CDLL``).
    """
    import sys as _sys

    # ── 1. Explicit environment variable ──────────────────────────────────────
    env_path = os.environ.get("QUANTX_LIB_PATH")
    if env_path:
      return env_path

    # ── 2. Wheel-bundled library in the package directory ─────────────────────
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if os.name == "nt":
      wheel_lib = os.path.join(pkg_dir, "quantx.dll")
    elif _sys.platform == "darwin":
      wheel_lib = os.path.join(pkg_dir, "libquantx.dylib")
    else:
      wheel_lib = os.path.join(pkg_dir, "libquantx.so")
    if os.path.exists(wheel_lib):
      return wheel_lib

    # ── 3. Development build directories ──────────────────────────────────────
    cwd = os.getcwd()
    if os.name == "nt":
      candidates = [
          os.path.join(cwd, "build", "Release", "quantx.dll"),
          os.path.join(cwd, "build", "Debug", "quantx.dll"),
          os.path.join(cwd, "build-vs", "Release", "quantx.dll"),
          os.path.join(cwd, "build-vs", "Debug", "quantx.dll"),
      ]
    elif _sys.platform == "darwin":
      candidates = [
          os.path.join(cwd, "build", "Release", "libquantx.dylib"),
          os.path.join(cwd, "build", "libquantx.dylib"),
          os.path.join(cwd, "build", "Debug", "libquantx.dylib"),
      ]
    else:
      candidates = [
          os.path.join(cwd, "build", "Release", "libquantx.so"),
          os.path.join(cwd, "build", "libquantx.so"),
          os.path.join(cwd, "build", "Debug", "libquantx.so"),
      ]
    for p in candidates:
      if os.path.exists(p):
        return p
    return candidates[0]  # best guess; ctypes will raise OSError if absent
