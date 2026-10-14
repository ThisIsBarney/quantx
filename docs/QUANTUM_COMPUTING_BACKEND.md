# Quantum Computing Backend Target

This document defines what "QuantX as a core execution backend" means.

## Target capabilities

- accept circuit-level workloads (not only Hamiltonian workloads)
- execute gate-model programs on statevector/tensor-network backends
- return framework-friendly results (statevector, counts, expectation values, metadata)
- provide explainable backend routing and capability introspection

## Required interfaces

## 1) Circuit input model

Minimum v1.1 circuit schema:

- `qubits`
- ordered `operations`
- operation fields: `name`, `targets`, optional `controls`, optional `params`
- optional `measurements` and classical register mapping

## 2) Execution API

C++:

- `run_circuit(const QuantumCircuit&, const RunOptions&) -> CircuitResult`
- `circuit_backend_capabilities() -> CircuitCapabilities`

C ABI:

- `qx_run_circuit(circuit_json, options_json) -> result_json`
- `qx_circuit_capabilities() -> capability_json`

Python:

- `engine.run_circuit(circuit, options)`
- `engine.run_qiskit_like(request, options)` (Qiskit-style adapter; supports `parameter_map`, `clbits`, `MEASURE`)
- `engine.run_cirq_like(request, options)` (Cirq-style adapter; supports `parameter_map`, `clbits`, `MEASURE`/`M`)
- `engine.capabilities()` — preferred alias for `circuit_capabilities()`
- `engine.circuit_capabilities()`

## 3) Capability query

The circuit backend exposes a stable, append-only capability object with:

- supported gates
- supported noise models
- `max_qubits`
- `max_shots`
- adapter support hints

Read the capability object first, then route requests based on the returned limits and supported vocabulary.

The current native gate-model executor accepts:

- single-qubit gates: `X`, `Y`, `Z`, `H`, `S`, `T`, `RX`, `RY`, `RZ`
- two-qubit gates: `CNOT`/`CX`, `CZ`, `SWAP`
- measurement: `MEASURE` (collapses statevector; `targets[0]` = qubit, `controls[0]` = classical bit index)

Parametric binding: set `param_names` on an op and supply a `parameter_map` on the circuit to resolve angles at runtime.

Supported noise model selectors:

- `none`
- `depolarizing` — the channel (1 − p)ρ + p·I/2 with p = `prob`: a random Pauli kick with probability 3p/4
- `amplitude_damping` — T1-style decay with `gamma`
- `bit_flip` — X applied with `prob`
- `phase_flip` — Z applied with `prob`
- `t1_decay` — amplitude damping with `gamma = 1 - exp(-dt/T1)`; params: `T1`, `dt`
- `t2_decay` — pure dephasing with `p = (1-exp(-dt/T2))/2`; params: `T2`, `dt`

The advertised limits are practical execution limits for the current statevector path and are intended for planning, validation, and adapter routing.

### Capability query examples

C++:

```cpp
const auto caps = qx::circuit_backend_capabilities();
std::cout << qx::circuit_capabilities_to_json(caps) << std::endl;
```

C ABI:

```c
const char* caps_json = qx_circuit_capabilities();
/* caps_json contains backend, supported_gates, supported_noise_models, max_qubits, max_shots */
qx_free_string(caps_json);
```

Python:

```python
caps = engine.circuit_capabilities()
print(caps["backend"])
print(caps["supported_gates"])
print(caps["supported_noise_models"])
```

## 4) Result model

- `ok` — bool; false when an error_code is present
- `counts` — shot-based bitstring counts `{"bitstring": count}`
- `expectation_exact` / `expectation_shot` — observable expectation values
- `measurement_results` — classical register outcomes from MEASURE ops (clbit index order)
- `metadata` — structured block: `engine_version`, `backend`, `elapsed_ms`
- `engine_version`, `backend`, `elapsed_ms` — top-level aliases for backward compat
- `state_size` — number of amplitudes in the final statevector

See `schemas/result.schema.json` for the full append-only field contract.

## 5) Error contract

Unsupported circuit inputs return machine-readable error payloads with:

- `ok: false`
- `error_code`
- `error_message`
- `error` as a human-readable compatibility alias

Client code should branch on `error_code` and display `error_message` to users. The `error` alias is preserved for compatibility, but the contract is `code + message`.

Examples:

- `QX_UNSUPPORTED_GATE`
- `QX_UNSUPPORTED_NOISE`
- `QX_INVALID_CIRCUIT`
- `QX_INVALID_QUBIT_INDEX`

### Error contract examples

C++:

```cpp
try {
  (void)qx::run_circuit(bad_circuit, options);
} catch (const qx::CircuitError& ex) {
  std::cout << ex.code() << ": " << ex.what() << std::endl;
}
```

C ABI:

```c
const char* json = qx_run_circuit(circuit_json, options_json);
/* parse ok=false, error_code, error_message */
qx_free_string(json);
```

Python:

```python
result = engine.run_circuit({"qubits": 1, "operations": [{"name": "foo", "targets": [0]}]}, {"enable_checkpoint": False})
print(result["ok"])
print(result["error_code"])
print(result["error_message"])
```

Minimal command line reproduction:

```powershell
@'
{"qubits":1,"operations":[{"name":"foo","targets":[0]}]}
'@ | Set-Content $env:TEMP\qx_bad_circuit.json -Encoding utf8
.\build-vs\Debug\qx_cli.exe --circuit $env:TEMP\qx_bad_circuit.json --options .\examples\options_default.json
```

## Integration principles

- backend-agnostic request model: frontends should not hardcode backend classes
- stable schema evolution: append-only fields for JSON interfaces
- deterministic test mode with fixed seeds
- clear error contracts for unsupported gates/noise/channels
