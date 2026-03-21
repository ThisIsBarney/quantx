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
- `engine.run_qiskit_like(request, options)` (minimal adapter payload)
- `engine.run_cirq_like(request, options)` (minimal adapter payload)
- `engine.circuit_capabilities()`

## 3) Capability query

The circuit backend exposes a stable, append-only capability object with:

- supported gates
- supported noise models
- `max_qubits`
- `max_shots`
- adapter support hints

The current native gate-model executor accepts:

- single-qubit gates: `X`, `Y`, `Z`, `H`, `S`, `T`, `RX`, `RY`, `RZ`
- two-qubit gates: `CNOT`/`CX`, `CZ`, `SWAP`

Supported noise model selectors:

- `none`
- `depolarizing`
- `amplitude_damping`

The advertised limits are practical execution limits for the current statevector path and are intended for planning, validation, and adapter routing.

## 4) Result model

- final state (optional)
- shot counts (`{"bitstring": count}`)
- expectation values (`{"obs_name": value}`)
- runtime metadata (backend, elapsed time, method, precision)

## 5) Error contract

Unsupported circuit inputs return machine-readable error payloads with:

- `ok: false`
- `error_code`
- `error_message`
- `error` as a human-readable compatibility alias

Examples:

- `QX_UNSUPPORTED_GATE`
- `QX_UNSUPPORTED_NOISE`
- `QX_INVALID_CIRCUIT`
- `QX_INVALID_QUBIT_INDEX`

## Integration principles

- backend-agnostic request model: frontends should not hardcode backend classes
- stable schema evolution: append-only fields for JSON interfaces
- deterministic test mode with fixed seeds
- clear error contracts for unsupported gates/noise/channels
