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

C ABI:

- `qx_run_circuit(circuit_json, options_json) -> result_json`

Python:

- `engine.run_circuit(circuit, options)`
- `engine.run_qiskit_like(request, options)` (minimal adapter payload)
- `engine.run_cirq_like(request, options)` (minimal adapter payload)

## 3) Result model

- final state (optional)
- shot counts (`{"bitstring": count}`)
- expectation values (`{"obs_name": value}`)
- runtime metadata (backend, elapsed time, method, precision)

## Integration principles

- backend-agnostic request model: frontends should not hardcode backend classes
- stable schema evolution: append-only fields for JSON interfaces
- deterministic test mode with fixed seeds
- clear error contracts for unsupported gates/noise/channels
