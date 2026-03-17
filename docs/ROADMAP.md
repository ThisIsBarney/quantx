# QuantX v1 Delivery Roadmap (12 Weeks)

Dual-track objective:

- Track A: high-performance quantum dynamics simulation
- Track B: become a reusable execution backend for quantum computing frameworks

## M1 (Week 1-2): Scaffold and API freeze

- finalize IR types (`QuantumProblem`, `RunOptions`, `OperatorGraph`, `EquationType`)
- lock C++ API, C ABI, and JSON schema fields
- baseline runtime/event model and error conventions

## M2 (Week 3-5): Statevector track

- CPU evolution path (OpenMP-ready)
- Trotter and Krylov strategy slots
- norm/unitarity validation metrics

## M3 (Week 6-8): MPS track

- TEBD-first implementation
- truncation/bond growth controls
- observability for entanglement and truncation error

## M4 (Week 9-10): Scale and resilience

- MPI+GPU execution plan hooks
- checkpoint/restore for long jobs
- baseline communication and recovery telemetry

## M5 (Week 11-12): Alignment and RC

- benchmark harness and reproducibility scripts
- compare reference results against QuTiP/QuEST/ITensor
- produce release candidate artifacts

## Next milestones (post-v1, quantum-computing backend track)

## M6: Circuit IR and gate execution

- add `QuantumCircuit` IR (gates, parameters, measurement ops)
- map standard gates (X, Y, Z, H, S, T, RX/RY/RZ, CNOT, CZ, SWAP) to statevector backend
- add circuit-to-Hamiltonian simulation bridge APIs

## M7: Sampling and noise

- support shot-based sampling outputs (counts and expectation estimates)
- add basic noise channels (depolarizing, amplitude damping, phase damping)
- add deterministic seeded sampling path for reproducible tests

## M8: Framework adapter layer

- add adapter API for external frameworks (Qiskit/Cirq-style execution request)
- normalize result schema for transpiler/runtime integration
- publish backend capability descriptor (supported gates, max qubits, noise support)
