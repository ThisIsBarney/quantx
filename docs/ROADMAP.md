# QuantX v1 Delivery Roadmap (12 Weeks)

Dual-track objective:

- Track A: high-performance quantum dynamics simulation
- Track B: become a reusable execution backend for quantum computing frameworks

Priority bands:

- P0: core simulator correctness and scalable execution backbone
- P1: quantum computing backend interoperability and production usability
- P2: ML/LLM-assisted optimization and workflow acceleration

Engineering policy:

- Prefer license-friendly mature OSS components for non-differentiating infrastructure (JSON/CLI/linear algebra/tooling).
- Avoid rebuilding generic infrastructure in-house unless there is a measured bottleneck or compatibility blocker.

## M1 (Week 1-2): Scaffold and API freeze

- finalize IR types (`QuantumProblem`, `RunOptions`, `OperatorGraph`, `EquationType`)
- lock C++ API, C ABI, and JSON schema fields
- baseline runtime/event model and error conventions

## M2 (Week 3-5): Statevector track

- CPU evolution path (OpenMP-ready)
- Trotter and Krylov strategy slots
- norm/unitarity validation metrics
- priority: P0

## M3 (Week 6-8): MPS track

- TEBD-first implementation
- truncation/bond growth controls
- observability for entanglement and truncation error
- priority: P0

## M4 (Week 9-10): Scale and resilience

- MPI+GPU execution plan hooks
- checkpoint/restore for long jobs
- baseline communication and recovery telemetry
- priority: P0

## M5 (Week 11-12): Alignment and RC

- benchmark harness and reproducibility scripts
- compare reference results against QuTiP/QuEST/ITensor
- produce release candidate artifacts
- priority: P0/P1

## v1.1-rc checklist

- run `.\scripts\test.ps1`
- run `.\scripts\benchmark_all.ps1 -NoiseSweep -RunGates -EnforceGate`
- inspect `benchmarks/reports/INDEX.md` and the latest `*_TREND.md` files
- generate visualization artifacts with `.\scripts\run_visualization.ps1`
- re-check `docs/DEPENDENCIES.md` before adding new third-party packages

## Next milestones (post-v1, quantum-computing backend track)

## M6: Circuit IR and gate execution

- add `QuantumCircuit` IR (gates, parameters, measurement ops)
- map standard gates (X, Y, Z, H, S, T, RX/RY/RZ, CNOT, CZ, SWAP) to statevector backend
- add circuit-to-Hamiltonian simulation bridge APIs
- priority: P1

## M7: Sampling and noise

- support shot-based sampling outputs (counts and expectation estimates)
- add basic noise channels (depolarizing, amplitude damping, phase damping)
- add deterministic seeded sampling path for reproducible tests
- priority: P1

## M8: Framework adapter layer

- add adapter API for external frameworks (Qiskit/Cirq-style execution request)
- normalize result schema for transpiler/runtime integration
- publish backend capability descriptor (supported gates, max qubits, noise support)
- priority: P1

## M9: ML-assisted scheduler and adaptive control

- collect runtime features and outcomes into a training dataset
- add learned backend chooser as optional policy (`heuristic` fallback stays default)
- add adaptive timestep policy recommendation for Hamiltonian evolution
- priority: P2

## M10: LLM-assisted workflow layer

- natural-language to `problem/circuit` template generation
- automated benchmark report drafting and anomaly summarization
- experiment suggestion loop with guardrails and reproducibility metadata
- priority: P2
