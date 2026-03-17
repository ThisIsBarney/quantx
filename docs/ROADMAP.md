# QuantX v1 Delivery Roadmap (12 Weeks)

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
