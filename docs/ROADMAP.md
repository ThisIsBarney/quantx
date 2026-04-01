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

Release execution policy:

- Split multi-round integration work into three explicit rounds when the surface area is large enough to risk cross-task interference.
- Keep each round in its own `codex/*` worktree on top of the latest `master` before editing.
- Merge a round only after its smoke, benchmark, and documentation checks are green.
- Rebase or recreate the round worktree when `master` advances instead of mixing stale and fresh branches.
- Remove the branch, worktree, and temporary stash entries after merge so the next round starts cleanly.

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

## Completed milestones (post-v1 physics track)

### M20: Lindblad open quantum systems
- Lindblad master equation solver; amplitude damping, dephasing, depolarizing, Rabi decay scenarios; von Neumann entropy and purity tracking.

### M21: Wigner quasi-probability and quantum transport
- Wigner function W(x,p) via spectral method; probability current density J; transmission/reflection coefficients; animated Wigner heatmaps.

### M22: Bipartite entanglement analysis
- Schmidt decomposition; von Neumann entropy S(ρ_A); mutual information; concurrence (2-qubit); logarithmic negativity; animated per-bond bond dimension charts.

### M23: Schwinger model (1+1D QED)
- Staggered-fermion lattice QED; chiral condensate; string tension; pair-creation dynamics; mass sweep.

### M24: φ⁴ Scalar QFT
- Truncated Fock basis; Z₂ symmetry breaking; phase-transition sweep; quantum quench dynamics.
- CSR sparse Hamiltonian + full-reorthogonalization Lanczos eigensolver; Arnoldi-Krylov time propagator.

### M25: Unified single-page dashboard
- All 30 scenarios aggregated in one `quantum_dashboard.html`; dark sidebar; search/keyboard nav; iframe srcdoc switching.

### M26: φ⁴ quantum quench dynamics + Lanczos
- Real-time quench dynamics (order parameter, bipartite entropy, per-site observables); stroboscopic heatmaps.

### M27: 1D Hubbard model + Floquet kicked-Ising
- **Hubbard**: fixed-(N↑,N↓) Fock basis; Jordan-Wigner CSR; Lanczos; double occupancy D, spin structure factor S(π), momentum distribution n(k); U/t Mott-insulator sweep.
- **Floquet**: stroboscopic Néel dynamics; exact diagonal H₀; single-qubit X kicks; Hessenberg QR quasi-energy spectrum; mean gap ratio r (Poisson/GUE); ergodic/MBL phase badge.

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
