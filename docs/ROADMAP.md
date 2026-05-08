# QuantX Delivery Roadmap

## Overview

QuantX is a high-performance quantum simulation platform with three delivery tracks:

| Track | Focus | Status |
|-------|-------|--------|
| **A — Core simulation** | Physics solvers, statevector, MPS, noise, circuit IR | ✅ Complete (M1–M53) |
| **B — Production backend** | REST API, SDK, auth, observability, deployment | ✅ Complete (M54–M108) |
| **C — Extreme performance** | GPU, large-scale Clifford, distributed statevector | ✅ Complete (M109–M111) |

All 111 milestones are complete as of 2026-05-06.

---

## Engineering policy

- Prefer license-friendly mature OSS for non-differentiating infrastructure.
- Each milestone lives on a `feature/mNNN-short-name` branch → implement → tests green → PR → Codex review → squash merge → delete branch.
- Benchmark and CI gates must be green before merge.

---

## Completed milestones

### M1–M5: Foundation

| Milestone | Delivered |
|-----------|-----------|
| **M1** | IR types (`QuantumProblem`, `RunOptions`, `OperatorGraph`, `EquationType`), C++ API freeze, C ABI, JSON schema |
| **M2** | CPU statevector evolution (OpenMP), Trotter/Krylov strategy slots, norm/unitarity validation |
| **M3** | MPS track — TEBD, truncation/bond-growth controls, entanglement observability |
| **M4** | MPI+GPU execution hooks, checkpoint/restore, communication telemetry |
| **M5** | Benchmark harness, QuTiP/QuEST/ITensor reference comparisons, v1 release candidate |

---

### M6–M19: Circuit execution & quantum computing backend

| Milestone | Delivered |
|-----------|-----------|
| **M6** | `QuantumCircuit` IR; standard gate mapping (X/Y/Z/H/S/T/Rx/Ry/Rz/CNOT/CZ/SWAP) to statevector |
| **M7** | Shot-based sampling, depolarizing/amplitude-damping/phase-damping noise channels, seeded sampling |
| **M8** | Adapter API for Qiskit/Cirq-style execution, result schema normalization, backend capability descriptor |
| **M9** | ML-assisted backend scheduler; learned policy with heuristic fallback; adaptive timestep for Hamiltonian evolution |
| **M10** | LLM-assisted workflow — NL-to-circuit templates, benchmark report drafting, experiment suggestion loop |
| **M11–M19** | Extended physics solvers: relativistic dynamics, 3D visualization, VQE, QPE, QEC primitives, topological phases, Floquet, many-body localization, lattice field theory |

---

### M20–M27: Physics track extensions (post-v1)

| Milestone | Delivered |
|-----------|-----------|
| **M20** | Lindblad master equation solver; amplitude damping, dephasing, depolarizing, Rabi decay; von Neumann entropy, purity |
| **M21** | Wigner quasi-probability — W(x,p) spectral method, probability current J, transmission/reflection, animated Wigner heatmaps |
| **M22** | Bipartite entanglement — Schmidt decomposition, von Neumann entropy S(ρ_A), mutual information, concurrence, logarithmic negativity |
| **M23** | Schwinger model (1+1D QED) — staggered-fermion lattice QED, chiral condensate, string tension, pair-creation dynamics |
| **M24** | φ⁴ scalar QFT — truncated Fock basis, Z₂ symmetry breaking, phase-transition sweep, quantum quench; CSR Lanczos eigensolver |
| **M25** | Unified single-page dashboard — 30 scenarios, `quantum_dashboard.html`, dark sidebar, search/keyboard nav |
| **M26** | φ⁴ quantum quench dynamics + Lanczos — real-time order parameter, bipartite entropy, stroboscopic heatmaps |
| **M27** | 1D Hubbard model (Jordan-Wigner CSR, Lanczos, Mott-insulator sweep) + Floquet kicked-Ising (quasi-energy spectrum, ergodic/MBL phase) |

---

### M28–M53: Advanced physics & algorithms

| Milestone | Delivered |
|-----------|-----------|
| **M28–M53** | Extended solver suite: quantum walks, Keldysh Green's functions, non-equilibrium dynamics, topological invariants, symmetry-protected phases, real-space RG, quantum chaos diagnostics, spectral functions, time-crystal detection, driven-dissipative phases, Holstein polaron, Bose-Hubbard superfluid/Mott, t-J model, frustrated magnetism, quantum billiards, random circuit sampling, quantum advantage benchmarks, error-rate scaling, quantum error correction codes |

---

### M54–M56: Python SDK & distribution

| Milestone | Delivered |
|-----------|-----------|
| **M54** | Python SDK — ctypes bridge to 37 C++ solvers, typed sync/async API, full test coverage |
| **M55** | FastAPI REST server — 37 solver endpoints, Docker image |
| **M56** | PyPI publish workflow — OIDC trusted publishing, automated releases |

---

### M57–M65: Ecosystem integrations & algorithms

| Milestone | Delivered |
|-----------|-----------|
| **M57** | OpenQASM 3.0 parser + transpiler → QuantX circuit JSON |
| **M58** | Noise model library — depolarizing, amplitude damping, thermal relaxation, readout error |
| **M59** | IBM Quantum connector — `circuit_to_qasm3`, `IBMQuantumBackend` |
| **M60** | Quantum chemistry — Jordan-Wigner transform, `MolecularVQE` (H₂, LiH, BeH₂) |
| **M61** | Quantum Machine Learning — `QuantumNeuralNetwork`, `QuantumClassifier` (parameter-shift gradients, Adam optimizer) |
| **M62** | QEC research toolkit — `RepetitionCode`, `SurfaceCode` (d=3+), `LogicalErrorSimulator`, `ThresholdResult` |
| **M63** | Tensor network methods — MPS, MPO, TEBD (Suzuki-Trotter), DMRG (two-site ground state) |
| **M64** | Quantum Boltzmann Machine — RBM, QBM with transverse-field quantum fluctuations, `BoltzmannSampler` |
| **M65** | Clifford+T fault-tolerant gate compilation — `GateSequence`, `CliffordGroup` (24 elements), `SolovayKitaev`, `TCountOptimizer` |

---

### M66–M80: Frameworks, benchmarking & advanced algorithms

| Milestone | Delivered |
|-----------|-----------|
| **M66** | PennyLane plugin — `QuantXDevice` (`quantx.statevector`), 18 gate kernels, wire-order-correct probabilities |
| **M67** | Lanczos/Arnoldi sparse eigensolver — `SparseHamiltonian`, `LanczosSolver` (re-orthogonalized), `ArnoldiSolver` (complex eigenpairs) |
| **M68** | Parallel parameter sweep — `ParameterSweep`, `grid_sweep`, `list_sweep` (ThreadPool/ProcessPool, cancel-on-timeout) |
| **M69** | GPU backend — `GpuBackend` (auto cupy/numpy), `GpuStatevector` (fp32/fp64), `GpuLanczos`, `GpuBatchedCircuit`, complex64 dtype |
| **M70** | Quantum error mitigation — ZNE (gate folding, Richardson/Poly/Exp extrapolators), PEC (`QuasiProbabilityDecomposition.from_depolarizing`), `zne_mitigate`, `pec_mitigate` |
| **M71** | Lindblad open-systems Python API — `LindbladSolver`, `LindbladResult`, `JumpOperator`, `LindbladFrame`, `EntanglementTrace` |
| **M72** | General `SparseHamiltonian` factory — `from_pauli_terms`, `heisenberg_1d/2d`, `bose_hubbard`, `hubbard_1d`, `xxz_chain` |
| **M73** | Spectral functions + Krylov time evolution — `SpectralResult`, `KrylovResult`, `KrylovEvolver` (Arnoldi-Krylov, imaginary-time) |
| **M74** | Finite-size scaling + quantum phase transitions — `FiniteSizeScaling`, `binder_cumulant`, `entanglement_scaling`, `ScalingResult` |
| **M75** | Quantum channel + steady-state solver — `QuantumChannel` (Kraus), `SteadyStateSolver` (Liouvillian), `process_fidelity` |
| **M76** | Hardware-aware circuit transpilation — `HardwareTranspiler` (SABRE routing), `CouplingMap`, native gate decomposition |
| **M77** | Google Cirq + Amazon Braket adapters — `to_cirq`/`from_cirq`, `to_braket`/`from_braket` |
| **M78** | QEC MWPM decoder — `MWPMDecoder` (minimum-weight perfect matching), `UnionFind` alternative |
| **M79** | Advanced quantum chemistry — `UCCSDSolver`, `ADAPTVQESolver`, `QAOASolver`, `QITESolver` |
| **M80** | Quantum kernel methods + QSVM — `FidelityQuantumKernel`, `ZZFeatureMap`, `QSVM`, target/Frobenius alignment |

---

### M81–M104: Benchmarking, advanced simulation & production services

| Milestone | Delivered |
|-----------|-----------|
| **M81** | Randomized benchmarking + quantum volume — `StandardRB` (EPC fit), `QuantumVolume` (heavy-output), XEB fidelity |
| **M82** | Cross-entropy benchmarking — `CrossEntropyBenchmarking`, `XEBResult` |
| **M83** | State and process tomography — `StateTomography`, `ProcessTomography` |
| **M84** | Quantum optimal control — `GRAPESolver` (gradient ascent), `CRABSolver` (random basis), `GRAPEResult`, `CRABResult` |
| **M85** | Estimator + Sampler primitives — `SparsePauliOp`, `Estimator`, `Sampler` (hardware-agnostic, 62 tests) |
| **M86** | Dynamic circuits — `DynamicCircuit` (mid-circuit measurement, `if_bit` feed-forward, reset, `repeat_until`), `DynamicSimulator` (43 tests) |
| **M87** | Noise characterization pipeline — `NoiseCharacterizer` (fit from RB/XEB/process tomography), `CharacterizedNoise` (56 tests) |
| **M88** | Circuit templates library — `hardware_efficient_ansatz`, `qaoa_circuit`, `ghz_circuit`, `quantum_fourier_transform`, `uccsd_ansatz`, `iqp_circuit` (84 tests) |
| **M89** | Pulse-level simulator — `PulseSchedule` (Gaussian/DRAG/FlatTop/Arbitrary waveforms), `PulseSimulator` (Schrödinger + Lindblad ODE via scipy), T1/T2 decay, DRAG suppression (72 tests) |
| **M90** | Crosstalk + ZZ coupling model — `CouplingGraph`, `CrosstalkModel`, ZZ Hamiltonian, `echo_sequence`, `zz_routing_cost`, `cheapest_path` (50 tests) |
| **M91** | Fault-tolerant logical qubits — `LogicalQubit` (repetition code, Pauli-frame tracking), `MagicStateDistillery` (15-to-1 cubic suppression), `FaultTolerantCircuit` (67 tests) |
| **M92** | Floquet engineering + Bloch-Floquet band structure — `FloquetSystem` (RK4 propagator), `BlochFloquetSolver` (quasi-energy spectrum, H_eff via matrix log, micromotion), `floquet_spectrum`, `floquet_effective_hamiltonian` (64 tests) |
| **M93** | Real hardware backend integration — `IBMHardwareBackend`, `IonQBackend`, `GoogleBackend`, `calibration_to_noise_model`, gate normalization, cirq joint bitstring decoder |
| **M94** | Time-dependent DMRG — `TDMRGSolver`, `real_time_evolve`, `imaginary_time_evolve`, entropy truncation |
| **M95** | Noise-aware VQE + SPSA — `NoiseAwareVQE`, `SPSAOptimizer`, Nelder-Mead |
| **M96** | Quantum network simulation — `QuantumNode`, `QuantumChannel`, `QuantumNetwork`, `BellPairGenerator`, `EntanglementSwapper`, `QuantumRepeater`, Werner state physics, DEJMPS purification (108 tests) |
| **M97** | Quantum communication protocols — `BB84Protocol` (Eve detection, Shor-Preskill rate), `E91Protocol` (optimal CHSH angles), `TeleportationProtocol`, `DenseCodingProtocol` (69 tests) |
| **M98** | REST API expansion — FastAPI endpoints for pulse simulator, fault-tolerant qubits, quantum network, communication protocols, noise-aware VQE |
| **M99** | GitHub Actions CI pipeline — matrix jobs Python 3.11/3.12/3.13 for SDK + server; `ci-gate` branch protection; `windows-benchmark-gate` |
| **M100** | Quantum walk algorithms — `DiscreteTimeWalk` (flip-flop shift, Hadamard/Grover/DFT coins), `ContinuousTimeQuantumWalk` (Arnoldi-Krylov), `GroverWalk` (oracle search, quadratic speedup) (72 tests) |
| **M101** | Suzuki-Trotter Hamiltonian simulation — `TrotterSimulator` (order 1/2/4 product formulas), `trotter_error_bound`, `optimal_steps`, `heisenberg_trotter`, `ising_trotter` (93 tests) |
| **M102** | Many-body physics enhancements — `compute_otoc` (quantum scrambling), `SYKModel`/`syk_hamiltonian` (Jordan-Wigner q=4 Majorana fermions), `dmrg_2d`/`SnakeMapper` (2D DMRG snake-path) |
| **M103** | Async task queue + result cache — `JobStore` (asyncio.Lock, TTL eviction), `ResultCache` (SHA-256 keyed, LRU, env-var config), `POST /jobs/submit`, `GET /jobs/{job_id}/status` (33 server tests) |
| **M104** | MPS simulator 50+ qubits — `AdaptiveTruncation` (eps-relative threshold), `MPSSimulator` (chi_l,d,chi_r tensors, GHZ n=50 fidelity=1.0), `CachedDMRG` (L-env caching), `mps_benchmark` (56 tests) |

---

### M105–M108: Production readiness

| Milestone | Delivered |
|-----------|-----------|
| **M105** | Persistent `JobStore` backends — `AbstractJobStoreBackend` ABC, `InMemoryBackend`, `SqliteBackend` (aiosqlite), `RedisBackend` (redis.asyncio); `QUANTX_JOB_STORE_URL` env var factory |
| **M106** | API authentication — `X-API-Key` header, JWT (HS256, configurable TTL), `POST /auth/token`, `GET /auth/whoami`; sliding-window rate limiter (`QUANTX_RATE_LIMIT_RPM`), HTTP 429 + `Retry-After` |
| **M107** | Observability — Prometheus metrics (`quantx_requests_total`, `quantx_request_duration_seconds`, cache/job gauges, `GET /metrics`), OpenTelemetry tracing (OTLP exporter, no-op by default) |
| **M108** | Deployment packaging — Docker Compose (`quantx-api`, Redis, Prometheus, Grafana), Helm chart (`Chart.yaml`, `values.yaml`, HPA), `GET /ready` Kubernetes readiness endpoint |

---

### M109–M111: Extreme performance track

| Milestone | Delivered |
|-----------|-----------|
| **M109** | GPU-accelerated MPS tensor networks — CuPy/cuTensorNet two-qubit SVD, `MPSSimulator(backend="cuda")`, CPU fallback, n=30 GPU vs CPU benchmark |
| **M110** | Large-scale Clifford simulation — Aaronson-Gottesman binary symplectic tableau, vectorized O(n) gate ops (H/S/CX), 10 000-qubit GHZ, `StabilizerState`, `CliffordSimulator`, `StimCircuit` import (70 tests, PR #101) |
| **M111** | Distributed statevector via Ray — `DistributedSVSimulator` (2^n amplitudes sharded across `num_shards=2^s` blocks, local/global gate dispatch, CPU fallback when Ray absent), `dist_sv_benchmark`, `is_ray_available` (54 tests, PR #102) |

---

## Future directions

The following tracks represent potential areas for future development:

### Track C — Frontier algorithms

- **M112**: Fault-tolerant QC end-to-end pipeline — Surface code encode → logical gate → syndrome extraction → MWPM decode → logical error rate vs p_phys threshold curve
- **M113**: Excited-state quantum chemistry — EOM-VQE for excitation energies, `ExcitedStateVQE`, PySCF interface, conical intersection detection
- **M114**: Non-equilibrium quantum dynamics — Keldysh Green's function, real-time many-body entanglement entropy, Kibble-Zurek scaling
- **M115**: Variational quantum optimization at scale — QAOA with CVaR objective, MaxCut/TSP/integer programming, parameter landscape visualization

### Track D — Ecosystem depth

- **M116**: Qiskit 1.x compatibility — `QuantXSamplerV2`/`QuantXEstimatorV2` implementing Primitives V2 interface
- **M117**: Jupyter interactive widgets — `CircuitWidget` (drag-and-drop), `BlochSphereWidget` (real-time), `JobMonitorWidget`; ipywidgets + anywidget
- **M118**: Neutral-atom backend — QuEra Aquila / Pasqal analog Hamiltonian, Rydberg blockade model, `RydbergSimulator`

---

## Release checklist

Before tagging any release:

```
.\scripts\test.ps1
.\scripts\benchmark_all.ps1 -NoiseSweep -RunGates -EnforceGate
```

- Inspect `benchmarks/reports/INDEX.md` and latest `*_TREND.md` files
- Run `.\scripts\run_visualization.ps1`
- Review `docs/DEPENDENCIES.md` before adding new third-party packages
- Confirm all CI jobs green on `master` (including `windows-benchmark-gate`)
