# Changelog

All notable changes to **quantx-engine** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html): `MAJOR.MINOR.PATCH`.
While the version is 0.x, a MINOR increment may change the API; PATCH increments are
bug-fixes and packaging corrections only.

---

## [Unreleased]

*(No changes yet)*

---

## [0.1.0] — unreleased  *(first public release, in preparation)*

The first version published to PyPI. The native engine is now installable; what it can
do from Python is gated against exact references.

Version numbers restart here, so the API stability guarantee of the internal 1.0.0
(below) does not carry over: as a 0.x release, any minor version may change the API.

### Added

- **Platform wheels that carry the native engine** (#194). There is one
  `py3-none-<platform>` wheel per platform, which serves every CPython 3: Linux x86_64
  (manylinux_2_28), macOS arm64 and x86_64 (10.15+), and Windows AMD64. Any other
  platform installs the sdist, a pure-Python wheel that says so.
- **`quantx_engine.models` and `dmrg.ground_state`** (#196): ground states of the XXZ,
  J1-J2, spin-1 Heisenberg and Fermi-Hubbard chains in a chosen symmetry sector on the
  block-sparse engine. They are gated against exact diagonalisation to 1e-10.
- **`models.HubbardLattice` and `quantx_engine.hva`** (#197): Hubbard ground states on
  ladders, strips and cylinders, and fermionic HVA circuits on the same lattices
  (`hva.circuit`, `hva.layout`), returning energy, entanglement entropies and
  truncation. The circuits are gated against an independent state vector: to 1e-10
  from a product start, and to 1e-6 from the default U = 0 start, which carries its
  DMRG preparation's residual (measured at about 1e-8). Lattice DMRG is gated against
  exact diagonalisation on the geometries where the fermion signs are physical.
- **The native engine these sit on**, built since the internal 1.0.0 (the git history
  has the details):
  - two-site DMRG on a block-sparse MPS that conserves U(1) or packed U(1)×U(1)
    charges, for any charge-conserving MPO, with a DMRG3S mixer;
  - SU(2)-symmetric DMRG for Heisenberg chains of any spin;
  - TDVP and TEBD;
  - finite-temperature statics and dynamics by purification;
  - Hubbard chains, ladders and cylinders, and fermionic HVA circuits;
  - a matrix-free Lindblad solver;
  - optional cuSOLVER acceleration in source builds.

  Only part of it is reachable from Python so far; the README lists which.

### Changed

- **The README and the PyPI page lead with the engine:** what it does from Python,
  and what each capability is checked against. Everything else in the package is
  marked experimental (M7). The package description now says the same thing.

### Fixed

- Shots on the native circuit backend were sampled from one random pass, so any
  randomness in the circuit made them unanimous.
  - **Measured circuits:** a measured Bell pair read `00` on all 1000 shots, as did
    every QASM program that ends in `measure`. When every measurement is terminal and
    no noise acts, the shots are now sampled from the state before measurement: the
    exact distribution, still from one pass. An observable with X or Y on a measured
    qubit then reads 0.
  - **Noise:** `|1>` under `DepolarizingNoise(0.5)` read all 0 or all 1, never the
    mixed state's 3/4. With noise, or a measurement that later gates depend on, every
    shot now runs its own pass. This costs one full pass per shot, and
    `expectation_exact` becomes the average over the passes.
  - With `shots = 0`, a noisy circuit still returns one random pass.
  - The classical register and the returned state come from one pass.
  - A noise channel at strength zero counts as no noise.
  - `max_shots` (1 000 000) is now enforced, not only advertised. A shot count must be
    a non-negative integer, and one past 2³² is refused: 2³² + 10 used to wrap to 10
    shots.
  - Amplitude damping runs in place, which makes the noisy per-shot path about three
    times faster (a 16-qubit GHZ circuit at 1000 shots: 47 s, from about 150 s).
- The native depolarizing channel applied a random Pauli with probability p, which is
  (1 − p)ρ + (p/3)(XρX + YρY + ZρZ). `DepolarizingNoise` documents, and the
  density-matrix simulator in `primitives` runs, (1 − p)ρ + p·I/2. The native backend
  now runs the documented channel too. Two readers of the parameter were also wrong:
  - `hardware.calibration_to_noise_model` passed a gate's average infidelity r as p.
    It now inverts the channel for the gate's arity: p = 2r for a one-qubit gate, and
    about 5r/6 for a two-qubit gate, where the channel acts on both qubits. The worst
    gate sets p.
  - `noise_char`'s process-tomography fit used the average-gate factor (d + 1)/d where
    the depolarizing parameter needs d²/(d² − 1). It read `DepolarizingNoise(0.1)`
    back as 0.1125.

  PEC's `from_depolarizing` keeps its Pauli-error parameter and now says how it maps:
  `DepolarizingNoise(p)` is `noise_prob = 3p/4`.
- The native circuit backend returned ⟨Y⟩ with the wrong sign, exact and sampled
  alike: `S H|0⟩ = |+i⟩` read −1. Every observable with an odd number of Y factors
  had its sign flipped, and so did any energy with such terms (a real Hamiltonian has
  none). The statevector, Lindblad and MPS code already used the right sign.
- The Python simulators read noise models on their own terms. The density-matrix
  simulator in `primitives` (behind `Estimator`, `Sampler` and `noise_vqe`):
  - put noise only on the qubits a gate acts on;
  - never applied `ReadoutNoise`'s bit flip;
  - took T1 decay from `noise_metadata`, and only when `amplitude_damping_gamma` was
    unset.

  The dynamic-circuit simulator treated amplitude damping as a depolarizing
  probability and dropped every other channel. Now all three simulators apply the
  same channels, resolved by the new `noise.native_channels`: every qubit after every
  gate, as on the native backend, each rate clamped to [0, 1] as the backend clamps it.
  For dynamic circuits, `DynamicResult.metadata` reports `noise_channels` in place of
  the old `p_noise`. The density-matrix simulator now applies gates by contracting over
  their qubits, and one-qubit channels block by block, instead of multiplying
  2^n x 2^n matrices: a two-layer, 8-qubit circuit with depolarizing and T1 noise
  takes 0.2 s, from 1.2 s, although every qubit now gets the noise.
- The block engine's lossless centre moves padded every bond they crossed with
  zero-weight states, so reported bond dimensions were inflated (a circuit capped at
  `max_bond` 4 reported 10). Energies and entropies were unaffected (#197).
- `CompositeNoise` no longer drops a channel silently (#198). `to_dict()` merged its
  models with `dict.update`, so a second `noise`-kind model replaced the first:
  `CompositeNoise([DepolarizingNoise(...), ThermalRelaxationNoise(...), ReadoutNoise(...)])`
  ran depolarizing + bit-flip with no T1 channel. Two models that emit the same key
  (`noise`, `depolarizing_prob`, `amplitude_damping_gamma`, `noise_metadata`, ...) now
  raise `NoiseModelError` when the composite is built, including through `+`. So does
  a custom model's `noise` selector of kind `depolarizing` or `amplitude_damping` that
  carries a value, next to `DepolarizingNoise` or `AmplitudeDampingNoise`: the keys
  differ, but the backend writes the selector's value into the same field.
  Depolarizing and amplitude damping still combine with one `noise`-kind model.
- `ReadoutNoise`'s docstring said its bit flip follows each measurement. The backend
  applies it to every qubit after every gate except measurements, so it is gate noise,
  not a readout error (#198).
- Eigen 3.4.0's BDCSVD can read out of bounds on some rank-deficient blocks. It crashed
  the DMRG's single-site splits, which since use JacobiSVD, and the HVA centre moves
  before #197. The wheels build without LAPACKE, so the two-site DMRG, the dense DMRG,
  TDVP and the SU(2) DMRG still ran it. The build now applies Eigen 3.4.1's two BDCSVD
  fixes to Eigen 3.4.0: that case reports a numerical issue, and the total-deflation
  test uses |col0|. The engine redoes any SVD that BDCSVD reports as failed with
  JacobiSVD, and a new test (`qx_robust_svd_test`) drives that path. The build stays on
  3.4.0, the version the engine was validated with, and takes only these two fixes from
  3.4.1.
- The DMRG C API failed on every odd-length Heisenberg chain, with an error that blamed
  `max_bond` (#196).
- An unknown model type in `qx_dmrg_sweep` silently ran as an empty Hamiltonian and
  returned E = 0; it is now an error (#196).
- At n = 2 the five DMRG and dynamics entry points of the C API returned an internal
  error from inside the block engine, which needs n ≥ 3. `qx_dmrg_sweep` and
  `qx_tdvp_evolve` now answer 2-site chains with their dense and non-block solvers;
  the other three refuse with a message that says why (#195).
- Four tutorial notebooks called APIs that do not exist, and printed claims their own
  outputs contradicted (#195).

### Known issues

- **The VQE showcase's bond dimensions** (`docs/universal_mpo.md` and the `chi` columns
  of `docs/vqe_*.csv`) were recorded before the #197 fix and overstate the true bond
  dimensions. Energies, entropies and truncation errors are unaffected.

---

## Development history before the first public release

The versions below were internal milestones of the project. None was published to PyPI,
and none has a release tag, apart from an old `v0.2.0` tag that corresponds to no
published package. They are kept as a record of what was built and when. Version
numbering restarts at 0.1.0 with the first public release.

### 1.0.0 (internal) — 2026-05-25  *(Phase 2 complete)*

#### Added

- **M167** — Benchmark suite: `benchmarks/compare_backends.py` covering Lanczos,
  Lindblad, DMRG, TEBD, VQE, QEC; JSON result artefacts; `benchmarks-ci.yml` workflow.
- **M168** — GPU wheels: `quantx-engine[gpu]` optional extra (CuPy-cuda12x runtime);
  `gpu-wheels.yml` CI with mock unit tests and optional self-hosted CUDA smoke test.
- **M169** — IBM Quantum real hardware round-trip: `IBMHardwareBackend.submit`,
  `HardwareJob.fetch`; `examples/ibm_hardware_demo.py`; 17 mock tests;
  `hardware-tests.yml` CI with gated real-hardware job (requires `QISKIT_IBM_TOKEN`).
- **M170** — `quantx_engine.ibm`: `circuit_to_qasm3`, QASM 3.0 serialisation
  compatible with Qiskit 1.x Primitives V2.
- **M171** — Community infrastructure: GitHub issue templates (bug, feature, question),
  PR template, `SECURITY.md` (private vulnerability reporting), `CODE_OF_CONDUCT_CN.md`
  (Contributor Covenant 2.1), expanded `CONTRIBUTING.md`, `docs/add_your_hamiltonian.md`
  contributor guide.
- **M172** — v1.0.0 release: version bump to 1.0.0, `release.yml` automated PyPI +
  GitHub Release workflow, API stability guarantee (see table below).

#### API stability guarantee (v1.0+)

The following symbols are **stable** and will not change without a deprecation cycle:

| Module | Stable symbols |
|--------|---------------|
| `quantx_engine` | `backend_info`, `Engine`, `LanczosSolver`, `MPSSimulator`, `CachedDMRG` |
| `quantx_engine.gpu` | `GpuBackend`, `GpuStatevector`, `gpu_info` |
| `quantx_engine.hardware` | `HardwareJob`, `HardwareJobResult`, `IBMHardwareBackend`, `HardwareError` |
| `quantx_engine.ibm` | `circuit_to_qasm3` |
| `quantx_engine.lindblad` | `LindbladSolver`, `JumpOperator`, `LindbladResult` |
| `quantx_engine.tensor_network` | `MPO` |
| `quantx_engine.pennylane_plugin` | `QuantXDevice` |

Internal modules (prefixed `_`) and modules noted as experimental are exempt.

---

### 0.4.0 (internal) — 2026-05-25  *(Phase 2 Track M — SDK Usability)*

#### Added

- **M162** — Test coverage hardening: skip rate reduced to < 1.4 % (49 / 3 699).
  Pure-Python Lindblad steady-state solver via Kronecker-product superoperator + SVD
  null-space; `JumpOperator(matrix)` matrix-form constructor; `GroverWalk` uniform
  initial superposition; `GKPState.norm()`, `CatQubit.energy_spectrum()`,
  `TeleportationResult.fidelity`, `QLDPCCode.n` alias, `MBQC.fidelity()`,
  `ZNEMitigator` alias, `RichardsonExtrapolation` class.
- **M163** — GitHub Pages documentation site: Sphinx + Furo theme, `sphinx-apidoc`
  auto-generated API stubs, `docs.yml` CI workflow, `sphinx_copybutton` extension,
  Docs badge in README.
- **M164** — Binder + Colab notebook compatibility: `notebooks/requirements.txt`,
  Colab/Binder launch badges in all 7 tutorial notebooks (01–07), pip-install guard
  cells, `scripts/add_notebook_badges.py` with dynamic branch detection,
  updated `notebooks-ci.yml`.
- **M165** — Interactive examples gallery: two new tutorial notebooks —
  `08_quantum_walk_algorithms.ipynb` (DTQW, CTQW, Grover walk, walk spectrum)
  and `09_noise_mitigation.ipynb` (ZNE, Richardson extrapolation, PEC, RB).
- **M166** — Versioned changelog + deprecation policy (this release):
  `CHANGELOG.md` updated, `quantx_engine._deprecation.warn()` helper,
  `@deprecated` decorator, version bumped to **0.4.0**.

#### Fixed

- `GroverWalk.run()`: AttributeError `_walk._walk_op` → `_walk._W` (Codex P1).
- `KrylovEvolver.run()`: off-by-one `n_steps+1` → `n_steps` snapshot count (Codex P2).
- `sphinx-build | tail -20` masked Sphinx exit code; replaced with explicit `if !` guard.
- `add_notebook_badges.py`: hardcoded `BRANCH = "master"` → dynamic `_default_branch()`.

#### Deprecated

- Nothing deprecated in this release.  The deprecation framework (`_deprecation.warn`)
  is in place for future cycles.

---

### 0.2.0 (internal) — 2026-05-20

#### Added
- `quantx_engine.__version__` attribute (`"0.2.0"`)
- `QuantXNativeUnavailable` exception — raised by `Engine.__init__` with a helpful message when
  the native C++ shared library cannot be loaded; all pure-Python modules remain functional
- `CVaRQAOAResult` top-level export — `qaoa_optim.QAOAResult` renamed in the public namespace
  to avoid clash with the earlier `advanced_chemistry.QAOAResult` (M79)
- **M127** packaging fixes: `pyproject.toml` optional-dependency groups `gpu`, `distributed`, `all`

#### Fixed
- `test_advanced_chemistry_importable_from_package` now passes — `quantx_engine.QAOAResult`
  correctly refers to `advanced_chemistry.QAOAResult` (M79), not `qaoa_optim.QAOAResult` (M115)
- All 16 Codex review findings on PRs #111–#118 fixed (bosonic codes, magic-state distillation,
  bicycle LDPC, McLachlan dynamics, QSP/QSVT, OO-VQE, MBQC, dynamical decoupling)

#### Physics modules added in this release cycle (M119–M126)
| Milestone | Module | Key classes |
|-----------|--------|-------------|
| M119 | `bosonic.py` | `CatQubit`, `GKPState`, `GKPDecoder`, `CatNoiseModel` |
| M120 | `ft_tgate.py` | `FTTGate`, `FTCCXGate`, `MagicStateFactory` |
| M121 | `qldpc.py` | `ClassicalCode`, `HypergraphProductCode`, `LDPCDecoder`, `bicycle_code` |
| M122 | `mclachlan.py` | `McLachlanSolver`, `McLachlanResult` |
| M123 | `qsp_qsvt.py` | `QSPAnglesFinder`, `QSVTCircuit` |
| M124 | `oovqe.py` | `OOVQESolver`, `OOVQEResult` |
| M125 | `mbqc.py` | `GraphState`, `MBQCPattern`, `MeasurementResult` |
| M126 | `dyndecouple.py` | `DynamicalDecoupling`, `QuantumMemory`, `DDSequence` |

---

### 0.1.0 (internal) — 2026-04-15

First internal milestone, covering M1–M118 (133 modules, 2000+ tests).

#### Track A — Core engine (M1–M20)
- `Engine` — ctypes bridge to native C++ quantum simulation runtime
- `CircuitError`, `ParseError`, OpenQASM 3.0 parser (`parse_qasm_to_circuit`)
- `NoiseModel`, `DepolarizingNoise`, `AmplitudeDampingNoise`, `ThermalRelaxationNoise`,
  `ReadoutNoise`, `CompositeNoise`
- `IBMQuantumBackend`, `circuit_to_qasm3`

#### Track B — Chemistry & VQE (M21–M50)
- `PauliTerm`, `PauliHamiltonian`, `jordan_wigner`, `MolecularVQE` (`chemistry.py`)
- `UCCSDSolver`, `ADAPTVQESolver`, `QAOASolver`, `QITESolver` (`advanced_chemistry.py`)
- `FidelityQuantumKernel`, `QSVM` (`quantum_kernel.py`)
- `QuantumNeuralNetwork`, `QuantumClassifier` (`qml.py`)
- `SPSAOptimizer`, `NoiseAwareVQE` (`noise_vqe.py`)
- `ExcitedStateVQE`, `detect_conical_intersections` (`excited_state.py`)

#### Track C — QEC & Fault tolerance (M51–M70)
- `RepetitionCode`, `SurfaceCode`, `MWPMDecoder`, `LogicalErrorSimulator` (`qec.py`)
- `LogicalQubit`, `MagicStateDistillery`, `FaultTolerantCircuit` (`fault_tolerant.py`)
- `GateSequence`, `SolovayKitaev`, `TCountOptimizer` (`clifford_t.py`)
- `StabilizerState`, `CliffordSimulator` (`clifford_sim.py`)
- `FTQCPipeline`, `threshold_curve` (`ftqc_pipeline.py`)

#### Track D — Simulation backends (M71–M90)
- `MPS`, `MPO`, `TEBD`, `DMRG` (`tensor_network.py`)
- `LanczosSolver`, `ArnoldiSolver` (`lanczos.py`)
- `MPSSimulator`, `CachedDMRG` (`mps_simulator.py`)
- `GpuStatevector`, `GpuLanczos` (`gpu.py`)
- `DistributedSVSimulator` via Ray (`ray_sim.py`)
- `LindbladSolver` (`lindblad.py`)
- `KrylovEvolver` (`spectral.py`)
- `TDMRGSolver` (`tdmrg.py`)
- `TrotterSimulator` (`trotter.py`)
- `SYKModel`, `compute_otoc` (`many_body.py`)

#### Track E–G — Algorithms & Protocols (M91–M118)
- Quantum walk algorithms: `DiscreteTimeWalk`, `ContinuousTimeQuantumWalk`, `GroverWalk`
- Quantum network: `QuantumRepeater`, `EntanglementSwapper`, `BellPairGenerator`
- Communication protocols: `BB84Protocol`, `E91Protocol`, `TeleportationProtocol`
- `BlochFloquetSolver`, `FloquetSystem` (`floquet.py`)
- `GRAPESolver`, `CRABSolver` (`optimal_control.py`)
- `CrossEntropyBenchmarking` (`xeb.py`)
- `StateTomography`, `ProcessTomography` (`tomography.py`)
- `CVaRQAOA`, `QAOALandscape` (`qaoa_optim.py`)
- `QuantXSamplerV2`, `QuantXEstimatorV2` (`qiskit_compat.py`)
- `CircuitWidget`, `BlochSphereWidget`, `JobMonitorWidget` (`jupyter_widgets.py`)
- `RydbergSimulator` (`rydberg.py`)
- Pulse-level simulation: `PulseSimulator`, `DRAGWaveform` (`pulse.py`)
- `NonEquilError`, `KeldyshSolver`, `KibbleZurekScaling` (`neqdyn.py`)
- `IBMHardwareBackend`, `IonQBackend`, `GoogleBackend` (`hardware.py`)
- `QuantXDevice` PennyLane plugin (`pennylane_plugin.py`)
- `DynamicCircuit`, `DynamicSimulator` (`dynamic.py`)
- `HardwareTranspiler` (`transpiler.py`)
- `ZNEResult`, `zne_mitigate`, `pec_mitigate` (`mitigation.py`)
- `FiniteSizeScaling`, `binder_cumulant` (`finite_size.py`)
- `SteadyStateSolver` (`quantum_channel.py`)
- `QuantumBoltzmannMachine`, `BoltzmannSampler` (`qbm.py`)
- `RBM` (`qbm.py`), `StandardRB`, `QuantumVolume` (`rb_qv.py`)
- `ParameterSweep`, `grid_sweep` (`sweep.py`)
- `CrosstalkModel`, `CouplingGraph` (`crosstalk.py`)
- `NoiseCharacterizer` (`noise_char.py`)
- `CircuitTemplate`, `hardware_efficient_ansatz`, `uccsd_ansatz` (`templates.py`)
- `to_cirq`, `from_cirq`, `to_braket`, `from_braket` (`adapters.py`)

---

[Unreleased]: https://github.com/ThisIsBarney/quantx/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ThisIsBarney/quantx/releases/tag/v0.1.0
