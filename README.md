# QuantX Engine

QuantX is a C++20 quantum simulation engine with 37 physics solvers, a Python SDK, a REST API server, and cloud-hardware connectors.

[![CI](https://github.com/ThisIsBarney/quantx/actions/workflows/benchmark-gate.yml/badge.svg)](https://github.com/ThisIsBarney/quantx/actions/workflows/benchmark-gate.yml)
[![PyPI](https://img.shields.io/pypi/v/quantx-engine)](https://pypi.org/project/quantx-engine/)
[![Docs](https://github.com/ThisIsBarney/quantx/actions/workflows/docs.yml/badge.svg)](https://thisIsBarney.github.io/quantx/)

## Features

| Layer | What's included |
|---|---|
| **C++ Core** | Statevector, MPS, continuous-space (1D/2D/3D) and field-theory backends; Trotter + Arnoldi-Krylov evolution; noise channels |
| **37 Solvers** | VQE, QPE, QEC, Schwinger QFT, φ⁴, Hubbard, Floquet, Heisenberg, Bose-Hubbard, TFI, SSH, Kitaev chain, Anderson, Chern insulator, Hofstadter, MBL, spectral function, ten-fold way classifier, 2nd Chern, Wilson loop, edge states, Haldane, Weyl, Fu-Kane, Lutchyn-Oreg, Kitaev honeycomb, BBH, Floquet topo, non-Hermitian, AAH, SYK, Kondo, Luttinger, toric code, MIPT, RG flow, entanglement |
| **Python SDK** | `pip install quantx-engine`; ctypes bridge; 37 typed sync + async solver methods; OpenQASM 3.0 parser; noise model library; IBM Quantum connector; quantum chemistry (Jordan-Wigner + VQE) |
| **REST API** | FastAPI server with 37 solver endpoints; Docker image; OpenAPI docs at `/docs` |
| **Hardware** | IBM Quantum connector — circuit transpiler → QASM 3.0 → Qiskit Runtime Sampler |
| **Chemistry** | Jordan-Wigner transform; pre-built H₂, LiH, BeH₂ Hamiltonians; `MolecularVQE.to_vqe_options()` |
| **Visualization** | 1D SVG animations, 2D/3D heatmaps, standalone Three.js 3D point-cloud player |

## Quick Start

### Build (Windows / Visual Studio 2022)

```powershell
.\scripts\build.ps1
.\scripts\test.ps1
```

### Python SDK

```bash
pip install quantx-engine
```

```python
from quantx_engine import Engine

engine = Engine()
result = engine.run_vqe({
    "cost_hamiltonian": [
        {"label": "Z0", "coefficient": -0.5},
        {"label": "Z1", "coefficient": -0.5},
        {"label": "Z0Z1", "coefficient": 0.25},
    ],
    "ansatz_type": "uccsd",
    "max_iterations": 200,
})
```

### Quantum Chemistry

```python
from quantx_engine.chemistry import MolecularVQE

problem = MolecularVQE.h2(bond_length=0.735)
opts = problem.to_vqe_options(ansatz="uccsd")
result = engine.run_vqe(opts)
print(f"Ground state energy: {result['energy']:.6f} Ha (FCI ref: {problem.fci_energy:.6f} Ha)")
```

### OpenQASM 3.0

```python
from quantx_engine.qasm import parse_qasm_to_circuit

circuit = parse_qasm_to_circuit("""
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
""")
result = engine.run_circuit(circuit)
```

### Noise Models

```python
from quantx_engine.noise import DepolarizingNoise, ThermalRelaxationNoise, CompositeNoise

# Depolarizing combines with at most one `noise`-kind model (ThermalRelaxationNoise
# or ReadoutNoise): the native backend runs one per circuit, so a second raises.
model = CompositeNoise([
    DepolarizingNoise(prob=0.01),
    ThermalRelaxationNoise(T1=50e-6, T2=30e-6, t_gate=50e-9),
])
circuit.update(model.to_dict())
result = engine.run_circuit(circuit)
```

### REST API Server

```bash
# Docker
docker run -p 8000:8000 quantx-engine-server

# Or directly
uvicorn server.main:app --port 8000
```

```bash
# Run a named solver — body is the solver options JSON directly
curl -X POST http://localhost:8000/solve/vqe \
  -H 'Content-Type: application/json' \
  -d '{"cost_hamiltonian": [{"label": "Z0", "coefficient": -0.5}], "ansatz_type": "uccsd"}'
```

## Notebooks

Interactive tutorials — run instantly in Colab or Binder (no local install needed):

| Notebook | Topic | Colab | Binder |
|----------|-------|-------|--------|
| [01 — Getting Started](notebooks/01_getting_started.ipynb) | Lanczos, Trotter, noise channels | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThisIsBarney/quantx/blob/master/notebooks/01_getting_started.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ThisIsBarney/quantx/master?filepath=notebooks/01_getting_started.ipynb) |
| [02 — VQE Chemistry](notebooks/02_vqe_chemistry.ipynb) | UCCSD-VQE, noise-aware VQE, CVaR-QAOA | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThisIsBarney/quantx/blob/master/notebooks/02_vqe_chemistry.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ThisIsBarney/quantx/master?filepath=notebooks/02_vqe_chemistry.ipynb) |
| [03 — Quantum Error Correction](notebooks/03_quantum_error_correction.ipynb) | RepetitionCode, SurfaceCode, MWPM | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThisIsBarney/quantx/blob/master/notebooks/03_quantum_error_correction.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ThisIsBarney/quantx/master?filepath=notebooks/03_quantum_error_correction.ipynb) |
| [04 — Quantum Networking](notebooks/04_quantum_networking.ipynb) | BB84, teleportation, entanglement swapping | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThisIsBarney/quantx/blob/master/notebooks/04_quantum_networking.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ThisIsBarney/quantx/master?filepath=notebooks/04_quantum_networking.ipynb) |
| [05 — Many-Body MPS](notebooks/05_many_body_mps.ipynb) | DMRG, TEBD, MPS entanglement | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThisIsBarney/quantx/blob/master/notebooks/05_many_body_mps.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ThisIsBarney/quantx/master?filepath=notebooks/05_many_body_mps.ipynb) |
| [06 — QEC Research](notebooks/06_qec_research.ipynb) | QLDPC codes, bicycle codes, BP decoder | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThisIsBarney/quantx/blob/master/notebooks/06_qec_research.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ThisIsBarney/quantx/master?filepath=notebooks/06_qec_research.ipynb) |
| [07 — Bosonic QEC](notebooks/07_bosonic_qec.ipynb) | Cat qubits, GKP codes, Wigner function | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThisIsBarney/quantx/blob/master/notebooks/07_bosonic_qec.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ThisIsBarney/quantx/master?filepath=notebooks/07_bosonic_qec.ipynb) |
| [08 — Quantum Walk Algorithms](notebooks/08_quantum_walk_algorithms.ipynb) | Discrete/continuous walks, Grover search, walk spectrum | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThisIsBarney/quantx/blob/master/notebooks/08_quantum_walk_algorithms.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ThisIsBarney/quantx/master?filepath=notebooks/08_quantum_walk_algorithms.ipynb) |
| [09 — Noise Mitigation](notebooks/09_noise_mitigation.ipynb) | ZNE, Richardson extrapolation, PEC, readout mitigation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThisIsBarney/quantx/blob/master/notebooks/09_noise_mitigation.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ThisIsBarney/quantx/master?filepath=notebooks/09_noise_mitigation.ipynb) |

## CLI

```powershell
.\build-vs\Debug\qx_cli.exe --vqe .\examples\vqe_ising.json --options .\examples\options_default.json
.\build-vs\Debug\qx_cli.exe --circuit .\examples\circuit_bell.json --options .\examples\options_default.json
```

## Repository Layout

```
include/qx/       Public C++ headers + C ABI (qx_run_solver, qx_list_solvers)
src/              Core solvers, backends, noise channels, JSON utilities
python/           Python SDK (quantx_engine package)
  quantx_engine/
    engine.py     ctypes bridge — 37 sync/async solver methods
    qasm.py       OpenQASM 3.0 parser → QuantX circuit JSON
    noise.py      Noise model library
    ibm.py        IBM Quantum connector
    chemistry.py  Jordan-Wigner transform + molecular VQE
server/           FastAPI REST server (37 endpoints, Docker)
schemas/          JSON input/output schemas
tests/            C++ smoke + integration tests
scripts/          Build, test, benchmark, visualization scripts
docs/             Architecture, usage cookbook, roadmap
```

## Architecture

- **IR**: `QuantumProblem`, `RunOptions`, `OperatorGraph`, `EquationType`
- **Scheduler**: backend auto-selection with explanation reasons
- **Backend**: unified interface `initialize / evolve / measure / checkpoint / restore`
- **C ABI**: `qx_run_solver(solver, opts_json)` — single entry point for all 37 solvers
- **Python**: ctypes bridge with typed methods + async variants for all solvers

## Benchmarks & CI

Every push runs the full benchmark gate on Windows/Visual Studio 2022:

```powershell
.\scripts\benchmark_all.ps1 -NoiseSweep -RunGates -EnforceGate -SkipBuild
```

Reports are uploaded as CI artifacts. See `benchmarks/reports/INDEX.md` for trends.

## Notes

- CUDA/cuFFT acceleration: `-DQX_ENABLE_CUDA=ON` (requires CUDA Toolkit)
- MPI support: prepared at CMake level; full HPC kernels in roadmap
- OpenQASM parser supports 30+ gate types including parametric gates (`rx`, `ry`, `rz`, `p`, `crz`, etc.) and classical control flow
- IBM connector transpiles QuantX circuits → QASM 3.0 → Qiskit ISA circuits for hardware execution
- Noise backend: depolarizing, amplitude damping, thermal relaxation (T1/T2), readout error, bit-flip, phase-flip

## License

MIT
