# QuantX

QuantX is a tensor-network engine for quantum many-body physics. It runs block-sparse
MPS algorithms (DMRG, TDVP, and fermionic HVA circuits) that conserve U(1), U(1)×U(1)
and SU(2) charges exactly. The core is C++20, with optional CUDA acceleration, and it
has a Python API.

This page lists only what the engine does from Python and how each capability is
checked. Everything else in the package is marked experimental below.

## What is verified, and against what

| Capability | Python | Checked against |
|---|---|---|
| Ground states of the XXZ, J1-J2, spin-1 Heisenberg and Fermi-Hubbard chains, in a chosen symmetry sector | `dmrg.ground_state` with `models.XXZChain`, `J1J2Chain`, `Spin1Chain`, `HubbardChain` | exact diagonalisation, written independently of the engine, to 1e-10 |
| Hubbard ground states on ladders, strips and cylinders | `dmrg.ground_state(models.HubbardLattice(...))` | exact diagonalisation on a ladder and a cylinder, where the fermion signs are physical, to 1e-10 |
| Fermionic HVA (VQE) circuits on chains, ladders, strips and cylinders: energy, entanglement entropy, truncation | `hva.circuit`, `hva.layout` | an independent state-vector simulation: energy and entropies to 1e-10 from a product start (`init="product"`); to 1e-6 from the default U = 0 start, measured at about 1e-8 (see below) |
| The gradient of an HVA circuit's energy in every angle, by the adjoint method, at a cost that does not grow with the number of angles | `hva.circuit(..., gradient=True)` | the exact gradient of the same independent state vector, and central differences of the engine's own energy: to 1e-8 from a product start |
| Finite-temperature statics, the dynamical structure factor S(k, ω, T) and the optical conductivity σ(ω, T) of the XXZ, spin-1 and Hubbard chains | `thermal.statics`, `thermal.structure_factor(...).spectrum()`, `thermal.current_correlation(...).conductivity()` | exact diagonalisation of the whole Hilbert space, to the second-order Trotter error of the steps: below 1e-3 at the default step 0.05, and falling by 4 at half the step; the transforms against Lehmann sums to 1e-8 |

These checks run in CI against a fresh Linux build of the engine (`test_models_ed.py`,
`test_hva_statevector.py`, `test_hva_gradient.py` and `test_thermal_ed.py` in `python/quantx_engine/tests/`). The macOS and Windows
wheels are checked only for loading the engine.

## Install

```bash
pip install quantx-engine        # from 0.1.0
```

Documentation: **<https://thisisbarney.github.io/quantx/>** — including
[Results](https://thisisbarney.github.io/quantx/results.html), which groups the figures
by how strong a check is behind each (exact answers, published values, a convergence
study, demonstrations and a benchmark) and names the data file each was rendered from.

Wheels carry the native engine for Linux x86_64, macOS 10.15+ (arm64 and x86_64) and
Windows AMD64, on any CPython 3.10 or later. On other platforms pip installs a
pure-Python package without the engine, and `quantx_engine.backend_info()["native"]` is
`False`. The wheels are CPU-only; the macOS ones also run without OpenMP.

`quantx_engine.thermal` needs 0.2.0 or later.

## Quick start

```python
from quantx_engine.models import HubbardChain, XXZChain
from quantx_engine.dmrg import ground_state

gs = ground_state(HubbardChain(L=16, U=4.0), n_up=8, n_dn=8, max_bond=256)
gs.energy, gs.truncation_error, gs.sector

gs = ground_state(XXZChain(L=21, delta=0.5), sz=0.5)   # an odd chain, S^z = 1/2
```

```python
from quantx_engine.models import HubbardLattice
from quantx_engine import hva

cyl = HubbardLattice(Lx=4, W=3, U=4.0, y_periodic=True)
angles = [[0.1] * hva.layout(cyl).angles_per_layer] * 3      # 3 layers
r = hva.circuit(cyl, angles, n_up=6, n_dn=6, max_bond=256)   # ~10 s
r.energy, r.entropy_max, r.truncation_error, r.init_energy_error
# This cylinder already truncates at max_bond 256: truncation_error is about 5e-4, and
# the energy moves by 3e-3 at max_bond 1024. Raise max_bond until it stops moving.
```

```python
from quantx_engine.models import XXZChain
from quantx_engine import thermal

sf = thermal.structure_factor(XXZChain(L=12), beta=1.0, t_max=4.0)  # ~10 s on a desktop CPU
k, w, S = sf.spectrum()                     # S(k, omega, T), both signs of omega
# Converged at the default max_bond 128: the correlations move by 1e-5 at 256.
```

Finite temperature works by purification, in the grand-canonical ensemble. The Hubbard
chain needs a larger `max_bond` than the spin chains for the same times; see
[docs/finite_temperature.md](docs/finite_temperature.md).

[`examples/quickstart.py`](examples/quickstart.py) runs all three, a Hubbard ground
state, a finite-temperature spectral function and an HVA circuit, and asserts each
against an exact answer computed without the engine. CI runs it on every wheel.

**Conventions.**
- The spin models use physical spin operators S = σ/2 throughout.
- Every chain is open.
- Hubbard is H = −t Σ (c†c + h.c.) + U Σ n↑n↓ − μ Σ n.
- A ground-state or circuit solver works in one symmetry sector: total S^z, or the
  numbers of up and down electrons. Finite temperature is grand canonical, over every
  sector, with μ setting the Hubbard filling.
- An unreachable sector, or an unknown argument, is an error rather than a silent
  default.

**Reading the results.**
- `truncation_error` is the weight the bond-dimension cap discarded. For
  `ground_state` it is summed over the last sweep; for `hva.circuit` it is the largest
  single truncation.
- An HVA circuit started from the U = 0 ground state (`init="free"`, the default)
  inherits that state's DMRG error. `init_energy_error` is the start's energy against
  the exact U = 0 energy. A large value flags a bad start, but a small one does not
  bound the circuit's error: the energy is stationary at the ground state, so a start
  right to 1e-14 in energy can still carry ~1e-8 into the circuit. For the full
  precision, start from `init="product"`.

## More in the C++ engine, partly exposed

- **Real-time evolution** (TDVP and TEBD). From Python, only the XXZ chain is reachable:
  `tdvp.TDVPSolver` and `tdvp.LoschmidtSolver`, and the zero-temperature S(k, ω) and
  current correlator through `structure_factor.StructureFactorSolver` and
  `ConductivitySolver`. These keep the engine's Pauli convention, J σ·σ = 4 J S·S. See
  [docs/dmrg_tdvp_report.md](docs/dmrg_tdvp_report.md).
- **SU(2)-symmetric DMRG.** From Python, only the Heisenberg chain is reachable, through
  `dmrg.DMRGSolver(engine).solve(..., method="su2")`: J σ·σ for spin 1/2, and J S·S
  with `phys_spin=1`. See
  [docs/dmrg_su2_benchmark_report.md](docs/dmrg_su2_benchmark_report.md).
- **GPU acceleration** (cuSOLVER) of the block-sparse kernels, in source builds:
  [docs/gpu_acceleration_report.md](docs/gpu_acceleration_report.md).

The comparisons with TeNPy in those reports depend on hardware, bond dimension and build
options. Read them there, with their caveats, rather than as a single number;
[benchmarks/REPRODUCING.md](benchmarks/REPRODUCING.md) says how to rerun each one.

## Experimental

Everything else in the package is experimental, and lives in `quantx_engine.contrib`: its
API may change, and the checks above do not cover it. That is 63 modules, almost all pure
Python without the engine (VQE and quantum chemistry, quantum error correction and
decoders, noise models and error mitigation, quantum walks, bosonic codes, an IBM Quantum
connector, a PennyLane device, and more); `lindblad` and `cached_dmrg` are older
interfaces to parts of the engine. Import them from there, as in
`from quantx_engine.contrib.qec import SurfaceCode`. Their old paths
(`import quantx_engine.qec`, `from quantx_engine import SurfaceCode`) still work until
0.4.0, with a `DeprecationWarning`. The tutorial notebooks
in [`notebooks/`](notebooks/) use several of them, and run in CI. The two in
[`notebooks/engine/`](notebooks/engine/) run the engine itself against exact references:
Hubbard ground states against exact diagonalisation and the Lieb–Wu solution, and
finite-temperature spectra against exact diagonalisation.

The repository also has a REST server ([`server/`](server/)) and a command-line
interface (`qx_cli`).

## Building from source

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # -DQX_USE_CUDA=ON for the cuSOLVER path
cmake --build build --parallel
ctest --test-dir build
```

Point the Python package at the library with `QUANTX_LIB_PATH`, or build a wheel as in
[RELEASING.md](RELEASING.md).

## Roadmap and contributing

Where the project is going, phase by phase, is in [ROADMAP.md](ROADMAP.md).
Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers the development
setup and the standard every engine change is held to: a test against an independent
reference. Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

MIT, © 2026 Marshall WU. See [LICENSE](LICENSE).
