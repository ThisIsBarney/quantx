# quantx-engine

QuantX is a tensor-network engine for quantum many-body physics. It runs block-sparse
MPS algorithms (DMRG, TDVP, and fermionic HVA circuits) that conserve U(1), U(1)×U(1)
and SU(2) charges exactly. The core is C++20, with a Python API.

```bash
pip install quantx-engine
```

Wheels carry the native engine for Linux x86_64, macOS 10.15+ (arm64 and x86_64) and
Windows AMD64, on any CPython 3.10 or later. On other platforms pip installs a
pure-Python package without the engine, and `quantx_engine.backend_info()["native"]` is
`False`.

## What is verified

| Capability | Python | Checked against |
|---|---|---|
| Ground states of the XXZ, J1-J2, spin-1 Heisenberg and Fermi-Hubbard chains, in a chosen symmetry sector | `dmrg.ground_state` with `models.XXZChain`, `J1J2Chain`, `Spin1Chain`, `HubbardChain` | exact diagonalisation to 1e-10 |
| Hubbard on ladders, strips and cylinders | `dmrg.ground_state(models.HubbardLattice(...))` | exact diagonalisation on a ladder and a cylinder, to 1e-10 |
| Fermionic HVA circuits on chains, ladders, strips and cylinders | `hva.circuit`, `hva.layout` | an independent state vector: energy and entanglement entropies to 1e-10 from a product start; to 1e-6 from the default U = 0 start (about 1e-8 measured) |

```python
from quantx_engine.models import HubbardChain, HubbardLattice
from quantx_engine.dmrg import ground_state
from quantx_engine import hva

gs = ground_state(HubbardChain(L=16, U=4.0), n_up=8, n_dn=8, max_bond=256)
gs.energy, gs.truncation_error

cyl = HubbardLattice(Lx=4, W=3, U=4.0, y_periodic=True)
angles = [[0.1] * hva.layout(cyl).angles_per_layer] * 3
r = hva.circuit(cyl, angles, n_up=6, n_dn=6, max_bond=256)
r.energy, r.entropy_max, r.truncation_error, r.init_energy_error
# At max_bond 256 this cylinder already truncates (truncation_error ~5e-4; the energy
# moves by 3e-3 at 1024): raise max_bond until the energy stops moving.
```

Spin models use S = σ/2, and every chain is open. A solver works in one symmetry sector,
and an unreachable sector is an error rather than a silent default. The gates run on a
Linux build in CI; the macOS and Windows wheels are checked for loading the engine.

## Experimental

Everything else in the package is experimental: its API may change, and the checks above
do not cover it. That is more than 50 pure-Python modules that do not use the engine (VQE
and chemistry, error correction, noise and mitigation, quantum walks, bosonic codes,
hardware connectors and more), plus a few older interfaces to parts of the engine.

Documentation, benchmarks and the C++ engine are in the
[repository](https://github.com/ThisIsBarney/quantx). Its
[roadmap](https://github.com/ThisIsBarney/quantx/blob/master/ROADMAP.md) says where the
project is going, and the
[contributing guide](https://github.com/ThisIsBarney/quantx/blob/master/CONTRIBUTING.md)
says how to take part.

MIT License, © 2026 Marshall WU.
