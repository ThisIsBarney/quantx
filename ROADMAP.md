# QuantX roadmap

*Last updated 2026-10-18.* The work is organised in phases, each closed by a gate whose
condition can be checked. Dates are targets. The status of each item is kept current
here as work merges.

**Status:** ✅ done · 🔄 in progress · ⬜ not started

Two items of the internal plan are not engineering work and are not listed here: a market
study (M10) and a commercial pilot (M22). That is why the numbering skips them.

## Where QuantX is going

QuantX is a symmetry-conserving many-body engine: block-sparse MPS algorithms that
conserve U(1), U(1)×U(1) and SU(2) charges exactly. It aims to fill two roles:

1. **An engine that others embed.** It should be usable from Python directly, and
   later as a simulator backend for platforms such as CUDA-Q and PennyLane, for
   problems where the symmetry makes the difference.
2. **A classical referee.** It should produce careful classical baselines for
   symmetric problems that are put forward as quantum-advantage candidates, or show
   clearly why a baseline is out of reach.

What we will not do is listed at the end.

## Phase 1 (2026-10 to 2026-12): expose what the engine already does

The engine has run DMRG, TDVP, finite-temperature purification and fermionic circuits
for a while, mostly from C++. This phase writes little new physics. It puts the
verified capabilities in users' hands.

### Phase 1A: the open-source release, 0.1.0 (gate G0, target 2026-11-06)

| | Item | Done when | Status |
|---|---|---|---|
| R1 | Release artefacts | Wheels and sdist; one version number (0.1.0); MIT license; changelog; a release workflow that refuses a tag that does not match the version (#194, #199) | ✅ |
| R2 | A clear product | The README and the PyPI page describe only verified engine capabilities (#200). The documentation site says the same, and the API reference separates the engine from the experimental modules (#203). | ✅ |
| R3 | Contributing | A contributing guide that matches how the project works, an English code of conduct, issue and PR templates, code owners | ✅ |
| R4 | A public roadmap | This file | ✅ |
| R5 | No known serious bugs | The full Python suite, the C++ tests and the notebooks pass in the configuration the wheels use. Known problems are fixed (#204–#207), or recorded (the 0.1.0 known issues in the changelog, #208). | ✅ |
| R6 | Open-source hygiene | Third-party notices shipped with the wheels; no secrets in the history; no local paths in the docs (#202) | ✅ |
| **G0** | **Gate: open source** | The repository is public, CI is green on `master`, and 0.1.0 is on PyPI with `pip install` giving `backend_info()["native"] == True` on Linux, macOS and Windows. Released 2026-10-16: [PyPI](https://pypi.org/project/quantx-engine/0.1.0/), [release notes](https://github.com/ThisIsBarney/quantx/releases/tag/v0.1.0) | ✅ |

### Phase 1B: gate G1

| | Item | Done when | Status |
|---|---|---|---|
| M1 | Release pipeline | Platform wheels carry the native engine (#194) | ✅ |
| M2 | Engineering health | `run_dmrg_blocks` no longer crashes at n = 2; the CI configurations are fixed (#195) | ✅ |
| M3 | Gradient-variance study | Barren plateaus versus width, published with its caveats (#193) | ✅ |
| M4 | General models from Python | XXZ, J1-J2, spin-1 and Hubbard ground states from Python, against exact diagonalisation to 1e-10 (#196) | ✅ |
| M5 | Fermionic circuits from Python | HVA circuits on chains, ladders, strips and cylinders, against a state vector (#197) | ✅ |
| M6 | Finite-temperature spectroscopy from Python | S(k, ω, T) and σ(ω, T) for the M4 models, against exact diagonalisation. Done for the XXZ, spin-1 and Hubbard chains, against a full-space ED (#215); J1-J2, whose next-nearest coupling has no purified form in the engine yet, is left for later | ✅ |
| M7 | A smaller product surface | The README describes the engine (#200); the pure-Python modules move to `quantx_engine.contrib`, marked experimental (planned for 0.2). The 63 modules outside the engine API moved, and `import quantx_engine` loads the engine alone; the old paths warn until 0.4.0 | ✅ |
| M8 | Referee feasibility | A one-page assessment of a classical baseline for SU(2) lattice gauge-theory dynamics (60 sites, conserved charge): the bond dimension and error needed, against the published tensor-network baseline. The one-pager is in, [docs/referee_su2_hadron.md](docs/referee_su2_hadron.md) (#217): the instance is already matched classically (an MPS at χ = 800, under an hour on a laptop), and it is the tracker's only one with a conserved U(1) charge. Measured since: the physical observable needs χ = 1600, where n_f(20) = 0.12 ± 0.04 (fidelity estimate 0.995, the error bar the change from χ = 800); every published classical value lies 0.13 to 0.29 above it (`benchmarks/su2_hadron_quimb.py`) | ✅ |
| M9 | Credibility material | Reproduction scripts for the TeNPy comparisons; notebooks for Hubbard and finite-temperature spectra, each reproducing its exact reference. The scripts run anywhere, each with a quick check ([benchmarks/REPRODUCING.md](benchmarks/REPRODUCING.md), #220); the notebooks check the engine against exact diagonalisation and the Lieb–Wu solution ([notebooks/engine/](notebooks/engine/), #221) | ✅ |
| **G1** | **Gate: installable and usable** | After `pip install`, 20 lines of Python give a Hubbard ground state, a finite-temperature spectral function and an HVA circuit, each matching its exact reference. [examples/quickstart.py](examples/quickstart.py) does it, and CI runs it on every wheel it builds (#223). Released 2026-10-18 in 0.2.0 ([PyPI](https://pypi.org/project/quantx-engine/0.2.0/), [release notes](https://github.com/ThisIsBarney/quantx/releases/tag/v0.2.0)): from a fresh `pip install`, the quick start passes in 2.4 s | ✅ |

## Phase 2 (2027-01 to 2027-06): from one-shot solvers to a composable library

| | Item | Done when | Status |
|---|---|---|---|
| M11 | MPS and MPO as Python objects | One notebook chains DMRG, applying an operator, TDVP and correlation functions, never passing through JSON. In progress: `quantx_engine.mps` keeps states in the engine under handles (ground states, one-site diagonal operators, TDVP, matrix elements), gated against exact diagonalisation; next, the notebook, then operators that change the charge and MPOs | 🔄 |
| M12 | cuTENSOR block-sparse backend | At χ = 256 no slower than TeNPy; at χ = 1024 the current lead is kept; the CPU path stays | ⬜ |
| M13 | Adjoint gradients | The full gradient of an HVA circuit costs the same regardless of the number of parameters, and matches central differences to 1e-8 | ⬜ |
| M14 | CUDA-Q backend | An NVQIR plugin (`--target quantx`), starting with particle-number-conserving circuits; CUDA-Q examples match a state vector | ⬜ |
| M15 | PennyLane device on the engine | The `quantx.statevector` device runs on the native engine, plus an MPS device; PennyLane's own examples run | ⬜ |
| M16 | First classical-referee submission | A classical result and its run time submitted for a symmetric quantum-advantage candidate, and listed publicly. First target, after M8: the SU(2) hadron instance, with the charge conserved exactly and an error bar on the physical observable. In progress: `quantx_engine.u1_circuit` runs the instance's circuits on the block-sparse engine with the charge exact; next, the instance itself at the bond dimensions M8 needed, against the quimb trajectory | 🔄 |
| **G2** | **Gate: part of the ecosystem** | No slower than TeNPy at small χ; CUDA-Q can select QuantX; one public referee submission | ⬜ |

## Phase 3 (2027-07 to 2028-03): the symmetric engine as infrastructure

| | Item | Done when | Status |
|---|---|---|---|
| M17 | Stable C ABI | Symmetric block-sparse tensors and the core algorithms (DMRG, TDVP, purification, circuits) as a separately versioned library with a stable ABI | ⬜ |
| M18 | Infinite systems | iDMRG, then VUMPS: the infinite spin-1/2 Heisenberg chain's energy matches the Bethe ansatz to 1e-8 | ⬜ |
| M19 | An external library on QuantX | A tensor-network library (quimb or TeNPy are the candidates) merges an adapter that uses QuantX as a backend | ⬜ |
| M20 | General non-abelian symmetry | SU(2) beyond the Heisenberg chain (the spin SU(2) of Hubbard, general MPOs), matching the U(1)×U(1) path and faster at equal accuracy | ⬜ |
| M21 | Routine refereeing | Every new observable-estimation candidate gets a classical baseline, or a public explanation of why there is none, within a month | ⬜ |
| **G3** | **Gate: used by others** | At least one external project calls QuantX from its main branch | ⬜ |

## What we will not do

- **A quantum SDK or development platform.** CUDA-Q and Qiskit fill that layer;
  QuantX aims to be a backend for them.
- **Our own GPU kernels for block-sparse contraction.** Those go to cuTENSOR.
- **New features in the pure-Python modules.** Error correction, quantum networks,
  bosonic codes, MBQC, Clifford simulation and similar modules are frozen in
  `quantx_engine.contrib`, with minimal maintenance.
- **A quantum-chemistry DMRG stack.** block2 is the standard; at most an FCIDUMP
  reader and cross-checks against block2.
- **PEPS, for now.** quimb and tenax have it, and the referee targets do not need it.
- **Solver count as a goal.**

## Influencing the roadmap

Open an issue to propose an item or to argue for a different order. The
[contributing guide](CONTRIBUTING.md) describes how work gets in. The earlier roadmap
(a 172-milestone list kept in `docs/ROADMAP.md` until 2026-10) is in the git history.
