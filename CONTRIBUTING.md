# Contributing to QuantX

Thank you for considering a contribution. QuantX is a block-sparse tensor-network engine
for quantum many-body physics (see the [README](README.md)); this guide covers how to set
up a development environment, the standard every engine change is held to, and how a
pull request gets merged.

Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md)
([中文版](CODE_OF_CONDUCT_CN.md)). Report security problems privately, as described in
[SECURITY.md](SECURITY.md), never in a public issue.

## Contents

1. [Where to start](#where-to-start)
2. [The verification standard](#the-verification-standard)
3. [Development environment](#development-environment)
4. [Building the native engine](#building-the-native-engine)
5. [Running the tests](#running-the-tests)
6. [Pull requests](#pull-requests)
7. [Coding conventions](#coding-conventions)
8. [Licensing of contributions](#licensing-of-contributions)

## Where to start

- **Bugs, questions and feature requests** go to
  [GitHub Issues](https://github.com/ThisIsBarney/quantx/issues), using the templates.
  Search first; for a bug, include `quantx_engine.__version__` and the output of
  `quantx_engine.backend_info()`, which says whether the native engine was loaded.
- **What the project is working towards** is in [ROADMAP.md](ROADMAP.md). Issues
  labelled `good first issue` or `help wanted` are open for anyone.
- **Large changes** (a new solver, a new public API, a change to the C API): open an
  issue first, so the design can be agreed before you write the code.

The package has two parts, held to different bars:

| Part | What it is | Bar |
|---|---|---|
| The engine | The C++ core in `src/` and `include/qx/`, its C API (`include/qx/c_api.h`), and the Python modules that drive it: `models`, `dmrg`, `hva`, `tdvp`, `structure_factor` | A new capability comes with a gate (next section). Today `dmrg.ground_state` and `hva` are gated; `tdvp`, `structure_factor` and the SU(2) path of `DMRGSolver` are validated in the C++ tests but have no Python gate yet. Changes to `c_api.h` keep existing callers working. |
| Experimental modules | The pure-Python modules that do not use the engine (VQE, error correction, noise, quantum walks, connectors, ...) | Tests for what they claim. Their API may change, and they may move to a separate `quantx_engine.contrib` namespace. |

## The verification standard

This is the rule that matters most. **Every engine capability exposed to Python ships
with a gate:** a test that compares the engine against a reference computed
independently of it, at a stated tolerance. The reference can be exact
diagonalisation, a state vector, or an analytic result. (The older `tdvp`,
`structure_factor` and SU(2) entry points predate the rule; giving them gates is open
work.) The existing gates show the pattern:

- `python/quantx_engine/tests/test_models_ed.py`: every model's ground state against
  an exact diagonalisation written separately from the engine, to 1e-10.
- `python/quantx_engine/tests/test_hva_statevector.py`: HVA circuits against an
  independent Fock-space simulator: energies and entanglement entropies.
- `python/quantx_engine/tests/test_noise.py`: the noise-model rules against the native
  circuit backend.

Three habits follow from it:

1. **A gate must fail when it cannot run.** Native gates skip when the library is
   missing, but fail instead under `QX_REQUIRE_NATIVE=1`, which the Linux native CI job
   sets. A gate that silently skips proves nothing.
2. **Refuse rather than guess.** An unknown argument, an unreachable symmetry sector or a
   degenerate start is an error with a message, never a silent default.
3. **Say only what a test checks.** Docstrings, the README and the changelog state the
   tolerance a gate actually holds. If a claim has no test, write the test or drop the
   claim.

## Development environment

You need Python 3.10 or later. For the native engine you also need CMake 3.20 or later
and a C++20 compiler (GCC 11+, Clang 14+, or Visual Studio 2022).

```bash
git clone https://github.com/<your-fork>/quantx.git
cd quantx
python -m venv python/.venv
source python/.venv/bin/activate          # Windows: python\.venv\Scripts\Activate.ps1
pip install -e "python/[dev]"
```

Without the native library the package runs in pure-Python mode:
`quantx_engine.backend_info()["native"]` is `False`, and the engine gates skip.

## Building the native engine

```bash
# Linux (Ubuntu/Debian: apt install cmake g++ ninja-build libgomp1) and macOS
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

```powershell
# Windows (Visual Studio 2022): the Visual Studio generator, 64-bit
cmake -S . -B build -A x64
cmake --build build --config Release --parallel
```

On Windows, build 64-bit: a 64-bit Python cannot load a 32-bit DLL. The plain
"Developer PowerShell for VS 2022" shortcut sets up the 32-bit (x86) compiler, so for
Ninja use the "x64 Native Tools Command Prompt" instead, or run
`Enter-VsDevShell -Arch amd64 -HostArch amd64` first.

The shared library lands in `build/Release/` (`build/<config>/` for any configuration):
`libquantx.so` on Linux, `libquantx.dylib` on macOS, `quantx.dll` on Windows. Point
Python at it with `QUANTX_LIB_PATH`, on Linux (on macOS, name `libquantx.dylib`):

```bash
export QUANTX_LIB_PATH="$PWD/build/Release/libquantx.so"
python -c "import quantx_engine; print(quantx_engine.backend_info())"
```

and on Windows:

```powershell
$env:QUANTX_LIB_PATH = "$PWD\build\Release\quantx.dll"
python -c "import quantx_engine; print(quantx_engine.backend_info())"
```

Options worth knowing:

| Option | Default | What it does |
|---|---|---|
| `QX_ENABLE_OPENMP` | ON | Threads the block-sparse kernels |
| `QX_ENABLE_BLAS` | OFF | Links OpenBLAS/LAPACKE for the large-block SVDs (faster at large bond dimension). Needs `-DQX_BLAS_INCLUDE_DIR=<dir with lapacke.h>` and `-DQX_BLAS_LIB=<OpenBLAS library>`; on Windows the OpenBLAS DLL must be on `PATH` at run time |
| `QX_USE_CUDA` | OFF | The cuSOLVER GPU path for the block-sparse SVDs (`-DCMAKE_CUDA_ARCHITECTURES=89` etc.) |
| `QX_ENABLE_CUDA` | OFF | cuFFT for the FFT provider (separate from `QX_USE_CUDA`) |

The published wheels are built with the defaults: no BLAS, no CUDA. Test engine changes
in that configuration too.

## Running the tests

| What | Command |
|---|---|
| Python package (pure-Python mode) | `pytest python/quantx_engine/tests -q` |
| Engine gates (needs the library) | `QX_REQUIRE_NATIVE=1 QUANTX_LIB_PATH=... pytest python/quantx_engine/tests/test_models_ed.py python/quantx_engine/tests/test_hva_statevector.py python/quantx_engine/tests/test_noise.py` |
| C++ tests | `ctest --test-dir build --output-on-failure` |
| Tutorial notebooks | `pip install -r notebooks/requirements.txt nbmake && pytest --nbmake notebooks/` |
| REST server | `pip install fastapi httpx anyio pydantic aiosqlite pytest-asyncio prometheus_client && pytest server/tests` |

Some C++ suites are long: `qx_hubbard_test` takes tens of minutes. Run the ones your
change touches while you work (`ctest -R <name>`), and the full set before asking for a
review. Run the Python suite with the library loaded as well, so the engine paths are
exercised.

## Pull requests

1. Branch from `master` and keep each PR to one change.
2. Name commits with a type prefix: `feat`, `fix`, `perf`, `docs`, `test`, `ci`,
   `chore`, e.g. `fix(dmrg): refuse an odd chain in an even sector`. Say in the body
   what changed and why, and what you ran to check it.
3. Add tests (see the verification standard) and a line under `[Unreleased]` in
   [CHANGELOG.md](CHANGELOG.md) for anything a user would notice.
4. Fill in the PR template, including the test plan.
5. A maintainer reviews the PR. Expect questions about the reference your gate compares
   against and about its tolerance.
6. Once the review is addressed and the checks pass, a maintainer squash-merges it.

Releases are cut by maintainers; see [RELEASING.md](RELEASING.md).

## Coding conventions

- **Python:** PEP 8; NumPy-style docstrings, with a `References` section citing the
  paper (with DOI) for every scientific algorithm; type hints on public functions.
- **C++:** C++20. Follow the indentation and naming of the file you are editing (the
  codebase mixes 2- and 4-space indentation); `snake_case` for functions and
  variables, `PascalCase` for types. When you add or change a function in
  `include/qx/c_api.h`, document its JSON inputs, outputs and conventions there.
- **Physics conventions:** state them. The spin models in `quantx_engine.models` use
  S = σ/2; the older spin-1/2 Heisenberg entry points (`DMRGSolver`, `tdvp`,
  `structure_factor`) use the Pauli convention J σ·σ. Every chain is open unless a
  parameter says otherwise.

## Licensing of contributions

QuantX is released under the [MIT License](LICENSE). By submitting a contribution you
agree that it is licensed under the same terms, and you certify the
[Developer Certificate of Origin 1.1](https://developercertificate.org/): sign off each
commit with `git commit -s`, which adds a `Signed-off-by: Your Name <you@example.com>`
line.

The sign-off is required. A DCO check runs on every pull request and fails if any
commit lacks a `Signed-off-by:` line with the email of that commit's author (merge
commits are exempt). To sign off commits you have already made, run
`git rebase --signoff origin/master` and force-push the branch. Run the same check
locally before you push:

```bash
scripts/check_dco.sh origin/master HEAD
```
