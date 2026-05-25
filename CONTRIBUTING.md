# Contributing to QuantX

Thank you for your interest in contributing to QuantX! This guide covers the
full contribution lifecycle — from filing issues to getting your PR merged —
as well as development setup, build instructions, and coding conventions.

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Table of Contents

1. [Community channels](#community-channels)
2. [How to contribute](#how-to-contribute)
   - [Reporting bugs](#reporting-bugs)
   - [Requesting features](#requesting-features)
   - [Submitting a pull request](#submitting-a-pull-request)
   - [Developer Certificate of Origin](#developer-certificate-of-origin)
3. [Development environment](#development-environment)
4. [Pure-Python SDK](#pure-python-sdk)
5. [Building the native C++ engine](#building-the-native-c-engine)
   - [Linux / macOS](#linux--macos)
   - [Windows](#windows)
   - [MSYS2 / MinGW (Windows)](#msys2--mingw-windows)
6. [Verifying the build](#verifying-the-build)
7. [Wiring Python to the native library](#wiring-python-to-the-native-library)
8. [Running the test suite](#running-the-test-suite)
9. [Coding conventions](#coding-conventions)

---

## Community channels

| Channel | Purpose |
|---------|---------|
| [GitHub Issues](https://github.com/ThisIsBarney/quantx/issues) | Bug reports, feature requests, questions |
| [GitHub Discussions](https://github.com/ThisIsBarney/quantx/discussions) | General Q&A, ideas, show-and-tell |
| [Discord](https://discord.gg/quantx) | Real-time chat — `#help`, `#dev`, `#hardware` channels |

For security vulnerabilities please read [SECURITY.md](SECURITY.md) and
**do not** open a public issue.

---

## How to contribute

### Reporting bugs

Before opening an issue, search [existing issues](https://github.com/ThisIsBarney/quantx/issues)
to avoid duplicates.

Use the **Bug Report** template when filing a bug.  Include:
- QuantX version (`python -c "import quantx_engine; print(quantx_engine.__version__)"`)
- Python version and operating system
- Minimal reproducible example
- Full traceback (if applicable)

### Requesting features

Use the **Feature Request** template.  Describe the quantum simulation use-case
you are trying to address and, if possible, a sketch of the proposed API.

### Submitting a pull request

1. **Fork** the repository and create a branch:
   ```bash
   git checkout -b feat/short-description
   ```
2. **Write tests** — every new feature needs at least one `test_*.py` file.
3. **Run the test suite** locally (see [Running the test suite](#running-the-test-suite)).
4. **Open a PR** against `master` using the PR template.
5. A maintainer will review within a few business days.
6. Address any feedback; the PR is merged once CI passes and at least one
   approval is received.

**PR scope guidelines**:
- Keep PRs focused — one feature or fix per PR.
- Include a `CHANGELOG.md` entry under `[Unreleased]` for user-visible changes.
- Reference the issue number in the PR description (`Closes #NNN`).

### Developer Certificate of Origin

By submitting a pull request you certify that you have the right to submit
the contribution under the project's MIT licence and that you agree to the
[Developer Certificate of Origin v1.1](https://developercertificate.org/).

Add a sign-off to each commit:
```bash
git commit -s -m "feat: add my contribution"
```
or append manually:
```
Signed-off-by: Your Name <your@email.com>
```

---

## Development environment

**Python requirements**: Python ≥ 3.11  
**C++ requirements**: CMake ≥ 3.20, a C++20-capable compiler (GCC ≥ 11, Clang ≥ 14, or MSVC ≥ 19.30)

```bash
# Clone the repo
git clone https://github.com/yourusername/quantx.git
cd quantx

# Create a virtual environment
python -m venv python/.venv
source python/.venv/bin/activate        # Linux / macOS
python\.venv\Scripts\Activate.ps1       # Windows PowerShell

# Install the Python SDK in editable mode with all dev dependencies
pip install scipy networkx pennylane
pip install -e "python/[dev]"
```

---

## Pure-Python SDK

The vast majority of QuantX functionality is pure Python and requires **no C++
build**. All modules in `python/quantx_engine/` except `engine.py` work
without the native library.

```bash
# Run the full pure-Python test suite
pytest python/quantx_engine/tests/ -q --tb=short
```

---

## Building the native C++ engine

The native engine (`libquantx.so` / `quantx.dll`) provides C++ implementations
of Lanczos diagonalization, Trotter simulation, MPS backends, and 30+ physics
solvers. It is an optional performance accelerator; the Python SDK falls back
gracefully when it is unavailable.

### Linux / macOS

```bash
# Install prerequisites (Ubuntu / Debian)
sudo apt-get install cmake g++ ninja-build libgomp1

# macOS with Homebrew
brew install cmake ninja libomp

# Configure
cmake -S . -B build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DQX_ENABLE_OPENMP=ON \
  -DQX_ENABLE_MPI=OFF \
  -DQX_ENABLE_PYBIND=OFF

# Build (uses all available CPU cores)
cmake --build build --parallel $(nproc)

# The shared library is at:
#   build/Release/libquantx.so   (Linux)
#   build/Release/libquantx.dylib (macOS)
```

### Windows

Using **Visual Studio 2022** (MSVC):

```powershell
# From a Developer PowerShell for VS 2022 prompt
cmake -S . -B build `
  -DCMAKE_BUILD_TYPE=Release `
  -DQX_ENABLE_OPENMP=OFF `
  -DQX_ENABLE_MPI=OFF `
  -DQX_ENABLE_PYBIND=OFF

cmake --build build --config Release --parallel

# DLL is at: build\Release\quantx.dll
```

### MSYS2 / MinGW (Windows)

If you prefer GCC on Windows via [MSYS2](https://www.msys2.org/):

```bash
# In an MSYS2 UCRT64 shell — install build tools
pacman -S mingw-w64-ucrt-x86_64-cmake \
           mingw-w64-ucrt-x86_64-gcc \
           mingw-w64-ucrt-x86_64-ninja

# Configure + build
cmake -S . -B build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DQX_ENABLE_OPENMP=ON

cmake --build build --parallel $(nproc)

# DLL is at: build/Release/quantx.dll
```

---

## Verifying the build

After a successful build, run the C++ smoke tests:

```bash
# Linux
./build/Release/qx_smoke
./build/Release/qx_fft_smoke

# Windows
.\build\Release\qx_smoke.exe
.\build\Release\qx_fft_smoke.exe
```

Check that the shared library exports the expected symbols:

```bash
# Linux
nm -D build/Release/libquantx.so | grep "qx_run\|qx_version\|qx_list_solvers"

# Windows (from a VS Developer Command Prompt)
dumpbin /exports build\Release\quantx.dll | findstr qx_run
```

---

## Wiring Python to the native library

Set the `QUANTX_LIB_PATH` environment variable to the path of the compiled
shared library, then import QuantX as usual:

```python
import os
os.environ["QUANTX_LIB_PATH"] = "/path/to/build/Release/libquantx.so"

from quantx_engine import backend_info
print(backend_info())
# {'native': True, 'gpu': False, 'version': '0.1.0', 'solvers': [...]}

from quantx_engine.engine import Engine
e = Engine()
result = e.run_solver("heisenberg", {"n_sites": 8, "J": 1.0, "scenario": "ground_state"})
print(result["ground_energy"])
```

When `QUANTX_LIB_PATH` is set, `LanczosSolver`, `TrotterSimulator`, and
`MPSSimulator` automatically route eligible problems through the native
C++ backend (≥10× faster for large systems), falling back to pure Python
for unsupported problem shapes.

---

## Running the test suite

```bash
# Fast pure-Python unit tests
pytest python/quantx_engine/tests/ -q --tb=short

# Literature-validation tests (M134-M148) — reproduce paper results
pytest python/quantx_engine/tests/test_literature_*.py -v

# Full suite including server tests
pytest python/ server/tests/ -q --tb=short
```

With the native library available:

```bash
QUANTX_LIB_PATH=/path/to/libquantx.so \
  pytest python/quantx_engine/tests/ -q --tb=short
```

---

## Coding conventions

- **Python**: Follow PEP 8. NumPy-style docstrings with `References` section
  citing DOI-linked papers for all scientific algorithms.
- **C++**: C++20, 2-space indentation, `snake_case` for functions/variables,
  `PascalCase` for structs/classes.
- **Tests**: Every new feature needs a `test_*.py` file. Literature-validation
  tests (`test_literature_mNNN.py`) must cite the paper and reproduce a specific
  numerical result. Use `pytest.skip()` for graceful API-mismatch handling.
- **References**: Use the `.. [N] Author (year). Title. Journal. DOI` format
  in module docstrings.
