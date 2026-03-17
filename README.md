# QuantX Engine (v1 Scaffold)

QuantX is a C++20-first quantum simulation engine scaffold with:

- C++ core runtime
- plugin-like backend registry
- statevector + MPS built-in backends
- scheduler/explain API
- C ABI (`qx_run`, `qx_explain_backend_choice`)
- Python SDK wrapper (ctypes)

## Quick Start

```powershell
.\scripts\build.ps1
.\scripts\test.ps1
.\scripts\demo.ps1
```

## Architecture

- `IR`: `QuantumProblem`, `RunOptions`, `OperatorGraph`, `EquationType`
- `Scheduler`: backend auto-selection + explanation reasons
- `Backend`: unified interface `initialize/evolve/measure/checkpoint/restore`
- `Runtime`: events and resource metadata
- `API`:
  - C++: `qx::run`, `qx::explain_backend_choice`
  - C ABI: `qx_run`, `qx_explain_backend_choice`
  - Python: `Engine.run`, `Engine.explain`

## Repository Layout

- `include/qx/` public headers
- `src/` core implementation
- `schemas/` v1 JSON input schema
- `python/` SDK package
- `bindings/python/` optional pybind11 module
- `tests/` smoke tests
- `examples/` runnable demos
- `scripts/` convenience scripts for build/test/demo

## Notes

- Statevector supports Trotter and Krylov evolution with one-body and two-body Pauli-string terms.
- MPI/CUDA/pybind11 flags are prepared at CMake level; full HPC kernels are next milestone.
