# QuantX Engine (v1 Scaffold)

QuantX is a C++20-first quantum simulation engine scaffold with:

- C++ core runtime
- plugin-like backend registry
- statevector + MPS built-in backends
- scheduler/explain API
- C ABI (`qx_run`, `qx_explain_backend_choice`)
- Python SDK wrapper (ctypes)

## Product Goals

- quantum many-body and dynamics simulation engine
- core execution backend for quantum computing frameworks

## Quick Start

```powershell
.\scripts\build.ps1
.\scripts\test.ps1
.\scripts\demo.ps1
.\scripts\run_cli.ps1
```

## Architecture

- `IR`: `QuantumProblem`, `RunOptions`, `OperatorGraph`, `EquationType`
- `Scheduler`: backend auto-selection + explanation reasons
- `Backend`: unified interface `initialize/evolve/measure/checkpoint/restore`
- `Runtime`: events and resource metadata
- `API`:
  - C++: `qx::run`, `qx::run_circuit`, `qx::explain_backend_choice`
  - C ABI: `qx_run`, `qx_run_circuit`, `qx_explain_backend_choice`
  - Python: `Engine.run`, `Engine.run_circuit`, `Engine.explain`

## Repository Layout

- `include/qx/` public headers
- `src/` core implementation
- `schemas/` v1 JSON input schema
- `python/` SDK package
- `bindings/python/` optional pybind11 module
- `tests/` smoke tests
- `examples/` runnable demos
- `scripts/` convenience scripts for build/test/demo
- `docs/USAGE.md` quick usage cookbook (CLI/Python/C ABI)
- `docs/ML_LLM_INTEGRATION.md` P2 ML/LLM roadmap and guardrails

## Notes

- Statevector supports Trotter and Krylov evolution with one-body and two-body Pauli-string terms.
- MPI/CUDA/pybind11 flags are prepared at CMake level; full HPC kernels are next milestone.
- Quantum-computing backend roadmap is tracked in `docs/ROADMAP.md` and `docs/QUANTUM_COMPUTING_BACKEND.md`.
