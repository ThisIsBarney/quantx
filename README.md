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
.\scripts\run_benchmark.ps1
.\scripts\benchmark_all.ps1 -NoiseSweep
.\scripts\benchmark_all.ps1 -NoiseSweep -RunGates -EnforceGate
.\scripts\run_verification_matrix.ps1
.\scripts\run_benchmark_and_gate.ps1 -Suite benchmarks/suites/circuit_accuracy_smoke.json
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

- Statevector supports Trotter and Arnoldi-Krylov evolution (configurable `krylov_dim`) with one-body and two-body Pauli-string terms.
- MPI/CUDA/pybind11 flags are prepared at CMake level; full HPC kernels are next milestone.
- Optional CUDA/cuFFT acceleration path can be enabled with `-DQX_ENABLE_CUDA=ON` when CUDA Toolkit is installed.
- Quantum-computing backend roadmap is tracked in `docs/ROADMAP.md` and `docs/QUANTUM_COMPUTING_BACKEND.md`.
- `scripts/benchmark_all.ps1` is the unified benchmark entrypoint for circuit accuracy/noise sweep plus continuous 2D/3D smoke and 3D accuracy suites.
- `scripts/run_verification_matrix.ps1` is the one-shot pre-commit flow: smoke, adapters, visualization, and the selected benchmark gate.
- Subagent workflow, branch isolation, and code-quality guardrails are documented in `docs/ENGINEERING_STANDARDS.md`.
- Smoke tests are expected to report failures on stderr/stdout instead of opening modal debug dialogs on Windows.

## v1.1-rc Checklist

- Run `.\scripts\test.ps1` before release tagging.
- Run `.\scripts\benchmark_all.ps1 -NoiseSweep -RunGates -EnforceGate` for the reproducible benchmark gate.
- Inspect `benchmarks/reports/INDEX.md` and the latest `*_TREND.md` files for regressions.
- Generate the visualization artifacts with `.\scripts\run_visualization.ps1`.
- Re-check `docs/DEPENDENCIES.md` before adding any new third-party package.
