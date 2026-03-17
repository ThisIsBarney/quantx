# Usage Guide

## 1) CLI usage (no coding required)

Build once:

```powershell
.\scripts\build.ps1
```

Run simulation:

```powershell
.\build-vs\Debug\qx_cli.exe --problem .\examples\problem_rabi.json --options .\examples\options_default.json
```

Run circuit:

```powershell
.\build-vs\Debug\qx_cli.exe --circuit .\examples\circuit_bell.json --options .\examples\options_default.json
```

Run noisy circuit:

```powershell
.\build-vs\Debug\qx_cli.exe --circuit .\examples\circuit_bell_noisy.json --options .\examples\options_default.json
```

Explain backend selection:

```powershell
.\build-vs\Debug\qx_cli.exe --problem .\examples\problem_xx.json --options .\examples\options_default.json --explain
```

Run canned CLI examples:

```powershell
.\scripts\run_cli.ps1
```

## 1.1) Benchmark usage

Run accuracy smoke benchmark:

```powershell
.\scripts\run_benchmark.ps1
```

Run both accuracy + noise sweep and regenerate index:

```powershell
.\scripts\benchmark_all.ps1 -NoiseSweep
```

Run with threshold gating (fails if any suite breaches configured gates):

```powershell
.\scripts\benchmark_all.ps1 -NoiseSweep -EnforceGate
```

Reports are written to `benchmarks/reports/`:

- suite JSON report: `*.json`
- suite markdown summary: `*.md`
- cumulative index: `INDEX.md`
- suite trend report: `*_TREND.md`

CI integration:

- workflow file: `.github/workflows/benchmark-gate.yml`
- runs build + tests + accuracy benchmark + gate check on Windows

## 2) Python SDK

```powershell
.\scripts\demo.ps1
```

Or call directly:

```python
from quantx_engine import Engine
e = Engine(r"D:\WUJ\coding\QuantX\build-vs\Debug\quantx.dll")
res = e.run({"qubits": 1, "method": "krylov", "total_time": 1.57, "dt": 0.01,
             "hamiltonian_terms":[{"label":"X0","coefficient":1.0}]},
            {"enable_checkpoint": False})
print(res)
```

Circuit call:

```python
from quantx_engine import Engine
e = Engine(r"D:\WUJ\coding\QuantX\build-vs\Debug\quantx.dll")
cres = e.run_circuit({
    "qubits": 2,
    "shots": 1000,
    "observables": ["Z0", "Z1", "Z0Z1"],
    "depolarizing_prob": 0.01,
    "amplitude_damping_gamma": 0.01,
    "seed": 42,
    "operations": [
      {"name": "H", "targets": [0]},
      {"name": "CNOT", "controls": [0], "targets": [1]}
    ]
}, {"enable_checkpoint": False})
print(cres)
```

Output includes:

- `expectation_exact`: exact expectation from final statevector
- `expectation_shot`: estimated expectation from sampled counts
- `estimation_error`: `expectation_shot - expectation_exact`
- `shot_variance`: per-observable Bernoulli variance estimate (`1 - mean^2`)
- `shot_stderr`: per-observable standard error (`sqrt(variance/shots)`)
- `error_summary`: aggregate error stats (`mean_abs_error`, `max_abs_error`, count)

Observable support:

- currently supports Pauli-string observables such as `Z0`, `Z0Z1`, `X0X1`, `Y0`
- multi-body combinations are supported, e.g. `Y0Y1`, `X0Y1`
- for `X/Y` observables, shot estimation is done via basis rotation + sampling

## 3) C ABI

Exports:

- `qx_run(problem_json, options_json)`
- `qx_run_circuit(circuit_json, options_json)`
- `qx_explain_backend_choice(problem_json, options_json)`
- `qx_free_string(ptr)`

Use `quantx.dll` in `build-vs/Debug`.
