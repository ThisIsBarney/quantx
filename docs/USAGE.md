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

Run with explicit log level:

```powershell
.\build-vs\Debug\qx_cli.exe --problem .\examples\problem_rabi.json --options .\examples\options_default.json --log-level debug
```

`--log-level` supports `trace|debug|info|warn|error|off` (default: `warn`).

Run MPS simulation with explicit TEBD controls:

```powershell
.\build-vs\Debug\qx_cli.exe --problem .\examples\problem_mps_tebd.json --options .\examples\options_default.json
```

Run circuit:

```powershell
.\build-vs\Debug\qx_cli.exe --circuit .\examples\circuit_bell.json --options .\examples\options_default.json
```

Run noisy circuit:

```powershell
.\build-vs\Debug\qx_cli.exe --circuit .\examples\circuit_bell_noisy.json --options .\examples\options_default.json
```

Run continuous-space 2D problem:

```powershell
.\build-vs\Debug\qx_cli.exe --problem .\examples\problem_continuous2d_free.json --options .\examples\options_default.json
```

Explain backend selection:

```powershell
.\build-vs\Debug\qx_cli.exe --problem .\examples\problem_xx.json --options .\examples\options_default.json --explain
```

Show engine version:

```powershell
.\build-vs\Debug\qx_cli.exe --version
```

Validate input JSON only (no simulation run):

```powershell
.\build-vs\Debug\qx_cli.exe --problem .\examples\problem_rabi.json --validate-only
```

Run canned CLI examples:

```powershell
.\scripts\run_cli.ps1
```

Run only the MPS control example:

```powershell
.\scripts\run_mps_example.ps1
```

Generate visualization pages (HTML + SVG):

```powershell
.\scripts\run_visualization.ps1
```

Generated files:

- `out/visualizations/rabi_population.html`
- `out/visualizations/mps_diagnostics.html`
- `out/visualizations/quantum_spatial_wavepacket_barrier.html`
- `out/visualizations/quantum_spatial3d_wavepacket_barrier_periodic.html`

Each page includes animation controls (`Play`, `Pause`, timeline slider).
`quantum_spatial_wavepacket_barrier.html` shows a 1D spatial wavepacket evolution (`|psi(x,t)|^2`) solved by finite difference + Crank-Nicolson, with phase color, potential profile, and sampled position cloud.
`quantum_spatial3d_wavepacket_barrier_periodic.html` shows 3D evolution slices (XY/XZ/YZ) of `|psi(x,y,z,t)|^2` solved by split-step Fourier.

Generate a focused 3D page with selectable boundary condition:

```powershell
python .\scripts\visualize_simulation.py --scenario spatial3d --spatial3d-potential barrier --spatial3d-boundary absorbing
```

Export 3D frames as JSON and prepare standalone JS player assets:

```powershell
.\scripts\run_spatial3d_js_export.ps1
```

Outputs include:

- `out/visualizations/quantum_spatial3d_wavepacket_barrier_periodic.json`
- `out/visualizations/spatial3d_player.html`

Open `spatial3d_player.html`, then load the exported JSON (path input or local file picker).

Player interaction features:

- Per-panel slice remap: each panel can switch among `XY` / `XZ` / `YZ`.
- Frame navigation: `Prev` / `Next`, slider, keyboard `Left` / `Right` arrows.
- Playback hotkey: `Space` toggles `Play` / `Pause`.
- Intensity scales: `Linear`, `Sqrt`, `Log`.
- Colormaps: `QuantX`, `Grayscale`, `Thermal`.
- Render cache warmup for smoother playback on long sequences.

Generate a spatial potential gallery:

```powershell
.\scripts\run_spatial_gallery.ps1
```

Gallery outputs:

- `out/visualizations/quantum_spatial_wavepacket_free.html`
- `out/visualizations/quantum_spatial_wavepacket_barrier.html`
- `out/visualizations/quantum_spatial_wavepacket_harmonic.html`
- `out/visualizations/quantum_spatial_wavepacket_well.html`
- `out/visualizations/quantum_spatial2d_wavepacket_free.html`
- `out/visualizations/quantum_spatial2d_wavepacket_barrier.html`
- `out/visualizations/quantum_spatial2d_wavepacket_harmonic.html`
- `out/visualizations/quantum_spatial2d_wavepacket_well.html`
- `out/visualizations/quantum_spatial3d_wavepacket_barrier_absorbing.html`

2D pages use a numerical split-step Fourier solver and render `|psi(x,y,t)|^2` as animated heatmaps.
3D pages render orthogonal slices (XY/XZ/YZ) of `|psi(x,y,z,t)|^2` and support `periodic`, `absorbing`, and `dirichlet` boundary options.

## 1.1) Benchmark usage

Run accuracy smoke benchmark:

```powershell
.\scripts\run_benchmark.ps1
```

Run continuous-space 2D smoke benchmark (uses `Engine.run(...)` with problem JSON):

```powershell
.\scripts\run_problem_benchmark.ps1
```

Run both accuracy + noise sweep and regenerate index:

```powershell
.\scripts\benchmark_all.ps1 -NoiseSweep
```

Run with threshold gating (fails if any suite breaches configured gates):

```powershell
.\scripts\benchmark_all.ps1 -NoiseSweep -EnforceGate
```

Run a single suite with gate in one command:

```powershell
.\scripts\run_benchmark_and_gate.ps1 -Suite benchmarks/suites/circuit_accuracy_smoke.json
```

Reports are written to `benchmarks/reports/`:

- suite JSON report: `*.json`
- suite markdown summary: `*.md`
- cumulative index: `INDEX.md`
- suite trend report: `*_TREND.md`

CI integration:

- workflow file: `.github/workflows/benchmark-gate.yml`
- runs build + tests + both benchmark suites + gate checks on Windows
- uploads `benchmarks/reports/` as CI artifact

## 2) Python SDK

```powershell
.\scripts\demo.ps1
```

Or call directly:

```python
from quantx_engine import Engine
e = Engine(r"D:\WUJ\coding\QuantX\build-vs\Debug\quantx.dll")
print("engine version:", e.version())
res = e.run({"qubits": 1, "method": "krylov", "total_time": 1.57, "dt": 0.01,
             "krylov_dim": 8,
             "hamiltonian_terms":[{"label":"X0","coefficient":1.0}]},
            {"enable_checkpoint": False})
print(res)
print(e.validate_problem({"qubits": 1, "hamiltonian_terms":[{"label":"Z0","coefficient":1.0}]}))
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

Continuous-space 2D call (same unified `Engine.run(...)` API):

```python
from quantx_engine import Engine
e = Engine(r"D:\\WUJ\\coding\\QuantX\\build-vs\\Debug\\quantx.dll")
pres = e.run({
  "kind": "continuous_space_2d",
  "continuous_space_2d": {
    "nx": 128,
    "ny": 128,
    "dx": 0.1,
    "dy": 0.1,
    "dt": 0.001,
    "steps": 100,
    "boundary": "periodic",
    "potential_kind": "none",
    "potential_params": {"psi_sigma": 0.25, "psi_kx": 0.0, "psi_ky": 0.0}
  }
}, {"enable_checkpoint": False})
print(pres["backend"], pres["observables"]["norm"])
```

Continuous-space 3D call (same unified `Engine.run(...)` API):

```python
from quantx_engine import Engine
e = Engine(r"D:\\WUJ\\coding\\QuantX\\build-vs\\Debug\\quantx.dll")
pres3d = e.run({
  "kind": "continuous_space_3d",
  "continuous_space_3d": {
    "nx": 96,
    "ny": 96,
    "nz": 96,
    "dx": 0.08,
    "dy": 0.08,
    "dz": 0.08,
    "dt": 0.0005,
    "steps": 240,
    "boundary": "periodic",
    "potential_kind": "harmonic",
    "potential_params": {"kx": 0.4, "ky": 0.4, "kz": 0.4, "psi_sigma": 0.25}
  }
}, {"enable_checkpoint": False})
print(pres3d["backend"], pres3d["observables"]["sigma_z"])
```

Standard template files:

- `examples/problem_continuous2d_template.json`
- `examples/problem_continuous3d_template.json`

Framework-adapter call (Qiskit-like request):

```python
qiskit_req = {
  "num_qubits": 2,
  "shots": 1000,
  "operations": [
    {"name": "h", "qubits": [0]},
    {"name": "cx", "qubits": [0, 1]}
  ]
}
print(e.run_qiskit_like(qiskit_req, {"enable_checkpoint": False}))
```

Output includes:

- `expectation_exact`: exact expectation from final statevector
- `expectation_shot`: estimated expectation from sampled counts
- `estimation_error`: `expectation_shot - expectation_exact`
- `shot_variance`: per-observable Bernoulli variance estimate (`1 - mean^2`)
- `shot_stderr`: per-observable standard error (`sqrt(variance/shots)`)
- `error_summary`: aggregate error stats (`mean_abs_error`, `max_abs_error`, count)
- `engine_version`: producing engine version string (for reproducibility/audit)

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

`continuous_space_2d` also goes through `qx_run(problem_json, options_json)` (no extra ABI entrypoint):

```json
{
  "kind": "continuous_space_2d",
  "continuous_space_2d": {
    "nx": 128,
    "ny": 128,
    "dx": 0.1,
    "dy": 0.1,
    "dt": 0.001,
    "steps": 100,
    "boundary": "periodic",
    "potential_kind": "none"
  }
}
```

`continuous_space_3d` also goes through the same `qx_run(problem_json, options_json)` entrypoint:

```json
{
  "kind": "continuous_space_3d",
  "continuous_space_3d": {
    "nx": 96,
    "ny": 96,
    "nz": 96,
    "dx": 0.08,
    "dy": 0.08,
    "dz": 0.08,
    "dt": 0.0005,
    "steps": 240,
    "boundary": "periodic",
    "potential_kind": "harmonic"
  }
}
```

## 4) Problem JSON fields (core)

- `qubits`: number of qubits.
- `equation`: equation family (v1 uses `Schrodinger`).
- `total_time`: total evolution time.
- `dt`: timestep size.
- `method`: evolution method.
- statevector methods: `trotter`, `krylov`.
- MPS methods: `tebd`, `tdvp`.
- `krylov_dim`: Arnoldi/Krylov subspace dimension for `krylov`.
- `mps_max_bond`: MPS max bond dimension cap (>=2).
- `mps_trunc_tol`: MPS truncation tolerance (>0).
- `hamiltonian_terms`: array of terms like `{ "label": "X0X1", "coefficient": 1.0 }`.
- floating-point fields support scientific notation, e.g. `1e-6`.
