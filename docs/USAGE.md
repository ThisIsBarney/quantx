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

Export native continuous-frame payloads from the CLI:

```powershell
.\build-vs\Debug\qx_cli.exe --problem .\examples\problem_continuous2d_free.json --options .\examples\options_default.json --export-continuous-frames
```

`--export-continuous-frames` only affects continuous-space problems. It keeps JSON/API behavior append-only by adding optional payload fields without changing existing result keys.

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

Generate visualization with explicit solver mode:

```powershell
.\scripts\run_visualization.ps1 -SolverMode auto
```

Force engine-first visualization output:

```powershell
.\scripts\run_visualization.ps1 -SolverMode engine
```

Solver modes:

- `auto` (default): use engine-exported continuous frames first, fallback to Python reference solver if export is unavailable.
- `engine`: require engine-exported continuous frames (fails fast if export is unavailable).
- `python`: force Python reference solver for continuous scenarios.

When engine export is used, QuantX emits `continuous_export.frames` in run results with native frame payloads (2D density grids, 3D slices and point-cloud payloads).
If an engine export payload is partial or corrupt, the visualization layer prints a clear console diagnostic and either fails fast (`engine`) or falls back to the Python reference solver (`auto`).

For subagent work, follow the checklist in `docs/ENGINEERING_STANDARDS.md` and keep each agent in its own `codex/*` worktree.

Render from a pre-exported continuous frame JSON:

```powershell
python .\scripts\visualize_simulation.py --scenario spatial3d --continuous-export-json out/visualizations/quantum_spatial3d_wavepacket_barrier_periodic.json
```

Generated files:

- `out/visualizations/rabi_population.html`
- `out/visualizations/mps_diagnostics.html`
- `out/visualizations/quantum_spatial_wavepacket_barrier.html`
- `out/visualizations/quantum_spatial3d_wavepacket_barrier_periodic.html`

Each page includes animation controls (`Play`, `Pause`, timeline slider).
`quantum_spatial_wavepacket_barrier.html` shows a 1D spatial wavepacket evolution (`|psi(x,t)|^2`) solved by finite difference + Crank-Nicolson, with phase color, potential profile, and sampled position cloud.
`quantum_spatial3d_wavepacket_barrier_periodic.html` shows 3D evolution solved by split-step Fourier and can be explored in a true WebGL world view via the standalone player.

Generate a focused 3D page with selectable boundary condition:

```powershell
python .\scripts\visualize_simulation.py --scenario spatial3d --spatial3d-potential barrier --spatial3d-boundary absorbing
```

Export 3D frames as JSON and prepare standalone JS player assets:

```powershell
.\scripts\run_spatial3d_js_export.ps1
```

The 3D export helper defaults to `-SolverMode engine`, reuses an existing native DLL from the current repo or a parent worktree when available, and writes the exported frame JSON plus `spatial3d_player.html` into the requested output directory.

Outputs include:

- `out/visualizations/quantum_spatial3d_wavepacket_barrier_periodic.json`
- `out/visualizations/spatial3d_player.html`

Open `spatial3d_player.html`, then load the exported JSON (path input or local file picker).

Player interaction features (true 3D viewer):

- WebGL point-cloud world rendering with orbit camera controls (rotate/pan/zoom).
- Frame navigation: `Prev` / `Next`, slider, keyboard `Left` / `Right` arrows.
- Playback hotkey: `Space` toggles `Play` / `Pause`.
- Point rendering controls: point size, density cutoff, playback speed, auto-rotate.
- Real-time diagnostics: rendered point count, frame vmax, boundary/potential/norm and `<x,y,z>`.

Notes:

- The JSON export now includes per-frame volumetric points as `frame.points` entries in `[x, y, z, density]` format.
- The standalone bundle is offline-first: `spatial3d_player.html` loads `./vendor/three.min.js` and `./vendor/OrbitControls.js` locally.
- If local vendor files are missing, the player will try CDN fallback from `unpkg`.

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
3D pages support `periodic`, `absorbing`, and `dirichlet` boundary options; the standalone player renders the exported volumetric point cloud in true 3D.

## 1.1) Benchmark usage

Run the unified benchmark flow with gates:

```powershell
.\scripts\benchmark_all.ps1 -NoiseSweep -RunGates -EnforceGate
```

Run the unified benchmark flow without gates:

```powershell
.\scripts\benchmark_all.ps1 -NoiseSweep
```

Run a single suite directly when you are iterating on one target:

```powershell
.\scripts\run_benchmark.ps1 -Suite benchmarks/suites/circuit_accuracy_smoke.json
```

Legacy continuous-space alias for the unified dispatcher:

```powershell
.\scripts\run_problem_benchmark.ps1 -Suite benchmarks/suites/continuous2d_smoke.json
```

The unified benchmark entrypoint covers:

- `circuit_accuracy_smoke`
- `circuit_noise_sweep`
- `continuous2d_smoke`
- `continuous3d_smoke`
- `continuous3d_accuracy`

Reports are written to `benchmarks/reports/`:

- suite JSON report: `*.json`
- suite markdown summary: `*.md`
- cumulative index: `INDEX.md`
- suite trend report: `*_TREND.md`

CI integration uses the same script:

```powershell
.\scripts\benchmark_all.ps1 -NoiseSweep -RunGates -EnforceGate -SkipBuild
```

- workflow file: `.github/workflows/benchmark-gate.yml`
- runs build + tests + unified benchmark gate on Windows
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

Capability query:

```python
caps = e.circuit_capabilities()
print(caps["supported_gates"], caps["supported_noise_models"], caps["max_qubits"])
```

Output includes:

- `expectation_exact`: exact expectation from final statevector
- `expectation_shot`: estimated expectation from sampled counts
- `estimation_error`: `expectation_shot - expectation_exact`
- `shot_variance`: per-observable Bernoulli variance estimate (`1 - mean^2`)
- `shot_stderr`: per-observable standard error (`sqrt(variance/shots)`)
- `error_summary`: aggregate error stats (`mean_abs_error`, `max_abs_error`, count)
- `engine_version`: producing engine version string (for reproducibility/audit)
- on failure, `ok: false` plus `error_code` and `error_message`

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
