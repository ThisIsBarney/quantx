# Usage Guide

## 1) CLI usage (no coding required)

Build once:

```powershell
.\scripts\build.ps1
```

Build probe (no compile):

```powershell
powershell -Command "$ErrorActionPreference='Stop'; $vswhere='C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'; if (-not (Test-Path $vswhere)) { throw 'vswhere.exe not found' }; $install=& $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath -format value | Select-Object -First 1; if (-not $install) { throw 'Visual Studio VC toolchain not found' }; $dev=Join-Path $install 'Common7\Tools\VsDevCmd.bat'; if (-not (Test-Path $dev)) { throw \"VsDevCmd not found: $dev\" }; cmd /c \"`\"$dev`\" -arch=x64 -host_arch=x64 >nul && where cl\""
```

Probe troubleshooting:

- `vswhere.exe not found`: install Visual Studio Installer components or run on an image with VS Build Tools.
- `Visual Studio VC toolchain not found`: ensure `Microsoft.VisualStudio.Component.VC.Tools.x86.x64` is installed.
- `VsDevCmd not found`: verify `VSINSTALLDIR` or the discovered installation path.
- `where cl` fails: toolchain environment failed to initialize; reopen shell and rerun the probe.

The probe only checks toolchain discovery and compiler path resolution. It does not run `cmake` and does not trigger compilation.

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

Additional continuous-export knobs:

```powershell
.\build-vs\Debug\qx_cli.exe --problem .\examples\problem_continuous2d_free.json --options .\examples\options_default.json --export-continuous-frames --export-frame-stride 4 --export-points-max 32 --export-density-floor-ratio 0.2
```

- `--export-frame-stride` samples the public `continuous_export.frames` wrapper every N steps while preserving the full internal `continuous_density_frames` timeline.
- `--export-points-max` caps the sparse 3D point-cloud payload size.
- `--export-density-floor-ratio` filters sparse 3D points below a relative density threshold before the max-point cap is applied.

The public export wrapper now includes:

- `continuous_export.schema_version`
- `continuous_export.export_config`
- `continuous_export.frames`

`export_config` echoes the effective values used for the run so downstream tooling can reason about the export contract without guessing defaults.

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

When engine export is used, QuantX emits `continuous_export.frames` in run results with native frame payloads (2D density grids, 3D slices and point-cloud payloads). The wrapper is stride-sampled, while the internal `continuous_density_frames` list still preserves the full solver timeline for compatibility.
If an engine export payload is partial or corrupt, the visualization layer prints a clear console diagnostic and either fails fast (`engine`) or falls back to the Python reference solver (`auto`).
The benchmark/reporting layer treats these exports as append-only and now records platform/profile metadata alongside the extracted frame payloads.

## Continuous Export Schema

`continuous_export` is treated as an append-only contract.

- `schema_version` identifies the wrapper schema that downstream tooling should validate first.
- `export_config` echoes the effective export knobs used for the run, so consumers do not need to guess defaults.
- `frames` contains the actual sampled payloads and may gain new optional fields over time.
- Unknown fields should be ignored by non-strict readers, not treated as breaking changes.

Strict consumers should validate `continuous_export.schema_version`, require the documented fields they depend on, and fail fast if the wrapper is incomplete.
Non-strict consumers should prefer `--SolverMode auto` or equivalent fallback behavior so they can keep running when engine export is unavailable.

Example commands:

```powershell
.\scripts\run_visualization.ps1 -SolverMode engine
.\scripts\run_visualization.ps1 -SolverMode auto
```

For subagent work, follow the checklist in `docs/ENGINEERING_STANDARDS.md` and keep each agent in its own `codex/*` worktree.

Render from a pre-exported continuous frame JSON:

```powershell
python .\scripts\visualize_simulation.py --scenario spatial3d --continuous-export-json out/visualizations/quantum_spatial3d_wavepacket_barrier_periodic.json
```

Generated files (selection):

- `out/visualizations/quantum_dashboard.html` — unified single-page dashboard (all 31 scenarios)
- `out/visualizations/rabi_population.html`
- `out/visualizations/mps_diagnostics.html`
- `out/visualizations/quantum_spatial_wavepacket_barrier.html`
- `out/visualizations/quantum_spatial3d_wavepacket_barrier_periodic.html`
- `out/visualizations/quantum_phi4_dynamics.html`
- `out/visualizations/quantum_schwinger_dynamics.html`
- `out/visualizations/quantum_hubbard.html` — Hubbard Mott insulator (M27)
- `out/visualizations/quantum_floquet.html` — Floquet kicked-Ising (M27)

Each page includes animation controls (`Play`, `Pause`, timeline slider).
`quantum_spatial_wavepacket_barrier.html` shows a 1D spatial wavepacket evolution (`|psi(x,t)|^2`) solved by finite difference + Crank-Nicolson, with phase color, potential profile, and sampled position cloud.
`quantum_spatial3d_wavepacket_barrier_periodic.html` shows 3D evolution solved by split-step Fourier and can be explored in a true WebGL world view via the standalone player.
`quantum_hubbard.html` shows spin correlations, momentum distribution, and Mott-transition sweep charts.
`quantum_floquet.html` shows stroboscopic ⟨m_z⟩/entropy/energy dynamics, level-spacing histogram with Poisson/GUE reference lines, and ergodic/MBL phase badge.

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

The benchmark reports include:

- `platform_profile` and `perf_warning_policy`
- continuous-space performance metadata:
  - `grid`
  - `export_volume`
  - `fft_provider`
  - `avg_step_ms_mean`

CI runs default to the `github-windows` platform profile, while local runs default to `local`. Performance threshold violations remain visible as warnings, but only accuracy/stability thresholds block the gate.

For release work, keep the three-round collaboration convention in mind:

- use one dedicated `codex/*` worktree per round
- rebase onto the latest `master` before starting a new round if the base moved
- merge only after the round-specific smoke and benchmark commands pass

Local and CI runs may differ in how much they can verify in one pass:

- local runs are optimized for iteration speed and may skip heavier checks when the native DLL is not architecture-compatible
- CI keeps the same console output model, but smoke-suite performance warnings remain non-blocking while accuracy/stability thresholds stay strict

CI integration uses the same script:

```powershell
.\scripts\benchmark_all.ps1 -NoiseSweep -RunGates -EnforceGate -SkipBuild
```

- workflow file: `.github/workflows/benchmark-gate.yml`
- runs build + tests + unified benchmark gate on Windows
- uploads `benchmarks/reports/` as CI artifact

Minimal local and CI commands:

- local minimal verification:
  - run the `Build probe (no compile)` command in this section
  - `.\scripts\test.ps1 -NoPopup`
- CI-style minimal gate:
  - `.\scripts\benchmark_all.ps1 -NoiseSweep -RunGates -EnforceGate -SkipBuild -EnforceWorktreeGuard`

Pre-commit verification matrix:

```powershell
.\scripts\run_verification_matrix.ps1
```

This one-shot script runs the smoke suite, Python adapter checks, visualization smoke, and the selected benchmark gate in a single console-only flow.

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

Circuit capability and error contract:

```python
caps = e.circuit_capabilities()
print(caps["backend"], caps["supported_gates"], caps["supported_noise_models"])

bad = e.run_circuit({
    "qubits": 1,
    "operations": [
      {"name": "foo", "targets": [0]}
    ]
}, {"enable_checkpoint": False})

if not bad["ok"]:
  print(bad["error_code"])
  print(bad["error_message"])
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

The circuit API uses a machine-readable `error_code` plus a human-readable `error_message`. Downstream code should branch on `error_code` and display `error_message`; the legacy `error` field is kept only as a compatibility alias.

Minimal command-line repro for an unsupported gate:

```powershell
@'
{"qubits":1,"operations":[{"name":"foo","targets":[0]}]}
'@ | Set-Content $env:TEMP\qx_bad_circuit.json -Encoding utf8
.\build-vs\Debug\qx_cli.exe --circuit $env:TEMP\qx_bad_circuit.json --options .\examples\options_default.json
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
- on failure, `ok: false` plus `error_code` and `error_message` (the `error` alias remains for compatibility)

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

## 3b) Lattice physics modes (CLI)

### 1D Hubbard model (`--hubbard`)

Ground state and Mott-insulator characterization for the 1D Hubbard chain:

```powershell
# Single-point ground state at U/t = 4
.\build-vs\Debug\qx_cli.exe --hubbard .\examples\hubbard_ground_state.json

# U/t sweep (Mott transition)
.\build-vs\Debug\qx_cli.exe --hubbard .\examples\hubbard_sweep.json
```

Example `hubbard_ground_state.json`:

```json
{
  "n_sites": 6,
  "t_hop": 1.0,
  "U": 4.0,
  "scenario": "ground_state"
}
```

Example `hubbard_sweep.json`:

```json
{
  "n_sites": 6,
  "t_hop": 1.0,
  "U": 4.0,
  "scenario": "sweep",
  "n_sweep": 20,
  "U_min": 0.0,
  "U_max": 12.0
}
```

Key result fields:

- `ground_energy`: E₀
- `energy_gap`: Δ = E₁ − E₀
- `double_occupancy`: D = ⟨Σ n_{i↑}n_{i↓}⟩/N (→ 0 in Mott insulator)
- `spin_structure_pi`: S(k=π) antiferromagnetic structure factor
- `spin_corr`: ⟨S⁰ᶻSʳᶻ⟩ for r = 0…N−1
- `momentum_dist`: n(k) for k = 0…N−1
- `sweep`: array of `{ U, ground_energy, energy_gap, double_occupancy, spin_structure_pi }`

Limits: `n_sites` ∈ [2, 14]; `n_up/n_dn` ∈ [0, n_sites] (default: half-filling).

### Floquet kicked-Ising model (`--floquet`)

Stroboscopic dynamics and quasi-energy spectrum for the periodically driven Ising chain:

```powershell
# Stroboscopic dynamics
.\build-vs\Debug\qx_cli.exe --floquet .\examples\floquet_dynamics.json

# Quasi-energy spectrum scan
.\build-vs\Debug\qx_cli.exe --floquet .\examples\floquet_spectrum.json
```

Example `floquet_dynamics.json`:

```json
{
  "n_sites": 8,
  "J": 1.0,
  "h_z": 0.809,
  "drive_amp": 1.5,
  "drive_omega": 4.5,
  "n_periods": 40,
  "scenario": "dynamics"
}
```

Example `floquet_spectrum.json`:

```json
{
  "n_sites": 6,
  "J": 1.0,
  "h_z": 0.809,
  "drive_amp": 1.5,
  "drive_omega": 4.5,
  "scenario": "spectrum",
  "n_scan": 20,
  "scan_amp_max": 3.0
}
```

Key result fields:

- `frames`: stroboscopic snapshots `{ period, time, magnetization_z, energy, entropy }`
- `quasi_energies`: sorted quasi-energies θ_j ∈ [−π/T, π/T)
- `mean_gap_ratio`: level-spacing ratio r (Poisson ≈ 0.39 → MBL/integrable; GUE ≈ 0.60 → ergodic/chaotic)
- `scan_amps` / `scan_qe`: amplitude scan arrays for spectrum scenario

Limits: `n_sites` ∈ [2, 12]; `drive_omega` must be > 0; full diagonalization only when dim ≤ 4096 (N ≤ 12).

### Visualization: Hubbard and Floquet pages

```powershell
# Hubbard page (4-panel: spin correlations, n(k), D vs U/t, S(π) vs U/t)
python .\scripts\visualize_simulation.py --scenario hubbard --hubbard-sites 6 --hubbard-U 4.0

# Floquet page (4-panel: ⟨m_z⟩, entropy, energy, level-spacing histogram)
python .\scripts\visualize_simulation.py --scenario floquet --floquet-sites 8 --floquet-amp 1.5

# Both included in --scenario all dashboard
python .\scripts\visualize_simulation.py --scenario all
```

Additional Hubbard knobs:

- `--hubbard-sites N` (2–12, default 6)
- `--hubbard-t FLOAT` hopping amplitude (default 1.0)
- `--hubbard-U FLOAT` on-site repulsion (default 4.0)
- `--hubbard-n-up / --hubbard-n-dn` electron numbers (default: half-filling)
- `--hubbard-sweep-points N` points in U/t sweep (default 20)
- `--hubbard-U-max FLOAT` (default 12.0)

Additional Floquet knobs:

- `--floquet-sites N` (2–8, default 8)
- `--floquet-J FLOAT` Ising coupling (default 1.0)
- `--floquet-hz FLOAT` longitudinal field (default 0.809)
- `--floquet-amp FLOAT` kick amplitude h_x (default 1.5)
- `--floquet-omega FLOAT` drive frequency ω (default 4.5)
- `--floquet-periods N` stroboscopic steps (default 40)
- `--floquet-scenario dynamics|spectrum`

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
