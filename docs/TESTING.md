# Testing Notes

## Automated smoke coverage

- backend routing:
  - small qubit case -> `statevector`
  - large qubit case -> `mps`
- statevector physics sanity:
  - 1-qubit Rabi drive with `H = X0` at `t = pi/2`
    - `trotter` path reaches `population_1 ~= 1`
    - `krylov` path reaches `population_1 ~= 1`
  - 2-qubit XX coupling with `H = X0X1` at `t = pi/2`
    - reaches `population_last ~= 1` (`|11>`)
- circuit backend sanity:
  - Bell circuit (`H(0); CNOT(0,1)`) final state near `( |00> + |11> ) / sqrt(2)`
  - `RX(pi)` on `|0>` rotates near `|1>`
  - shot sampling returns expected Bell-support outcomes (`00`, `11`)
  - amplitude damping (`gamma=1`) drives `|1>` to `|0>`
- expectation estimation:
    - exact and shot estimates for `Z0`, `Z1`, `Z0Z1`, `X0X1`, `Y0Y1`, `X0Y1` on Bell circuit are consistent
    - exact and shot estimates for `Y0` on a prepared Y-eigenstate are consistent
    - per-observable stderr/variance and aggregate error summary are emitted
- norm conservation:
  - `|psi|^2` stays near 1.0 in smoke scenarios
- smoke diagnostics:
  - `qx_smoke` and `qx_fft_smoke` route CRT failures to stderr, so Windows debug runs stay console-only
  - `scripts/test.ps1 -NoPopup` enables a console-only native error mode before any child process starts
  - `scripts/run_cli.ps1 -NoPopup` uses the same console-only native error mode before build and CLI execution
  - `scripts/test.ps1` only runs the Python adapter and visualization smoke when it finds a compatible x64 native DLL; otherwise it prints a clear console skip message and continues with the lightweight schema smoke
- continuous export payload checks:
  - 2D export payloads are validated to match `nx * ny`
  - 3D export payloads are validated for `xy`, `xz`, `yz` slice shapes and `[x,y,z,density]` point quads
  - the schema smoke in `scripts/test.ps1` validates the append-only wrapper fields `schema_version`, `export_config`, and `frames`

## Workflow Checks

- Task scripts warn when launched from `master` or `main`.
- `scripts/ensure_subagent_worktree.ps1` supports three guard modes:
  - `warn` prints a warning and continues
  - `enforce` throws on protected branches
  - `silent` suppresses guard output, which is the default in CI
- Start each new agent task from a dedicated `codex/*` worktree.
- Failures must surface in the console; do not use debugger popups or modal error dialogs.
- See `docs/ENGINEERING_STANDARDS.md` for the full workflow and quality checklist.

## Script Summary Format

`scripts/test.ps1` writes one machine-parseable record per step and one final summary line:

- `QX_TEST_STEP|{...}`
- `QX_TEST_SUMMARY|{...}`

Each step record includes:

- `name`
- `status` (`pass`, `skip`, or `fail`)
- `reason`
- `duration_ms`
- `details`

The summary record includes:

- `pass`
- `skip`
- `fail`
- `total`
- `no_popup`
- `worktree_guard`
- `failed_step`

## Benchmark coverage

- unified benchmark gate:
  - `.\scripts\benchmark_all.ps1 -NoiseSweep -RunGates -EnforceGate`
  - covers `circuit_accuracy_smoke`, `circuit_noise_sweep`, `continuous2d_smoke`, `continuous3d_smoke`, and `continuous3d_accuracy`
- benchmark-only refresh:
  - `.\scripts\benchmark_all.ps1 -NoiseSweep`
- platform profiles:
  - local runs default to the `local` profile
  - GitHub Actions Windows runs default to `github-windows`
  - performance thresholds are emitted as warnings, while accuracy/stability checks remain blocking
- benchmark reporting smoke:
  - `python .\scripts\test_benchmark_reporting.py`
  - verifies platform profile detection, continuous performance metadata, gate classification, and backward-compatible frame extraction

## Schema Compatibility

- `continuous_export` should be treated as append-only by readers.
- strict consumers must validate `schema_version` and the documented wrapper fields before using the payload.
- non-strict consumers should use fallback behavior instead of assuming the engine export is always available.
- `scripts/test.ps1` includes a lightweight schema smoke so regressions are caught before the heavier visualization smoke test.

The benchmark reports and trend files now include:

- `platform_profile` and `perf_warning_policy`
- continuous-space metadata such as `grid`, `export_volume`, `fft_provider`, and `avg_step_ms_mean`
- a trend label that explicitly marks suites whose performance warnings are non-blocking
- pre-commit verification matrix:
  - `.\scripts\run_verification_matrix.ps1`
  - runs smoke, adapter, visualization, and the selected benchmark gate in one console-only pass
  - `.\scripts\test_worktree_guard.ps1`
    - validates the worktree guard's silent, enforce, and protected-branch warn paths

## How to run

```powershell
.\scripts\test.ps1
.\scripts\test.ps1 -NoPopup
```

CLI sanity:

```powershell
.\scripts\run_cli.ps1
.\scripts\run_cli.ps1 -NoPopup
```

CLI argument contract sanity:

```powershell
.\scripts\test_cli_args.ps1
```

Dependency license policy check:

```powershell
.\scripts\check_dependency_policy.ps1
```

Smoke binaries:

```powershell
.\build-vs\Debug\qx_smoke.exe
.\build-vs\Debug\qx_fft_smoke.exe
```

These binaries should fail in the terminal with `FAIL:` or `EXCEPTION:` output instead of opening debugger popups.

Visualization smoke test:

```powershell
python .\scripts\test_visualization.py
```
