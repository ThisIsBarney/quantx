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
- continuous export payload checks:
  - 2D export payloads are validated to match `nx * ny`
  - 3D export payloads are validated for `xy`, `xz`, `yz` slice shapes and `[x,y,z,density]` point quads

## Workflow Checks

- Task scripts warn when launched from `master` or `main`.
- Start each new agent task from a dedicated `codex/*` worktree.
- Failures must surface in the console; do not use debugger popups or modal error dialogs.
- See `docs/ENGINEERING_STANDARDS.md` for the full workflow and quality checklist.

## Benchmark coverage

- unified benchmark gate:
  - `.\scripts\benchmark_all.ps1 -NoiseSweep -RunGates -EnforceGate`
  - covers `circuit_accuracy_smoke`, `circuit_noise_sweep`, `continuous2d_smoke`, `continuous3d_smoke`, and `continuous3d_accuracy`
- benchmark-only refresh:
  - `.\scripts\benchmark_all.ps1 -NoiseSweep`

## How to run

```powershell
.\scripts\test.ps1
```

CLI sanity:

```powershell
.\scripts\run_cli.ps1
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
