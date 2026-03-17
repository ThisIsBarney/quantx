param(
  [switch]$NoiseSweep,
  [switch]$EnforceGate
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($EnforceGate) {
  .\scripts\run_benchmark_and_gate.ps1 -Suite benchmarks/suites/circuit_accuracy_smoke.json
  if ($LASTEXITCODE -ne 0) { throw "accuracy gate failed: exit code $LASTEXITCODE" }
} else {
  .\scripts\run_benchmark.ps1 -Suite benchmarks/suites/circuit_accuracy_smoke.json
  if ($LASTEXITCODE -ne 0) { throw "accuracy benchmark failed: exit code $LASTEXITCODE" }
}

if ($NoiseSweep) {
  if ($EnforceGate) {
    .\scripts\run_benchmark_and_gate.ps1 -Suite benchmarks/suites/circuit_noise_sweep.json
    if ($LASTEXITCODE -ne 0) { throw "noise-sweep gate failed: exit code $LASTEXITCODE" }
  } else {
    .\scripts\run_benchmark.ps1 -Suite benchmarks/suites/circuit_noise_sweep.json
    if ($LASTEXITCODE -ne 0) { throw "noise-sweep benchmark failed: exit code $LASTEXITCODE" }
  }
}

python .\benchmarks\summarize_reports.py
if ($LASTEXITCODE -ne 0) { throw "benchmark summarize failed: exit code $LASTEXITCODE" }

python .\benchmarks\compare_latest_reports.py --suite circuit_accuracy_smoke
if ($LASTEXITCODE -ne 0) { throw "benchmark trend failed for circuit_accuracy_smoke: exit code $LASTEXITCODE" }

if ($NoiseSweep) {
  python .\benchmarks\compare_latest_reports.py --suite circuit_noise_sweep
  if ($LASTEXITCODE -ne 0) { throw "benchmark trend failed for circuit_noise_sweep: exit code $LASTEXITCODE" }
}
