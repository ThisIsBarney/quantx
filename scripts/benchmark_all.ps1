param(
  [switch]$NoiseSweep
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\run_benchmark.ps1 -Suite benchmarks/suites/circuit_accuracy_smoke.json
if ($LASTEXITCODE -ne 0) { throw "accuracy benchmark failed: exit code $LASTEXITCODE" }

if ($NoiseSweep) {
  .\scripts\run_benchmark.ps1 -Suite benchmarks/suites/circuit_noise_sweep.json
  if ($LASTEXITCODE -ne 0) { throw "noise-sweep benchmark failed: exit code $LASTEXITCODE" }
}

python .\benchmarks\summarize_reports.py
if ($LASTEXITCODE -ne 0) { throw "benchmark summarize failed: exit code $LASTEXITCODE" }
