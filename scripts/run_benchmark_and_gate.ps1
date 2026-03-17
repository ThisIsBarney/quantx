param(
  [string]$Suite = "benchmarks/suites/circuit_accuracy_smoke.json"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\run_benchmark.ps1 -Suite $Suite
if ($LASTEXITCODE -ne 0) { throw "benchmark run failed: exit code $LASTEXITCODE" }

.\scripts\benchmark_gate.ps1 -Suite $Suite
if ($LASTEXITCODE -ne 0) { throw "benchmark gate failed: exit code $LASTEXITCODE" }
