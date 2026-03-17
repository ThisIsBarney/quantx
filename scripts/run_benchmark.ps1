param(
  [string]$Suite = "benchmarks/suites/circuit_accuracy_smoke.json"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\build.ps1
python .\benchmarks\run_circuit_benchmark.py --suite $Suite
if ($LASTEXITCODE -ne 0) { throw "benchmark runner failed: exit code $LASTEXITCODE" }
