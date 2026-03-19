param(
  [string]$Suite = "benchmarks/suites/circuit_accuracy_smoke.json",
  [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $SkipBuild) {
  .\scripts\build.ps1
}

# Dispatch to the circuit or continuous-space benchmark runner from one entrypoint.
$suiteLower = $Suite.ToLowerInvariant()
$runner = if ($suiteLower -match "continuous") {
  ".\benchmarks\run_problem_benchmark.py"
} else {
  ".\benchmarks\run_circuit_benchmark.py"
}

python $runner --suite $Suite
if ($LASTEXITCODE -ne 0) { throw "benchmark runner failed: exit code $LASTEXITCODE" }
