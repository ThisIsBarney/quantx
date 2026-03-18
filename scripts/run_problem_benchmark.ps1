param(
  [string]$Suite = "benchmarks/suites/continuous2d_smoke.json"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\build.ps1
python .\benchmarks\run_problem_benchmark.py --suite $Suite
if ($LASTEXITCODE -ne 0) { throw "problem benchmark runner failed: exit code $LASTEXITCODE" }
