param(
  [string]$Suite = "benchmarks/suites/circuit_accuracy_smoke.json"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python .\benchmarks\check_report_thresholds.py --suite $Suite
if ($LASTEXITCODE -ne 0) { throw "benchmark gate failed: exit code $LASTEXITCODE" }
