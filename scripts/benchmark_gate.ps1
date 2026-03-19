param(
  [string]$Suite = "benchmarks/suites/circuit_accuracy_smoke.json",
  [switch]$PerfWarnOnly
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($PerfWarnOnly) {
  python .\benchmarks\check_report_thresholds.py --suite $Suite --perf-warn-only
} else {
  python .\benchmarks\check_report_thresholds.py --suite $Suite
}
if ($LASTEXITCODE -ne 0) { throw "benchmark gate failed: exit code $LASTEXITCODE" }
