param(
  [string]$Suite = "benchmarks/suites/circuit_accuracy_smoke.json",
  [switch]$PerfWarnOnly,
  [string]$PlatformProfile = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not [string]::IsNullOrWhiteSpace($PlatformProfile)) {
  $env:QX_BENCHMARK_PLATFORM_PROFILE = $PlatformProfile
}

$pythonArgs = @(".\benchmarks\check_report_thresholds.py", "--suite", $Suite)
if (-not [string]::IsNullOrWhiteSpace($env:QX_BENCHMARK_PLATFORM_PROFILE)) {
  $pythonArgs += @("--platform-profile", $env:QX_BENCHMARK_PLATFORM_PROFILE)
}
if ($PerfWarnOnly) {
  $pythonArgs += "--perf-warn-only"
  python @pythonArgs
} else {
  python @pythonArgs
}
if ($LASTEXITCODE -ne 0) { throw "benchmark gate failed: exit code $LASTEXITCODE" }
