param(
  [string]$Suite = "benchmarks/suites/circuit_accuracy_smoke.json",
  [switch]$SkipBuild,
  [switch]$PerfWarnOnly,
  [string]$PlatformProfile = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\ensure_subagent_worktree.ps1

if (-not [string]::IsNullOrWhiteSpace($PlatformProfile)) {
  $env:QX_BENCHMARK_PLATFORM_PROFILE = $PlatformProfile
}

# Reuse the shared benchmark runner and let callers suppress the extra build in batch mode.
$benchmarkArgs = @{
  Suite = $Suite
}
if ($SkipBuild) {
  $benchmarkArgs.SkipBuild = $true
}
if (-not [string]::IsNullOrWhiteSpace($PlatformProfile)) {
  $benchmarkArgs.PlatformProfile = $PlatformProfile
}
.\scripts\run_benchmark.ps1 @benchmarkArgs
if ($LASTEXITCODE -ne 0) { throw "benchmark run failed: exit code $LASTEXITCODE" }

# Forward the caller's gate mode so the same script can be used in CI and local smoke runs.
$gateArgs = @{
  Suite = $Suite
}
if ($PerfWarnOnly) {
  $gateArgs.PerfWarnOnly = $true
}
if (-not [string]::IsNullOrWhiteSpace($PlatformProfile)) {
  $gateArgs.PlatformProfile = $PlatformProfile
}
.\scripts\benchmark_gate.ps1 @gateArgs
if ($LASTEXITCODE -ne 0) { throw "benchmark gate failed: exit code $LASTEXITCODE" }
