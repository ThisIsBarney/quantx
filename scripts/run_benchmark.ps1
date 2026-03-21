param(
  [string]$Suite = "benchmarks/suites/circuit_accuracy_smoke.json",
  [switch]$SkipBuild,
  [string]$PlatformProfile = "",
  [switch]$EnforceWorktreeGuard
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($EnforceWorktreeGuard) {
  .\scripts\ensure_subagent_worktree.ps1 -Enforce
} else {
  .\scripts\ensure_subagent_worktree.ps1
}

if (-not [string]::IsNullOrWhiteSpace($PlatformProfile)) {
  $env:QX_BENCHMARK_PLATFORM_PROFILE = $PlatformProfile
}

if (-not $SkipBuild) {
  .\scripts\build.ps1 -EnforceWorktreeGuard:$EnforceWorktreeGuard
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
