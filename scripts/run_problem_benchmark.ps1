param(
  [string]$Suite = "benchmarks/suites/continuous2d_smoke.json",
  [switch]$SkipBuild,
  [switch]$EnforceWorktreeGuard
  [string]$PlatformProfile = ""
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

# Keep the legacy continuous-space entrypoint, but reuse the unified dispatcher.
$runArgs = @{
  Suite = $Suite
  SkipBuild = $SkipBuild
  PlatformProfile = $PlatformProfile
}
if ($EnforceWorktreeGuard) {
  $runArgs.EnforceWorktreeGuard = $true
}
.\scripts\run_benchmark.ps1 @runArgs
