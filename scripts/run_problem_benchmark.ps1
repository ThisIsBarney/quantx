param(
  [string]$Suite = "benchmarks/suites/continuous2d_smoke.json",
  [string]$PlatformProfile = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\ensure_subagent_worktree.ps1

if (-not [string]::IsNullOrWhiteSpace($PlatformProfile)) {
  $env:QX_BENCHMARK_PLATFORM_PROFILE = $PlatformProfile
}

# Keep the legacy continuous-space entrypoint, but reuse the unified dispatcher.
.\scripts\run_benchmark.ps1 -Suite $Suite -PlatformProfile $PlatformProfile
