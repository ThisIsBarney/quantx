param(
  [switch]$NoiseSweep,
  [switch]$RunGates,
  [switch]$EnforceGate,
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

 # Resolve the effective platform profile for benchmark metadata and report labeling.
function Get-BenchmarkPlatformProfile {
  param([string]$ExplicitProfile)
  if (-not [string]::IsNullOrWhiteSpace($ExplicitProfile)) {
    return $ExplicitProfile
  }
  if (-not [string]::IsNullOrWhiteSpace($env:QX_BENCHMARK_PLATFORM_PROFILE)) {
    return $env:QX_BENCHMARK_PLATFORM_PROFILE
  }
  if (-not [string]::IsNullOrWhiteSpace($env:GITHUB_ACTIONS) -or -not [string]::IsNullOrWhiteSpace($env:CI)) {
    return "github-windows"
  }
  return "local"
}

$platformProfile = Get-BenchmarkPlatformProfile -ExplicitProfile $PlatformProfile
$env:QX_BENCHMARK_PLATFORM_PROFILE = $platformProfile

if (-not $SkipBuild) {
  # Build once, then fan out over every suite so the benchmark entry flow stays consistent.
  .\scripts\build.ps1 -EnforceWorktreeGuard:$EnforceWorktreeGuard
  if ($LASTEXITCODE -ne 0) { throw "build failed: exit code $LASTEXITCODE" }
}

# Return the canonical benchmark suite order so benchmark, gate, and trend generation stay aligned.
function Get-BenchmarkSuites {
  param([switch]$IncludeNoiseSweep)
  $suites = @(
    "benchmarks/suites/circuit_accuracy_smoke.json",
    "benchmarks/suites/continuous2d_smoke.json",
    "benchmarks/suites/continuous3d_smoke.json",
    "benchmarks/suites/continuous3d_accuracy.json"
  )
  if ($IncludeNoiseSweep) {
    $suites += "benchmarks/suites/circuit_noise_sweep.json"
  }
  return $suites
}

$suites = Get-BenchmarkSuites -IncludeNoiseSweep:$NoiseSweep

foreach ($suite in $suites) {
  if ($RunGates) {
    $gateArgs = @{
      Suite = $suite
      SkipBuild = $true
      PlatformProfile = $platformProfile
    }
    if ($EnforceWorktreeGuard) {
      $gateArgs.EnforceWorktreeGuard = $true
    }
    .\scripts\run_benchmark_and_gate.ps1 @gateArgs
    if ($LASTEXITCODE -ne 0) { throw "benchmark gate failed for ${suite}: exit code $LASTEXITCODE" }
  } else {
    $runArgs = @{
      Suite = $suite
      SkipBuild = $true
      PlatformProfile = $platformProfile
    }
    if ($EnforceWorktreeGuard) {
      $runArgs.EnforceWorktreeGuard = $true
    }
    .\scripts\run_benchmark.ps1 @runArgs
    if ($LASTEXITCODE -ne 0) { throw "benchmark failed for ${suite}: exit code $LASTEXITCODE" }
  }
}

python .\benchmarks\summarize_reports.py
if ($LASTEXITCODE -ne 0) { throw "benchmark summarize failed: exit code $LASTEXITCODE" }

foreach ($suite in $suites) {
  $suiteName = [System.IO.Path]::GetFileNameWithoutExtension($suite)
  python .\benchmarks\compare_latest_reports.py --suite $suiteName
  if ($LASTEXITCODE -ne 0) { throw "benchmark trend failed for ${suiteName}: exit code $LASTEXITCODE" }
}
