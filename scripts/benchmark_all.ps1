param(
  [switch]$NoiseSweep,
  [switch]$RunGates,
  [switch]$EnforceGate,
  [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\ensure_subagent_worktree.ps1

if (-not $SkipBuild) {
  # Build once, then fan out over every suite so the benchmark entry flow stays consistent.
  .\scripts\build.ps1
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

# Mark the smoke suites that should default to perf-warn-only gating unless the caller forces strict mode.
function Use-PerfWarnOnly {
  param([string]$SuiteName)
  if ($EnforceGate) {
    return $false
  }
  return $SuiteName -in @(
    "benchmarks/suites/circuit_noise_sweep.json",
    "benchmarks/suites/continuous2d_smoke.json",
    "benchmarks/suites/continuous3d_smoke.json"
  )
}

$suites = Get-BenchmarkSuites -IncludeNoiseSweep:$NoiseSweep

foreach ($suite in $suites) {
  if ($RunGates) {
    $gateArgs = @{
      Suite = $suite
      SkipBuild = $true
    }
    if (Use-PerfWarnOnly -SuiteName $suite) {
      $gateArgs.PerfWarnOnly = $true
    }
    .\scripts\run_benchmark_and_gate.ps1 @gateArgs
    if ($LASTEXITCODE -ne 0) { throw "benchmark gate failed for ${suite}: exit code $LASTEXITCODE" }
  } else {
    .\scripts\run_benchmark.ps1 -Suite $suite -SkipBuild
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
