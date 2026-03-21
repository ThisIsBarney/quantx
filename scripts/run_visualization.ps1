param(
  [ValidateSet("auto", "engine", "python")]
  [string]$SolverMode = "auto",
  [string]$ContinuousExportJson = "",
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

function Get-NativeDllPath {
  param([string]$StartDir)
  $current = Get-Item $StartDir
  while ($null -ne $current) {
    $candidate = Join-Path $current.FullName "build-vs\Debug\quantx.dll"
    if (Test-Path $candidate) {
      return $candidate
    }
    $parent = $current.Parent
    if ($null -eq $parent) {
      break
    }
    $current = $parent
  }
  return $null
}

if (-not (Get-NativeDllPath -StartDir $root)) {
  .\scripts\build.ps1 -EnforceWorktreeGuard:$EnforceWorktreeGuard
}

# Keep the wrapper ergonomic while exposing the continuous solver mode.
$arguments = @(
  ".\scripts\visualize_simulation.py"
  "--scenario"
  "all"
  "--spatial-potential"
  "barrier"
  "--out-dir"
  "out/visualizations"
  "--solver-mode"
  $SolverMode
)
if ($ContinuousExportJson) {
  $arguments += @("--continuous-export-json", $ContinuousExportJson)
}
python @arguments
if ($LASTEXITCODE -ne 0) { throw "visualization generation failed: exit code $LASTEXITCODE" }
