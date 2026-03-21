<#
.SYNOPSIS
  Render QuantX visualization pages from the command line.

.DESCRIPTION
  Generates HTML visualizations for spatial and circuit scenarios. Use
  -ContinuousExportJson when you already have an engine-export payload and want
  to render it directly without re-running the solver.

.PARAMETER Scenario
  Which visualization family to render.

.PARAMETER SolverMode
  Selects how continuous-space scenes source their data.

.PARAMETER ContinuousExportJson
  Optional JSON file produced by the native exporter.

.PARAMETER StrictExportSchema
  Fail fast when the export schema version is not compatible with this viewer.

.EXAMPLE
  .\scripts\run_visualization.ps1 -Scenario spatial2d -SolverMode python

.EXAMPLE
  .\scripts\run_visualization.ps1 -Scenario spatial2d -SolverMode engine -ContinuousExportJson out/export.json -StrictExportSchema

.EXAMPLE
  .\scripts\run_visualization.ps1 -Scenario spatial3d -SolverMode python -ContinuousExportJson out/export.json
#>
param(
  [ValidateSet("rabi", "mps", "spatial", "spatial2d", "spatial3d", "all")]
  [string]$Scenario = "all",
  [ValidateSet("auto", "engine", "python")]
  [string]$SolverMode = "auto",
  [string]$ContinuousExportJson = "",
  [switch]$StrictExportSchema,
  [switch]$EnforceWorktreeGuard
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($ContinuousExportJson -and (-not (Test-Path $ContinuousExportJson))) {
  Write-Error @"
continuous export json file not found: $ContinuousExportJson
hint: pass an existing file path, or omit -ContinuousExportJson to let QuantX solve live
example: .\scripts\run_visualization.ps1 -Scenario spatial2d -SolverMode python
"@
  exit 1
}

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

function Needs-NativeRuntime {
  param(
    [string]$SelectedScenario,
    [string]$SelectedSolverMode,
    [string]$ExportJson
  )

  if ($SelectedScenario -in @("rabi", "mps")) {
    return $true
  }
  if ($SelectedSolverMode -eq "engine" -and -not $ExportJson) {
    return $true
  }
  return $false
}

if ((Needs-NativeRuntime -SelectedScenario $Scenario -SelectedSolverMode $SolverMode -ExportJson $ContinuousExportJson) -and (-not (Get-NativeDllPath -StartDir $root))) {
  .\scripts\build.ps1 -EnforceWorktreeGuard:$EnforceWorktreeGuard
}

# Keep the wrapper ergonomic while exposing the continuous solver mode.
$arguments = @(
  ".\scripts\visualize_simulation.py"
  "--scenario"
  $Scenario
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
if ($StrictExportSchema) {
  $arguments += "--strict-export-schema"
}
python @arguments
if ($LASTEXITCODE -ne 0) { throw "visualization generation failed: exit code $LASTEXITCODE" }
