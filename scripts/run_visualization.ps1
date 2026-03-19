param(
  [ValidateSet("auto", "engine", "python")]
  [string]$SolverMode = "auto",
  [string]$ContinuousExportJson = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\build.ps1

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
