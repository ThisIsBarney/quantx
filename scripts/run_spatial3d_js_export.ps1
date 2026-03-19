param(
  [ValidateSet("auto", "engine", "python")]
  [string]$SolverMode = "engine",
  [string]$ContinuousExportJson = "",
  [string]$OutDir = "out/visualizations"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\ensure_subagent_worktree.ps1

function Resolve-VisualizationOutDir {
  param(
    [string]$RootDir,
    [string]$RequestedOutDir
  )
  if ([System.IO.Path]::IsPathRooted($RequestedOutDir)) {
    return [System.IO.Path]::GetFullPath($RequestedOutDir)
  }
  return [System.IO.Path]::GetFullPath((Join-Path $RootDir $RequestedOutDir))
}

$outDir = Resolve-VisualizationOutDir -RootDir $root -RequestedOutDir $OutDir
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

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
  .\scripts\build.ps1
}

$arguments = @(
  ".\scripts\visualize_simulation.py"
  "--scenario"
  "spatial3d"
  "--solver-mode"
  $SolverMode
  "--spatial3d-potential"
  "barrier"
  "--spatial3d-boundary"
  "periodic"
  "--spatial3d-export-json"
  "--out-dir"
  $OutDir
)
if ($ContinuousExportJson) {
  $arguments += @("--continuous-export-json", $ContinuousExportJson)
}
python @arguments
if ($LASTEXITCODE -ne 0) { throw "spatial3d export failed: exit code $LASTEXITCODE" }

Copy-Item .\scripts\spatial3d_player.html (Join-Path $outDir "spatial3d_player.html") -Force
Copy-Item .\scripts\vendor (Join-Path $outDir "vendor") -Recurse -Force
Write-Host "wrote JS player bundle to: $outDir"
