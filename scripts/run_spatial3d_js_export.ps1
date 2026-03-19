$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$outDir = Join-Path $root "out/visualizations"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

python .\scripts\visualize_simulation.py `
  --scenario spatial3d `
  --spatial3d-potential barrier `
  --spatial3d-boundary periodic `
  --spatial3d-export-json `
  --out-dir out/visualizations
if ($LASTEXITCODE -ne 0) { throw "spatial3d export failed: exit code $LASTEXITCODE" }

Copy-Item .\scripts\spatial3d_player.html (Join-Path $outDir "spatial3d_player.html") -Force
Write-Host "wrote JS player bundle to: $outDir"
