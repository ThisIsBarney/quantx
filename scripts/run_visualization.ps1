$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\build.ps1

python .\scripts\visualize_simulation.py --scenario all --spatial-potential barrier --out-dir out/visualizations
if ($LASTEXITCODE -ne 0) { throw "visualization generation failed: exit code $LASTEXITCODE" }
