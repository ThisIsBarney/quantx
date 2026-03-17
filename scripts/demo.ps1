$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\build.ps1
$env:PYTHONPATH = Join-Path $root "python"
python .\examples\python_sdk_demo.py
if ($LASTEXITCODE -ne 0) { throw "python demo failed: exit code $LASTEXITCODE" }
