$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python .\scripts\check_dependency_policy.py
if ($LASTEXITCODE -ne 0) { throw "dependency policy check failed: exit code $LASTEXITCODE" }
