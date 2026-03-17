$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\build.ps1
ctest --test-dir build-vs -C Debug --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "ctest failed: exit code $LASTEXITCODE" }
.\build-vs\Debug\qx_smoke.exe
if ($LASTEXITCODE -ne 0) { throw "qx_smoke failed: exit code $LASTEXITCODE" }
