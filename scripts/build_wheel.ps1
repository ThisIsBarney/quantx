# Build the quantx-engine Python wheel locally.
# Usage: .\scripts\build_wheel.ps1 [-Output dist]
param(
    [string]$Output = "python/dist"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== Building quantx-engine wheel ===" -ForegroundColor Cyan

Push-Location "$PSScriptRoot\..\python"
try {
    python -m pip install --upgrade build --quiet
    python -m build
    Write-Host "`nWheel built successfully. Artifacts in: $((Resolve-Path dist).Path)" -ForegroundColor Green
} finally {
    Pop-Location
}
