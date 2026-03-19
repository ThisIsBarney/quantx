$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\ensure_subagent_worktree.ps1

if (-not (Test-Path "build-vs")) {
  cmake -S . -B build-vs -G "Visual Studio 18 2026" -A x64
  if ($LASTEXITCODE -ne 0) { throw "cmake configure failed: exit code $LASTEXITCODE" }
}

cmake --build build-vs --config Debug
if ($LASTEXITCODE -ne 0) { throw "cmake build failed: exit code $LASTEXITCODE" }
Write-Host "Build completed: build-vs/Debug"
