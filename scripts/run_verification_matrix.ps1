$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Pre-commit verification should be strict about the branch/worktree location.
.\scripts\ensure_subagent_worktree.ps1 -Enforce

.\scripts\test.ps1 -EnforceWorktreeGuard
if ($LASTEXITCODE -ne 0) { throw "smoke verification failed: exit code $LASTEXITCODE" }

.\scripts\benchmark_all.ps1 -NoiseSweep -RunGates -EnforceGate -SkipBuild -EnforceWorktreeGuard
if ($LASTEXITCODE -ne 0) { throw "benchmark verification failed: exit code $LASTEXITCODE" }

Write-Host "verification matrix completed"
