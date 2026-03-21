param(
  [switch]$EnforceWorktreeGuard
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($EnforceWorktreeGuard) {
  .\scripts\ensure_subagent_worktree.ps1 -Enforce
} else {
  .\scripts\ensure_subagent_worktree.ps1
}

.\scripts\build.ps1 -EnforceWorktreeGuard:$EnforceWorktreeGuard

& .\build-vs\Debug\qx_cli.exe --problem .\examples\problem_mps_tebd.json --options .\examples\options_default.json
if ($LASTEXITCODE -ne 0) { throw "qx_cli mps tebd failed: exit code $LASTEXITCODE" }

& .\build-vs\Debug\qx_cli.exe --problem .\examples\problem_mps_tebd.json --options .\examples\options_default.json --explain
if ($LASTEXITCODE -ne 0) { throw "qx_cli mps explain failed: exit code $LASTEXITCODE" }
