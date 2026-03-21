param(
  [string]$Suite = "benchmarks/suites/continuous2d_smoke.json",
  [switch]$SkipBuild,
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

# Keep the legacy continuous-space entrypoint, but reuse the unified dispatcher.
if ($EnforceWorktreeGuard) {
  .\scripts\run_benchmark.ps1 -Suite $Suite -SkipBuild:$SkipBuild -EnforceWorktreeGuard
} else {
  .\scripts\run_benchmark.ps1 -Suite $Suite -SkipBuild:$SkipBuild
}
