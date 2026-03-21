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

$potentials = @("free", "barrier", "harmonic", "well")
foreach ($p in $potentials) {
  python .\scripts\visualize_simulation.py --scenario spatial --spatial-potential $p --out-dir out/visualizations
  if ($LASTEXITCODE -ne 0) { throw "spatial visualization ($p) failed: exit code $LASTEXITCODE" }
  python .\scripts\visualize_simulation.py --scenario spatial2d --spatial2d-potential $p --out-dir out/visualizations
  if ($LASTEXITCODE -ne 0) { throw "spatial2d visualization ($p) failed: exit code $LASTEXITCODE" }
}
