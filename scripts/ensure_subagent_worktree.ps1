param(
  [switch]$Enforce
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# CI jobs run on shared default branches; skip warning noise unless explicit enforcement is requested.
$isCI = $env:CI -eq "true" -or $env:GITHUB_ACTIONS -eq "true"
if ($isCI -and -not $Enforce) {
  return
}

# Warn when task scripts are run from a protected default branch instead of an isolated codex/* worktree.
try {
  $safeRoot = ($root -replace '\\', '/')
  $branch = (& git -c "safe.directory=$safeRoot" -C $root rev-parse --abbrev-ref HEAD 2>$null).Trim()
} catch {
  Write-Warning "QuantX worktree guard could not resolve the current git branch."
  return
}

if ([string]::IsNullOrWhiteSpace($branch) -or $branch -eq "HEAD" -or $branch -in @("master", "main")) {
  $message = "QuantX task scripts should run from a dedicated codex/* worktree; current branch is '$branch'."
  if ($Enforce) {
    throw $message
  }
  Write-Warning $message
}
