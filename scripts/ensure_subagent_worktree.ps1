param(
  [ValidateSet("warn", "enforce", "silent")]
  [string]$Mode,
  [switch]$Enforce
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Resolve the effective guard mode while keeping the legacy -Enforce switch working.
$isCI = $env:CI -eq "true" -or $env:GITHUB_ACTIONS -eq "true"
if ($Enforce) {
  $Mode = "enforce"
} elseif ([string]::IsNullOrWhiteSpace($Mode)) {
  $Mode = if ($isCI) { "silent" } else { "warn" }
}

if ($Mode -eq "silent") {
  return
}

function Get-QuantXWorktreeBranch {
  <#
    Returns the current git branch name for the active worktree.
  #>
  param(
    [Parameter(Mandatory = $true)]
    [string]$RootPath
  )

  $safeRoot = ($RootPath -replace '\\', '/')
  return (& git -c "safe.directory=$safeRoot" -C $RootPath rev-parse --abbrev-ref HEAD 2>$null).Trim()
}

# Warn or fail when task scripts are run from a protected default branch instead of an isolated codex/* worktree.
try {
  $branch = Get-QuantXWorktreeBranch -RootPath $root
} catch {
  Write-Warning "QuantX worktree guard could not resolve the current git branch."
  return
}

if ([string]::IsNullOrWhiteSpace($branch) -or $branch -eq "HEAD" -or $branch -in @("master", "main")) {
  $message = "QuantX task scripts should run from a dedicated codex/* worktree; current branch is '$branch'."
  if ($Mode -eq "enforce") {
    throw $message
  }
  Write-Warning $message
}
