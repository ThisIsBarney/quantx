param()

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Capture the worktree guard output for a single mode so the smoke stays lightweight.
function Invoke-GuardScript {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Mode
  )

  return (& {
    .\scripts\ensure_subagent_worktree.ps1 -Mode $Mode
  } 3>&1 2>&1 | Out-String)
}

# Replace git just long enough to make the warn path behave like a protected branch.
function Invoke-MockedWarnPath {
  <#
    Executes the guard script with a temporary git shim that reports master so
    the warning branch can be verified without touching the current worktree.
  #>
  $realGit = (Get-Command git -CommandType Application).Source

  function git {
    param(
      [Parameter(ValueFromRemainingArguments = $true)]
      [string[]]$Args
    )

    if ($Args -contains "rev-parse" -and $Args -contains "--abbrev-ref" -and $Args -contains "HEAD") {
      Write-Output "master"
      return
    }

    & $script:RealGitPath @Args
  }

  try {
    $script:RealGitPath = $realGit
    return Invoke-GuardScript -Mode warn
  } finally {
    Remove-Item Function:git -Force -ErrorAction SilentlyContinue
    Remove-Variable -Name RealGitPath -Scope Script -ErrorAction SilentlyContinue
  }
}

$silentOutput = Invoke-GuardScript -Mode silent
if (-not [string]::IsNullOrWhiteSpace($silentOutput)) {
  throw "worktree guard smoke failed: silent mode produced output: $silentOutput"
}

$enforceOutput = Invoke-GuardScript -Mode enforce
if (-not [string]::IsNullOrWhiteSpace($enforceOutput)) {
  throw "worktree guard smoke failed: enforce mode produced output on the current codex worktree: $enforceOutput"
}

$warnOutput = Invoke-MockedWarnPath
if ($warnOutput -notmatch "QuantX task scripts should run from a dedicated codex/\* worktree") {
  throw "worktree guard smoke failed: warn mode did not report the protected-branch warning: $warnOutput"
}

Write-Host "worktree guard smoke passed"
