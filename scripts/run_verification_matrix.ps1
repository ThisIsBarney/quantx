$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Get-NativeDllCandidate {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Root
  )

  $current = Get-Item -LiteralPath $Root
  while ($null -ne $current) {
    $candidate = Join-Path $current.FullName "build-vs\Debug\quantx.dll"
    if (Test-Path -LiteralPath $candidate) {
      return $candidate
    }
    $current = $current.Parent
  }

  return $null
}

function Test-X64PeFile {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  try {
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -lt 0x40) {
      return $false
    }
    if ($bytes[0] -ne 0x4D -or $bytes[1] -ne 0x5A) {
      return $false
    }
    $peOffset = [BitConverter]::ToInt32($bytes, 0x3C)
    if ($peOffset -lt 0 -or ($peOffset + 6) -gt $bytes.Length) {
      return $false
    }
    $machine = [BitConverter]::ToUInt16($bytes, $peOffset + 4)
    return ($machine -eq 0x8664)
  } catch {
    return $false
  }
}

function Invoke-VerificationStage {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [Parameter(Mandatory = $true)]
    [scriptblock]$Action,
    [Parameter(Mandatory = $true)]
    [string]$FailureHint
  )

  Write-Host ("==> starting stage: {0}" -f $Name)
  try {
    & $Action
    Write-Host ("==> finished stage: {0}" -f $Name)
  } catch {
    Write-Host ("==> failed stage: {0}" -f $Name)
    Write-Host ("hint: {0}" -f $FailureHint)
    throw
  }
}

Invoke-VerificationStage -Name "worktree guard" -FailureHint "verify that the task runs from a codex/* worktree based on the latest master" -Action {
  .\scripts\ensure_subagent_worktree.ps1 -Enforce
}

Invoke-VerificationStage -Name "smoke and adapter checks" -FailureHint "rerun .\scripts\test.ps1 -EnforceWorktreeGuard to see the console-only smoke output" -Action {
  .\scripts\test.ps1 -EnforceWorktreeGuard
  if ($LASTEXITCODE -ne 0) { throw "smoke verification failed: exit code $LASTEXITCODE" }
}

$nativeDll = Get-NativeDllCandidate -Root $root
if ($null -eq $nativeDll) {
  Write-Host "==> skipped stage: benchmark gate"
  Write-Host "hint: build a native quantx.dll first, then rerun the matrix to enable benchmark coverage"
} elseif (-not (Test-X64PeFile -Path $nativeDll)) {
  Write-Host "==> skipped stage: benchmark gate"
  Write-Host ("hint: found a native DLL, but it is not x64-compatible: {0}" -f $nativeDll)
} else {
  Invoke-VerificationStage -Name "benchmark gate" -FailureHint "inspect the latest file in benchmarks/reports/ and compare the suite gate output" -Action {
    .\scripts\benchmark_all.ps1 -NoiseSweep -RunGates -EnforceGate -SkipBuild -EnforceWorktreeGuard
    if ($LASTEXITCODE -ne 0) { throw "benchmark verification failed: exit code $LASTEXITCODE" }
  }
}

Write-Host "verification matrix completed"
