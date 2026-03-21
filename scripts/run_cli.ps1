param(
  [switch]$EnforceWorktreeGuard,
  [switch]$NoPopup
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Configure the current process so child native failures stay on the console.
function Enable-ConsoleOnlyNativeFailures {
  <#
    Applies the Windows error-mode flags that suppress modal error dialogs for
    native child processes launched from this script.
  #>
  if (-not ("QxNativeErrorMode" -as [type])) {
    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class QxNativeErrorMode {
  [DllImport("kernel32.dll")]
  private static extern uint SetErrorMode(uint uMode);

  private const uint SEM_FAILCRITICALERRORS = 0x0001;
  private const uint SEM_NOGPFAULTERRORBOX = 0x0002;
  private const uint SEM_NOOPENFILEERRORBOX = 0x8000;

  public static void EnableConsoleOnly() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
  }
}
"@
  }

  [QxNativeErrorMode]::EnableConsoleOnly()
  $env:QX_NO_POPUP = "1"
}

# Run one CLI scenario and fail with a clear console message when it exits non-zero.
function Invoke-QxCliScenario {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Label,

    [Parameter(Mandatory = $true)]
    [string[]]$Arguments
  )

  & .\build-vs\Debug\qx_cli.exe @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "qx_cli $Label failed: exit code $LASTEXITCODE"
  }
}

if ($NoPopup) {
  Enable-ConsoleOnlyNativeFailures
}

if ($EnforceWorktreeGuard) {
  .\scripts\ensure_subagent_worktree.ps1 -Mode enforce
} else {
  .\scripts\ensure_subagent_worktree.ps1
}

.\scripts\build.ps1 -EnforceWorktreeGuard:$EnforceWorktreeGuard

Invoke-QxCliScenario -Label "rabi" -Arguments @("--problem", ".\examples\problem_rabi.json", "--options", ".\examples\options_default.json")
Invoke-QxCliScenario -Label "xx" -Arguments @("--problem", ".\examples\problem_xx.json", "--options", ".\examples\options_default.json")
Invoke-QxCliScenario -Label "xx explain" -Arguments @("--problem", ".\examples\problem_xx.json", "--options", ".\examples\options_default.json", "--explain")
Invoke-QxCliScenario -Label "mps tebd" -Arguments @("--problem", ".\examples\problem_mps_tebd.json", "--options", ".\examples\options_default.json")
Invoke-QxCliScenario -Label "mps explain" -Arguments @("--problem", ".\examples\problem_mps_tebd.json", "--options", ".\examples\options_default.json", "--explain")
Invoke-QxCliScenario -Label "circuit" -Arguments @("--circuit", ".\examples\circuit_bell.json", "--options", ".\examples\options_default.json")
Invoke-QxCliScenario -Label "noisy circuit" -Arguments @("--circuit", ".\examples\circuit_bell_noisy.json", "--options", ".\examples\options_default.json")
