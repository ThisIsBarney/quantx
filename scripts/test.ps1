param(
  [switch]$EnforceWorktreeGuard,
  [switch]$NoPopup,
  [switch]$IncludeBuildProbe
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Keep native failures on the console when the caller requests a no-popup run.
function Enable-ConsoleOnlyNativeFailures {
  <#
    Configures the current PowerShell process so child native processes fail in
    the terminal instead of opening Windows error dialogs.
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

# Convert a record into a single-line JSON payload for machine parsing.
function ConvertTo-QxJsonLine {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Value
  )

  return ($Value | ConvertTo-Json -Compress -Depth 10)
}

# Emit a structured per-step record that downstream tools can parse reliably.
function Write-QxStepRecord {
  param(
    [Parameter(Mandatory = $true)]
    [hashtable]$Record
  )

  Write-Host ("QX_TEST_STEP|" + (ConvertTo-QxJsonLine -Value $Record))
}

# Record a step result and update the aggregate counters.
function Add-QxStepResult {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name,

    [Parameter(Mandatory = $true)]
    [ValidateSet("pass", "skip", "fail")]
    [string]$Status,

    [string]$Reason = "",
    [hashtable]$Details = @{},
    [double]$DurationMs = 0.0
  )

  $record = [ordered]@{
    name = $Name
    status = $Status
    reason = $Reason
    duration_ms = [math]::Round($DurationMs, 3)
    details = $Details
  }
  $script:QxStepResults.Add([pscustomobject]$record) | Out-Null
  Write-QxStepRecord -Record $record
}

# Run one test step, capture its result, and stop the suite on the first failure.
function Invoke-QxTestStep {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name,

    [Parameter(Mandatory = $true)]
    [scriptblock]$Action,

    [string]$SkipReason = "",
    [hashtable]$Details = @{}
  )

  if ($script:QxTestFailed) {
    Add-QxStepResult -Name $Name -Status "skip" -Reason "skipped after earlier failure" -Details $Details
    return
  }

  if (-not [string]::IsNullOrWhiteSpace($SkipReason)) {
    Add-QxStepResult -Name $Name -Status "skip" -Reason $SkipReason -Details $Details
    return
  }

  $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
  try {
    & $Action
    Add-QxStepResult -Name $Name -Status "pass" -Details $Details -DurationMs $stopwatch.Elapsed.TotalMilliseconds
  } catch {
    $script:QxTestFailed = $true
    Add-QxStepResult -Name $Name -Status "fail" -Reason $_.Exception.Message -Details $Details -DurationMs $stopwatch.Elapsed.TotalMilliseconds
  } finally {
    $stopwatch.Stop()
  }
}

# Locate the newest compatible native DLL before deciding whether to run the heavier Python smoke checks.
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
    $candidate = Join-Path $current.FullName "build\quantx.dll"
    if (Test-Path -LiteralPath $candidate) {
      return $candidate
    }
    $current = $current.Parent
  }

  return $null
}

# Check that a DLL is x64 so the Python runtime does not trip a loader mismatch.
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

if ($NoPopup) {
  Enable-ConsoleOnlyNativeFailures
}

if ($EnforceWorktreeGuard) {
  .\scripts\ensure_subagent_worktree.ps1 -Mode enforce
} else {
  .\scripts\ensure_subagent_worktree.ps1
}

$script:QxStepResults = New-Object System.Collections.Generic.List[object]
$script:QxTestFailed = $false

$nativeDll = Get-NativeDllCandidate -Root $root
$hasCompatibleNativeDll = $false
if ($null -ne $nativeDll) {
  $hasCompatibleNativeDll = Test-X64PeFile -Path $nativeDll
}

$continuousExportSchema = @'
{
  "continuous_export": {
    "schema_version": 1,
    "export_config": {
      "frame_stride": 4,
      "points_max": 32,
      "density_floor_ratio": 0.2
    },
    "frames": [
      {
        "t": 0.0,
        "density": [[1.0]],
        "exp_x": 0.0,
        "exp_y": 0.0
      }
    ]
  }
}
'@

$schemaCheck = @'
import json
import sys

if len(sys.argv) < 2:
    raise SystemExit("continuous export schema smoke failed: missing fixture path")

with open(sys.argv[1], "r", encoding="utf-8-sig") as handle:
    data = json.load(handle)

export = data.get("continuous_export")
if not isinstance(export, dict):
    raise SystemExit("continuous export schema smoke failed: missing continuous_export wrapper")
if export.get("schema_version") != 1:
    raise SystemExit("continuous export schema smoke failed: missing continuous_export.schema_version=1")
if not isinstance(export.get("export_config"), dict):
    raise SystemExit("continuous export schema smoke failed: missing continuous_export.export_config")
frames = export.get("frames")
if not isinstance(frames, list) or not frames:
    raise SystemExit("continuous export schema smoke failed: missing continuous_export.frames")
first = frames[0]
if not isinstance(first, dict) or "t" not in first:
    raise SystemExit("continuous export schema smoke failed: frame is missing t")
print("continuous export schema smoke passed")
'@

$schemaScript = Join-Path $env:TEMP ("quantx_continuous_export_schema_{0}.py" -f ([guid]::NewGuid().ToString("N")))
$schemaFixture = Join-Path $env:TEMP ("quantx_continuous_export_fixture_{0}.json" -f ([guid]::NewGuid().ToString("N")))
try {
  Set-Content -Path $schemaFixture -Value $continuousExportSchema -Encoding utf8
  Set-Content -Path $schemaScript -Value $schemaCheck -Encoding utf8
  Invoke-QxTestStep -Name "worktree_guard_smoke" -Details @{ mode = if ($EnforceWorktreeGuard) { "enforce" } else { "default" } } -Action {
    .\scripts\test_worktree_guard.ps1
  }
  Invoke-QxTestStep -Name "check_dependency_policy" -Action {
    .\scripts\check_dependency_policy.ps1
  }
  Invoke-QxTestStep -Name "benchmark_reporting" -Action {
    python .\scripts\test_benchmark_reporting.py
    if ($LASTEXITCODE -ne 0) {
      throw "benchmark reporting smoke failed: exit code $LASTEXITCODE"
    }
  }
  Invoke-QxTestStep -Name "continuous_export_schema" -Action {
    python $schemaScript $schemaFixture
    if ($LASTEXITCODE -ne 0) {
      throw "continuous export schema smoke failed: exit code $LASTEXITCODE"
    }
  }
  Invoke-QxTestStep -Name "build_probe" -SkipReason $(if (-not $IncludeBuildProbe) { "build probe not requested; pass -IncludeBuildProbe to enable" } else { "" }) -Action {
    .\scripts\test_cli_args.ps1 -BuildProbeOnly -EnforceWorktreeGuard:$EnforceWorktreeGuard
    if ($LASTEXITCODE -ne 0) {
      throw "build probe failed: exit code $LASTEXITCODE"
    }
  }

  if ($hasCompatibleNativeDll) {
    Invoke-QxTestStep -Name "python_adapters" -Details @{ native_dll = $nativeDll } -Action {
      python .\scripts\test_python_adapters.py
      if ($LASTEXITCODE -ne 0) {
        throw "python adapter smoke failed: exit code $LASTEXITCODE"
      }
    }
    Invoke-QxTestStep -Name "visualization" -Details @{ native_dll = $nativeDll } -Action {
      python .\scripts\test_visualization.py
      if ($LASTEXITCODE -ne 0) {
        throw "visualization smoke failed: exit code $LASTEXITCODE"
      }
    }
  } else {
    $skipReason = if ($null -eq $nativeDll) {
      "no native quantx.dll was found in this worktree or any ancestor build directory"
    } else {
      "found native DLL but it is not x64-compatible for the current Python runtime: $nativeDll"
    }
    Add-QxStepResult -Name "python_adapters" -Status "skip" -Reason $skipReason -Details @{ native_dll = $nativeDll }
    Add-QxStepResult -Name "visualization" -Status "skip" -Reason $skipReason -Details @{ native_dll = $nativeDll }
  }
} finally {
  Remove-Item $schemaScript -Force -ErrorAction SilentlyContinue
  Remove-Item $schemaFixture -Force -ErrorAction SilentlyContinue
}

$passCount = ($script:QxStepResults | Where-Object { $_.status -eq "pass" }).Count
$skipCount = ($script:QxStepResults | Where-Object { $_.status -eq "skip" }).Count
$failCount = ($script:QxStepResults | Where-Object { $_.status -eq "fail" }).Count
$failedStep = ($script:QxStepResults | Where-Object { $_.status -eq "fail" } | Select-Object -First 1).name

$summary = [ordered]@{
  type = "summary"
  pass = $passCount
  skip = $skipCount
  fail = $failCount
  total = $script:QxStepResults.Count
  no_popup = [bool]$NoPopup
  worktree_guard = if ($EnforceWorktreeGuard) { "enforce" } else { "warn" }
  failed_step = $failedStep
}
Write-Host ("QX_TEST_SUMMARY|" + (ConvertTo-QxJsonLine -Value $summary))

if ($failCount -gt 0) {
  exit 1
}
