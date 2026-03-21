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

.\scripts\check_dependency_policy.ps1
python .\scripts\test_benchmark_reporting.py
if ($LASTEXITCODE -ne 0) { throw "benchmark reporting smoke failed: exit code $LASTEXITCODE" }

# Detect whether a compatible native DLL is available before running the heavier
# Python adapter and visualization smoke checks.
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

$nativeDll = Get-NativeDllCandidate -Root $root
$hasCompatibleNativeDll = $false
if ($null -ne $nativeDll) {
  $hasCompatibleNativeDll = Test-X64PeFile -Path $nativeDll
}

# Lightweight schema smoke for the append-only continuous_export wrapper.
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
  python $schemaScript $schemaFixture
  if ($LASTEXITCODE -ne 0) { throw "continuous export schema smoke failed: exit code $LASTEXITCODE" }
} finally {
  Remove-Item $schemaScript -Force -ErrorAction SilentlyContinue
  Remove-Item $schemaFixture -Force -ErrorAction SilentlyContinue
}

if ($hasCompatibleNativeDll) {
  Write-Host ("running native adapter and visualization smoke with: {0}" -f $nativeDll)
  python .\scripts\test_python_adapters.py
  if ($LASTEXITCODE -ne 0) { throw "python adapter smoke failed: exit code $LASTEXITCODE" }
  python .\scripts\test_visualization.py
  if ($LASTEXITCODE -ne 0) { throw "visualization smoke failed: exit code $LASTEXITCODE" }
} else {
  if ($null -eq $nativeDll) {
    Write-Host "skipping python adapter and visualization smoke: no native quantx.dll was found in this worktree or any ancestor build directory"
  } else {
    Write-Host ("skipping python adapter and visualization smoke: found native DLL but it is not x64-compatible for the current Python runtime: {0}" -f $nativeDll)
  }
}
