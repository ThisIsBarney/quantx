param(
  [switch]$EnforceWorktreeGuard,
  [switch]$SkipBuildProbe,
  [switch]$BuildProbeOnly,
  # Skip the build probe and the build step entirely.
  # Use this when the binary was already compiled by a prior CI step.
  [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($EnforceWorktreeGuard) {
  .\scripts\ensure_subagent_worktree.ps1 -Enforce
} else {
  .\scripts\ensure_subagent_worktree.ps1
}

# Locate a Visual Studio developer command prompt script for probing build prerequisites.
function Get-VsDevCmdPath {
  $candidates = @(
    "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
  )
  foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) {
      return $candidate
    }
  }
  return $null
}

# Run a lightweight probe so failures are diagnosed before a full build starts.
function Invoke-BuildProbe {
  param([switch]$ThrowOnFailure)

  $diagnostics = New-Object System.Collections.Generic.List[string]

  try {
    $cmakeVersion = (& cmake --version 2>&1 | Select-Object -First 1)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace(($cmakeVersion | Out-String))) {
      throw "cmake is not available on PATH"
    }
    $diagnostics.Add("cmake=$cmakeVersion") | Out-Null
  } catch {
    $diagnostics.Add("cmake=missing") | Out-Null
    $diagnostics.Add("cmake_error=$($_.Exception.Message)") | Out-Null
    if ($ThrowOnFailure) {
      Write-Host "build probe diagnostics:"
      foreach ($line in $diagnostics) { Write-Host "  $line" }
      throw "build probe failed: cmake is required before invoking scripts/build.ps1"
    }
    return $false
  }

  $vsDevCmd = Get-VsDevCmdPath
  if ([string]::IsNullOrWhiteSpace($vsDevCmd)) {
    $diagnostics.Add("vsdevcmd=missing") | Out-Null
    if ($ThrowOnFailure) {
      Write-Host "build probe diagnostics:"
      foreach ($line in $diagnostics) { Write-Host "  $line" }
      throw "build probe failed: Visual Studio developer command prompt was not found"
    }
    return $false
  }
  $diagnostics.Add("vsdevcmd=$vsDevCmd") | Out-Null

  $compilerPath = & cmd /c "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && where cl" 2>&1
  if ($LASTEXITCODE -ne 0) {
    $diagnostics.Add("cl=missing") | Out-Null
    $diagnostics.Add("cl_error=$(($compilerPath | Out-String).Trim())") | Out-Null
    if ($ThrowOnFailure) {
      Write-Host "build probe diagnostics:"
      foreach ($line in $diagnostics) { Write-Host "  $line" }
      throw "build probe failed: cl.exe is not available from VsDevCmd"
    }
    return $false
  }

  $clResolved = (($compilerPath | Select-Object -First 1) | Out-String).Trim()
  if ([string]::IsNullOrWhiteSpace($clResolved)) {
    $diagnostics.Add("cl=empty") | Out-Null
    if ($ThrowOnFailure) {
      Write-Host "build probe diagnostics:"
      foreach ($line in $diagnostics) { Write-Host "  $line" }
      throw "build probe failed: cl.exe path could not be resolved"
    }
    return $false
  }
  $diagnostics.Add("cl=$clResolved") | Out-Null

  Write-Host "build probe passed:"
  foreach ($line in $diagnostics) { Write-Host "  $line" }
  return $true
}

if ($SkipBuild) {
  Write-Host "skip-build mode: assuming binary was already compiled by a prior step."
} else {
  if (-not $SkipBuildProbe) {
    $null = Invoke-BuildProbe -ThrowOnFailure
  }

  if ($BuildProbeOnly) {
    Write-Host "build probe only mode: skipping build and CLI argument checks."
    exit 0
  }

  .\scripts\build.ps1 -EnforceWorktreeGuard:$EnforceWorktreeGuard
}

# --help should succeed.
& .\build-vs\Debug\qx_cli.exe --help | Out-Null
if ($LASTEXITCODE -ne 0) { throw "qx_cli --help failed: exit code $LASTEXITCODE" }

# --version should succeed and print a non-empty string.
$version = & .\build-vs\Debug\qx_cli.exe --version
if ($LASTEXITCODE -ne 0) { throw "qx_cli --version failed: exit code $LASTEXITCODE" }
if ([string]::IsNullOrWhiteSpace(($version | Out-String))) { throw "qx_cli --version returned empty output" }

# invalid --log-level should fail.
cmd /c ".\\build-vs\\Debug\\qx_cli.exe --problem .\\examples\\problem_rabi.json --log-level invalid >NUL 2>NUL"
if ($LASTEXITCODE -eq 0) { throw "qx_cli invalid --log-level should fail" }

# mutually-exclusive --problem/--circuit should fail.
cmd /c ".\\build-vs\\Debug\\qx_cli.exe --problem .\\examples\\problem_rabi.json --circuit .\\examples\\circuit_bell.json >NUL 2>NUL"
if ($LASTEXITCODE -eq 0) { throw "qx_cli with both --problem and --circuit should fail" }

# valid path should still work.
& .\build-vs\Debug\qx_cli.exe --problem .\examples\problem_rabi.json --options .\examples\options_default.json --log-level warn | Out-Null
if ($LASTEXITCODE -ne 0) { throw "qx_cli valid problem run failed: exit code $LASTEXITCODE" }

# validate-only should pass without execution.
$validate = & .\build-vs\Debug\qx_cli.exe --problem .\examples\problem_rabi.json --validate-only
if ($LASTEXITCODE -ne 0) { throw "qx_cli --validate-only failed: exit code $LASTEXITCODE" }
if ((($validate | Out-String) -notmatch '"validated":"problem"')) { throw "qx_cli --validate-only output mismatch" }
