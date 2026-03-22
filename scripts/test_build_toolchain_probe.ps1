param()

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Run the build script in probe-only mode and capture console output for contract checks.
$output = (& powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build.ps1 -ProbeToolchainOnly 2>&1 | Out-String)

if ([string]::IsNullOrWhiteSpace($output)) {
  throw "build toolchain probe smoke failed: probe output was empty"
}

$line = ($output -split "`r?`n" | Where-Object { $_ -like "QX_BUILD_PROBE|*" } | Select-Object -Last 1)
if ([string]::IsNullOrWhiteSpace($line)) {
  throw "build toolchain probe smoke failed: missing QX_BUILD_PROBE line"
}

$jsonText = $line.Substring("QX_BUILD_PROBE|".Length)
try {
  $probe = $jsonText | ConvertFrom-Json
} catch {
  throw "build toolchain probe smoke failed: probe payload is not valid JSON"
}

if (-not $probe.ok) {
  $detail = if ($probe.error) { $probe.error } else { "unknown probe failure" }
  throw "build toolchain probe smoke failed: probe reported failure: $detail"
}

if ([string]::IsNullOrWhiteSpace($probe.vsdevcmd)) {
  throw "build toolchain probe smoke failed: vsdevcmd path is missing"
}
if ([string]::IsNullOrWhiteSpace($probe.cl)) {
  throw "build toolchain probe smoke failed: cl path is missing"
}

if (-not (Test-Path -LiteralPath $probe.vsdevcmd)) {
  throw "build toolchain probe smoke failed: vsdevcmd path does not exist: $($probe.vsdevcmd)"
}
if (-not (Test-Path -LiteralPath $probe.cl)) {
  throw "build toolchain probe smoke failed: cl path does not exist: $($probe.cl)"
}

Write-Host "build toolchain probe smoke passed"
