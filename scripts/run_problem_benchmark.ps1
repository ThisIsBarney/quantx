param(
  [string]$Suite = "benchmarks/suites/continuous2d_smoke.json"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Keep the legacy continuous-space entrypoint, but reuse the unified dispatcher.
.\scripts\run_benchmark.ps1 -Suite $Suite
