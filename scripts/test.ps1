$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\ensure_subagent_worktree.ps1

.\scripts\build.ps1
ctest --test-dir build-vs -C Debug --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "ctest failed: exit code $LASTEXITCODE" }
# Smoke binaries must report failures on the console, not via modal debugger popups.
.\build-vs\Debug\qx_smoke.exe
if ($LASTEXITCODE -ne 0) { throw "qx_smoke failed: exit code $LASTEXITCODE" }
.\scripts\test_cli_args.ps1
.\scripts\check_dependency_policy.ps1
python .\scripts\test_python_adapters.py
if ($LASTEXITCODE -ne 0) { throw "python adapter smoke failed: exit code $LASTEXITCODE" }
python .\scripts\test_benchmark_reporting.py
if ($LASTEXITCODE -ne 0) { throw "benchmark reporting smoke failed: exit code $LASTEXITCODE" }
python .\scripts\test_visualization.py
if ($LASTEXITCODE -ne 0) { throw "visualization smoke failed: exit code $LASTEXITCODE" }
