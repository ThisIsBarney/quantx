$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\ensure_subagent_worktree.ps1

.\scripts\build.ps1

& .\build-vs\Debug\qx_cli.exe --problem .\examples\problem_rabi.json --options .\examples\options_default.json
if ($LASTEXITCODE -ne 0) { throw "qx_cli rabi failed: exit code $LASTEXITCODE" }

& .\build-vs\Debug\qx_cli.exe --problem .\examples\problem_xx.json --options .\examples\options_default.json
if ($LASTEXITCODE -ne 0) { throw "qx_cli xx failed: exit code $LASTEXITCODE" }

& .\build-vs\Debug\qx_cli.exe --problem .\examples\problem_xx.json --options .\examples\options_default.json --explain
if ($LASTEXITCODE -ne 0) { throw "qx_cli explain failed: exit code $LASTEXITCODE" }

& .\build-vs\Debug\qx_cli.exe --problem .\examples\problem_mps_tebd.json --options .\examples\options_default.json
if ($LASTEXITCODE -ne 0) { throw "qx_cli mps tebd failed: exit code $LASTEXITCODE" }

& .\build-vs\Debug\qx_cli.exe --problem .\examples\problem_mps_tebd.json --options .\examples\options_default.json --explain
if ($LASTEXITCODE -ne 0) { throw "qx_cli mps explain failed: exit code $LASTEXITCODE" }

& .\build-vs\Debug\qx_cli.exe --circuit .\examples\circuit_bell.json --options .\examples\options_default.json
if ($LASTEXITCODE -ne 0) { throw "qx_cli circuit failed: exit code $LASTEXITCODE" }

& .\build-vs\Debug\qx_cli.exe --circuit .\examples\circuit_bell_noisy.json --options .\examples\options_default.json
if ($LASTEXITCODE -ne 0) { throw "qx_cli noisy circuit failed: exit code $LASTEXITCODE" }
