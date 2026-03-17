$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

.\scripts\build.ps1

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
