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

function Get-VsDevCmdPath {
  $candidates = @(
    "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
  )

  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
      return $candidate
    }
  }

  throw "Visual Studio developer command prompt not found."
}

function Get-VsCompilerPath {
  param([string]$VsDevCmdPath)

  $compilerPath = & cmd /c "`"$VsDevCmdPath`" -arch=x64 -host_arch=x64 >nul && where cl"
  if ($LASTEXITCODE -ne 0) {
    throw "failed to resolve cl.exe from the Visual Studio developer command prompt"
  }

  $resolved = ($compilerPath | Select-Object -First 1).Trim()
  if ([string]::IsNullOrWhiteSpace($resolved)) {
    throw "Visual Studio compiler path was empty."
  }
  return $resolved
}

function Invoke-VsCommand {
  param(
    [string]$VsDevCmdPath,
    [string]$CommandLine
  )

  $prefix = "`"$VsDevCmdPath`" -arch=x64 -host_arch=x64 >nul && "
  & cmd /c ($prefix + $CommandLine)
}

function Reset-InvalidBuildDirectory {
  param(
    [string]$BuildDir,
    [string]$ExpectedGenerator
  )

  $cachePath = Join-Path $BuildDir "CMakeCache.txt"
  if (-not (Test-Path $cachePath)) {
    return
  }

  $hasCompiler = Select-String -Path $cachePath -Pattern '^CMAKE_CXX_COMPILER:' -Quiet
  $generatorLine = Select-String -Path $cachePath -Pattern '^CMAKE_GENERATOR:INTERNAL=' | Select-Object -First 1
  $generatorMismatch = $true
  if ($generatorLine) {
    $generatorMismatch = ($generatorLine.Line -notmatch [regex]::Escape($ExpectedGenerator))
  }

  if (-not $hasCompiler -or $generatorMismatch) {
    Remove-Item -Recurse -Force $BuildDir
  }
}

$vsDevCmd = Get-VsDevCmdPath
$clPath = (Get-VsCompilerPath -VsDevCmdPath $vsDevCmd).Replace('\', '/')

$fallbackDepsDir = [System.IO.Path]::GetFullPath((Join-Path $root "..\..\build-vs\_deps"))
$fetchContentPrefix = ""
if (Test-Path $fallbackDepsDir) {
  $fetchContentPrefix = "set `"FETCHCONTENT_BASE_DIR=$fallbackDepsDir`" && "
}

function Get-FetchContentSourceArgs {
  param([string]$DepsDir)

  $sourceArgs = @()
  $depMap = @(
    @{ Var = "NLOHMANN_JSON"; Dir = "nlohmann_json-src" }
    @{ Var = "SPDLOG"; Dir = "spdlog-src" }
    @{ Var = "CLI11"; Dir = "cli11-src" }
    @{ Var = "EIGEN"; Dir = "eigen-src" }
  )

  foreach ($dep in $depMap) {
    $candidate = Join-Path $DepsDir $dep.Dir
    if (Test-Path $candidate) {
      $sourceArgs += "-DFETCHCONTENT_SOURCE_DIR_$($dep.Var)=`"$candidate`""
    }
  }

  return ($sourceArgs -join " ")
}

Reset-InvalidBuildDirectory -BuildDir "build-vs" -ExpectedGenerator "NMake Makefiles"

$fetchContentSourceArgs = Get-FetchContentSourceArgs -DepsDir $fallbackDepsDir
$configureCommand = $fetchContentPrefix + "cmake -S . -B build-vs -G `"NMake Makefiles`" -DCMAKE_BUILD_TYPE=Debug " + $fetchContentSourceArgs + " -DCMAKE_C_COMPILER=`"$clPath`" -DCMAKE_CXX_COMPILER=`"$clPath`""
Invoke-VsCommand -VsDevCmdPath $vsDevCmd -CommandLine $configureCommand
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed: exit code $LASTEXITCODE" }

$buildCommand = $fetchContentPrefix + "cmake --build build-vs"
Invoke-VsCommand -VsDevCmdPath $vsDevCmd -CommandLine $buildCommand
if ($LASTEXITCODE -ne 0) { throw "cmake build failed: exit code $LASTEXITCODE" }

Write-Host "Build completed: build-vs/Debug"
