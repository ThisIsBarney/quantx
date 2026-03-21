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
  # Collect candidate locations in priority order and deduplicate while preserving order.
  $candidates = New-Object System.Collections.Generic.List[string]
  $seen = New-Object System.Collections.Generic.HashSet[string]

  function Add-Candidate([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
      return
    }
    $normalized = [System.IO.Path]::GetFullPath($Path)
    if ($seen.Add($normalized)) {
      $candidates.Add($normalized)
    }
  }

  # Prefer already-exported VS installation hints when present.
  if (-not [string]::IsNullOrWhiteSpace($env:VSINSTALLDIR)) {
    Add-Candidate (Join-Path $env:VSINSTALLDIR "Common7\Tools\VsDevCmd.bat")
  }

  # Known install layouts across local/CI images.
  $roots = @(
    "C:\Program Files\Microsoft Visual Studio"
    "C:\Program Files (x86)\Microsoft Visual Studio"
  )
  $versions = @("18", "2022")
  $editions = @("Enterprise", "Professional", "Community", "BuildTools")
  foreach ($root in $roots) {
    foreach ($version in $versions) {
      foreach ($edition in $editions) {
        Add-Candidate (Join-Path $root "$version\$edition\Common7\Tools\VsDevCmd.bat")
      }
    }
  }

  # Fall back to vswhere when available (works on GitHub-hosted Windows runners).
  $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path $vswhere) {
    $installPaths = & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath -format value 2>$null
    foreach ($installPath in $installPaths) {
      Add-Candidate (Join-Path $installPath "Common7\Tools\VsDevCmd.bat")
    }
  }

  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
      return $candidate
    }
  }

  $checked = ($candidates | ForEach-Object { "  - $_" }) -join [Environment]::NewLine
  throw "Visual Studio developer command prompt not found. checked:`n$checked"
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
