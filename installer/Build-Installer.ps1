<#
.SYNOPSIS
    Build the CV2T installer package (PyInstaller binary + Inno Setup wizard).

.DESCRIPTION
    Two-step build:
      1. pyinstaller cv2t.spec  → dist/cv2t/
      2. iscc installer/cv2t-setup.iss → installer/Output/CV2T-Setup-<version>.exe

    Run from the repository root. Requires:
      - Python venv with PyInstaller (uv sync --extra whisper --extra dev)
      - Inno Setup 6.x with iscc.exe on PATH or at the default install location

.PARAMETER Clean
    Force a full PyInstaller rebuild (passes --clean). By default the build
    reuses PyInstaller's cached dependency analysis in build/.

.PARAMETER InnoOnly
    Skip the PyInstaller step and jump straight to Inno Setup compilation.
    Useful when iterating on installer UI or Pascal code without changing
    the application binary.

.PARAMETER Fast
    Use fast compression for Inno Setup (lzma2/fast, non-solid) instead of
    the release-quality lzma2/ultra64 solid compression. Produces a larger
    installer but compiles much faster. Intended for dev/test builds.

.NOTES
    Usage:
        .\installer\Build-Installer.ps1                  # incremental release build
        .\installer\Build-Installer.ps1 -Fast             # fast dev build
        .\installer\Build-Installer.ps1 -Clean            # force full rebuild
        .\installer\Build-Installer.ps1 -InnoOnly -Fast   # repackage existing dist/ quickly
#>
param(
    [switch]$Clean,
    [switch]$InnoOnly,
    [switch]$Fast
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $RepoRoot

function Write-Step($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }

# ── Source-hash helper ────────────────────────────────────────────────────────
# Computes a hash over all files that affect the PyInstaller output so we can
# skip the (slow) PyInstaller step when nothing has changed.
function Get-SourceHash {
    $hashInput = @()
    # Python source + spec + project config
    $files = @(Get-ChildItem -Path "cv2t" -Recurse -Include "*.py" -File) +
             @(Get-Item "cv2t.spec") +
             @(Get-Item "pyproject.toml")
    foreach ($f in $files | Sort-Object FullName) {
        $h = (Get-FileHash -Path $f.FullName -Algorithm SHA256).Hash
        $hashInput += "$($f.FullName)|$h"
    }
    # Hash the combined list
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($hashInput -join "`n")
    $sha = [System.Security.Cryptography.SHA256]::Create()
    return [BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-', '')
}

$HashFile = "build\.cv2t-build-hash"

# ── Pre-flight checks ────────────────────────────────────────────────────────
Write-Step "Checking prerequisites..."

if (-not (Test-Path "cv2t.spec")) {
    Write-Host "  ERROR: cv2t.spec not found. Run this script from the repository root." -ForegroundColor Red
    Pop-Location
    exit 1
}
Write-Ok "cv2t.spec found"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "  ERROR: uv not found on PATH." -ForegroundColor Red
    Write-Host "  Install it: irm https://astral.sh/uv/install.ps1 | iex" -ForegroundColor Yellow
    Pop-Location
    exit 1
}
Write-Ok "uv found: $(uv --version 2>$null)"

# Find iscc.exe early so we don't waste time on PyInstaller if it's missing
$iscc = Get-Command iscc -ErrorAction SilentlyContinue
if (-not $iscc) {
    $defaultPaths = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    foreach ($p in $defaultPaths) {
        if (Test-Path $p) {
            $iscc = Get-Item $p
            break
        }
    }
}
if (-not $iscc) {
    Write-Host "  ERROR: Inno Setup compiler (iscc.exe) not found." -ForegroundColor Red
    Write-Host "  Install it: winget install JRSoftware.InnoSetup" -ForegroundColor Yellow
    Write-Host "  Or download from https://jrsoftware.org/isdl.php" -ForegroundColor Yellow
    Pop-Location
    exit 1
}
Write-Ok "Inno Setup found: $($iscc)"

# ── Step 1: PyInstaller ──────────────────────────────────────────────────────
if ($InnoOnly) {
    Write-Step "Skipping PyInstaller (-InnoOnly flag set)"
    if (-not (Test-Path "dist\cv2t\cv2t.exe")) {
        Write-Host "  ERROR: dist\cv2t\cv2t.exe not found. Run a full build first." -ForegroundColor Red
        Pop-Location
        exit 1
    }
    Write-Ok "Using existing binary: dist\cv2t\cv2t.exe"
} else {
    # Check source hash to skip PyInstaller when nothing changed
    $skipPyInstaller = $false
    if (-not $Clean) {
        $currentHash = Get-SourceHash
        if ((Test-Path $HashFile) -and (Test-Path "dist\cv2t\cv2t.exe")) {
            $savedHash = Get-Content $HashFile -Raw
            if ($savedHash.Trim() -eq $currentHash) {
                $skipPyInstaller = $true
            }
        }
    }

    if ($skipPyInstaller) {
        Write-Step "PyInstaller skipped (source unchanged since last build)"
        Write-Ok "Using cached binary: dist\cv2t\cv2t.exe"
    } else {
        Write-Step "Building CV2T binary with PyInstaller..."

        $pyiArgs = @("pyinstaller", "cv2t.spec", "--noconfirm")
        if ($Clean) { $pyiArgs += "--clean" }

        $prevPref = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            uv run @pyiArgs 2>&1 | ForEach-Object { Write-Host "  $_" }
        } finally {
            $ErrorActionPreference = $prevPref
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ERROR: PyInstaller build failed." -ForegroundColor Red
            Pop-Location
            exit 1
        }

        if (-not (Test-Path "dist\cv2t\cv2t.exe")) {
            Write-Host "  ERROR: dist\cv2t\cv2t.exe not found after build." -ForegroundColor Red
            Pop-Location
            exit 1
        }
        Write-Ok "Binary built: dist\cv2t\cv2t.exe"

        # Persist source hash for next run
        $currentHash = Get-SourceHash
        if (-not (Test-Path "build")) { New-Item -ItemType Directory -Path "build" | Out-Null }
        $currentHash | Set-Content $HashFile -NoNewline
        Write-Ok "Build hash saved"
    }
}

# ── Step 2: Inno Setup ──────────────────────────────────────────────────────
Write-Step "Building installer with Inno Setup..."

Write-Host "  Using: $($iscc)"
$isccArgs = @("installer\cv2t-setup.iss")
if ($Fast) {
    $isccArgs = @("/DFastCompress") + $isccArgs
    Write-Host "  Mode: fast compression (dev build)" -ForegroundColor Yellow
} else {
    Write-Host "  Mode: ultra64 solid compression (release build)"
}
$prevPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $iscc @isccArgs 2>&1 | ForEach-Object { Write-Host "  $_" }
} finally {
    $ErrorActionPreference = $prevPref
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Inno Setup compilation failed." -ForegroundColor Red
    Pop-Location
    exit 1
}

$setupExe = Get-ChildItem "installer\Output\CV2T-Setup-*.exe" | Select-Object -First 1
if ($setupExe) {
    Write-Ok "Installer built: $($setupExe.FullName)"
    Write-Host ""
    Write-Host "  File size: $([math]::Round($setupExe.Length / 1MB, 1)) MB" -ForegroundColor DarkGray
} else {
    Write-Host "  WARNING: Expected output not found in installer\Output\" -ForegroundColor Yellow
}

Pop-Location
