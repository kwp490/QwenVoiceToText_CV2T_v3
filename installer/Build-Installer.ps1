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

.NOTES
    Usage:
        .\installer\Build-Installer.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $RepoRoot

function Write-Step($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }

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
Write-Step "Building CV2T binary with PyInstaller..."

$prevPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    uv run pyinstaller cv2t.spec --noconfirm 2>&1 | ForEach-Object { Write-Host "  $_" }
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

# ── Step 2: Inno Setup ──────────────────────────────────────────────────────
Write-Step "Building installer with Inno Setup..."

Write-Host "  Using: $($iscc)"
$prevPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $iscc "installer\cv2t-setup.iss" 2>&1 | ForEach-Object { Write-Host "  $_" }
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
