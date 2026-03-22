<#
.SYNOPSIS
    Install CV2T from source (developer/contributor path — supports both engines).

.DESCRIPTION
    Clones the CV2T repository, installs Python 3.11 via uv, syncs all
    dependencies, downloads model weights, and creates a desktop shortcut.

    Requires Administrator elevation. Installs to C:\Program Files\CV2T\.
    Models stored at %LOCALAPPDATA%\CV2T\models (user-writable).

.NOTES
    Run in an elevated PowerShell session:
        Set-ExecutionPolicy Bypass -Scope Process -Force
        .\Install-CV2T-Source.ps1
#>

#Requires -RunAsAdministrator

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$InstallDir = "C:\Program Files\CV2T"
$ModelsDir = "$env:LOCALAPPDATA\CV2T\models"
$RepoUrl = "https://github.com/kwp490/cv2t.git"

function Write-Step($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }

# ── WIN-01: Check NVIDIA GPU ─────────────────────────────────────────────────
Write-Step "Checking for NVIDIA GPU..."
try {
    $gpu = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
    if ($gpu) {
        Write-Host "  GPU detected: $($gpu.Trim())" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: No NVIDIA GPU detected. GPU acceleration will not be available." -ForegroundColor Yellow
    }
} catch {
    Write-Host "  WARNING: nvidia-smi not found. GPU acceleration may not be available." -ForegroundColor Yellow
}

# ── Install uv ───────────────────────────────────────────────────────────────
Write-Step "Installing uv package manager..."
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
}
Write-Host "  uv: $(uv --version)" -ForegroundColor Green

# ── Install Python 3.11 ─────────────────────────────────────────────────────
Write-Step "Ensuring Python 3.11..."
uv python install 3.11

# ── Clone repo ───────────────────────────────────────────────────────────────
Write-Step "Cloning CV2T repository..."
if (Test-Path $InstallDir) {
    Write-Host "  $InstallDir already exists — pulling latest..." -ForegroundColor Yellow
    Push-Location $InstallDir
    git pull
    Pop-Location
} else {
    git clone $RepoUrl $InstallDir
}

# ── Install dependencies ─────────────────────────────────────────────────────
Write-Step "Installing dependencies (all engines)..."
Push-Location $InstallDir
uv sync --extra all --extra dev
Pop-Location

# ── Download models ──────────────────────────────────────────────────────────
Write-Step "Downloading Whisper model..."
Push-Location $InstallDir
uv run cv2t download-model --engine whisper --target-dir $ModelsDir
Pop-Location

# ── Set permissions ──────────────────────────────────────────────────────────
Write-Step "Setting directory permissions..."
$acl = Get-Acl $InstallDir
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    "BUILTIN\Users", "ReadAndExecute", "ContainerInherit,ObjectInherit", "None", "Allow"
)
$acl.SetAccessRule($rule)
Set-Acl $InstallDir $acl

# ── Create desktop shortcut ──────────────────────────────────────────────────
Write-Step "Creating desktop shortcut..."
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "CV2T.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "uv"
$shortcut.Arguments = "run cv2t"
$shortcut.WorkingDirectory = $InstallDir
$shortcut.Description = "CV2T — Voice to Text"
$shortcut.Save()

# ── Windows Defender exclusions ──────────────────────────────────────────────
Write-Step "Adding Windows Defender exclusions..."
try {
    Add-MpPreference -ExclusionPath $InstallDir -ErrorAction SilentlyContinue
    Add-MpPreference -ExclusionPath "$env:APPDATA\CV2T" -ErrorAction SilentlyContinue
    Write-Host "  Defender exclusions added" -ForegroundColor Green
} catch {
    Write-Host "  Could not add Defender exclusions (non-critical)" -ForegroundColor Yellow
}

Write-Host "`n=== CV2T installed successfully ===" -ForegroundColor Green
Write-Host "  Install dir: $InstallDir"
Write-Host "  Models dir:  $ModelsDir"
Write-Host "  Launch with: cd '$InstallDir' && uv run cv2t"
