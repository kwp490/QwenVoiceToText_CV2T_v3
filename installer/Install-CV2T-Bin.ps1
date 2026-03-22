<#
.SYNOPSIS
    Install CV2T binary (end-user path — Whisper-only prebuilt .exe).

.DESCRIPTION
    Extracts the prebuilt CV2T binary from a release zip, downloads the
    Whisper model, and creates a desktop shortcut.

    Requires Administrator elevation. Installs to C:\Program Files\CV2T\.
    Does NOT require Python, uv, or git.

.PARAMETER ZipPath
    Path to the release zip file containing the cv2t/ directory.

.NOTES
    Run in an elevated PowerShell session:
        Set-ExecutionPolicy Bypass -Scope Process -Force
        .\Install-CV2T-Bin.ps1 -ZipPath ".\cv2t-v2.0.0-win64.zip"
#>

#Requires -RunAsAdministrator

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$ZipPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$InstallDir = "C:\Program Files\CV2T"
$ModelsDir = "$env:LOCALAPPDATA\CV2T\models"
$ExePath = "$InstallDir\cv2t.exe"

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

# ── Extract / copy files ─────────────────────────────────────────────────────
Write-Step "Installing CV2T binary..."
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

if ($ZipPath -and (Test-Path $ZipPath)) {
    Write-Host "  Extracting from $ZipPath..."
    Expand-Archive -Path $ZipPath -DestinationPath $InstallDir -Force
} else {
    # Assume the script is next to the cv2t/ directory
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $srcDir = Join-Path $scriptDir "cv2t"
    if (Test-Path $srcDir) {
        Write-Host "  Copying from $srcDir..."
        Copy-Item -Path "$srcDir\*" -Destination $InstallDir -Recurse -Force
    } else {
        Write-Host "  ERROR: No zip provided and no cv2t/ directory found next to this script." -ForegroundColor Red
        Write-Host "  Usage: .\Install-CV2T-Bin.ps1 -ZipPath 'path\to\cv2t-release.zip'"
        exit 1
    }
}

# ── Download Whisper model ───────────────────────────────────────────────────
Write-Step "Downloading Whisper model..."
New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null
& $ExePath download-model --engine whisper --target-dir $ModelsDir

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
$shortcut.TargetPath = $ExePath
$shortcut.WorkingDirectory = $InstallDir
$shortcut.Description = "CV2T — Voice to Text"
$shortcut.Save()

# ── Windows Defender exclusions ──────────────────────────────────────────────
Write-Step "Adding Windows Defender exclusions..."
try {
    Add-MpPreference -ExclusionPath $InstallDir -ErrorAction SilentlyContinue
    Add-MpPreference -ExclusionProcess $ExePath -ErrorAction SilentlyContinue
    Add-MpPreference -ExclusionPath "$env:APPDATA\CV2T" -ErrorAction SilentlyContinue
    Write-Host "  Defender exclusions added" -ForegroundColor Green
} catch {
    Write-Host "  Could not add Defender exclusions (non-critical)" -ForegroundColor Yellow
}

Write-Host "`n=== CV2T installed successfully ===" -ForegroundColor Green
Write-Host "  Install dir: $InstallDir"
Write-Host "  Models dir:  $ModelsDir"
Write-Host "  Launch: $ExePath"
