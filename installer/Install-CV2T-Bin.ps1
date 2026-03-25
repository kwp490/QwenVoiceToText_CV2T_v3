<#
.SYNOPSIS
    Install CV2T binary (end-user path — prebuilt .exe).

.DESCRIPTION
    Extracts the prebuilt CV2T binary from a release zip, prompts the user
    to select Whisper, Canary, or both engines, downloads the chosen model(s),
    and creates a desktop shortcut.

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
function Write-Already($msg) { Write-Host "  [SKIP] $msg" -ForegroundColor DarkGray }
function Write-Ok($msg) { Write-Host "  [OK]   $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }

function Invoke-NativeCommand {
    <# Run a native command, print indented output, and throw on failure. #>
    param([string]$Label, [scriptblock]$Command)
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $Command 2>&1
        foreach ($line in $output) { Write-Host "  $line" }
    } finally {
        $ErrorActionPreference = $prevPref
    }
    if ($LASTEXITCODE -ne 0) { throw "$Label failed (exit code $LASTEXITCODE)" }
}

function Invoke-StreamingCommand {
    <# Run a native command, streaming output line-by-line for real-time progress. #>
    param([string]$Label, [scriptblock]$Command)
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Command 2>&1 | ForEach-Object { Write-Host "  $_" }
    } finally {
        $ErrorActionPreference = $prevPref
    }
    if ($LASTEXITCODE -ne 0) { throw "$Label failed (exit code $LASTEXITCODE)" }
}

# ── WIN-01: Check NVIDIA GPU ─────────────────────────────────────────────────
Write-Step "Checking for NVIDIA GPU..."
try {
    $gpu = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
    if ($gpu) {
        Write-Ok "GPU detected: $($gpu.Trim())"
    } else {
        Write-Warn "No NVIDIA GPU detected. GPU acceleration will not be available."
    }
} catch {
    Write-Warn "nvidia-smi not found. GPU acceleration may not be available."
}

# ── Engine selection ─────────────────────────────────────────────────────────
Write-Step "Select speech engine(s) to install"
Write-Host "  [1] Whisper only  (default - fast, lighter on VRAM)"
Write-Host "  [2] Canary only   (NeMo/torch - higher accuracy, ~5 GB VRAM)"
Write-Host "  [3] Both engines"
Write-Host ""
$engineChoice = Read-Host "  Enter choice [1/2/3] (default: 1)"
switch ($engineChoice) {
    '2'     { $installWhisper = $false; $installCanary = $true  }
    '3'     { $installWhisper = $true;  $installCanary = $true  }
    default { $installWhisper = $true;  $installCanary = $false }
}
$selectedNames = @()
if ($installWhisper) { $selectedNames += 'whisper' }
if ($installCanary)  { $selectedNames += 'canary' }
Write-Ok "Selected engine(s): $($selectedNames -join ', ')"

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

# ── Download models ───────────────────────────────────────────────────────────
New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null

if ($installWhisper) {
    Write-Step "Checking Whisper model..."
    $whisperFiles = @("config.json", "model.bin", "tokenizer.json")
    $whisperReady = $true
    foreach ($f in $whisperFiles) {
        if (-not (Test-Path (Join-Path $ModelsDir $f))) { $whisperReady = $false; break }
    }
    if ($whisperReady) {
        Write-Already "Whisper model already present in $ModelsDir"
    } else {
        Write-Host "  Downloading Whisper model 'large-v3-turbo'..."
        Write-Host "  Using repo mobiuslabsgmbh/faster-whisper-large-v3-turbo"
        Invoke-StreamingCommand 'Model download' { & $ExePath download-model --engine whisper --target-dir $ModelsDir }
        Write-Ok "Whisper model downloaded to $ModelsDir"
    }
}

if ($installCanary) {
    Write-Step "Checking Canary model..."
    $canaryDir = Join-Path $ModelsDir "canary"
    if (Test-Path $canaryDir) {
        Write-Already "Canary model already present in $canaryDir"
    } else {
        Write-Host "  Downloading Canary model..."
        Invoke-StreamingCommand 'Canary model download' { & $ExePath download-model --engine canary --target-dir $ModelsDir }
        Write-Ok "Canary model downloaded to $ModelsDir"
    }
}

# ── Write default engine to settings ─────────────────────────────────────────
Write-Step "Configuring default engine..."
$settingsDir = "$env:APPDATA\CV2T"
$settingsFile = Join-Path $settingsDir "settings.json"
if (-not (Test-Path $settingsDir)) {
    New-Item -ItemType Directory -Path $settingsDir -Force | Out-Null
}
if ($installCanary -and -not $installWhisper) {
    $defaultEngine = 'canary'
} else {
    $defaultEngine = 'whisper'
}
if (Test-Path $settingsFile) {
    $cfg = Get-Content $settingsFile -Raw | ConvertFrom-Json
}
if (-not $cfg) {
    $cfg = [pscustomobject]@{}
}
if (-not ($cfg.PSObject.Properties.Name -contains 'engine')) {
    $cfg | Add-Member -NotePropertyName 'engine' -NotePropertyValue $defaultEngine
} else {
    $cfg.engine = $defaultEngine
}
$cfg | ConvertTo-Json -Depth 10 | Set-Content $settingsFile -Encoding UTF8
Write-Ok "Default engine set to '$defaultEngine' in $settingsFile"

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
$shortcut.Description = "CV2T - Voice to Text"
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
