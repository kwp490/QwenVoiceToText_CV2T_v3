<#
.SYNOPSIS
    Install CV2T from source (developer/contributor path — supports both engines).

.DESCRIPTION
    Copies the local CV2T source tree to the install directory, installs
    Python 3.11 via uv, syncs all dependencies, downloads model weights,
    and creates a desktop shortcut.

    Requires Administrator elevation. Installs to C:\Program Files\CV2T\.
    Models stored at %LOCALAPPDATA%\CV2T\models (user-writable).

.NOTES
    Run in an elevated PowerShell session from within the repo:
        Set-ExecutionPolicy Bypass -Scope Process -Force
        .\installer\Install-CV2T-Source.ps1
#>

#Requires -RunAsAdministrator

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$InstallDir = "C:\Program Files\CV2T"
$ModelsDir = "$env:LOCALAPPDATA\CV2T\models"
$RepoName = Split-Path -Leaf $PWD.Path

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

function Sync-SourceTree {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceDir,
        [Parameter(Mandatory = $true)]
        [string]$DestinationDir
    )

    if (-not (Test-Path $DestinationDir)) {
        New-Item -ItemType Directory -Path $DestinationDir -Force | Out-Null
    }

    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        robocopy $SourceDir $DestinationDir /MIR /XD .git __pycache__ .venv /XF "*.pyc" /NFL /NDL /NJH /NJS /NC /NS /NP 2>&1 | Out-Null
    } finally {
        $ErrorActionPreference = $prevPref
    }
    if ($LASTEXITCODE -gt 7) { throw "robocopy failed (exit code $LASTEXITCODE)" }
    $LASTEXITCODE = 0
}

function Assert-ValidInstallLayout {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallRoot,
        [Parameter(Mandatory = $true)]
        [string]$NestedRepoPath
    )

    if (Test-Path $NestedRepoPath) {
        throw "Invalid install layout: nested repo directory still exists at $NestedRepoPath"
    }

    $requiredPaths = @(
        (Join-Path $InstallRoot "pyproject.toml"),
        (Join-Path $InstallRoot "download_model.py"),
        (Join-Path $InstallRoot "cv2t\__main__.py")
    )
    foreach ($path in $requiredPaths) {
        if (-not (Test-Path $path)) {
            throw "Invalid install layout: missing required path $path"
        }
    }
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

# ── Install uv ───────────────────────────────────────────────────────────────
Write-Step "Checking for uv package manager..."
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Already "uv already installed: $(uv --version)"
} else {
    Write-Host "  Installing uv..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
    Write-Ok "uv installed: $(uv --version)"
}

# ── Install Python 3.11 ─────────────────────────────────────────────────────
Write-Step "Checking for Python 3.11..."
$py311 = uv python find 3.11 2>$null
if ($py311) {
    Write-Already "Python 3.11 already available: $py311"
} else {
    Write-Host "  Installing Python 3.11 via uv..."
    uv python install 3.11
    Write-Ok "Python 3.11 installed"
}

# ── Copy/sync source to install dir ──────────────────────────────────────────
Write-Step "Setting up CV2T repository..."
# Determine the repo root: the parent of the directory containing this script.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$RepoName = Split-Path -Leaf $RepoRoot
$NestedRepoDir = Join-Path $InstallDir $RepoName

# Verify we have a valid source tree next to this script
if (-not (Test-Path (Join-Path $RepoRoot "pyproject.toml"))) {
    Write-Host "  ERROR: Cannot find pyproject.toml in $RepoRoot" -ForegroundColor Red
    Write-Host "  Run this script from its location inside the CV2T repository."
    exit 1
}

if ($RepoRoot -eq $InstallDir) {
    Write-Already "Running from install directory — skipping copy"
} elseif (Test-Path (Join-Path $InstallDir ".git")) {
    # Install dir is a git clone — remove and replace with local source
    Write-Warn "$InstallDir contains an old git clone — replacing with local source..."
    Remove-Item -Recurse -Force $InstallDir
    Write-Host "  Syncing local source contents from $RepoRoot..."
    Sync-SourceTree -SourceDir $RepoRoot -DestinationDir $InstallDir
    Write-Ok "Source installed to $InstallDir from local tree"
} elseif (Test-Path $InstallDir) {
    Write-Warn "$InstallDir exists — updating with local source..."
    if (Test-Path $NestedRepoDir) {
        Write-Warn "Removing stale nested repo copy at $NestedRepoDir..."
        Remove-Item -Recurse -Force $NestedRepoDir
    }
    Sync-SourceTree -SourceDir $RepoRoot -DestinationDir $InstallDir
    Write-Ok "Source synced to $InstallDir"
} else {
    Write-Host "  Syncing local source contents from $RepoRoot..."
    Sync-SourceTree -SourceDir $RepoRoot -DestinationDir $InstallDir
    Write-Ok "Source installed to $InstallDir"
}

Assert-ValidInstallLayout -InstallRoot $InstallDir -NestedRepoPath $NestedRepoDir
Write-Ok "Install layout verified"

# ── Install dependencies ─────────────────────────────────────────────────────
Write-Step "Syncing dependencies ($($selectedNames -join ' + '))..."
Write-Host "  Running uv sync (will skip already-installed packages)..."
$uvExtras = '--extra dev'
if ($installWhisper) { $uvExtras += ' --extra whisper' }
if ($installCanary)  { $uvExtras += ' --extra canary' }
Push-Location $InstallDir
Invoke-NativeCommand 'uv sync' ([scriptblock]::Create("uv sync $uvExtras"))
if ($installWhisper) {
    Invoke-NativeCommand 'Refresh faster-whisper' { uv pip install --python .venv\Scripts\python.exe --upgrade "faster-whisper>=1.1" }
}
Pop-Location
Write-Ok "Dependencies synced"

# ── Verify CUDA DLLs ─────────────────────────────────────────────────────────
Write-Step "Verifying CUDA runtime libraries..."
$cudaOk = $false
try {
    $pyScript = @'
import os, sys, importlib.util
spec = importlib.util.find_spec('nvidia')
if spec is None:
    print('SKIP: nvidia pip packages not installed')
    sys.exit(0)
found = []
for sp in spec.submodule_search_locations:
    if not os.path.isdir(sp): continue
    for child in os.listdir(sp):
        bdir = os.path.join(sp, child, 'bin')
        if os.path.isdir(bdir):
            dlls = [f for f in os.listdir(bdir) if f.endswith('.dll')]
            if dlls: found.extend(dlls)
if not found:
    print('WARN: No NVIDIA DLLs found in pip packages')
    sys.exit(1)
cublas = [f for f in found if 'cublas' in f.lower()]
cudnn  = [f for f in found if 'cudnn' in f.lower()]
if cublas: print(f'OK: cuBLAS DLLs: {cublas}')
else: print('WARN: cublas DLL not found'); sys.exit(1)
if cudnn: print(f'OK: cuDNN DLLs (sample): {cudnn[:3]}')
else: print('WARN: cuDNN DLL not found (non-critical)')
'@
    $cudaCheck = & "$InstallDir\.venv\Scripts\python.exe" -c $pyScript 2>&1
    foreach ($line in $cudaCheck) { Write-Host "  $line" }
    if ($LASTEXITCODE -eq 0) {
        $cudaOk = $true
        Write-Ok "CUDA runtime libraries verified"
    } else {
        Write-Warn "CUDA DLLs missing — GPU acceleration may fall back to CPU"
    }
} catch {
    Write-Warn "Could not verify CUDA DLLs: $_"
}

# ── Download models ──────────────────────────────────────────────────────────
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
        Push-Location $InstallDir
        Invoke-StreamingCommand 'Model download' { uv run cv2t download-model --engine whisper --target-dir $ModelsDir }
        Pop-Location
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
        Push-Location $InstallDir
        Invoke-StreamingCommand 'Canary model download' { uv run cv2t download-model --engine canary --target-dir $ModelsDir }
        Pop-Location
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
Write-Step "Checking directory permissions..."
$acl = Get-Acl $InstallDir
$existingRule = $acl.Access | Where-Object {
    $_.IdentityReference.Value -eq "BUILTIN\Users" -and
    $_.FileSystemRights -band [System.Security.AccessControl.FileSystemRights]::ReadAndExecute
}
if ($existingRule) {
    Write-Already "BUILTIN\\Users already has ReadAndExecute on $InstallDir"
} else {
    $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
        "BUILTIN\Users", "ReadAndExecute", "ContainerInherit,ObjectInherit", "None", "Allow"
    )
    $acl.SetAccessRule($rule)
    Set-Acl $InstallDir $acl
    Write-Ok "Permissions set on $InstallDir"
}

# ── Create desktop shortcut ──────────────────────────────────────────────────
Write-Step "Checking desktop shortcut..."
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "CV2T.lnk"
if (Test-Path $shortcutPath) {
    Write-Already "Desktop shortcut already exists at $shortcutPath"
} else {
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = "uv"
    $shortcut.Arguments = "run cv2t"
    $shortcut.WorkingDirectory = $InstallDir
    $shortcut.Description = "CV2T - Voice to Text"
    $shortcut.Save()
    Write-Ok "Shortcut created at $shortcutPath"
}

# ── Windows Defender exclusions ──────────────────────────────────────────────
Write-Step "Checking Windows Defender exclusions..."
try {
    $prefs = Get-MpPreference -ErrorAction SilentlyContinue
    $pathExclusions = @($prefs.ExclusionPath)
    $needInstallDir = $InstallDir -notin $pathExclusions
    $appDataDir = "$env:APPDATA\CV2T"
    $needAppData = $appDataDir -notin $pathExclusions

    if ($needInstallDir -or $needAppData) {
        if ($needInstallDir) { Add-MpPreference -ExclusionPath $InstallDir -ErrorAction SilentlyContinue }
        if ($needAppData)    { Add-MpPreference -ExclusionPath $appDataDir -ErrorAction SilentlyContinue }
        Write-Ok "Defender exclusions added"
    } else {
        Write-Already "Defender exclusions already configured"
    }
} catch {
    Write-Warn "Could not check/add Defender exclusions (non-critical)"
}

Write-Host "`n=== CV2T installed successfully ===" -ForegroundColor Green
Write-Host "  Install dir: $InstallDir"
Write-Host "  Models dir:  $ModelsDir"
Write-Host "  Launch with: cd '$InstallDir'; uv run cv2t"
