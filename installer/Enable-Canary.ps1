<#
.SYNOPSIS
    Install the Canary speech engine add-on for CV2T.

.DESCRIPTION
    Creates a separate Python environment (canary-env) alongside the
    CV2T binary install with torch, NeMo, and all Canary dependencies.
    Also downloads the Canary model (~3 GB).

    After completion, restart CV2T and select "canary" in Settings.

    Requires Administrator elevation and an internet connection.
    Approximate download: ~6 GB (PyTorch + NeMo + model).

.NOTES
    Run from the CV2T install directory:
        Right-click > Run with PowerShell
    Or from an elevated prompt:
        Set-ExecutionPolicy Bypass -Scope Process -Force
        .\Enable-Canary.ps1
#>

#Requires -Version 5.1

param(
    # Override the application directory. When launched from CV2T,
    # PyInstaller 6+ places data files in _internal/ instead of beside
    # the exe, so the caller passes the correct app root.
    [string]$AppDir,

    # Skip the automatic CV2T restart at the end.  Used by Test-CV2T.ps1
    # which launches the app itself after Enable-Canary finishes.
    [switch]$NoLaunch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# -- Determine install directory -----------------------------------------------
if (-not $AppDir) {
    $AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    # If we are inside _internal/ or installer/, move up to the real app directory
    if ((Split-Path -Leaf $AppDir) -in '_internal', 'installer') {
        $AppDir = Split-Path -Parent $AppDir
    }
}
$CanaryEnvDir = "$AppDir\canary-env"
$ModelsDir = "$AppDir\models"
$CanaryModelDir = "$ModelsDir\canary"
$VenvDir = "$CanaryEnvDir\.venv"

function Write-Step($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  [OK]   $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Skip($msg) { Write-Host "  [SKIP] $msg" -ForegroundColor DarkGray }

function Invoke-NativeCommand {
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

# -- Header -------------------------------------------------------------------
Write-Host ""
Write-Host "  +==============================================================+" -ForegroundColor Cyan
Write-Host "  |  CV2T -- Enable Canary Engine                               |" -ForegroundColor Cyan
Write-Host "  |                                                            |" -ForegroundColor Cyan
Write-Host "  |  This will install PyTorch + NVIDIA NeMo into a separate   |" -ForegroundColor Cyan
Write-Host "  |  Python environment alongside your CV2T installation.      |" -ForegroundColor Cyan
Write-Host "  |                                                            |" -ForegroundColor Cyan
Write-Host "  |  Approximate download: ~6 GB                               |" -ForegroundColor Cyan
Write-Host "  |  Disk space required:  ~10 GB                              |" -ForegroundColor Cyan
Write-Host "  +==============================================================+" -ForegroundColor Cyan
Write-Host ""

# -- Pre-flight: check for NVIDIA GPU -----------------------------------------
Write-Step "Checking for NVIDIA GPU..."
try {
    $gpu = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
    if ($gpu) {
        Write-Ok "GPU detected: $($gpu.Trim())"
    } else {
        Write-Warn "No NVIDIA GPU detected. Canary requires CUDA for inference."
        $proceed = Read-Host "  Continue anyway? [y/N]"
        if ($proceed -ne 'y') { exit 0 }
    }
} catch {
    Write-Warn "nvidia-smi not found. GPU status unknown."
}

# -- Check / install uv ------------------------------------------------------
Write-Step "Checking for uv package manager..."
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Skip "uv already installed: $(uv --version)"
} else {
    Write-Host "  Installing uv..."
    try {
        winget install --id astral-sh.uv --exact --accept-package-agreements --accept-source-agreements
        # Refresh PATH
        $machPath = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
        $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
        $env:PATH = "$userPath;$machPath"
    } catch {
        Write-Host "  winget failed; trying pip fallback..." -ForegroundColor Yellow
        pip install uv 2>$null
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "Failed to install uv. Install it manually: https://docs.astral.sh/uv/"
    }
    Write-Ok "uv installed: $(uv --version)"
}

# Tell uv to use the system Python
$env:UV_PYTHON_PREFERENCE = 'only-system'

# -- Check / install Python 3.11 ---------------------------------------------
Write-Step "Checking for Python 3.11..."
$py311 = (Get-Command python3.11 -ErrorAction SilentlyContinue).Source
if (-not $py311) {
    try { $py311 = (& py -3.11 -c "import sys; print(sys.executable)" 2>$null).Trim() } catch { $py311 = $null }
}
if ($py311) {
    Write-Skip "Python 3.11 available: $py311"
} else {
    Write-Host "  Installing Python 3.11 via winget..."
    winget install --id Python.Python.3.11 --exact --accept-package-agreements --accept-source-agreements
    $machPath = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
    $env:PATH = "$userPath;$machPath"
    $py311 = (Get-Command python3.11 -ErrorAction SilentlyContinue).Source
    if (-not $py311) {
        try { $py311 = (& py -3.11 -c "import sys; print(sys.executable)" 2>$null).Trim() } catch { $py311 = $null }
    }
    if (-not $py311) {
        throw "Python 3.11 not found after installation. Restart PowerShell and re-run."
    }
    Write-Ok "Python 3.11 installed"
}
Write-Ok "Using Python: $py311"

# -- Create canary-env --------------------------------------------------------
Write-Step "Setting up Canary environment at $CanaryEnvDir..."
if (-not (Test-Path $CanaryEnvDir)) {
    New-Item -ItemType Directory -Path $CanaryEnvDir -Force | Out-Null
}

# Create a minimal pyproject.toml for the canary-env
$canaryToml = @"
[project]
name = "cv2t-canary-env"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "nemo_toolkit[asr]>=2.0,<3",
    "torch>=2.1,<2.8",
    "accelerate",
    "huggingface-hub>=0.34.0",
    "datasets>=4.0",
    "onnxruntime>=1.16",
    "soundfile>=0.12",
    "numpy>=1.24",
]

[tool.uv]
required-environments = ["sys_platform == 'win32' and platform_machine == 'AMD64'"]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
"@

# Check whether the environment is already up-to-date
$tomlPath = "$CanaryEnvDir\pyproject.toml"
$canaryPython = "$VenvDir\Scripts\python.exe"
$envUpToDate = $false
if ((Test-Path $tomlPath) -and (Test-Path $canaryPython)) {
    $existingToml = (Get-Content $tomlPath -Raw -ErrorAction SilentlyContinue) -replace '\r\n', "`n"
    $normalizedNew = $canaryToml -replace '\r\n', "`n"
    if ($existingToml.Trim() -eq $normalizedNew.Trim()) {
        # Verify the lockfile exists (uv sync was completed previously)
        if (Test-Path "$CanaryEnvDir\uv.lock") {
            $envUpToDate = $true
        }
    }
}

if ($envUpToDate) {
    Write-Skip "Canary environment already up-to-date"
} else {
    $canaryToml | Out-File -FilePath $tomlPath -Encoding utf8
    Write-Host "  Running uv sync (this may take several minutes)..."
    Push-Location $CanaryEnvDir
    try {
        Invoke-NativeCommand 'uv sync' { uv sync --python $py311 }
    } finally {
        Pop-Location
    }
    Write-Ok "Canary environment created"
}

# -- Verify PyTorch CUDA -----------------------------------------------------
Write-Step "Verifying PyTorch CUDA support..."
$canaryPython = "$VenvDir\Scripts\python.exe"
if (-not (Test-Path $canaryPython)) {
    throw "Canary venv Python not found at $canaryPython"
}

$prevPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $canaryPython -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>&1 | Out-Null
} finally {
    $ErrorActionPreference = $prevPref
}
if ($LASTEXITCODE -ne 0) {
    Write-Warn "PyTorch CUDA not working -- reinstalling with CUDA 12.8..."
    Push-Location $CanaryEnvDir
    try {
        Invoke-NativeCommand 'Reinstall torch' {
            uv pip install --python .venv\Scripts\python.exe `
                --index-url https://download.pytorch.org/whl/cu128 `
                --upgrade --force-reinstall torch
        }
    } finally {
        Pop-Location
    }
}
Write-Ok "PyTorch CUDA verified"

# -- Verify NeMo import ------------------------------------------------------
Write-Step "Verifying all Canary dependencies..."
$depCheckScript = @'
import sys
errors = []

# Core packages used directly by canary_worker.py
for mod in ('numpy', 'soundfile', 'torch', 'accelerate', 'huggingface_hub'):
    try:
        __import__(mod)
    except ImportError as e:
        errors.append(f'{mod}: {e}')

# onnxruntime — used by NeMo ASR internals (not declared as a nemo dep)
try:
    from onnxruntime import InferenceSession
except ImportError as e:
    errors.append(f'onnxruntime (InferenceSession): {e}')

# Transitive deps required at runtime by NeMo's import chain
for mod in ('transformers', 'sentencepiece', 'omegaconf', 'lightning', 'peft', 'datasets.distributed'):
    try:
        __import__(mod)
    except ImportError as e:
        errors.append(f'{mod}: {e}')

# The actual model class import
try:
    from nemo.collections.speechlm2.models import SALM
except ImportError as e:
    errors.append(f'nemo SALM: {e}')

if errors:
    print('MISSING DEPENDENCIES:', file=sys.stderr)
    for e in errors:
        print(f'  - {e}', file=sys.stderr)
    sys.exit(1)
else:
    print('All Canary dependencies verified.')
'@
$prevPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $canaryPython -c $depCheckScript 2>&1 | ForEach-Object { Write-Host "  $_" }
} finally {
    $ErrorActionPreference = $prevPref
}
if ($LASTEXITCODE -ne 0) {
    Write-Warn "One or more Canary dependencies are missing or broken."
    Write-Host "  Try: cd '$CanaryEnvDir'; uv sync" -ForegroundColor Yellow
} else {
    Write-Ok "All Canary dependencies verified"
}

# -- Download / validate Canary model -----------------------------------------
Write-Step "Checking Canary model (~3 GB)..."

# Required files and minimum size for model.safetensors (~3 GB)
$requiredFiles = @('config.json', 'model.safetensors')
$minModelSizeBytes = 1GB   # safeguard against truncated downloads

$modelValid = $true
foreach ($f in $requiredFiles) {
    $fp = Join-Path $CanaryModelDir $f
    if (-not (Test-Path $fp)) {
        Write-Host "  Missing: $f" -ForegroundColor Yellow
        $modelValid = $false
    }
}
if ($modelValid) {
    $safetensorsSize = (Get-Item "$CanaryModelDir\model.safetensors").Length
    if ($safetensorsSize -lt $minModelSizeBytes) {
        Write-Host "  model.safetensors is only $([math]::Round($safetensorsSize / 1MB)) MB (expected ~3 GB) -- likely truncated" -ForegroundColor Yellow
        $modelValid = $false
    }
}

if ($modelValid) {
    Write-Skip "Canary model already present and valid in $CanaryModelDir"
    # Check for upstream updates (fast — only fetches metadata)
    Write-Host "  Checking for model updates..." -ForegroundColor DarkGray
}

# snapshot_download is safe to re-run: it checks local cache etags/metadata
# against the remote and only downloads changed or missing files.
if (-not $modelValid) {
    New-Item -ItemType Directory -Path $CanaryModelDir -Force | Out-Null
    Write-Host "  Downloading Canary model files..."
}
$downloadScript = @'
import sys, os
target_dir = sys.argv[1]
from huggingface_hub import snapshot_download
try:
    snapshot_download(
        repo_id='nvidia/canary-qwen-2.5b',
        local_dir=target_dir,
        local_files_only=False,
    )
    print('Download complete.')
except Exception as e:
    print('ERROR: ' + str(e), file=sys.stderr)
    sys.exit(1)
'@
$prevPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $canaryPython -c $downloadScript "$CanaryModelDir" 2>&1 | ForEach-Object { Write-Host "  $_" }
} finally {
    $ErrorActionPreference = $prevPref
}
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Canary model download/update failed."
    Write-Host "  You can retry later by re-running this script." -ForegroundColor Yellow
} else {
    # Final validation after download
    $postValid = $true
    foreach ($f in $requiredFiles) {
        if (-not (Test-Path (Join-Path $CanaryModelDir $f))) {
            Write-Warn "Expected file still missing after download: $f"
            $postValid = $false
        }
    }
    if ($postValid) {
        $finalSize = (Get-Item "$CanaryModelDir\model.safetensors").Length
        if ($finalSize -lt $minModelSizeBytes) {
            Write-Warn "model.safetensors is only $([math]::Round($finalSize / 1MB)) MB -- download may be incomplete"
        } else {
            Write-Ok "Canary model verified in $CanaryModelDir ($([math]::Round($finalSize / 1GB, 1)) GB)"
        }
    }
}

# -- Summary ------------------------------------------------------------------
Write-Host ""
Write-Host "  +==============================================================+" -ForegroundColor Green
Write-Host "  |  Canary Engine Installation Complete                        |" -ForegroundColor Green
Write-Host "  +==============================================================+" -ForegroundColor Green
Write-Host ""
Write-Host "  Environment: $CanaryEnvDir" -ForegroundColor DarkGray
Write-Host "  Model:       $CanaryModelDir" -ForegroundColor DarkGray
Write-Host ""

# -- Restart CV2T -------------------------------------------------------------
$cv2tExe = Join-Path $AppDir "cv2t.exe"
if ($NoLaunch) {
    Write-Skip "Auto-launch skipped (-NoLaunch)"
} elseif (Test-Path $cv2tExe) {
    Write-Step "Restarting CV2T..."
    # Stop the running instance so the new launch does not hit the
    # "Another instance is already running" single-instance guard.
    Get-Process -Name 'cv2t' -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2   # allow the process to fully exit
    Start-Process -FilePath $cv2tExe -WorkingDirectory $AppDir
    Write-Ok "CV2T launched - select 'canary' as the Engine in Settings."
} else {
    Write-Host "  Next steps:" -ForegroundColor White
    Write-Host "    1. Launch CV2T" -ForegroundColor White
    Write-Host "    2. Open Settings" -ForegroundColor White
    Write-Host "    3. Select 'canary' as the Engine" -ForegroundColor White
}
Write-Host ""

# -- Auto-close after 20 seconds ----------------------------------------------
Write-Host "  This window will close in 20 seconds..." -ForegroundColor DarkGray
Start-Sleep -Seconds 20
