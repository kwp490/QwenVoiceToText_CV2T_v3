<#
.SYNOPSIS
    Install CV2T from source (developer/contributor path — supports both engines).

.DESCRIPTION
    Copies the local CV2T source tree to the install directory, installs
    Python 3.11 via uv, syncs all dependencies, downloads model weights,
    and creates a desktop shortcut.

    Requires Administrator elevation. Installs everything to C:\Program Files\CV2T\
    (binaries, models, config, logs, temp).

.NOTES
    Run in an elevated PowerShell session from within the repo:
        Set-ExecutionPolicy Bypass -Scope Process -Force
        .\installer\Install-CV2T-Source.ps1
#>

#Requires -RunAsAdministrator
#Requires -Version 5.1

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$InstallDir = "C:\Program Files\CV2T"
$ModelsDir = "$InstallDir\models"
$ConfigDir = "$InstallDir\config"
$LogsDir   = "$InstallDir\logs"
$TempDir   = "$InstallDir\temp"
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

function Update-OutdatedFiles {
    <# Compare every file in SourceDir against DestDir; force-copy any that
       are missing or older in the destination.  Returns the count of files updated. #>
    param(
        [Parameter(Mandatory)] [string]$SourceDir,
        [Parameter(Mandatory)] [string]$DestDir,
        [string[]]$ExcludeDirs = @('.git', '__pycache__', '.venv'),
        [string[]]$ExcludeExts = @('.pyc')
    )

    $updated = 0
    $srcItems = Get-ChildItem -Path $SourceDir -File -Recurse -Force
    foreach ($srcFile in $srcItems) {
        $rel = $srcFile.FullName.Substring($SourceDir.TrimEnd('\').Length + 1)

        # Skip excluded directories
        $skip = $false
        foreach ($exDir in $ExcludeDirs) {
            if ($rel -like "$exDir\*" -or $rel -like "*\$exDir\*") { $skip = $true; break }
        }
        if ($skip) { continue }

        # Skip excluded extensions
        foreach ($exExt in $ExcludeExts) {
            if ($srcFile.Extension -eq $exExt) { $skip = $true; break }
        }
        if ($skip) { continue }

        $destFile = Join-Path $DestDir $rel
        if (-not (Test-Path $destFile)) {
            $destParent = Split-Path $destFile -Parent
            if (-not (Test-Path $destParent)) {
                New-Item -ItemType Directory -Path $destParent -Force | Out-Null
            }
            Copy-Item -Path $srcFile.FullName -Destination $destFile -Force
            Write-Host "  [NEW]  $rel" -ForegroundColor Yellow
            $updated++
        } elseif ($srcFile.LastWriteTimeUtc -gt (Get-Item $destFile).LastWriteTimeUtc) {
            Copy-Item -Path $srcFile.FullName -Destination $destFile -Force
            Write-Host "  [UPD]  $rel" -ForegroundColor Yellow
            $updated++
        }
    }
    return $updated
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
        robocopy $SourceDir $DestinationDir /MIR /XD .git __pycache__ .venv models config logs temp installer /XF "*.pyc" /NFL /NDL /NJH /NJS /NC /NS /NP 2>&1 | Out-Null
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

# ── Verify & patch outdated files ─────────────────────────────────────────────
if ($RepoRoot -ne $InstallDir) {
    Write-Step "Checking for outdated files in $InstallDir..."
    $outdated = Update-OutdatedFiles -SourceDir $RepoRoot -DestDir $InstallDir
    if ($outdated -eq 0) {
        Write-Already "All files are up-to-date"
    } else {
        Write-Ok "$outdated file(s) updated"
    }
}

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

# ── Validate virtual environment and core imports ─────────────────────────────
Write-Step "Validating virtual environment..."
$venvPython = "$InstallDir\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "  ERROR: Virtual environment not found at $InstallDir\.venv" -ForegroundColor Red
    Write-Host "  Try deleting $InstallDir\.venv and re-running this installer." -ForegroundColor Red
    exit 1
}
$pyVer = & $venvPython --version 2>&1
Write-Ok "venv Python: $pyVer"

Write-Step "Verifying core Python imports..."
$coreImportScript = @'
import sys, importlib
failed = []
for mod in ['PySide6', 'sounddevice', 'soundfile', 'numpy', 'keyboard']:
    try:
        importlib.import_module(mod)
    except ImportError as e:
        failed.append(f'{mod}: {e}')
if failed:
    for f in failed: print(f'FAIL: {f}')
    sys.exit(1)
else:
    print('All core imports OK')
'@
$prevPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    $importResult = & $venvPython -c $coreImportScript 2>&1
    foreach ($line in $importResult) { Write-Host "  $line" }
} finally {
    $ErrorActionPreference = $prevPref
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Core dependencies are missing. Try:" -ForegroundColor Red
    Write-Host "    cd '$InstallDir'; uv sync $uvExtras" -ForegroundColor Yellow
    exit 1
}
Write-Ok "Core imports verified"

if ($installWhisper) {
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $venvPython -c "import faster_whisper" 2>&1 | Out-Null }
    finally { $ErrorActionPreference = $prevPref }
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "faster-whisper import failed. Whisper engine will not work."
        Write-Host "  Try: cd '$InstallDir'; uv pip install --python .venv\Scripts\python.exe --upgrade 'faster-whisper>=1.1'" -ForegroundColor Yellow
    } else {
        Write-Ok "faster-whisper import OK"
    }
}

if ($installCanary) {
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $venvPython -c "import torch" 2>&1 | Out-Null }
    finally { $ErrorActionPreference = $prevPref }
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "torch import failed. Canary engine will not work."
        Write-Host "  Try: cd '$InstallDir'; uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 torch" -ForegroundColor Yellow
    } else {
        Write-Ok "torch import OK"
    }
}

# ── Ensure PyTorch has CUDA support (Canary) ──────────────────────────────────
if ($installCanary) {
    Write-Step "Verifying PyTorch CUDA support..."
    $venvPython = "$InstallDir\.venv\Scripts\python.exe"
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $venvPython -c "import torch; assert torch.cuda.is_available()" 2>&1 | Out-Null }
    finally { $ErrorActionPreference = $prevPref }
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "PyTorch does not have CUDA support — reinstalling with CUDA 12.8..."
        Push-Location $InstallDir
        Invoke-NativeCommand 'Install torch+CUDA' {
            uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 --upgrade --force-reinstall torch
        }
        Pop-Location
        Write-Ok "PyTorch with CUDA reinstalled"
    } else {
        Write-Already "PyTorch has CUDA support"
    }

    # Verify GPU kernels actually work (catches arch mismatch, e.g. Blackwell + cu124)
    Write-Step "Verifying PyTorch GPU kernel compatibility..."
    $venvPython = "$InstallDir\.venv\Scripts\python.exe"
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $venvPython -c "import torch; torch.zeros(1, device='cuda')" 2>&1 | Out-Null }
    finally { $ErrorActionPreference = $prevPref }
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "PyTorch CUDA kernels failed — GPU arch may require a newer CUDA toolkit"
        Write-Host "  Reinstalling torch from cu128 index (includes Blackwell/sm_120 support)..."
        Push-Location $InstallDir
        Invoke-NativeCommand 'Upgrade torch for GPU arch' {
            uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 --upgrade --force-reinstall torch
        }
        Pop-Location
        # Re-verify after reinstall
        $prevPref2 = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try { & $venvPython -c "import torch; torch.zeros(1, device='cuda')" 2>&1 | Out-Null }
        finally { $ErrorActionPreference = $prevPref2 }
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "GPU kernel test still fails after torch reinstall — Canary will fall back to CPU"
        } else {
            Write-Ok "PyTorch GPU kernels working after reinstall"
        }
    } else {
        Write-Ok "PyTorch GPU kernels verified for this GPU"
    }
}

# ── Verify huggingface-hub compatibility (Canary) ────────────────────────────
if ($installCanary) {
    Write-Step "Checking huggingface-hub version compatibility..."
    $venvPython = "$InstallDir\.venv\Scripts\python.exe"

    # Ensure wandb is functional (NeMo imports it transitively)
    Write-Host "  Verifying wandb installation..."
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $venvPython -c "from wandb.proto.wandb_telemetry_pb2 import Imports" 2>&1 | Out-Null }
    finally { $ErrorActionPreference = $prevPref }
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "wandb is broken or missing — reinstalling..."
        Push-Location $InstallDir
        Invoke-NativeCommand 'Fix wandb' { uv pip install --python .venv\Scripts\python.exe --force-reinstall wandb }
        Pop-Location
        Write-Ok "wandb reinstalled"
    } else {
        Write-Already "wandb is functional"
    }

    $hfVer = & $venvPython -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>$null
    if (-not $hfVer) {
        Write-Host "  huggingface-hub not found, installing compatible version..."
        Push-Location $InstallDir
        Invoke-NativeCommand 'Install huggingface-hub' { uv pip install --python .venv\Scripts\python.exe "huggingface-hub>=0.34.0,<1.0" }
        Pop-Location
        Write-Ok "huggingface-hub installed"
    } else {
        $parts = $hfVer.Trim().Split('.')
        $major = [int]$parts[0]
        $minor = [int]$parts[1]
        if ($major -ge 1 -or ($major -eq 0 -and $minor -lt 34)) {
            Write-Warn "huggingface-hub $hfVer is outside required range (>=0.34.0,<1.0) for transformers"
            Write-Host "  Installing compatible version..."
            Push-Location $InstallDir
            Invoke-NativeCommand 'Fix huggingface-hub' { uv pip install --python .venv\Scripts\python.exe "huggingface-hub>=0.34.0,<1.0" }
            Pop-Location
            Write-Ok "huggingface-hub pinned to compatible version"
        } else {
            Write-Already "huggingface-hub $hfVer is compatible"
        }
    }
}

# ── Verify CUDA DLLs ─────────────────────────────────────────────────────────
Write-Step "Verifying CUDA runtime libraries..."
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
        Write-Ok "CUDA runtime libraries verified"
    } else {
        Write-Warn "CUDA DLLs missing — GPU acceleration may fall back to CPU"
    }
} catch {
    Write-Warn "Could not verify CUDA DLLs: $_"
}

# ── Download models ──────────────────────────────────────────────────────────
foreach ($dir in @($ModelsDir, $ConfigDir, $LogsDir, $TempDir)) {
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
}

# ── Migrate existing data from old locations ──────────────────────────────────
Write-Step "Checking for data to migrate from previous install..."

$oldSettingsFile = "$env:APPDATA\CV2T\settings.json"
$newSettingsFile = Join-Path $ConfigDir "settings.json"
if ((Test-Path $oldSettingsFile) -and -not (Test-Path $newSettingsFile)) {
    Copy-Item -Path $oldSettingsFile -Destination $newSettingsFile -Force
    Write-Ok "Migrated settings.json from $oldSettingsFile"
} else {
    Write-Already "No settings to migrate (already present or no old settings found)"
}

$oldModelsDir = "$env:LOCALAPPDATA\CV2T\models"
if (Test-Path $oldModelsDir) {
    $migrated = 0
    foreach ($engineDir in (Get-ChildItem -Path $oldModelsDir -Directory)) {
        $destEngine = Join-Path $ModelsDir $engineDir.Name
        if (-not (Test-Path $destEngine)) {
            Write-Host "  Migrating $($engineDir.Name) model..."
            Copy-Item -Path $engineDir.FullName -Destination $destEngine -Recurse -Force
            $migrated++
            Write-Ok "Migrated $($engineDir.Name) model from $($engineDir.FullName)"
        }
    }
    if ($migrated -eq 0) {
        Write-Already "No models to migrate (already present in new location)"
    }
} else {
    Write-Already "No old model directory found at $oldModelsDir"
}

$oldLogDir = "$env:APPDATA\CV2T"
foreach ($logFile in @("cv2t.log", "cv2t.log.1", "cv2t.log.2")) {
    $oldLog = Join-Path $oldLogDir $logFile
    $newLog = Join-Path $LogsDir $logFile
    if ((Test-Path $oldLog) -and -not (Test-Path $newLog)) {
        Copy-Item -Path $oldLog -Destination $newLog -Force
    }
}
if ($installWhisper) {
    Write-Step "Checking Whisper model..."
    $whisperDir = Join-Path $ModelsDir "whisper"
    $whisperFiles = @("config.json", "model.bin", "tokenizer.json")
    $whisperReady = $true
    foreach ($f in $whisperFiles) {
        if (-not (Test-Path (Join-Path $whisperDir $f))) { $whisperReady = $false; break }
    }
    if ($whisperReady) {
        Write-Already "Whisper model already present in $whisperDir"
    } else {
        Write-Host "  Downloading Whisper model 'large-v3-turbo'..."
        Write-Host "  Using repo mobiuslabsgmbh/faster-whisper-large-v3-turbo"
        Push-Location $InstallDir
        Invoke-StreamingCommand 'Model download' { uv run cv2t download-model --engine whisper --target-dir $ModelsDir }
        Pop-Location
        Write-Ok "Whisper model downloaded to $whisperDir"
    }
}

if ($installCanary) {
    Write-Step "Checking Canary model..."
    $canaryDir = Join-Path $ModelsDir "canary"
    # Validate it's a NeMo SALM model (not an old ONNX download)
    $canaryConfigFile = Join-Path $canaryDir "config.json"
    $canaryIsNemo = $false
    if (Test-Path $canaryConfigFile) {
        $canaryConfig = Get-Content $canaryConfigFile -Raw | ConvertFrom-Json
        if ($canaryConfig.PSObject.Properties.Match('audio_locator_tag').Count -gt 0) {
            $canaryIsNemo = $true
        }
    }
    if ($canaryIsNemo) {
        Write-Already "Canary NeMo model already present in $canaryDir"
    } else {
        if (Test-Path $canaryDir) {
            Write-Warn "Removing incompatible Canary model (ONNX) from $canaryDir..."
            Remove-Item -Recurse -Force $canaryDir
        }
        Write-Host "  Downloading Canary NeMo SALM model (nvidia/canary-qwen-2.5b)..."
        Push-Location $InstallDir
        Invoke-StreamingCommand 'Canary model download' { uv run cv2t download-model --engine canary --target-dir $ModelsDir }
        Pop-Location
        Write-Ok "Canary model downloaded to $canaryDir"
    }
}

# ── Write default engine to settings ─────────────────────────────────────────
Write-Step "Configuring default engine..."
$settingsFile = Join-Path $ConfigDir "settings.json"
$cfg = $null
if ($installCanary -and -not $installWhisper) {
    $defaultEngine = 'canary'
} else {
    $defaultEngine = 'whisper'
}
if (Test-Path $settingsFile) {
    $rawSettings = Get-Content $settingsFile -Raw
    if (-not [string]::IsNullOrWhiteSpace($rawSettings)) {
        $cfg = $rawSettings | ConvertFrom-Json
    }
}
if (-not $cfg) {
    $cfg = [pscustomobject]@{}
}
if ($cfg.PSObject.Properties.Match('engine').Count -eq 0) {
    $cfg | Add-Member -NotePropertyName 'engine' -NotePropertyValue $defaultEngine
} else {
    $cfg.engine = $defaultEngine
}
$jsonText = $cfg | ConvertTo-Json -Depth 10
[System.IO.File]::WriteAllText($settingsFile, $jsonText, (New-Object System.Text.UTF8Encoding $false))
Write-Ok "Default engine set to '$defaultEngine' in $settingsFile"

# ── Set permissions (current user gets Modify on install dir) ────────────────
Write-Step "Checking directory permissions..."
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$acl = Get-Acl $InstallDir
$existingRule = $acl.Access | Where-Object {
    $_.IdentityReference.Value -eq $currentUser -and
    $_.FileSystemRights -band [System.Security.AccessControl.FileSystemRights]::Modify
}
if ($existingRule) {
    Write-Already "$currentUser already has Modify on $InstallDir"
} else {
    $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
        $currentUser, "Modify", "ContainerInherit,ObjectInherit", "None", "Allow"
    )
    $acl.SetAccessRule($rule)
    Set-Acl $InstallDir $acl
    Write-Ok "Granted Modify permission to $currentUser on $InstallDir"
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

    if ($needInstallDir) {
        Add-MpPreference -ExclusionPath $InstallDir -ErrorAction SilentlyContinue
        Write-Ok "Defender exclusions added"
    } else {
        Write-Already "Defender exclusions already configured"
    }
} catch {
    Write-Warn "Could not check/add Defender exclusions (non-critical)"
}

# ── Final smoke test ──────────────────────────────────────────────────────────
Write-Step "Running smoke test..."
$prevPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    Push-Location $InstallDir
    $smokeResult = & uv run cv2t --version 2>&1
    Pop-Location
    foreach ($line in $smokeResult) { Write-Host "  $line" }
} finally {
    $ErrorActionPreference = $prevPref
}
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Smoke test failed — CV2T may not launch correctly"
    Write-Host "  Try running manually: cd '$InstallDir'; uv run cv2t --version" -ForegroundColor Yellow
    Write-Host "  Check logs at: $LogsDir\cv2t.log" -ForegroundColor Yellow
} else {
    Write-Ok "Smoke test passed: $($smokeResult -join ' ')"
}

Write-Host "`n=== CV2T installed successfully ===" -ForegroundColor Green
Write-Host "  Install dir: $InstallDir"
Write-Host "  Models dir:  $ModelsDir"
Write-Host "  Config dir:  $ConfigDir"
Write-Host "  Logs dir:    $LogsDir"
Write-Host "  Launch with: cd '$InstallDir'; uv run cv2t"
