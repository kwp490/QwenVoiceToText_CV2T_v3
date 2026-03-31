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
        .\Install-CV2T-Bin.ps1 -ZipPath ".\cv2t-v3.0.0-win64.zip"
#>

#Requires -RunAsAdministrator
#Requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$ZipPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$InstallDir = "C:\Program Files\CV2T"
$ModelsDir = "$InstallDir\models"
$ConfigDir = "$InstallDir\config"
$LogsDir   = "$InstallDir\logs"
$TempDir   = "$InstallDir\temp"
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

function Update-OutdatedFiles {
    <# Compare every file in SourceDir against DestDir; force-copy any that
       are missing or older in the destination.  Returns the count of files updated. #>
    param(
        [Parameter(Mandatory)] [string]$SourceDir,
        [Parameter(Mandatory)] [string]$DestDir,
        [string[]]$ExcludeDirs = @(),
        [string[]]$ExcludeExts = @()
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

# ── Verify & patch outdated files ─────────────────────────────────────────────
if ($ZipPath -and (Test-Path $ZipPath)) {
    # Zip was already extracted above; use a temp dir to compare
    $tempVerify = Join-Path $env:TEMP "cv2t-verify-$(Get-Random)"
    Expand-Archive -Path $ZipPath -DestinationPath $tempVerify -Force
    $verifySrc = $tempVerify
    # If the zip contains a nested directory, use that
    $nested = Get-ChildItem -Path $tempVerify -Directory
    if ($nested.Count -eq 1) { $verifySrc = $nested[0].FullName }
} else {
    $verifySrc = $srcDir
}

if ($verifySrc -and (Test-Path $verifySrc)) {
    Write-Step "Checking for outdated files in $InstallDir..."
    $outdated = Update-OutdatedFiles -SourceDir $verifySrc -DestDir $InstallDir
    if ($outdated -eq 0) {
        Write-Already "All files are up-to-date"
    } else {
        Write-Ok "$outdated file(s) updated"
    }
}

if ($tempVerify -and (Test-Path $tempVerify)) {
    Remove-Item -Recurse -Force $tempVerify
}

# ── Verify binary exists ─────────────────────────────────────────────────────
Write-Step "Verifying CV2T binary..."
if (Test-Path $ExePath) {
    Write-Ok "CV2T binary found at $ExePath"
} else {
    Write-Host "  ERROR: cv2t.exe not found at $ExePath" -ForegroundColor Red
    Write-Host "  The release zip may be structured differently. Check that cv2t.exe" -ForegroundColor Red
    Write-Host "  exists inside the extracted archive and re-run with the correct -ZipPath." -ForegroundColor Red
    exit 1
}

# ── Create data subdirectories ────────────────────────────────────────────────
foreach ($dir in @($ModelsDir, $ConfigDir, $LogsDir, $TempDir)) {
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
}

# ── Verify Canary dependencies ────────────────────────────────────────────────
if ($installCanary) {
    $venvPython = "$InstallDir\.venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        # Show antimalware notice if uv is available (it will be used for package repairs)
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            Write-Host ""
            Write-Host "  ┌─────────────────────────────────────────────────────────────────┐" -ForegroundColor Yellow
            Write-Host "  │  ANTIMALWARE NOTICE                                            │" -ForegroundColor Yellow
            Write-Host "  │                                                                │" -ForegroundColor Yellow
            Write-Host "  │  The next steps may use uv.exe (by Astral) to repair Python    │" -ForegroundColor Yellow
            Write-Host "  │  packages. Some antimalware tools (e.g. Malwarebytes) may      │" -ForegroundColor Yellow
            Write-Host "  │  flag or quarantine uv.exe as a false positive.                │" -ForegroundColor Yellow
            Write-Host "  │                                                                │" -ForegroundColor Yellow
            Write-Host "  │  If this happens, add uv.exe to your antimalware allow-list.  │" -ForegroundColor Yellow
            Write-Host "  │  uv is an open-source Python package manager:                  │" -ForegroundColor Yellow
            Write-Host "  │  https://github.com/astral-sh/uv                              │" -ForegroundColor Yellow
            Write-Host "  └─────────────────────────────────────────────────────────────────┘" -ForegroundColor Yellow
            Write-Host ""
            Read-Host "  Press Enter to continue"
        }

        # Ensure wandb is functional (NeMo imports it transitively)
        Write-Step "Verifying wandb installation..."
        $prevPref = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try { & $venvPython -c "from wandb.proto.wandb_telemetry_pb2 import Imports" 2>&1 | Out-Null }
        finally { $ErrorActionPreference = $prevPref }
        if ($LASTEXITCODE -ne 0) {
            if (Get-Command uv -ErrorAction SilentlyContinue) {
                Write-Warn "wandb is broken or missing — reinstalling..."
                Push-Location $InstallDir
                Invoke-NativeCommand 'Fix wandb' { uv pip install --python .venv\Scripts\python.exe --reinstall-package wandb wandb }
                Pop-Location
                Write-Ok "wandb reinstalled"
            } else {
                Write-Warn "wandb is broken (uv not available to fix automatically)"
            }
        } else {
            Write-Already "wandb is functional"
        }

        # Ensure PyTorch has CUDA support
        Write-Step "Verifying PyTorch CUDA support..."
        $prevPref = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try { & $venvPython -c "import torch; assert torch.cuda.is_available()" 2>&1 | Out-Null }
        finally { $ErrorActionPreference = $prevPref }
        if ($LASTEXITCODE -ne 0) {
            if (Get-Command uv -ErrorAction SilentlyContinue) {
                Write-Warn "PyTorch does not have CUDA support — reinstalling with CUDA 12.8..."
                Push-Location $InstallDir
                Invoke-NativeCommand 'Install torch+CUDA' {
                    uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 --upgrade --force-reinstall torch
                }
                Pop-Location
                Write-Ok "PyTorch with CUDA reinstalled"
            } else {
                Write-Warn "PyTorch lacks CUDA support (uv not available to fix automatically)"
            }
        } else {
            Write-Already "PyTorch has CUDA support"
        }

        # Verify GPU kernels actually work (catches arch mismatch, e.g. Blackwell + cu124)
        Write-Step "Verifying PyTorch GPU kernel compatibility..."
        $prevPref = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try { & $venvPython -c "import torch; torch.zeros(1, device='cuda')" 2>&1 | Out-Null }
        finally { $ErrorActionPreference = $prevPref }
        if ($LASTEXITCODE -ne 0) {
            if (Get-Command uv -ErrorAction SilentlyContinue) {
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
                Write-Warn "PyTorch GPU kernels failed (uv not available to fix automatically)"
            }
        } else {
            Write-Ok "PyTorch GPU kernels verified for this GPU"
        }

        # Verify huggingface-hub compatibility
        Write-Step "Checking huggingface-hub version compatibility..."
        $hfVer = & $venvPython -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>$null
        $needsFix = $false
        if (-not $hfVer) {
            $needsFix = $true
            Write-Warn "huggingface-hub not found in virtual environment"
        } else {
            $parts = $hfVer.Trim().Split('.')
            $major = [int]$parts[0]
            $minor = [int]$parts[1]
            if ($major -ge 1 -or ($major -eq 0 -and $minor -lt 34)) {
                $needsFix = $true
                Write-Warn "huggingface-hub $hfVer is outside required range (>=0.34.0,<1.0) for transformers"
            } else {
                Write-Already "huggingface-hub $hfVer is compatible"
            }
        }
        if ($needsFix -and (Get-Command uv -ErrorAction SilentlyContinue)) {
            Write-Host "  Installing compatible version..."
            Push-Location $InstallDir
            Invoke-NativeCommand 'Fix huggingface-hub' { uv pip install --python .venv\Scripts\python.exe "huggingface-hub>=0.34.0,<1.0" }
            Pop-Location
            Write-Ok "huggingface-hub pinned to compatible version"
        } elseif ($needsFix) {
            Write-Warn "Cannot fix automatically (uv not available)."
            Write-Host "  If uv is installed, run manually: uv pip install --python .venv\Scripts\python.exe `"huggingface-hub>=0.34.0,<1.0`"" -ForegroundColor Yellow
        }
    } else {
        Write-Warn "Cannot verify Canary dependencies — no virtual environment found."
        Write-Host "  Binary installs bundle dependencies. If Canary fails at runtime:" -ForegroundColor Yellow
        Write-Host "    1. Install uv: winget install astral-sh.uv" -ForegroundColor Yellow
        Write-Host "    2. Install torch: uv pip install --index-url https://download.pytorch.org/whl/cu128 torch" -ForegroundColor Yellow
        Write-Host "    3. Or switch to Whisper engine in Settings." -ForegroundColor Yellow
    }
}

# ── Verify CUDA DLLs (Whisper needs cublas/cudnn) ─────────────────────────
if ($installWhisper) {
    Write-Step "Verifying CUDA runtime libraries for Whisper..."
    $internalDlls = Get-ChildItem -Path $InstallDir -Recurse -Filter "cublas64*.dll" -ErrorAction SilentlyContinue
    if ($internalDlls) {
        Write-Ok "cuBLAS DLL found: $($internalDlls[0].Name)"
    } else {
        Write-Warn "cuBLAS DLL not found in install directory"
        Write-Host "  Whisper (CTranslate2) requires NVIDIA cuBLAS and cuDNN DLLs." -ForegroundColor Yellow
        Write-Host "  If GPU transcription fails, install the CUDA Toolkit from:" -ForegroundColor Yellow
        Write-Host "    https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
        Write-Host "  Or ensure nvidia-cublas-cu12 and nvidia-cudnn-cu12 pip packages" -ForegroundColor Yellow
        Write-Host "  are bundled in the release build." -ForegroundColor Yellow
    }
    $cudnnDlls = Get-ChildItem -Path $InstallDir -Recurse -Filter "cudnn*.dll" -ErrorAction SilentlyContinue
    if ($cudnnDlls) {
        Write-Ok "cuDNN DLL found: $($cudnnDlls[0].Name)"
    } else {
        Write-Warn "cuDNN DLL not found — GPU acceleration may fall back to CPU"
    }
}

# ── Migrate existing data from old locations ──────────────────────────────────
Write-Step "Checking for data to migrate from previous install..."

# Migrate settings.json from %APPDATA%\CV2T
$oldSettingsFile = "$env:APPDATA\CV2T\settings.json"
$newSettingsFile = Join-Path $ConfigDir "settings.json"
if ((Test-Path $oldSettingsFile) -and -not (Test-Path $newSettingsFile)) {
    Copy-Item -Path $oldSettingsFile -Destination $newSettingsFile -Force
    Write-Ok "Migrated settings.json from $oldSettingsFile"
} else {
    Write-Already "No settings to migrate (already present or no old settings found)"
}

# Migrate models from %LOCALAPPDATA%\CV2T\models
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

# Migrate log files from %APPDATA%\CV2T
$oldLogDir = "$env:APPDATA\CV2T"
foreach ($logFile in @("cv2t.log", "cv2t.log.1", "cv2t.log.2")) {
    $oldLog = Join-Path $oldLogDir $logFile
    $newLog = Join-Path $LogsDir $logFile
    if ((Test-Path $oldLog) -and -not (Test-Path $newLog)) {
        Copy-Item -Path $oldLog -Destination $newLog -Force
    }
}

# ── Download models ───────────────────────────────────────────────────────────

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
        Invoke-StreamingCommand 'Model download' { & $ExePath download-model --engine whisper --target-dir $ModelsDir }
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
        Invoke-StreamingCommand 'Canary model download' { & $ExePath download-model --engine canary --target-dir $ModelsDir }
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
$cfg | ConvertTo-Json -Depth 10 | ForEach-Object {
    [System.IO.File]::WriteAllText($settingsFile, $_, (New-Object System.Text.UTF8Encoding $false))
}
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
    Write-Host "  Defender exclusions added" -ForegroundColor Green
} catch {
    Write-Host "  Could not add Defender exclusions (non-critical)" -ForegroundColor Yellow
}

Write-Host "`n=== CV2T installed successfully ===" -ForegroundColor Green
Write-Host "  Install dir: $InstallDir"
Write-Host "  Models dir:  $ModelsDir"
Write-Host "  Config dir:  $ConfigDir"
Write-Host "  Logs dir:    $LogsDir"
Write-Host "  Launch:      $ExePath"
Write-Host ""
Write-Host "  If transcription fails with a CUDA error:" -ForegroundColor Yellow
Write-Host "    - Ensure you have the latest NVIDIA GPU driver from https://www.nvidia.com/drivers" -ForegroundColor Yellow
Write-Host "    - RTX 50-series (Blackwell) requires driver 560+ and CUDA 12.8+" -ForegroundColor Yellow
