<#
.SYNOPSIS
    Quick-test launcher for CV2T -- run tests and launch the app without manual build/install cycles.

.DESCRIPTION
    Two testing modes:

      1. Release Build Test
         Builds the installer (PyInstaller + Inno Setup), silently uninstalls
         the old version (keeping models), silently installs the new build,
         and launches cv2t.exe.  Requires admin -- auto-elevates if needed.

      2. Source Test
         Runs directly from source with CV2T_HOME pointed at a dev-temp
         folder.  No system changes -- the installed release build is untouched.

    Both modes sync dependencies and run the test suite before launching.

.PARAMETER Mode
    Skip the interactive menu.  Valid values: Release, Source.

.EXAMPLE
    .\Test-CV2T.ps1                 # interactive menu
    .\Test-CV2T.ps1 -Mode Source    # skip menu, run from source
    .\Test-CV2T.ps1 -Mode Release   # skip menu, full release cycle

.NOTES
    Run from the repository root.
#>

[CmdletBinding()]
param(
    [ValidateSet('Release', 'Source')]
    [string]$Mode
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# -- Resolve repo root ---------------------------------------------------------
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $RepoRoot

# -- Helpers (match Build-Installer.ps1 style) --------------------------------
function Write-Step($msg)  { Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "  [ERROR] $msg" -ForegroundColor Red }
function Write-Info($msg)  { Write-Host "  $msg" -ForegroundColor DarkGray }

function Exit-Script([int]$code = 1) {
    Pop-Location
    exit $code
}

# -- Interactive menu ----------------------------------------------------------
if (-not $Mode) {
    Write-Host ""
    Write-Host "  =========================================" -ForegroundColor Cyan
    Write-Host "       CV2T -- Quick Test Launcher         " -ForegroundColor Cyan
    Write-Host "  =========================================" -ForegroundColor Cyan
    Write-Host "  1) Release Build Test                    " -ForegroundColor White
    Write-Host "     Build .exe, uninstall, install, launch" -ForegroundColor DarkGray
    Write-Host "     (requires admin)                      " -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  2) Source Test                           " -ForegroundColor White
    Write-Host "     Run from source, no system changes    " -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  Q) Quit                                  " -ForegroundColor White
    Write-Host "  =========================================" -ForegroundColor Cyan
    Write-Host ""

    $choice = Read-Host "  Select an option"
    switch ($choice.ToLower()) {
        '1' { $Mode = 'Release' }
        '2' { $Mode = 'Source'  }
        'q' { Write-Host "  Bye."; Exit-Script 0 }
        default {
            Write-Err "Invalid choice '$choice'"
            Exit-Script 1
        }
    }
}

Write-Step "Mode: $Mode"


# ==============================================================================
#  PHASE 1 -- Pre-flight checks (shared)
# ==============================================================================

Write-Step "Pre-flight checks..."

# 1a. Verify uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Err "uv not found on PATH."
    Write-Info "Install it: irm https://astral.sh/uv/install.ps1 | iex"
    Exit-Script 1
}
Write-Ok "uv found: $(uv --version 2>$null)"

# 1b. Verify pyproject.toml exists
if (-not (Test-Path "pyproject.toml")) {
    Write-Err "pyproject.toml not found. Run this script from the repository root."
    Exit-Script 1
}
Write-Ok "pyproject.toml found"

# 1c. Sync dependencies
Write-Step "Syncing dependencies (uv sync --extra all --extra dev)..."
$prevPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    uv sync --extra all --extra dev 2>&1 | ForEach-Object { Write-Host "  $_" }
} finally {
    $ErrorActionPreference = $prevPref
}
if ($LASTEXITCODE -ne 0) {
    Write-Err "uv sync failed (exit code $LASTEXITCODE)."
    Exit-Script 1
}
Write-Ok "Dependencies synced (whisper + canary + dev)"


# ==============================================================================
#  PHASE 2 -- Run test suite (shared)
# ==============================================================================

Write-Step "Running test suite (uv run pytest tests/ -v)..."
$prevPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    uv run pytest tests/ -v 2>&1 | ForEach-Object { Write-Host "  $_" }
} finally {
    $ErrorActionPreference = $prevPref
}

$testExit = $LASTEXITCODE
if ($testExit -ne 0) {
    Write-Host ""
    Write-Warn "Test suite failed (exit code $testExit)."
    Write-Host ""
    $continue = Read-Host "  Continue anyway? (y/N)"
    if ($continue -notin 'y', 'Y') {
        Write-Host "  Aborted." -ForegroundColor Red
        Exit-Script 1
    }
    Write-Warn "Continuing despite test failures."
} else {
    Write-Ok "All tests passed"
}


# ==============================================================================
#  MODE: Release Build Test
# ==============================================================================

if ($Mode -eq 'Release') {

    # -- Admin elevation -------------------------------------------------------
    $isAdmin = ([Security.Principal.WindowsPrincipal] `
        [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if (-not $isAdmin) {
        Write-Warn "Release mode requires admin privileges. Elevating..."
        $scriptPath = $MyInvocation.MyCommand.Path
        try {
            Start-Process powershell.exe -Verb RunAs -ArgumentList @(
                '-NoProfile', '-ExecutionPolicy', 'Bypass',
                '-File', "`"$scriptPath`"",
                '-Mode', 'Release'
            )
        } catch {
            Write-Err "Failed to elevate: $_"
            Exit-Script 1
        }
        Write-Info "Elevated process started. This window can be closed."
        Exit-Script 0
    }

    Write-Ok "Running as Administrator"

    # -- Step 1: Sync canary-env dependencies ----------------------------------
    $canaryEnvDir = Join-Path $RepoRoot "canary-env"
    if (Test-Path (Join-Path $canaryEnvDir "pyproject.toml")) {
        Write-Step "Syncing canary-env dependencies..."
        Push-Location $canaryEnvDir
        $prevPref = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            uv sync 2>&1 | ForEach-Object { Write-Host "  $_" }
        } finally {
            $ErrorActionPreference = $prevPref
        }
        Pop-Location
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "canary-env sync failed (exit code $LASTEXITCODE). Canary may not work."
        } else {
            Write-Ok "canary-env dependencies synced"
        }
    }

    # -- Step 2: Build installer -----------------------------------------------
    Write-Step "Building installer..."
    $buildScript = Join-Path $RepoRoot "installer\Build-Installer.ps1"
    if (-not (Test-Path $buildScript)) {
        Write-Err "installer\Build-Installer.ps1 not found."
        Exit-Script 1
    }

    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $buildScript 2>&1 | ForEach-Object { Write-Host "  $_" }
    } finally {
        $ErrorActionPreference = $prevPref
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Build-Installer.ps1 failed (exit code $LASTEXITCODE)."
        Exit-Script 1
    }
    Write-Ok "Installer built successfully"

    # -- Step 3: Silent uninstall ----------------------------------------------
    Write-Step "Checking for existing CV2T installation..."

    $uninstallKey = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{B7E9F3A1-9C2D-4E5F-8A1B-3D7C6E4F2A9B}_is1'
    # Also check WOW6432Node for 32-bit Inno Setup entries
    $uninstallKeyWow = 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\{B7E9F3A1-9C2D-4E5F-8A1B-3D7C6E4F2A9B}_is1'

    $uninstallString = $null
    foreach ($key in @($uninstallKey, $uninstallKeyWow)) {
        if (Test-Path $key) {
            $regEntry = Get-ItemProperty -Path $key -ErrorAction SilentlyContinue
            if ($regEntry.UninstallString) {
                $uninstallString = $regEntry.UninstallString
                break
            }
        }
    }

    if ($uninstallString) {
        # Strip any existing quotes from the path
        $uninstallerPath = $uninstallString -replace '"', ''
        Write-Info "Found uninstaller: $uninstallerPath"
        Write-Step "Silently uninstalling old version (models will be preserved)..."

        $prevPref = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $proc = Start-Process -FilePath $uninstallerPath `
                -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART' `
                -Wait -PassThru
        } finally {
            $ErrorActionPreference = $prevPref
        }

        if ($proc.ExitCode -ne 0) {
            Write-Warn "Uninstaller exited with code $($proc.ExitCode) (may be okay)."
        } else {
            Write-Ok "Old version uninstalled (models preserved)"
        }
    } else {
        Write-Info "No existing CV2T installation found -- skipping uninstall."
    }

    # -- Step 4: Silent install ------------------------------------------------
    Write-Step "Installing new build..."

    $setupExe = Get-ChildItem "installer\Output\CV2T-Setup-*.exe" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $setupExe) {
        Write-Err "No installer found in installer\Output\. Build may have failed."
        Exit-Script 1
    }

    Write-Info "Installing: $($setupExe.Name) ($([math]::Round($setupExe.Length / 1MB, 1)) MB)"

    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $proc = Start-Process -FilePath $setupExe.FullName `
            -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART' `
            -Wait -PassThru
    } finally {
        $ErrorActionPreference = $prevPref
    }

    if ($proc.ExitCode -ne 0) {
        Write-Err "Installer failed (exit code $($proc.ExitCode))."
        Exit-Script 1
    }
    Write-Ok "CV2T installed successfully"

    # -- Step 5: Re-run Enable-Canary if canary-env exists ---------------------
    # After reinstall the bundled Enable-Canary.ps1 may have updated deps.
    # Re-running it mirrors what a user would do and ensures the installed
    # canary-env stays in sync -- no repo-specific workarounds.
    $installedCanaryEnv = 'C:\Program Files\CV2T\canary-env'
    $installedEnableCanary = 'C:\Program Files\CV2T\_internal\Enable-Canary.ps1'
    if ((Test-Path $installedCanaryEnv) -and (Test-Path $installedEnableCanary)) {
        Write-Step "Re-running Enable-Canary.ps1 to sync installed canary-env..."
        $prevPref = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & $installedEnableCanary -NoLaunch 2>&1 | ForEach-Object { Write-Host "  $_" }
        } finally {
            $ErrorActionPreference = $prevPref
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Enable-Canary.ps1 exited with code $LASTEXITCODE."
        } else {
            Write-Ok "Canary environment up to date"
        }
    } else {
        Write-Info "No installed canary-env found -- skipping Enable-Canary."
    }

    # -- Step 6: Launch --------------------------------------------------------
    Write-Step "Launching CV2T (installed build)..."

    $installedExe = 'C:\Program Files\CV2T\cv2t.exe'
    if (-not (Test-Path $installedExe)) {
        Write-Err "cv2t.exe not found at $installedExe"
        Exit-Script 1
    }

    # Stop any running instance so the new launch does not hit the
    # "Another instance is already running" single-instance guard.
    Get-Process -Name 'cv2t' -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2

    Start-Process $installedExe
    Write-Ok "CV2T launched from $installedExe"

    Write-Host ""
    Write-Host "  =========================================" -ForegroundColor Green
    Write-Host "  Release build test complete." -ForegroundColor Green
    Write-Host "  =========================================" -ForegroundColor Green
    Write-Host ""

    Exit-Script 0
}


# ==============================================================================
#  MODE: Source Test (no system changes)
# ==============================================================================

if ($Mode -eq 'Source') {

    $devTemp = Join-Path $RepoRoot 'dev-temp'

    # -- Set up dev-temp directory ---------------------------------------------
    Write-Step "Setting up dev-temp environment..."

    foreach ($sub in @('config', 'logs', 'temp')) {
        $dir = Join-Path $devTemp $sub
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    Write-Ok "dev-temp directories ready at $devTemp"

    # -- Model access via directory junction -----------------------------------
    $devModels = Join-Path $devTemp 'models'
    $installedModels = 'C:\Program Files\CV2T\models'
    $repoModels = Join-Path $RepoRoot 'models'

    if (Test-Path $devModels) {
        # Already exists (junction or directory from previous run)
        Write-Ok "dev-temp\models already exists"
    } elseif (Test-Path $installedModels) {
        # Create junction to installed models (no admin, no duplication)
        Write-Info "Creating junction: dev-temp\models -> $installedModels"
        cmd /c mklink /J "$devModels" "$installedModels" | Out-Null
        if (Test-Path $devModels) {
            Write-Ok "Junction created to installed models"
        } else {
            Write-Warn "Junction creation failed. Falling back to repo models."
            if (Test-Path $repoModels) {
                cmd /c mklink /J "$devModels" "$repoModels" | Out-Null
            }
        }
    } elseif (Test-Path $repoModels) {
        # No installed models -- link to repo models folder
        Write-Info "No installed models found. Creating junction to repo models."
        cmd /c mklink /J "$devModels" "$repoModels" | Out-Null
        Write-Ok "Junction created to repo models"
    } else {
        Write-Warn "No model files found. The app may prompt you to download models."
    }

    # -- Launch from source ----------------------------------------------------
    Write-Step "Launching CV2T from source..."
    Write-Info "CV2T_HOME = $devTemp"
    Write-Info "Config, logs, and temp files go to dev-temp/ (not Program Files)."
    Write-Host ""

    $env:CV2T_HOME = $devTemp

    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        uv run python -m cv2t 2>&1 | ForEach-Object { Write-Host "  $_" }
    } finally {
        $ErrorActionPreference = $prevPref
    }

    # Clean up env var so it doesn't leak to the rest of the session
    Remove-Item Env:\CV2T_HOME -ErrorAction SilentlyContinue

    Write-Host ""
    Write-Host "  =========================================" -ForegroundColor Green
    Write-Host "  Source test session ended." -ForegroundColor Green
    Write-Host "  =========================================" -ForegroundColor Green
    Write-Info "dev-temp/ persists between runs. Delete it manually to reset."
    Write-Host ""

    Exit-Script 0
}
