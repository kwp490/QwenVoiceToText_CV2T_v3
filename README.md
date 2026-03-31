# CV2T — Native Windows Voice-to-Text

**Real-time speech transcription on Windows using NVIDIA GPUs — no Docker, no WSL, no HTTP server.**

CV2T replaces the v1 Docker-based architecture with **native in-process inference**. Press a hotkey, speak, and your transcribed text is pasted into the active window.

## Features

- **Two engine options**: NVIDIA Canary Qwen 2.5B (NeMo) or Faster-Whisper (CTranslate2)
- **Global hotkeys**: Start/stop recording from any application
- **Auto-paste**: Transcribed text goes directly to your active window
- **GPU-accelerated**: Both engines leverage NVIDIA CUDA
- **No Docker, no WSL, no HTTP** — everything runs natively on Windows

## Requirements

- Windows 11 (64-bit)
- NVIDIA GPU (RTX 30-series or newer recommended, 6+ GB VRAM)
- NVIDIA Driver 525+ with CUDA support (RTX 50-series Blackwell requires 560+)

## Quick Start

### Recommended: GUI Installer

Download **CV2T-Setup-3.0.0.exe** from [Releases](https://github.com/kwp490/QwenVoiceToText_CV2T_v3/releases) and double-click it. The installer will:

1. Extract application files to `C:\Program Files\CV2T`
2. Let you choose a speech engine (Whisper or Canary)
3. Download the model (~1–3 GB)
4. Create desktop and Start Menu shortcuts

No Python, no command line required.

> **Silent / unattended install:**
> ```powershell
> CV2T-Setup-3.0.0.exe /VERYSILENT /ENGINE=whisper
> ```
> Accepted `/ENGINE=` values: `whisper`, `canary`, `both`

### Source Install (both engines)

```powershell
# 1. Install uv
irm https://astral.sh/uv/install.ps1 | iex

# 2. Clone and install
git clone https://github.com/kwp490/QwenVoiceToText_CV2T_v3.git
cd QwenVoiceToText_CV2T_v3
uv sync --extra all

# 3. Download model and launch
uv run cv2t download-model --engine canary
uv run cv2t
```

### PowerShell Binary Install (alternative)

For sysadmins or automated deployments, a PowerShell installer is also available:

```powershell
# Run as Administrator
Set-ExecutionPolicy Bypass -Scope Process -Force
.\installer\Install-CV2T-Bin.ps1 -ZipPath ".\cv2t-v3.0.0-win64.zip"
```

## Settings

| Setting | Default | Description |
|---|---|---|
| `engine` | `whisper` | Speech engine: `canary` or `whisper` |
| `model_path` | `C:\Program Files\CV2T\models` | Directory for model weights |
| `device` | `cuda` | Inference device: `cuda` or `cpu` |
| `language` | `en` | Language code |
| `inference_timeout` | `30` | Max seconds per transcription |
| `sample_rate` | `16000` | Recording sample rate (Hz) — audio is always resampled to 16 kHz for engines |
| `silence_threshold` | `0.0015` | RMS threshold for silence detection |
| `auto_copy` | `true` | Auto-copy transcription to clipboard |
| `auto_paste` | `true` | Auto-paste via Ctrl+V after transcription |

Settings are stored at `C:\Program Files\CV2T\config\settings.json`.

## Hotkeys

| Hotkey | Action |
|---|---|
| `Ctrl+Alt+P` | Start recording |
| `Ctrl+Alt+L` | Stop recording & transcribe |
| `Ctrl+Alt+Q` | Quit application |

Hotkeys are configurable in Settings.

## Architecture

```
┌──────────────────────┐
│      CV2T GUI        │
│  (PySide6 / Qt)      │
├──────────────────────┤
│   Engine Abstraction │
│   ┌───────┐ ┌──────┐│
│   │Canary │ │Whisper││
│   │(NeMo) │ │(CT2) ││
│   └───┬───┘ └──┬───┘│
│       │        │     │
│       ▼        ▼     │
│     NVIDIA GPU       │
│     (CUDA)           │
└──────────────────────┘
```

## Model Comparison

| | Canary (NeMo) | Whisper (CTranslate2) |
|---|---|---|
| **Model** | nvidia/canary-qwen-2.5b | large-v3-turbo (via faster-whisper) |
| **VRAM** | ~5 GB | ~3 GB |
| **Accuracy** | Excellent | Very good |
| **Speed** | Fast | Very fast |
| **torch required** | Yes | No |
| **Distribution** | Source install only | Source or .exe |
| **Max input** | 40s (auto-chunked) | Unlimited (VAD) |

## GPU Dependency Matrix

| Component | Whisper (binary) | Canary (source) |
|---|---|---|
| CUDA Toolkit | 12.x | 12.x |
| cuDNN | Not required | 9.x (via torch) |
| CTranslate2 | 4.5.x | — |
| torch | NOT required | 2.1+ (via NeMo) |
| NeMo | — | 2.0+ |

## Windows Defender

The `keyboard` library uses low-level keyboard hooks (`SetWindowsHookEx`) which antivirus software may flag. Add these exclusions:

```powershell
# Run as Administrator
Add-MpPreference -ExclusionPath "C:\Program Files\CV2T"
Add-MpPreference -ExclusionProcess "C:\Program Files\CV2T\cv2t.exe"
```

## Building the Installer

The binary build is **Whisper-only** (excludes torch/NeMo):

```bash
# 1. Build the binary
uv sync --extra whisper --extra dev
uv run pyinstaller cv2t.spec
# Output: dist/cv2t/cv2t.exe

# 2. Build the GUI installer (requires Inno Setup 6.x)
iscc installer\cv2t-setup.iss
# Output: installer/Output/CV2T-Setup-3.0.0.exe
```

Or use the combined build script:

```powershell
.\installer\Build-Installer.ps1
```

## CLI

```bash
cv2t                                                    # Launch GUI
cv2t download-model --engine whisper --target-dir DIR   # Download model
cv2t download-model --engine canary --target-dir DIR    # Download Canary model (no auth required)
cv2t --version                                          # Print version
```

## License

MIT — see [LICENSE](LICENSE).
