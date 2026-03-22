# CV2T — Native Windows Voice-to-Text

**Real-time speech transcription on Windows using NVIDIA GPUs — no Docker, no WSL, no HTTP server.**

CV2T replaces the v1 Docker-based architecture with **native in-process inference**. Press a hotkey, speak, and your transcribed text is pasted into the active window.

## Features

- **Two engine options**: NVIDIA Canary Qwen 2.5B (NeMo/torch) or Faster-Whisper (CTranslate2)
- **Global hotkeys**: Start/stop recording from any application
- **Auto-paste**: Transcribed text goes directly to your active window
- **GPU-accelerated**: Both engines leverage NVIDIA CUDA
- **No Docker, no WSL, no HTTP** — everything runs natively on Windows

## Requirements

- Windows 11 (64-bit)
- NVIDIA GPU (RTX 30-series or newer recommended, 6+ GB VRAM)
- NVIDIA Driver 525+ with CUDA support

## Quick Start

### Source Install (both engines)

```powershell
# 1. Install uv
irm https://astral.sh/uv/install.ps1 | iex

# 2. Clone and install
git clone https://github.com/kwp490/cv2t.git
cd cv2t
uv sync --extra all

# 3. Download model and launch
uv run cv2t download-model --engine whisper --target-dir "$env:LOCALAPPDATA\CV2T\models"
uv run cv2t
```

### Binary Install (Whisper-only .exe)

Download the latest release zip from [Releases](https://github.com/kwp490/cv2t/releases), then run the installer:

```powershell
# Run as Administrator
Set-ExecutionPolicy Bypass -Scope Process -Force
.\installer\Install-CV2T-Bin.ps1 -ZipPath ".\cv2t-v2.0.0-win64.zip"
```

## Settings

| Setting | Default | Description |
|---|---|---|
| `engine` | `whisper` | Speech engine: `canary` or `whisper` |
| `model_path` | `%LOCALAPPDATA%\CV2T\models` | Directory for model weights |
| `device` | `cuda` | Inference device: `cuda` or `cpu` |
| `language` | `en` | Language code |
| `inference_timeout` | `30` | Max seconds per transcription |
| `sample_rate` | `16000` | Recording sample rate (Hz) — audio is always resampled to 16 kHz for engines |
| `silence_threshold` | `0.0015` | RMS threshold for silence detection |
| `auto_copy` | `true` | Auto-copy transcription to clipboard |
| `auto_paste` | `true` | Auto-paste via Ctrl+V after transcription |

Settings are stored at `%APPDATA%\CV2T\settings.json`.

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
| **Model** | nvidia/canary-qwen-2.5b | large-v3-turbo |
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
| cuDNN | 9.x | 9.x (via torch) |
| CTranslate2 | 4.5.x | — |
| torch | NOT required | 2.1+ (via NeMo) |
| NeMo | — | 2.0+ |

## Windows Defender

The `keyboard` library uses low-level keyboard hooks (`SetWindowsHookEx`) which antivirus software may flag. Add these exclusions:

```powershell
# Run as Administrator
Add-MpPreference -ExclusionPath "C:\Program Files\CV2T"
Add-MpPreference -ExclusionProcess "C:\Program Files\CV2T\cv2t.exe"
Add-MpPreference -ExclusionPath "$env:APPDATA\CV2T"
```

## Building the .exe

The binary build is **Whisper-only** (excludes torch/NeMo):

```bash
uv sync --extra whisper --extra dev
uv run pyinstaller cv2t.spec
# Output: dist/cv2t/cv2t.exe
```

## CLI

```bash
cv2t                                                    # Launch GUI
cv2t download-model --engine whisper --target-dir DIR   # Download model
cv2t download-model --engine canary --target-dir DIR    # Download Canary model
cv2t --version                                          # Print version
```

## License

MIT — see [LICENSE](LICENSE).
