# CV2T — Native Windows Voice-to-Text

**Real-time speech transcription on Windows using NVIDIA GPUs.**

Press a hotkey, speak, and your transcribed text is pasted into the active window. GPU-accelerated, runs natively — no setup complexity.

## Download & Install

> **[Download CV2T-Setup-3.0.0.exe](https://github.com/kwp490/QwenVoiceToText_CV2T_v3/releases/latest)**
>
> Double-click the installer and follow the prompts. No Python, no command line required.

The installer will:

1. Extract application files to `C:\Program Files\CV2T`
2. Let you choose a speech engine (Whisper or Canary)
3. Download the model (~1–3 GB)
4. Create desktop and Start Menu shortcuts

**Requirements:** Windows 10/11 (64-bit), NVIDIA GPU (RTX 30-series or newer, 6+ GB VRAM), NVIDIA Driver 525+ (RTX 50-series Blackwell requires 560+).

> **Silent / unattended install:**
> ```powershell
> CV2T-Setup-3.0.0.exe /VERYSILENT /ENGINE=whisper
> ```
> Accepted `/ENGINE=` values: `whisper`, `canary`, `both`

## Features

- **Two engine options**: NVIDIA Canary Qwen 2.5B (NeMo) or Faster-Whisper (CTranslate2)
- **Professional Mode**: AI-powered text cleanup via OpenAI API with a preset system — 5 built-in presets, custom presets, domain vocabulary preservation, and per-preset model selection
- **Global hotkeys**: Start/stop recording from any application (configurable bindings)
- **Auto-paste**: Transcribed text goes directly to your active window
- **GPU-accelerated**: Both engines leverage NVIDIA CUDA with automatic Blackwell (RTX 50-series) workarounds
- **Microphone selection**: Choose a specific input device or use the system default
- **Sleep/wake recovery**: Hotkeys automatically re-register after Windows resume from sleep
- **Single-instance guard**: Prevents multiple CV2T processes from running simultaneously
- **Real-time resource monitoring**: RAM, VRAM, and GPU temperature displayed in the diagnostics panel
- **Audio feedback**: Beep tones on recording start/stop
- **Canary Bridge Engine**: Frozen binary builds can use Canary via a subprocess bridge (`canary-env`)
- **Runs natively on Windows** — single installer, no dependencies

## Source Install (both engines)

For developers or users who want the Canary engine (requires Python):

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

## Settings

| Setting              | Default                            | Description                                              |
|----------------------|------------------------------------|----------------------------------------------------------|
| `engine`             | `whisper`                          | Speech engine: `canary` or `whisper`                     |
| `model_path`         | `C:\Program Files\CV2T\models`     | Directory for model weights                              |
| `device`             | `cuda`                             | Inference device: `cuda` or `cpu`                        |
| `language`           | `en`                               | Language code                                            |
| `inference_timeout`  | `30`                               | Max seconds per transcription                            |
| `force_cuda_sync`    | `auto`                             | CUDA sync mode: `auto`, `on`, `off` — Blackwell GPU workaround |
| `auto_copy`          | `true`                             | Auto-copy transcription to clipboard                     |
| `auto_paste`         | `true`                             | Auto-paste via Ctrl+V after transcription                |
| `hotkeys_enabled`    | `true`                             | Master toggle for global hotkeys                         |
| `hotkey_start`       | `ctrl+alt+p`                       | Start-recording hotkey                                   |
| `hotkey_stop`        | `ctrl+alt+l`                       | Stop/transcribe hotkey                                   |
| `hotkey_quit`        | `ctrl+alt+q`                       | Quit application hotkey                                  |
| `clear_logs_on_exit` | `true`                             | Clear log files when the application exits               |
| `mic_device_index`   | `-1`                               | Microphone device index (`-1` = system default)          |
| `sample_rate`        | `16000`                            | Recording sample rate (Hz) — resampled to 16 kHz         |
| `silence_threshold`  | `0.0015`                           | RMS threshold for silence detection                      |
| `silence_margin_ms`  | `500`                              | Silence margin (ms) added around voiced regions          |
| `professional_mode`  | `false`                            | Enable AI text cleanup (requires OpenAI API key)         |
| `pro_active_preset`  | `General Professional`             | Active Professional Mode preset name                     |
| `store_api_key`      | `false`                            | Persist API key in Windows Credential Manager            |

Settings are stored at `C:\Program Files\CV2T\config\settings.json`.

> **Note:** The OpenAI API key is **never** stored in `settings.json`. It is held in memory only, unless you enable "Remember API key", which saves it securely via Windows Credential Manager (DPAPI).

## Hotkeys

| Hotkey (default)     | Action                               |
| -------------------- | ------------------------------------ |
| `Ctrl+Alt+P`         | Start recording                      |
| `Ctrl+Alt+L`         | Stop recording & transcribe          |
| `Ctrl+Alt+Q`         | Quit application                     |

All hotkey bindings are configurable in Settings. Hotkeys can also be disabled entirely via the `hotkeys_enabled` toggle. After Windows resumes from sleep, hotkeys are automatically re-registered.

## Professional Mode

Optional AI-powered post-processing that cleans up your dictated text before it reaches the clipboard. Configure it via the **Professional Mode Settings** button in the main window.

**What it does:**
- **Fix tone** — rewrites emotional, aggressive, or unprofessional language while preserving meaning
- **Fix grammar** — corrects grammar errors
- **Fix punctuation** — adds proper punctuation and capitalization
- **Custom instructions** — free-text system prompt per preset for fine-tuning AI behavior
- **Vocabulary preservation** — domain-specific terms (comma/newline-separated) are preserved verbatim during cleanup

Each option is configured per preset — you can have different cleanup rules for different contexts. When enabled, the transcription history shows both the original and cleaned text.

### Presets

Professional Mode uses a **preset system**. Five built-in presets are included:

| Preset | Description |
|---|---|
| **General Professional** | Neutral business tone, clear and concise |
| **Technical / Engineering** | Preserves jargon, acronyms, and technical terminology |
| **Casual / Friendly** | Warm, approachable, conversational tone |
| **Email / Correspondence** | Professional email with greeting/sign-off, short paragraphs |
| **Simplified (8th Grade)** | Short sentences, common words, simple structures |

You can also create, duplicate, and delete custom presets. Each preset has its own toggle settings, custom system prompt, vocabulary list, and optional model override.

**Requirements:** An OpenAI API key. Enter it in Professional Mode Settings — the key is held in memory only by default and is **never** written to `settings.json` or any log file. Optionally check "Remember API key" to store it securely via Windows Credential Manager.

**Example:**

| Input (dictated) | Output (General Professional preset) |
|---|---|
| *"I am having a horrible day at work and I am angry and frustrated at you"* | *"I am having a challenging day at work and would like to discuss some concerns with you."* |

## Architecture

```
┌───────────────────────────────────────┐
│          CV2T GUI  (PySide6 / Qt)     │
│ ┌────────────┐  ┌──────────────────┐  │
│ │ Hotkey Mgr │  │ Resource Monitor │  │
│ │ (sleep/    │  │ (RAM + VRAM +    │  │
│ │  wake safe)│  │  GPU temp)       │  │
│ └────────────┘  └──────────────────┘  │
├───────────────────────────────────────┤
│   Engine Abstraction                  │
│   ┌──────────┐ ┌──────────────────┐   │
│   │ Canary   │ │ Whisper (CT2)    │   │
│   │ (NeMo or │ │                  │   │
│   │  Bridge) │ │                  │   │
│   └────┬─────┘ └──────┬──────────┘   │
│        │               │              │
│        ▼               ▼              │
│      NVIDIA GPU (CUDA)                │
├───────────────────────────────────────┤
│   Professional Mode (optional)        │
│   ┌─────────────────────────────────┐ │
│   │ ProPreset → TextProcessor →     │ │
│   │ OpenAI API                      │ │
│   │ (5 built-in + custom presets,   │ │
│   │  vocabulary, custom prompts)    │ │
│   └─────────────────────────────────┘ │
└───────────────────────────────────────┘
```

## Model Comparison

|                      | Canary (NeMo)                       | Whisper (CTranslate2)                      |
| -------------------- | ----------------------------------- | ------------------------------------------ |
| **Model**            | nvidia/canary-qwen-2.5b             | large-v3-turbo (via faster-whisper)        |
| **VRAM**             | ~5 GB                               | ~3 GB                                      |
| **Accuracy**         | Excellent                           | Very good                                  |
| **Speed**            | Fast                                | Very fast                                  |
| **torch required**   | Yes                                 | No                                         |
| **Distribution**     | Source install only                  | Source or .exe                              |
| **Max input**        | 40 s (auto-chunked)                 | Unlimited (VAD)                            |

## GPU Dependency Matrix

| Component            | Whisper (binary)                    | Canary (source)                            |
| -------------------- | ----------------------------------- | ------------------------------------------ |
| CUDA Toolkit         | 12.x                                | 12.x                                       |
| cuDNN                | Not required                        | 9.x (via torch)                            |
| CTranslate2          | 4.5.x                               | —                                          |
| ONNX Runtime         | 1.17+ (Silero VAD)                  | —                                          |
| torch                | NOT required                        | 2.1+ (via NeMo)                            |
| NeMo                 | —                                   | 2.0+                                       |

## Antivirus & Anti-Malware Notes

CV2T uses low-level keyboard hooks (`SetWindowsHookEx`) for global hotkeys. Some antivirus or anti-malware tools may flag this as suspicious — it is a **false positive**. The application only listens for the specific hotkey combinations you configure; it does not capture or log general keystrokes.

### Recommended exclusions

Add these exclusions in your antivirus / anti-malware software to prevent interference:

| Path / Process                             | Why                                                    |
| ------------------------------------------ | ------------------------------------------------------ |
| `C:\Program Files\CV2T\`                   | Install directory — contains the app and CUDA DLLs     |
| `C:\Program Files\CV2T\cv2t.exe`           | Main executable — flagged due to keyboard hooks         |
| `uv.exe` (if using source install)         | Python package manager — sometimes flagged by Malwarebytes and other tools |

**Windows Defender (PowerShell, run as Administrator):**

```powershell
Add-MpPreference -ExclusionPath "C:\Program Files\CV2T"
Add-MpPreference -ExclusionProcess "C:\Program Files\CV2T\cv2t.exe"
```

> **Note:** The GUI installer adds these Defender exclusions automatically during setup.

**Other antivirus / anti-malware software:** Look for "Exclusions", "Allow list", or "Authorized applications" in your security software settings and add the paths above. If `uv.exe` is quarantined during a source install, restore it and add it to your allow list — [uv](https://github.com/astral-sh/uv) is a widely used open-source Python package manager by Astral.

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

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.
