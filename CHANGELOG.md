# Changelog

All notable changes to CV2T are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Professional Mode**: Optional AI-powered text cleanup via OpenAI API
  - Fix tone (rewrite emotional/unprofessional language)
  - Fix grammar
  - Fix punctuation and capitalization
  - Each option configured per preset
- **Preset system** for Professional Mode — 5 built-in presets (General Professional, Technical / Engineering, Casual / Friendly, Email / Correspondence, Simplified 8th Grade) plus user-created custom presets
- **Custom system prompts** per preset for fine-tuning AI cleanup behavior
- **Domain vocabulary preservation** — comma/newline-separated terms protected during cleanup
- **Per-preset model override** — use different OpenAI models (e.g. `gpt-5.4-mini`, `gpt-5.4-nano`) per preset
- **Professional Mode Settings dialog** — dedicated scrollable dialog for managing presets, API key, custom instructions, and vocabulary
- API key validation button — verifies key works before saving
- "Remember API key" option — stores key securely via Windows Credential Manager (DPAPI)
- Transcription history shows both original and cleaned text when Professional Mode is active
- Graceful degradation — if the OpenAI API fails, raw transcription is used as fallback
- **Canary Bridge Engine** — subprocess-based Canary for frozen/PyInstaller binary builds via separate `canary-env` Python environment
- **Install Canary Engine** button in Settings — launches `Enable-Canary.ps1` to install torch/NeMo into `canary-env`
- **Validate Canary Setup** button in Settings — checks canary-env Python, dependencies, CUDA, and model files
- **System resource monitor** — real-time RAM + GPU VRAM + temperature polling via Win32 API and NVML
- **Microphone device selection** in Settings (system default or specific device)
- **Silence margin setting** (`silence_margin_ms`) for voice activity detection tuning
- **Configurable hotkey bindings** — start, stop, quit hotkeys editable in Settings and persisted
- **Hotkeys enabled master toggle** — disable all global hotkeys without changing bindings
- **System sleep/wake recovery** — automatic hotkey re-registration after Windows resume from sleep
- **Single-instance mutex guard** — Windows mutex prevents multiple CV2T processes
- **Blackwell GPU (sm_120+) detection** — auto-enables `CUDA_LAUNCH_BLOCKING` and `TORCHDYNAMO_DISABLE` for stability
- **Force CUDA sync setting** (`auto`/`on`/`off`) for manual Blackwell workaround override
- **Collapsible Advanced Diagnostics panel** — engine status, logs, and GPU metrics hidden by default
- **Audio beep feedback** on recording start/stop
- **Engine selection dialog** at startup when multiple engines + models are installed
- **Clear logs on exit** toggle setting (`clear_logs_on_exit`)

### Changed
- Professional Mode settings moved from main Settings dialog to a dedicated Professional Mode Settings dialog
- Individual `pro_fix_tone`, `pro_fix_grammar`, `pro_fix_punctuation`, and `pro_model` settings replaced by per-preset fields in `ProPreset`

### Fixed
- Installer now bundles `onnxruntime` — fixes CUDA→CPU fallback caused by missing Silero VAD dependency

### Security
- API keys are **never** stored in `settings.json` or written to log files
- API key field is password-masked with a toggle to reveal
- All error messages are sanitized to redact API key content

## [3.0.0] — 2026-03-31

### Added
- Native Windows voice-to-text with GPU acceleration
- **Whisper engine** (CTranslate2) — fast, lightweight (~3 GB VRAM), bundled in binary installer
- **Canary engine** (NeMo/torch) — higher accuracy (~5 GB VRAM), available via source install
- GUI installer (Inno Setup) with engine selection and model download
- Silent / unattended install: `CV2T-Setup-3.0.0.exe /VERYSILENT /ENGINE=whisper`
- Global hotkeys for start/stop recording and quit (configurable)
- Auto-paste transcription to the active window via Ctrl+V
- Auto-copy transcription to clipboard
- Real-time GPU resource monitoring (VRAM, utilization)
- Automatic 40-second audio chunking for Canary engine
- Data migration from earlier install locations
- Windows Defender exclusion setup in installer
- Source install path with `uv sync` for developers and Canary users
- `download-model` CLI command for offline model setup

### Fixed
- Settings: changing device (CUDA → CPU) now correctly prompts to reload the model
- Settings: device selector is disabled for Canary engine (requires CUDA)

### Changed
- "Server" panel replaced with "Model Engine" panel in the GUI

### Removed
- NeMo ONNX runtime (replaced with native NeMo/torch)
