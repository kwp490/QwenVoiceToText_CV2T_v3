# Changelog

All notable changes to CV2T are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Professional Mode**: Optional AI-powered text cleanup via OpenAI API
  - Fix tone (rewrite emotional/unprofessional language)
  - Fix grammar
  - Fix punctuation and capitalization
  - Each option independently toggleable in Settings
- Professional Mode settings group in Settings dialog with model selector, API key field, and sub-options
- API key validation button in Settings — verifies key works before saving
- "Remember API key" option — stores key securely via Windows Credential Manager (DPAPI)
- Transcription history shows both original and cleaned text when Professional Mode is active
- Graceful degradation — if the OpenAI API fails, raw transcription is used as fallback

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
