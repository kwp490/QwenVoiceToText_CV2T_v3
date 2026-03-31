# Changelog

All notable changes to CV2T are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [3.0.0] — 2026-03-31

### Added
- Native Windows voice-to-text — no Docker, no WSL, no HTTP server
- **Whisper engine** (CTranslate2) — fast, lightweight (~3 GB VRAM), bundled in binary installer
- **Canary engine** (NeMo/torch) — higher accuracy (~5 GB VRAM), available via source install
- GUI installer (Inno Setup) with engine selection and model download
- Silent / unattended install: `CV2T-Setup-3.0.0.exe /VERYSILENT /ENGINE=whisper`
- Global hotkeys for start/stop recording and quit (configurable)
- Auto-paste transcription to the active window via Ctrl+V
- Auto-copy transcription to clipboard
- Real-time GPU resource monitoring (VRAM, utilization)
- Automatic 40-second audio chunking for Canary engine
- Data migration from v1/v2 install locations
- Windows Defender exclusion setup in installer
- Source install path with `uv sync` for developers and Canary users
- `download-model` CLI command for offline model setup

### Changed
- Complete rewrite from Docker-based v1 architecture to native in-process inference
- "Server" panel replaced with "Model Engine" panel in the GUI

### Removed
- Docker and WSL2 dependencies
- HTTP server for inference
- NeMo ONNX runtime (replaced with native NeMo/torch)
