# Contributing to CV2T

## Dev Setup

```bash
# Clone and install all dependencies (including dev tools)
git clone https://github.com/kwp490/QwenVoiceToText_CV2T_v3.git
cd QwenVoiceToText_CV2T_v3
uv sync --extra all --extra dev
```

## Running Tests

```bash
uv run pytest
```

## Compile Check

```bash
uv run python -m compileall cv2t
```

## Verify Engine Availability

```bash
uv run python -c "from cv2t.engine import ENGINES; print(list(ENGINES.keys()))"
```

## Code Style

- Use type hints where practical
- Follow existing patterns in the codebase
- Keep imports sorted (stdlib → third-party → local)

## Architecture Notes

- **Thread safety**: Clipboard writes (`set_clipboard_text`) must only happen on the main Qt thread. Worker threads emit signals; connected slots run on the main thread.
- **Audio format**: All engine calls receive 1D float32 mono numpy arrays. Audio is resampled to 16 kHz before engine input, regardless of recording sample rate.
- **Canary chunking**: Audio longer than 40 seconds is automatically chunked (30s windows with 2s overlap) and stitched.
- **Canary Bridge**: In frozen (PyInstaller) builds, the Canary engine runs as a subprocess via `CanaryBridgeEngine`. The bridge communicates with `canary_worker.py` over JSON lines on stdin/stdout, using a separate `canary-env` Python environment with torch/NeMo installed.
- **Canary inference thread**: The native `CanaryEngine` pins all PyTorch/CUDA operations to a single dedicated thread to prevent cross-thread CUDA context corruption.
- **Blackwell workarounds**: GPUs with compute capability ≥ sm_120 (Blackwell / RTX 50-series) automatically get `CUDA_LAUNCH_BLOCKING=1` and `TORCHDYNAMO_DISABLE=1` to prevent NeMo SALM hangs. Controlled by the `force_cuda_sync` setting (`auto`/`on`/`off`).
- **GPU cleanup**: `unload()` methods must explicitly `del` the model, call `gc.collect()`, and `torch.cuda.empty_cache()` (if torch is available).
- **Professional Mode**: Text cleanup runs on a `Worker` thread via the OpenAI API (no GPU conflict, no need to pause the resource monitor). The API key is held in memory on `MainWindow._api_key` — it must **never** be logged, printed, or serialized to `settings.json`. Use `_sanitize_error()` from `text_processor.py` when handling API exceptions.
- **Preset system**: Professional Mode uses `ProPreset` dataclass instances. Five built-in presets are always available; user presets are stored as JSON files in `config/presets/`. Built-in presets cannot be deleted.
- **Sleep/wake recovery**: `HotkeyManager.re_register()` is called on `WM_POWERBROADCAST` / `PBT_APMRESUMEAUTOMATIC` to restore keyboard hooks invalidated during sleep.
- **Single-instance guard**: A Windows named mutex (`Global\CV2TMutex`) prevents multiple processes. Released via `release_single_instance_mutex()` before restart.

## Building the Binary

The `.exe` build is Whisper-only:

```bash
uv sync --extra whisper --extra dev
uv run pyinstaller cv2t.spec
```

## Building the Installer

After building the binary, compile the Inno Setup installer:

```bash
# Requires Inno Setup 6.x — https://jrsoftware.org/isdl.php
iscc installer\cv2t-setup.iss
# Output: installer/Output/CV2T-Setup-3.0.0.exe
```

Or run the combined build script:

```powershell
.\installer\Build-Installer.ps1
```

The Inno Setup script ([installer/cv2t-setup.iss](installer/cv2t-setup.iss)) bundles the
PyInstaller output from `dist/cv2t/`, adds a custom engine-selection wizard page,
downloads models post-install via `cv2t.exe download-model`, and creates shortcuts.

## Filing Issues

Please include the output of "Copy Diagnostics" from the GUI when reporting bugs.
