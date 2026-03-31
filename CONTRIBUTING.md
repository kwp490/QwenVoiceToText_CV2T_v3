# Contributing to CV2T

## Dev Setup

```bash
# Clone and install all dependencies (including dev tools)
git clone https://github.com/kwp490/cv2t.git
cd cv2t
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
- **GPU cleanup**: `unload()` methods must explicitly `del` the model, call `gc.collect()`, and `torch.cuda.empty_cache()` (if torch is available).

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
# Output: installer/Output/CV2T-Setup-2.0.0.exe
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
