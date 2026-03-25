## Plan: CV2T v2.0 — Native Windows Voice-to-Text (No Docker)

**TL;DR**: Build a brand-new project `kwp490/cv2t` that replaces v1's WSL2 → Docker → NeMo HTTP stack with **native in-process inference** on Windows. Default model: NVIDIA Canary Qwen 2.5B (via NeMo/torch). Alternative: Faster-Whisper (CTranslate2, no torch). Uses `uv` for deps. No Docker, no WSL, no HTTP server. UI stays similar, with "Server" panel becoming "Model Engine" panel. Distributed as: **Whisper-only standalone `.exe`** (no torch) + **source install for Canary** (requires torch/NeMo). A single all-engines `.exe` is deferred until a torch-free Canary runtime is proven.

> **CRITICAL**: This is a BRAND NEW project. Do NOT modify any files in the old `canary-voice-to-text` / `QwenVoiceToText_Docker` repository. All files are created from scratch in a new directory and pushed to a new GitHub repo.

---

### Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Canary inference runtime** | NeMo/torch (default, proven) → ONNX Runtime (experimental branch) | The only officially documented runtime for `nvidia/canary-qwen-2.5b` is NeMo via `SALM.from_pretrained()`. The community ONNX port is unvalidated (no model card, low downloads, config identifies `Qwen3ForCausalLM` not an ASR contract). **NeMo+torch is the default Canary implementation.** ONNX may be explored as a lighter-weight experiment once parity with NeMo is proven on real audio. |
| **Whisper inference** | `faster-whisper` (CTranslate2) | Proven, no torch needed, native Windows CUDA support |
| **torch dependency** | Required for Canary (NeMo); not needed for Whisper-only | NeMo pulls in torch. For Whisper-only installs/builds, torch is not required. The Whisper-only `.exe` excludes torch. Canary is source-install only until a torch-free runtime is proven. |
| **Canary audio chunking** | Mandatory 40-second chunking with transcript stitching | NVIDIA's model card states maximum training audio duration was 40 seconds; accuracy degrades beyond that. The official Space chunks into 40s windows. A chunking layer is mandatory — never pass arbitrary-length recordings into a single inference call. |
| **Inference sample rate** | Always 16 kHz mono float32 internally | Both Canary (NeMo) and Whisper (CTranslate2) expect 16 kHz mono input. `sample_rate` in settings is a **recording-only** parameter. Audio is always resampled to 16 kHz before any engine call, regardless of the recording sample rate. |
| **Clipboard thread safety** | Clipboard writes on main Qt thread ONLY | The Win32 `OpenClipboard()` API requires the calling thread to own the clipboard. Worker threads must emit Qt signals; clipboard operations happen in the connected main-thread slot. This matches v1's proven pattern. |
| **Installation** | Binaries in `C:\Program Files\CV2T\`, models in `%LOCALAPPDATA%\CV2T\models` | Install directory requires admin. Models default to `%LOCALAPPDATA%\CV2T\models` (user-writable) to avoid permission friction for model downloads, updates, and engine switches. User config remains at `%APPDATA%\CV2T\`. Overridable via settings. |
| **GPU memory cleanup** | Explicit `del model` + `gc.collect()` + `torch.cuda.empty_cache()` in `unload()` | NeMo/torch, ONNX Runtime, and CTranslate2 can all leak GPU memory without explicit cleanup. |

---

### Steps

**Prompt 1 — Foundation (Steps 1–9)**

1. **Project scaffolding** — Create `cv2t/` directory with `pyproject.toml` (using `uv` + `hatchling`), `.gitignore`, MIT LICENSE, `.python-version` pinning 3.11. Structure: `cv2t/` package, `cv2t/engine/` subpackage, `tests/`, `installer/`. Initialize git. Create GitHub repo via `gh repo create kwp490/cv2t`.

   `pyproject.toml` dependency groups:
   ```toml
   [project]
   dependencies = [
       "PySide6>=6.6,<7",
       "sounddevice>=0.4",
       "soundfile>=0.12",
       "numpy>=1.24",
       "keyboard>=0.13",
       "nvidia-ml-py>=12.0; sys_platform == 'win32'",
   ]

   [project.optional-dependencies]
   canary = [
       # NeMo path (default, proven):
       "nemo_toolkit[asr]>=2.0",
       "torch>=2.1",
       "accelerate",
   ]
   canary-onnx = [
       # Experimental ONNX path (no torch) — only viable once parity is proven:
       "onnxruntime-gpu==1.19.*",
   ]
   whisper = [
       "faster-whisper>=1.0",
   ]
   all = ["cv2t[canary,whisper]"]
   dev = [
       "pyinstaller>=6.0",
       "pytest>=7.0",
   ]
   ```

2. **Engine abstraction** — Create `cv2t/engine/base.py` defining a `SpeechEngine` Protocol with: `name`, `is_loaded`, `vram_estimate_gb` properties; `load(model_path, device)`, `transcribe(audio_numpy, sample_rate) → str`, `unload()` methods. Engines receive raw numpy float32 audio directly — no WAV, no HTTP.

   **Resampling contract**: The base class or a shared utility (`cv2t/engine/audio_utils.py`) must provide `ensure_16khz(audio: np.ndarray, source_sr: int) -> np.ndarray` which resamples to 16 kHz if `source_sr != 16000`. Every engine's `transcribe()` calls this first. Use `soundfile`/`numpy` linear interpolation or `scipy.signal.resample` (add `scipy` as a dep if needed, or use a lightweight resampler).

   **Chunking contract**: Engines that have a maximum input duration (e.g., Canary's 40s limit) must implement chunking internally. The base protocol's `transcribe()` accepts arbitrary-length audio; the engine is responsible for splitting, transcribing chunks, and stitching results. A shared `chunk_audio(audio, sr, max_seconds, overlap_seconds)` utility in `audio_utils.py` provides this.

   **Critical**: `unload()` must explicitly delete the model object/session and call `gc.collect()` to prevent GPU memory leaks:
   ```python
   def unload(self) -> None:
       if self._model is not None:
           del self._model
           self._model = None
       import gc; gc.collect()
       # If torch was used:
       import torch; torch.cuda.empty_cache()
   ```

3. **Canary engine — NeMo is the default implementation** — Create `cv2t/engine/canary.py`.

   The official runtime for `nvidia/canary-qwen-2.5b` is NeMo. Community ONNX exports are unvalidated and should only be explored after NeMo parity is established.

   **Default Path — NeMo (proven, matches v1):**
   1. Load the model using `nemo.collections.speechlm2.models.SALM.from_pretrained(model_path)`.
   2. Use the NeMo prompt/audio API — **NOT** Whisper-style special tokens. The correct prompting protocol is conversation-based:
      ```python
      conversation = [[{
          "role": "user",
          "content": f"Transcribe the following: {model.audio_locator_tag}",
          "audio": [audio_file_path],
      }]]
      with torch.inference_mode():
          response = model.generate(
              prompts=conversation,
              max_new_tokens=max_tokens,
              temperature=0.0,
              top_k=1,
          )[0]
      text = model.tokenizer.ids_to_text(response.cpu())
      text = text.replace("<|endoftext|>", "").strip()
      ```
   3. **NeMo requires file paths, not numpy arrays.** The engine's `transcribe()` must write audio to a temporary WAV file (use `tempfile.NamedTemporaryFile` or `/dev/shm/` equivalent), pass the path into the conversation dict, then clean up.
   4. Dynamic `max_new_tokens` based on audio duration: `max(64, int(duration_seconds * 20))` (~20 tokens/sec of speech).
   5. dtype selection: `torch.bfloat16` on Ampere+ GPUs, `torch.float16` otherwise.
   6. Warmup with dummy inference (4000-sample silence WAV) after loading for CUDA kernel compilation.

   **Mandatory 40-second chunking:**
   Canary was trained on a maximum of 40 seconds of audio. The official NVIDIA Space chunks into 40s windows. The engine must:
   1. Split audio longer than 40 seconds into chunks (e.g., 30s chunks with 2s overlap to avoid word splits at boundaries).
   2. Transcribe each chunk independently.
   3. Stitch transcripts together, deduplicating overlapping text at boundaries.
   Use the shared `chunk_audio()` utility from `cv2t/engine/audio_utils.py`.

   **Experimental Path — ONNX (deferred, not in v2.0 scope):**
   If explored later, an ONNX branch should:
   1. Validate the community port (`onnx-community/canary-qwen-2.5b-ONNX` or similar) produces equivalent output to NeMo on real speech audio.
   2. Document the ONNX loading method and any manual token handling required.
   3. Only graduate to default once parity is proven. Track in a GitHub issue.

   **Important**: Do NOT use Whisper-style control tokens (`<|startoftranscript|>`, `<|en|>`, `<|transcribe|>`, `<|notimestamps|>`) for Canary. These are a different model family's protocol.

4. **Whisper engine** — Create `cv2t/engine/whisper.py` using `faster-whisper` (CTranslate2 backend — no torch needed).

   **Critical**: Separate `model_id` (e.g., `"large-v3-turbo"`) from `models_dir` (the directory to store downloads). Always pass `download_root=models_dir` regardless of whether the subdirectory already exists — this ensures first-time downloads go to the configured location, not the default HuggingFace cache:
   ```python
   def load(self, model_id: str, models_dir: str, device: str = "cuda") -> None:
       from faster_whisper import WhisperModel
       # model_id: "large-v3-turbo", "large-v3", "turbo", etc.
       # models_dir: configured directory for model storage (e.g., %LOCALAPPDATA%\CV2T\models)
       # download_root: always set to models_dir so downloads never go to ~/.cache/
       compute = "float16" if device == "cuda" else "int8"
       self._model = WhisperModel(
           model_id,
           device=device,
           compute_type=compute,
           download_root=models_dir,
       )
   ```

   **Model name**: Use `"large-v3-turbo"` as the default model identifier (the library's built-in name). This is well-documented in faster-whisper's own API. Avoid hardcoding HuggingFace repo paths — the library resolves `large-v3-turbo` to `mobiuslabsgmbh/faster-whisper-large-v3-turbo` automatically.

   **GPU dependency matrix** (document in README and validate in CI):
   - faster-whisper/CTranslate2 GPU requires **CUDA 12.x** and **cuDNN 9.x**
   - Pin tested versions: `faster-whisper==1.1.*`, `ctranslate2==4.5.*` (update after testing)
   - Windows users must either have CUDA Toolkit + cuDNN installed system-wide, or the bundled DLLs from ctranslate2's pip package must be collected by PyInstaller

5. **Engine registry** — `cv2t/engine/__init__.py` with lazy registration: try-import each engine, register if deps available. `ENGINES` dict maps `"canary"` → `CanaryEngine`, `"whisper"` → `WhisperEngine`.

6. **GPU monitor** — Create `cv2t/gpu_monitor.py` using `nvidia-ml-py` (the official NVIDIA NVML Python binding — replaces the now-deprecated `pynvml` wrapper) for VRAM/temp/GPU name, plus `ctypes` `GlobalMemoryStatusEx` for system RAM (reuse v1's `docker_manager.py` lines 230-250 pattern). Returns `SystemMetrics` dataclass with `GpuMetrics` nested inside. Import as `import pynvml` (nvidia-ml-py provides the `pynvml` module name).

7. **Config** — Create `cv2t/config.py`. New `Settings` dataclass with fields: `engine` ("canary"/"whisper"), `model_path` (default: `%LOCALAPPDATA%\CV2T\models` — user-writable, avoids admin friction for downloads/updates), `device` ("cuda"/"cpu"), `language` ("en"), `inference_timeout` (30s), `auto_copy`, `auto_paste`, `hotkeys_enabled`, `hotkey_start/stop/quit`, `mic_device_index`, `sample_rate` (**recording-only** — audio is always resampled to 16 kHz before engine calls), `silence_threshold`, `silence_margin_ms`, `clear_logs_on_exit`. Config dir: `%APPDATA%\CV2T\settings.json` (user-writable, separate from install dir). Same `save()`/`load()` pattern as v1's `config.py`. **Removed from v1**: endpoint, container_name, image_name, server_port, health_timeout, health_poll_interval, shm_size, stop_server_on_exit, api_key.

8. **Port core modules** — Copy v1's `audio.py`, `clipboard.py`, `hotkeys.py`, `workers.py` into `cv2t/`.

   **`audio.py` changes:**
   - Add `get_raw_audio() → np.ndarray` that returns the recorded audio as a **1D float32 array of shape `(samples,)`**. This is mandatory — both CTranslate2 (faster-whisper) and ONNX Runtime expect 1D mono audio. Implementation:
     ```python
     def get_raw_audio(self) -> Optional[np.ndarray]:
         """Return recorded audio as 1D float32 mono array (samples,)."""
         audio = self.stop_recording()  # returns (samples, channels) or None
         if audio is None:
             return None
         # Downmix to mono if multi-channel, then flatten to 1D
         if audio.ndim == 2:
             if audio.shape[1] > 1:
                 audio = np.mean(audio, axis=1)  # multi-channel → mono
             else:
                 audio = audio[:, 0]  # single-channel (samples,1) → (samples,)
         return audio.astype(np.float32)
     ```
   - Note: v1 records with `channels=1` so audio is typically `(samples, 1)`, but this defensive downmix handles any edge case.

   **`clipboard.py` — copy verbatim from v1.** No changes needed. The Win32 ctypes approach avoids subprocess issues in PyInstaller `--noconsole` builds.

   **`hotkeys.py`, `workers.py` — copy verbatim from v1.**

9. **Foundation tests** — `tests/test_config.py` (adapt v1's `test_config.py` for new fields), `tests/test_engine_base.py` (verify both engines implement the protocol, mock model loading).

**Deliverable**: Compiles, tests pass, no GUI yet.

---

**Prompt 2 — GUI (Steps 10–14)**

10. **Main window** — Create `cv2t/main_window.py` adapted from v1's `main_window.py`. Key changes:
    - **"Model Engine" group** replaces "Server" group: shows engine name (`_lbl_engine`), model status (`_lbl_model_status`: Loading/Ready/Error), VRAM (`_lbl_vram`), RAM (`_lbl_ram`), GPU info (`_lbl_gpu_info`). Buttons: Reload Model, Validate, Copy Diagnostics.
    - **`ModelStatus` enum** replaces `ServerStatus`: `NOT_LOADED`, `LOADING`, `READY`, `VALIDATING`, `VALIDATED`, `ERROR` (6 states instead of 10).
    - **`DictationState`** enum: same as v1 minus `UPLOADING` (no HTTP upload).
    - **Startup flow**: Window appears → worker calls `engine.load()` → shows elapsed time ("Loading model… 45s") → on success set READY, on error set ERROR. No health polling loop needed.
    - **Transcription flow** (thread-safety critical):
      1. `_on_stop_and_transcribe()` creates a `Worker` wrapping `_process()`.
      2. `_process()` runs on a **worker thread**: calls `recorder.get_raw_audio()` → `recorder.trim_silence()` → `engine.transcribe(audio, sr)` → returns text string. **NO clipboard or keyboard operations in _process().**
      3. `worker.signals.result` connects to `_on_transcription_result()` which runs on the **main Qt thread** (via Qt signal marshaling).
      4. `_on_transcription_result()` calls `set_clipboard_text(text)` on the main thread (safe for Win32 `OpenClipboard()`), then dispatches `simulate_paste()` to a separate worker for the modifier-release spin-wait.
      This pattern matches v1 exactly and prevents `OpenClipboard access denied` crashes.
    - **Metrics timer**: 5-second interval, calls `get_system_metrics()` on worker, updates labels with same color-coding as v1.
    - **Diagnostics**: Collects engine name/status, model path, GPU info (via pynvml), Python/package versions, recent logs. No Docker logs.
    - **Validate**: Use a bundled short speech audio fixture (e.g., a 3-second WAV of a known phrase like "testing one two three") stored in `cv2t/assets/`. Run `engine.transcribe()` on it and assert a loose expected transcript match (e.g., check that the result contains "testing" or "one two three"). **Do NOT use a sine wave** — ASR models are built for speech, not tone detection, and a non-empty result on a sine wave is meaningless.
    - **Keep identical from v1**: Dictation group, history panel (+`_HistoryEntry` widget), log panel (+`QtLogHandler`), bottom buttons, `closeEvent` (calls `engine.unload()`), beep sounds.

11. **Settings dialog** — Create `cv2t/settings_dialog.py` adapted from v1's `settings_dialog.py`. Three groups: "Model Engine" (engine combo, model path + browse, device combo, language, inference timeout), "Audio" (identical to v1), "Dictation UX" (same as v1 minus stop-server-on-exit). Engine combo change updates model path placeholder. If engine/model_path changed on OK → prompt to reload model.

12. **Entry point** — Create `cv2t/__main__.py` adapted from v1's `__main__.py`. Mutex: `"Global\\CV2TMutex"`. App name: `"CV2T"`. Log file: `cv2t.log` in `%APPDATA%\CV2T\`. Remove API key generation. On startup: load settings → instantiate engine → create MainWindow → begin model loading. Keep v1's stdout/stderr safety (replace `None` with `io.StringIO()` for PyInstaller `--noconsole`).

    **CLI contract** (required for binary installer): Support `argparse`-based subcommands in addition to the default GUI launch:
    - `cv2t` (no args) — launch GUI (default)
    - `cv2t download-model --engine canary|whisper --target-dir <path>` — download model weights, then exit
    - `cv2t --version` — print version and exit

    This allows `Install-CV2T-Bin.ps1` to call `cv2t.exe download-model --engine whisper --target-dir "..."` without needing a separate `download_model.exe`.

13. **Package init** — `cv2t/__init__.py`: `__version__ = "2.0.0"`, docstring.

14. **Wire together** — Ensure main_window receives engine instance, settings, connects all signals/slots, timers work.

**Deliverable**: Fully functional GUI that records, transcribes in-process, and pastes.

---

**Prompt 3 — Polish & Distribution (Steps 15–22)**

15. **Complete tests** — `tests/test_audio.py` (silence trimming, WAV encoding, `get_raw_audio` 1D mono guarantee). Expand `test_config.py`. Ensure `uv run pytest` passes.

16. **PyInstaller spec** — Create `cv2t.spec`: entry `cv2t/__main__.py`. **This binary is Whisper-only.** Canary requires NeMo/torch which is too large and fragile to bundle — Canary users use the source install path.
    - **Build mode**: Use **`onedir`** (not `onefile`) for GPU builds. GPU builds bundle large CUDA/cuDNN DLLs; `onefile` extracts to a temp directory on every launch and has historical issues with binaries >2 GB. `onedir` produces `dist/cv2t/cv2t.exe` with DLLs alongside it.
    - **Hidden imports**: PySide6.QtWidgets, PySide6.QtCore, PySide6.QtGui, sounddevice, soundfile, _soundfile_data, numpy, keyboard, nvidia-ml-py (pynvml), faster_whisper, ctranslate2.
    - **Excludes**: `torch`, `torchaudio`, `torchvision`, `nemo`, `nemo_toolkit`, `tkinter`, `matplotlib`, `scipy`, `pandas`, `PIL`. Torch is excluded because this binary only ships the Whisper/CTranslate2 engine which does not need it.
    - **Binaries — sounddevice**: Use PyInstaller's `collect_dynamic_libs` utility instead of manual path construction for robustness across build environments:
      ```python
      from PyInstaller.utils.hooks import collect_dynamic_libs
      binaries = collect_dynamic_libs('sounddevice')
      ```
    - **Binaries — CUDA shared libraries**: `ctranslate2` (used by `faster-whisper`) bundles CUDA/cuDNN shared libraries (`.dll` files) inside its site-packages directory. PyInstaller does not reliably collect these automatically. Use `collect_dynamic_libs`:
      ```python
      from PyInstaller.utils.hooks import collect_dynamic_libs
      binaries = collect_dynamic_libs('sounddevice')
      # Collect CUDA/cuDNN DLLs from ctranslate2
      try:
          binaries += collect_dynamic_libs('ctranslate2')
      except Exception:
          pass
      ```
      **Fallback**: If the bundled DLLs still make the output too large, document that the target machine must have the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) installed system-wide, and exclude GPU DLLs from the bundle. Provide a separate CPU-only build without CUDA DLLs for users without NVIDIA GPUs.
    - `console=False`, output `dist/cv2t/cv2t.exe` (onedir).

17. **Installer — two distribution paths.** The TL;DR promises both a pip/source install and a standalone `.exe`. These are fundamentally different workflows and must not be conflated in a single script.

    **17a. Source installer** — Create `installer/Install-CV2T-Source.ps1` (developer/contributor path, supports both engines).
    - **Requires Administrator**: Installs to `C:\Program Files\CV2T\`. Prompts for elevation if not already admin.
    - Steps: check admin elevation (prompt if needed), check NVIDIA GPU/driver (reuse v1's WIN-02 pattern from `Install-Canary.ps1`), install `uv` via `irm https://astral.sh/uv/install.ps1 | iex`, install Python via `uv python install 3.11`, clone repo into `C:\Program Files\CV2T\`, `uv sync --extra all` (canary+whisper), run `cv2t download-model --engine canary --target-dir "$env:LOCALAPPDATA\CV2T\models"` (and/or `--engine whisper`), create desktop shortcut, launch.
    - After install, grant standard users read+execute permissions on the install directory so the app can run without admin. Models at `%LOCALAPPDATA%\CV2T\models` are inherently user-writable.
    - User settings remain at `%APPDATA%\CV2T\settings.json` (writable without admin).

    **17b. Binary installer** — Create `installer/Install-CV2T-Bin.ps1` (end-user path for the prebuilt `.exe`). **This is Whisper-only** — Canary requires the source install.
    - **Requires Administrator**: Installs to `C:\Program Files\CV2T\`.
    - **Does NOT require** `uv`, Python, or `git`. The user downloads a release `.zip` containing the `cv2t/` directory (from PyInstaller onedir) and this script.
    - Steps: check admin elevation, check NVIDIA GPU/driver, extract/copy `cv2t/` directory to `C:\Program Files\CV2T\`, run `cv2t.exe download-model --engine whisper --target-dir "$env:LOCALAPPDATA\CV2T\models"` (uses the CLI contract defined in Step 12), set permissions, create desktop shortcut, add Windows Defender exclusions.
    - Consider Inno Setup or NSIS as a follow-up for a traditional Windows installer wizard with uninstall support.

18. **Model downloader** — Create `download_model.py` at root (also callable via `cv2t download-model` CLI): supports both Canary and Whisper downloads. CLI: `--engine canary|whisper --target-dir "%LOCALAPPDATA%\CV2T\models"`.
    - **Canary target**: `onnx-community/canary-qwen-2.5b-ONNX` via `huggingface_hub.snapshot_download()`. This is an open model (CC-BY-4.0) — no authentication required.
    - **Whisper target**: `large-v3-turbo` via `faster_whisper.utils.download_model()` (resolves to `mobiuslabsgmbh/faster-whisper-large-v3-turbo`). Open model — no authentication required.
    - **Default target directory**: `%LOCALAPPDATA%\CV2T\models` (user-writable, no admin required).
    - **No authentication needed**: Both models are open and publicly accessible. No `--hf-token` argument, no `HF_TOKEN` environment variable, and no `huggingface-cli login` prompts.

19. **README.md** — What it does, requirements (Windows 11, NVIDIA 30-series+), quick start (3 commands), settings table, hotkeys table, architecture diagram (GUI ↔ Engine ↔ GPU), model comparison table (Canary vs Whisper), building .exe, contributing.

    **Include a GPU dependency matrix section** documenting exact tested versions:
    | Component | Whisper (binary) | Canary (source) |
    |---|---|---|
    | CUDA Toolkit | 12.x | 12.x |
    | cuDNN | 9.x | 9.x (via torch) |
    | CTranslate2 | 4.5.x (pinned) | — |
    | ONNX Runtime | — | — (deferred) |
    | torch | NOT required | 2.1+ (via NeMo) |
    | NeMo | — | 2.0+ |

    Pin and document these exact versions. Test the matrix in CI or release validation. Do not leave dependency ranges open-ended (e.g., `>=1.17`) without a known-good combination.

    **Include a Windows Defender section**: The `keyboard` library uses low-level keyboard hooks (`SetWindowsHookEx`) which antivirus software may flag. Document the recommended Defender exclusions:
    - Folder exclusion: `C:\Program Files\CV2T\`
    - Process exclusion: `C:\Program Files\CV2T\cv2t.exe`
    - Folder exclusion: `%APPDATA%\CV2T`

    Provide exact PowerShell commands:
    ```powershell
    # Run as Administrator
    Add-MpPreference -ExclusionPath "C:\Program Files\CV2T"
    Add-MpPreference -ExclusionProcess "C:\Program Files\CV2T\cv2t.exe"
    Add-MpPreference -ExclusionPath "$env:APPDATA\CV2T"
    ```

20. **CONTRIBUTING.md** — Dev setup with `uv sync --extra dev`, test commands, compile check.

21. **Create GitHub repo and push**:
    ```bash
    cd /mnt/c/Coding_Projects/QwenVoiceToText_CV2T_v2
    git init
    git add .
    git commit -m "CV2T v2.0 — native Windows voice-to-text, no Docker"
    gh repo create kwp490/cv2t --public --source=. --remote=origin --push
    ```

22. **Final verification** — Run the full verification checklist (see below).

**Deliverable**: Complete project, documented, pushed to `kwp490/cv2t`.

---

### Relevant files (v1 reference for implementing LLM)

The v1 project is at `canary-voice-to-text` / `QwenVoiceToText_Docker`. Read these files for reference patterns — do NOT modify them:

- `canary_gui/main_window.py` — Primary UI reference. Adapt Server group → Model Engine group, keep Dictation/History/Log groups. **Pay attention to the thread-safety pattern**: `_on_transcription_result()` does clipboard on main thread, `simulate_paste()` on worker.
- `canary_gui/settings_dialog.py` — Form layout pattern, mic dropdown, browse button. Replace Docker fields → engine fields.
- `canary_gui/audio.py` — Base recording logic. Add `get_raw_audio()` with 1D mono guarantee.
- `canary_gui/clipboard.py` — Copy verbatim. Win32 ctypes clipboard + simulate_paste.
- `canary_gui/hotkeys.py` — Copy verbatim.
- `canary_gui/workers.py` — Copy verbatim. `Worker` + `WorkerSignals` pattern.
- `canary_gui/config.py` — Rewrite fields, keep `save()`/`load()` pattern.
- `canary_gui/__main__.py` — Adapt mutex/app names, keep stdout safety, remove API key gen.
- `canary_qwen_docker/canary_server_docker.py` — Lines 260-440: reference model loading (dtype selection, warmup with dummy inference) and inference logic (audio_locator_tag conversation-style prompt format, generate params, tokenizer decode, dynamic max_tokens) for the NeMo Canary implementation. **This is the canonical reference for Canary prompting** — use conversation dicts with `model.audio_locator_tag`, NOT Whisper-style special tokens.
- `installer/Install-Canary.ps1` — Reference GPU check logic (WIN-02) for simplified installer.
- `tests/test_config.py` — Adapt for new Settings fields.

### Verification

1. `uv run pytest` — all tests pass
2. `uv run python -m compileall cv2t` — no syntax errors
3. `uv run python -c "from cv2t.engine import ENGINES; print(list(ENGINES.keys()))"` — shows available engines
4. `uv run cv2t` — window appears, model loads, status shows "Ready"
5. Record → Stop → transcribed text in history → auto-paste to Notepad
6. Validate button → uses bundled speech fixture → "Validation OK" with expected transcript match
7. Settings changes persist across restarts
8. VRAM/RAM metrics update every 5 seconds
9. Switch engine in settings → reload → new engine works
10. `uv run pyinstaller cv2t.spec` → `dist/cv2t/cv2t.exe` (onedir) runs without console
11. Second launch → "already running" dialog
12. Close app → `nvidia-smi` confirms VRAM freed
13. **Thread safety**: Confirm no clipboard operations happen on worker threads (search for `set_clipboard_text` — must only appear in main-thread slots)
14. **Audio shape**: Confirm `get_raw_audio()` returns shape `(N,)` not `(N, 1)` — add an `assert audio.ndim == 1` in tests
15. **Resampling**: Confirm audio is resampled to 16 kHz before engine calls, regardless of recording sample rate setting
16. **Canary chunking**: Test with audio >40 seconds — confirm it is chunked, each chunk transcribed, and results stitched correctly
17. **CLI**: `cv2t.exe download-model --engine whisper --target-dir <path>` downloads the model to the specified directory and exits
18. **GPU deps**: Verify the pinned CUDA/cuDNN/CTranslate2 version matrix works on a clean Windows machine

### Decisions

- **New repo `kwp490/cv2t`** — old `canary-voice-to-text` is NOT modified
- **Canary loading method**: NeMo/torch is the default (proven in v1). ONNX is an experimental branch, deferred until parity is proven on real audio. Do NOT use Whisper-style special tokens for Canary prompting — use the NeMo conversation/audio API.
- **Canary 40-second chunking**: Mandatory. Audio longer than 40 seconds is split into overlapping chunks, transcribed independently, and stitched. Never pass arbitrary-length recordings into a single Canary inference call.
- **torch dependency**: Required for Canary (NeMo). Not needed for Whisper-only installs. The `.exe` is Whisper-only; Canary users use the source install path.
- **Two distribution paths**: **Whisper-only standalone `.exe`** (no torch, onedir build) + **source install for Canary** (requires torch/NeMo via `uv sync --extra canary`). A single all-engines binary is deferred until a torch-free Canary runtime is proven.
- **No Docker/WSL/HTTP** anywhere in the new project
- **Config path**: `%APPDATA%\CV2T\` (separate from v1's `%APPDATA%\CanaryVoiceToText\`)
- **Model storage**: `%LOCALAPPDATA%\CV2T\models` (user-writable, no admin needed for downloads/updates). Binaries install to `C:\Program Files\CV2T\`.
- **Optional dependencies**: `cv2t[canary]` (NeMo+torch), `cv2t[canary-onnx]` (experimental), `cv2t[whisper]`, `cv2t[all]` — core GUI installs without model deps
- **`uv`** for all dependency management, replacing pip and requirements.txt
- **`nvidia-ml-py` for GPU monitoring** (replaces `pynvml` which is now deprecated and wraps `nvidia-ml-py` anyway). Import remains `import pynvml`.
- **Two installer paths** — Source installer (`Install-CV2T-Source.ps1`) for developers (both engines), binary installer (`Install-CV2T-Bin.ps1`) for end users (Whisper-only `.exe`). These must not be conflated.
- **Sample rate**: `sample_rate` in settings is a recording parameter only. Audio is always resampled to 16 kHz mono float32 before engine calls.
- **Whisper model default**: `"large-v3-turbo"` (library-resolved name), not a hardcoded HuggingFace repo path.
- **Validation**: Use a bundled short speech fixture, not a sine wave. Assert on loose transcript match.
- **CLI contract**: `cv2t download-model --engine ... --target-dir ...` is defined in the entry point (Step 12) so the binary installer can call it.
- **PyInstaller build mode**: `onedir` (not `onefile`) for GPU builds to avoid >2 GB extraction issues and temp directory overhead.
- **GPU dependency matrix**: Pin and document exact CUDA 12.x / cuDNN 9.x / CTranslate2 version combinations. Test in CI.

### Critical Implementation Constraints

These constraints apply across ALL prompts. Violating any of them will produce a broken application:

1. **Audio must be 1D mono float32 `(samples,)` at 16 kHz for engine calls** — Both CTranslate2 and NeMo expect 16 kHz mono input. The `get_raw_audio()` method must flatten/downmix before returning. A shared `ensure_16khz()` utility resamples from the recording sample rate to 16 kHz before any engine call. The `sample_rate` setting controls recording only, never engine input.

2. **Canary audio must be chunked to ≤40 seconds** — The model was trained on a maximum of 40 seconds of audio. The Canary engine must split longer recordings into overlapping chunks (e.g., 30s with 2s overlap), transcribe each, and stitch the results. This is mandatory, not optional.

3. **Canary uses NeMo conversation-style prompts, NOT Whisper-style tokens** — The correct API is `model.generate(prompts=[[{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}", "audio": [path]}]], ...)`. Do NOT use `<|startoftranscript|>`, `<|en|>`, or similar Whisper control tokens.

4. **Clipboard writes ONLY on the main Qt thread** — `set_clipboard_text()` uses Win32 `OpenClipboard()` which is thread-affine. The transcription worker emits a `result` signal; the connected slot runs on the main thread and does the clipboard write. `simulate_paste()` (which uses `keyboard.send()`, not clipboard) is safe to dispatch to a worker.

5. **Explicit GPU memory cleanup in `unload()`** — NeMo/torch, ONNX Runtime `InferenceSession`, and CTranslate2 models can hold GPU memory after Python reference is dropped. Use `del self._model; self._model = None; gc.collect()`. For torch-based engines, also call `torch.cuda.empty_cache()`.

6. **The `.exe` is Whisper-only** — Exclude `torch`, `torchaudio`, `torchvision`, `nemo`, `nemo_toolkit` from PyInstaller builds. Canary requires NeMo/torch and is only available via source install. Do not promise a single all-engines binary until a torch-free Canary runtime is proven.

7. **Bundle PortAudio DLL** — `sounddevice` requires `libportaudio64bit.dll` which PyInstaller does not collect automatically. Use `collect_dynamic_libs('sounddevice')` in the spec file.

8. **Document Windows Defender exclusions** — The `keyboard` library's `SetWindowsHookEx` calls trigger heuristic AV detections. The README must include exclusion instructions or users will report the `.exe` as "blocked by antivirus."

9. **Bundle CUDA shared libraries OR document system-wide CUDA requirement** — `ctranslate2` ships CUDA/cuDNN DLLs inside its pip package. PyInstaller does not auto-collect these binary dependencies. Either use `collect_dynamic_libs('ctranslate2')` in `cv2t.spec`, or document that CUDA 12.x + cuDNN 9.x must be installed system-wide on the target machine.

10. **Pin GPU dependency versions** — Do not leave GPU dependency ranges open-ended. Pin and test exact combinations: CUDA 12.x, cuDNN 9.x, CTranslate2 4.5.x, faster-whisper 1.1.x (or current known-good versions). Document this matrix in README and validate in CI or release testing.

### Further Considerations

1. **Cold start time**: Model loading takes 30-120s. Show a progress-style label in the Model Engine panel ("Loading model… 45s elapsed") with an elapsed-time counter updated by a QTimer.
2. **Nuitka vs PyInstaller**: Nuitka produces smaller/faster executables but is harder to configure. Start with PyInstaller (proven in v1), evaluate Nuitka as a follow-up.
3. **Whisper model auto-download**: `faster-whisper` auto-downloads models on first use to `~/.cache/huggingface/`. Always pass `download_root=models_dir` to redirect to the configured model path (`%LOCALAPPDATA%\CV2T\models`). Support `cv2t download-model --engine whisper` for offline prep.
4. **DirectML fallback**: `onnxruntime-directml` could enable AMD/Intel GPU support in the future. Not in scope for v2.0 but worth noting in the architecture for extensibility.
5. **Hotkey fallback for managed environments**: The `keyboard` library uses global hooks (`SetWindowsHookEx`) which are commonly flagged by security tools in enterprise environments. Keep `keyboard` as the default, but add a fallback using Qt-native shortcuts (`QShortcut` / `QAction` with `Qt.ApplicationShortcut`) when the global hook layer fails to register or is unavailable. This ensures the app remains functional (with window-focused hotkeys only) even when global hooks are blocked.
6. **ONNX Canary experiment**: Track in a GitHub issue. Only graduate to default once NeMo parity is proven on real speech audio with benchmarks. The community ONNX port's config identifies `Qwen3ForCausalLM` (a text LM), not an ASR-specific contract — thorough validation is needed.
