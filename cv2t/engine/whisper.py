"""
Whisper engine — faster-whisper (CTranslate2) implementation.

No torch dependency. Native Windows CUDA support.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from typing import Optional

import numpy as np
from huggingface_hub import snapshot_download

from .audio_utils import ensure_16khz
from .base import _cleanup_gpu_memory

log = logging.getLogger(__name__)


def _add_nvidia_dll_dirs() -> None:
    """Add nvidia pip-package DLL directories to PATH on Windows.

    Packages like nvidia-cublas-cu12 install DLLs into
    ``site-packages/nvidia/<lib>/bin/`` which is not on the default
    DLL search path.  We add every such ``bin/`` directory so that
    CTranslate2 can find cublas64_12.dll, cudnn*.dll, etc.
    """
    if sys.platform != "win32":
        return

    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return

    added: list[str] = []
    for search_path in spec.submodule_search_locations:
        nvidia_root = os.path.normpath(search_path)
        if not os.path.isdir(nvidia_root):
            continue
        for child in os.listdir(nvidia_root):
            bin_dir = os.path.join(nvidia_root, child, "bin")
            if os.path.isdir(bin_dir) and bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                # Also register with os.add_dll_directory (Python 3.8+)
                try:
                    os.add_dll_directory(bin_dir)
                except OSError:
                    pass
                added.append(bin_dir)

    if added:
        log.info("Added NVIDIA DLL directories to PATH: %s", added)

_WHISPER_MODEL_ID = "large-v3-turbo"
_WHISPER_REPO_ID = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
_WHISPER_REQUIRED_FILES = ("config.json", "model.bin", "tokenizer.json")
_WHISPER_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


def _whisper_model_ready(model_dir: str) -> bool:
    return all(os.path.isfile(os.path.join(model_dir, name)) for name in _WHISPER_REQUIRED_FILES)


def _log_runtime_diagnostics() -> None:
    try:
        import faster_whisper
        import faster_whisper.utils as faster_whisper_utils

        models = getattr(faster_whisper_utils, "_MODELS", None)
        resolved_repo = None
        if isinstance(models, dict):
            resolved_repo = models.get(_WHISPER_MODEL_ID)

        log.info("faster_whisper.__file__ = %s", getattr(faster_whisper, "__file__", "<unknown>"))
        log.info(
            "faster_whisper.utils.__file__ = %s",
            getattr(faster_whisper_utils, "__file__", "<unknown>"),
        )
        log.info("_MODELS['%s'] = %r", _WHISPER_MODEL_ID, resolved_repo)
    except Exception as exc:
        log.warning("Unable to inspect faster-whisper runtime diagnostics: %s", exc)


class WhisperEngine:
    """Faster-whisper (CTranslate2) speech engine."""

    def __init__(self) -> None:
        self._model = None

    @property
    def name(self) -> str:
        return "whisper"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def vram_estimate_gb(self) -> float:
        return 3.0

    def load(self, model_path: str, device: str = "cuda") -> None:
        """Load a Whisper model via faster-whisper from a local model directory."""
        _add_nvidia_dll_dirs()
        from faster_whisper import WhisperModel

        compute = "float16" if device == "cuda" else "auto"
        _log_runtime_diagnostics()

        if os.path.isdir(model_path) and _whisper_model_ready(model_path):
            local_dir = model_path
            log.info("Whisper model files found locally: %s", local_dir)
        else:
            log.info(
                "Downloading Whisper model '%s' from %s into %s",
                _WHISPER_MODEL_ID,
                _WHISPER_REPO_ID,
                model_path,
            )
            local_dir = snapshot_download(
                repo_id=_WHISPER_REPO_ID,
                local_dir=model_path,
                local_files_only=False,
                allow_patterns=_WHISPER_ALLOW_PATTERNS,
            )
            log.info("Model downloaded to %s", local_dir)

        log.info(
            "Loading WhisperModel from %s (device=%s, compute=%s)",
            local_dir, device, compute,
        )
        self._model = WhisperModel(
            local_dir,
            device=device,
            compute_type=compute,
        )

        # Verify CUDA actually works end-to-end by running a full
        # mini-transcription (encode + decode + VAD).  cuBLAS / cuDNN
        # failures can surface at any stage; testing only encode is not
        # enough — the original verification missed decode/beam-search
        # crashes that killed the process with no Python traceback.
        if device == "cuda":
            try:
                _test = np.zeros(16000, dtype=np.float32)
                # Full pipeline: feature extraction → encode → decode → VAD
                segs, _ = self._model.transcribe(
                    _test, language="en", beam_size=1, vad_filter=False,
                )
                list(segs)  # force decode
                segs2, _ = self._model.transcribe(
                    _test, language="en", beam_size=1, vad_filter=True,
                )
                list(segs2)  # force VAD path
                log.info("CUDA verification passed")
            except Exception as exc:
                log.warning("CUDA inference failed (%s); falling back to CPU", exc)
                del self._model
                self._model = None
                _cleanup_gpu_memory()
                compute = "auto"
                device = "cpu"
                log.info(
                    "Reloading WhisperModel from %s (device=%s, compute=%s)",
                    local_dir, device, compute,
                )
                self._model = WhisperModel(
                    local_dir,
                    device=device,
                    compute_type=compute,
                )

        log.info("Whisper model loaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio. Accepts arbitrary-length 1D float32 mono."""
        if self._model is None:
            raise RuntimeError("Whisper model not loaded")

        # Resample to 16 kHz
        audio_16k = ensure_16khz(audio, sample_rate)
        if len(audio_16k) == 0:
            return ""

        # CTranslate2 requires a contiguous float32 array — numpy views/slices
        # from trim_silence can be non-contiguous and cause native crashes.
        audio_16k = np.ascontiguousarray(audio_16k, dtype=np.float32)

        log.debug(
            "Transcribing: shape=%s dtype=%s contiguous=%s",
            audio_16k.shape, audio_16k.dtype, audio_16k.flags["C_CONTIGUOUS"],
        )

        # Flush log handlers so crash diagnostics survive a native abort.
        for h in logging.getLogger().handlers:
            h.flush()

        segments, info = self._model.transcribe(
            audio_16k,
            language="en",
            beam_size=5,
            vad_filter=True,
        )

        # Eagerly consume the generator — lazy iteration can crash in native
        # code with no opportunity for Python-level error recovery.
        log.info("Consuming transcription segments…")
        text_parts = [seg.text.strip() for seg in segments]
        log.info("Transcription complete (%d segment(s))", len(text_parts))
        return " ".join(text_parts)

    def unload(self) -> None:
        """Release model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        _cleanup_gpu_memory()
        log.info("Whisper model unloaded")
