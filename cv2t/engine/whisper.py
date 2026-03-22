"""
Whisper engine — faster-whisper (CTranslate2) implementation.

No torch dependency. Native Windows CUDA support.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import numpy as np

from .audio_utils import ensure_16khz
from .base import _cleanup_gpu_memory

log = logging.getLogger(__name__)


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
        """Load a Whisper model via faster-whisper.

        *model_path* can be either:
        - A model ID like "large-v3-turbo" (library resolves the repo)
        - A local directory containing the model + a `models_dir` parent
        """
        from faster_whisper import WhisperModel

        compute = "float16" if device == "cuda" else "int8"

        # If model_path is a directory containing a model, use it directly.
        # Otherwise, treat it as a models_dir and use the default model ID.
        import os
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "model.bin")):
            # model_path is a full model directory
            log.info("Loading Whisper model from local path: %s", model_path)
            self._model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute,
            )
        else:
            # model_path is a directory for storing downloads
            model_id = "large-v3-turbo"
            log.info(
                "Loading Whisper model %s (download_root=%s, device=%s)",
                model_id, model_path, device,
            )
            self._model = WhisperModel(
                model_id,
                device=device,
                compute_type=compute,
                download_root=model_path,
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

        segments, info = self._model.transcribe(
            audio_16k,
            language="en",
            beam_size=5,
            vad_filter=True,
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        return " ".join(text_parts)

    def unload(self) -> None:
        """Release model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        _cleanup_gpu_memory()
        log.info("Whisper model unloaded")
