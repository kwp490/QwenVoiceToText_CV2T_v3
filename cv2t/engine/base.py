"""
Engine abstraction — ABC defining the contract all speech engines must follow.
"""

from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod

import numpy as np

from .audio_utils import ensure_16khz

log = logging.getLogger(__name__)


class SpeechEngine(ABC):
    """Base class for all speech-to-text engines.

    Provides shared ``is_loaded``, ``_release_model()``, and a
    ``transcribe()`` wrapper that resamples to 16 kHz and guards
    against empty audio / unloaded model.  Subclasses implement
    ``_transcribe_impl()`` for the engine-specific inference path.
    """

    def __init__(self) -> None:
        self._model = None

    # ── Abstract interface ───────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""

    @property
    @abstractmethod
    def vram_estimate_gb(self) -> float:
        """Estimated VRAM usage in GB when loaded."""

    @abstractmethod
    def load(self, model_path: str, device: str = "cuda") -> None:
        """Load model weights into memory/VRAM."""

    @abstractmethod
    def _transcribe_impl(self, audio_16k: np.ndarray) -> str:
        """Engine-specific transcription of 16 kHz mono float32 audio."""

    @abstractmethod
    def unload(self) -> None:
        """Release model and free GPU memory."""

    # ── Concrete shared logic ────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded and ready for inference."""
        return self._model is not None

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Resample to 16 kHz, guard empty/unloaded, then delegate."""
        if self._model is None:
            raise RuntimeError(f"{self.name} model not loaded")
        audio_16k = ensure_16khz(audio, sample_rate)
        if len(audio_16k) == 0:
            return ""
        return self._transcribe_impl(audio_16k)

    def _release_model(self) -> None:
        """Delete model reference and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        _cleanup_gpu_memory()
        log.info("%s model unloaded", self.name)


def _cleanup_gpu_memory() -> None:
    """Best-effort GPU memory cleanup."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
