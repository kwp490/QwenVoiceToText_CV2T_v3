"""
Engine abstraction — Protocol defining the contract all speech engines must follow.
"""

from __future__ import annotations

import gc
from typing import Protocol, runtime_checkable


@runtime_checkable
class SpeechEngine(Protocol):
    """Contract for all speech-to-text engines."""

    @property
    def name(self) -> str:
        """Human-readable engine name."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded and ready for inference."""
        ...

    @property
    def vram_estimate_gb(self) -> float:
        """Estimated VRAM usage in GB when loaded."""
        ...

    def load(self, model_path: str, device: str = "cuda") -> None:
        """Load model weights into memory/VRAM."""
        ...

    def transcribe(self, audio: "np.ndarray", sample_rate: int) -> str:
        """Transcribe audio. Accepts arbitrary-length 1D float32 mono at any sample rate.

        The engine is responsible for resampling to 16 kHz and chunking
        if the model has a maximum input duration.
        """
        ...

    def unload(self) -> None:
        """Release model and free GPU memory."""
        ...


def _cleanup_gpu_memory() -> None:
    """Best-effort GPU memory cleanup."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
