"""
Canary engine — NeMo/torch implementation (default, proven).

Uses NVIDIA NeMo's SALM.from_pretrained() and conversation-style prompts.
Mandatory 40-second chunking for audio longer than 40s.
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf

from .audio_utils import chunk_audio, ensure_16khz, stitch_transcripts
from .base import _cleanup_gpu_memory

log = logging.getLogger(__name__)

_MAX_CHUNK_SECONDS = 30.0
_OVERLAP_SECONDS = 2.0


class CanaryEngine:
    """NeMo-based Canary Qwen 2.5B speech engine."""

    def __init__(self) -> None:
        self._model = None
        self._device: str = "cuda"

    @property
    def name(self) -> str:
        return "canary"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def vram_estimate_gb(self) -> float:
        return 5.0

    def load(self, model_path: str, device: str = "cuda") -> None:
        """Load Canary model via NeMo SALM.from_pretrained()."""
        import torch
        from nemo.collections.speechlm2.models import SALM

        self._device = device

        # Use engine-specific subdirectory; fall back to base path for
        # existing installations.
        canary_subdir = os.path.join(model_path, "canary")
        if os.path.isdir(canary_subdir):
            model_path = canary_subdir

        # dtype selection: bfloat16 on Ampere+, float16 otherwise
        load_dtype = torch.float16
        if torch.cuda.is_available():
            try:
                if torch.cuda.is_bf16_supported():
                    load_dtype = torch.bfloat16
            except Exception:
                pass

        log.info("Loading Canary model from %s (dtype=%s)", model_path, load_dtype)
        self._model = SALM.from_pretrained(model_path)
        self._model = self._model.eval().to(device=device, dtype=load_dtype)
        log.info("Canary model loaded")

        # Warmup with dummy inference for CUDA kernel compilation
        self._warmup()

    def _warmup(self) -> None:
        """Run a dummy inference to compile CUDA kernels."""
        import torch

        try:
            fd, warmup_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(warmup_path, np.zeros(4000, dtype=np.float32), 16000)

            conversation = [[{
                "role": "user",
                "content": f"Transcribe: {self._model.audio_locator_tag}",
                "audio": [warmup_path],
            }]]

            with torch.inference_mode():
                self._model.generate(prompts=conversation, max_new_tokens=2)
            log.info("Canary warmup complete")
        except Exception as exc:
            log.warning("Canary warmup skipped: %s", exc)
        finally:
            try:
                os.unlink(warmup_path)
            except OSError:
                pass

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio with mandatory 40s chunking.

        Accepts arbitrary-length 1D float32 mono audio at any sample rate.
        """
        import torch

        if self._model is None:
            raise RuntimeError("Canary model not loaded")

        # Resample to 16 kHz
        audio_16k = ensure_16khz(audio, sample_rate)
        if len(audio_16k) == 0:
            return ""

        # Chunk audio (30s chunks, 2s overlap — well within 40s limit)
        chunks = chunk_audio(audio_16k, 16000, _MAX_CHUNK_SECONDS, _OVERLAP_SECONDS)
        log.info("Transcribing %d chunk(s)", len(chunks))

        texts = []
        for i, chunk in enumerate(chunks):
            text = self._transcribe_chunk(chunk, torch)
            log.info("Chunk %d/%d: %s", i + 1, len(chunks), text[:80] if text else "(empty)")
            texts.append(text)

        return stitch_transcripts(texts)

    def _transcribe_chunk(self, chunk: np.ndarray, torch_module) -> str:
        """Transcribe a single audio chunk via NeMo conversation API."""
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp_path, chunk, 16000)

            duration = len(chunk) / 16000
            max_tokens = max(64, int(duration * 20))

            conversation = [[{
                "role": "user",
                "content": f"Transcribe the following: {self._model.audio_locator_tag}",
                "audio": [tmp_path],
            }]]

            with torch_module.inference_mode():
                response = self._model.generate(
                    prompts=conversation,
                    max_new_tokens=max_tokens,
                    temperature=0.0,
                    top_k=1,
                )[0]

            text = self._model.tokenizer.ids_to_text(response.cpu())
            text = text.replace("<|endoftext|>", "").strip()
            return text
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def unload(self) -> None:
        """Release model and free VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
        _cleanup_gpu_memory()
        log.info("Canary model unloaded")
