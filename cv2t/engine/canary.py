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


def _get_temp_dir() -> str:
    """Return a writable temp directory for transient WAV files.

    Uses the user's TEMP directory rather than the install directory
    to avoid Windows Defender real-time scanning and UAC issues under
    C:\\Program Files.
    """
    d = os.path.join(tempfile.gettempdir(), "cv2t")
    os.makedirs(d, exist_ok=True)
    return d


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
        import importlib.machinery
        import sys
        import types

        # Disable torch.compile / Dynamo before importing torch.  NeMo/SALM
        # can trigger compiled CUDA graphs that cache input shapes; the
        # second generate() call with different audio length hangs because
        # the cached graph shape no longer matches.
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

        # Force synchronous CUDA kernel execution.  Blackwell (sm_120)
        # with NeMo SALM can hang on subsequent generate() calls when
        # async-launched kernels silently corrupt state.  The ~10-20%
        # throughput cost is negligible for a voice-to-text app.
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

        import torch

        torch.compiler.disable(recursive=True)

        # NeMo transitively imports wandb (via TTS helpers / accelerate) at
        # module level.  wandb is not needed for inference and is often
        # broken.  Install a stub package so the entire import chain
        # succeeds, including importlib.util.find_spec("wandb") checks.
        if "wandb" not in sys.modules:
            _wandb = types.ModuleType("wandb")
            _wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
            _wandb.__path__ = []
            _wandb.__package__ = "wandb"
            _wandb.__version__ = "0.0.0"
            sys.modules["wandb"] = _wandb

        # NeMo → lightning → torchmetrics → transformers triggers a
        # runtime version check that rejects huggingface-hub >= 1.0.
        # Older transformers pins <1.0 in its dependency_versions_table
        # but NeMo 2.x pulls huggingface-hub 1.x.  Stub the check
        # module before the import chain reaches it.  The stub must
        # expose dep_version_check() because other transformers
        # submodules import it by name.
        if "transformers.dependency_versions_check" not in sys.modules:
            _dvc = types.ModuleType("transformers.dependency_versions_check")
            _dvc.dep_version_check = lambda pkg, hint=None: None
            sys.modules["transformers.dependency_versions_check"] = _dvc

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

        # Blackwell (sm_120) may have incomplete flash/memory-efficient
        # SDPA kernels in current PyTorch builds.  Fall back to the
        # portable math implementation to avoid native CUDA crashes.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        self._model = SALM.from_pretrained(model_path)
        self._model = self._model.eval().to(device=device, dtype=load_dtype)
        log.info("Canary model loaded")

        # Warmup with dummy inference for CUDA kernel compilation
        self._warmup()

    def _warmup(self) -> None:
        """Run a dummy inference to compile CUDA kernels."""
        import torch

        try:
            fd, warmup_path = tempfile.mkstemp(suffix=".wav", dir=_get_temp_dir())
            os.close(fd)
            sf.write(warmup_path, np.zeros(4000, dtype=np.float32), 16000)

            conversation = [[{
                "role": "user",
                "content": f"Transcribe: {self._model.audio_locator_tag}",
                "audio": [warmup_path],
            }]]

            with torch.no_grad():
                self._model.generate(prompts=conversation, max_new_tokens=2)
            torch.cuda.synchronize()
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

        result = stitch_transcripts(texts)

        # Release Python references to transient CUDA tensors from the
        # generate() call.  Without this, dead tensor objects accumulate
        # and can confuse PyTorch's caching allocator on subsequent calls.
        gc.collect()

        return result

    def _transcribe_chunk(self, chunk: np.ndarray, torch_module) -> str:
        """Transcribe a single audio chunk via NeMo conversation API."""
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=_get_temp_dir())
        os.close(fd)
        try:
            sf.write(tmp_path, chunk, 16000)

            duration = len(chunk) / 16000
            max_tokens = max(64, int(duration * 20))
            log.info("chunk %.1fs → max_tokens=%d, file=%s", duration, max_tokens, tmp_path)

            conversation = [[{
                "role": "user",
                "content": f"Transcribe the following: {self._model.audio_locator_tag}",
                "audio": [tmp_path],
            }]]

            log.info("entering generate()")
            logging.getLogger().handlers[0].flush() if logging.getLogger().handlers else None

            with torch_module.no_grad():
                response = self._model.generate(
                    prompts=conversation,
                    max_new_tokens=max_tokens,
                )[0]

            torch_module.cuda.synchronize()
            log.info("generate() returned %d token(s)", len(response))
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
