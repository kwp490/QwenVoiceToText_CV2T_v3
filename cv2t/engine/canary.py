"""
Canary engine — NeMo/torch implementation (default, proven).

Uses NVIDIA NeMo's SALM.from_pretrained() and conversation-style prompts.
Mandatory 40-second chunking for audio longer than 40s.
"""

from __future__ import annotations

import gc
import logging
import os
import queue
import tempfile
import threading
from typing import Any, Optional

import numpy as np
import soundfile as sf

from .audio_utils import chunk_audio, ensure_16khz, stitch_transcripts
from .base import SpeechEngine, _cleanup_gpu_memory

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


class CanaryEngine(SpeechEngine):
    """NeMo-based Canary Qwen 2.5B speech engine.

    All PyTorch / CUDA operations are pinned to a single dedicated thread
    to prevent cross-thread CUDA context corruption.  QThreadPool assigns
    arbitrary worker threads for each task; NeMo SALM's internal state
    (KV caches, attention buffers) is not safe across OS threads.
    """

    def __init__(self) -> None:
        super().__init__()
        self._device: str = "cuda"

        # Dedicated inference thread — all CUDA / PyTorch work runs here.
        self._inf_queue: queue.Queue = queue.Queue()
        self._inf_thread = threading.Thread(
            target=self._inference_loop,
            name="canary-inference",
            daemon=True,
        )
        self._inf_thread.start()

    # ── Inference-thread helpers ──────────────────────────────────────────

    def _inference_loop(self) -> None:
        """Consume callables on the dedicated thread until sentinel."""
        while True:
            item = self._inf_queue.get()
            if item is None:          # shutdown sentinel
                break
            fn, args, kwargs, holder = item
            try:
                holder["value"] = fn(*args, **kwargs)
            except BaseException as exc:
                holder["error"] = exc
            finally:
                holder["done"].set()

    def _run_on_inf_thread(self, fn, *args: Any, **kwargs: Any) -> Any:
        """Submit *fn* to the inference thread and block until complete."""
        holder: dict = {"done": threading.Event(), "value": None, "error": None}
        self._inf_queue.put((fn, args, kwargs, holder))
        holder["done"].wait()
        if holder["error"] is not None:
            raise holder["error"]
        return holder["value"]

    @property
    def name(self) -> str:
        return "canary"

    @property
    def vram_estimate_gb(self) -> float:
        return 5.0

    def load(self, model_path: str, device: str = "cuda") -> None:
        """Load Canary model — delegates to the dedicated inference thread."""
        self._run_on_inf_thread(self._load_impl, model_path, device)

    def _load_impl(self, model_path: str, device: str) -> None:
        """Load Canary model via NeMo SALM.from_pretrained() (inference thread)."""
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

        # Suppress noisy one-time warnings from NeMo's import chain and
        # model loading: Megatron/Apex fallback, pydub/ffmpeg, torch
        # distributed redirects, triton, torch_dtype deprecation,
        # generation_config defaults.  All harmless for inference.
        import warnings
        for _pat in (
            r"Couldn't find ffmpeg",
            r"Redirects are currently not supported",
            r"triton not found",
            r"torch_dtype.*deprecated",
            r"generation_config.*default values",
        ):
            warnings.filterwarnings("ignore", message=_pat)
        _noisy_loggers = (
            "nemo", "nemo_logger", "nemo.utils.nemo_logging",
            "transformers", "transformers.generation.configuration_utils",
        )
        for _name in _noisy_loggers:
            logging.getLogger(_name).setLevel(logging.ERROR)

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

        # Restore third-party log levels so real warnings are still visible
        for _name in _noisy_loggers:
            logging.getLogger(_name).setLevel(logging.WARNING)

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

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str = "en") -> str:
        """Resample then delegate to the dedicated inference thread."""
        # Base-class guard + resampling on the calling thread (cheap),
        # then heavy inference on the dedicated CUDA thread.
        if self._model is None:
            raise RuntimeError(f"{self.name} model not loaded")
        audio_16k = ensure_16khz(audio, sample_rate)
        if len(audio_16k) == 0:
            return ""
        return self._run_on_inf_thread(self._transcribe_impl, audio_16k, language)

    def _transcribe_impl(self, audio_16k: np.ndarray, language: str = "en") -> str:
        """Transcribe 16 kHz audio with mandatory chunking (inference thread)."""
        import torch

        # Chunk audio (30s chunks, 2s overlap — well within 40s limit)
        chunks = chunk_audio(audio_16k, 16000, _MAX_CHUNK_SECONDS, _OVERLAP_SECONDS)
        log.info("Transcribing %d chunk(s)", len(chunks))

        texts = []
        for i, chunk in enumerate(chunks):
            text = self._transcribe_chunk(chunk, torch)
            log.info("Chunk %d/%d: %d chars", i + 1, len(chunks), len(text))
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
            log.info("chunk %.1fs -> max_tokens=%d, file=%s", duration, max_tokens, tmp_path)

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
        """Release model, free VRAM, and stop the inference thread."""
        if not self._inf_thread.is_alive():
            return
        # Drain any pending items so the sentinel is processed promptly.
        while not self._inf_queue.empty():
            try:
                self._inf_queue.get_nowait()
            except queue.Empty:
                break
        # Submit the cleanup, then the sentinel.  Use a short timeout so
        # we don't block forever if generate() is still running (the
        # daemon thread will be killed at process exit anyway).
        holder: dict = {"done": threading.Event(), "value": None, "error": None}
        self._inf_queue.put((self._unload_impl, (), {}, holder))
        self._inf_queue.put(None)          # shutdown sentinel
        holder["done"].wait(timeout=5)
        self._inf_thread.join(timeout=2)

    def _unload_impl(self) -> None:
        """Release model and free VRAM (inference thread)."""
        self._release_model()
