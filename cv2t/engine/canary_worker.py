#!/usr/bin/env python3
"""
Canary inference worker — runs in a separate Python environment with torch/NeMo.

This script is launched by CV2T's CanaryBridgeEngine as a subprocess when
the Canary engine is installed via Enable-Canary.ps1 but torch/NeMo are not
available in the main frozen binary.

Protocol: JSON lines over stdin/stdout.

  -> {"command": "load", "model_path": "...", "device": "cuda"}
  <- {"status": "ready"} | {"status": "error", "message": "..."}

  -> {"command": "transcribe", "audio_file": "...", "language": "en"}
  <- {"status": "ok", "text": "..."} | {"status": "error", "message": "..."}

  -> {"command": "shutdown"}
  <- (process exits)

Dependencies (installed in canary-env):
    torch, nemo_toolkit[asr], soundfile, numpy
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import tempfile

log = logging.getLogger(__name__)

# ── Audio utilities (standalone — no cv2t imports) ───────────────────────────


def _ensure_16khz(audio, source_sr: int):
    """Resample audio to 16 kHz if needed."""
    import numpy as np

    if source_sr == 16000:
        return audio
    duration = len(audio) / source_sr
    target_len = int(duration * 16000)
    if target_len == 0:
        return np.array([], dtype=np.float32)
    indices = np.linspace(0, len(audio) - 1, target_len)
    left = np.floor(indices).astype(int)
    right = np.minimum(left + 1, len(audio) - 1)
    frac = (indices - left).astype(np.float32)
    return audio[left] * (1.0 - frac) + audio[right] * frac


def _chunk_audio(audio, sr: int, max_seconds: float = 30.0, overlap_seconds: float = 2.0):
    """Split audio into overlapping chunks."""
    max_samples = int(max_seconds * sr)
    overlap_samples = int(overlap_seconds * sr)
    step = max_samples - overlap_samples
    if len(audio) <= max_samples:
        return [audio]
    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + max_samples, len(audio))
        chunks.append(audio[start:end])
        if end >= len(audio):
            break
        start += step
    return chunks


def _stitch_transcripts(texts: list) -> str:
    """Join chunk transcripts, deduplicating overlap at boundaries."""
    if not texts:
        return ""
    result = texts[0]
    for nxt in texts[1:]:
        if not nxt:
            continue
        if not result:
            result = nxt
            continue
        words_r = result.split()
        words_n = nxt.split()
        best_overlap = 0
        max_check = min(len(words_r), len(words_n), 10)
        for k in range(1, max_check + 1):
            if words_r[-k:] == words_n[:k]:
                best_overlap = k
        if best_overlap > 0:
            result = result + " " + " ".join(words_n[best_overlap:])
        else:
            result = result + " " + nxt
    return result.strip()


# ── Model state ──────────────────────────────────────────────────────────────

_model = None
_MAX_CHUNK_SECONDS = 30.0
_OVERLAP_SECONDS = 2.0


def _get_temp_dir() -> str:
    d = os.path.join(tempfile.gettempdir(), "cv2t")
    os.makedirs(d, exist_ok=True)
    return d


def _load_model(model_path: str, device: str = "cuda") -> None:
    """Load Canary NeMo SALM model."""
    global _model
    import importlib.machinery
    import types
    import warnings

    import numpy as np

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    import torch

    torch.compiler.disable(recursive=True)

    # Stub wandb — NeMo transitively imports it but it's not needed for inference
    if "wandb" not in sys.modules:
        _wandb = types.ModuleType("wandb")
        _wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
        _wandb.__path__ = []
        _wandb.__package__ = "wandb"
        _wandb.__version__ = "0.0.0"
        sys.modules["wandb"] = _wandb

    # Stub datasets.distributed — NeMo imports it transitively but inference
    # doesn't need distributed dataset splitting.  If the real submodule is
    # missing (e.g. stripped or incompatible datasets version), provide a shim.
    if "datasets.distributed" not in sys.modules:
        try:
            import datasets.distributed  # noqa: F401 — use real module if available
        except (ImportError, ModuleNotFoundError):
            _dd = types.ModuleType("datasets.distributed")
            _dd.__spec__ = importlib.machinery.ModuleSpec("datasets.distributed", None)
            _dd.__package__ = "datasets"
            _dd.split_dataset_by_node = lambda dataset, rank=0, world_size=1: dataset
            sys.modules["datasets.distributed"] = _dd

    # Stub transformers version check (NeMo 2.x vs. newer huggingface-hub)
    if "transformers.dependency_versions_check" not in sys.modules:
        _dvc = types.ModuleType("transformers.dependency_versions_check")
        _dvc.dep_version_check = lambda pkg, hint=None: None
        sys.modules["transformers.dependency_versions_check"] = _dvc

    # Suppress noisy warnings from NeMo's import chain
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

    # Use engine-specific subdirectory
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

    # Disable SDPA kernels that may be incomplete on newer architectures
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    _model = SALM.from_pretrained(model_path)
    _model = _model.eval().to(device=device, dtype=load_dtype)
    log.info("Canary model loaded")

    # Warmup with dummy inference
    import soundfile as sf

    try:
        fd, warmup_path = tempfile.mkstemp(suffix=".wav", dir=_get_temp_dir())
        os.close(fd)
        sf.write(warmup_path, np.zeros(4000, dtype=np.float32), 16000)
        conversation = [[{
            "role": "user",
            "content": f"Transcribe: {_model.audio_locator_tag}",
            "audio": [warmup_path],
        }]]
        with torch.no_grad():
            _model.generate(prompts=conversation, max_new_tokens=2)
        torch.cuda.synchronize()
        log.info("Canary warmup complete")
    except Exception as exc:
        log.warning("Canary warmup skipped: %s", exc)
    finally:
        try:
            os.unlink(warmup_path)
        except OSError:
            pass

    # Restore log levels
    for _name in _noisy_loggers:
        logging.getLogger(_name).setLevel(logging.WARNING)


def _transcribe(audio_file: str, language: str = "en") -> str:
    """Transcribe audio from a WAV file."""
    import numpy as np
    import soundfile as sf
    import torch

    audio, sr = sf.read(audio_file, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_16k = _ensure_16khz(audio, sr)
    if len(audio_16k) == 0:
        return ""

    chunks = _chunk_audio(audio_16k, 16000, _MAX_CHUNK_SECONDS, _OVERLAP_SECONDS)
    texts = []
    for chunk in chunks:
        fd, chunk_path = tempfile.mkstemp(suffix=".wav", dir=_get_temp_dir())
        os.close(fd)
        try:
            sf.write(chunk_path, chunk, 16000)

            duration = len(chunk) / 16000
            max_tokens = max(64, int(duration * 20))

            conversation = [[{
                "role": "user",
                "content": (
                    f"Transcribe the following: "
                    f"{_model.audio_locator_tag}"
                ),
                "audio": [chunk_path],
            }]]
            with torch.no_grad():
                output = _model.generate(
                    prompts=conversation, max_new_tokens=max_tokens,
                )

            # generate() returns a list of token-ID tensors, not strings.
            # Decode via the model's tokenizer, matching the native engine.
            if isinstance(output, list) and output:
                response = output[0]
            else:
                response = output
            if isinstance(response, str):
                text = response
            elif hasattr(response, "cpu"):
                # Tensor of token IDs — decode to text.
                # The tensor may be multi-dimensional (e.g. [1, seq_len])
                # so squeeze and convert to a flat Python list of ints
                # before passing to ids_to_text.
                ids = response.cpu().squeeze().tolist()
                if isinstance(ids, int):
                    ids = [ids]
                text = _model.tokenizer.ids_to_text(ids)
                text = text.replace("<|endoftext|>", "")
            else:
                text = str(response)
            texts.append(text.strip())
        finally:
            try:
                os.unlink(chunk_path)
            except OSError:
                pass

    return _stitch_transcripts(texts)


# ── JSON-lines protocol ─────────────────────────────────────────────────────

# Keep a reference to the real stdout for protocol messages.
# sys.stdout is redirected to stderr in main() so that stray print()
# calls from NeMo / torch / third-party code do not corrupt the
# JSON-lines channel.
_proto_stdout = sys.stdout


def _send(obj: dict) -> None:
    """Write a JSON line to the protocol channel (real stdout)."""
    _proto_stdout.write(json.dumps(obj) + "\n")
    _proto_stdout.flush()


def _recv() -> dict | None:
    """Read a JSON line from stdin.  Returns None on EOF."""
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line.strip())


def main() -> int:
    global _model

    # Redirect stdout -> stderr so that stray print() calls from NeMo,
    # torch, transformers, etc. do not land on the JSON-lines protocol
    # channel.  Protocol messages use _proto_stdout (saved above).
    sys.stdout = sys.stderr

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [canary-worker] %(levelname)s %(message)s",
        stream=sys.stderr,  # Logs go to stderr; protocol goes to _proto_stdout
    )

    log.info("Canary worker started (pid=%d)", os.getpid())

    while True:
        msg = _recv()
        if msg is None:
            break

        cmd = msg.get("command")

        if cmd == "load":
            try:
                _load_model(msg["model_path"], msg.get("device", "cuda"))
                _send({"status": "ready"})
            except Exception as exc:
                log.error("Model load failed: %s", exc, exc_info=True)
                _send({"status": "error", "message": str(exc)})

        elif cmd == "transcribe":
            if _model is None:
                _send({"status": "error", "message": "Model not loaded"})
                continue
            try:
                text = _transcribe(msg["audio_file"], msg.get("language", "en"))
                _send({"status": "ok", "text": text})
            except Exception as exc:
                log.error("Transcription failed: %s", exc, exc_info=True)
                _send({"status": "error", "message": str(exc)})

        elif cmd == "shutdown":
            log.info("Shutdown requested")
            break

        else:
            _send({"status": "error", "message": f"Unknown command: {cmd}"})

    # Cleanup
    if _model is not None:
        del _model
        _model = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    log.info("Canary worker exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
