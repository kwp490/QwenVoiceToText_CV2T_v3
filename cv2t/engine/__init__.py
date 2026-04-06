"""
Engine registry — lazy-loads available engines based on installed dependencies.
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, Type

log = logging.getLogger(__name__)

ENGINES: Dict[str, Type] = {}

# ── Canary ───────────────────────────────────────────────────────────────────
#
# In a frozen (PyInstaller) build torch/NeMo are *excluded* by cv2t.spec, so
# native Canary is never available — only the subprocess bridge can provide it.
#
# In a source install we attempt ``import torch`` to verify torch is genuinely
# loadable.  ``importlib.util.find_spec`` is NOT used anywhere because it is
# unreliable inside frozen builds: it can discover packages from the *system*
# Python even when they are listed in the spec's ``excludes``.

if getattr(sys, "frozen", False):
    # ── Frozen binary: only the subprocess bridge is possible ────────────
    try:
        from .canary_bridge import CanaryBridgeEngine, canary_env_available
        if canary_env_available():
            ENGINES["canary"] = CanaryBridgeEngine
            log.debug("Canary engine available via subprocess bridge (canary-env)")
        else:
            log.debug("Canary bridge: canary-env not found")
    except ImportError:
        log.debug("Canary bridge: canary_bridge module not importable")
else:
    # ── Source install: try native torch first, then bridge ──────────────
    try:
        import torch as _torch  # noqa: F401 — fails fast without torch
        del _torch
        from .canary import CanaryEngine
        ENGINES["canary"] = CanaryEngine
    except ImportError:
        try:
            from .canary_bridge import CanaryBridgeEngine, canary_env_available
            if canary_env_available():
                ENGINES["canary"] = CanaryBridgeEngine
                log.debug("Canary engine available via subprocess bridge (canary-env)")
            else:
                log.debug("Canary engine unavailable (no torch, no canary-env)")
        except ImportError:
            log.debug("Canary engine unavailable (no torch, bridge not importable)")

# Whisper — requires faster-whisper (CTranslate2)
try:
    from .whisper import WhisperEngine
    ENGINES["whisper"] = WhisperEngine
except ImportError:
    log.debug("Whisper engine unavailable (faster-whisper not installed)")


# ── Model-file detection ─────────────────────────────────────────────────────

_WHISPER_REQUIRED_FILES = ("config.json", "model.bin", "tokenizer.json")


def _model_files_exist(engine_name: str, model_path: str) -> bool:
    """Return True if the model files for *engine_name* are present on disk."""
    import os

    if engine_name == "whisper":
        whisper_dir = os.path.join(model_path, "whisper")
        return all(
            os.path.isfile(os.path.join(whisper_dir, f))
            for f in _WHISPER_REQUIRED_FILES
        )
    if engine_name == "canary":
        canary_dir = os.path.join(model_path, "canary")
        return os.path.isdir(canary_dir) and os.path.isfile(
            os.path.join(canary_dir, "config.json")
        )
    return False


def get_available_engines(model_path: str) -> list:
    """Return engine names whose dependencies AND model files are installed."""
    return [
        name for name in ENGINES if _model_files_exist(name, model_path)
    ]
