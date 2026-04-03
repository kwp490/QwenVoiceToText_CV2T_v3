"""
Engine registry — lazy-loads available engines based on installed dependencies.
"""

from __future__ import annotations

import logging
from typing import Dict, Type

log = logging.getLogger(__name__)

ENGINES: Dict[str, Type] = {}

# Canary — requires NeMo + torch (direct import for source installs)
try:
    from .canary import CanaryEngine
    ENGINES["canary"] = CanaryEngine
except ImportError:
    # Canary subprocess bridge — for binary installs with canary-env
    try:
        from .canary_bridge import CanaryBridgeEngine, canary_env_available
        if canary_env_available():
            ENGINES["canary"] = CanaryBridgeEngine
            log.debug("Canary engine available via subprocess bridge (canary-env)")
        else:
            log.debug("Canary engine unavailable (NeMo/torch not installed, no canary-env)")
    except ImportError:
        log.debug("Canary engine unavailable (NeMo/torch not installed)")

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
