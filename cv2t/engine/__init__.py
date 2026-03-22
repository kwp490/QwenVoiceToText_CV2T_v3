"""
Engine registry — lazy-loads available engines based on installed dependencies.
"""

from __future__ import annotations

import logging
from typing import Dict, Type

log = logging.getLogger(__name__)

ENGINES: Dict[str, Type] = {}

# Canary — requires NeMo + torch
try:
    from .canary import CanaryEngine
    ENGINES["canary"] = CanaryEngine
except ImportError:
    log.debug("Canary engine unavailable (NeMo/torch not installed)")

# Whisper — requires faster-whisper (CTranslate2)
try:
    from .whisper import WhisperEngine
    ENGINES["whisper"] = WhisperEngine
except ImportError:
    log.debug("Whisper engine unavailable (faster-whisper not installed)")
