"""
Configuration persistence for CV2T.

All data lives under the install directory (default C:\Program Files\CV2T).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

log = logging.getLogger(__name__)

INSTALL_DIR = Path(os.environ.get("CV2T_HOME", r"C:\Program Files\CV2T"))

DEFAULT_CONFIG_DIR = INSTALL_DIR / "config"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "settings.json"
DEFAULT_MODELS_DIR = str(INSTALL_DIR / "models")
DEFAULT_LOG_DIR = INSTALL_DIR / "logs"
DEFAULT_TEMP_DIR = INSTALL_DIR / "temp"


@dataclass
class Settings:
    """All user-configurable settings with sensible defaults."""

    # ── Model Engine ──────────────────────────────────────────────────────────
    engine: str = "whisper"
    model_path: str = DEFAULT_MODELS_DIR
    device: str = "cuda"
    language: str = "en"
    inference_timeout: int = 30

    # ── Dictation UX ─────────────────────────────────────────────────────────
    auto_copy: bool = True
    auto_paste: bool = True
    hotkeys_enabled: bool = True
    hotkey_start: str = "ctrl+alt+p"
    hotkey_stop: str = "ctrl+alt+l"
    hotkey_quit: str = "ctrl+alt+q"
    clear_logs_on_exit: bool = False

    # ── Audio ─────────────────────────────────────────────────────────────────
    mic_device_index: int = -1           # -1 = system default
    sample_rate: int = 16000             # recording only — always resampled to 16 kHz for engines
    silence_threshold: float = 0.0015
    silence_margin_ms: int = 500

    # ── Helpers ───────────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> None:
        """Persist settings to JSON file."""
        path = path or DEFAULT_CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        log.info("Settings saved to %s", path)

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        """Load settings from JSON, falling back to defaults for missing keys."""
        path = path or DEFAULT_CONFIG_FILE
        if not path.exists():
            log.info("No settings file found; using defaults")
            return cls()
        try:
            with open(path, encoding="utf-8-sig") as fh:
                data = json.load(fh)
            known = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in known})
        except Exception:
            log.warning("Failed to load settings; using defaults", exc_info=True)
            return cls()
