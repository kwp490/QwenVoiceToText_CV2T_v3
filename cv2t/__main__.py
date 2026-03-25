"""
CV2T — entry point.

Usage:
    python -m cv2t                                          # launch GUI
    python -m cv2t download-model --engine whisper          # download model
    python -m cv2t --version                                # print version

Handles single-instance guard, logging setup, and Qt application lifecycle.
"""

from __future__ import annotations

import argparse
import ctypes
import faulthandler
import io
import logging
import logging.handlers
import os
import sys


_WHISPER_MODEL_ID = "large-v3-turbo"
_WHISPER_REPO_ID = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
_WHISPER_REQUIRED_FILES = ("config.json", "model.bin", "tokenizer.json")
_WHISPER_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]

# ── Stdout/stderr safety (needed for PyInstaller --noconsole builds) ─────────
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()


# ── Single-instance mutex (Windows) ──────────────────────────────────────────

_MUTEX_NAME = "Global\\CV2TMutex"
_mutex_handle = None


def release_single_instance_mutex() -> None:
    """Release the single-instance mutex so a restarted process can acquire it."""
    global _mutex_handle
    if _mutex_handle is not None:
        try:
            ctypes.windll.kernel32.CloseHandle(_mutex_handle)  # type: ignore[attr-defined]
        except Exception:
            pass
        _mutex_handle = None


def _ensure_single_instance() -> bool:
    """Return True if this is the only running instance."""
    global _mutex_handle
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        _mutex_handle = kernel32.CreateMutexW(None, True, _MUTEX_NAME)
        if kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            kernel32.CloseHandle(_mutex_handle)
            _mutex_handle = None
            return False
        return True
    except Exception:
        # Non-Windows or ctypes not available — skip guard
        return True


# ── Logging ──────────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    from .config import DEFAULT_LOG_DIR

    log_dir = str(DEFAULT_LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "cv2t.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.handlers.RotatingFileHandler(
                log_path, maxBytes=2 * 1024 * 1024, backupCount=2, encoding="utf-8"
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("cv2t").info("=== CV2T starting (log: %s) ===", log_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cv2t",
        description="CV2T — Native Windows Voice-to-Text",
    )
    parser.add_argument(
        "--version", action="store_true", help="Print version and exit"
    )
    parser.add_argument(
        "--skip-engine-prompt",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,  # internal: skip engine selection after settings-driven restart
    )

    sub = parser.add_subparsers(dest="command")

    dl = sub.add_parser("download-model", help="Download model weights")
    dl.add_argument(
        "--engine",
        choices=["canary", "whisper"],
        required=True,
        help="Engine whose model to download",
    )
    dl.add_argument(
        "--target-dir",
        default=None,
        help="Directory to store models (default: C:\\Program Files\\CV2T\\models)",
    )

    return parser


def _cmd_download_model(args: argparse.Namespace) -> int:
    """Handle the download-model subcommand."""
    from .config import DEFAULT_MODELS_DIR

    target_dir = args.target_dir or DEFAULT_MODELS_DIR
    os.makedirs(target_dir, exist_ok=True)

    if args.engine == "whisper":
        engine_dir = os.path.join(target_dir, "whisper")
        os.makedirs(engine_dir, exist_ok=True)
        return _download_whisper(engine_dir)
    elif args.engine == "canary":
        engine_dir = os.path.join(target_dir, "canary")
        os.makedirs(engine_dir, exist_ok=True)
        return _download_canary(engine_dir)
    return 1


def _is_whisper_model(model_dir: str) -> bool:
    """Return True if config.json in *model_dir* describes a Whisper model."""
    import json as _json
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = _json.load(f)
        archs = cfg.get("architectures", [])
        if archs and not any("whisper" in a.lower() for a in archs):
            return False
        model_type = cfg.get("model_type", "")
        if model_type and model_type.lower() not in ("whisper", ""):
            return False
        return True
    except (ValueError, OSError):
        return False


def _whisper_model_ready(target_dir: str) -> bool:
    """Return True if the whisper model files already exist locally."""
    return (
        all(os.path.isfile(os.path.join(target_dir, f)) for f in _WHISPER_REQUIRED_FILES)
        and _is_whisper_model(target_dir)
    )


def _print_whisper_runtime_diagnostics() -> None:
    """Print runtime details for the active faster-whisper installation."""
    try:
        import faster_whisper
        import faster_whisper.utils as faster_whisper_utils

        models = getattr(faster_whisper_utils, "_MODELS", None)
        resolved_repo = None
        if isinstance(models, dict):
            resolved_repo = models.get(_WHISPER_MODEL_ID)

        print(f"faster_whisper.__file__ = {getattr(faster_whisper, '__file__', '<unknown>')}")
        print(
            "faster_whisper.utils.__file__ = "
            f"{getattr(faster_whisper_utils, '__file__', '<unknown>')}"
        )
        print(f"_MODELS['{_WHISPER_MODEL_ID}'] = {resolved_repo!r}")
    except Exception as exc:
        print(f"WARNING: Unable to inspect faster-whisper runtime diagnostics: {exc}")


def _download_whisper(target_dir: str) -> int:
    if _whisper_model_ready(target_dir):
        print(f"Whisper model already present in {target_dir} — skipping download.")
        return 0
    _print_whisper_runtime_diagnostics()
    return _download_model("Whisper", _WHISPER_REPO_ID, target_dir,
                           allow_patterns=_WHISPER_ALLOW_PATTERNS)


def _download_canary(target_dir: str) -> int:
    """Download Canary NeMo SALM model via huggingface_hub."""
    return _download_model("Canary", "nvidia/canary-qwen-2.5b", target_dir)


def _download_model(
    engine_label: str,
    repo_id: str,
    target_dir: str,
    *,
    allow_patterns: list | None = None,
) -> int:
    """Generic HuggingFace model download with standard error handling."""
    print(f"Downloading {engine_label} model from {repo_id} to {target_dir}...")
    try:
        from huggingface_hub import snapshot_download

        kwargs: dict = {
            "repo_id": repo_id,
            "local_dir": target_dir,
            "local_files_only": False,
        }
        if allow_patterns:
            kwargs["allow_patterns"] = allow_patterns
        snapshot_download(**kwargs)
        print("Download complete.")
        return 0
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "Repository Not Found" in msg:
            print(f"ERROR: Repo '{repo_id}' not found or access denied: {exc}")
        else:
            print(f"ERROR: Download failed: {exc}")
        return 1


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.version:
        from . import __version__
        print(f"CV2T {__version__}")
        return 0

    if args.command == "download-model":
        _setup_logging()
        return _cmd_download_model(args)

    # Default: launch GUI
    faulthandler.enable()

    if not _ensure_single_instance():
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            _app = QApplication(sys.argv)
            QMessageBox.warning(None, "CV2T", "Another instance is already running.")
        except Exception:
            print("ERROR: Another instance of CV2T is already running.")
        return 1

    _setup_logging()

    from PySide6.QtWidgets import QApplication
    from .config import Settings
    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("CV2T")
    app.setOrganizationName("CV2T")

    settings = Settings.load()

    # If more than one engine + model is installed, let the user choose
    # which engine to load before the main window opens.
    # Skip when restarting after an engine change in settings (the user
    # already made their choice and it is persisted).
    if not args.skip_engine_prompt:
        from .engine import get_available_engines
        available = get_available_engines(settings.model_path)
        if len(available) > 1:
            chosen = _ask_engine_selection(available, settings.engine)
            if chosen is None:
                # User closed the dialog — exit gracefully
                return 0
            if chosen != settings.engine:
                settings.engine = chosen
                settings.save()

    window = MainWindow(settings)
    window.show()

    return app.exec()


def _ask_engine_selection(available: list, current_default: str) -> str | None:
    """Show a dialog asking the user which engine to load.

    Returns the chosen engine name, or *None* if the dialog was dismissed.
    """
    from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QDialogButtonBox
    from PySide6.QtCore import Qt

    _ENGINE_LABELS = {
        "whisper": "Whisper  (CTranslate2 — ~3 GB VRAM, faster startup)",
        "canary": "Canary  (NeMo/PyTorch — ~5 GB VRAM, higher accuracy)",
    }

    dlg = QDialog()
    dlg.setWindowTitle("CV2T — Select Engine")
    dlg.setMinimumWidth(420)
    dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

    layout = QVBoxLayout(dlg)
    layout.addWidget(QLabel(
        "Multiple speech engines are installed.\n"
        "Choose which engine to load:"
    ))

    combo = QComboBox()
    for name in available:
        label = _ENGINE_LABELS.get(name) or name
        combo.addItem(label, userData=name)
    # Pre-select the current default if it's in the list
    default_idx = combo.findData(current_default)
    if default_idx >= 0:
        combo.setCurrentIndex(default_idx)

    layout.addWidget(combo)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    if dlg.exec() == QDialog.DialogCode.Accepted:
        return combo.currentData()
    return None


if __name__ == "__main__":
    sys.exit(main())
