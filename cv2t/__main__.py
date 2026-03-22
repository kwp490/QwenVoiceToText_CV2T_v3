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
import io
import logging
import logging.handlers
import os
import sys

# ── Stdout/stderr safety (needed for PyInstaller --noconsole builds) ─────────
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()


# ── Single-instance mutex (Windows) ──────────────────────────────────────────

_MUTEX_NAME = "Global\\CV2TMutex"
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
    log_dir = os.path.join(
        os.environ.get("APPDATA", os.path.expanduser("~")),
        "CV2T",
    )
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
        help="Directory to store models (default: %%LOCALAPPDATA%%\\CV2T\\models)",
    )

    return parser


def _cmd_download_model(args: argparse.Namespace) -> int:
    """Handle the download-model subcommand."""
    from .config import DEFAULT_MODELS_DIR
    from huggingface_hub import snapshot_download

    target_dir = args.target_dir or DEFAULT_MODELS_DIR
    os.makedirs(target_dir, exist_ok=True)

    if args.engine == "canary":
        repo_id = "onnx-community/canary-qwen-2.5b-ONNX"
    elif args.engine == "whisper":
        repo_id = "Systran/faster-whisper-large-v3-turbo"
    else:
        return 1

    print(f"Downloading {repo_id} to {target_dir}...")
    try:
        snapshot_download(repo_id=repo_id, local_dir=target_dir)
        print("Download complete.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
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
    window = MainWindow(settings)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
