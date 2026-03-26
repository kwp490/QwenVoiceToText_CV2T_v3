#!/usr/bin/env python3
"""
Model downloader for CV2T.

Standalone script that can also be invoked via `cv2t download-model`.

Usage:
    python download_model.py --engine whisper --target-dir "C:\Program Files\CV2T\models"
    python download_model.py --engine canary --target-dir "C:\Program Files\CV2T\models"
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Download local AI models for CV2T.")
    parser.add_argument(
        "--engine",
        choices=["canary", "whisper"],
        required=True,
        help="Engine whose model to download",
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="Directory to store models (default: C:\\Program Files\\CV2T\\models)",
    )
    args = parser.parse_args()

    default_dir = os.path.join(
        os.environ.get("CV2T_HOME", r"C:\Program Files\CV2T"),
        "models",
    )
    target_dir = args.target_dir or default_dir
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


def _whisper_model_ready(target_dir: str) -> bool:
    """Return True if the whisper model files already exist locally."""
    return all(
        os.path.isfile(os.path.join(target_dir, f))
        for f in _WHISPER_REQUIRED_FILES
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


if __name__ == "__main__":
    sys.exit(main())