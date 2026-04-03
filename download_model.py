#!/usr/bin/env python3
"""
Model downloader for CV2T.

Standalone script that can also be invoked via ``cv2t download-model``.

Usage:
    python download_model.py --engine whisper --target-dir "C:\\Program Files\\CV2T\\models"
    python download_model.py --engine canary --target-dir "C:\\Program Files\\CV2T\\models"
"""

from __future__ import annotations

import argparse
import os
import sys



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
        from cv2t.model_downloader import download_whisper_model
        return download_whisper_model(engine_dir)
    elif args.engine == "canary":
        engine_dir = os.path.join(target_dir, "canary")
        os.makedirs(engine_dir, exist_ok=True)
        return _download_canary(engine_dir)
    return 1


def _download_canary(target_dir: str) -> int:
    """Download Canary NeMo SALM model via huggingface_hub."""
    import importlib
    try:
        hf_hub = importlib.import_module("huggingface_hub")
    except ImportError:
        print("ERROR: Canary model download requires huggingface-hub.")
        print("Install it: pip install huggingface-hub")
        return 1
    print(f"Downloading Canary model from nvidia/canary-qwen-2.5b to {target_dir}...")
    try:
        hf_hub.snapshot_download(
            repo_id="nvidia/canary-qwen-2.5b",
            local_dir=target_dir,
            local_files_only=False,
        )
        print("Download complete.")
        return 0
    except Exception as exc:
        print(f"ERROR: Download failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())