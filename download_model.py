#!/usr/bin/env python3
"""
Model downloader for CV2T.

Standalone script that can also be invoked via `cv2t download-model`.

Usage:
    python download_model.py --engine whisper --target-dir "%LOCALAPPDATA%\\CV2T\\models"
    python download_model.py --engine canary --target-dir "%LOCALAPPDATA%\\CV2T\\models"
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
        help="Directory to store models (default: %%LOCALAPPDATA%%\\CV2T\\models)",
    )
    args = parser.parse_args()

    default_dir = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
        "CV2T", "models",
    )
    target_dir = args.target_dir or default_dir
    os.makedirs(target_dir, exist_ok=True)

    from huggingface_hub import snapshot_download

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


if __name__ == "__main__":
    sys.exit(main())