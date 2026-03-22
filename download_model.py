#!/usr/bin/env python3
"""
Model downloader for CV2T.

Standalone script that can also be invoked via `cv2t download-model`.

Usage:
    python download_model.py --engine whisper --target-dir "%LOCALAPPDATA%\\CV2T\\models"
    python download_model.py --engine canary --target-dir "%LOCALAPPDATA%\\CV2T\\models" --hf-token TOKEN
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CV2T model weights")
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
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token for gated models (Canary)",
    )
    args = parser.parse_args()

    default_dir = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
        "CV2T", "models",
    )
    target_dir = args.target_dir or default_dir
    os.makedirs(target_dir, exist_ok=True)

    if args.engine == "whisper":
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("ERROR: faster-whisper not installed. Install with: uv sync --extra whisper")
            return 1

        print(f"Downloading Whisper large-v3-turbo to {target_dir}…")
        try:
            WhisperModel(
                "large-v3-turbo",
                device="cpu",
                compute_type="int8",
                download_root=target_dir,
            )
            print("Whisper model downloaded successfully.")
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}")
            return 1

    elif args.engine == "canary":
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("ERROR: huggingface_hub not installed. Install with: uv sync --extra canary")
            return 1

        repo_id = "nvidia/canary-qwen-2.5b"
        token = args.hf_token or os.environ.get("HF_TOKEN")
        print(f"Downloading {repo_id} to {target_dir}…")
        if not token:
            print(
                "NOTE: nvidia/canary-qwen-2.5b is a gated model.\n"
                "If download fails with 401/403, you must either:\n"
                "  1. Run: huggingface-cli login\n"
                "  2. Set HF_TOKEN environment variable\n"
                "  3. Pass --hf-token <token>\n"
            )
        try:
            snapshot_download(
                repo_id,
                local_dir=os.path.join(target_dir, "canary-qwen-2.5b"),
                token=token,
            )
            print("Canary model downloaded successfully.")
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}")
            return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
