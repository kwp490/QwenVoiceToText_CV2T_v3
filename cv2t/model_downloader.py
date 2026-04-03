"""
Lightweight model downloader using only Python stdlib (urllib).

Replaces huggingface_hub for Whisper model downloads in the frozen binary,
eliminating ~400 MB of transitive dependencies from the installer.

Canary model downloads use huggingface_hub from within the canary-env
Python environment (installed separately via Enable-Canary.ps1).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

_HF_CDN = "https://huggingface.co"

# ── Whisper model constants (single source of truth) ─────────────────────────

WHISPER_REPO_ID = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
WHISPER_MODEL_ID = "large-v3-turbo"
WHISPER_REQUIRED_FILES = ("config.json", "model.bin", "tokenizer.json")
WHISPER_OPTIONAL_FILES = (
    "preprocessor_config.json",
    "vocabulary.json",
    "vocabulary.txt",
)


def _hf_file_url(repo_id: str, filename: str, revision: str = "main") -> str:
    """Build a HuggingFace CDN URL for a single file."""
    return f"{_HF_CDN}/{repo_id}/resolve/{revision}/{filename}"


def download_file(url: str, dest_path: str, *, label: str = "") -> bool:
    """Download a single file from *url* to *dest_path*.

    Shows progress on stdout.  Returns True on success, False on HTTP error.
    """
    display = label or os.path.basename(dest_path)

    # Skip if file already exists and has nonzero size
    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
        log.info("  %s — already exists, skipping", display)
        return True

    tmp_path = dest_path + ".tmp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cv2t/3.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            total = int(resp.headers.get("Content-Length", 0))
            total_mb = total / (1024 * 1024) if total else 0

            downloaded = 0
            buf_size = 256 * 1024  # 256 KB chunks

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(buf_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0 and sys.stdout and hasattr(sys.stdout, "write"):
                        pct = downloaded * 100 / total
                        dl_mb = downloaded / (1024 * 1024)
                        try:
                            sys.stdout.write(
                                f"\r  {display}: {dl_mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)"
                            )
                            sys.stdout.flush()
                        except (OSError, ValueError):
                            pass  # stdout may be unavailable in frozen builds

            try:
                sys.stdout.write(f"\r  {display}: {total_mb:.1f} MB — done\n")
                sys.stdout.flush()
            except (OSError, ValueError):
                pass

        os.replace(tmp_path, dest_path)
        return True

    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            log.debug("  %s — not found (404), skipping", display)
            return False
        log.error("  %s — HTTP %d: %s", display, exc.code, exc.reason)
        return False
    except Exception as exc:
        log.error("  %s — download failed: %s", display, exc)
        return False
    finally:
        try:
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass


def download_whisper_model(target_dir: str) -> int:
    """Download all Whisper model files to *target_dir*.

    Returns 0 on success, 1 on failure.
    """
    os.makedirs(target_dir, exist_ok=True)

    if whisper_model_ready(target_dir):
        print(f"Whisper model already present in {target_dir} — skipping download.")
        return 0

    print(f"Downloading Whisper model from {WHISPER_REPO_ID}...")

    # Download required files
    for filename in WHISPER_REQUIRED_FILES:
        url = _hf_file_url(WHISPER_REPO_ID, filename)
        dest = os.path.join(target_dir, filename)
        if not download_file(url, dest, label=filename):
            print(f"ERROR: Failed to download required file: {filename}")
            return 1

    # Download optional files (ignore failures)
    for filename in WHISPER_OPTIONAL_FILES:
        url = _hf_file_url(WHISPER_REPO_ID, filename)
        dest = os.path.join(target_dir, filename)
        download_file(url, dest, label=f"{filename} (optional)")

    if whisper_model_ready(target_dir):
        print("Whisper model download complete.")
        return 0
    else:
        print("ERROR: Model files are incomplete after download.")
        return 1


def is_whisper_model(model_dir: str) -> bool:
    """Return True if config.json in *model_dir* describes a Whisper model."""
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        archs = cfg.get("architectures", [])
        if archs and not any("whisper" in a.lower() for a in archs):
            return False
        model_type = cfg.get("model_type", "")
        if model_type and model_type.lower() not in ("whisper", ""):
            return False
        return True
    except (json.JSONDecodeError, OSError):
        return False


def whisper_model_ready(model_dir: str) -> bool:
    """Return True if all required Whisper model files exist."""
    return all(
        os.path.isfile(os.path.join(model_dir, f)) for f in WHISPER_REQUIRED_FILES
    ) and is_whisper_model(model_dir)
