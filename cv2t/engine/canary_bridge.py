"""
Canary subprocess bridge engine.

Wraps the standalone canary_worker.py as a SpeechEngine, allowing the
frozen PyInstaller binary to use Canary inference via a separate Python
environment that has torch/NeMo installed (canary-env).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional

import numpy as np
import soundfile as sf

from .base import SpeechEngine

log = logging.getLogger(__name__)


def _get_app_dir() -> str:
    """Return the application root directory."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    # Source install — repo root is two levels up from cv2t/engine/
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_canary_paths() -> tuple[Optional[str], Optional[str]]:
    """Discover canary-env Python and worker script paths.

    Returns ``(python_path, worker_path)`` or ``(None, None)`` if not found.
    """
    app_dir = _get_app_dir()

    python_path = os.path.join(app_dir, "canary-env", ".venv", "Scripts", "python.exe")

    # PyInstaller 6+ places data files in _internal/ (sys._MEIPASS),
    # not beside the exe.
    data_dir = getattr(sys, "_MEIPASS", app_dir)
    worker_path = os.path.join(data_dir, "canary_worker.py")
    if not os.path.isfile(worker_path):
        # Fallback: beside the exe (legacy layout or dev)
        worker_path = os.path.join(app_dir, "canary_worker.py")

    if os.path.isfile(python_path) and os.path.isfile(worker_path):
        return python_path, worker_path
    return None, None


def canary_env_available() -> bool:
    """Return True if the canary subprocess environment is installed."""
    python_path, worker_path = _get_canary_paths()
    return python_path is not None


def get_app_dir() -> str:
    """Public accessor for the application root directory."""
    return _get_app_dir()


class CanaryBridgeEngine(SpeechEngine):
    """Canary engine that delegates to a subprocess with torch/NeMo.

    The subprocess runs ``canary_worker.py`` using the Python from
    ``canary-env/.venv/`` and communicates via JSON lines over
    stdin/stdout.
    """

    def __init__(self) -> None:
        super().__init__()
        self._process: Optional[subprocess.Popen] = None
        self._python_path: Optional[str] = None
        self._worker_path: Optional[str] = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "canary"

    @property
    def vram_estimate_gb(self) -> float:
        return 5.0

    def _send(self, obj: dict) -> None:
        """Write a JSON line to the worker's stdin."""
        assert self._process is not None and self._process.stdin is not None
        line = json.dumps(obj) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

    def _drain_stderr(self) -> None:
        """Continuously read stderr in a background thread.

        Without this, the worker can deadlock when its stderr pipe buffer
        fills (64 KB on Windows) during model load — the bridge is blocked
        on stdout.readline() while the worker is blocked on a stderr write.
        """
        assert self._process is not None and self._process.stderr is not None
        try:
            for line in self._process.stderr:
                stripped = line.rstrip()
                if stripped:
                    self._stderr_lines.append(stripped)
                    log.debug("[canary-worker] %s", stripped)
        except ValueError:
            pass  # stderr closed

    def _recv(self, timeout: float | None = None) -> dict:
        """Read a JSON line from the worker's stdout.

        If *timeout* is given (seconds), raises ``TimeoutError`` when
        no complete line arrives before the deadline.
        """
        assert self._process is not None and self._process.stdout is not None

        if timeout is not None:
            # Windows selectors don't support pipe file-objects, so use a
            # background thread to perform the blocking readline.
            result_box: list[str | None] = [None]

            def _read() -> None:
                try:
                    result_box[0] = self._process.stdout.readline()
                except Exception:
                    result_box[0] = None

            reader = threading.Thread(target=_read, daemon=True,
                                      name="canary-recv")
            reader.start()
            reader.join(timeout=timeout)

            if reader.is_alive():
                # readline still blocking — timeout expired
                # Kill the subprocess so the thread unblocks.
                try:
                    self._process.kill()
                except Exception:
                    pass
                reader.join(timeout=5)
                stderr_text = "\n".join(self._stderr_lines[-50:])
                detail = ""
                if stderr_text:
                    log.error("Canary worker stderr:\n%s", stderr_text.strip())
                    detail = f"\n{stderr_text.strip()}"
                raise TimeoutError(
                    f"Canary worker did not respond within {timeout:.0f}s"
                    f"{detail}"
                )

            line = result_box[0]
            if not line:
                # Worker died — fall through to error handling below
                pass
            else:
                return json.loads(line.strip())

        else:
            line = self._process.stdout.readline()

        if not line:
            # Worker died — collect stderr from the drain thread
            stderr_text = "\n".join(self._stderr_lines[-50:])
            rc = self._process.poll()
            detail = f" (exit code {rc})" if rc is not None else ""
            if stderr_text:
                log.error("Canary worker stderr:\n%s", stderr_text.strip())
                detail += f"\n{stderr_text.strip()}"
            raise RuntimeError(
                f"Canary worker process terminated unexpectedly{detail}"
            )
        return json.loads(line.strip())

    def load(self, model_path: str, device: str = "cuda") -> None:
        """Start the canary worker subprocess and load the model."""
        self._python_path, self._worker_path = _get_canary_paths()
        if self._python_path is None:
            raise RuntimeError(
                "Canary environment not found. "
                "Install it via Settings \u2192 Install Canary Engine."
            )

        log.info(
            "Starting Canary worker: %s %s",
            self._python_path, self._worker_path,
        )

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        self._process = subprocess.Popen(
            [self._python_path, self._worker_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )

        # Drain stderr in a background thread to prevent pipe-buffer
        # deadlocks (the worker redirects stdout→stderr, so *all* log
        # output from NeMo/torch goes there).
        self._stderr_lines = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name="canary-stderr",
        )
        self._stderr_thread.start()

        # Send load command
        self._send({
            "command": "load",
            "model_path": model_path,
            "device": device,
        })

        # Wait for model to load (can take 30-60 s on first run, up to
        # several minutes if NeMo compiles kernels).  Cap at 5 minutes
        # so the UI doesn't hang indefinitely.
        response = self._recv(timeout=300)
        if response.get("status") != "ready":
            error_msg = response.get("message", "Unknown error")
            self._cleanup_process()
            raise RuntimeError(f"Canary model load failed: {error_msg}")

        # Mark as loaded (base class protocol)
        self._model = "subprocess"
        log.info("Canary worker ready (subprocess bridge)")

    def _transcribe_impl(self, audio_16k: np.ndarray, language: str = "en") -> str:
        """Save audio to temp file and send to worker for transcription."""
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Canary worker process is not running")

        # Write audio to temp WAV file
        tmp_dir = os.path.join(tempfile.gettempdir(), "cv2t")
        os.makedirs(tmp_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
        os.close(fd)

        try:
            sf.write(tmp_path, audio_16k, 16000)

            self._send({
                "command": "transcribe",
                "audio_file": tmp_path,
                "language": language,
            })

            response = self._recv()
            if response.get("status") != "ok":
                error_msg = response.get("message", "Unknown error")
                raise RuntimeError(f"Canary transcription failed: {error_msg}")

            return response.get("text", "")

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def unload(self) -> None:
        """Shut down the canary worker subprocess."""
        if self._process is not None and self._process.poll() is None:
            try:
                self._send({"command": "shutdown"})
                self._process.wait(timeout=10)
            except Exception:
                self._process.kill()
        self._cleanup_process()
        self._model = None
        log.info("Canary bridge engine unloaded")

    def _cleanup_process(self) -> None:
        """Clean up subprocess resources."""
        if self._process is not None:
            try:
                if self._process.poll() is None:
                    self._process.kill()
                if self._process.stdin:
                    self._process.stdin.close()
                if self._process.stdout:
                    self._process.stdout.close()
                if self._process.stderr:
                    self._process.stderr.close()
            except Exception:
                pass
            self._process = None
