"""
Settings dialog for CV2T.

Provides a form to edit engine, model path, microphone, toggles, etc.
Changes are written back to the ``Settings`` dataclass on accept.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .audio import AudioRecorder
from .config import Settings
from .engine import ENGINES, get_available_engines

log = logging.getLogger(__name__)

# Engines that require CUDA and cannot run on CPU.
_CUDA_ONLY_ENGINES = frozenset({"canary"})


class SettingsDialog(QDialog):
    """Modal dialog for editing application settings."""

    def __init__(
        self,
        settings: Settings,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        self._build_ui()
        self._populate()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Model Engine group ───────────────────────────────────────────────
        engine_group = QGroupBox("Model Engine")
        engine_form = QFormLayout()

        self._engine_combo = QComboBox()
        for name in ENGINES:
            self._engine_combo.addItem(name)
        # Disable engine selection when only one engine has model files installed
        available = get_available_engines(self.settings.model_path)
        if len(available) <= 1:
            self._engine_combo.setEnabled(False)
            self._engine_combo.setToolTip(
                "Only one engine is installed. Install both engines to switch."
            )
        engine_form.addRow("Engine:", self._engine_combo)

        # ── Canary add-on status / install button ────────────────────────────
        canary_row = QHBoxLayout()
        self._canary_status_label = QLabel()
        canary_row.addWidget(self._canary_status_label)
        self._btn_install_canary = QPushButton("Install Canary Engine")
        self._btn_install_canary.setToolTip(
            "Install PyTorch + NeMo into a separate environment.\n"
            "Requires ~10 GB disk space and an internet connection."
        )
        self._btn_install_canary.clicked.connect(self._on_install_canary)
        canary_row.addWidget(self._btn_install_canary)

        self._btn_validate_canary = QPushButton("Validate Canary Setup")
        self._btn_validate_canary.setToolTip(
            "Check that the canary-env Python, all dependencies,\n"
            "CUDA support, and model files are correctly installed."
        )
        self._btn_validate_canary.clicked.connect(self._on_validate_canary)
        canary_row.addWidget(self._btn_validate_canary)

        self._lbl_canary_validate = QLabel("")
        canary_row.addWidget(self._lbl_canary_validate)

        canary_row.addStretch()
        engine_form.addRow("Canary engine:", canary_row)
        self._update_canary_status()

        model_row = QHBoxLayout()
        self._model_path = QLineEdit()
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_model_path)
        model_row.addWidget(self._model_path)
        model_row.addWidget(btn_browse)
        engine_form.addRow("Model path:", model_row)

        self._device_combo = QComboBox()
        self._device_combo.addItems(["cuda", "cpu"])
        engine_form.addRow("Device:", self._device_combo)

        # Lock device to CUDA when a CUDA-only engine is selected
        self._engine_combo.currentTextChanged.connect(self._on_engine_changed)

        self._language = QLineEdit()
        self._language.setMaximumWidth(100)
        engine_form.addRow("Language:", self._language)

        self._inference_timeout = QSpinBox()
        self._inference_timeout.setRange(5, 300)
        self._inference_timeout.setSuffix(" s")
        engine_form.addRow("Inference timeout:", self._inference_timeout)

        engine_group.setLayout(engine_form)
        layout.addWidget(engine_group)

        # ── Audio group ──────────────────────────────────────────────────────
        audio_group = QGroupBox("Audio")
        audio_form = QFormLayout()

        self._mic_combo = QComboBox()
        self._mic_combo.addItem("System default", -1)
        try:
            for idx, name in AudioRecorder.list_input_devices():
                self._mic_combo.addItem(f"[{idx}] {name}", idx)
        except Exception:
            log.warning("Could not enumerate audio devices", exc_info=True)
        audio_form.addRow("Microphone:", self._mic_combo)

        self._silence_threshold = QDoubleSpinBox()
        self._silence_threshold.setRange(0.0001, 0.1)
        self._silence_threshold.setDecimals(4)
        self._silence_threshold.setSingleStep(0.0005)
        audio_form.addRow("Silence threshold (RMS):", self._silence_threshold)

        self._silence_margin = QSpinBox()
        self._silence_margin.setRange(50, 1000)
        self._silence_margin.setSuffix(" ms")
        audio_form.addRow("Silence margin:", self._silence_margin)

        self._sample_rate = QSpinBox()
        self._sample_rate.setRange(8000, 48000)
        self._sample_rate.setSingleStep(8000)
        self._sample_rate.setSuffix(" Hz")
        audio_form.addRow("Sample rate (recording):", self._sample_rate)

        audio_group.setLayout(audio_form)
        layout.addWidget(audio_group)

        # ── Dictation UX group ───────────────────────────────────────────────
        ux_group = QGroupBox("Dictation UX")
        ux_form = QFormLayout()

        self._auto_copy = QCheckBox("Auto-copy transcription to clipboard")
        ux_form.addRow(self._auto_copy)

        self._auto_paste = QCheckBox("Auto-paste (Ctrl+V) after copy")
        ux_form.addRow(self._auto_paste)

        self._hotkeys_enabled = QCheckBox("Enable global hotkeys")
        ux_form.addRow(self._hotkeys_enabled)

        self._hotkey_start = QLineEdit()
        ux_form.addRow("Start recording hotkey:", self._hotkey_start)

        self._hotkey_stop = QLineEdit()
        ux_form.addRow("Stop/transcribe hotkey:", self._hotkey_stop)

        self._hotkey_quit = QLineEdit()
        ux_form.addRow("Quit hotkey:", self._hotkey_quit)

        self._clear_logs_on_exit = QCheckBox("Clear logs on application exit")
        ux_form.addRow(self._clear_logs_on_exit)

        ux_group.setLayout(ux_form)
        layout.addWidget(ux_group)

        # ── Button box ───────────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ── Populate / Save ──────────────────────────────────────────────────────

    def _populate(self) -> None:
        s = self.settings
        idx = self._engine_combo.findText(s.engine)
        if idx >= 0:
            self._engine_combo.setCurrentIndex(idx)
        self._model_path.setText(s.model_path)
        idx = self._device_combo.findText(s.device)
        if idx >= 0:
            self._device_combo.setCurrentIndex(idx)
        self._language.setText(s.language)
        self._inference_timeout.setValue(s.inference_timeout)
        self._silence_threshold.setValue(s.silence_threshold)
        self._silence_margin.setValue(s.silence_margin_ms)
        self._sample_rate.setValue(s.sample_rate)
        self._auto_copy.setChecked(s.auto_copy)
        self._auto_paste.setChecked(s.auto_paste)
        self._hotkeys_enabled.setChecked(s.hotkeys_enabled)
        self._hotkey_start.setText(s.hotkey_start)
        self._hotkey_stop.setText(s.hotkey_stop)
        self._hotkey_quit.setText(s.hotkey_quit)
        self._clear_logs_on_exit.setChecked(s.clear_logs_on_exit)

        # Select current mic device
        idx = self._mic_combo.findData(s.mic_device_index)
        if idx >= 0:
            self._mic_combo.setCurrentIndex(idx)

        # Apply engine-specific device constraints
        self._on_engine_changed(self._engine_combo.currentText())

    def _save_and_accept(self) -> None:
        s = self.settings
        s.engine = self._engine_combo.currentText()
        s.model_path = self._model_path.text().strip()
        s.device = self._device_combo.currentText()
        s.language = self._language.text().strip() or "en"
        s.inference_timeout = self._inference_timeout.value()
        s.silence_threshold = self._silence_threshold.value()
        s.silence_margin_ms = self._silence_margin.value()
        s.sample_rate = self._sample_rate.value()
        s.auto_copy = self._auto_copy.isChecked()
        s.auto_paste = self._auto_paste.isChecked()
        s.hotkeys_enabled = self._hotkeys_enabled.isChecked()
        s.hotkey_start = self._hotkey_start.text().strip() or "ctrl+alt+p"
        s.hotkey_stop = self._hotkey_stop.text().strip() or "ctrl+alt+l"
        s.hotkey_quit = self._hotkey_quit.text().strip() or "ctrl+alt+q"
        s.clear_logs_on_exit = self._clear_logs_on_exit.isChecked()
        s.mic_device_index = self._mic_combo.currentData()

        s.save()
        log.info("Settings saved")
        self.accept()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _on_engine_changed(self, engine_name: str) -> None:
        """Enable or disable the device combo based on engine CPU support."""
        if engine_name in _CUDA_ONLY_ENGINES:
            self._device_combo.setCurrentText("cuda")
            self._device_combo.setEnabled(False)
            self._device_combo.setToolTip(
                f"The {engine_name} engine requires CUDA and cannot run on CPU."
            )
        else:
            self._device_combo.setEnabled(True)
            self._device_combo.setToolTip("")

    def _browse_model_path(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Model Directory",
            self._model_path.text(),
        )
        if path:
            self._model_path.setText(path)

    # ── Canary add-on helpers ───────────────────────────────────────────────

    def _update_canary_status(self) -> None:
        """Refresh the Canary status label and install/validate button state."""
        installed = False

        # Check the ENGINES registry (populated once at import time via real
        # ``import torch``, which is reliable in both source and frozen builds).
        if "canary" in ENGINES:
            if "canary_bridge" in ENGINES["canary"].__module__:
                self._canary_status_label.setText("\u2705 Installed (canary-env)")
            else:
                self._canary_status_label.setText("\u2705 Installed (native)")
            self._canary_status_label.setStyleSheet("color: #2e7d32;")
            installed = True
        else:
            # Canary-env may have been installed after app startup — check live.
            try:
                from cv2t.engine.canary_bridge import canary_env_available
                if canary_env_available():
                    self._canary_status_label.setText(
                        "\u2705 Installed (canary-env) \u2014 restart to activate")
                    self._canary_status_label.setStyleSheet("color: #2e7d32;")
                    installed = True
            except ImportError:
                pass

        if installed:
            self._btn_install_canary.setVisible(False)
            self._btn_validate_canary.setVisible(True)
        else:
            self._canary_status_label.setText("Not installed")
            self._canary_status_label.setStyleSheet("color: #757575;")
            self._btn_install_canary.setVisible(True)
            self._btn_validate_canary.setVisible(False)
            self._lbl_canary_validate.setText("")

    def _on_install_canary(self, *, skip_confirm: bool = False) -> None:
        """Launch Enable-Canary.ps1 in an elevated terminal."""
        import os
        import subprocess
        import sys

        # Find Enable-Canary.ps1
        if getattr(sys, "frozen", False):
            app_dir = os.path.dirname(sys.executable)
            # PyInstaller 6+ places data files in _internal/ (sys._MEIPASS)
            data_dir = getattr(sys, "_MEIPASS", app_dir)
        else:
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = app_dir

        script_path = os.path.join(app_dir, "Enable-Canary.ps1")
        if not os.path.isfile(script_path):
            # PyInstaller 6+ _internal/ directory
            script_path = os.path.join(data_dir, "Enable-Canary.ps1")
        if not os.path.isfile(script_path):
            # Check installer directory (dev/source layout)
            script_path = os.path.join(app_dir, "installer", "Enable-Canary.ps1")

        if not os.path.isfile(script_path):
            QMessageBox.warning(
                self,
                "Enable-Canary.ps1 Not Found",
                "The Canary installation script was not found.\n\n"
                "Expected location:\n"
                f"  {os.path.join(app_dir, 'Enable-Canary.ps1')}\n\n"
                "Please reinstall CV2T or download the script from GitHub.",
            )
            return

        if not skip_confirm:
            reply = QMessageBox.question(
                self,
                "Install Canary Engine",
                "This will install PyTorch + NVIDIA NeMo (~10 GB disk space).\n\n"
                "A PowerShell window will open with Administrator privileges.\n"
                "The installation may take 10-20 minutes.\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            # Launch elevated PowerShell with the Enable-Canary script.
            #
            # Quoting chain: Python → outer powershell → Start-Process → inner
            # powershell.  We use single-quotes inside the inner command so
            # paths with spaces (e.g. "C:\Program Files\CV2T") survive intact.
            # In PowerShell, single-quotes are escaped by doubling them ('').
            #
            # The outer -ArgumentList is one string using `\"…\"` so that the
            # inner single-quotes are preserved through Start-Process.
            esc_script = script_path.replace("'", "''")
            esc_app = app_dir.replace("'", "''")
            inner_cmd = (
                "Set-ExecutionPolicy Bypass -Scope Process -Force; "
                f"& '{esc_script}' -AppDir '{esc_app}'"
            )
            # Wrap for Start-Process: escape inner double-quotes for the
            # outer PowerShell layer, pass the command via -ArgumentList.
            outer_cmd = (
                'Start-Process powershell.exe '
                '-ArgumentList @('
                "'-NoProfile',"
                "'-Command',"
                f"'{inner_cmd.replace(chr(39), chr(39)*2)}'"
                ') -Verb RunAs'
            )
            subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-Command", outer_cmd],
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            QMessageBox.information(
                self,
                "Installation Started",
                "The Canary installation has started in a new window.\n\n"
                "After it completes, restart CV2T and select 'canary' "
                "as the Engine in Settings.",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Launch Failed",
                f"Failed to start the Canary installer:\n{exc}",
            )

    # ── Canary validation helpers ─────────────────────────────────────────

    def _on_validate_canary(self) -> None:
        """Run canary-env dependency checks on a background thread."""
        self._btn_validate_canary.setEnabled(False)
        self._lbl_canary_validate.setText("Validating\u2026")
        self._lbl_canary_validate.setStyleSheet("color: #757575;")

        model_path = self._model_path.text().strip() or self.settings.model_path

        from PySide6.QtCore import QThreadPool

        from .workers import Worker

        worker = Worker(self._do_canary_validation, model_path)
        worker.signals.result.connect(self._on_canary_validate_result)
        worker.signals.error.connect(self._on_canary_validate_error)
        QThreadPool.globalInstance().start(worker)

    @staticmethod
    def _do_canary_validation(model_path: str) -> list[tuple[str, bool, str]]:
        """Check canary-env, dependencies, CUDA, and model files.

        Runs in a worker thread.  Each check is a ``(name, passed, detail)``
        tuple so the UI can display a structured report.
        """
        import os
        import subprocess
        import sys

        results: list[tuple[str, bool, str]] = []

        # ── 1. Locate canary-env Python ──────────────────────────────────
        if getattr(sys, "frozen", False):
            app_dir = os.path.dirname(sys.executable)
        else:
            app_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )

        python_path = os.path.join(
            app_dir, "canary-env", ".venv", "Scripts", "python.exe",
        )
        if os.path.isfile(python_path):
            results.append(("canary-env Python", True, python_path))
        else:
            results.append((
                "canary-env Python", False,
                f"Not found: {python_path}",
            ))
            # Cannot run further subprocess checks without the interpreter.
            return results

        # ── 2. Dependency imports + CUDA check (subprocess) ──────────────
        # The canary-env packages (NeMo, torch, transformers, etc.) may
        # print warnings or banners to stdout during import.  Redirect
        # stdout to stderr *before* any imports so only the final JSON
        # line appears on stdout.
        dep_script = (
            "import sys, json, os, io\n"
            "_real_stdout = sys.stdout\n"
            "sys.stdout = sys.stderr\n"
            "os.environ['TORCHDYNAMO_DISABLE'] = '1'\n"
            "results = []\n"
            "for mod in ('numpy', 'soundfile', 'torch', 'accelerate',\n"
            "            'huggingface_hub', 'onnxruntime',\n"
            "            'transformers', 'sentencepiece', 'omegaconf',\n"
            "            'lightning', 'peft'):\n"
            "    try:\n"
            "        __import__(mod)\n"
            "        results.append((mod, True, 'OK'))\n"
            "    except ImportError as e:\n"
            "        results.append((mod, False, str(e)))\n"
            "try:\n"
            "    from nemo.collections.speechlm2.models import SALM\n"
            "    results.append(('nemo SALM', True, 'OK'))\n"
            "except ImportError as e:\n"
            "    results.append(('nemo SALM', False, str(e)))\n"
            "try:\n"
            "    import torch\n"
            "    cuda_ok = torch.cuda.is_available()\n"
            "    results.append(('CUDA', cuda_ok,\n"
            "        'available' if cuda_ok else 'torch.cuda.is_available() returned False'))\n"
            "except Exception as e:\n"
            "    results.append(('CUDA', False, str(e)))\n"
            "_real_stdout.write(json.dumps(results) + '\\n')\n"
            "_real_stdout.flush()\n"
        )

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        try:
            proc = subprocess.run(
                [python_path, "-c", dep_script],
                capture_output=True,
                text=True,
                timeout=120,
                creationflags=creationflags,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                import json
                dep_results = json.loads(proc.stdout.strip())
                results.extend(
                    (name, ok, detail) for name, ok, detail in dep_results
                )
            else:
                stderr_preview = (proc.stderr or "").strip()[:300]
                results.append((
                    "Dependency check", False,
                    f"Subprocess failed (exit {proc.returncode}): {stderr_preview}",
                ))
        except subprocess.TimeoutExpired:
            results.append(("Dependency check", False, "Timed out after 120 s"))
        except Exception as exc:
            results.append(("Dependency check", False, str(exc)))

        # ── 3. Model files ───────────────────────────────────────────────
        canary_dir = os.path.join(model_path, "canary")
        config_path = os.path.join(canary_dir, "config.json")
        safetensors_path = os.path.join(canary_dir, "model.safetensors")

        if os.path.isfile(config_path):
            results.append(("Model config.json", True, config_path))
        else:
            results.append(("Model config.json", False, f"Missing: {config_path}"))

        if os.path.isfile(safetensors_path):
            size_bytes = os.path.getsize(safetensors_path)
            if size_bytes >= 1_000_000_000:  # ~1 GB
                size_gb = round(size_bytes / (1024 ** 3), 1)
                results.append((
                    "Model weights", True,
                    f"{safetensors_path} ({size_gb} GB)",
                ))
            else:
                size_mb = round(size_bytes / (1024 ** 2))
                results.append((
                    "Model weights", False,
                    f"File too small ({size_mb} MB) — likely truncated",
                ))
        else:
            results.append((
                "Model weights", False,
                f"Missing: {safetensors_path}",
            ))

        return results

    def _on_canary_validate_result(self, results: list) -> None:
        """Handle canary validation results from the worker thread."""
        self._btn_validate_canary.setEnabled(True)

        failures = [(name, detail) for name, ok, detail in results if not ok]

        if not failures:
            self._lbl_canary_validate.setText("\u2705 All checks passed")
            self._lbl_canary_validate.setStyleSheet("color: #2e7d32;")
            return

        # Show inline summary
        self._lbl_canary_validate.setText(
            f"\u274c {len(failures)} check(s) failed"
        )
        self._lbl_canary_validate.setStyleSheet("color: #c62828;")

        # Build detailed report
        lines = []
        for name, ok, detail in results:
            icon = "\u2705" if ok else "\u274c"
            lines.append(f"{icon}  {name}: {detail}")
        report = "\n".join(lines)

        reply = QMessageBox.question(
            self,
            "Canary Validation Results",
            f"{report}\n\n"
            f"{len(failures)} check(s) failed.\n\n"
            "Launch Enable-Canary.ps1 to repair the installation?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._on_install_canary(skip_confirm=True)

    def _on_canary_validate_error(self, err: str) -> None:
        """Handle unexpected errors from the canary validation worker."""
        self._btn_validate_canary.setEnabled(True)
        self._lbl_canary_validate.setText(f"\u274c {err}")
        self._lbl_canary_validate.setStyleSheet("color: #c62828;")
