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
from .text_processor import (
    TextProcessor,
    delete_api_key_from_keyring,
    load_api_key_from_keyring,
    save_api_key_to_keyring,
)

log = logging.getLogger(__name__)

# Engines that require CUDA and cannot run on CPU.
_CUDA_ONLY_ENGINES = frozenset({"canary"})


class SettingsDialog(QDialog):
    """Modal dialog for editing application settings."""

    def __init__(
        self,
        settings: Settings,
        parent: Optional[QWidget] = None,
        api_key: str = "",
    ):
        super().__init__(parent)
        self.settings = settings
        self._api_key = api_key
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

        # ── Professional Mode group ──────────────────────────────────────────
        pro_group = QGroupBox("Professional Mode")
        pro_form = QFormLayout()

        self._pro_enabled = QCheckBox("Enable Professional Mode")
        pro_form.addRow(self._pro_enabled)

        self._pro_model = QComboBox()
        self._pro_model.setEditable(True)
        self._pro_model.addItems(["gpt-5.4-mini", "gpt-5.4-nano"])
        pro_form.addRow("Model:", self._pro_model)

        # API key row: password field + eye toggle
        key_row = QHBoxLayout()
        self._pro_api_key = QLineEdit()
        self._pro_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._pro_api_key.setPlaceholderText("sk-…")
        key_row.addWidget(self._pro_api_key)

        self._btn_eye = QPushButton("\U0001f441")
        self._btn_eye.setFixedWidth(32)
        self._btn_eye.setCheckable(True)
        self._btn_eye.setToolTip("Show / hide API key")
        self._btn_eye.toggled.connect(self._toggle_key_visibility)
        key_row.addWidget(self._btn_eye)
        pro_form.addRow("API key:", key_row)

        self._pro_store_key = QCheckBox("Remember API key (Windows Credential Manager)")
        pro_form.addRow(self._pro_store_key)

        # Validate button + inline result label
        validate_row = QHBoxLayout()
        self._btn_validate_key = QPushButton("Validate API Key")
        self._btn_validate_key.clicked.connect(self._on_validate_api_key)
        validate_row.addWidget(self._btn_validate_key)
        self._lbl_validate_result = QLabel("")
        validate_row.addWidget(self._lbl_validate_result)
        validate_row.addStretch()
        pro_form.addRow(validate_row)

        # Cleanup sub-options
        self._pro_fix_tone = QCheckBox("Fix tone (rewrite unprofessional language)")
        pro_form.addRow(self._pro_fix_tone)

        self._pro_fix_grammar = QCheckBox("Fix grammar")
        pro_form.addRow(self._pro_fix_grammar)

        self._pro_fix_punctuation = QCheckBox("Fix punctuation && capitalization")
        pro_form.addRow(self._pro_fix_punctuation)

        pro_group.setLayout(pro_form)
        layout.addWidget(pro_group)

        # Wire master toggle to enable/disable sub-widgets
        self._pro_sub_widgets = [
            self._pro_model,
            self._pro_api_key,
            self._btn_eye,
            self._pro_store_key,
            self._btn_validate_key,
            self._pro_fix_tone,
            self._pro_fix_grammar,
            self._pro_fix_punctuation,
        ]
        self._pro_enabled.toggled.connect(self._on_pro_toggled)

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

        # Professional Mode
        self._pro_enabled.setChecked(s.professional_mode)
        idx = self._pro_model.findText(s.pro_model)
        if idx >= 0:
            self._pro_model.setCurrentIndex(idx)
        else:
            self._pro_model.setCurrentText(s.pro_model)
        self._pro_store_key.setChecked(s.store_api_key)
        self._pro_fix_tone.setChecked(s.pro_fix_tone)
        self._pro_fix_grammar.setChecked(s.pro_fix_grammar)
        self._pro_fix_punctuation.setChecked(s.pro_fix_punctuation)

        # Load API key: from caller (in-memory) or keyring (if stored)
        if self._api_key:
            self._pro_api_key.setText(self._api_key)
        elif s.store_api_key:
            stored = load_api_key_from_keyring()
            if stored:
                self._pro_api_key.setText(stored)

        # Apply master toggle state to sub-widgets
        self._on_pro_toggled(s.professional_mode)

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

        # Professional Mode settings (API key is NEVER saved to settings JSON)
        s.professional_mode = self._pro_enabled.isChecked()
        s.pro_model = self._pro_model.currentText().strip() or "gpt-5.4-mini"
        s.pro_fix_tone = self._pro_fix_tone.isChecked()
        s.pro_fix_grammar = self._pro_fix_grammar.isChecked()
        s.pro_fix_punctuation = self._pro_fix_punctuation.isChecked()
        s.store_api_key = self._pro_store_key.isChecked()

        # API key: persist to keyring or delete, expose via property
        self._api_key = self._pro_api_key.text().strip()
        if s.store_api_key and self._api_key:
            save_api_key_to_keyring(self._api_key)
        elif not s.store_api_key:
            delete_api_key_from_keyring()

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

    # ── Professional Mode helpers ────────────────────────────────────────────

    # ── Canary add-on helpers ───────────────────────────────────────────────

    def _update_canary_status(self) -> None:
        """Refresh the Canary status label and install button state."""
        # Check if canary is available via direct import (source install)
        if "canary" in ENGINES:
            try:
                from cv2t.engine.canary import CanaryEngine
                self._canary_status_label.setText("\u2705 Installed (native)")
                self._canary_status_label.setStyleSheet("color: #2e7d32;")
                self._btn_install_canary.setVisible(False)
                return
            except ImportError:
                pass

            # Check if canary is available via subprocess bridge
            try:
                from cv2t.engine.canary_bridge import canary_env_available
                if canary_env_available():
                    self._canary_status_label.setText("\u2705 Installed (canary-env)")
                    self._canary_status_label.setStyleSheet("color: #2e7d32;")
                    self._btn_install_canary.setVisible(False)
                    return
            except ImportError:
                pass

        self._canary_status_label.setText("Not installed")
        self._canary_status_label.setStyleSheet("color: #757575;")
        self._btn_install_canary.setVisible(True)

    def _on_install_canary(self) -> None:
        """Launch Enable-Canary.ps1 in an elevated terminal."""
        import os
        import subprocess
        import sys

        # Find Enable-Canary.ps1
        if getattr(sys, "frozen", False):
            app_dir = os.path.dirname(sys.executable)
        else:
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        script_path = os.path.join(app_dir, "Enable-Canary.ps1")
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
            # Launch elevated PowerShell with the Enable-Canary script
            cmd = (
                f'Set-ExecutionPolicy Bypass -Scope Process -Force; '
                f'& "{script_path}"'
            )
            subprocess.Popen(
                [
                    "powershell.exe", "-Command",
                    f'Start-Process powershell.exe '
                    f'-ArgumentList \'-NoExit\',\'-Command\',\'{cmd}\' '
                    f'-Verb RunAs',
                ],
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

    # ── Professional Mode helpers ────────────────────────────────────────────

    def _on_pro_toggled(self, enabled: bool) -> None:
        """Enable or disable all Professional Mode sub-widgets."""
        for w in self._pro_sub_widgets:
            w.setEnabled(enabled)

    def _toggle_key_visibility(self, show: bool) -> None:
        if show:
            self._pro_api_key.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self._pro_api_key.setEchoMode(QLineEdit.EchoMode.Password)

    def _on_validate_api_key(self) -> None:
        key = self._pro_api_key.text().strip()
        if not key:
            self._lbl_validate_result.setText("\u274c No API key entered")
            self._lbl_validate_result.setStyleSheet("color: #c62828;")
            return

        self._lbl_validate_result.setText("Validating…")
        self._lbl_validate_result.setStyleSheet("color: #757575;")
        self._btn_validate_key.setEnabled(False)

        model = self._pro_model.currentText()

        def _do_validate():
            processor = TextProcessor(api_key=key, model=model)
            return processor.validate_key()

        from PySide6.QtCore import QThreadPool

        from .workers import Worker

        worker = Worker(_do_validate)
        worker.signals.result.connect(self._on_validate_result)
        worker.signals.error.connect(self._on_validate_error)
        QThreadPool.globalInstance().start(worker)

    def _on_validate_result(self, result: tuple) -> None:
        self._btn_validate_key.setEnabled(True)
        ok, msg = result
        if ok:
            self._lbl_validate_result.setText(f"\u2705 {msg}")
            self._lbl_validate_result.setStyleSheet("color: #2e7d32;")
        else:
            self._lbl_validate_result.setText(f"\u274c {msg}")
            self._lbl_validate_result.setStyleSheet("color: #c62828;")

    def _on_validate_error(self, err: str) -> None:
        self._btn_validate_key.setEnabled(True)
        self._lbl_validate_result.setText(f"\u274c {err}")
        self._lbl_validate_result.setStyleSheet("color: #c62828;")

    @property
    def api_key(self) -> str:
        """Return the API key entered by the user (in-memory only)."""
        return self._api_key
