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
    QLineEdit,
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

    def __init__(self, settings: Settings, parent: Optional[QWidget] = None):
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
