"""
Main application window for CV2T.

Integrates model engine lifecycle, audio recording, transcription,
clipboard, hotkeys, and history into a single cohesive window.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from .audio import AudioRecorder, play_beep
from .clipboard import set_clipboard_text, simulate_paste
from ._constants import (
    COLOR_DIMMED,
    COLOR_ERROR,
    COLOR_IDLE,
    COLOR_INFO,
    COLOR_NEUTRAL,
    COLOR_SUCCESS,
    COLOR_VALIDATED,
    COLOR_WARNING,
    LOADING_TICK_MS,
    METRICS_POLL_MS,
    PBT_APMRESUMEAUTOMATIC,
    PBT_APMRESUMESUSPEND,
    STATE_RESET_ERROR_MS,
    STATE_RESET_IDLE_MS,
    SYSTEM_RESUME_DEBOUNCE_S,
    SYSTEM_RESUME_DELAY_MS,
    WM_POWERBROADCAST,
)
from .config import DEFAULT_LOG_DIR, DEFAULT_PRESETS_DIR, Settings
from .engine import ENGINES
from .hotkeys import HotkeyManager
from ._resource_monitor import ResourceMonitor
from .pro_preset import ProPreset, bootstrap_presets, load_all_presets
from .text_processor import TextProcessor, load_api_key_from_keyring
from .workers import Worker

# Engine families that cannot coexist in a single process due to
# CTranslate2 / PyTorch CUDA runtime conflicts.
_CTRANSLATE2_ENGINES = frozenset({"whisper"})
_TORCH_ENGINES = frozenset({"canary"})

log = logging.getLogger(__name__)


# ── Qt-compatible log handler ─────────────────────────────────────────────────


class _QtLogEmitter(QObject):
    log_signal = Signal(str)


class QtLogHandler(logging.Handler):
    """Routes log records to a Qt signal for display in the log panel."""

    def __init__(self) -> None:
        super().__init__()
        self.emitter = _QtLogEmitter()

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.emitter.log_signal.emit(msg)


# ── State enums ───────────────────────────────────────────────────────────────


class DictationState(str, Enum):
    IDLE = "Idle"
    RECORDING = "Recording…"
    PROCESSING = "Processing…"
    SUCCESS = "Success"
    ERROR = "Error"


class ModelStatus(str, Enum):
    NOT_LOADED = "Not loaded"
    LOADING = "Loading…"
    READY = "Ready"
    VALIDATING = "Validating…"
    VALIDATED = "Validated"
    ERROR = "Error"


# ── History entry ─────────────────────────────────────────────────────────────


class _HistoryEntry(QWidget):
    """Single row in the transcription history."""

    def __init__(
        self,
        timestamp: str,
        text: str,
        success: bool,
        parent: Optional[QWidget] = None,
        original_text: Optional[str] = None,
    ):
        super().__init__(parent)
        self._text = text
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 2, 4, 2)

        icon = "\u2705" if success else "\u274c"
        time_label = QLabel(f"<b>{timestamp}</b>")
        time_label.setFixedWidth(70)
        status_label = QLabel(icon)
        status_label.setFixedWidth(22)

        # Build text column — one or two labels depending on professional mode
        if original_text is not None:
            text_col = QVBoxLayout()
            text_col.setContentsMargins(0, 0, 0, 0)
            text_col.setSpacing(1)

            orig_display = original_text if len(original_text) <= 120 else original_text[:117] + "…"
            orig_label = QLabel(f'<span style="color:{COLOR_DIMMED}">Original: {orig_display}</span>')
            orig_label.setWordWrap(True)
            orig_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            orig_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            text_col.addWidget(orig_label)

            clean_display = text if len(text) <= 120 else text[:117] + "…"
            clean_label = QLabel(f"Cleaned: {clean_display}")
            clean_label.setWordWrap(True)
            clean_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            clean_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            text_col.addWidget(clean_label)

            text_widget = QWidget()
            text_widget.setLayout(text_col)
            text_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        else:
            display = text if len(text) <= 120 else text[:117] + "…"
            text_widget = QLabel(display)
            text_widget.setWordWrap(True)
            text_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            text_widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        copy_btn = QPushButton("Copy")
        copy_btn.setFixedWidth(50)
        copy_btn.clicked.connect(self._copy)

        row.addWidget(time_label)
        row.addWidget(status_label)
        row.addWidget(text_widget)
        row.addWidget(copy_btn)

    def _copy(self) -> None:
        set_clipboard_text(self._text)


# ═════════════════════════════════════════════════════════════════════════════
# Main Window
# ═════════════════════════════════════════════════════════════════════════════


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self, settings: Settings, engine=None):
        super().__init__()
        self.settings = settings
        self._pool = QThreadPool.globalInstance()

        # ── Engine ───────────────────────────────────────────────────────────
        if engine is not None:
            self._engine = engine
        else:
            engine_cls = ENGINES.get(settings.engine)
            if engine_cls is None:
                available = list(ENGINES.keys())
                if available:
                    fallback = available[0]
                    log.warning(
                        "Engine '%s' not available; falling back to '%s'. "
                        "Installed engines: %s",
                        settings.engine, fallback, available,
                    )
                    settings.engine = fallback
                    settings.save()
                    engine_cls = ENGINES[fallback]
                else:
                    raise RuntimeError(
                        "No speech engines available. Re-install the "
                        "application or check that dependencies are intact."
                    )
            self._engine = engine_cls()

        # ── Audio ────────────────────────────────────────────────────────────
        self._recorder = AudioRecorder(
            sample_rate=settings.sample_rate,
            silence_threshold=settings.silence_threshold,
            silence_margin_ms=settings.silence_margin_ms,
            device=settings.mic_device_index if settings.mic_device_index >= 0 else None,
        )
        self._hotkey_mgr = HotkeyManager(parent=self)

        # ── State ────────────────────────────────────────────────────────────
        self._dictation_state = DictationState.IDLE
        self._model_status = ModelStatus.NOT_LOADED
        self._model_load_start: float = 0.0
        self._last_resume_time: float = 0.0

        # ── Resource monitor ─────────────────────────────────────────────────
        self._res_monitor = ResourceMonitor(
            pool=self._pool, interval_ms=METRICS_POLL_MS, parent=self,
        )
        self._res_monitor.metrics_updated.connect(self._on_metrics_result)
        self._res_monitor.metrics_error.connect(
            lambda err: log.error("Metrics poll error: %s", err)
        )

        # ── Professional Mode ────────────────────────────────────────────────
        self._api_key: str = ""
        self._text_processor: Optional[TextProcessor] = None
        self._pro_worker: Optional[Worker] = None
        self._pro_context: Optional[tuple[str, str]] = None  # (ts, original)
        self._pro_timeout: Optional[QTimer] = None
        self._pro_presets: dict[str, ProPreset] = {}
        self._active_preset: Optional[ProPreset] = None

        # Bootstrap presets directory and load presets
        bootstrap_presets(DEFAULT_PRESETS_DIR)
        self._pro_presets = load_all_presets(DEFAULT_PRESETS_DIR)
        self._active_preset = self._pro_presets.get(settings.pro_active_preset)

        if settings.store_api_key:
            self._api_key = load_api_key_from_keyring()
        if settings.professional_mode and self._api_key and self._active_preset:
            self._text_processor = TextProcessor(
                api_key=self._api_key,
                model=self._active_preset.model or "gpt-5.4-mini",
            )
        elif settings.professional_mode and not self._api_key:
            log.warning("Professional Mode enabled but no API key configured")

        # ── Build UI ─────────────────────────────────────────────────────────
        self.setWindowTitle("CV2T — Voice to Text")
        self.setMinimumSize(640, 700)
        self.resize(720, 820)
        self._build_ui()
        self._setup_logging()
        self._setup_timers()
        self._connect_hotkeys()

        # ── Open mic stream ──────────────────────────────────────────────────
        try:
            self._recorder.open_stream()
            self._log_ui("Microphone stream opened")
        except Exception as exc:
            self._log_ui(f"Microphone error: {exc}", error=True)

        # ── Begin model loading ──────────────────────────────────────────────
        self._load_model()

    # ═════════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ═════════════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)

        # ── Transcription section (dominant) ─────────────────────────────────
        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("\U0001f3a4  Start Recording  (Ctrl+Alt+P)")
        self._btn_start.setMinimumHeight(52)
        self._btn_start.setStyleSheet("font-size: 14px;")
        self._btn_start.clicked.connect(self._on_start_recording)
        self._btn_stop = QPushButton("\u23f9  Stop && Transcribe  (Ctrl+Alt+L)")
        self._btn_stop.setMinimumHeight(52)
        self._btn_stop.setStyleSheet("font-size: 14px;")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop_and_transcribe)
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        root.addLayout(btn_row)

        # ── Status indicators (model + dictation) ────────────────────────────
        status_row_top = QHBoxLayout()
        self._lbl_global_status = QLabel()
        self._lbl_global_status.setFont(QFont("Segoe UI", 10))
        status_row_top.addWidget(self._lbl_global_status)
        status_row_top.addStretch()
        root.addLayout(status_row_top)
        self._update_global_status()

        toggle_row = QHBoxLayout()
        self._chk_auto_copy = QCheckBox("Auto-copy to clipboard")
        self._chk_auto_copy.setChecked(self.settings.auto_copy)
        self._chk_auto_paste = QCheckBox("Auto-paste (Ctrl+V)")
        self._chk_auto_paste.setChecked(self.settings.auto_paste)
        self._chk_hotkeys = QCheckBox("Enable global hotkeys")
        self._chk_hotkeys.setChecked(self.settings.hotkeys_enabled)
        self._chk_hotkeys.toggled.connect(self._on_hotkeys_toggled)
        self._chk_professional = QCheckBox("Professional Mode")
        self._chk_professional.setChecked(self.settings.professional_mode)
        self._chk_professional.toggled.connect(self._on_professional_toggled)
        toggle_row.addWidget(self._chk_auto_copy)
        toggle_row.addWidget(self._chk_auto_paste)
        toggle_row.addWidget(self._chk_hotkeys)
        toggle_row.addWidget(self._chk_professional)
        toggle_row.addStretch()
        root.addLayout(toggle_row)

        # History header with contextual Clear button
        history_header = QHBoxLayout()
        history_header.addWidget(QLabel("<b>Transcription History</b>"))
        history_header.addStretch()
        self._btn_clear_history = QPushButton("\U0001f5d1  Clear History")
        self._btn_clear_history.clicked.connect(self._on_clear_history)
        history_header.addWidget(self._btn_clear_history)
        root.addLayout(history_header)

        self._history_widget = QWidget()
        self._history_layout = QVBoxLayout(self._history_widget)
        self._history_layout.setContentsMargins(0, 0, 0, 0)
        self._history_layout.setSpacing(2)
        self._history_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._history_widget)
        scroll.setMinimumHeight(200)
        root.addWidget(scroll, stretch=1)

        # ── Collapsible Advanced Diagnostics panel ───────────────────────────
        self._diag_toggle = QPushButton("\u25b6 Advanced Diagnostics")
        self._diag_toggle.setFlat(True)
        self._diag_toggle.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; padding: 4px; }"
        )
        self._diag_toggle.clicked.connect(self._toggle_diagnostics)
        root.addWidget(self._diag_toggle)

        self._diag_content = QWidget()
        diag_layout = QVBoxLayout(self._diag_content)
        diag_layout.setContentsMargins(0, 0, 0, 0)

        # Model Engine panel
        engine_group = QGroupBox("Model Engine")
        eg_layout = QVBoxLayout()

        status_row = QHBoxLayout()
        self._lbl_engine = QLabel(f"Engine: {self._engine.name}")
        self._lbl_model_status = QLabel("Status: Not loaded")
        self._lbl_engine.setFont(QFont("Segoe UI", 10))
        self._lbl_model_status.setFont(QFont("Segoe UI", 10))
        status_row.addWidget(self._lbl_engine)
        status_row.addWidget(self._lbl_model_status)
        status_row.addStretch()
        eg_layout.addLayout(status_row)

        metrics_row = QHBoxLayout()
        self._lbl_ram = QLabel("RAM: —")
        self._lbl_vram = QLabel("VRAM: —")
        self._lbl_gpu_info = QLabel("GPU: —")
        self._lbl_ram.setFont(QFont("Segoe UI", 9))
        self._lbl_vram.setFont(QFont("Segoe UI", 9))
        self._lbl_gpu_info.setFont(QFont("Segoe UI", 9))
        metrics_row.addWidget(self._lbl_ram)
        metrics_row.addWidget(self._lbl_vram)
        metrics_row.addWidget(self._lbl_gpu_info)
        metrics_row.addStretch()
        eg_layout.addLayout(metrics_row)

        btn_row_engine = QHBoxLayout()
        self._btn_reload = QPushButton("Reload Model")
        self._btn_reload.clicked.connect(self._on_reload_model)
        self._btn_validate = QPushButton("Validate")
        self._btn_validate.clicked.connect(self._on_validate)
        btn_row_engine.addWidget(self._btn_reload)
        btn_row_engine.addWidget(self._btn_validate)
        btn_row_engine.addStretch()
        eg_layout.addLayout(btn_row_engine)

        engine_group.setLayout(eg_layout)

        # Log panel with contextual buttons in header
        log_group = QGroupBox("Log")
        lg_layout = QVBoxLayout()

        log_header = QHBoxLayout()
        log_header.addStretch()
        self._btn_clear_logs = QPushButton("\U0001f5d1  Clear Logs")
        self._btn_clear_logs.clicked.connect(self._on_clear_logs)
        self._btn_copy_logs = QPushButton("\U0001f4cb  Copy Logs")
        self._btn_copy_logs.clicked.connect(self._on_copy_logs)
        log_header.addWidget(self._btn_clear_logs)
        log_header.addWidget(self._btn_copy_logs)
        lg_layout.addLayout(log_header)

        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumBlockCount(500)
        self._log_text.setFont(QFont("Consolas", 9))
        lg_layout.addWidget(self._log_text)
        log_group.setLayout(lg_layout)

        # Splitter for engine + log inside diagnostics
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(engine_group)
        splitter.addWidget(log_group)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        diag_layout.addWidget(splitter)

        self._diag_content.setVisible(False)
        root.addWidget(self._diag_content)

        # ── Bottom buttons ───────────────────────────────────────────────────
        bottom_row = QHBoxLayout()
        btn_settings = QPushButton("\u2699  Settings")
        btn_settings.clicked.connect(self._on_open_settings)
        btn_pro_settings = QPushButton("\u2695  Professional Mode Settings")
        btn_pro_settings.clicked.connect(self._on_open_pro_settings)
        btn_quit = QPushButton("Quit")
        btn_quit.clicked.connect(self.close)
        bottom_row.addWidget(btn_settings)
        bottom_row.addWidget(btn_pro_settings)
        bottom_row.addStretch()
        bottom_row.addWidget(btn_quit)
        root.addLayout(bottom_row)



    def _toggle_diagnostics(self) -> None:
        """Show or hide the Advanced Diagnostics panel."""
        was_hidden = self._diag_content.isHidden()
        self._diag_content.setVisible(was_hidden)
        self._diag_toggle.setText(
            "\u25bc Advanced Diagnostics" if was_hidden
            else "\u25b6 Advanced Diagnostics"
        )

    def _update_global_status(self) -> None:
        """Refresh the unified status bar with model, dictation, and professional mode state."""
        model_color_map = {
            ModelStatus.READY: COLOR_SUCCESS,
            ModelStatus.VALIDATED: COLOR_VALIDATED,
            ModelStatus.LOADING: COLOR_WARNING,
            ModelStatus.NOT_LOADED: COLOR_NEUTRAL,
            ModelStatus.VALIDATING: COLOR_INFO,
            ModelStatus.ERROR: COLOR_ERROR,
        }
        dict_color_map = {
            DictationState.IDLE: COLOR_IDLE,
            DictationState.RECORDING: COLOR_ERROR,
            DictationState.PROCESSING: COLOR_WARNING,
            DictationState.SUCCESS: COLOR_SUCCESS,
            DictationState.ERROR: COLOR_ERROR,
        }
        m_color = model_color_map.get(self._model_status, COLOR_NEUTRAL)
        d_color = dict_color_map.get(self._dictation_state, COLOR_IDLE)

        # Professional mode status
        if self.settings.professional_mode and self._text_processor is not None:
            preset_name = self.settings.pro_active_preset
            pro_text = f'Active ({preset_name})'
            pro_color = COLOR_SUCCESS
        elif self.settings.professional_mode:
            pro_text = 'No API Key'
            pro_color = COLOR_WARNING
        else:
            pro_text = 'Inactive'
            pro_color = COLOR_NEUTRAL

        self._lbl_global_status.setText(
            f'Model: <span style="color:{m_color}"><b>{self._model_status.value}</b></span>'
            f'  \u00b7  '
            f'Dictation: <span style="color:{d_color}"><b>{self._dictation_state.value}</b></span>'
            f'  \u00b7  '
            f'Professional: <span style="color:{pro_color}"><b>{pro_text}</b></span>'
        )

    # ═════════════════════════════════════════════════════════════════════════
    # LOGGING INTEGRATION
    # ═════════════════════════════════════════════════════════════════════════

    def _setup_logging(self) -> None:
        handler = QtLogHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%H:%M:%S"))
        handler.emitter.log_signal.connect(self._append_log)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

    @Slot(str)
    def _append_log(self, msg: str) -> None:
        self._log_text.appendPlainText(msg)

    def _log_ui(self, msg: str, error: bool = False) -> None:
        if error:
            log.error(msg)
        else:
            log.info(msg)

    # ═════════════════════════════════════════════════════════════════════════
    # TIMERS
    # ═════════════════════════════════════════════════════════════════════════

    def _setup_timers(self) -> None:
        # Model loading elapsed timer (updates label during loading)
        self._loading_timer = QTimer(self)
        self._loading_timer.timeout.connect(self._update_loading_label)
        self._loading_timer.setInterval(LOADING_TICK_MS)

        # Start resource-metrics polling
        self._res_monitor.start()

    # ═════════════════════════════════════════════════════════════════════════
    # HOTKEYS
    # ═════════════════════════════════════════════════════════════════════════

    def _connect_hotkeys(self) -> None:
        self._hotkey_mgr.start_requested.connect(self._on_start_recording)
        self._hotkey_mgr.stop_requested.connect(self._on_stop_and_transcribe)
        self._hotkey_mgr.quit_requested.connect(self.close)
        if self.settings.hotkeys_enabled:
            self._hotkey_mgr.register(
                self.settings.hotkey_start,
                self.settings.hotkey_stop,
                self.settings.hotkey_quit,
            )

    @Slot(bool)
    def _on_hotkeys_toggled(self, enabled: bool) -> None:
        if enabled:
            self._hotkey_mgr.register(
                self.settings.hotkey_start,
                self.settings.hotkey_stop,
                self.settings.hotkey_quit,
            )
            self._log_ui("Global hotkeys enabled")
        else:
            self._hotkey_mgr.unregister()
            self._log_ui("Global hotkeys disabled")

    # ═════════════════════════════════════════════════════════════════════════
    # MODEL ENGINE MANAGEMENT
    # ═════════════════════════════════════════════════════════════════════════

    def _set_model_status(self, status: ModelStatus) -> None:
        self._model_status = status
        color_map = {
            ModelStatus.READY: COLOR_SUCCESS,
            ModelStatus.VALIDATED: COLOR_VALIDATED,
            ModelStatus.LOADING: COLOR_WARNING,
            ModelStatus.NOT_LOADED: COLOR_NEUTRAL,
            ModelStatus.VALIDATING: COLOR_INFO,
            ModelStatus.ERROR: COLOR_ERROR,
        }
        color = color_map.get(status, COLOR_NEUTRAL)
        self._lbl_model_status.setText(
            f'Status: <span style="color:{color}"><b>{status.value}</b></span>'
        )
        self._update_global_status()
        self._refresh_dictation_buttons()

    def _load_model(self) -> None:
        """Begin model loading on a worker thread."""
        self._set_model_status(ModelStatus.LOADING)
        self._model_load_start = time.time()
        self._loading_timer.start()
        self._log_ui(f"Loading {self._engine.name} model…")

        def _do_load():
            if hasattr(self._engine, "force_cuda_sync"):
                self._engine.force_cuda_sync = self.settings.force_cuda_sync
            self._engine.load(self.settings.model_path, self.settings.device)

        worker = Worker(_do_load)
        worker.signals.result.connect(self._on_model_loaded)
        worker.signals.error.connect(self._on_model_load_error)
        self._pool.start(worker)

    @Slot(object)
    def _on_model_loaded(self, _result) -> None:
        self._loading_timer.stop()
        elapsed = time.time() - self._model_load_start
        self._set_model_status(ModelStatus.READY)
        self._lbl_engine.setText(f"Engine: {self._engine.name}")
        self._log_ui(f"Model loaded in {elapsed:.1f}s")

    @Slot(str)
    def _on_model_load_error(self, err: str) -> None:
        self._loading_timer.stop()
        self._set_model_status(ModelStatus.ERROR)
        self._log_ui(f"Model load failed: {err}", error=True)

    def _update_loading_label(self) -> None:
        """Update the status label with elapsed loading time."""
        if self._model_status == ModelStatus.LOADING:
            elapsed = int(time.time() - self._model_load_start)
            self._lbl_model_status.setText(
                f'Status: <span style="color:{COLOR_WARNING}"><b>Loading… {elapsed}s</b></span>'
            )

    @Slot()
    def _on_reload_model(self) -> None:
        """Unload then reload the model."""
        self._log_ui("Reloading model…")

        def _do_reload():
            self._engine.unload()
            if hasattr(self._engine, "force_cuda_sync"):
                self._engine.force_cuda_sync = self.settings.force_cuda_sync
            self._engine.load(self.settings.model_path, self.settings.device)

        self._set_model_status(ModelStatus.LOADING)
        self._model_load_start = time.time()
        self._loading_timer.start()

        worker = Worker(_do_reload)
        worker.signals.result.connect(self._on_model_loaded)
        worker.signals.error.connect(self._on_model_load_error)
        self._pool.start(worker)

    # ── Resource metrics ──────────────────────────────────────────────────────

    @Slot(object)
    def _on_metrics_result(self, metrics) -> None:
        if metrics.ram_total_gb > 0:
            self._lbl_ram.setText(
                f"RAM: {metrics.ram_used_gb:.1f} / {metrics.ram_total_gb:.1f} GB "
                f"({metrics.ram_percent:.0f}%)"
            )
        else:
            self._lbl_ram.setText("RAM: —")

        gpu = metrics.gpu
        if gpu.vram_total_gb > 0:
            pct = gpu.vram_percent
            if pct > 90:
                color = COLOR_ERROR
            elif pct > 75:
                color = COLOR_WARNING
            else:
                color = COLOR_SUCCESS
            self._lbl_vram.setText(
                f'VRAM: <span style="color:{color}"><b>{gpu.vram_used_gb:.1f}</b></span>'
                f" / {gpu.vram_total_gb:.1f} GB ({pct:.0f}%)"
            )
            self._lbl_gpu_info.setText(f"GPU: {gpu.name} ({gpu.temperature_c}°C)")
        else:
            self._lbl_vram.setText("VRAM: —")
            self._lbl_gpu_info.setText("GPU: —")

    # ── Validate ──────────────────────────────────────────────────────────────

    @Slot()
    def _on_validate(self) -> None:
        if not self._engine.is_loaded:
            self._log_ui("Cannot validate — model not loaded", error=True)
            return
        self._set_model_status(ModelStatus.VALIDATING)
        self._log_ui("Running functional validation…")

        def _do_validate():
            # Use bundled speech fixture
            fixture_path = Path(__file__).parent / "assets" / "validation.wav"
            if not fixture_path.exists():
                return False, "Validation fixture not found"
            import numpy as np
            import soundfile as sf
            audio, sr = sf.read(fixture_path, dtype="float32")
            if audio.ndim == 2:
                audio = audio[:, 0]
            text = self._engine.transcribe(audio, sr)
            # Loose match — just check for some expected words
            text_lower = text.lower()
            if any(w in text_lower for w in ("testing", "one", "two", "three")):
                return True, f"OK: \"{text}\""
            elif text.strip():
                return True, f"Got text (unexpected): \"{text}\""
            else:
                return False, "Empty transcription result"

        worker = Worker(_do_validate)
        worker.signals.result.connect(self._on_validate_result)
        worker.signals.error.connect(lambda e: self._on_validate_result((False, str(e))))
        self._pool.start(worker)

    @Slot(object)
    def _on_validate_result(self, result: tuple) -> None:
        ok, msg = result
        if ok:
            self._set_model_status(ModelStatus.VALIDATED)
            self._log_ui(f"Validation passed: {msg}")
        else:
            self._set_model_status(ModelStatus.ERROR)
            self._log_ui(f"Validation failed: {msg}", error=True)

    # ═════════════════════════════════════════════════════════════════════════
    # DICTATION
    # ═════════════════════════════════════════════════════════════════════════

    def _set_dictation_state(self, state: DictationState) -> None:
        self._dictation_state = state
        self._update_global_status()
        self._refresh_dictation_buttons()

    def _refresh_dictation_buttons(self) -> None:
        """Enable/disable Start & Stop buttons based on dictation + model state."""
        is_idle = self._dictation_state == DictationState.IDLE
        is_recording = self._dictation_state == DictationState.RECORDING
        model_ready = self._model_status in (ModelStatus.READY, ModelStatus.VALIDATED)
        self._btn_start.setEnabled(is_idle and model_ready)
        self._btn_stop.setEnabled(is_recording)

    @Slot()
    def _on_start_recording(self) -> None:
        if self._dictation_state != DictationState.IDLE:
            return
        if self._model_status not in (ModelStatus.READY, ModelStatus.VALIDATED):
            self._log_ui("Cannot record — model not ready yet", error=True)
            return
        play_beep((600, 900))   # ascending chirp → "go!"
        self._recorder.start_recording()
        self._set_dictation_state(DictationState.RECORDING)
        self._log_ui("Recording started")

    @Slot()
    def _on_stop_and_transcribe(self) -> None:
        """Stop recording, trim, transcribe in-process, clipboard, paste — threaded."""
        if self._dictation_state != DictationState.RECORDING:
            return
        play_beep((900, 500))   # descending chirp → "done"
        self._set_dictation_state(DictationState.PROCESSING)

        # Pause NVML polling — concurrent driver calls can
        # deadlock against CUDA kernel launches in generate().
        self._res_monitor.stop()

        # Wait for any in-flight metrics poll to finish before dispatching
        # the transcription worker (avoids NVML / CUDA overlap).
        import time as _time
        _deadline = _time.monotonic() + 2.0
        while self._res_monitor.is_in_flight and _time.monotonic() < _deadline:
            from PySide6.QtCore import QCoreApplication
            QCoreApplication.processEvents()
            _time.sleep(0.05)

        # Get raw audio (fast, on main thread)
        audio = self._recorder.get_raw_audio()
        if audio is None:
            self._log_ui("No audio recorded", error=True)
            self._res_monitor.start()
            self._set_dictation_state(DictationState.IDLE)
            return

        self._log_ui(f"Recording stopped \u2014 captured {len(audio)/self.settings.sample_rate:.1f}s of audio")

        # Heavy work on thread pool — NO clipboard ops here
        def _process():
            # Trim silence
            trim_result = self._recorder.trim_silence(audio)
            if trim_result is None:
                raise RuntimeError("No speech detected — audio was pure silence")
            trimmed, pct = trim_result
            if pct > 1:
                log.info("Trimmed %.0f%% silence", pct)

            # Contiguous copy — trim_silence returns a view/slice that can
            # cause native-code crashes in CTranslate2 / CUDA.
            trimmed = np.ascontiguousarray(trimmed, dtype=np.float32)

            # Transcribe in-process
            text = self._engine.transcribe(
                trimmed, self.settings.sample_rate, self.settings.language
            )
            return text

        worker = Worker(_process)
        worker.signals.result.connect(self._on_transcription_result)
        worker.signals.error.connect(self._on_transcription_error)
        self._pool.start(worker)

    @Slot(object)
    def _on_transcription_result(self, text: str) -> None:
        """Handle transcription result — runs on MAIN THREAD (safe for clipboard)."""
        self._res_monitor.start()
        text = str(text).strip()
        ts = datetime.datetime.now().strftime("%H:%M:%S")

        if text:
            self._set_dictation_state(DictationState.SUCCESS)
            self._log_ui(f"Transcribed: {len(text)} chars")

            # Professional Mode: send to OpenAI for cleanup
            if (
                self.settings.professional_mode
                and self._text_processor is not None
                and self._active_preset is not None
            ):
                self._log_ui("Cleaning up text…")

                preset = self._active_preset

                def _cleanup():
                    result = self._text_processor.process(
                        text,
                        preset=preset,
                    )
                    log.info("Professional cleanup worker finished (%d chars)", len(result))
                    return result

                # Store context for the bound-method handlers so we
                # don't need lambdas (lambdas prevent QObject connection
                # tracking and allow the Worker to be GC'd prematurely).
                self._pro_context = (ts, text)
                self._pro_worker = Worker(_cleanup)
                self._pro_worker.setAutoDelete(False)  # we manage lifetime
                self._pro_worker.signals.result.connect(self._on_professional_result)
                self._pro_worker.signals.error.connect(self._on_professional_error)
                self._pro_worker.signals.finished.connect(self._on_professional_finished)
                self._pool.start(self._pro_worker)

                # Safety timeout — if signal delivery fails for any
                # reason, fall back after the API timeout + buffer.
                self._pro_timeout = QTimer(self)
                self._pro_timeout.setSingleShot(True)
                self._pro_timeout.timeout.connect(self._on_professional_timeout)
                self._pro_timeout.start(20_000)  # 20 s
                return

            self._add_history(ts, text, success=True)

            copied = True
            if self._chk_auto_copy.isChecked():
                copied = set_clipboard_text(text)  # MAIN THREAD — safe
                if copied:
                    self._log_ui("Copied to clipboard")
                else:
                    self._log_ui("Failed to copy to clipboard", error=True)

            if copied and self._chk_auto_paste.isChecked():
                # Run paste in a thread to avoid blocking UI during modifier wait
                def _paste():
                    simulate_paste(wait_for_modifiers=self._chk_hotkeys.isChecked())
                w = Worker(_paste)
                self._pool.start(w)
        else:
            self._log_ui("Transcription returned empty text")
            self._add_history(ts, "(empty)", success=True)
            self._set_dictation_state(DictationState.SUCCESS)

        QTimer.singleShot(
            STATE_RESET_IDLE_MS,
            lambda: self._set_dictation_state(DictationState.IDLE)
            if self._dictation_state in (DictationState.SUCCESS, DictationState.ERROR)
            else None,
        )

    @Slot(object)
    def _on_professional_result(self, cleaned_raw: object) -> None:
        """Handle the cleaned text from Professional Mode."""
        log.info("Professional result signal delivered to main thread")
        ctx = self._pro_context  # read BEFORE cancel clears it
        self._cancel_pro_timeout()
        if ctx is None:
            return  # already handled (e.g. by timeout)
        ts, original = ctx
        cleaned = str(cleaned_raw).strip()
        if cleaned and cleaned != original:
            self._log_ui(f"Professional cleanup: {len(original)} -> {len(cleaned)} chars")
            self._add_history(ts, cleaned, success=True, original_text=original)
            output = cleaned
        else:
            self._log_ui("Professional cleanup returned unchanged text")
            self._add_history(ts, original, success=True)
            output = original

        copied = True
        if self._chk_auto_copy.isChecked():
            copied = set_clipboard_text(output)
            if copied:
                self._log_ui("Copied to clipboard")
            else:
                self._log_ui("Failed to copy to clipboard", error=True)

        if copied and self._chk_auto_paste.isChecked():
            def _paste():
                simulate_paste(wait_for_modifiers=self._chk_hotkeys.isChecked())
            w = Worker(_paste)
            self._pool.start(w)

        QTimer.singleShot(
            STATE_RESET_IDLE_MS,
            lambda: self._set_dictation_state(DictationState.IDLE)
            if self._dictation_state in (DictationState.SUCCESS, DictationState.ERROR)
            else None,
        )

    @Slot(str)
    def _on_professional_error(self, err: str) -> None:
        """Professional Mode cleanup failed — fall back to raw text."""
        log.info("Professional error signal delivered to main thread")
        ctx = self._pro_context  # read BEFORE cancel clears it
        self._cancel_pro_timeout()
        if ctx is None:
            return  # already handled (e.g. by timeout)
        ts, original = ctx
        self._log_ui(f"Professional cleanup failed: {err}", error=True)
        self._add_history(ts, original, success=True)

        copied = True
        if self._chk_auto_copy.isChecked():
            copied = set_clipboard_text(original)
            if copied:
                self._log_ui("Copied original text to clipboard (cleanup failed)")
            else:
                self._log_ui("Failed to copy to clipboard", error=True)

        if copied and self._chk_auto_paste.isChecked():
            def _paste():
                simulate_paste(wait_for_modifiers=self._chk_hotkeys.isChecked())
            w = Worker(_paste)
            self._pool.start(w)

        QTimer.singleShot(
            STATE_RESET_IDLE_MS,
            lambda: self._set_dictation_state(DictationState.IDLE)
            if self._dictation_state in (DictationState.SUCCESS, DictationState.ERROR)
            else None,
        )

    @Slot()
    def _on_professional_finished(self) -> None:
        """Worker done — drop the reference (prevent leak)."""
        self._pro_worker = None

    def _cancel_pro_timeout(self) -> None:
        """Stop the safety timer and clear professional-mode context."""
        if self._pro_timeout is not None:
            self._pro_timeout.stop()
            self._pro_timeout.deleteLater()
            self._pro_timeout = None
        self._pro_context = None

    @Slot()
    def _on_professional_timeout(self) -> None:
        """Safety net — professional cleanup did not complete in time."""
        ctx = self._pro_context
        self._pro_timeout = None
        self._pro_context = None
        self._pro_worker = None
        if ctx is None:
            return  # result/error already handled normally
        ts, original = ctx
        log.warning("Professional cleanup timed out — falling back to original text")
        self._log_ui("Professional cleanup timed out — using original text", error=True)
        self._add_history(ts, original, success=True)

        copied = True
        if self._chk_auto_copy.isChecked():
            copied = set_clipboard_text(original)
            if copied:
                self._log_ui("Copied original text to clipboard")
            else:
                self._log_ui("Failed to copy to clipboard", error=True)

        if copied and self._chk_auto_paste.isChecked():
            def _paste():
                simulate_paste(wait_for_modifiers=self._chk_hotkeys.isChecked())
            w = Worker(_paste)
            self._pool.start(w)

        QTimer.singleShot(
            STATE_RESET_IDLE_MS,
            lambda: self._set_dictation_state(DictationState.IDLE)
            if self._dictation_state in (DictationState.SUCCESS, DictationState.ERROR)
            else None,
        )

    @Slot(str)
    def _on_transcription_error(self, err: str) -> None:
        self._res_monitor.start()
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._set_dictation_state(DictationState.ERROR)
        self._log_ui(f"Transcription error: {err}", error=True)
        self._add_history(ts, f"Error: {err}", success=False)
        QTimer.singleShot(
            STATE_RESET_ERROR_MS,
            lambda: self._set_dictation_state(DictationState.IDLE)
            if self._dictation_state in (DictationState.SUCCESS, DictationState.ERROR)
            else None,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # HISTORY
    # ═════════════════════════════════════════════════════════════════════════

    def _add_history(
        self,
        timestamp: str,
        text: str,
        success: bool,
        original_text: Optional[str] = None,
    ) -> None:
        entry = _HistoryEntry(
            timestamp, text, success, parent=self._history_widget,
            original_text=original_text,
        )
        count = self._history_layout.count()
        self._history_layout.insertWidget(max(0, count - 1), entry)

    # ═════════════════════════════════════════════════════════════════════════
    # CLEAR LOGS & HISTORY
    # ═════════════════════════════════════════════════════════════════════════

    @Slot()
    def _on_clear_history(self) -> None:
        """Clear the in-memory transcription history."""
        while self._history_layout.count() > 1:
            item = self._history_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._log_ui("History cleared")

    @Slot()
    def _on_clear_logs(self) -> None:
        """Clear the UI log panel and on-disk log files."""
        self._log_text.clear()
        self._delete_log_files()
        self._log_ui("Logs cleared")

    @Slot()
    def _on_copy_logs(self) -> None:
        """Copy all visible log text to the clipboard."""
        text = self._log_text.toPlainText()
        if text:
            if set_clipboard_text(text):
                self._log_ui("Logs copied to clipboard")
            else:
                self._log_ui("Failed to copy logs to clipboard", error=True)
        else:
            self._log_ui("No log text to copy")

    def _delete_log_files(self) -> None:
        """Remove the rotating log files from disk."""
        log_dir = DEFAULT_LOG_DIR
        for pattern in ("cv2t.log", "cv2t.log.*"):
            for f in log_dir.glob(pattern):
                try:
                    f.unlink()
                except OSError:
                    pass

    # ═════════════════════════════════════════════════════════════════════════
    # SETTINGS
    # ═════════════════════════════════════════════════════════════════════════

    @Slot()
    def _on_open_settings(self) -> None:
        from .settings_dialog import SettingsDialog

        old_engine = self.settings.engine
        old_model_path = self.settings.model_path
        old_device = self.settings.device

        dlg = SettingsDialog(self.settings, parent=self)
        if dlg.exec() == SettingsDialog.DialogCode.Accepted:
            self._apply_settings()

            # If engine, model path, or device changed, prompt to reload
            if (
                self.settings.engine != old_engine
                or self.settings.model_path != old_model_path
                or self.settings.device != old_device
            ):
                # Switching between CTranslate2-based and PyTorch-based engines
                # requires a full process restart (CUDA runtimes conflict).
                engine_family_changed = (
                    self.settings.engine != old_engine
                    and self._engines_need_restart(old_engine, self.settings.engine)
                )
                if engine_family_changed:
                    reply = QMessageBox.question(
                        self,
                        "Restart Required",
                        f"Switching from {old_engine} to {self.settings.engine} "
                        "requires restarting CV2T (incompatible CUDA runtimes).\n\n"
                        "Restart now?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes,
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self._restart_application()
                    return

                reply = QMessageBox.question(
                    self,
                    "Reload Model?",
                    "Engine or model path changed. Reload model now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    # If engine changed, swap engine instance
                    if self.settings.engine != old_engine:
                        engine_cls = ENGINES.get(self.settings.engine)
                        if engine_cls:
                            self._engine.unload()
                            self._engine = engine_cls()
                            self._lbl_engine.setText(f"Engine: {self._engine.name}")
                    self._on_reload_model()

    @staticmethod
    def _engines_need_restart(old_engine: str, new_engine: str) -> bool:
        """Return True if switching between these engines requires a restart."""
        old_is_ct2 = old_engine in _CTRANSLATE2_ENGINES
        new_is_ct2 = new_engine in _CTRANSLATE2_ENGINES
        return old_is_ct2 != new_is_ct2

    def _restart_application(self) -> None:
        """Save state, release the single-instance mutex, and re-exec."""
        import subprocess

        from .__main__ import release_single_instance_mutex

        self._log_ui("Restarting CV2T for engine switch…")
        log.info("Restarting CV2T for engine switch")

        # Graceful shutdown without accepting close (we re-launch ourselves)
        self._loading_timer.stop()
        self._res_monitor.stop()
        self._hotkey_mgr.unregister()
        self._recorder.close_stream()
        self._engine.unload()

        release_single_instance_mutex()

        # Re-launch, skipping the engine prompt since the user already
        # chose the engine in settings.
        if getattr(sys, "frozen", False):
            # PyInstaller frozen build — sys.executable is the .exe itself
            subprocess.Popen(
                [sys.executable, "--skip-engine-prompt"],
                cwd=os.environ.get("CV2T_HOME", r"C:\Program Files\CV2T"),
            )
        else:
            # Source install — use python -m cv2t
            subprocess.Popen(
                [sys.executable, "-m", "cv2t", "--skip-engine-prompt"],
                cwd=os.environ.get("CV2T_HOME", r"C:\Program Files\CV2T"),
            )

        # Exit current process
        from PySide6.QtWidgets import QApplication
        QApplication.instance().quit()

    def _apply_settings(self) -> None:
        """Re-apply changed settings to live components."""
        s = self.settings

        # Audio (need to re-open stream if device changed)
        new_dev = s.mic_device_index if s.mic_device_index >= 0 else None
        if new_dev != self._recorder.device:
            self._recorder.close_stream()
            self._recorder.device = new_dev
            try:
                self._recorder.open_stream()
                self._log_ui("Microphone stream re-opened")
            except Exception as exc:
                self._log_ui(f"Microphone error: {exc}", error=True)
        self._recorder.sample_rate = s.sample_rate
        self._recorder.silence_threshold = s.silence_threshold
        self._recorder.silence_margin = int(s.sample_rate * s.silence_margin_ms / 1000)

        # Hotkeys
        if s.hotkeys_enabled:
            self._hotkey_mgr.register(s.hotkey_start, s.hotkey_stop, s.hotkey_quit)
        else:
            self._hotkey_mgr.unregister()
        self._chk_hotkeys.setChecked(s.hotkeys_enabled)
        self._chk_auto_copy.setChecked(s.auto_copy)
        self._chk_auto_paste.setChecked(s.auto_paste)

        # Professional Mode
        self._active_preset = self._pro_presets.get(s.pro_active_preset)
        if s.professional_mode and self._api_key and self._active_preset:
            model = self._active_preset.model or "gpt-5.4-mini"
            self._text_processor = TextProcessor(
                api_key=self._api_key, model=model,
            )
            self._log_ui("Professional Mode enabled")
        else:
            self._text_processor = None
            if s.professional_mode and not self._api_key:
                self._log_ui(
                    "Professional Mode enabled but no API key configured",
                    error=True,
                )

        self._log_ui("Settings applied")
        self._update_global_status()

    @Slot()
    def _on_open_pro_settings(self) -> None:
        """Open the Professional Mode settings dialog."""
        from .pro_settings_dialog import ProSettingsDialog

        dlg = ProSettingsDialog(
            settings=self.settings,
            presets=self._pro_presets,
            presets_dir=DEFAULT_PRESETS_DIR,
            parent=self,
            api_key=self._api_key,
        )
        if dlg.exec() == ProSettingsDialog.DialogCode.Accepted:
            self._api_key = dlg.api_key
            self._pro_presets = dlg.presets
            self._active_preset = self._pro_presets.get(
                self.settings.pro_active_preset,
            )

            # Re-create or destroy TextProcessor based on new state
            if self.settings.professional_mode and self._api_key and self._active_preset:
                model = self._active_preset.model or "gpt-5.4-mini"
                self._text_processor = TextProcessor(
                    api_key=self._api_key, model=model,
                )
                self._log_ui("Professional Mode enabled")
            else:
                self._text_processor = None
                if self.settings.professional_mode and not self._api_key:
                    self._log_ui(
                        "Professional Mode enabled but no API key configured",
                        error=True,
                    )

            self._chk_professional.blockSignals(True)
            self._chk_professional.setChecked(self.settings.professional_mode)
            self._chk_professional.blockSignals(False)
            self._update_global_status()

    def _on_professional_toggled(self, checked: bool) -> None:
        """Handle the Professional Mode checkbox in the main toggle row."""
        if checked:
            if not self._api_key:
                self._chk_professional.blockSignals(True)
                self._chk_professional.setChecked(False)
                self._chk_professional.blockSignals(False)
                reply = QMessageBox.question(
                    self,
                    "API Key Required",
                    "Professional Mode requires an OpenAI API key.\n\n"
                    "Would you like to open Professional Mode Settings "
                    "to configure one?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._on_open_pro_settings()
                return
            if self._active_preset is None:
                self._chk_professional.blockSignals(True)
                self._chk_professional.setChecked(False)
                self._chk_professional.blockSignals(False)
                reply = QMessageBox.question(
                    self,
                    "No Preset Configured",
                    "Professional Mode requires an active preset.\n\n"
                    "Would you like to open Professional Mode Settings "
                    "to configure one?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._on_open_pro_settings()
                return
            # All prerequisites met — enable
            self.settings.professional_mode = True
            model = self._active_preset.model or "gpt-5.4-mini"
            self._text_processor = TextProcessor(
                api_key=self._api_key, model=model,
            )
            self._log_ui("Professional Mode enabled")
        else:
            self.settings.professional_mode = False
            self._text_processor = None
            self._log_ui("Professional Mode disabled")
        self.settings.save()
        self._update_global_status()

    # ═════════════════════════════════════════════════════════════════════════
    # SLEEP / WAKE RECOVERY
    # ═════════════════════════════════════════════════════════════════════════

    def nativeEvent(self, event_type, message):
        """Intercept Windows power-management broadcasts."""
        if event_type == b"windows_generic_MSG":
            try:
                import ctypes.wintypes

                msg = ctypes.wintypes.MSG.from_address(int(message))
                if msg.message == WM_POWERBROADCAST and msg.wParam in (
                    PBT_APMRESUMEAUTOMATIC,
                    PBT_APMRESUMESUSPEND,
                ):
                    now = time.time()
                    if now - self._last_resume_time > SYSTEM_RESUME_DEBOUNCE_S:
                        self._last_resume_time = now
                        QTimer.singleShot(SYSTEM_RESUME_DELAY_MS, self._on_system_resume)
            except Exception:
                log.debug("nativeEvent parsing failed", exc_info=True)
        return super().nativeEvent(event_type, message)

    def _on_system_resume(self) -> None:
        """Re-register hotkeys and re-open the mic stream after sleep/wake."""
        log.info("System resume from sleep detected")
        self._log_ui("System resume detected — re-registering hotkeys")

        if self._chk_hotkeys.isChecked():
            self._hotkey_mgr.re_register()

        try:
            self._recorder.close_stream()
            self._recorder.open_stream()
            self._log_ui("Microphone stream re-opened after resume")
        except Exception as exc:
            self._log_ui(f"Microphone error after resume: {exc}", error=True)

    # ═════════════════════════════════════════════════════════════════════════
    # CLEANUP
    # ═════════════════════════════════════════════════════════════════════════

    def closeEvent(self, event) -> None:
        """Graceful shutdown."""
        self._log_ui("Shutting down…")
        self._loading_timer.stop()
        self._res_monitor.stop()
        self._hotkey_mgr.unregister()
        self._recorder.close_stream()
        self._engine.unload()
        # Wait for any in-flight thread-pool workers (transcription, model
        # load, metrics poll) to finish so the process can exit cleanly.
        self._pool.waitForDone(5000)
        if self.settings.clear_logs_on_exit:
            self._delete_log_files()
        event.accept()
