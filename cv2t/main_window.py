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
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from .audio import AudioRecorder, play_beep
from .clipboard import set_clipboard_text, simulate_paste
from .config import Settings
from .engine import ENGINES
from .gpu_monitor import get_system_metrics
from .hotkeys import HotkeyManager
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

        display = text if len(text) <= 120 else text[:117] + "…"
        text_label = QLabel(display)
        text_label.setWordWrap(True)
        text_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        copy_btn = QPushButton("Copy")
        copy_btn.setFixedWidth(50)
        copy_btn.clicked.connect(self._copy)

        row.addWidget(time_label)
        row.addWidget(status_label)
        row.addWidget(text_label)
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
                raise RuntimeError(
                    f"Engine '{settings.engine}' not available. "
                    f"Installed engines: {available}"
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
        self._metrics_poll_in_flight: bool = False
        self._last_resume_time: float = 0.0

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

        # ── Model Engine panel ───────────────────────────────────────────────
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

        # Resource metrics row
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

        btn_row = QHBoxLayout()
        self._btn_reload = QPushButton("Reload Model")
        self._btn_reload.clicked.connect(self._on_reload_model)
        self._btn_validate = QPushButton("Validate")
        self._btn_validate.clicked.connect(self._on_validate)
        self._btn_diagnostics = QPushButton("Copy Diagnostics")
        self._btn_diagnostics.clicked.connect(self._on_copy_diagnostics)
        btn_row.addWidget(self._btn_reload)
        btn_row.addWidget(self._btn_validate)
        btn_row.addWidget(self._btn_diagnostics)
        btn_row.addStretch()
        eg_layout.addLayout(btn_row)

        engine_group.setLayout(eg_layout)
        root.addWidget(engine_group)

        # ── Dictation panel ──────────────────────────────────────────────────
        dict_group = QGroupBox("Dictation")
        dg_layout = QVBoxLayout()

        self._lbl_dictation_state = QLabel("State: Idle")
        self._lbl_dictation_state.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        dg_layout.addWidget(self._lbl_dictation_state)

        btn_row2 = QHBoxLayout()
        self._btn_start = QPushButton("\U0001f3a4  Start Recording  (Ctrl+Alt+P)")
        self._btn_start.setMinimumHeight(38)
        self._btn_start.clicked.connect(self._on_start_recording)
        self._btn_stop = QPushButton("\u23f9  Stop && Transcribe  (Ctrl+Alt+L)")
        self._btn_stop.setMinimumHeight(38)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop_and_transcribe)
        btn_row2.addWidget(self._btn_start)
        btn_row2.addWidget(self._btn_stop)
        dg_layout.addLayout(btn_row2)

        toggle_row = QHBoxLayout()
        self._chk_auto_copy = QCheckBox("Auto-copy to clipboard")
        self._chk_auto_copy.setChecked(self.settings.auto_copy)
        self._chk_auto_paste = QCheckBox("Auto-paste (Ctrl+V)")
        self._chk_auto_paste.setChecked(self.settings.auto_paste)
        self._chk_hotkeys = QCheckBox("Enable global hotkeys")
        self._chk_hotkeys.setChecked(self.settings.hotkeys_enabled)
        self._chk_hotkeys.toggled.connect(self._on_hotkeys_toggled)
        toggle_row.addWidget(self._chk_auto_copy)
        toggle_row.addWidget(self._chk_auto_paste)
        toggle_row.addWidget(self._chk_hotkeys)
        toggle_row.addStretch()
        dg_layout.addLayout(toggle_row)

        dict_group.setLayout(dg_layout)
        root.addWidget(dict_group)

        # ── History panel ────────────────────────────────────────────────────
        hist_group = QGroupBox("Transcription History")
        hg_layout = QVBoxLayout()

        self._history_widget = QWidget()
        self._history_layout = QVBoxLayout(self._history_widget)
        self._history_layout.setContentsMargins(0, 0, 0, 0)
        self._history_layout.setSpacing(2)
        self._history_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._history_widget)
        scroll.setMinimumHeight(120)
        hg_layout.addWidget(scroll)

        hist_group.setLayout(hg_layout)

        # ── Log panel ────────────────────────────────────────────────────────
        log_group = QGroupBox("Log")
        lg_layout = QVBoxLayout()
        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumBlockCount(500)
        self._log_text.setFont(QFont("Consolas", 9))
        lg_layout.addWidget(self._log_text)
        log_group.setLayout(lg_layout)

        # ── Splitter for history + log ───────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(hist_group)
        splitter.addWidget(log_group)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        # ── Bottom buttons ───────────────────────────────────────────────────
        bottom_row = QHBoxLayout()
        btn_settings = QPushButton("\u2699  Settings")
        btn_settings.clicked.connect(self._on_open_settings)
        btn_clear = QPushButton("\U0001f5d1  Clear Logs && History")
        btn_clear.clicked.connect(self._on_clear_logs_and_history)
        btn_quit = QPushButton("Quit")
        btn_quit.clicked.connect(self.close)
        bottom_row.addWidget(btn_settings)
        bottom_row.addWidget(btn_clear)
        bottom_row.addStretch()
        bottom_row.addWidget(btn_quit)
        root.addLayout(bottom_row)

        # ── Status bar ───────────────────────────────────────────────────────
        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage("Starting up…")

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
        self._loading_timer.setInterval(1000)

        # Resource-metrics timer
        self._metrics_timer = QTimer(self)
        self._metrics_timer.timeout.connect(self._poll_metrics)
        self._metrics_timer.start(5000)

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
            ModelStatus.READY: "#2e7d32",
            ModelStatus.VALIDATED: "#1b5e20",
            ModelStatus.LOADING: "#f57f17",
            ModelStatus.NOT_LOADED: "#757575",
            ModelStatus.VALIDATING: "#1565c0",
            ModelStatus.ERROR: "#c62828",
        }
        color = color_map.get(status, "#757575")
        self._lbl_model_status.setText(
            f'Status: <span style="color:{color}"><b>{status.value}</b></span>'
        )
        self.statusBar().showMessage(f"Model: {status.value}")
        self._refresh_dictation_buttons()

    def _load_model(self) -> None:
        """Begin model loading on a worker thread."""
        self._set_model_status(ModelStatus.LOADING)
        self._model_load_start = time.time()
        self._loading_timer.start()
        self._log_ui(f"Loading {self._engine.name} model…")

        def _do_load():
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
                f'Status: <span style="color:#f57f17"><b>Loading… {elapsed}s</b></span>'
            )

    @Slot()
    def _on_reload_model(self) -> None:
        """Unload then reload the model."""
        self._log_ui("Reloading model…")

        def _do_reload():
            self._engine.unload()
            self._engine.load(self.settings.model_path, self.settings.device)

        self._set_model_status(ModelStatus.LOADING)
        self._model_load_start = time.time()
        self._loading_timer.start()

        worker = Worker(_do_reload)
        worker.signals.result.connect(self._on_model_loaded)
        worker.signals.error.connect(self._on_model_load_error)
        self._pool.start(worker)

    # ── Resource metrics ──────────────────────────────────────────────────────

    def _poll_metrics(self) -> None:
        """Periodic resource usage poll — runs on thread pool."""
        if self._metrics_poll_in_flight:
            return
        self._metrics_poll_in_flight = True

        worker = Worker(get_system_metrics)
        worker.signals.result.connect(self._on_metrics_result)
        worker.signals.error.connect(self._on_metrics_error)
        self._pool.start(worker)

    @Slot(str)
    def _on_metrics_error(self, err: str) -> None:
        self._metrics_poll_in_flight = False
        log.error("Metrics worker error: %s", err)

    @Slot(object)
    def _on_metrics_result(self, metrics) -> None:
        self._metrics_poll_in_flight = False
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
                color = "#c62828"
            elif pct > 75:
                color = "#f57f17"
            else:
                color = "#2e7d32"
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

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @Slot()
    def _on_copy_diagnostics(self) -> None:
        def _collect():
            import sys
            metrics = get_system_metrics()
            lines = [
                "=== CV2T Diagnostics ===",
                f"Engine: {self._engine.name}",
                f"Model status: {self._model_status.value}",
                f"Model path: {self.settings.model_path}",
                f"Device: {self.settings.device}",
                f"Python: {sys.version}",
                f"GPU: {metrics.gpu.name or 'N/A'}",
                f"VRAM: {metrics.gpu.vram_used_gb:.1f}/{metrics.gpu.vram_total_gb:.1f} GB",
                f"RAM: {metrics.ram_used_gb:.1f}/{metrics.ram_total_gb:.1f} GB",
            ]
            # Package versions
            try:
                import cv2t
                lines.append(f"CV2T version: {cv2t.__version__}")
            except Exception:
                pass
            try:
                import PySide6
                lines.append(f"PySide6: {PySide6.__version__}")
            except Exception:
                pass
            return "\n".join(lines)

        worker = Worker(_collect)
        worker.signals.result.connect(self._on_diagnostics_ready)
        self._pool.start(worker)

    @Slot(object)
    def _on_diagnostics_ready(self, diag: str) -> None:
        set_clipboard_text(str(diag))
        self._log_ui("Diagnostics copied to clipboard")

    # ═════════════════════════════════════════════════════════════════════════
    # DICTATION
    # ═════════════════════════════════════════════════════════════════════════

    def _set_dictation_state(self, state: DictationState) -> None:
        self._dictation_state = state
        color_map = {
            DictationState.IDLE: "#424242",
            DictationState.RECORDING: "#c62828",
            DictationState.PROCESSING: "#f57f17",
            DictationState.SUCCESS: "#2e7d32",
            DictationState.ERROR: "#c62828",
        }
        color = color_map.get(state, "#424242")
        self._lbl_dictation_state.setText(
            f'State: <span style="color:{color}"><b>{state.value}</b></span>'
        )
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
        self._metrics_timer.stop()

        # Wait for any in-flight metrics poll to finish before dispatching
        # the transcription worker (avoids NVML / CUDA overlap).
        import time as _time
        _deadline = _time.monotonic() + 2.0
        while self._metrics_poll_in_flight and _time.monotonic() < _deadline:
            from PySide6.QtCore import QCoreApplication
            QCoreApplication.processEvents()
            _time.sleep(0.05)

        # Get raw audio (fast, on main thread)
        audio = self._recorder.get_raw_audio()
        if audio is None:
            self._log_ui("No audio recorded", error=True)
            self._metrics_timer.start()
            self._set_dictation_state(DictationState.IDLE)
            return

        self._log_ui(f"Captured {len(audio)/self.settings.sample_rate:.1f}s of audio")

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
            text = self._engine.transcribe(trimmed, self.settings.sample_rate)
            return text

        worker = Worker(_process)
        worker.signals.result.connect(self._on_transcription_result)
        worker.signals.error.connect(self._on_transcription_error)
        self._pool.start(worker)

    @Slot(object)
    def _on_transcription_result(self, text: str) -> None:
        """Handle transcription result — runs on MAIN THREAD (safe for clipboard)."""
        self._metrics_timer.start()
        text = str(text).strip()
        ts = datetime.datetime.now().strftime("%H:%M:%S")

        if text:
            self._set_dictation_state(DictationState.SUCCESS)
            self._log_ui(f"Transcribed: {text}")
            self._add_history(ts, text, success=True)

            if self._chk_auto_copy.isChecked():
                set_clipboard_text(text)  # MAIN THREAD — safe
                self._log_ui("Copied to clipboard")

            if self._chk_auto_paste.isChecked():
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
            1500,
            lambda: self._set_dictation_state(DictationState.IDLE)
            if self._dictation_state in (DictationState.SUCCESS, DictationState.ERROR)
            else None,
        )

    @Slot(str)
    def _on_transcription_error(self, err: str) -> None:
        self._metrics_timer.start()
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._set_dictation_state(DictationState.ERROR)
        self._log_ui(f"Transcription error: {err}", error=True)
        self._add_history(ts, f"Error: {err}", success=False)
        QTimer.singleShot(
            2000,
            lambda: self._set_dictation_state(DictationState.IDLE)
            if self._dictation_state in (DictationState.SUCCESS, DictationState.ERROR)
            else None,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # HISTORY
    # ═════════════════════════════════════════════════════════════════════════

    def _add_history(self, timestamp: str, text: str, success: bool) -> None:
        entry = _HistoryEntry(timestamp, text, success, parent=self._history_widget)
        count = self._history_layout.count()
        self._history_layout.insertWidget(max(0, count - 1), entry)

    # ═════════════════════════════════════════════════════════════════════════
    # CLEAR LOGS & HISTORY
    # ═════════════════════════════════════════════════════════════════════════

    @Slot()
    def _on_clear_logs_and_history(self) -> None:
        """Clear the in-memory history, UI log panel, and on-disk log files."""
        while self._history_layout.count() > 1:
            item = self._history_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._log_text.clear()
        self._delete_log_files()
        self._log_ui("Logs and history cleared")

    def _delete_log_files(self) -> None:
        """Remove the rotating log files from disk."""
        log_dir = Path(
            os.environ.get("APPDATA", str(Path.home()))
        ) / "CV2T"
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

        dlg = SettingsDialog(self.settings, parent=self)
        if dlg.exec() == SettingsDialog.DialogCode.Accepted:
            self._apply_settings()

            # If engine or model path changed, prompt to reload
            if self.settings.engine != old_engine or self.settings.model_path != old_model_path:
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
        self._metrics_timer.stop()
        self._hotkey_mgr.unregister()
        self._recorder.close_stream()
        self._engine.unload()

        release_single_instance_mutex()

        # Re-launch via the same executable
        subprocess.Popen(
            [sys.executable, "-m", "cv2t"],
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

        self._log_ui("Settings applied")

    # ═════════════════════════════════════════════════════════════════════════
    # SLEEP / WAKE RECOVERY
    # ═════════════════════════════════════════════════════════════════════════

    def nativeEvent(self, event_type, message):
        """Intercept Windows power-management broadcasts."""
        WM_POWERBROADCAST = 0x0218
        PBT_APMRESUMEAUTOMATIC = 0x0012
        PBT_APMRESUMESUSPEND = 0x0007

        if event_type == b"windows_generic_MSG":
            try:
                import ctypes.wintypes

                msg = ctypes.wintypes.MSG.from_address(int(message))
                if msg.message == WM_POWERBROADCAST and msg.wParam in (
                    PBT_APMRESUMEAUTOMATIC,
                    PBT_APMRESUMESUSPEND,
                ):
                    now = time.time()
                    if now - self._last_resume_time > 10:
                        self._last_resume_time = now
                        QTimer.singleShot(2000, self._on_system_resume)
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
        self._metrics_timer.stop()
        self._hotkey_mgr.unregister()
        self._recorder.close_stream()
        self._engine.unload()
        if self.settings.clear_logs_on_exit:
            self._delete_log_files()
        event.accept()
