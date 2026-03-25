"""Periodic GPU / RAM metrics polling via Qt signals."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from .gpu_monitor import get_system_metrics
from .workers import Worker

log = logging.getLogger(__name__)


class ResourceMonitor(QObject):
    """Polls system metrics on a thread pool and emits results via signals.

    Signals
    -------
    metrics_updated(object)
        Emitted with a ``SystemMetrics`` dataclass on each successful poll.
    metrics_error(str)
        Emitted when the poll worker raises an exception.
    """

    metrics_updated = Signal(object)
    metrics_error = Signal(str)

    def __init__(self, pool, interval_ms: int = 5000, parent=None) -> None:
        super().__init__(parent)
        self._pool = pool
        self._in_flight = False

        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._poll)

    # ── Public control ───────────────────────────────────────────────────

    def start(self) -> None:
        """Start the periodic polling timer."""
        self._timer.start()

    def stop(self) -> None:
        """Stop the periodic polling timer."""
        self._timer.stop()

    @property
    def is_in_flight(self) -> bool:
        """True if a metrics worker is currently running."""
        return self._in_flight

    # ── Internal ─────────────────────────────────────────────────────────

    @Slot()
    def _poll(self) -> None:
        if self._in_flight:
            return
        self._in_flight = True
        worker = Worker(get_system_metrics)
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(self._on_error)
        self._pool.start(worker)

    @Slot(object)
    def _on_result(self, metrics) -> None:
        self._in_flight = False
        self.metrics_updated.emit(metrics)

    @Slot(str)
    def _on_error(self, err: str) -> None:
        self._in_flight = False
        log.error("Metrics worker error: %s", err)
        self.metrics_error.emit(err)
