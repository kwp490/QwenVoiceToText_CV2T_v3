"""
QThreadPool-based workers for background operations.

Each heavy operation (model inference, GPU monitoring, etc.)
runs on a pooled thread and communicates results back to the Qt main thread
via ``WorkerSignals``.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

log = logging.getLogger(__name__)


class WorkerSignals(QObject):
    """Signals emitted by ``Worker`` instances."""

    finished = Signal()
    error = Signal(str)
    result = Signal(object)


class Worker(QRunnable):
    """Generic runnable that executes *fn* on a ``QThreadPool``.

    Usage::

        worker = Worker(some_blocking_fn, arg1, arg2, kw=val)
        worker.signals.result.connect(handle_result)
        worker.signals.error.connect(handle_error)
        QThreadPool.globalInstance().start(worker)
    """

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:
            tb = traceback.format_exc()
            log.error("Worker error: %s\n%s", exc, tb)
            self.signals.error.emit(str(exc))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
