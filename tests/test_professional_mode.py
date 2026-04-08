"""Tests for Professional Mode worker lifetime and signal delivery.

These tests catch the class of bugs where:
  - The Worker is garbage-collected before its signals are delivered.
  - Lambda signal connections prevent QObject connection tracking.
  - Auto-delete races with Python-side references.
  - Missing safety timeout leaves the UI frozen if signal delivery fails.

The structural tests (AST-based) run without a Qt event loop and verify
the source code itself.  The integration tests use a real QApplication +
QThreadPool to confirm end-to-end signal delivery.
"""

import ast
import sys
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MAIN_WINDOW_PATH = _REPO_ROOT / "cv2t" / "main_window.py"


# ═════════════════════════════════════════════════════════════════════════════
# Structural tests — verify the source code meets safety invariants
# ═════════════════════════════════════════════════════════════════════════════


class TestProModeWorkerLifetimeInvariants(unittest.TestCase):
    """Source-level checks for professional mode worker lifetime safety.

    These detect regressions toward the original bug (lambda connections,
    missing stored reference, auto-delete not disabled).
    """

    @classmethod
    def setUpClass(cls):
        cls._source = _MAIN_WINDOW_PATH.read_text(encoding="utf-8")
        cls._tree = ast.parse(cls._source, filename="main_window.py")

        # Locate the MainWindow class node
        cls._mw_class = None
        for node in ast.walk(cls._tree):
            if isinstance(node, ast.ClassDef) and node.name == "MainWindow":
                cls._mw_class = node
                break
        assert cls._mw_class is not None, "MainWindow class not found"

    # ── 1. Worker must be stored on self ─────────────────────────────────

    def test_pro_worker_stored_as_instance_attr(self):
        """_pro_worker must be assigned in __init__ (not just a local)."""
        init_src = self._get_method_source("__init__")
        self.assertIn(
            "self._pro_worker",
            init_src,
            "MainWindow.__init__ must declare self._pro_worker",
        )

    def test_active_preset_stored_as_instance_attr(self):
        """_active_preset must be assigned in __init__."""
        init_src = self._get_method_source("__init__")
        self.assertIn(
            "self._active_preset",
            init_src,
            "MainWindow.__init__ must declare self._active_preset",
        )

    def test_pro_worker_assigned_before_pool_start(self):
        """self._pro_worker must be set BEFORE self._pool.start().

        Regression: If the worker is a local ``worker`` and only passed to
        pool.start(), it can be GC'd before signals are delivered.
        """
        result_src = self._get_method_source("_on_transcription_result")

        # The worker must be stored on self
        self.assertIn(
            "self._pro_worker",
            result_src,
            "_on_transcription_result must store the worker as self._pro_worker",
        )
        # Must appear before pool.start
        store_pos = result_src.find("self._pro_worker")
        start_pos = result_src.find("self._pool.start(self._pro_worker)")
        self.assertGreater(
            start_pos, store_pos,
            "self._pro_worker must be assigned before self._pool.start()",
        )

    # ── 2. No lambda signal connections ──────────────────────────────────

    def test_no_lambda_in_professional_signal_connections(self):
        """Professional mode signal connections must use bound methods, not lambdas.

        Regression: Lambda connections prevent QObject-to-QObject connection
        tracking.  Without a QObject receiver, Qt cannot prevent the sender's
        WorkerSignals from being GC'd, causing silent signal loss.
        """
        result_src = self._get_method_source("_on_transcription_result")
        lines = result_src.splitlines()

        for i, line in enumerate(lines):
            if ".signals.result.connect(" in line or ".signals.error.connect(" in line:
                # Check this line and the next few for lambda
                context = "\n".join(lines[i : i + 5])
                self.assertNotIn(
                    "lambda",
                    context,
                    f"Professional mode signal connection must not use lambda "
                    f"(line ~{i+1} of _on_transcription_result):\n{context}",
                )

    # ── 3. Auto-delete must be disabled ──────────────────────────────────

    def test_auto_delete_disabled_for_pro_worker(self):
        """The professional mode worker must call setAutoDelete(False).

        Regression: With autoDelete=True (the default), the C++ QRunnable
        is destroyed by QThreadPool immediately after run() returns —
        before the queued cross-thread signals reach the main event loop.
        """
        result_src = self._get_method_source("_on_transcription_result")
        self.assertIn(
            "setAutoDelete(False)",
            result_src,
            "Professional mode worker must disable auto-delete "
            "(setAutoDelete(False)) — we manage its lifetime via self._pro_worker",
        )

    # ── 4. Safety timeout must exist ─────────────────────────────────────

    def test_safety_timeout_created(self):
        """A safety timeout must be started when the cleanup worker is dispatched.

        This prevents permanent UI freeze if signal delivery fails.
        """
        result_src = self._get_method_source("_on_transcription_result")
        self.assertIn(
            "self._pro_timeout",
            result_src,
            "_on_transcription_result must create a safety timeout",
        )

    def test_timeout_handler_exists(self):
        """_on_professional_timeout must be defined as a method."""
        method_names = [
            node.name
            for node in ast.walk(self._mw_class)
            if isinstance(node, ast.FunctionDef)
        ]
        self.assertIn(
            "_on_professional_timeout",
            method_names,
            "MainWindow must define _on_professional_timeout() as a safety net",
        )

    # ── 5. Context stored on self (not captured by closure) ──────────────

    def test_pro_context_stored_on_self(self):
        """Transcription timestamp and text must be stored as self._pro_context.

        Regression: Passing these via lambda closure creates the same
        signal-tracking problem as lambda connections.
        """
        result_src = self._get_method_source("_on_transcription_result")
        self.assertIn(
            "self._pro_context",
            result_src,
            "_on_transcription_result must store (ts, text) as self._pro_context",
        )

    # ── 6. Handlers read context BEFORE clearing it ────────────────────

    def test_result_handler_reads_context_before_cancel(self):
        """_on_professional_result must read _pro_context BEFORE _cancel_pro_timeout.

        Regression: _cancel_pro_timeout() sets self._pro_context = None.
        If the handler calls cancel first and reads context second, ctx is
        always None and the handler silently returns without processing.
        """
        src = self._get_method_source("_on_professional_result")
        cancel_pos = src.find("_cancel_pro_timeout")
        context_read_pos = src.find("self._pro_context")
        self.assertGreater(
            cancel_pos, context_read_pos,
            "_on_professional_result must read self._pro_context BEFORE calling "
            "_cancel_pro_timeout() (which clears it)",
        )

    def test_error_handler_reads_context_before_cancel(self):
        """_on_professional_error must read _pro_context BEFORE _cancel_pro_timeout.

        Same regression as the result handler — cancel clears context.
        """
        src = self._get_method_source("_on_professional_error")
        cancel_pos = src.find("_cancel_pro_timeout")
        context_read_pos = src.find("self._pro_context")
        self.assertGreater(
            cancel_pos, context_read_pos,
            "_on_professional_error must read self._pro_context BEFORE calling "
            "_cancel_pro_timeout() (which clears it)",
        )

    def test_error_handler_clears_context(self):
        """_on_professional_error must clear _pro_context to prevent double-fire."""
        src = self._get_method_source("_on_professional_error")
        self.assertIn(
            "_cancel_pro_timeout",
            src,
            "_on_professional_error must call _cancel_pro_timeout()",
        )

    def test_finished_handler_clears_worker_ref(self):
        """_on_professional_finished must set self._pro_worker = None."""
        src = self._get_method_source("_on_professional_finished")
        self.assertIn(
            "self._pro_worker = None",
            src,
            "_on_professional_finished must clear the worker reference",
        )

    # ── 7. finished signal connected ─────────────────────────────────────

    def test_finished_signal_connected(self):
        """The worker's finished signal must be connected to _on_professional_finished."""
        result_src = self._get_method_source("_on_transcription_result")
        self.assertIn(
            ".signals.finished.connect(",
            result_src,
            "Professional mode worker must connect the finished signal "
            "to clear self._pro_worker",
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_method_source(self, method_name: str) -> str:
        """Extract the source text of a method from MainWindow."""
        for node in ast.walk(self._mw_class):
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                return ast.get_source_segment(self._source, node) or ""
        self.fail(f"Method '{method_name}' not found in MainWindow")


# ═════════════════════════════════════════════════════════════════════════════
# Integration tests — verify signal delivery with a real Qt environment
# ═════════════════════════════════════════════════════════════════════════════


def _qt_available() -> bool:
    """Check whether PySide6 is importable and a QApp can be created."""
    try:
        from PySide6.QtWidgets import QApplication
        return True
    except ImportError:
        return False


@unittest.skipUnless(_qt_available(), "PySide6 not available")
class TestProModeWorkerSignalDelivery(unittest.TestCase):
    """Live integration: Worker signals must reach the main thread reliably.

    This is a focused reproduction of the original bug where the Worker
    was GC'd before signals arrived.
    """

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication([])

    def test_worker_result_delivered_with_stored_ref(self):
        """Worker result signal MUST be delivered when we store a reference."""
        from PySide6.QtCore import QThreadPool, QTimer
        from cv2t.workers import Worker

        pool = QThreadPool.globalInstance()
        result_box = [None]
        finished = [False]

        def on_result(val):
            result_box[0] = val

        def on_finished():
            finished[0] = True

        worker = Worker(lambda: "cleaned text")
        worker.setAutoDelete(False)
        worker.signals.result.connect(on_result)
        worker.signals.finished.connect(on_finished)

        # Store reference (the fix)
        self._worker = worker
        pool.start(worker)

        # Process events until finished or 5s timeout
        import time
        deadline = time.monotonic() + 5.0
        while not finished[0] and time.monotonic() < deadline:
            self._app.processEvents()
            time.sleep(0.01)

        self.assertTrue(finished[0], "Worker finished signal was never delivered")
        self.assertEqual(result_box[0], "cleaned text",
                         "Worker result signal was never delivered")

        self._worker = None  # cleanup

    def test_worker_error_delivered_with_stored_ref(self):
        """Worker error signal MUST be delivered when we store a reference."""
        from PySide6.QtCore import QThreadPool
        from cv2t.workers import Worker

        pool = QThreadPool.globalInstance()
        error_box = [None]
        finished = [False]

        def on_error(err):
            error_box[0] = err

        def on_finished():
            finished[0] = True

        def _fail():
            raise ValueError("API timeout")

        worker = Worker(_fail)
        worker.setAutoDelete(False)
        worker.signals.error.connect(on_error)
        worker.signals.finished.connect(on_finished)

        self._worker = worker
        pool.start(worker)

        import time
        deadline = time.monotonic() + 5.0
        while not finished[0] and time.monotonic() < deadline:
            self._app.processEvents()
            time.sleep(0.01)

        self.assertTrue(finished[0], "Worker finished signal was never delivered")
        self.assertIn("API timeout", error_box[0],
                       "Worker error signal was never delivered")
        self._worker = None

    def test_rapid_sequential_workers_all_deliver(self):
        """10 rapid-fire workers must ALL deliver their result signals.

        Regression: intermittent GC of Workers caused some results to
        silently disappear under load (e.g. rapid dictation cycles).
        """
        from PySide6.QtCore import QThreadPool
        from cv2t.workers import Worker

        pool = QThreadPool.globalInstance()
        n = 10
        results = []
        finished_count = [0]

        def on_result(val):
            results.append(val)

        def on_finished():
            finished_count[0] += 1

        workers = []
        for i in range(n):
            w = Worker(lambda idx=i: f"result-{idx}")
            w.setAutoDelete(False)
            w.signals.result.connect(on_result)
            w.signals.finished.connect(on_finished)
            workers.append(w)
            pool.start(w)

        import time
        deadline = time.monotonic() + 10.0
        while finished_count[0] < n and time.monotonic() < deadline:
            self._app.processEvents()
            time.sleep(0.01)

        self.assertEqual(
            finished_count[0], n,
            f"Only {finished_count[0]}/{n} workers finished — "
            f"signals for {n - finished_count[0]} workers were silently lost",
        )
        self.assertEqual(
            len(results), n,
            f"Only {len(results)}/{n} results delivered — "
            f"{n - len(results)} result signals were silently lost",
        )
        workers.clear()


# ═════════════════════════════════════════════════════════════════════════════
# Structural tests — preset-based architecture invariants
# ═════════════════════════════════════════════════════════════════════════════


class TestPresetArchitectureInvariants(unittest.TestCase):
    """Verify the new preset-based professional mode architecture.

    These are AST/source-level checks that the refactoring
    preserved all safety invariants and correctly wired the
    new preset system.
    """

    @classmethod
    def setUpClass(cls):
        cls._source = _MAIN_WINDOW_PATH.read_text(encoding="utf-8")
        cls._tree = ast.parse(cls._source, filename="main_window.py")
        cls._mw_class = None
        for node in ast.walk(cls._tree):
            if isinstance(node, ast.ClassDef) and node.name == "MainWindow":
                cls._mw_class = node
                break
        assert cls._mw_class is not None

    def _get_method_source(self, method_name: str) -> str:
        for node in ast.walk(self._mw_class):
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                return ast.get_source_segment(self._source, node) or ""
        self.fail(f"Method '{method_name}' not found in MainWindow")

    # ── Init invariants ──────────────────────────────────────────────────

    def test_pro_presets_stored_in_init(self):
        """MainWindow.__init__ must declare self._pro_presets."""
        src = self._get_method_source("__init__")
        self.assertIn("self._pro_presets", src)

    def test_active_preset_stored_in_init(self):
        """MainWindow.__init__ must declare self._active_preset."""
        src = self._get_method_source("__init__")
        self.assertIn("self._active_preset", src)

    def test_bootstrap_presets_called_in_init(self):
        """Presets directory must be bootstrapped on startup."""
        src = self._get_method_source("__init__")
        self.assertIn("bootstrap_presets", src)

    def test_toggle_button_text_no_longer_in_build_ui(self):
        """PRO toggle button must NOT be in _build_ui (moved to Settings)."""
        src = self._get_method_source("_build_ui")
        self.assertNotIn("PRO: ON", src)
        self.assertNotIn("PRO: OFF", src)

    # ── Transcription uses preset ────────────────────────────────────────

    def test_transcription_uses_preset_kwarg(self):
        """_on_transcription_result must pass preset= to process()."""
        src = self._get_method_source("_on_transcription_result")
        self.assertIn("preset=", src,
                       "_on_transcription_result must pass a preset to process()")

    def test_transcription_checks_active_preset(self):
        """_on_transcription_result must check self._active_preset is not None."""
        src = self._get_method_source("_on_transcription_result")
        self.assertIn("self._active_preset is not None", src)

    def test_transcription_captures_preset_locally(self):
        """Preset must be captured as a local variable before worker dispatch.

        This prevents a race where the user changes presets mid-cleanup.
        """
        src = self._get_method_source("_on_transcription_result")
        self.assertIn("preset = self._active_preset", src)

    # ── No references to removed settings fields ─────────────────────────

    def test_no_pro_fix_tone_in_main_window(self):
        """Removed field pro_fix_tone must not appear in main_window.py."""
        self.assertNotIn("pro_fix_tone", self._source)

    def test_no_pro_fix_grammar_in_main_window(self):
        """Removed field pro_fix_grammar must not appear in main_window.py."""
        self.assertNotIn("pro_fix_grammar", self._source)

    def test_no_pro_fix_punctuation_in_main_window(self):
        """Removed field pro_fix_punctuation must not appear in main_window.py."""
        self.assertNotIn("pro_fix_punctuation", self._source)

    def test_no_pro_model_in_main_window(self):
        """Removed field settings.pro_model must not appear in main_window.py."""
        # pro_model can appear in preset context, but not as s.pro_model or settings.pro_model
        self.assertNotIn("settings.pro_model", self._source)
        self.assertNotIn("s.pro_model", self._source)

    # ── Professional Mode toggle ─────────────────────────────────────────

    def test_on_pro_toggle_removed_from_main_window(self):
        """_on_pro_toggle must no longer exist in MainWindow (moved to SettingsDialog)."""
        method_names = [
            n.name for n in ast.walk(self._mw_class)
            if isinstance(n, ast.FunctionDef)
        ]
        self.assertNotIn("_on_pro_toggle", method_names)

    def test_on_open_pro_settings_exists_in_main_window(self):
        """_on_open_pro_settings must exist in MainWindow (opens dedicated dialog)."""
        method_names = [
            n.name for n in ast.walk(self._mw_class)
            if isinstance(n, ast.FunctionDef)
        ]
        self.assertIn("_on_open_pro_settings", method_names)

    def test_refresh_preset_combo_removed_from_main_window(self):
        """_refresh_preset_combo must no longer exist in MainWindow."""
        method_names = [
            n.name for n in ast.walk(self._mw_class)
            if isinstance(n, ast.FunctionDef)
        ]
        self.assertNotIn("_refresh_preset_combo", method_names)

    def test_pro_toggle_saves_settings_in_pro_dialog(self):
        """ProSettingsDialog._save_and_accept must persist professional_mode."""
        pro_src = (_REPO_ROOT / "cv2t" / "pro_settings_dialog.py").read_text(encoding="utf-8")
        self.assertIn("professional_mode", pro_src)

    def test_preset_combo_changed_saves_settings_in_pro_dialog(self):
        """ProSettingsDialog._save_and_accept must persist pro_active_preset."""
        pro_src = (_REPO_ROOT / "cv2t" / "pro_settings_dialog.py").read_text(encoding="utf-8")
        self.assertIn("pro_active_preset", pro_src)

    # ── Settings dialog no longer has pro fields ─────────────────────────

    def test_settings_dialog_no_api_key_param(self):
        """SettingsDialog no longer accepts api_key parameter."""
        settings_src = (_REPO_ROOT / "cv2t" / "settings_dialog.py").read_text(encoding="utf-8")
        tree = ast.parse(settings_src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SettingsDialog":
                for item in ast.walk(node):
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        arg_names = [a.arg for a in item.args.args]
                        self.assertNotIn("api_key", arg_names,
                                         "SettingsDialog.__init__ must not accept api_key")
                        return
        self.fail("SettingsDialog.__init__ not found")

    def test_settings_dialog_no_professional_mode_section(self):
        """SettingsDialog must NOT contain Professional Mode section (moved to dedicated dialog)."""
        settings_src = (_REPO_ROOT / "cv2t" / "settings_dialog.py").read_text(encoding="utf-8")
        self.assertNotIn("Professional Mode", settings_src)
        self.assertNotIn("_pro_enabled", settings_src)
        self.assertNotIn("_pro_preset_combo", settings_src)


# ═════════════════════════════════════════════════════════════════════════════
# Backward compatibility — old settings.json with removed fields
# ═════════════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility(unittest.TestCase):
    """Verify old settings.json files with removed fields load gracefully."""

    def test_old_settings_with_pro_flags_loads_ok(self):
        """A settings.json containing removed pro_fix_* fields must load
        without error, silently ignoring the unknown keys."""
        import json
        import tempfile
        from cv2t.config import Settings

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "settings.json"
            old_data = {
                "engine": "whisper",
                "professional_mode": True,
                "pro_fix_tone": False,
                "pro_fix_grammar": True,
                "pro_fix_punctuation": False,
                "pro_model": "gpt-5.4-nano",
                "store_api_key": True,
            }
            path.write_text(json.dumps(old_data), encoding="utf-8")
            loaded = Settings.load(path)
            # Must load without error; removed fields silently dropped
            self.assertTrue(loaded.professional_mode)
            self.assertTrue(loaded.store_api_key)
            # New field gets its default
            self.assertEqual(loaded.pro_active_preset, "General Professional")
            # Removed fields must NOT be attributes
            self.assertFalse(hasattr(loaded, "pro_fix_tone"))
            self.assertFalse(hasattr(loaded, "pro_model"))

    def test_new_settings_round_trip_no_removed_fields(self):
        """Saved settings.json must not contain removed fields."""
        import json
        import tempfile
        from cv2t.config import Settings

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "settings.json"
            Settings().save(path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("pro_fix_tone", raw)
            self.assertNotIn("pro_fix_grammar", raw)
            self.assertNotIn("pro_fix_punctuation", raw)
            self.assertNotIn("pro_model", raw)
            self.assertIn("pro_active_preset", raw)


# ═════════════════════════════════════════════════════════════════════════════
# Text processor preset integration
# ═════════════════════════════════════════════════════════════════════════════


class TestTextProcessorPresetIntegration(unittest.TestCase):
    """Verify TextProcessor.process() correctly uses ProPreset objects."""

    def test_preset_model_overrides_constructor_model(self):
        """When a preset specifies a model, it should be used instead of
        the constructor model."""
        from unittest.mock import MagicMock
        from cv2t.text_processor import TextProcessor
        from cv2t.pro_preset import ProPreset

        proc = TextProcessor(api_key="sk-test", model="gpt-5.4-mini")
        proc._client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "cleaned"
        proc._client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        preset = ProPreset(name="Test", model="gpt-5.4-nano", fix_tone=True)
        proc.process("test", preset=preset)

        call_kwargs = proc._client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs.get("model"), "gpt-5.4-nano")

    def test_preset_empty_model_falls_back_to_constructor(self):
        """When preset.model is empty, fall back to the constructor model."""
        from unittest.mock import MagicMock
        from cv2t.text_processor import TextProcessor
        from cv2t.pro_preset import ProPreset

        proc = TextProcessor(api_key="sk-test", model="gpt-5.4-mini")
        proc._client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "cleaned"
        proc._client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        preset = ProPreset(name="Test", model="", fix_tone=True)
        proc.process("test", preset=preset)

        call_kwargs = proc._client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs.get("model"), "gpt-5.4-mini")

    def test_preset_flags_override_kwargs(self):
        """Preset flags must take priority over keyword arguments."""
        from unittest.mock import MagicMock
        from cv2t.text_processor import TextProcessor
        from cv2t.pro_preset import ProPreset

        proc = TextProcessor(api_key="sk-test", model="gpt-5.4-mini")
        proc._client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "cleaned"
        proc._client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        # kwargs say all on, but preset says all off with no custom prompt
        preset = ProPreset(
            name="Test", fix_tone=False, fix_grammar=False,
            fix_punctuation=False, system_prompt="",
        )
        result = proc.process(
            "test", fix_tone=True, fix_grammar=True,
            fix_punctuation=True, preset=preset,
        )
        # All flags off + no custom prompt → should return original
        self.assertEqual(result, "test")
        proc._client.chat.completions.create.assert_not_called()

    def test_vocabulary_preservation_in_prompt(self):
        """Vocabulary terms must appear in the system prompt sent to the API."""
        from unittest.mock import MagicMock
        from cv2t.text_processor import TextProcessor
        from cv2t.pro_preset import ProPreset

        proc = TextProcessor(api_key="sk-test", model="gpt-5.4-mini")
        proc._client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "cleaned"
        proc._client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        preset = ProPreset(
            name="Test", fix_tone=True, vocabulary="Kubernetes, gRPC\nOAuth2",
        )
        proc.process("test text", preset=preset)

        call_args = proc._client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages")
        system_content = messages[0]["content"]
        self.assertIn("Kubernetes", system_content)
        self.assertIn("gRPC", system_content)
        self.assertIn("OAuth2", system_content)

    def test_custom_system_prompt_in_api_call(self):
        """Custom system prompt from preset must appear in the API call."""
        from unittest.mock import MagicMock
        from cv2t.text_processor import TextProcessor
        from cv2t.pro_preset import ProPreset

        proc = TextProcessor(api_key="sk-test", model="gpt-5.4-mini")
        proc._client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "cleaned"
        proc._client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        preset = ProPreset(
            name="Legal", fix_tone=True,
            system_prompt="Rewrite in formal legal language.",
        )
        proc.process("test text", preset=preset)

        call_args = proc._client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages")
        system_content = messages[0]["content"]
        self.assertIn("formal legal language", system_content)
        # Default tone instruction should NOT appear
        self.assertNotIn("professional and neutral", system_content)

    def test_all_builtin_presets_generate_valid_prompts(self):
        """Every built-in preset must generate a non-empty system prompt."""
        from cv2t.text_processor import _build_system_prompt
        from cv2t.pro_preset import get_builtin_presets

        for name, preset in get_builtin_presets().items():
            prompt = _build_system_prompt(
                preset.fix_tone, preset.fix_grammar, preset.fix_punctuation,
                custom_prompt=preset.system_prompt,
                vocabulary=preset.vocabulary,
            )
            self.assertTrue(
                prompt.strip(),
                f"Built-in preset '{name}' generates an empty system prompt",
            )


if __name__ == "__main__":
    unittest.main()
