"""Tests for the refactored MainWindow layout.

Verifies:
  Phase 1 – Transcription section is dominant; buttons enlarged, no Dictation GroupBox.
  Phase 2 – Diagnostics panel collapsed by default; toggling shows/hides.
  Phase 3 – Status indicators placed between buttons and checkboxes (no QStatusBar).
  Phase 4 – Clear/Copy buttons are contextually placed in panel headers.
"""

import ast
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MAIN_WINDOW_PATH = _REPO_ROOT / "cv2t" / "main_window.py"


def _qt_available() -> bool:
    try:
        from PySide6.QtWidgets import QApplication
        return True
    except ImportError:
        return False


# ═════════════════════════════════════════════════════════════════════════════
# Structural (AST) tests — no Qt needed
# ═════════════════════════════════════════════════════════════════════════════


class TestLayoutStructure(unittest.TestCase):
    """AST-level checks on _build_ui layout changes."""

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

    # ── Phase 1 ──────────────────────────────────────────────────────────

    def test_no_dictation_groupbox(self):
        """The 'Dictation' QGroupBox must no longer appear in _build_ui."""
        src = self._get_method_source("_build_ui")
        self.assertNotIn('QGroupBox("Dictation")', src)

    def test_start_button_height_increased(self):
        """Start Recording button must have minimumHeight >= 50."""
        src = self._get_method_source("_build_ui")
        self.assertIn("setMinimumHeight(52)", src)

    def test_start_button_font_increased(self):
        """Start/Stop buttons must have increased font-size."""
        src = self._get_method_source("_build_ui")
        self.assertIn("font-size: 14px", src)

    def test_history_min_height_increased(self):
        """History scroll area must have minimumHeight >= 200."""
        src = self._get_method_source("_build_ui")
        self.assertIn("setMinimumHeight(200)", src)

    def test_no_lbl_dictation_state_in_build_ui(self):
        """_lbl_dictation_state must not be created in _build_ui."""
        src = self._get_method_source("_build_ui")
        self.assertNotIn("_lbl_dictation_state", src)

    # ── Phase 2 ──────────────────────────────────────────────────────────

    def test_diag_toggle_exists(self):
        """_build_ui must create self._diag_toggle."""
        src = self._get_method_source("_build_ui")
        self.assertIn("self._diag_toggle", src)

    def test_diag_content_exists(self):
        """_build_ui must create self._diag_content."""
        src = self._get_method_source("_build_ui")
        self.assertIn("self._diag_content", src)

    def test_diag_hidden_by_default(self):
        """Diagnostics content must be hidden by default."""
        src = self._get_method_source("_build_ui")
        self.assertIn("self._diag_content.setVisible(False)", src)

    def test_toggle_diagnostics_method_exists(self):
        """_toggle_diagnostics method must exist."""
        method_names = [
            n.name for n in ast.walk(self._mw_class)
            if isinstance(n, ast.FunctionDef)
        ]
        self.assertIn("_toggle_diagnostics", method_names)

    # ── Phase 3 ──────────────────────────────────────────────────────────

    def test_lbl_global_status_created(self):
        """_build_ui must create self._lbl_global_status."""
        src = self._get_method_source("_build_ui")
        self.assertIn("self._lbl_global_status", src)

    def test_no_status_bar_in_build_ui(self):
        """QStatusBar must not be used in _build_ui (status moved inline)."""
        src = self._get_method_source("_build_ui")
        self.assertNotIn("QStatusBar", src)
        self.assertNotIn("setStatusBar", src)

    def test_update_global_status_method_exists(self):
        """_update_global_status method must exist."""
        method_names = [
            n.name for n in ast.walk(self._mw_class)
            if isinstance(n, ast.FunctionDef)
        ]
        self.assertIn("_update_global_status", method_names)

    def test_set_model_status_calls_update_global(self):
        """_set_model_status must call _update_global_status."""
        src = self._get_method_source("_set_model_status")
        self.assertIn("_update_global_status", src)

    def test_set_dictation_state_calls_update_global(self):
        """_set_dictation_state must call _update_global_status."""
        src = self._get_method_source("_set_dictation_state")
        self.assertIn("_update_global_status", src)

    # ── Phase 4 ──────────────────────────────────────────────────────────

    def test_on_clear_history_method_exists(self):
        """_on_clear_history must be defined as a separate method."""
        method_names = [
            n.name for n in ast.walk(self._mw_class)
            if isinstance(n, ast.FunctionDef)
        ]
        self.assertIn("_on_clear_history", method_names)

    def test_on_clear_logs_method_exists(self):
        """_on_clear_logs must be defined as a separate method."""
        method_names = [
            n.name for n in ast.walk(self._mw_class)
            if isinstance(n, ast.FunctionDef)
        ]
        self.assertIn("_on_clear_logs", method_names)

    def test_old_clear_method_removed(self):
        """_on_clear_logs_and_history must no longer exist."""
        method_names = [
            n.name for n in ast.walk(self._mw_class)
            if isinstance(n, ast.FunctionDef)
        ]
        self.assertNotIn("_on_clear_logs_and_history", method_names)

    def test_clear_history_button_in_build_ui(self):
        """_btn_clear_history must be created in _build_ui."""
        src = self._get_method_source("_build_ui")
        self.assertIn("_btn_clear_history", src)

    def test_clear_logs_button_in_build_ui(self):
        """_btn_clear_logs must be created in _build_ui."""
        src = self._get_method_source("_build_ui")
        self.assertIn("_btn_clear_logs", src)

    def test_copy_logs_button_in_build_ui(self):
        """_btn_copy_logs must be created in _build_ui."""
        src = self._get_method_source("_build_ui")
        self.assertIn("_btn_copy_logs", src)

    # ── Phase 5 (professional mode button in footer) ───────────────────

    def test_no_pro_toggle_in_build_ui(self):
        """PRO toggle button must not be in _build_ui (was removed long ago)."""
        src = self._get_method_source("_build_ui")
        self.assertNotIn("_btn_pro_toggle", src)

    def test_no_combo_preset_in_build_ui(self):
        """Preset combo must not be in _build_ui (moved to Pro Settings dialog)."""
        src = self._get_method_source("_build_ui")
        self.assertNotIn("_combo_preset", src)

    def test_no_pro_toggle_method(self):
        """_on_pro_toggle must not be defined in MainWindow."""
        method_names = [
            n.name for n in ast.walk(self._mw_class)
            if isinstance(n, ast.FunctionDef)
        ]
        self.assertNotIn("_on_pro_toggle", method_names)

    def test_pro_settings_button_in_build_ui(self):
        """Professional Mode Settings button must be in _build_ui footer."""
        src = self._get_method_source("_build_ui")
        self.assertIn("Professional Mode Settings", src)

    def test_on_open_pro_settings_method_exists(self):
        """_on_open_pro_settings must be defined in MainWindow."""
        method_names = [
            n.name for n in ast.walk(self._mw_class)
            if isinstance(n, ast.FunctionDef)
        ]
        self.assertIn("_on_open_pro_settings", method_names)

    def test_no_refresh_preset_combo_method(self):
        """_refresh_preset_combo must not be in MainWindow (moved to ProSettingsDialog)."""
        method_names = [
            n.name for n in ast.walk(self._mw_class)
            if isinstance(n, ast.FunctionDef)
        ]
        self.assertNotIn("_refresh_preset_combo", method_names)

    def test_update_global_status_includes_professional(self):
        """_update_global_status must include Professional mode status."""
        src = self._get_method_source("_update_global_status")
        self.assertIn("Professional", src)

    # ── Phase 6 (professional mode quick-toggle checkbox) ────────────

    def test_chk_professional_in_build_ui(self):
        """_chk_professional checkbox must be created in _build_ui."""
        src = self._get_method_source("_build_ui")
        self.assertIn("self._chk_professional", src)
        self.assertIn('QCheckBox("Professional Mode")', src)

    def test_chk_professional_connected_to_toggled(self):
        """_chk_professional must be connected to _on_professional_toggled."""
        src = self._get_method_source("_build_ui")
        self.assertIn("_chk_professional.toggled.connect(self._on_professional_toggled)", src)

    def test_on_professional_toggled_method_exists(self):
        """_on_professional_toggled must be defined in MainWindow."""
        method_names = [
            n.name for n in ast.walk(self._mw_class)
            if isinstance(n, ast.FunctionDef)
        ]
        self.assertIn("_on_professional_toggled", method_names)

    def test_on_professional_toggled_handles_no_api_key(self):
        """_on_professional_toggled must handle the no-API-key case."""
        src = self._get_method_source("_on_professional_toggled")
        self.assertIn("API Key Required", src)

    def test_on_professional_toggled_handles_no_preset(self):
        """_on_professional_toggled must handle the no-preset case."""
        src = self._get_method_source("_on_professional_toggled")
        self.assertIn("No Preset Configured", src)

    def test_pro_settings_syncs_checkbox(self):
        """_on_open_pro_settings must sync _chk_professional after dialog."""
        src = self._get_method_source("_on_open_pro_settings")
        self.assertIn("_chk_professional.setChecked", src)
        self.assertIn("_chk_professional.blockSignals", src)


# ═════════════════════════════════════════════════════════════════════════════
# Live widget tests — require PySide6
# ═════════════════════════════════════════════════════════════════════════════


@unittest.skipUnless(_qt_available(), "PySide6 not available")
class TestDiagnosticsToggleLive(unittest.TestCase):
    """Integration tests for diagnostics collapse/expand with a real QApp."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication, QMessageBox
        cls._app = QApplication.instance() or QApplication([])
        # Make QMessageBox available to all test methods in this class
        globals()["QMessageBox"] = QMessageBox

    def _make_window(self):
        """Create a MainWindow with mocked engine for testing."""
        from unittest.mock import MagicMock, PropertyMock
        from cv2t.config import Settings
        import tempfile

        settings = Settings()
        settings.hotkeys_enabled = False

        engine = MagicMock()
        engine.name = "mock"
        type(engine).is_loaded = PropertyMock(return_value=False)

        from cv2t.main_window import MainWindow
        win = MainWindow(settings, engine=engine)
        return win

    def test_diagnostics_hidden_by_default(self):
        """Diagnostics content must be hidden on construction."""
        win = self._make_window()
        try:
            self.assertTrue(win._diag_content.isHidden())
        finally:
            win.close()

    def test_toggle_shows_diagnostics(self):
        """Clicking the toggle must show diagnostics content."""
        win = self._make_window()
        try:
            win._diag_toggle.click()
            self.assertFalse(win._diag_content.isHidden())
        finally:
            win.close()

    def test_toggle_twice_hides_diagnostics(self):
        """Two clicks must hide diagnostics again."""
        win = self._make_window()
        try:
            win._diag_toggle.click()
            win._diag_toggle.click()
            self.assertTrue(win._diag_content.isHidden())
        finally:
            win.close()

    def test_log_captured_while_hidden(self):
        """Log text must accumulate even when diagnostics is collapsed."""
        win = self._make_window()
        try:
            self.assertFalse(win._diag_content.isVisible())
            win._log_text.appendPlainText("test log message")
            self.assertIn("test log message", win._log_text.toPlainText())
        finally:
            win.close()

    def test_global_status_label_exists(self):
        """The inline status label must exist and contain state info."""
        win = self._make_window()
        try:
            text = win._lbl_global_status.text()
            self.assertIn("Model:", text)
            self.assertIn("Dictation:", text)
        finally:
            win.close()

    def test_start_button_height(self):
        """Start button must have minimum height >= 50."""
        win = self._make_window()
        try:
            self.assertGreaterEqual(win._btn_start.minimumHeight(), 50)
        finally:
            win.close()

    def test_stop_button_height(self):
        """Stop button must have minimum height >= 50."""
        win = self._make_window()
        try:
            self.assertGreaterEqual(win._btn_stop.minimumHeight(), 50)
        finally:
            win.close()

    # ── Phase 6 (professional mode quick-toggle live tests) ──────────

    def test_chk_professional_exists(self):
        """Professional Mode checkbox must exist on the main window."""
        win = self._make_window()
        try:
            self.assertIsNotNone(win._chk_professional)
            self.assertEqual(win._chk_professional.text(), "Professional Mode")
        finally:
            win.close()

    def test_chk_professional_default_unchecked(self):
        """Professional Mode checkbox defaults to unchecked (pro mode off)."""
        win = self._make_window()
        try:
            self.assertFalse(win._chk_professional.isChecked())
        finally:
            win.close()

    def test_chk_professional_reflects_settings(self):
        """Professional Mode checkbox reflects settings.professional_mode."""
        from unittest.mock import MagicMock, PropertyMock, patch
        from cv2t.config import Settings
        from cv2t.main_window import MainWindow

        settings = Settings()
        settings.professional_mode = True
        settings.hotkeys_enabled = False

        engine = MagicMock()
        engine.name = "mock"
        type(engine).is_loaded = PropertyMock(return_value=False)

        win = MainWindow(settings, engine=engine)
        try:
            self.assertTrue(win._chk_professional.isChecked())
        finally:
            win.close()

    def test_toggle_on_without_api_key_reverts(self):
        """Enabling Professional Mode without an API key must revert the checkbox."""
        from unittest.mock import patch
        win = self._make_window()
        try:
            win._api_key = ""
            # Patch QMessageBox to auto-click No (cancel)
            with patch(
                "cv2t.main_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.No,
            ):
                win._chk_professional.setChecked(True)
            self.assertFalse(win._chk_professional.isChecked())
            self.assertFalse(win.settings.professional_mode)
        finally:
            win.close()

    def test_toggle_on_with_api_key_enables(self):
        """Enabling Professional Mode with API key and preset must succeed."""
        from unittest.mock import MagicMock
        from cv2t.pro_preset import ProPreset

        win = self._make_window()
        try:
            win._api_key = "sk-test-key"
            win._active_preset = ProPreset(name="Test")
            win._chk_professional.setChecked(True)
            self.assertTrue(win.settings.professional_mode)
            self.assertIsNotNone(win._text_processor)
        finally:
            win.close()

    def test_toggle_off_disables(self):
        """Disabling Professional Mode must clear TextProcessor."""
        from unittest.mock import MagicMock
        from cv2t.pro_preset import ProPreset

        win = self._make_window()
        try:
            # First enable
            win._api_key = "sk-test-key"
            win._active_preset = ProPreset(name="Test")
            win._chk_professional.setChecked(True)
            self.assertIsNotNone(win._text_processor)
            # Then disable
            win._chk_professional.setChecked(False)
            self.assertFalse(win.settings.professional_mode)
            self.assertIsNone(win._text_processor)
        finally:
            win.close()
