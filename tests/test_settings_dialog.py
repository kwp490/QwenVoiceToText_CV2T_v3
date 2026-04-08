"""Tests for SettingsDialog after Professional Mode extraction.

Verifies:
  - Professional Mode section has been removed from SettingsDialog.
  - No pro_presets, api_key params or properties remain.
  - Core settings (engine, audio, UX) still function.
"""

import ast
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SETTINGS_DIALOG_PATH = _REPO_ROOT / "cv2t" / "settings_dialog.py"


def _qt_available() -> bool:
    try:
        from PySide6.QtWidgets import QApplication
        return True
    except ImportError:
        return False


# ═════════════════════════════════════════════════════════════════════════════
# Structural (AST) tests
# ═════════════════════════════════════════════════════════════════════════════


class TestSettingsDialogProModeRemoved(unittest.TestCase):
    """AST-level checks confirming Professional Mode is NOT in SettingsDialog."""

    @classmethod
    def setUpClass(cls):
        cls._source = _SETTINGS_DIALOG_PATH.read_text(encoding="utf-8")
        cls._tree = ast.parse(cls._source, filename="settings_dialog.py")
        cls._sd_class = None
        for node in ast.walk(cls._tree):
            if isinstance(node, ast.ClassDef) and node.name == "SettingsDialog":
                cls._sd_class = node
                break
        assert cls._sd_class is not None

    def _get_method_source(self, method_name: str) -> str:
        for node in ast.walk(self._sd_class):
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                return ast.get_source_segment(self._source, node) or ""
        self.fail(f"Method '{method_name}' not found in SettingsDialog")

    def _get_method_names(self):
        return [
            n.name for n in ast.walk(self._sd_class)
            if isinstance(n, ast.FunctionDef)
        ]

    def test_no_professional_mode_groupbox_in_build_ui(self):
        """_build_ui must NOT contain a 'Professional Mode' QGroupBox."""
        src = self._get_method_source("_build_ui")
        self.assertNotIn("Professional Mode", src)

    def test_no_pro_enabled_checkbox(self):
        """_build_ui must NOT create self._pro_enabled."""
        src = self._get_method_source("_build_ui")
        self.assertNotIn("_pro_enabled", src)

    def test_no_pro_preset_combo(self):
        """_build_ui must NOT create self._pro_preset_combo."""
        src = self._get_method_source("_build_ui")
        self.assertNotIn("_pro_preset_combo", src)

    def test_no_on_open_pro_settings_method(self):
        """SettingsDialog must NOT have _on_open_pro_settings."""
        self.assertNotIn("_on_open_pro_settings", self._get_method_names())

    def test_no_refresh_preset_combo_method(self):
        """SettingsDialog must NOT have _refresh_preset_combo."""
        self.assertNotIn("_refresh_preset_combo", self._get_method_names())

    def test_no_pro_presets_property(self):
        """SettingsDialog must NOT expose pro_presets property."""
        self.assertNotIn("pro_presets", self._source)

    def test_no_api_key_property(self):
        """SettingsDialog must NOT expose api_key property."""
        # api_key can appear in comments; check for the property definition
        self.assertNotIn("def api_key", self._source)

    def test_save_does_not_write_professional_mode(self):
        """_save_and_accept must NOT write s.professional_mode."""
        src = self._get_method_source("_save_and_accept")
        self.assertNotIn("professional_mode", src)

    def test_save_does_not_write_pro_active_preset(self):
        """_save_and_accept must NOT write s.pro_active_preset."""
        src = self._get_method_source("_save_and_accept")
        self.assertNotIn("pro_active_preset", src)

    def test_init_no_pro_presets_kwarg(self):
        """__init__ must NOT accept pro_presets parameter."""
        src = self._get_method_source("__init__")
        self.assertNotIn("pro_presets", src)

    def test_init_no_api_key_kwarg(self):
        """__init__ must NOT accept api_key parameter."""
        src = self._get_method_source("__init__")
        self.assertNotIn("api_key", src)

    # ── Core settings still present ──────────────────────────────────────

    def test_engine_group_in_build_ui(self):
        """_build_ui must still contain Model Engine group."""
        src = self._get_method_source("_build_ui")
        self.assertIn("Model Engine", src)

    def test_audio_group_in_build_ui(self):
        """_build_ui must still contain Audio group."""
        src = self._get_method_source("_build_ui")
        self.assertIn("Audio", src)

    def test_ux_group_in_build_ui(self):
        """_build_ui must still contain Dictation UX group."""
        src = self._get_method_source("_build_ui")
        self.assertIn("Dictation UX", src)


# ═════════════════════════════════════════════════════════════════════════════
# Live widget tests — require PySide6
# ═════════════════════════════════════════════════════════════════════════════


@unittest.skipUnless(_qt_available(), "PySide6 not available")
class TestSettingsDialogLive(unittest.TestCase):
    """Integration tests for SettingsDialog without pro mode."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication([])

    def _make_dialog(self):
        from cv2t.config import Settings
        from cv2t.settings_dialog import SettingsDialog

        settings = Settings()
        dlg = SettingsDialog(settings)
        return dlg, settings

    def test_dialog_opens_without_pro_params(self):
        """SettingsDialog must open without pro_presets or api_key."""
        dlg, _ = self._make_dialog()
        try:
            self.assertIsNotNone(dlg)
        finally:
            dlg.close()

    def test_no_pro_enabled_attr(self):
        """SettingsDialog must not have _pro_enabled attribute."""
        dlg, _ = self._make_dialog()
        try:
            self.assertFalse(hasattr(dlg, "_pro_enabled"))
        finally:
            dlg.close()

    def test_no_pro_preset_combo_attr(self):
        """SettingsDialog must not have _pro_preset_combo attribute."""
        dlg, _ = self._make_dialog()
        try:
            self.assertFalse(hasattr(dlg, "_pro_preset_combo"))
        finally:
            dlg.close()
