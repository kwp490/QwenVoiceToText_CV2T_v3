"""Tests for ProPreset dataclass and preset management."""

import json
import tempfile
import unittest
from pathlib import Path

from cv2t.pro_preset import (
    BUILTIN_PRESET_NAMES,
    ProPreset,
    _safe_filename,
    bootstrap_presets,
    delete_preset,
    get_builtin_presets,
    load_all_presets,
    save_preset,
)


class ProPresetSerializationTests(unittest.TestCase):
    """Round-trip JSON serialization of ProPreset."""

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.json"
            preset = ProPreset(
                name="My Preset",
                system_prompt="Be formal.",
                fix_tone=False,
                fix_grammar=True,
                fix_punctuation=False,
                vocabulary="API, gRPC, OAuth2",
                model="gpt-5.4-nano",
            )
            preset.save(path)
            loaded = ProPreset.load(path)

            self.assertEqual(loaded.name, "My Preset")
            self.assertEqual(loaded.system_prompt, "Be formal.")
            self.assertFalse(loaded.fix_tone)
            self.assertTrue(loaded.fix_grammar)
            self.assertFalse(loaded.fix_punctuation)
            self.assertEqual(loaded.vocabulary, "API, gRPC, OAuth2")
            self.assertEqual(loaded.model, "gpt-5.4-nano")

    def test_load_ignores_unknown_fields(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.json"
            path.write_text(
                json.dumps({"name": "Test", "unknown_field": "ignore"}),
                encoding="utf-8",
            )
            loaded = ProPreset.load(path)
            self.assertEqual(loaded.name, "Test")
            self.assertFalse(hasattr(loaded, "unknown_field"))

    def test_validate_empty_name(self):
        preset = ProPreset(name="")
        preset.validate()
        self.assertEqual(preset.name, "Untitled Preset")

    def test_validate_empty_model(self):
        preset = ProPreset(model="")
        preset.validate()
        self.assertEqual(preset.model, "gpt-5.4-mini")


class BuiltinPresetsTests(unittest.TestCase):
    """Tests for built-in preset availability."""

    def test_builtin_presets_exist(self):
        presets = get_builtin_presets()
        self.assertGreater(len(presets), 0)

    def test_expected_builtin_names(self):
        expected = {
            "General Professional",
            "Technical / Engineering",
            "Casual / Friendly",
            "Email / Correspondence",
            "Simplified (8th Grade)",
        }
        self.assertEqual(BUILTIN_PRESET_NAMES, expected)

    def test_builtin_presets_are_copies(self):
        presets1 = get_builtin_presets()
        presets2 = get_builtin_presets()
        # Modifying one dict should not affect the other
        presets1["General Professional"].fix_tone = False
        self.assertTrue(presets2["General Professional"].fix_tone)


class PresetManagerTests(unittest.TestCase):
    """Tests for load_all_presets, save_preset, delete_preset."""

    def test_load_all_includes_builtins(self):
        with tempfile.TemporaryDirectory() as td:
            presets = load_all_presets(Path(td))
            for name in BUILTIN_PRESET_NAMES:
                self.assertIn(name, presets)

    def test_save_and_load_user_preset(self):
        with tempfile.TemporaryDirectory() as td:
            presets_dir = Path(td)
            user_preset = ProPreset(name="Custom", system_prompt="Test")
            save_preset(user_preset, presets_dir)

            presets = load_all_presets(presets_dir)
            self.assertIn("Custom", presets)
            self.assertEqual(presets["Custom"].system_prompt, "Test")

    def test_user_preset_overrides_builtin(self):
        with tempfile.TemporaryDirectory() as td:
            presets_dir = Path(td)
            override = ProPreset(
                name="General Professional",
                system_prompt="Custom override",
            )
            save_preset(override, presets_dir)

            presets = load_all_presets(presets_dir)
            self.assertEqual(
                presets["General Professional"].system_prompt, "Custom override"
            )

    def test_delete_user_preset(self):
        with tempfile.TemporaryDirectory() as td:
            presets_dir = Path(td)
            user_preset = ProPreset(name="ToDelete", system_prompt="Bye")
            save_preset(user_preset, presets_dir)

            result = delete_preset("ToDelete", presets_dir)
            self.assertTrue(result)

            presets = load_all_presets(presets_dir)
            self.assertNotIn("ToDelete", presets)

    def test_cannot_delete_builtin(self):
        with tempfile.TemporaryDirectory() as td:
            result = delete_preset("General Professional", Path(td))
            self.assertFalse(result)

    def test_delete_nonexistent_returns_false(self):
        with tempfile.TemporaryDirectory() as td:
            result = delete_preset("DoesNotExist", Path(td))
            self.assertFalse(result)

    def test_corrupt_preset_file_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            presets_dir = Path(td)
            bad = presets_dir / "bad.json"
            bad.write_text("not valid json {{{", encoding="utf-8")

            presets = load_all_presets(presets_dir)
            # Should still have builtins, bad file skipped
            for name in BUILTIN_PRESET_NAMES:
                self.assertIn(name, presets)


class BootstrapPresetsTests(unittest.TestCase):
    """Tests for bootstrap_presets directory creation."""

    def test_bootstrap_creates_directory_and_files(self):
        with tempfile.TemporaryDirectory() as td:
            presets_dir = Path(td) / "presets"
            bootstrap_presets(presets_dir)

            self.assertTrue(presets_dir.is_dir())
            files = list(presets_dir.glob("*.json"))
            self.assertEqual(len(files), len(BUILTIN_PRESET_NAMES))

    def test_bootstrap_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            presets_dir = Path(td) / "presets"
            bootstrap_presets(presets_dir)
            bootstrap_presets(presets_dir)  # second call should not fail

            files = list(presets_dir.glob("*.json"))
            self.assertEqual(len(files), len(BUILTIN_PRESET_NAMES))


class SafeFilenameTests(unittest.TestCase):
    """Tests for _safe_filename sanitisation."""

    def test_normal_name(self):
        self.assertEqual(_safe_filename("General Professional"), "General Professional")

    def test_slashes_replaced(self):
        result = _safe_filename("Technical / Engineering")
        self.assertNotIn("/", result)
        self.assertNotIn("\\", result)

    def test_special_chars_replaced(self):
        result = _safe_filename('a<b>c:d"e|f?g*h')
        for ch in '<>:"|?*':
            self.assertNotIn(ch, result)

    def test_empty_returns_preset(self):
        self.assertEqual(_safe_filename(""), "preset")

    def test_whitespace_only_returns_preset(self):
        self.assertEqual(_safe_filename("   "), "preset")


class PresetNameCollisionTests(unittest.TestCase):
    """Tests for edge cases in preset naming and file management."""

    def test_preset_with_slash_in_name_saves_correctly(self):
        """Names with special chars (e.g., 'Technical / Engineering')
        must round-trip through save/load correctly."""
        with tempfile.TemporaryDirectory() as td:
            presets_dir = Path(td)
            preset = ProPreset(
                name="Technical / Engineering", system_prompt="Be precise."
            )
            save_preset(preset, presets_dir)
            presets = load_all_presets(presets_dir)
            self.assertIn("Technical / Engineering", presets)
            self.assertEqual(
                presets["Technical / Engineering"].system_prompt, "Be precise."
            )

    def test_multiple_presets_with_similar_names(self):
        """Distinct presets with similar names must coexist."""
        with tempfile.TemporaryDirectory() as td:
            presets_dir = Path(td)
            for name in ["Test A", "Test B", "Test C"]:
                save_preset(ProPreset(name=name), presets_dir)

            presets = load_all_presets(presets_dir)
            for name in ["Test A", "Test B", "Test C"]:
                self.assertIn(name, presets)


if __name__ == "__main__":
    unittest.main()
