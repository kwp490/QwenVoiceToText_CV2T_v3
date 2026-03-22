import json
import tempfile
import unittest
from pathlib import Path

from cv2t.config import Settings


class SettingsConfigTests(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "settings.json"

            settings = Settings(
                engine="canary",
                model_path="/tmp/models",
                device="cpu",
                auto_copy=False,
            )
            settings.save(config_path)

            loaded = Settings.load(config_path)

            self.assertEqual(loaded.engine, "canary")
            self.assertEqual(loaded.model_path, "/tmp/models")
            self.assertEqual(loaded.device, "cpu")
            self.assertFalse(loaded.auto_copy)

    def test_load_ignores_unknown_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "settings.json"
            config_path.write_text(
                json.dumps(
                    {
                        "engine": "whisper",
                        "language": "en",
                        "unexpected": "ignore-me",
                    }
                ),
                encoding="utf-8",
            )

            loaded = Settings.load(config_path)

            self.assertEqual(loaded.engine, "whisper")
            self.assertEqual(loaded.language, "en")
            self.assertFalse(hasattr(loaded, "unexpected"))

    def test_defaults(self):
        s = Settings()
        self.assertEqual(s.engine, "whisper")
        self.assertEqual(s.device, "cuda")
        self.assertEqual(s.sample_rate, 16000)
        self.assertTrue(s.auto_copy)
        self.assertTrue(s.auto_paste)
        self.assertTrue(s.hotkeys_enabled)

    def test_load_missing_file_returns_defaults(self):
        loaded = Settings.load(Path("/nonexistent/path/settings.json"))
        self.assertEqual(loaded.engine, "whisper")

    def test_load_corrupt_json_returns_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "settings.json"
            config_path.write_text("not valid json {{{", encoding="utf-8")
            loaded = Settings.load(config_path)
            self.assertEqual(loaded.engine, "whisper")


if __name__ == "__main__":
    unittest.main()
