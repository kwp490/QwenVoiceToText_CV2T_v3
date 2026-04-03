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
        # Professional Mode defaults
        self.assertFalse(s.professional_mode)
        self.assertTrue(s.pro_fix_tone)
        self.assertTrue(s.pro_fix_grammar)
        self.assertTrue(s.pro_fix_punctuation)
        self.assertEqual(s.pro_model, "gpt-5.4-mini")
        self.assertFalse(s.store_api_key)

    def test_load_missing_file_returns_defaults(self):
        loaded = Settings.load(Path("/nonexistent/path/settings.json"))
        self.assertEqual(loaded.engine, "whisper")

    def test_load_corrupt_json_returns_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "settings.json"
            config_path.write_text("not valid json {{{", encoding="utf-8")
            loaded = Settings.load(config_path)
            self.assertEqual(loaded.engine, "whisper")

    def test_professional_mode_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "settings.json"

            settings = Settings(
                professional_mode=True,
                pro_fix_tone=False,
                pro_fix_grammar=True,
                pro_fix_punctuation=False,
                pro_model="gpt-5.4-nano",
                store_api_key=True,
            )
            settings.save(config_path)
            loaded = Settings.load(config_path)

            self.assertTrue(loaded.professional_mode)
            self.assertFalse(loaded.pro_fix_tone)
            self.assertTrue(loaded.pro_fix_grammar)
            self.assertFalse(loaded.pro_fix_punctuation)
            self.assertEqual(loaded.pro_model, "gpt-5.4-nano")
            self.assertTrue(loaded.store_api_key)

    def test_api_key_never_in_settings_json(self):
        """The actual API key value must NEVER be serialised into settings.json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "settings.json"
            settings = Settings(professional_mode=True, store_api_key=True)
            settings.save(config_path)
            raw = config_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            # store_api_key (bool preference) is allowed, but there must be
            # no field that could hold an actual API key string value.
            self.assertNotIn("openai_api_key", data)
            self.assertNotIn("api_key", data)        # no bare 'api_key' field
            self.assertNotIn("openai", raw.lower().replace("store_api_key", ""))
            # Verify no string value looks like an API key
            for v in data.values():
                if isinstance(v, str):
                    self.assertFalse(
                        v.startswith("sk-"),
                        "A value resembling an API key was found in settings JSON",
                    )


if __name__ == "__main__":
    unittest.main()


class SettingsValidationTests(unittest.TestCase):
    """Tests for Settings.validate() correcting invalid values."""

    def test_invalid_engine_falls_back_to_whisper(self):
        s = Settings(engine="nonexistent")
        s.validate()
        self.assertEqual(s.engine, "whisper")

    def test_valid_engines_accepted(self):
        for engine in ("whisper", "canary"):
            s = Settings(engine=engine)
            s.validate()
            self.assertEqual(s.engine, engine)

    def test_invalid_device_falls_back_to_cuda(self):
        s = Settings(device="gpu")
        s.validate()
        self.assertEqual(s.device, "cuda")

    def test_valid_devices_accepted(self):
        for device in ("cuda", "cpu"):
            s = Settings(device=device)
            s.validate()
            self.assertEqual(s.device, device)

    def test_sample_rate_too_low_reset(self):
        s = Settings(sample_rate=100)
        s.validate()
        self.assertEqual(s.sample_rate, 16000)

    def test_sample_rate_too_high_reset(self):
        s = Settings(sample_rate=96000)
        s.validate()
        self.assertEqual(s.sample_rate, 16000)

    def test_valid_sample_rate_unchanged(self):
        for sr in (8000, 16000, 44100, 48000):
            s = Settings(sample_rate=sr)
            s.validate()
            self.assertEqual(s.sample_rate, sr)

    def test_inference_timeout_too_low_reset(self):
        s = Settings(inference_timeout=0)
        s.validate()
        self.assertEqual(s.inference_timeout, 30)

    def test_silence_threshold_zero_reset(self):
        s = Settings(silence_threshold=0)
        s.validate()
        self.assertEqual(s.silence_threshold, 0.0015)

    def test_load_invalid_engine_in_json_corrected(self):
        """Settings loaded from JSON with invalid engine should be corrected."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "settings.json"
            p.write_text(json.dumps({"engine": "invalid_eng"}), encoding="utf-8")
            s = Settings.load(p)
            self.assertEqual(s.engine, "whisper")
