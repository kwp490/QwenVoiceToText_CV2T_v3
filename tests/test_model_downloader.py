"""Tests for the urllib-based model downloader (cv2t.model_downloader)."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cv2t.model_downloader import (
    WHISPER_MODEL_ID,
    WHISPER_OPTIONAL_FILES,
    WHISPER_REPO_ID,
    WHISPER_REQUIRED_FILES,
    _hf_file_url,
    download_file,
    download_whisper_model,
    is_whisper_model,
    whisper_model_ready,
)


class TestConstants(unittest.TestCase):
    """Whisper model constants must be consistent and non-empty."""

    def test_repo_id_format(self):
        self.assertIn("/", WHISPER_REPO_ID)
        self.assertFalse(WHISPER_REPO_ID.startswith("http"))

    def test_required_files_non_empty(self):
        self.assertTrue(len(WHISPER_REQUIRED_FILES) >= 3)

    def test_model_id_is_string(self):
        self.assertIsInstance(WHISPER_MODEL_ID, str)
        self.assertTrue(len(WHISPER_MODEL_ID) > 0)


class TestHfFileUrl(unittest.TestCase):
    """URL construction for HuggingFace CDN."""

    def test_default_revision(self):
        url = _hf_file_url("org/model", "config.json")
        self.assertEqual(url, "https://huggingface.co/org/model/resolve/main/config.json")

    def test_custom_revision(self):
        url = _hf_file_url("org/model", "model.bin", revision="v1.0")
        self.assertEqual(url, "https://huggingface.co/org/model/resolve/v1.0/model.bin")

    def test_whisper_urls(self):
        for fname in WHISPER_REQUIRED_FILES:
            url = _hf_file_url(WHISPER_REPO_ID, fname)
            self.assertIn(WHISPER_REPO_ID, url)
            self.assertIn(fname, url)
            self.assertTrue(url.startswith("https://"))


class TestIsWhisperModel(unittest.TestCase):
    """Validation of Whisper model directories."""

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(is_whisper_model(d))

    def test_valid_whisper_config(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {"model_type": "whisper"}
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f)
            self.assertTrue(is_whisper_model(d))

    def test_non_whisper_model_type(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {"model_type": "bert"}
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f)
            self.assertFalse(is_whisper_model(d))

    def test_no_model_type_is_valid(self):
        """CTranslate2 conversions may lack model_type — should be valid."""
        with tempfile.TemporaryDirectory() as d:
            cfg = {}
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f)
            self.assertTrue(is_whisper_model(d))

    def test_invalid_json(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                f.write("not json")
            self.assertFalse(is_whisper_model(d))


class TestWhisperModelReady(unittest.TestCase):
    """Full model readiness check."""

    def test_all_files_present(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {"model_type": "whisper"}
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f)
            for fname in WHISPER_REQUIRED_FILES:
                if fname != "config.json":
                    with open(os.path.join(d, fname), "w") as f:
                        f.write("dummy")
            self.assertTrue(whisper_model_ready(d))

    def test_missing_required_file(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {"model_type": "whisper"}
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f)
            # Only config.json, missing model.bin and tokenizer.json
            self.assertFalse(whisper_model_ready(d))


class TestDownloadFile(unittest.TestCase):
    """File download with urllib — mocked network."""

    def test_skip_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "existing.json")
            with open(dest, "w") as f:
                f.write('{"x":1}')
            result = download_file("https://example.com/x", dest)
            self.assertTrue(result)

    @patch("cv2t.model_downloader.urllib.request.urlopen")
    def test_successful_download(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "5"}
        mock_resp.read.side_effect = [b"hello", b""]
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "test.txt")
            result = download_file("https://example.com/test.txt", dest)
            self.assertTrue(result)
            with open(dest) as f:
                self.assertEqual(f.read(), "hello")

    @patch("cv2t.model_downloader.urllib.request.urlopen")
    def test_404_returns_false(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://example.com/missing", 404, "Not Found", {}, None
        )
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "missing.txt")
            result = download_file("https://example.com/missing", dest)
            self.assertFalse(result)
            self.assertFalse(os.path.exists(dest))


class TestDownloadWhisperModel(unittest.TestCase):
    """Integration-style test for the full Whisper download flow (mocked)."""

    def test_skips_when_already_present(self):
        with tempfile.TemporaryDirectory() as d:
            # Create a complete model directory
            cfg = {"model_type": "whisper"}
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f)
            for fname in WHISPER_REQUIRED_FILES:
                if fname != "config.json":
                    with open(os.path.join(d, fname), "w") as f:
                        f.write("dummy")
            result = download_whisper_model(d)
            self.assertEqual(result, 0)

    @patch("cv2t.model_downloader.download_file")
    def test_downloads_required_and_optional_files(self, mock_download):
        mock_download.return_value = True

        with tempfile.TemporaryDirectory() as d:
            # download_file always returns True but doesn't create files,
            # so whisper_model_ready will fail at the end — that's expected
            result = download_whisper_model(d)
            # Should have been called for required + optional files
            total_calls = len(WHISPER_REQUIRED_FILES) + len(WHISPER_OPTIONAL_FILES)
            self.assertEqual(mock_download.call_count, total_calls)


if __name__ == "__main__":
    unittest.main()
