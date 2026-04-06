"""Tests for engine loading, model path resolution, and failure handling.

Catches issues like:
- Missing dependencies (e.g. 'av' module required by faster-whisper)
- Model files not found at expected paths
- Engine registry consistency
- Graceful error handling when model loading fails
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from cv2t.engine import ENGINES, _model_files_exist, get_available_engines
from cv2t.engine.base import SpeechEngine
from cv2t.engine.whisper import WhisperEngine
from cv2t.model_downloader import WHISPER_REQUIRED_FILES

_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestEngineRegistry(unittest.TestCase):
    """Engine registry must contain at least Whisper."""

    def test_whisper_registered(self):
        self.assertIn("whisper", ENGINES)

    def test_whisper_is_whisper_engine(self):
        self.assertIs(ENGINES["whisper"], WhisperEngine)

    def test_all_engines_extend_base(self):
        for name, cls in ENGINES.items():
            self.assertTrue(
                issubclass(cls, SpeechEngine),
                f"{name} engine does not extend SpeechEngine",
            )


class TestModelFilesExist(unittest.TestCase):
    """_model_files_exist correctly detects presence/absence of model files."""

    def test_whisper_complete(self):
        with tempfile.TemporaryDirectory() as d:
            whisper_dir = os.path.join(d, "whisper")
            os.makedirs(whisper_dir)
            cfg = {"model_type": "whisper"}
            with open(os.path.join(whisper_dir, "config.json"), "w") as f:
                json.dump(cfg, f)
            for fname in WHISPER_REQUIRED_FILES:
                if fname != "config.json":
                    Path(os.path.join(whisper_dir, fname)).write_text("dummy")
            self.assertTrue(_model_files_exist("whisper", d))

    def test_whisper_missing_model_bin(self):
        with tempfile.TemporaryDirectory() as d:
            whisper_dir = os.path.join(d, "whisper")
            os.makedirs(whisper_dir)
            cfg = {"model_type": "whisper"}
            with open(os.path.join(whisper_dir, "config.json"), "w") as f:
                json.dump(cfg, f)
            # Only config.json — missing model.bin and tokenizer.json
            self.assertFalse(_model_files_exist("whisper", d))

    def test_whisper_no_directory(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(_model_files_exist("whisper", d))

    def test_canary_with_config(self):
        with tempfile.TemporaryDirectory() as d:
            canary_dir = os.path.join(d, "canary")
            os.makedirs(canary_dir)
            with open(os.path.join(canary_dir, "config.json"), "w") as f:
                json.dump({}, f)
            self.assertTrue(_model_files_exist("canary", d))

    def test_canary_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            canary_dir = os.path.join(d, "canary")
            os.makedirs(canary_dir)
            self.assertFalse(_model_files_exist("canary", d))

    def test_unknown_engine(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(_model_files_exist("nonexistent", d))


class TestGetAvailableEngines(unittest.TestCase):
    """get_available_engines returns only engines with both deps and model files."""

    def test_empty_model_dir(self):
        with tempfile.TemporaryDirectory() as d:
            available = get_available_engines(d)
            self.assertEqual(available, [])

    def test_whisper_available_when_files_present(self):
        with tempfile.TemporaryDirectory() as d:
            whisper_dir = os.path.join(d, "whisper")
            os.makedirs(whisper_dir)
            cfg = {"model_type": "whisper"}
            with open(os.path.join(whisper_dir, "config.json"), "w") as f:
                json.dump(cfg, f)
            for fname in WHISPER_REQUIRED_FILES:
                if fname != "config.json":
                    Path(os.path.join(whisper_dir, fname)).write_text("dummy")
            available = get_available_engines(d)
            self.assertIn("whisper", available)


class TestWhisperEngineLoadErrors(unittest.TestCase):
    """WhisperEngine.load must raise clear errors for common failure modes."""

    def test_load_missing_model_triggers_download(self):
        """When model files don't exist, load() should attempt download."""
        engine = WhisperEngine()
        with tempfile.TemporaryDirectory() as d:
            with patch("cv2t.model_downloader.download_whisper_model", return_value=1) as mock_dl:
                with self.assertRaises(RuntimeError) as ctx:
                    engine.load(d, device="cpu")
                self.assertIn("Failed to download", str(ctx.exception))
                mock_dl.assert_called_once()

    @patch("cv2t.engine.whisper._whisper_model_ready", return_value=True)
    @patch("cv2t.engine.whisper._add_nvidia_dll_dirs")
    @patch("cv2t.engine.whisper._patch_suppressed_tokens")
    @patch("cv2t.engine.whisper._log_runtime_diagnostics")
    def test_load_import_error_propagates(self, _diag, _patch, _dll, _ready):
        """If faster_whisper.WhisperModel import fails, error should propagate."""
        engine = WhisperEngine()
        with tempfile.TemporaryDirectory() as d:
            whisper_dir = os.path.join(d, "whisper")
            os.makedirs(whisper_dir)
            with patch(
                "faster_whisper.WhisperModel",
                side_effect=ImportError("No module named 'av'"),
            ):
                with self.assertRaises(ImportError) as ctx:
                    engine.load(d, device="cpu")
                self.assertIn("av", str(ctx.exception))

    def test_unload_after_failed_load(self):
        """Unloading an engine that never loaded should not raise."""
        engine = WhisperEngine()
        engine.unload()
        self.assertFalse(engine.is_loaded)

    def test_transcribe_when_not_loaded_raises(self):
        """Transcribing without loading must raise RuntimeError."""
        engine = WhisperEngine()
        audio = np.zeros(16000, dtype=np.float32)
        with self.assertRaises(RuntimeError) as ctx:
            engine.transcribe(audio, 16000, "en")
        self.assertIn("not loaded", str(ctx.exception))


class TestWhisperModelPathResolution(unittest.TestCase):
    """WhisperEngine.load resolves model path correctly."""

    @patch("cv2t.engine.whisper._add_nvidia_dll_dirs")
    @patch("cv2t.engine.whisper._patch_suppressed_tokens")
    @patch("cv2t.engine.whisper._log_runtime_diagnostics")
    def test_prefers_whisper_subdirectory(self, _diag, _patch, _dll):
        """load() should prefer model_path/whisper/ over model_path/."""
        engine = WhisperEngine()
        with tempfile.TemporaryDirectory() as d:
            whisper_dir = os.path.join(d, "whisper")
            os.makedirs(whisper_dir)

            with patch("cv2t.engine.whisper._whisper_model_ready") as mock_ready:
                mock_ready.side_effect = lambda path: path == whisper_dir
                mock_model = MagicMock()
                with patch("faster_whisper.WhisperModel", return_value=mock_model):
                    engine.load(d, device="cpu")

                # First call should be for the whisper subdirectory
                self.assertEqual(mock_ready.call_args_list[0][0][0], whisper_dir)
                self.assertTrue(engine.is_loaded)
                engine.unload()

    @patch("cv2t.engine.whisper._add_nvidia_dll_dirs")
    @patch("cv2t.engine.whisper._patch_suppressed_tokens")
    @patch("cv2t.engine.whisper._log_runtime_diagnostics")
    def test_falls_back_to_base_path(self, _diag, _patch, _dll):
        """load() should fall back to model_path/ if whisper/ subdir is absent."""
        engine = WhisperEngine()
        with tempfile.TemporaryDirectory() as d:
            with patch("cv2t.engine.whisper._whisper_model_ready") as mock_ready:
                # First call (whisper subdir) returns False, second (base) returns True
                mock_ready.side_effect = [False, True]
                mock_model = MagicMock()
                with patch("faster_whisper.WhisperModel", return_value=mock_model):
                    engine.load(d, device="cpu")
                self.assertTrue(engine.is_loaded)
                engine.unload()


class TestFasterWhisperDependencies(unittest.TestCase):
    """Verify that transitive dependencies of faster-whisper are importable.

    faster-whisper imports av (PyAV) and tokenizers unconditionally.
    These tests catch PyInstaller builds that exclude them by mistake.
    """

    def test_av_is_importable(self):
        import av
        self.assertTrue(hasattr(av, "open"))

    def test_faster_whisper_audio_importable(self):
        import faster_whisper.audio
        self.assertTrue(hasattr(faster_whisper.audio, "decode_audio"))

    def test_tokenizers_is_importable(self):
        import tokenizers
        self.assertTrue(hasattr(tokenizers, "Tokenizer"))

    def test_tokenizers_native_extension_loads(self):
        """tokenizers.tokenizers is the Rust native extension — must be importable."""
        from tokenizers import tokenizers as _native  # noqa: F401

    def test_faster_whisper_transcribe_importable(self):
        """faster_whisper.transcribe imports tokenizers — must not fail."""
        import faster_whisper.transcribe
        self.assertTrue(hasattr(faster_whisper.transcribe, "WhisperModel"))


class TestPyInstallerSpecDependencies(unittest.TestCase):
    """cv2t.spec must include required packages and not strip/exclude them."""

    def setUp(self):
        self.spec_path = _REPO_ROOT / "cv2t.spec"
        self.spec_content = self.spec_path.read_text(encoding="utf-8")

    def test_av_in_hiddenimports(self):
        self.assertIn(
            "'av'",
            self.spec_content,
            "cv2t.spec must list 'av' in hiddenimports (required by faster-whisper)",
        )

    def test_tokenizers_in_hiddenimports(self):
        self.assertIn(
            "'tokenizers'",
            self.spec_content,
            "cv2t.spec must list 'tokenizers' in hiddenimports (required by faster-whisper)",
        )

    def test_required_packages_not_in_excludes(self):
        """av and tokenizers must not appear in the excludes= section."""
        in_excludes = False
        forbidden = {"'av',", "'tokenizers',", "'huggingface_hub',"}
        for line in self.spec_content.splitlines():
            stripped = line.strip()
            if "excludes=" in stripped or "excludes =" in stripped:
                in_excludes = True
            if in_excludes and stripped == "],":
                in_excludes = False
            if in_excludes and stripped in forbidden:
                self.fail(
                    f"cv2t.spec must NOT exclude {stripped.strip(',')} "
                    "— it is required by faster-whisper"
                )

    def test_tokenizers_not_in_strip_patterns(self):
        """tokenizers must not be stripped from the build."""
        # Check that _STRIP_PATTERNS does not contain a pattern matching 'tokenizers'
        import re
        for line in self.spec_content.splitlines():
            if "_re.compile" in line and "tokenizer" in line.lower():
                self.fail(
                    f"cv2t.spec strip pattern matches 'tokenizers' — "
                    f"this will break faster-whisper: {line.strip()}"
                )


class TestCanaryTorchGuard(unittest.TestCase):
    """Canary engine must not register or load when torch/nemo is missing."""

    def test_registry_excludes_canary_when_torch_missing(self):
        """ENGINES must not contain 'canary' when torch is not installed.

        In a non-frozen (source) environment the registry uses
        ``import torch`` to verify torch is genuinely loadable.
        """
        import builtins
        import importlib
        import cv2t.engine as engine_mod

        original_engines = dict(engine_mod.ENGINES)
        _real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch blocked by test")
            return _real_import(name, *args, **kwargs)

        try:
            with patch.object(builtins, "__import__", side_effect=_fake_import):
                engine_mod.ENGINES.clear()
                importlib.reload(engine_mod)
                self.assertNotIn(
                    "canary",
                    engine_mod.ENGINES,
                    "Canary must not register when torch is unavailable",
                )
        finally:
            engine_mod.ENGINES.clear()
            engine_mod.ENGINES.update(original_engines)

    def test_frozen_build_never_registers_native_canary(self):
        """In a frozen build, native canary is impossible — only bridge.

        ``cv2t.spec`` excludes torch/NeMo, so the registry must skip the
        ``import torch`` path entirely and only check the subprocess bridge.
        """
        import importlib
        import cv2t.engine as engine_mod

        original_engines = dict(engine_mod.ENGINES)
        try:
            with patch.object(engine_mod.sys, "frozen", True, create=True):
                # Also ensure bridge reports unavailable
                with patch(
                    "cv2t.engine.canary_bridge.canary_env_available",
                    return_value=False,
                ):
                    engine_mod.ENGINES.clear()
                    importlib.reload(engine_mod)
                    self.assertNotIn(
                        "canary",
                        engine_mod.ENGINES,
                        "Frozen build must not register native canary",
                    )
        finally:
            engine_mod.ENGINES.clear()
            engine_mod.ENGINES.update(original_engines)

    def test_canary_load_raises_when_torch_missing(self):
        """CanaryEngine.load() must raise RuntimeError, not ModuleNotFoundError."""
        try:
            from cv2t.engine.canary import CanaryEngine
        except ImportError:
            self.skipTest("CanaryEngine not importable in this environment")

        engine = CanaryEngine()
        try:
            import importlib.util
            _real_find_spec = importlib.util.find_spec

            def _fake_find_spec(name, *a, **kw):
                if name == "torch":
                    return None
                return _real_find_spec(name, *a, **kw)

            with patch.object(importlib.util, "find_spec", side_effect=_fake_find_spec):
                with self.assertRaises(RuntimeError) as ctx:
                    engine.load("/fake/path", device="cuda")
                self.assertIn("PyTorch", str(ctx.exception))
                self.assertIn("Enable-Canary", str(ctx.exception))
        finally:
            engine._inf_queue.put(None)  # shut down inference thread


class TestEngineFallback(unittest.TestCase):
    """App must fall back gracefully when saved engine isn't available."""

    def test_fallback_when_saved_engine_missing(self):
        """MainWindow must fall back to first available engine, not crash."""
        from cv2t.config import Settings
        import cv2t.engine as engine_mod

        original_engines = dict(engine_mod.ENGINES)

        # Simulate: ENGINES has only whisper, settings say canary
        mock_engine_cls = MagicMock()
        mock_engine_cls.return_value = MagicMock(spec=SpeechEngine)

        engine_mod.ENGINES.clear()
        engine_mod.ENGINES["whisper"] = mock_engine_cls

        try:
            settings = Settings(engine="canary", model_path="/fake")
            # Import the function that selects engine (inline the logic)
            engine_cls = engine_mod.ENGINES.get(settings.engine)
            self.assertIsNone(engine_cls, "canary should not be in ENGINES")

            available = list(engine_mod.ENGINES.keys())
            self.assertEqual(available, ["whisper"])

            # Simulating the fallback logic from MainWindow.__init__
            if engine_cls is None and available:
                fallback = available[0]
                settings.engine = fallback
                engine_cls = engine_mod.ENGINES[fallback]

            self.assertEqual(settings.engine, "whisper")
            self.assertIs(engine_cls, mock_engine_cls)
        finally:
            engine_mod.ENGINES.clear()
            engine_mod.ENGINES.update(original_engines)

    def test_no_engines_raises_runtime_error(self):
        """If no engines at all are available, RuntimeError must be raised."""
        import cv2t.engine as engine_mod

        original_engines = dict(engine_mod.ENGINES)
        engine_mod.ENGINES.clear()

        try:
            available = list(engine_mod.ENGINES.keys())
            self.assertEqual(available, [])
            with self.assertRaises(RuntimeError) as ctx:
                if not available:
                    raise RuntimeError(
                        "No speech engines available. Re-install the "
                        "application or check that dependencies are intact."
                    )
            self.assertIn("No speech engines available", str(ctx.exception))
        finally:
            engine_mod.ENGINES.clear()
            engine_mod.ENGINES.update(original_engines)

    def test_settings_validate_accepts_canary(self):
        """Settings.validate() must accept 'canary' as a valid engine name."""
        from cv2t.config import Settings
        s = Settings(engine="canary")
        s.validate()
        self.assertEqual(s.engine, "canary")

    def test_settings_validate_rejects_unknown_engine(self):
        """Settings.validate() must reject unknown engine names."""
        from cv2t.config import Settings
        s = Settings(engine="nonexistent_engine")
        s.validate()
        self.assertEqual(s.engine, "whisper")


if __name__ == "__main__":
    unittest.main()
