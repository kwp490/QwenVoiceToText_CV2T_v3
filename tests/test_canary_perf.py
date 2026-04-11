"""Tests for Canary engine performance optimizations.

Covers:
- _needs_blackwell_workarounds() GPU detection logic
- force_cuda_sync config field and validation
- num_beams=1 passed to generate()
- Periodic GC (every N transcriptions, not every call)
- 40-second chunk constant
- Timing instrumentation (returns correct values)
- Bridge forwards force_cuda_sync to worker
- SDPA backend configuration (flash off, mem-efficient+cuDNN on for Blackwell)
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ── _needs_blackwell_workarounds() ───────────────────────────────────────────

class TestNeedsBlackwellWorkarounds(unittest.TestCase):
    """Test GPU-architecture-conditional workaround detection."""

    def _get_fn(self):
        from cv2t.engine.canary import _needs_blackwell_workarounds
        return _needs_blackwell_workarounds

    def test_force_on_always_true(self):
        fn = self._get_fn()
        self.assertTrue(fn("on"))

    def test_force_off_always_false(self):
        fn = self._get_fn()
        self.assertFalse(fn("off"))

    def test_auto_detects_blackwell(self):
        fn = self._get_fn()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (12, 0)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            self.assertTrue(fn("auto"))

    def test_auto_detects_ada(self):
        fn = self._get_fn()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (8, 9)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            self.assertFalse(fn("auto"))

    def test_auto_detects_ampere(self):
        fn = self._get_fn()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (8, 0)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            self.assertFalse(fn("auto"))

    def test_auto_no_cuda_defaults_safe(self):
        fn = self._get_fn()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            self.assertTrue(fn("auto"))

    def test_auto_returns_bool(self):
        fn = self._get_fn()
        result = fn("auto")
        self.assertIsInstance(result, bool)


class TestWorkerNeedsBlackwellWorkarounds(unittest.TestCase):
    """Test the standalone worker's copy of _needs_blackwell_workarounds."""

    def _get_fn(self):
        from cv2t.engine.canary_worker import _needs_blackwell_workarounds
        return _needs_blackwell_workarounds

    def test_force_on(self):
        self.assertTrue(self._get_fn()("on"))

    def test_force_off(self):
        self.assertFalse(self._get_fn()("off"))

    def test_auto_returns_bool(self):
        result = self._get_fn()("auto")
        self.assertIsInstance(result, bool)


# ── force_cuda_sync config ───────────────────────────────────────────────────

class TestForceCudaSyncConfig(unittest.TestCase):
    """Test the force_cuda_sync settings field."""

    def test_default_is_auto(self):
        from cv2t.config import Settings
        s = Settings()
        self.assertEqual(s.force_cuda_sync, "auto")

    def test_valid_values_accepted(self):
        from cv2t.config import Settings
        for val in ("auto", "on", "off"):
            s = Settings(force_cuda_sync=val)
            s.validate()
            self.assertEqual(s.force_cuda_sync, val)

    def test_invalid_value_reset_to_auto(self):
        from cv2t.config import Settings
        s = Settings(force_cuda_sync="invalid")
        s.validate()
        self.assertEqual(s.force_cuda_sync, "auto")

    def test_round_trip(self):
        import tempfile
        from pathlib import Path
        from cv2t.config import Settings

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "settings.json"
            s = Settings(force_cuda_sync="off")
            s.save(p)
            loaded = Settings.load(p)
            self.assertEqual(loaded.force_cuda_sync, "off")

    def test_missing_field_gets_default(self):
        import tempfile
        from pathlib import Path
        from cv2t.config import Settings

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "settings.json"
            # Write JSON without force_cuda_sync (simulates old config)
            p.write_text(json.dumps({"engine": "canary"}), encoding="utf-8")
            loaded = Settings.load(p)
            self.assertEqual(loaded.force_cuda_sync, "auto")


# ── Chunk size constant ──────────────────────────────────────────────────────

class TestChunkSizeConstants(unittest.TestCase):
    """Verify the chunk constants are set to 40s."""

    def test_canary_engine_chunk_size(self):
        from cv2t.engine.canary import _MAX_CHUNK_SECONDS
        self.assertEqual(_MAX_CHUNK_SECONDS, 40.0)

    def test_canary_worker_chunk_size(self):
        from cv2t.engine.canary_worker import _MAX_CHUNK_SECONDS
        self.assertEqual(_MAX_CHUNK_SECONDS, 40.0)

    def test_overlap_seconds(self):
        from cv2t.engine.canary import _OVERLAP_SECONDS
        from cv2t.engine.canary_worker import _OVERLAP_SECONDS as worker_overlap
        self.assertEqual(_OVERLAP_SECONDS, 2.0)
        self.assertEqual(worker_overlap, 2.0)


# ── Periodic GC ──────────────────────────────────────────────────────────────

class TestPeriodicGC(unittest.TestCase):
    """GC should only run every _GC_INTERVAL transcriptions, not every call."""

    def test_gc_interval_defined(self):
        from cv2t.engine.canary import CanaryEngine
        self.assertGreater(CanaryEngine._GC_INTERVAL, 1)

    @patch("cv2t.engine.canary.gc.collect")
    def test_gc_not_called_on_first_transcription(self, mock_gc):
        from cv2t.engine.canary import CanaryEngine
        engine = CanaryEngine()
        engine._model = MagicMock()
        engine._transcribe_count = 0

        # Mock _transcribe_chunk to return a tuple (text, gen_time, io_time)
        engine._transcribe_chunk = MagicMock(return_value=("hello", 0.1, 0.01))

        # Mock chunk_audio to return one chunk
        with patch("cv2t.engine.canary.chunk_audio", return_value=[np.zeros(16000)]):
            with patch("cv2t.engine.canary.stitch_transcripts", return_value="hello"):
                # Need torch mock for import inside _transcribe_impl
                mock_torch = MagicMock()
                with patch.dict("sys.modules", {"torch": mock_torch}):
                    result = engine._transcribe_impl(np.zeros(16000))

        self.assertEqual(result, "hello")
        mock_gc.assert_not_called()

    @patch("cv2t.engine.canary.gc.collect")
    def test_gc_called_at_interval(self, mock_gc):
        from cv2t.engine.canary import CanaryEngine
        engine = CanaryEngine()
        engine._model = MagicMock()
        # Set count to one less than the interval
        engine._transcribe_count = CanaryEngine._GC_INTERVAL - 1

        engine._transcribe_chunk = MagicMock(return_value=("hello", 0.1, 0.01))

        with patch("cv2t.engine.canary.chunk_audio", return_value=[np.zeros(16000)]):
            with patch("cv2t.engine.canary.stitch_transcripts", return_value="hello"):
                mock_torch = MagicMock()
                with patch.dict("sys.modules", {"torch": mock_torch}):
                    engine._transcribe_impl(np.zeros(16000))

        mock_gc.assert_called_once()


# ── num_beams=1 in generate() ────────────────────────────────────────────────

class TestNumBeamsExplicit(unittest.TestCase):
    """Verify that num_beams=1 is explicitly passed to model.generate()."""

    def test_canary_engine_passes_num_beams(self):
        """CanaryEngine._transcribe_chunk must pass num_beams=1."""
        from cv2t.engine.canary import CanaryEngine
        engine = CanaryEngine()

        mock_model = MagicMock()
        mock_model.audio_locator_tag = "<|audio|>"
        mock_response = MagicMock()
        mock_response.cpu.return_value = [1, 2, 3]
        mock_response.__len__ = MagicMock(return_value=3)
        mock_model.generate.return_value = [mock_response]
        mock_model.tokenizer.ids_to_text.return_value = "hello"
        engine._model = mock_model

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        chunk = np.zeros(16000, dtype=np.float32)  # 1 second
        with patch("cv2t.engine.canary.tempfile.mkstemp", return_value=(0, "/tmp/test.wav")):
            with patch("cv2t.engine.canary.os.close"):
                with patch("cv2t.engine.canary.sf.write"):
                    with patch("cv2t.engine.canary.os.unlink"):
                        text, _, _ = engine._transcribe_chunk(chunk, mock_torch)

        # Verify num_beams=1 was passed
        call_kwargs = mock_model.generate.call_args
        self.assertEqual(call_kwargs.kwargs.get("num_beams"), 1)


# ── Bridge forwards force_cuda_sync ──────────────────────────────────────────

class TestBridgeForcesCudaSync(unittest.TestCase):
    """CanaryBridgeEngine must pass force_cuda_sync to the worker."""

    @patch("cv2t.engine.canary_bridge._get_canary_paths")
    @patch("cv2t.engine.canary_bridge.subprocess.Popen")
    def test_load_sends_force_cuda_sync(self, mock_popen, mock_paths):
        from cv2t.engine.canary_bridge import CanaryBridgeEngine

        mock_paths.return_value = ("python.exe", "worker.py")
        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdout.readline.return_value = json.dumps({"status": "ready"}) + "\n"
        proc.stdin = MagicMock()
        proc.stderr = MagicMock()
        mock_popen.return_value = proc

        engine = CanaryBridgeEngine()
        engine.force_cuda_sync = "off"
        engine.load("C:\\models", "cuda")

        # Find the load command that was sent
        written = proc.stdin.write.call_args[0][0]
        cmd = json.loads(written)
        self.assertEqual(cmd["command"], "load")
        self.assertEqual(cmd["force_cuda_sync"], "off")

    @patch("cv2t.engine.canary_bridge._get_canary_paths")
    @patch("cv2t.engine.canary_bridge.subprocess.Popen")
    def test_load_defaults_to_auto(self, mock_popen, mock_paths):
        from cv2t.engine.canary_bridge import CanaryBridgeEngine

        mock_paths.return_value = ("python.exe", "worker.py")
        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdout.readline.return_value = json.dumps({"status": "ready"}) + "\n"
        proc.stdin = MagicMock()
        proc.stderr = MagicMock()
        mock_popen.return_value = proc

        engine = CanaryBridgeEngine()
        engine.load("C:\\models", "cuda")

        written = proc.stdin.write.call_args[0][0]
        cmd = json.loads(written)
        self.assertEqual(cmd["force_cuda_sync"], "auto")


# ── Worker _load_model accepts force_cuda_sync ───────────────────────────────

class TestWorkerLoadModel(unittest.TestCase):
    """canary_worker._load_model must accept force_cuda_sync parameter."""

    def test_load_model_signature(self):
        import inspect
        from cv2t.engine.canary_worker import _load_model
        sig = inspect.signature(_load_model)
        self.assertIn("force_cuda_sync", sig.parameters)
        self.assertEqual(sig.parameters["force_cuda_sync"].default, "auto")


# ── Timing instrumentation ───────────────────────────────────────────────────

class TestTimingInstrumentation(unittest.TestCase):
    """Timing instrumentation must not break return values."""

    def test_transcribe_chunk_returns_tuple(self):
        """_transcribe_chunk must return (text, gen_time, io_time)."""
        from cv2t.engine.canary import CanaryEngine
        engine = CanaryEngine()

        mock_model = MagicMock()
        mock_model.audio_locator_tag = "<|audio|>"
        mock_response = MagicMock()
        mock_response.cpu.return_value = [1, 2, 3]
        mock_response.__len__ = MagicMock(return_value=3)
        mock_model.generate.return_value = [mock_response]
        mock_model.tokenizer.ids_to_text.return_value = "hello world"
        engine._model = mock_model

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        chunk = np.zeros(16000, dtype=np.float32)
        with patch("cv2t.engine.canary.tempfile.mkstemp", return_value=(0, "/tmp/test.wav")):
            with patch("cv2t.engine.canary.os.close"):
                with patch("cv2t.engine.canary.sf.write"):
                    with patch("cv2t.engine.canary.os.unlink"):
                        result = engine._transcribe_chunk(chunk, mock_torch)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        text, gen_time, io_time = result
        self.assertEqual(text, "hello world")
        self.assertIsInstance(gen_time, float)
        self.assertIsInstance(io_time, float)
        self.assertGreaterEqual(gen_time, 0)
        self.assertGreaterEqual(io_time, 0)


# ── Engine attribute defaults ────────────────────────────────────────────────

class TestEngineAttributes(unittest.TestCase):
    """Verify new engine attributes have correct defaults."""

    def test_canary_engine_force_cuda_sync_default(self):
        from cv2t.engine.canary import CanaryEngine
        engine = CanaryEngine()
        self.assertEqual(engine.force_cuda_sync, "auto")

    def test_bridge_engine_force_cuda_sync_default(self):
        from cv2t.engine.canary_bridge import CanaryBridgeEngine
        engine = CanaryBridgeEngine()
        self.assertEqual(engine.force_cuda_sync, "auto")

    def test_canary_engine_transcribe_count_starts_zero(self):
        from cv2t.engine.canary import CanaryEngine
        engine = CanaryEngine()
        self.assertEqual(engine._transcribe_count, 0)


# ── SDPA backend configuration on Blackwell ──────────────────────────────────

class TestBlackwellSDPAConfig(unittest.TestCase):
    """Verify that only flash SDP is disabled on Blackwell.

    Memory-efficient and cuDNN SDPA backends are confirmed working on
    sm_120 (Blackwell) with PyTorch 2.7.x and provide ~26x speedup
    over the math-only fallback.  The flash SDP kernel is not compiled
    into the Windows wheels so it must be disabled to prevent PyTorch
    from attempting it and falling through to math-only.
    """

    @patch("cv2t.engine.canary.SALM", create=True)
    def test_blackwell_disables_only_flash_sdp(self, mock_salm_cls):
        """On Blackwell, flash SDP must be off; mem-efficient must stay on."""
        from cv2t.engine.canary import CanaryEngine

        mock_salm = MagicMock()
        mock_salm.eval.return_value = mock_salm
        mock_salm.to.return_value = mock_salm
        mock_salm.audio_locator_tag = "<|audio|>"
        mock_salm.generate.return_value = [MagicMock(cpu=MagicMock(return_value=[]))]
        mock_salm.tokenizer.ids_to_text.return_value = ""
        mock_salm_cls.from_pretrained.return_value = mock_salm

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.is_bf16_supported.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (12, 0)
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float16 = "float16"

        enable_flash = MagicMock()
        enable_mem = MagicMock()
        mock_torch.backends.cuda.enable_flash_sdp = enable_flash
        mock_torch.backends.cuda.enable_mem_efficient_sdp = enable_mem

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "nemo": MagicMock(),
            "nemo.collections": MagicMock(),
            "nemo.collections.speechlm2": MagicMock(),
            "nemo.collections.speechlm2.models": MagicMock(SALM=mock_salm_cls),
        }):
            engine = CanaryEngine()
            engine.force_cuda_sync = "on"  # force blackwell path
            try:
                engine._load_impl(os.path.join(tempfile.gettempdir(), "models"), "cuda")
            except Exception:
                pass  # model loading internals may fail; we only check SDP calls

        # Flash SDP must be disabled
        enable_flash.assert_called_with(False)
        # Mem-efficient SDP must NOT be disabled
        enable_mem.assert_not_called()

    @patch("cv2t.engine.canary.SALM", create=True)
    def test_non_blackwell_leaves_all_sdp_enabled(self, mock_salm_cls):
        """On non-Blackwell GPUs, no SDP backends should be disabled."""
        from cv2t.engine.canary import CanaryEngine

        mock_salm = MagicMock()
        mock_salm.eval.return_value = mock_salm
        mock_salm.to.return_value = mock_salm
        mock_salm.audio_locator_tag = "<|audio|>"
        mock_salm.generate.return_value = [MagicMock(cpu=MagicMock(return_value=[]))]
        mock_salm.tokenizer.ids_to_text.return_value = ""
        mock_salm_cls.from_pretrained.return_value = mock_salm

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.is_bf16_supported.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (8, 9)  # Ada
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float16 = "float16"

        enable_flash = MagicMock()
        enable_mem = MagicMock()
        mock_torch.backends.cuda.enable_flash_sdp = enable_flash
        mock_torch.backends.cuda.enable_mem_efficient_sdp = enable_mem

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "nemo": MagicMock(),
            "nemo.collections": MagicMock(),
            "nemo.collections.speechlm2": MagicMock(),
            "nemo.collections.speechlm2.models": MagicMock(SALM=mock_salm_cls),
        }):
            engine = CanaryEngine()
            engine.force_cuda_sync = "off"  # non-blackwell path
            try:
                engine._load_impl(os.path.join(tempfile.gettempdir(), "models"), "cuda")
            except Exception:
                pass

        # Neither SDP backend should be touched on non-Blackwell
        enable_flash.assert_not_called()
        enable_mem.assert_not_called()


class TestWorkerBlackwellSDPAConfig(unittest.TestCase):
    """Verify the worker also only disables flash SDP on Blackwell."""

    def test_worker_source_only_disables_flash(self):
        """canary_worker.py must not call enable_mem_efficient_sdp(False)."""
        import ast
        from pathlib import Path

        worker_path = Path(__file__).resolve().parent.parent / "cv2t" / "engine" / "canary_worker.py"
        source = worker_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        flash_disabled = False
        mem_efficient_disabled = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_src = ast.get_source_segment(source, node)
                if call_src and "enable_flash_sdp(False)" in call_src:
                    flash_disabled = True
                if call_src and "enable_mem_efficient_sdp(False)" in call_src:
                    mem_efficient_disabled = True

        self.assertTrue(flash_disabled,
                        "canary_worker.py must disable flash SDP on Blackwell")
        self.assertFalse(mem_efficient_disabled,
                         "canary_worker.py must NOT disable mem-efficient SDP")

    def test_engine_source_only_disables_flash(self):
        """canary.py must not call enable_mem_efficient_sdp(False)."""
        import ast
        from pathlib import Path

        engine_path = Path(__file__).resolve().parent.parent / "cv2t" / "engine" / "canary.py"
        source = engine_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        flash_disabled = False
        mem_efficient_disabled = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_src = ast.get_source_segment(source, node)
                if call_src and "enable_flash_sdp(False)" in call_src:
                    flash_disabled = True
                if call_src and "enable_mem_efficient_sdp(False)" in call_src:
                    mem_efficient_disabled = True

        self.assertTrue(flash_disabled,
                        "canary.py must disable flash SDP on Blackwell")
        self.assertFalse(mem_efficient_disabled,
                         "canary.py must NOT disable mem-efficient SDP")


if __name__ == "__main__":
    unittest.main()
