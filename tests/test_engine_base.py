"""Tests for engine base protocol and audio utilities."""

import unittest

import numpy as np

from cv2t.engine.audio_utils import chunk_audio, ensure_16khz, stitch_transcripts


class TestEnsure16khz(unittest.TestCase):
    def test_passthrough_at_16khz(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = ensure_16khz(audio, 16000)
        np.testing.assert_array_equal(result, audio)

    def test_downsample_from_48khz(self):
        # 1 second at 48 kHz → should produce ~16000 samples
        audio = np.random.randn(48000).astype(np.float32)
        result = ensure_16khz(audio, 48000)
        self.assertEqual(len(result), 16000)

    def test_upsample_from_8khz(self):
        audio = np.random.randn(8000).astype(np.float32)
        result = ensure_16khz(audio, 8000)
        self.assertEqual(len(result), 16000)

    def test_empty_audio(self):
        audio = np.array([], dtype=np.float32)
        result = ensure_16khz(audio, 44100)
        self.assertEqual(len(result), 0)


class TestChunkAudio(unittest.TestCase):
    def test_short_audio_no_chunking(self):
        audio = np.zeros(16000 * 10, dtype=np.float32)  # 10 seconds
        chunks = chunk_audio(audio, 16000, max_seconds=30.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), len(audio))

    def test_long_audio_chunked(self):
        audio = np.zeros(16000 * 90, dtype=np.float32)  # 90 seconds
        chunks = chunk_audio(audio, 16000, max_seconds=30.0, overlap_seconds=2.0)
        self.assertGreater(len(chunks), 1)
        # Each chunk should be at most 30s
        for c in chunks:
            self.assertLessEqual(len(c), 16000 * 30)

    def test_exact_boundary(self):
        audio = np.zeros(16000 * 30, dtype=np.float32)  # exactly 30 seconds
        chunks = chunk_audio(audio, 16000, max_seconds=30.0)
        self.assertEqual(len(chunks), 1)


class TestStitchTranscripts(unittest.TestCase):
    def test_single_transcript(self):
        self.assertEqual(stitch_transcripts(["hello world"]), "hello world")

    def test_no_overlap(self):
        result = stitch_transcripts(["hello", "world"])
        self.assertEqual(result, "hello world")

    def test_with_overlap(self):
        result = stitch_transcripts(["the quick brown fox", "brown fox jumps over"])
        self.assertEqual(result, "the quick brown fox jumps over")

    def test_empty_list(self):
        self.assertEqual(stitch_transcripts([]), "")

    def test_empty_entries(self):
        result = stitch_transcripts(["hello", "", "world"])
        self.assertEqual(result, "hello world")


class TestEngineProtocol(unittest.TestCase):
    def test_canary_engine_has_protocol_attrs(self):
        from cv2t.engine.canary import CanaryEngine
        engine = CanaryEngine()
        self.assertEqual(engine.name, "canary")
        self.assertFalse(engine.is_loaded)
        self.assertIsInstance(engine.vram_estimate_gb, float)

    def test_whisper_engine_has_protocol_attrs(self):
        from cv2t.engine.whisper import WhisperEngine
        engine = WhisperEngine()
        self.assertEqual(engine.name, "whisper")
        self.assertFalse(engine.is_loaded)
        self.assertIsInstance(engine.vram_estimate_gb, float)

    def test_unload_when_not_loaded(self):
        """Unload on a not-loaded engine should not raise."""
        from cv2t.engine.canary import CanaryEngine
        from cv2t.engine.whisper import WhisperEngine
        CanaryEngine().unload()
        WhisperEngine().unload()


if __name__ == "__main__":
    unittest.main()
