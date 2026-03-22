"""Tests for audio recording utilities."""

import unittest
from unittest.mock import patch

import numpy as np

from cv2t.audio import AudioRecorder


class TestGetRawAudio(unittest.TestCase):
    def test_returns_1d_mono(self):
        """get_raw_audio must return shape (N,) not (N, 1)."""
        rec = AudioRecorder(sample_rate=16000)
        # Simulate recorded frames: (samples, 1) — typical sounddevice output
        fake_frames = np.random.randn(1600, 1).astype(np.float32)
        rec._queue.put(fake_frames)
        rec._recording.set()
        rec._recording.clear()

        audio = rec.get_raw_audio()
        self.assertIsNotNone(audio)
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(audio.shape, (1600,))
        self.assertEqual(audio.dtype, np.float32)

    def test_returns_none_when_empty(self):
        rec = AudioRecorder(sample_rate=16000)
        audio = rec.get_raw_audio()
        self.assertIsNone(audio)

    def test_multichannel_downmix(self):
        """Multi-channel audio should be averaged to mono."""
        rec = AudioRecorder(sample_rate=16000)
        # Simulate 2-channel audio
        fake_frames = np.ones((1600, 2), dtype=np.float32)
        fake_frames[:, 1] = 0.5
        rec._queue.put(fake_frames)
        rec._recording.set()
        rec._recording.clear()

        audio = rec.get_raw_audio()
        self.assertEqual(audio.ndim, 1)
        # Mean of 1.0 and 0.5 = 0.75
        np.testing.assert_allclose(audio, 0.75, atol=1e-6)


class TestTrimSilence(unittest.TestCase):
    def test_pure_silence_returns_none(self):
        rec = AudioRecorder(sample_rate=16000, silence_threshold=0.01)
        audio = np.zeros(16000, dtype=np.float32)
        result = rec.trim_silence(audio)
        self.assertIsNone(result)

    def test_loud_audio_returns_mostly_unchanged(self):
        rec = AudioRecorder(sample_rate=16000, silence_threshold=0.001)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = rec.trim_silence(audio)
        self.assertIsNotNone(result)
        trimmed, pct = result
        self.assertGreater(len(trimmed), 0)


if __name__ == "__main__":
    unittest.main()
