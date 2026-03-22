"""
Audio utilities for engine preprocessing — resampling and chunking.
"""

from __future__ import annotations

from typing import List

import numpy as np

TARGET_SR = 16000


def ensure_16khz(audio: np.ndarray, source_sr: int) -> np.ndarray:
    """Resample audio to 16 kHz if needed.  Input must be 1D float32."""
    if source_sr == TARGET_SR:
        return audio
    duration = len(audio) / source_sr
    target_len = int(duration * TARGET_SR)
    if target_len == 0:
        return np.array([], dtype=np.float32)
    indices = np.linspace(0, len(audio) - 1, target_len)
    left = np.floor(indices).astype(int)
    right = np.minimum(left + 1, len(audio) - 1)
    frac = (indices - left).astype(np.float32)
    return audio[left] * (1.0 - frac) + audio[right] * frac


def chunk_audio(
    audio: np.ndarray,
    sr: int,
    max_seconds: float = 30.0,
    overlap_seconds: float = 2.0,
) -> List[np.ndarray]:
    """Split audio into overlapping chunks.

    Returns a list of 1D float32 arrays, each at most *max_seconds* long.
    Adjacent chunks overlap by *overlap_seconds*.
    """
    max_samples = int(max_seconds * sr)
    overlap_samples = int(overlap_seconds * sr)
    step = max_samples - overlap_samples

    if len(audio) <= max_samples:
        return [audio]

    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + max_samples, len(audio))
        chunks.append(audio[start:end])
        if end >= len(audio):
            break
        start += step
    return chunks


def stitch_transcripts(texts: List[str]) -> str:
    """Join chunk transcripts, deduplicating overlap at boundaries.

    Uses a simple suffix/prefix match to remove repeated words at
    chunk boundaries.
    """
    if not texts:
        return ""
    result = texts[0]
    for nxt in texts[1:]:
        if not nxt:
            continue
        if not result:
            result = nxt
            continue
        # Try to find overlapping words
        words_r = result.split()
        words_n = nxt.split()
        best_overlap = 0
        max_check = min(len(words_r), len(words_n), 10)
        for k in range(1, max_check + 1):
            if words_r[-k:] == words_n[:k]:
                best_overlap = k
        if best_overlap > 0:
            result = result + " " + " ".join(words_n[best_overlap:])
        else:
            result = result + " " + nxt
    return result.strip()
