"""Audio array helpers shared by every pipeline stage."""

from __future__ import annotations

import numpy as np


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Squeeze and downmix to a 1-D float32 array (channels on either axis)."""
    wav = np.asarray(audio, dtype=np.float32).squeeze()
    if wav.ndim > 1:
        # Channels are the shorter axis
        wav = wav.mean(axis=0 if wav.shape[0] < wav.shape[1] else 1)
    return wav


def normalize_audio(wav: np.ndarray, peak: float = 0.9, max_gain_db: float = 20.0) -> np.ndarray:
    """Remove DC offset and scale to a target peak; near-silent input is left alone (no noise boosting)."""
    if len(wav) == 0:
        return wav
    wav = wav - wav.mean(dtype=np.float64).astype(np.float32)
    top = float(np.abs(wav).max())
    if top < 1e-4:
        return wav
    return (wav * min(peak / top, 10 ** (max_gain_db / 20))).astype(np.float32)
