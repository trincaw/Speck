"""Default-microphone capture (sounddevice): mono float32 blocks delivered to a callback."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import numpy as np


@contextmanager
def open_microphone(
    on_audio: Callable[[np.ndarray], None],
    sample_rate: int = 16000,
    block_seconds: float = 0.1,
) -> Iterator[None]:
    """Stream the default input device into `on_audio` while the context is open.

    `on_audio` runs on the audio thread: keep it short (push to a session or a queue).
    """
    import sounddevice as sd

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=int(sample_rate * block_seconds),
        callback=lambda indata, *_: on_audio(indata[:, 0].copy()),
    ):
        yield
