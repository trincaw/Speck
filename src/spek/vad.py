"""Voice activity detection: cuts a live audio stream into utterances (Silero VAD via sherpa-onnx)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from spek.utils import resolve_hf_file

VAD_REPO = "csukuangfj/vad"
VAD_FILE = "silero_vad.onnx"


@dataclass
class Utterance:
    start: float  # seconds from stream start
    samples: np.ndarray


class VadSegmenter:
    """Incremental Silero VAD: feed arbitrary-size chunks, get back finished utterances."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence: float = 0.4,
        min_speech: float = 0.25,
        max_speech: float = 15.0,
        model: Optional[str] = None,
        provider: str = "cpu",
    ):
        import sherpa_onnx

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = resolve_hf_file(model, VAD_REPO, VAD_FILE)
        cfg.silero_vad.threshold = threshold
        cfg.silero_vad.min_silence_duration = min_silence
        cfg.silero_vad.min_speech_duration = min_speech
        cfg.silero_vad.max_speech_duration = max_speech
        cfg.sample_rate = sample_rate
        cfg.num_threads = 1
        cfg.provider = provider

        self.sample_rate = sample_rate
        self._window = cfg.silero_vad.window_size
        self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=max_speech + 5)
        self._pending = np.zeros(0, dtype=np.float32)

    def accept(self, samples: np.ndarray) -> list[Utterance]:
        buf = np.concatenate([self._pending, samples]) if len(self._pending) else samples
        n = len(buf) - len(buf) % self._window
        for i in range(0, n, self._window):
            window = buf[i : i + self._window]
            self._vad.accept_waveform(window)
        self._pending = buf[n:].copy()
        return self._drain()

    def flush(self) -> list[Utterance]:
        if len(self._pending):
            self._vad.accept_waveform(np.pad(self._pending, (0, self._window - len(self._pending))))
            self._pending = np.zeros(0, dtype=np.float32)
        self._vad.flush()
        return self._drain()

    def _drain(self) -> list[Utterance]:
        out = []
        while not self._vad.empty():
            seg = self._vad.front
            out.append(Utterance(start=seg.start / self.sample_rate, samples=np.asarray(seg.samples, dtype=np.float32)))
            self._vad.pop()
        return out
