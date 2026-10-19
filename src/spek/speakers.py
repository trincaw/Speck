"""Voice activity detection and online speaker classification (sherpa-onnx).

Used for live microphone streams: the VAD cuts audio into utterances, and each utterance
is embedded and matched against known speakers (enrolled or discovered on the fly).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from spek.utils import resolve_hf_file, to_mono_float32

logger = logging.getLogger(__name__)

VAD_REPO = "csukuangfj/vad"
VAD_FILE = "silero_vad.onnx"

EMBEDDING_REPO = "csukuangfj/speaker-embedding-models"
EMBEDDING_FILE = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"


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
            self._vad.accept_waveform(buf[i : i + self._window])
        self._pending = buf[n:].copy()
        return self._drain()

    def flush(self) -> list[Utterance]:
        if len(self._pending):
            self._vad.accept_waveform(np.pad(self._pending, (0, self._window - len(self._pending))))
            self._pending = np.zeros(0, dtype=np.float32)
        self._vad.flush()
        return self._drain()

    @property
    def is_speech(self) -> bool:
        return self._vad.is_speech_detected()

    def _drain(self) -> list[Utterance]:
        out = []
        while not self._vad.empty():
            seg = self._vad.front
            out.append(Utterance(start=seg.start / self.sample_rate, samples=np.asarray(seg.samples, dtype=np.float32)))
            self._vad.pop()
        return out


class SpeakerEmbedder:
    """L2-normalized speaker embeddings (CAM++ / 3D-Speaker by default)."""

    def __init__(self, model: Optional[str] = None, num_threads: int = 2, provider: str = "cpu", sample_rate: int = 16000):
        import sherpa_onnx

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=resolve_hf_file(model, EMBEDDING_REPO, EMBEDDING_FILE),
            num_threads=num_threads,
            provider=provider,
        )
        if not cfg.validate():
            raise ValueError(f"Invalid speaker embedding config: {cfg}")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.sample_rate = sample_rate
        self.dim = self._extractor.dim

    def embed(self, samples: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(self.sample_rate, to_mono_float32(samples))
        stream.input_finished()
        emb = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)


@dataclass
class _Speaker:
    label: str
    centroid: np.ndarray
    count: int = 1
    enrolled: bool = False


@dataclass
class SpeakerMatch:
    label: str
    score: float
    is_new: bool = False


@dataclass
class SpeakerClassifier:
    """Online nearest-centroid speaker classifier.

    Each utterance embedding is compared (cosine) to every known speaker. Above `threshold`
    it is assigned to the best match and that centroid is refined; below it, a new
    "Speaker N" is created (up to `max_speakers`). Short utterances never create speakers,
    since their embeddings are unreliable.
    """

    embedder: SpeakerEmbedder
    threshold: float = 0.5
    max_speakers: int = 0  # <= 0: unlimited
    min_new_speaker_seconds: float = 1.5
    max_centroid_weight: int = 50  # caps history so centroids keep adapting
    speakers: list[_Speaker] = field(default_factory=list)

    def enroll(self, label: str, samples: np.ndarray) -> None:
        """Register a known person from a clean voice sample (a few seconds is enough)."""
        emb = self.embedder.embed(samples)
        existing = next((s for s in self.speakers if s.label == label), None)
        if existing:
            self._update(existing, emb)
        else:
            self.speakers.append(_Speaker(label, emb, enrolled=True))

    def classify(self, samples: np.ndarray) -> SpeakerMatch:
        emb = self.embedder.embed(samples)
        duration = len(samples) / self.embedder.sample_rate

        best, score = None, -1.0
        if self.speakers:
            scores = np.stack([s.centroid for s in self.speakers]) @ emb
            idx = int(np.argmax(scores))
            best, score = self.speakers[idx], float(scores[idx])

        can_create = duration >= self.min_new_speaker_seconds and (
            self.max_speakers <= 0 or len(self.speakers) < self.max_speakers
        )
        if best is None or (score < self.threshold and can_create):
            speaker = _Speaker(f"Speaker {sum(not s.enrolled for s in self.speakers)}", emb)
            self.speakers.append(speaker)
            return SpeakerMatch(speaker.label, 1.0, is_new=True)

        if score >= self.threshold:
            self._update(best, emb)
        return SpeakerMatch(best.label, score)

    def _update(self, speaker: _Speaker, emb: np.ndarray) -> None:
        w = min(speaker.count, self.max_centroid_weight)
        centroid = speaker.centroid * w + emb
        speaker.centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        speaker.count += 1
