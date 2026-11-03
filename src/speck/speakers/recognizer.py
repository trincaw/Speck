"""Pyannote.audio speaker diarization and voice recognition provider for Speck.

Integrates Pyannote.audio neural segmentation (pyannote/segmentation-3.0) and the
speaker-embedding model of the live classifier (TitaNet-small by default; see
docs/models_report.md) for speaker verification, closed-set identification,
real-time streaming inference, and offline diarization.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from speck.audio import to_mono_float32
from speck.hub import embedding_model, segmentation_model
from speck.speakers.classifier import SpeakerMatch, SpeakerTurn
from speck.speakers.embedding import SpeakerEmbedder

logger = logging.getLogger(__name__)

# Calibration of the default embedding model; instances use their own model's values
MATCH_THRESHOLD = embedding_model().match_threshold
CLUSTER_THRESHOLD = embedding_model().cluster_threshold


class PyannoteSpeakerRecognizer:
    """Production Pyannote.audio diarization & speaker recognition engine.

    Combines pyannote/segmentation-3.0 for detecting speaker boundaries and turns
    with a speaker-embedding model (TitaNet-small by default) for identification and verification.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        segmentation_model_path: Optional[str] = None,
        sample_rate: int = 16000,
        num_threads: int = 2,
        provider: str = "cpu",
    ):
        self.sr = sample_rate
        self.num_threads = num_threads
        self.provider = provider
        self._lock = threading.Lock()

        # model_path: embedding model id or .onnx file (default: SPECK_SPEAKER_MODEL)
        self._embedder = SpeakerEmbedder(model_path, num_threads=num_threads, provider=provider, sample_rate=sample_rate)
        self.embedding_model_path = self._embedder.model_path
        self.segmentation_model_path = segmentation_model().path(segmentation_model_path)
        self.match_threshold = self._embedder.spec.match_threshold
        self.cluster_threshold = self._embedder.spec.cluster_threshold
        self.dim = self._embedder.dim

        # Registry for enrolled speaker profiles: speaker_id -> L2-normalized embedding
        self.database: dict[str, np.ndarray] = {}

    @property
    def architecture_name(self) -> str:
        return f"Pyannote.audio (pyannote/segmentation-3.0 + {self._embedder.spec.name})"

    def extract_embedding(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        """Extract L2-normalized speaker voice embedding."""
        start = time.perf_counter()
        emb = self._embedder.embed(audio)
        return emb, time.perf_counter() - start

    def enroll(self, speaker_id: str, audio: np.ndarray) -> np.ndarray:
        """Enroll or update speaker profile with running average centroid."""
        emb, _ = self.extract_embedding(audio)
        if speaker_id in self.database:
            centroid = self.database[speaker_id] + emb
            self.database[speaker_id] = centroid / (np.linalg.norm(centroid) + 1e-9)
        else:
            self.database[speaker_id] = emb
        return self.database[speaker_id]

    def match(self, audio: np.ndarray) -> tuple[dict[str, float], float]:
        """Query input audio against all enrolled profiles using Cosine similarity."""
        start = time.perf_counter()
        emb, _ = self.extract_embedding(audio)
        scores = {spk_id: float(emb @ spk_emb) for spk_id, spk_emb in self.database.items()}
        return scores, time.perf_counter() - start

    def verify(self, audio_ref: np.ndarray, audio_query: np.ndarray) -> tuple[float, float]:
        """Direct 1:1 pairwise verification score between two audio clips."""
        start = time.perf_counter()
        e_ref, _ = self.extract_embedding(audio_ref)
        e_query, _ = self.extract_embedding(audio_query)
        return float(e_ref @ e_query), time.perf_counter() - start

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: int = -1,
        threshold: Optional[float] = None,
        min_duration_on: float = 0.2,
        min_duration_off: float = 0.3,
    ) -> list[SpeakerTurn]:
        """Diarize audio into single-speaker turns using Pyannote segmentation (threshold: model's calibration)."""
        import sherpa_onnx

        threshold = self.cluster_threshold if threshold is None else threshold
        wav = to_mono_float32(audio)
        duration = len(wav) / self.sr
        if duration <= 0:
            return []

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=self.segmentation_model_path
                ),
                num_threads=self.num_threads,
                provider=self.provider,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=self.embedding_model_path,
                num_threads=self.num_threads,
                provider=self.provider,
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_speakers, threshold=threshold
            ),
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )

        with self._lock:
            sd = sherpa_onnx.OfflineSpeakerDiarization(config)
            res = sd.process(wav)

        if res.num_segments == 0:  # very short or unsegmented audio: one turn
            if not self.database:
                return [SpeakerTurn(0.0, duration, SpeakerMatch("Speaker 1", 1.0, is_new=True))]
            scores, _ = self.match(wav)
            best_spk = max(scores, key=scores.get)
            return [SpeakerTurn(0.0, duration, SpeakerMatch(best_spk, scores[best_spk]))]

        turns: list[SpeakerTurn] = []
        for seg in res.sort_by_start_time():
            start_samp = int(seg.start * self.sr)
            end_samp = min(len(wav), int(seg.end * self.sr))
            label = f"Speaker {seg.speaker + 1}"
            score = float(seg.confidence) if seg.confidence > 0 else 1.0

            # If speaker profiles are enrolled, classify the segmented turn
            if self.database and (end_samp - start_samp) >= int(0.5 * self.sr):
                seg_scores, _ = self.match(wav[start_samp:end_samp])
                if seg_scores:
                    matched_spk = max(seg_scores, key=seg_scores.get)
                    if seg_scores[matched_spk] >= self.match_threshold:
                        label = matched_spk
                        score = seg_scores[matched_spk]

            match = SpeakerMatch(label=label, score=score)
            if turns and turns[-1].match.label == label:
                turns[-1].end = seg.end
            else:
                turns.append(SpeakerTurn(start=seg.start, end=seg.end, match=match))

        return turns

    def segment(self, audio: np.ndarray) -> list[SpeakerTurn]:
        """Alias for diarize() to match Speck SpeakerClassifier protocol."""
        return self.diarize(audio)

    def resolve(self, label: str) -> str:
        """Resolve speaker label alias (identity for pyannote)."""
        return label

    @property
    def embedder(self) -> PyannoteSpeakerRecognizer:
        return self

    @property
    def sample_rate(self) -> int:
        return self.sr


class RealTimeStreamingEngine:
    """Live "who is speaking" over a rolling context window (e.g. 500 ms chunks, 1.5 s window)."""

    def __init__(self, recognizer, sample_rate=16000, window_sec=1.5, chunk_sec=0.5, threshold=None):
        self.recognizer = recognizer
        self.sr = sample_rate
        self.window_size = int(window_sec * sample_rate)
        self.chunk_size = int(chunk_sec * sample_rate)
        self.threshold = getattr(recognizer, "match_threshold", MATCH_THRESHOLD) if threshold is None else threshold
        self.buffer = np.zeros(self.window_size, dtype=np.float32)

    def reset(self) -> None:
        self.buffer.fill(0.0)

    def process_chunk(self, chunk) -> tuple[str, float, float, float]:
        """Slide `chunk` into the window and match it. Returns (speaker, score, latency_s, rtf)."""
        chunk = np.asarray(chunk, dtype=np.float32)[-self.window_size :]
        n = len(chunk)
        self.buffer[: self.window_size - n] = self.buffer[n:]  # in-place shift, no reallocation
        self.buffer[self.window_size - n :] = chunk

        start = time.perf_counter()
        scores, _ = self.recognizer.match(self.buffer)
        latency = time.perf_counter() - start

        best_spk, best_score = "Unknown", 0.0
        if scores:
            best_spk = max(scores, key=scores.get)
            best_score = scores[best_spk]
            if best_score < self.threshold:
                best_spk = "Unknown"
        return best_spk, best_score, latency, latency / (n / self.sr)


__all__ = ["PyannoteSpeakerRecognizer", "RealTimeStreamingEngine"]
