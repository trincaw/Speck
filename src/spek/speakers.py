"""Online speaker classification (sherpa-onnx speaker embeddings).

Used for live streams: the VAD cuts audio into utterances, each utterance is scanned for
speaker changes and every resulting turn is embedded and matched against known speakers
(enrolled or discovered on the fly). While someone is still talking, `LiveSpeakerPreview`
gives a provisional label from the most recent audio.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from spek.utils import resolve_hf_file, to_mono_float32
from spek.vad import Utterance, VadSegmenter  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)

EMBEDDING_REPO = "csukuangfj/speaker-embedding-models"
# TitaNet keeps a stable similarity scale across utterance lengths (CAM++ swings from
# 0.2 to 0.8 between 1.5 s and 3 s clips), which a single online threshold depends on.
EMBEDDING_FILE = "nemo_en_titanet_small.onnx"


class SpeakerEmbedder:
    """L2-normalized speaker embeddings (NeMo TitaNet-small by default)."""

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
        self._lock = threading.Lock()  # shared by concurrent live sessions
        self.sample_rate = sample_rate
        self.dim = self._extractor.dim

    def embed(self, samples: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(self.sample_rate, to_mono_float32(samples))
        stream.input_finished()
        with self._lock:
            emb = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)


@dataclass(eq=False)  # identity semantics: centroids are arrays
class _Speaker:
    label: str
    centroid: np.ndarray
    count: int = 1
    enrolled: bool = False
    seconds: float = 0.0  # speech seen, used to weight the centroid


@dataclass
class SpeakerMatch:
    label: str
    score: float
    is_new: bool = False
    confident: bool = True  # False: below threshold, label is a best guess


@dataclass
class SpeakerTurn:
    """A single-speaker stretch of an utterance (times relative to the utterance start)."""

    start: float
    end: float
    match: SpeakerMatch


@dataclass
class SpeakerClassifier:
    """Online speaker classifier: nearest centroid with change detection and self-correction.

    - `segment()` slides a window over an utterance and splits it where adjacent windows
      stop sounding like the same voice, so fast hand-overs without a pause are caught.
    - Each turn is compared (cosine) to every known speaker, with a small bonus for whoever
      spoke last. Above `threshold` it is assigned and that centroid refined (weighted by
      duration); below it, a new "Speaker N" is created (up to `max_speakers`). Short turns
      never create speakers, since their embeddings are unreliable.
    - Discovered speakers whose centroids converge above `merge_threshold` are merged;
      `resolve()` maps a stale label to the speaker it was folded into.
    - `peek()` scores audio without learning from it, for provisional live labels.
    """

    embedder: SpeakerEmbedder
    threshold: float = 0.35
    max_speakers: int = 0  # <= 0: unlimited
    min_new_speaker_seconds: float = 1.5
    max_centroid_seconds: float = 120.0  # caps history so centroids keep adapting
    continuity_bonus: float = 0.05
    merge_threshold: float = 0.6
    change_window: float = 1.0
    change_hop: float = 0.25
    change_threshold: float = 0.3
    min_turn_seconds: float = 1.0
    speakers: list[_Speaker] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
    _last: Optional[_Speaker] = field(default=None, repr=False)
    _next_id: int = field(default=0, repr=False)

    def enroll(self, label: str, samples: np.ndarray) -> None:
        """Register a known person from a clean voice sample (a few seconds is enough)."""
        emb = self.embedder.embed(samples)
        duration = len(samples) / self.embedder.sample_rate
        existing = next((s for s in self.speakers if s.label == label), None)
        if existing:
            self._update(existing, emb, duration)
        else:
            self.speakers.append(_Speaker(label, emb, enrolled=True, seconds=duration))

    def resolve(self, label: str) -> str:
        """Current label of a speaker, following merges."""
        while label in self.aliases:
            label = self.aliases[label]
        return label

    def classify(self, samples: np.ndarray) -> SpeakerMatch:
        """Label one single-speaker stretch of audio and learn from it."""
        return self._classify(self.embedder.embed(samples), len(samples) / self.embedder.sample_rate)

    def peek(self, samples: np.ndarray) -> SpeakerMatch:
        """Best guess for `samples` without changing any state; "?" for an unknown voice."""
        best, score, _ = self._best(self.embedder.embed(samples))
        if best is None or score < self.threshold:
            return SpeakerMatch("?", max(score, 0.0), is_new=True, confident=False)
        return SpeakerMatch(best.label, score)

    def segment(self, samples: np.ndarray) -> list[SpeakerTurn]:
        """Split an utterance at speaker changes and classify each turn."""
        sr = self.embedder.sample_rate
        cuts = self._change_points(samples)
        bounds = [0, *cuts, len(samples)]
        turns: list[SpeakerTurn] = []
        for a, b in zip(bounds, bounds[1:]):
            match = self.classify(samples[a:b])
            if turns and self.resolve(turns[-1].match.label) == match.label:
                turns[-1].end = b / sr
                turns[-1].match.label = match.label
            else:
                turns.append(SpeakerTurn(a / sr, b / sr, match))
        for turn in turns:
            turn.match.label = self.resolve(turn.match.label)
        return turns

    def _change_points(self, samples: np.ndarray) -> list[int]:
        """Sample offsets where the voice changes, at least `min_turn_seconds` apart."""
        sr = self.embedder.sample_rate
        win, hop = int(self.change_window * sr), int(self.change_hop * sr)
        min_turn = int(self.min_turn_seconds * sr)
        if len(samples) < 2 * max(win, min_turn):
            return []
        starts = range(0, len(samples) - win + 1, hop)
        embs = np.stack([self.embedder.embed(samples[s : s + win]) for s in starts])
        step = win // hop  # windows i and i+step touch without overlapping
        if len(embs) <= step:
            return []
        sims = np.einsum("ij,ij->i", embs[:-step], embs[step:])
        boundary = [(i + step) * hop for i in range(len(sims))]

        cuts: list[int] = []
        for i in np.argsort(sims):
            if sims[i] >= self.change_threshold:
                break
            b = boundary[i]
            if min(b, len(samples) - b) >= min_turn and all(abs(b - c) >= min_turn for c in cuts):
                cuts.append(b)
        cuts = sorted(self._snap_to_pause(samples, c) for c in cuts)

        # Confirm on whole turns, which embed far more reliably than single windows
        while cuts:
            bounds = [0, *cuts, len(samples)]
            pieces = [self.embedder.embed(samples[a:b]) for a, b in zip(bounds, bounds[1:])]
            sims = [float(x @ y) for x, y in zip(pieces, pieces[1:])]
            i = int(np.argmax(sims))
            if sims[i] < self.threshold:
                break
            del cuts[i]
        return cuts

    def _snap_to_pause(self, samples: np.ndarray, cut: int) -> int:
        """Move a cut to the quietest 20 ms frame within half a hop, so turns sent to ASR
        rarely start or end mid-word."""
        sr = self.embedder.sample_rate
        frame, radius = sr // 50, int(self.change_hop * sr / 2)
        lo = max(0, cut - radius)
        n = (min(len(samples), cut + radius) - lo) // frame
        if n < 2:
            return cut
        energy = np.square(samples[lo : lo + n * frame].reshape(n, frame)).mean(axis=1)
        return lo + int(np.argmin(energy)) * frame + frame // 2

    def _best(self, emb: np.ndarray) -> tuple[Optional[_Speaker], float, float]:
        """(best speaker, raw cosine, cosine incl. continuity bonus)."""
        if not self.speakers:
            return None, -1.0, -1.0
        raw = np.stack([s.centroid for s in self.speakers]) @ emb
        biased = raw + np.array([self.continuity_bonus if s is self._last else 0.0 for s in self.speakers])
        idx = int(np.argmax(biased))
        return self.speakers[idx], float(raw[idx]), float(biased[idx])

    def _classify(self, emb: np.ndarray, duration: float) -> SpeakerMatch:
        best, score, biased = self._best(emb)
        can_create = duration >= self.min_new_speaker_seconds and (
            self.max_speakers <= 0 or len(self.speakers) < self.max_speakers
        )
        if best is None or (biased < self.threshold and can_create):
            speaker = _Speaker(f"Speaker {self._next_id}", emb, seconds=duration)
            self._next_id += 1
            self.speakers.append(speaker)
            self._last = speaker
            logger.debug("New %s (best score %.2f)", speaker.label, score)
            return SpeakerMatch(speaker.label, 1.0, is_new=True)

        confident = biased >= self.threshold
        if confident:
            self._update(best, emb, duration)
            best = self._merge_duplicates(best)
        self._last = best
        return SpeakerMatch(best.label, score, confident=confident)

    def _update(self, speaker: _Speaker, emb: np.ndarray, duration: float) -> None:
        w = min(speaker.seconds, self.max_centroid_seconds)
        centroid = speaker.centroid * w + emb * duration
        speaker.centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        speaker.seconds += duration
        speaker.count += 1

    def _merge_duplicates(self, speaker: _Speaker) -> _Speaker:
        """Fold discovered speakers that turned out to be `speaker` (or vice versa) together."""
        for other in list(self.speakers):
            if other is speaker or other not in self.speakers or (other.enrolled and speaker.enrolled):
                continue
            if float(other.centroid @ speaker.centroid) < self.merge_threshold:
                continue
            keep, gone = (other, speaker) if other.enrolled or (
                not speaker.enrolled and self.speakers.index(other) < self.speakers.index(speaker)
            ) else (speaker, other)
            centroid = keep.centroid * keep.seconds + gone.centroid * gone.seconds
            keep.centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            keep.seconds += gone.seconds
            keep.count += gone.count
            self.speakers.remove(gone)
            self.aliases[gone.label] = keep.label
            logger.debug("Merged %s into %s", gone.label, keep.label)
            speaker = keep
        return speaker


@dataclass
class LiveSpeakerPreview:
    """Provisional "who is talking now" while an utterance is still in progress.

    Call `update()` after each `VadSegmenter.accept()`; it returns a match at most every
    `interval` seconds of speech, scored on the last `window` seconds.
    """

    classifier: SpeakerClassifier
    vad: VadSegmenter
    interval: float = 0.5
    window: float = 2.0
    min_speech: float = 1.0
    _last_at: float = field(default=0.0, repr=False)

    def update(self) -> Optional[SpeakerMatch]:
        live = self.vad.live_seconds
        if live < self._last_at:  # a new utterance started
            self._last_at = 0.0
        if live < self.min_speech or live - self._last_at < self.interval:
            return None
        self._last_at = live
        return self.classifier.peek(self.vad.live_audio(self.window))
