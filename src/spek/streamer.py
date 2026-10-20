"""Push-based live streaming sessions (microphone, WebSocket, queues)."""

from __future__ import annotations

import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Generator, Optional

import numpy as np

from spek.utils import to_mono_float32

if TYPE_CHECKING:
    from spek.base import BaseASREngine
    from spek.speakers import SpeakerClassifier, SpeakerMatch
    from spek.vad import Utterance

logger = logging.getLogger(__name__)


class AudioBuffer:
    """Growable float32 buffer addressed by absolute sample index.

    Appends are amortized O(1) (capacity doubling) and consumed samples can be
    discarded, so long sessions don't grow memory or re-copy the whole history.
    """

    def __init__(self, capacity: int = 16000 * 10):
        self._data = np.empty(capacity, dtype=np.float32)
        self._start = 0  # absolute index of _data[0]
        self._size = 0

    def __len__(self) -> int:
        """Total samples ever appended (absolute end index)."""
        return self._start + self._size

    def append(self, chunk: np.ndarray) -> None:
        needed = self._size + len(chunk)
        if needed > len(self._data):
            grown = np.empty(max(needed, 2 * len(self._data)), dtype=np.float32)
            grown[: self._size] = self._data[: self._size]
            self._data = grown
        self._data[self._size : needed] = chunk
        self._size = needed

    def slice(self, start: int, end: int) -> np.ndarray:
        start, end = max(start, self._start), min(end, len(self))
        return self._data[start - self._start : max(start, end) - self._start].copy()

    def discard_before(self, index: int) -> None:
        drop = min(max(0, index - self._start), self._size)
        if drop:
            self._data[: self._size - drop] = self._data[drop : self._size]
            self._size -= drop
            self._start += drop


class BaseAudioStreamSession(ABC):
    """Base class for push-based audio streaming sessions.

    Producers call `push_audio()` then `finish()`; a worker thread consumes the audio and
    emits text through `on_token`, `iter_tokens()` and `transcript`.
    """

    def __init__(
        self,
        engine: BaseASREngine,
        language: str = "auto",
        lookahead: int = 6,
        skip_special_tokens: bool = True,
        on_token: Optional[Callable[[str], None]] = None,
    ):
        self.engine = engine
        self.language = language
        self.lookahead = lookahead
        self.skip_special_tokens = skip_special_tokens
        self.on_token = on_token
        self.sr = engine.sampling_rate

        self._buffer = AudioBuffer()
        self._cond = threading.Condition()
        self._finished = False
        self._stopped = False
        self._tokens: queue.Queue[Optional[str]] = queue.Queue()
        self._transcript: list[str] = []
        self._worker: Optional[threading.Thread] = None

    def start(self) -> BaseAudioStreamSession:
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        return self

    def push_audio(self, pcm_chunk: np.ndarray) -> None:
        """Push mono float32 audio at the engine's sampling rate."""
        if self._finished:
            return
        chunk = to_mono_float32(pcm_chunk).reshape(-1)
        with self._cond:
            self._buffer.append(chunk)
            self._cond.notify_all()

    def finish(self) -> None:
        """Signal end of audio; the worker flushes remaining output and exits."""
        with self._cond:
            self._finished = True
            self._cond.notify_all()

    def stop(self) -> None:
        """Abort the session immediately."""
        with self._cond:
            self._stopped = self._finished = True
            self._cond.notify_all()
        self.join(timeout=2.0)

    def join(self, timeout: Optional[float] = None) -> bool:
        """Wait for the worker to finish. Returns True if it did."""
        if self._worker:
            self._worker.join(timeout)
            return not self._worker.is_alive()
        return True

    def iter_tokens(self, timeout: Optional[float] = None) -> Generator[str, None, None]:
        while True:
            try:
                token = self._tokens.get(timeout=timeout)
            except queue.Empty:
                if self._stopped or (self._finished and not self._worker.is_alive()):
                    return
                continue
            if token is None:
                return
            yield token

    @property
    def transcript(self) -> str:
        return "".join(self._transcript).strip()

    def _read(self, start: int, end: int) -> Optional[np.ndarray]:
        """Block until samples [start, end) exist. Returns fewer samples only at end of
        stream, and None once the stream is exhausted or stopped."""
        with self._cond:
            self._cond.wait_for(lambda: self._finished or len(self._buffer) >= end)
            if self._stopped or (self._finished and start >= len(self._buffer)):
                return None
            chunk = self._buffer.slice(start, end)
            self._buffer.discard_before(start)
            return chunk

    def _read_available(self, start: int, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Return all samples from `start` (waiting up to `timeout`), None when exhausted."""
        with self._cond:
            self._cond.wait_for(lambda: self._finished or len(self._buffer) > start, timeout)
            if self._stopped or (self._finished and start >= len(self._buffer)):
                return None
            chunk = self._buffer.slice(start, len(self._buffer))
            self._buffer.discard_before(start + len(chunk))
            return chunk

    def _emit_token(self, token: str) -> None:
        if not token:
            return
        self._transcript.append(token)
        self._tokens.put(token)
        if self.on_token:
            try:
                self.on_token(token)
            except Exception as err:
                logger.error("Error in on_token callback: %s", err)

    def _run(self) -> None:
        try:
            self._run_inference_worker()
        except Exception as err:
            logger.error("%s worker error: %s", type(self).__name__, err, exc_info=True)
        finally:
            self._tokens.put(None)

    @abstractmethod
    def _run_inference_worker(self) -> None:
        """Consume audio via `_read*` and emit text via `_emit_token`."""


class UtteranceStreamSession(BaseAudioStreamSession):
    """VAD-segmented session: each finished utterance is processed exactly once.

    Avoids re-transcribing the whole growing buffer on every tick (O(n^2)), so cost
    is linear in audio length and latency is bounded by utterance length.
    """

    def __init__(self, *args, vad_min_silence: float = 0.4, vad_max_speech: float = 15.0, **kwargs):
        super().__init__(*args, **kwargs)
        from spek.vad import VadSegmenter
        self._vad = VadSegmenter(self.sr, min_silence=vad_min_silence, max_speech=vad_max_speech)

    def _run_inference_worker(self) -> None:
        cursor = 0
        while (chunk := self._read_available(cursor)) is not None:
            cursor += len(chunk)
            for utt in self._vad.accept(chunk):
                self._handle(utt)
            self._on_audio()
        if not self._stopped:
            for utt in self._vad.flush():
                self._handle(utt)

    def _on_audio(self) -> None:
        """Called after each chunk reaches the VAD (utterances may still be in progress)."""

    def _handle(self, utt: Utterance) -> None:
        try:
            self._emit_token(self._process_utterance(utt))
        except Exception as err:
            logger.warning("Utterance at %.2fs failed: %s", utt.start, err)

    def _process_utterance(self, utt: Utterance) -> str:
        text = self.engine.transcribe(
            utt.samples, language=self.language, skip_special_tokens=self.skip_special_tokens
        ).text.strip()
        return f" {text}" if text and self._transcript else text


class SpeakerStreamSession(UtteranceStreamSession):
    """Speaker-labelled live transcript from any engine: VAD → speaker turns → ASR per turn.

    Each finished utterance is first split where the voice changes (`SpeakerClassifier.segment`),
    then every single-speaker turn is transcribed on its own. Labels come from the audio
    itself, so no ASR timestamps are needed and any engine with `transcribe()` works.

    - `on_token` gets tagged text as turns finish (a tag whenever the speaker changes).
    - `on_segment` gets each turn as a dict (speaker, start, end, text, score, confident).
    - `on_speaker` gets a provisional `SpeakerMatch` while someone is still talking.
    - `segments` / `transcript` follow later speaker merges, so they are the corrected view.
    """

    def __init__(
        self,
        *args,
        speakers: SpeakerClassifier,
        tag_format: str = "block",
        on_segment: Optional[Callable[[dict], None]] = None,
        on_speaker: Optional[Callable[[SpeakerMatch], None]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from spek.mixed import TranscriptMixer
        from spek.speakers import LiveSpeakerPreview

        if speakers.embedder.sample_rate != self.sr:
            raise ValueError(f"Speaker embedder runs at {speakers.embedder.sample_rate} Hz, engine at {self.sr} Hz")
        self.classifier = speakers
        self.tag_format = tag_format
        self.on_segment = on_segment
        self.on_speaker = on_speaker
        self._mixer = TranscriptMixer(tag_format=tag_format, on_token=self._emit_token)
        self._preview = LiveSpeakerPreview(speakers, self._vad) if on_speaker else None
        self._turns: list[dict] = []

    def _on_audio(self) -> None:
        if self._preview and (match := self._preview.update()):
            self._notify(self.on_speaker, match)

    def _handle(self, utt: Utterance) -> None:
        try:
            turns = self.classifier.segment(utt.samples)
        except Exception as err:
            logger.warning("Speaker segmentation at %.2fs failed: %s", utt.start, err)
            return
        for turn in turns:
            piece = utt.samples[int(turn.start * self.sr) : int(turn.end * self.sr)]
            try:
                text = self.engine.transcribe(
                    piece, language=self.language, skip_special_tokens=self.skip_special_tokens
                ).text.strip()
            except Exception as err:
                logger.warning("Turn at %.2fs failed: %s", utt.start + turn.start, err)
                continue
            if not text:
                continue
            seg = {
                "speaker": turn.match.label,
                "start": round(utt.start + turn.start, 2),
                "end": round(utt.start + turn.end, 2),
                "text": text,
                "score": round(turn.match.score, 3),
                "confident": turn.match.confident,
            }
            self._turns.append(seg)
            self._mixer.emit(seg["speaker"], text, whole=True)
            self._notify(self.on_segment, dict(seg))

    def _notify(self, callback: Optional[Callable], value) -> None:
        if callback:
            try:
                callback(value)
            except Exception as err:
                logger.error("Error in speaker callback: %s", err)

    @property
    def segments(self) -> list[dict]:
        """Every turn so far, with labels of merged speakers resolved."""
        return [{**s, "speaker": self.classifier.resolve(s["speaker"])} for s in list(self._turns)]

    @property
    def transcript(self) -> str:
        from spek.mixed import TranscriptMixer

        mixer = TranscriptMixer(tag_format=self.tag_format)
        for s in self.segments:
            mixer.emit(s["speaker"], s["text"], whole=True)
        return mixer.transcript


def LiveAudioStreamSession(engine: BaseASREngine, **kwargs) -> BaseAudioStreamSession:
    """Create the engine-appropriate live session (kept for backward compatibility)."""
    return engine.create_stream_session(**kwargs)
