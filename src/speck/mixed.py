"""Mixed mode: microphone and system audio as two channels merged into one transcript.

Keeping the channels apart (instead of summing them) means the device already tells who
is talking: the microphone is you, system audio is everyone else.

- `TranscriptMixer` interleaves labelled text from several live sessions.
- `EchoGate` mutes microphone audio that is only the speakers' sound picked up again
  (laptop speakers, no headphones), so remote speech isn't transcribed twice.
- `MixedSession` wires one engine into a mic session and a system-audio session; with a
  `SpeakerClassifier`, remote audio is further split into "Speaker N" / enrolled names.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from speck.utils import to_mono_float32

if TYPE_CHECKING:
    from speck.speakers import SpeakerClassifier

logger = logging.getLogger(__name__)


class TranscriptMixer:
    """Merge text from several labelled sources into one speaker-tagged transcript.

    `emit(label, text)` accepts either incremental tokens (keep their own spacing) or
    whole turns (`whole=True`, joined with a space). A tag is written whenever the
    label changes, so interleaved channels stay readable.
    """

    def __init__(self, tag: bool = True, tag_format: str = "block", on_token: Optional[Callable[[str], None]] = None):
        self.tag = tag
        self.tag_format = tag_format
        self.on_token = on_token
        self.segments: list[dict[str, Any]] = []
        self._parts: list[str] = []
        self._last: Optional[str] = None
        self._lock = threading.Lock()

    def emit(self, label: str, text: str, whole: bool = False, **info: Any) -> None:
        if not text or not text.strip():
            return
        with self._lock:
            if label != self._last or not self.segments:
                text = text.lstrip()
                piece = self._header(label) + text
                self.segments.append({"speaker": label, "text": text, **info})
                self._last = label
            else:
                if whole and not text[0].isspace():
                    text = " " + text
                piece = text
                self.segments[-1]["text"] += text
                self.segments[-1].update(info)
            self._parts.append(piece)
        if self.on_token:
            try:
                self.on_token(piece)
            except Exception as err:
                logger.error("Error in on_token callback: %s", err)

    def _header(self, label: str) -> str:
        sep = "" if not self._parts else (" " if self.tag_format == "inline" or not self.tag else "\n")
        if not self.tag:
            return sep
        return f"{sep}[{label}] " if self.tag_format == "inline" else f"{sep}[{label}]: "

    @property
    def transcript(self) -> str:
        with self._lock:
            return "".join(self._parts).strip()


class EchoGate:
    """Mute microphone blocks whose loudness contour follows the system audio.

    Speaker sound picked up by the mic is a delayed, filtered copy of what is playing,
    so their short-time log-energy envelopes correlate strongly; your own voice doesn't
    track the remote audio. Each mic block is judged on the last `window` seconds, at
    delays up to `max_lag`. Heuristic: loud double-talk may be muted too, so prefer
    headphones or PipeWire's echo-cancel module when available.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window: float = 1.0,
        max_lag: float = 0.4,
        threshold: float = 0.75,
        frame: float = 0.02,
        min_reference_rms: float = 0.003,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_reference_rms = min_reference_rms
        self._frame = int(frame * sample_rate)
        self._win = int(window * sample_rate) // self._frame * self._frame
        self._lag = int(max_lag * sample_rate) // self._frame * self._frame
        self._ref = np.zeros(self._win + self._lag, dtype=np.float32)
        self._mic = np.zeros(self._win, dtype=np.float32)
        self._lock = threading.Lock()
        self.muted_seconds = 0.0

    def reference(self, block: np.ndarray) -> None:
        """Feed what the speakers are playing (system audio)."""
        block = to_mono_float32(block).reshape(-1)
        with self._lock:
            self._ref = np.concatenate([self._ref, block])[-len(self._ref) :]

    def filter(self, block: np.ndarray) -> np.ndarray:
        """Return the mic block, or silence if it is just echo."""
        block = to_mono_float32(block).reshape(-1)
        with self._lock:
            self._mic = np.concatenate([self._mic, block])[-len(self._mic) :]
            ref, mic = self._ref.copy(), self._mic.copy()
        if not self._is_echo(ref, mic):
            return block
        self.muted_seconds += len(block) / self.sample_rate
        return np.zeros_like(block)

    def _envelope(self, x: np.ndarray) -> np.ndarray:
        frames = x[: len(x) // self._frame * self._frame].reshape(-1, self._frame)
        return np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-8)

    def _is_echo(self, ref: np.ndarray, mic: np.ndarray) -> bool:
        if np.sqrt(np.mean(ref[-self._win :] ** 2)) < self.min_reference_rms:
            return False  # nothing is playing
        m = self._envelope(mic)
        r = self._envelope(ref)
        if m.std() < 1e-3:
            return False
        m = (m - m.mean()) / m.std()
        n, lags = len(m), len(r) - len(m)
        best = -1.0
        for lag in range(lags + 1):  # mic lags the reference by `lags - lag` frames
            seg = r[lag : lag + n]
            if seg.std() < 1e-3:
                continue
            best = max(best, float(np.mean(m * (seg - seg.mean()) / seg.std())))
        return best >= self.threshold


class MixedSession:
    """Live transcription of the microphone ("you") and system audio ("them") together.

    Feed `push_mic()` / `push_system()` from any capture (they may run on different
    threads). System audio is tagged `remote_label`, or with `speakers` each remote voice
    gets its own label (enrolled name, else "Speaker N" numbered as new voices appear).
    """

    def __init__(
        self,
        engine,
        me_label: str = "You",
        remote_label: str = "Remote",
        tag_speakers: bool = True,
        tag_format: str = "block",
        echo_gate: Optional[EchoGate] = None,
        on_token: Optional[Callable[[str], None]] = None,
        speakers: Optional[SpeakerClassifier] = None,
        **session_kwargs: Any,
    ):
        self.me_label, self.remote_label = me_label, remote_label
        self.mixer = TranscriptMixer(tag=tag_speakers, tag_format=tag_format, on_token=on_token)
        self.echo_gate = echo_gate
        self.speakers = speakers

        self.mic = engine.create_stream_session(
            on_token=lambda t: self.mixer.emit(me_label, t, channel="mic"), **session_kwargs
        )
        if speakers is None:
            self.system = engine.create_stream_session(
                on_token=lambda t: self.mixer.emit(remote_label, t, channel="system"), **session_kwargs
            )
        else:
            self.system = engine.create_stream_session(
                speakers=speakers,
                on_segment=lambda seg: self.mixer.emit(
                    seg["speaker"], seg["text"], whole=True, channel="system", start=seg["start"], end=seg["end"]
                ),
                **session_kwargs,
            )

    @property
    def sessions(self) -> tuple:
        return self.mic, self.system

    def start(self) -> MixedSession:
        for s in self.sessions:
            s.start()
        return self

    def push_mic(self, chunk: np.ndarray) -> None:
        self.mic.push_audio(self.echo_gate.filter(chunk) if self.echo_gate else chunk)

    def push_system(self, chunk: np.ndarray) -> None:
        if self.echo_gate:
            self.echo_gate.reference(chunk)
        self.system.push_audio(chunk)

    def finish(self) -> None:
        for s in self.sessions:
            s.finish()

    def join(self, timeout: Optional[float] = None) -> bool:
        return all([s.join(timeout) for s in self.sessions])

    def stop(self) -> None:
        for s in self.sessions:
            s.stop()

    @property
    def transcript(self) -> str:
        if self.speakers is None:
            return self.mixer.transcript
        mixer = TranscriptMixer(tag=self.mixer.tag, tag_format=self.mixer.tag_format)
        for seg in self.segments:  # re-render so merged remote speakers share one label
            mixer.emit(seg["speaker"], seg["text"], whole=True)
        return mixer.transcript

    @property
    def segments(self) -> list[dict[str, Any]]:
        """Channel turns; remote speakers that were merged later carry their final label."""
        with self.mixer._lock:
            segments = [dict(seg) for seg in self.mixer.segments]
        if self.speakers is not None:
            for seg in segments:
                if seg["channel"] == "system":
                    seg["speaker"] = self.speakers.resolve(seg["speaker"])
        return segments
