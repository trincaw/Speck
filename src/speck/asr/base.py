"""Model-agnostic abstract base classes and common data structures for ASR engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional, Union

import numpy as np
import torch

from speck.audio import normalize_audio, to_mono_float32
from speck.config import get_hotwords, get_normalize_audio

if TYPE_CHECKING:
    from speck.speakers import SpeakerClassifier
    from speck.streaming.sessions import BaseAudioStreamSession

AudioInput = Union[str, np.ndarray, torch.Tensor]


@dataclass
class TranscriptionResult:
    """ASR transcription result metadata."""
    text: str
    raw_text: str
    detected_language: Optional[str]
    duration_seconds: float
    inference_seconds: float
    real_time_factor: float
    segments: Optional[list[dict[str, Any]]] = None


def rtf(elapsed: float, duration: float) -> float:
    return elapsed / max(duration, 1e-4)


class BaseASREngine(ABC):
    """Abstract base class for all speech recognition engines in Speck."""

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        hotwords: Optional[list[str]] = None,
    ):
        self.model_id = model_id
        self.hotwords = hotwords if hotwords is not None else get_hotwords()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self._is_loaded = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of the model/engine."""

    @property
    @abstractmethod
    def sampling_rate(self) -> int:
        """Required audio sampling rate in Hz."""

    @property
    def supported_languages(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def load(self) -> BaseASREngine:
        """Load model weights or API client. Must be idempotent."""

    def _ensure_loaded(self) -> None:
        if not self._is_loaded:
            self.load()

    def _prepare_audio(self, audio: AudioInput, sampling_rate: Optional[int] = None) -> tuple[np.ndarray, float]:
        """Load and normalize audio to mono float32 at the engine's sampling rate."""
        target_sr = sampling_rate or self.sampling_rate
        if isinstance(audio, str):
            from transformers.audio_utils import load_audio
            wav = load_audio(audio, sampling_rate=target_sr)
        elif isinstance(audio, torch.Tensor):
            wav = audio.detach().cpu().numpy()
        else:
            wav = audio
        wav = to_mono_float32(wav)
        if get_normalize_audio():
            wav = normalize_audio(wav)
        return wav, len(wav) / target_sr

    def _postprocess(self, text: str) -> str:
        """Snap near-miss words to the configured hotword vocabulary (no-op without one)."""
        if not self.hotwords or not text:
            return text
        from speck.asr.postprocess import correct_vocabulary
        return correct_vocabulary(text, self.hotwords)

    @abstractmethod
    def transcribe(
        self,
        audio: AudioInput,
        language: str = "auto",
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Offline / batch audio transcription."""

    @abstractmethod
    def stream_transcribe(
        self,
        audio: AudioInput,
        language: str = "auto",
        skip_special_tokens: bool = True,
        simulate_realtime: bool = False,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Chunked streaming transcription yielding text increments."""

    def create_stream_session(
        self,
        language: str = "auto",
        skip_special_tokens: bool = True,
        on_token: Optional[Callable[[str], None]] = None,
        speakers: Optional[SpeakerClassifier] = None,
        **kwargs: Any,
    ) -> BaseAudioStreamSession:
        """Create a live push-based session. Default: VAD-segmented utterances sent to `transcribe`.

        With `speakers`, every utterance is split into single-speaker turns first and the
        output is speaker-labelled (`SpeakerStreamSession`), whatever the engine.
        """
        from speck.streaming.sessions import SpeakerStreamSession, UtteranceStreamSession
        if speakers is not None:
            kwargs["speakers"] = speakers
        return (SpeakerStreamSession if speakers is not None else UtteranceStreamSession)(
            engine=self,
            language=language,
            skip_special_tokens=skip_special_tokens,
            on_token=on_token,
            **kwargs,
        )
