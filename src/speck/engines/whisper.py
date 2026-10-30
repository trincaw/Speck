"""Whisper engine on CTranslate2 (faster-whisper): beam search, VAD filtering, hotword biasing."""

from __future__ import annotations

import logging
import time
from typing import Any, Generator, Optional

import numpy as np
import torch

from speck.base import AudioInput, BaseASREngine, TranscriptionResult, rtf
from speck.config import get_beam_size, get_language, get_whisper_compute_type
from speck.utils import LanguageTier, iso639_1

logger = logging.getLogger(__name__)

WHISPER_LANGUAGES = {
    "multilingual": LanguageTier(
        name="Whisper multilingual (99)",
        description="Any ISO 639-1 code (en, it, es, ...); 'auto' runs language identification.",
        locales=[],
    )
}


class WhisperASR(BaseASREngine):
    """OpenAI Whisper through faster-whisper. `model_id` is a size ('base', 'large-v3-turbo') or a CT2 repo."""

    def __init__(
        self,
        model_id: str = "large-v3-turbo",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        compute_type: Optional[str] = None,
        beam_size: Optional[int] = None,
        vad_filter: bool = True,
        hotwords: Optional[list[str]] = None,
    ):
        super().__init__(model_id=model_id, device=device, dtype=dtype, hotwords=hotwords)
        default_compute = get_whisper_compute_type() if self.device == "cuda" else "int8"
        self.compute_type = compute_type or default_compute
        self.beam_size = beam_size or get_beam_size()
        self.vad_filter = vad_filter
        self.model = None

    @property
    def name(self) -> str:
        return f"Whisper {self.model_id} (faster-whisper)"

    @property
    def sampling_rate(self) -> int:
        return 16000

    @property
    def supported_languages(self) -> dict[str, Any]:
        return WHISPER_LANGUAGES

    def load(self) -> WhisperASR:
        if self.model is None:
            from faster_whisper import WhisperModel

            logger.info("Loading Whisper %s on %s (%s)...", self.model_id, self.device, self.compute_type)
            self.model = WhisperModel(self.model_id, device=self.device, compute_type=self.compute_type)
        self._is_loaded = True
        return self

    def _segments(self, wav: np.ndarray, language: str, **kwargs: Any):
        """Lazy (segments, info). A fixed language skips the identification pass and avoids false switches."""
        lang = iso639_1(language) or iso639_1(get_language())
        options = dict(
            language=lang,
            beam_size=self.beam_size,
            # Each segment decoded on its own audio: stops repetition loops from propagating
            condition_on_previous_text=False,
            vad_filter=self.vad_filter,  # silence is where Whisper hallucinates
            hotwords=", ".join(self.hotwords) or None,
        )
        return self.model.transcribe(wav, **{**options, **kwargs})

    def transcribe(
        self,
        audio: AudioInput,
        language: str = "auto",
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> TranscriptionResult:
        self._ensure_loaded()
        wav, duration = self._prepare_audio(audio)

        start = time.perf_counter()
        segments, info = self._segments(wav, language, **kwargs)
        parsed = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
        elapsed = time.perf_counter() - start

        raw_text = " ".join(s["text"] for s in parsed).strip()
        return TranscriptionResult(
            text=self._postprocess(raw_text),
            raw_text=raw_text,
            detected_language=info.language,
            duration_seconds=duration,
            inference_seconds=elapsed,
            real_time_factor=rtf(elapsed, duration),
            segments=parsed,
        )

    def stream_transcribe(
        self,
        audio: AudioInput,
        language: str = "auto",
        lookahead: int = 6,
        skip_special_tokens: bool = True,
        simulate_realtime: bool = False,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Yields each segment as soon as it is decoded (Whisper segments are ~5-30 s, not tokens)."""
        self._ensure_loaded()
        wav, _ = self._prepare_audio(audio)
        segments, _ = self._segments(wav, language, **kwargs)
        for seg in segments:
            yield self._postprocess(seg.text.strip()) + " "
