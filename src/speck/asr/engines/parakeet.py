"""NVIDIA Parakeet-TDT engine (offline FastConformer-TDT via Transformers)."""

from __future__ import annotations

import logging
import time
from typing import Any, Generator, Optional

import numpy as np
import torch
from transformers import AutoModelForTDT, AutoProcessor

from speck.asr.base import AudioInput, BaseASREngine, TranscriptionResult, rtf
from speck.asr.languages import LanguageTier

logger = logging.getLogger(__name__)

PARAKEET_LANGUAGES = {
    "european": LanguageTier(
        name="Parakeet-TDT v3 (25 European languages)",
        description="Language is detected automatically; no language argument is needed.",
        locales=[
            "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it",
            "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
        ],
    )
}

# Full-attention encoder: memory grows quadratically, so longer audio is cut at pauses first
MAX_SINGLE_PASS_SECONDS = 40.0
MAX_SEGMENT_SECONDS = 30.0


class ParakeetASR(BaseASREngine):
    """Offline Parakeet-TDT: more accurate than the streaming Nemotron, at the cost of live latency."""

    def __init__(
        self,
        model_id: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        hotwords: Optional[list[str]] = None,
    ):
        super().__init__(model_id=model_id, device=device, dtype=dtype, hotwords=hotwords)
        self.processor = None
        self.model = None

    @property
    def name(self) -> str:
        return "NVIDIA Parakeet-TDT"

    @property
    def sampling_rate(self) -> int:
        return self.processor.feature_extractor.sampling_rate if self.processor else 16000

    @property
    def supported_languages(self) -> dict[str, Any]:
        return PARAKEET_LANGUAGES

    def load(self) -> ParakeetASR:
        if self.processor is None:
            logger.info("Loading processor for %s...", self.model_id)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        if self.model is None:
            logger.info("Loading model %s onto %s (%s)...", self.model_id, self.device, self.dtype)
            self.model = AutoModelForTDT.from_pretrained(self.model_id, dtype=self.dtype, device_map=self.device).eval()
        self._is_loaded = True
        return self

    def _decode(self, wav: np.ndarray) -> str:
        inputs = self.processor(wav, sampling_rate=self.sampling_rate, return_tensors="pt")
        inputs = inputs.to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=1024)
        sequences = output.sequences if hasattr(output, "sequences") else output
        return self.processor.decode(sequences[0], skip_special_tokens=True).strip()

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
        if duration <= MAX_SINGLE_PASS_SECONDS:
            text = self._decode(wav)
        else:
            from speck.streaming.vad import VadSegmenter

            vad = VadSegmenter(sample_rate=self.sampling_rate, max_speech=MAX_SEGMENT_SECONDS)
            utterances = vad.accept(wav) + vad.flush()
            text = " ".join(t for t in (self._decode(u.samples) for u in utterances) if t)
        elapsed = time.perf_counter() - start

        return TranscriptionResult(
            text=self._postprocess(text),
            raw_text=text,
            detected_language=None,
            duration_seconds=duration,
            inference_seconds=elapsed,
            real_time_factor=rtf(elapsed, duration),
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
        """Pseudo-streaming: decode the file, then emit it word by word."""
        words = self.transcribe(audio, language=language).text.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
