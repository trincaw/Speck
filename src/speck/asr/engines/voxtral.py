"""Mistral Voxtral Mini Transcribe 2 engine (Mistral cloud API or local Transformers weights)."""

from __future__ import annotations

import io
import logging
import time
from typing import Any, Generator, Optional, Union

import numpy as np
import requests
import soundfile as sf
import torch

from speck.asr.base import AudioInput, BaseASREngine, TranscriptionResult, rtf
from speck.config import get_mistral_api_key, get_voxtral_api_url, get_voxtral_backend, get_voxtral_model
from speck.asr.languages import LanguageTier, iso639_1 as _iso639_1

logger = logging.getLogger(__name__)

VOXTRAL_LANGUAGES = {
    "core": LanguageTier(
        name="Voxtral Core Languages (13)",
        description="High-accuracy transcription & audio understanding.",
        locales=["en", "fr", "es", "de", "it", "pt", "nl", "hi", "ja", "zh", "ru", "ar", "ko"],
    )
}

VOXTRAL_STREAMING_PRESETS = {
    0: {"name": "ultra-low-latency", "latency_ms": 200.0, "description": "200ms chunk (Real-time live)"},
    3: {"name": "balanced", "latency_ms": 500.0, "description": "500ms chunk (Balanced)"},
    6: {"name": "high-accuracy", "latency_ms": 1000.0, "description": "1000ms chunk (High Accuracy - default)"},
    13: {"name": "maximum-accuracy", "latency_ms": 2000.0, "description": "2000ms chunk (Maximum Accuracy)"},
}


class VoxtralASR(BaseASREngine):
    """Mistral Voxtral Mini Transcribe 2 via cloud API (default) or local Hugging Face weights."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        backend: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        super().__init__(model_id=model_id or get_voxtral_model(), device=device, dtype=dtype)
        self.backend = (backend or get_voxtral_backend()).lower()
        self.api_key = api_key if api_key is not None else get_mistral_api_key()
        self.api_url = api_url or get_voxtral_api_url()
        self._session: Optional[requests.Session] = None
        self._local_processor = None
        self._local_model = None

    @property
    def name(self) -> str:
        return f"Mistral Voxtral Mini Transcribe 2 ({'Cloud API' if self.backend == 'api' else 'Local'})"

    @property
    def sampling_rate(self) -> int:
        return 16000

    @property
    def supported_languages(self) -> dict[str, Any]:
        return VOXTRAL_LANGUAGES

    @property
    def streaming_presets(self) -> dict[Union[int, str], dict[str, Any]]:
        return VOXTRAL_STREAMING_PRESETS

    def load(self) -> VoxtralASR:
        if self._is_loaded:
            return self
        if self.backend == "api":
            # Keep-alive session: live mode sends many small requests
            self._session = requests.Session()
            logger.info("Voxtral API mode (model=%s, url=%s)", self.model_id, self.api_url)
        else:
            from transformers import AutoModelForCausalLM, AutoProcessor

            logger.info("Loading local Voxtral model %s on %s...", self.model_id, self.device)
            self._local_processor = AutoProcessor.from_pretrained(self.model_id)
            self._local_model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=self.dtype, device_map=self.device
            ).eval()
        self._is_loaded = True
        return self

    def transcribe(
        self,
        audio: AudioInput,
        language: str = "auto",
        skip_special_tokens: bool = True,
        diarize: bool = False,
        **kwargs: Any,
    ) -> TranscriptionResult:
        self._ensure_loaded()
        wav, duration = self._prepare_audio(audio)
        start = time.perf_counter()

        if self.backend == "api":
            text, detected_lang, segments = self._transcribe_api(wav, language, diarize)
        else:
            text, detected_lang, segments = self._transcribe_local(wav), _iso639_1(language), None

        elapsed = time.perf_counter() - start
        return TranscriptionResult(
            text=self._postprocess(text),
            raw_text=text,
            detected_language=detected_lang,
            duration_seconds=duration,
            inference_seconds=elapsed,
            real_time_factor=rtf(elapsed, duration),
            segments=segments,
        )

    def _transcribe_api(self, wav: np.ndarray, language: str, diarize: bool):
        if not (self.api_key or "").strip():
            raise ValueError(
                "Mistral API key is required for Voxtral Mini Transcribe 2 (API mode). "
                "Set MISTRAL_API_KEY in your .env file or environment."
            )

        buf = io.BytesIO()
        sf.write(buf, wav, self.sampling_rate, format="WAV", subtype="PCM_16")
        lang = _iso639_1(language)
        data = {
            "model": self.model_id,
            "diarize": str(diarize).lower(),
            "timestamp_granularities[]": "segment",
            **({"language": lang} if lang else {}),
        }
        post = self._session.post if self._session else requests.post
        try:
            response = post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": ("audio.wav", buf.getvalue(), "audio/wav")},
                data=data,
                timeout=120,
            )
            if response.status_code == 401:
                raise PermissionError("Mistral API authentication failed (401). Check MISTRAL_API_KEY.")
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Mistral Voxtral API error: {exc}") from exc

        return body.get("text", "").strip(), body.get("language", lang), body.get("segments", [])

    def _transcribe_local(self, wav: np.ndarray) -> str:
        inputs = self._local_processor(audio=wav, sampling_rate=self.sampling_rate, return_tensors="pt")
        inputs = inputs.to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            outputs = self._local_model.generate(**inputs, max_new_tokens=1024)
        return self._local_processor.decode(outputs[0], skip_special_tokens=True).strip()

    def stream_transcribe(
        self,
        audio: AudioInput,
        language: str = "auto",
        lookahead: int = 6,
        skip_special_tokens: bool = True,
        simulate_realtime: bool = False,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Pseudo-streaming: one request, then emit segments (or words) progressively."""
        result = self.transcribe(audio=audio, language=language, skip_special_tokens=skip_special_tokens, **kwargs)

        if result.segments:
            for seg in result.segments:
                if simulate_realtime:
                    time.sleep(max(0.05, seg.get("end", 0.0) - seg.get("start", 0.0)))
                yield seg.get("text", "") + " "
            return

        words = result.text.split()
        delay = min(0.2, result.duration_seconds / max(1, len(words))) if simulate_realtime else 0.0
        for i, word in enumerate(words):
            if delay:
                time.sleep(delay)
            yield word + (" " if i < len(words) - 1 else "")


# Backward-compatible alias
VoxtralMiniTranscribe = VoxtralASR
