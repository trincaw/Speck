"""NVIDIA Nemotron 3.5 ASR Streaming engine (Cache-Aware FastConformer-RNNT via Transformers)."""

from __future__ import annotations

import copy
import logging
import time
from threading import Thread
from typing import Any, Callable, Generator, Iterator, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer

from speck.base import AudioInput, BaseASREngine, TranscriptionResult, rtf
from speck.streamer import BaseAudioStreamSession
from speck.utils import LOOKAHEAD_CONFIGS, SUPPORTED_LANGUAGES, parse_language_tag

logger = logging.getLogger(__name__)

# read(start, end) -> samples in [start, end); shorter only at end of stream, None when exhausted
SampleReader = Callable[[int, int], Optional[np.ndarray]]


def _pad_to(chunk: np.ndarray, size: int) -> np.ndarray:
    return np.pad(chunk, (0, size - len(chunk))) if len(chunk) < size else chunk


class NemotronASR(BaseASREngine):
    """NVIDIA Nemotron 3.5 ASR streaming & offline inference engine."""

    def __init__(
        self,
        model_id: str = "nvidia/nemotron-3.5-asr-streaming-0.6b",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(model_id=model_id, device=device, dtype=dtype)
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[AutoModelForRNNT] = None

    @property
    def name(self) -> str:
        return "Nemotron 3.5 ASR"

    @property
    def supported_languages(self) -> dict[str, Any]:
        return SUPPORTED_LANGUAGES

    @property
    def streaming_presets(self) -> dict[Union[int, str], dict[str, Any]]:
        return LOOKAHEAD_CONFIGS

    @property
    def sampling_rate(self) -> int:
        # Fixed by the checkpoint; avoids loading the processor just to answer this
        return self.processor.feature_extractor.sampling_rate if self.processor else 16000

    def load(self) -> NemotronASR:
        if self.processor is None:
            logger.info("Loading processor for %s...", self.model_id)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        if self.model is None:
            logger.info("Loading model %s onto %s (%s)...", self.model_id, self.device, self.dtype)
            self.model = AutoModelForRNNT.from_pretrained(self.model_id, dtype=self.dtype, device_map=self.device)
            self.model.eval()
        self._is_loaded = True
        return self

    def _features(self, wav: np.ndarray, language: str, **kwargs: Any):
        inputs = self.processor(wav, sampling_rate=self.sampling_rate, language=language, return_tensors="pt", **kwargs)
        return inputs.to(self.device, dtype=self.dtype)

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
        with torch.inference_mode():
            output = self.model.generate(
                **self._features(wav, language), return_dict_in_generate=True, max_new_tokens=1024
            )
        sequence = output.sequences[0]
        elapsed = time.perf_counter() - start

        raw_text = self.processor.decode(sequence, skip_special_tokens=False).strip()
        _, detected_lang = parse_language_tag(raw_text)
        text = self.processor.decode(sequence, skip_special_tokens=True).strip() if skip_special_tokens else raw_text
        text = self._postprocess(text)

        return TranscriptionResult(
            text=text,
            raw_text=raw_text,
            detected_language=detected_lang or (language if language != "auto" else None),
            duration_seconds=duration,
            inference_seconds=elapsed,
            real_time_factor=rtf(elapsed, duration),
        )

    def stream_tokens(
        self,
        read: SampleReader,
        language: str = "auto",
        lookahead: int = 6,
        skip_special_tokens: bool = True,
    ) -> Iterator[str]:
        """Cache-aware streaming decode over any sample source (file, mic, socket).

        Audio is pulled chunk by chunk through `read`, so the same code path drives
        offline streaming and live sessions.
        """
        self._ensure_loaded()
        p = self.processor
        p.set_num_lookahead_tokens(lookahead)
        hop, n_fft = p.feature_extractor.hop_length, p.feature_extractor.n_fft
        chunk_len = p.num_samples_per_audio_chunk

        first = read(0, p.num_samples_first_audio_chunk)
        if first is None:
            return
        first_inputs = self._features(
            _pad_to(first, p.num_samples_first_audio_chunk), language, is_streaming=True, is_first_audio_chunk=True
        )

        def features() -> Iterator[torch.Tensor]:
            yield first_inputs.input_features[:, : p.num_mel_frames_first_audio_chunk, :]
            frame = p.num_mel_frames_first_audio_chunk
            while True:
                start = frame * hop - n_fft // 2
                chunk = read(max(start, 0), start + chunk_len)
                if chunk is None:
                    return
                if start < 0:
                    chunk = np.pad(chunk, (-start, 0))
                yield self._features(
                    _pad_to(chunk, chunk_len), language, is_streaming=True, is_first_audio_chunk=False
                ).input_features
                frame += p.num_mel_frames_per_audio_chunk

        streamer = TextIteratorStreamer(p.tokenizer, skip_special_tokens=skip_special_tokens)

        # generate() keeps per-call streaming state on the model object itself, so concurrent
        # streams (e.g. mic + system audio in mixed mode) each get a shallow copy: own
        # attributes, shared weights and submodules (which hold no per-call state).
        model = copy.copy(self.model)

        def generate() -> None:
            try:
                with torch.inference_mode():
                    model.generate(
                        **{**first_inputs, "input_features": features()}, streamer=streamer, max_new_tokens=1_000_000
                    )
            except Exception:
                logger.exception("Nemotron streaming generation failed")
                streamer.end()  # unblock the consumer instead of hanging forever

        thread = Thread(target=generate, daemon=True)
        thread.start()
        try:
            yield from streamer
        finally:
            thread.join()

    def stream_transcribe(
        self,
        audio: AudioInput,
        language: str = "auto",
        lookahead: int = 6,
        skip_special_tokens: bool = True,
        simulate_realtime: bool = False,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        wav, _ = self._prepare_audio(audio)

        def read(start: int, end: int) -> Optional[np.ndarray]:
            if start >= len(wav):
                return None
            if simulate_realtime:
                time.sleep((end - start) / self.sampling_rate)
            return wav[start:end]

        yield from self.stream_tokens(read, language, lookahead, skip_special_tokens)

    def create_stream_session(self, **kwargs: Any) -> BaseAudioStreamSession:
        self._ensure_loaded()
        if kwargs.get("speakers") is not None:  # speaker turns need utterance-level decoding
            return super().create_stream_session(**kwargs)
        kwargs.pop("speakers", None)
        return NemotronLiveStreamSession(engine=self, **kwargs)


class NemotronLiveStreamSession(BaseAudioStreamSession):
    """Live session feeding pushed audio straight into Nemotron's cache-aware decoder."""

    engine: NemotronASR

    def _run_inference_worker(self) -> None:
        for token in self.engine.stream_tokens(self._read, self.language, self.lookahead, self.skip_special_tokens):
            if self._stopped:
                break
            self._emit_token(token)
