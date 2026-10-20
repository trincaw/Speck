"""Sherpa-ONNX engine: speaker diarization + per-turn ASR with speaker-tagged output.

Offline: Pyannote segmentation + CAM++ embeddings cluster speakers, then Whisper (ONNX)
transcribes every turn in one batched decode.
Live: Silero VAD cuts utterances, speaker changes inside them are detected, an online
classifier labels each turn and Whisper transcribes it.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Union

import numpy as np
import torch

from spek.base import AudioInput, BaseASREngine, TranscriptionResult, rtf
from spek.config import (
    get_sherpa_asr_decoder,
    get_sherpa_asr_encoder,
    get_sherpa_asr_tokens,
    get_sherpa_asr_type,
    get_sherpa_embedding_model,
    get_sherpa_num_clusters,
    get_sherpa_num_threads,
    get_sherpa_segmentation_model,
    get_sherpa_tag_format,
    get_sherpa_threshold,
    get_speaker_threshold,
)
from spek.speakers import (
    EMBEDDING_FILE,
    EMBEDDING_REPO,
    LiveSpeakerPreview,
    SpeakerClassifier,
    SpeakerEmbedder,
    Utterance,
)
from spek.streamer import UtteranceStreamSession
from spek.utils import LanguageTier, resolve_hf_file

logger = logging.getLogger(__name__)

DEFAULT_SEGMENTATION_REPO = "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0"
DEFAULT_SEGMENTATION_FILE = "model.int8.onnx"

DEFAULT_WHISPER_REPO = "csukuangfj/sherpa-onnx-whisper-tiny.en"
DEFAULT_WHISPER_ENCODER = "tiny.en-encoder.int8.onnx"
DEFAULT_WHISPER_DECODER = "tiny.en-decoder.int8.onnx"
DEFAULT_WHISPER_TOKENS = "tiny.en-tokens.txt"

MIN_SEGMENT_SECONDS = 0.1

SHERPA_LANGUAGES = {
    "core": LanguageTier(
        name="Sherpa-ONNX Core Languages",
        description="Multilingual speech recognition & speaker diarization.",
        locales=["en", "es", "fr", "de", "it", "zh", "ja", "ko"],
    )
}

SHERPA_STREAMING_PRESETS = {
    0: {"name": "ultra-low-latency", "latency_ms": 300.0, "description": "300ms chunk (Live stream)"},
    3: {"name": "balanced", "latency_ms": 600.0, "description": "600ms chunk (Balanced)"},
    6: {"name": "high-accuracy", "latency_ms": 1200.0, "description": "1200ms chunk (High Accuracy - default)"},
    13: {"name": "maximum-accuracy", "latency_ms": 2500.0, "description": "2500ms chunk (Maximum Accuracy)"},
}


def merge_speaker_segments(segments: list[dict[str, Any]], max_gap: float = 0.8) -> list[dict[str, Any]]:
    """Merge consecutive segments of the same speaker separated by at most `max_gap` seconds."""
    merged: list[dict[str, Any]] = []
    for seg in segments:
        prev = merged[-1] if merged else None
        if (
            prev
            and seg.get("speaker_id") == prev.get("speaker_id")
            and seg.get("start", 0.0) - prev.get("end", 0.0) <= max_gap
        ):
            prev["end"] = max(prev.get("end", 0.0), seg.get("end", 0.0))
            prev["text"] = " ".join(t for t in (prev.get("text", "").strip(), seg.get("text", "").strip()) if t)
        else:
            merged.append(dict(seg))
    return merged


def format_speaker_tagged_transcript(segments: list[dict[str, Any]], format_style: str = "block") -> str:
    """'block' -> '[Speaker 0]: text' per line; 'inline' -> '[Speaker 0] text [Speaker 1] text'."""
    valid = [s for s in segments if s.get("text", "").strip()]
    if format_style.lower() == "inline":
        return " ".join(f"[{s['speaker']}] {s['text'].strip()}" for s in valid)
    return "\n".join(f"[{s['speaker']}]: {s['text'].strip()}" for s in valid)


class SherpaSpeakerTagger:
    """Offline speaker diarization using sherpa-onnx."""

    def __init__(
        self,
        segmentation_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        num_clusters: int = -1,
        threshold: float = 0.9,
        min_duration_on: float = 0.2,
        min_duration_off: float = 0.3,
        num_threads: int = 2,
        provider: str = "cpu",
    ):
        self.segmentation_model_path = segmentation_model
        self.embedding_model_path = embedding_model
        self.num_clusters = num_clusters
        self.threshold = threshold
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.num_threads = num_threads
        self.provider = provider
        self._diarizer = None
        self._config = None
        self._clustering = (num_clusters, threshold)

    def load(self) -> None:
        if self._diarizer is not None:
            return
        import sherpa_onnx

        seg_path = resolve_hf_file(self.segmentation_model_path, DEFAULT_SEGMENTATION_REPO, DEFAULT_SEGMENTATION_FILE)
        embed_path = resolve_hf_file(self.embedding_model_path, EMBEDDING_REPO, EMBEDDING_FILE)

        self._config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg_path),
                num_threads=self.num_threads,
                provider=self.provider,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=embed_path, num_threads=self.num_threads, provider=self.provider
            ),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=self.num_clusters, threshold=self.threshold),
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )
        if not self._config.validate():
            raise ValueError(f"Invalid Sherpa speaker diarization configuration: {self._config}")

        self._diarizer = sherpa_onnx.OfflineSpeakerDiarization(self._config)
        logger.info("SherpaSpeakerTagger loaded (seg=%s, embed=%s)", Path(seg_path).name, Path(embed_path).name)

    def _set_clustering(self, num_clusters: int, threshold: float) -> None:
        """Apply per-call clustering overrides without reloading the models."""
        if (num_clusters, threshold) == self._clustering:
            return
        import sherpa_onnx

        self._config.clustering = sherpa_onnx.FastClusteringConfig(num_clusters=num_clusters, threshold=threshold)
        self._diarizer.set_config(self._config)
        self._clustering = (num_clusters, threshold)

    def diarize(
        self,
        samples: np.ndarray,
        num_clusters: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Diarize 16 kHz mono audio into speaker segments sorted by start time."""
        self.load()
        self._set_clustering(
            self.num_clusters if num_clusters is None else num_clusters,
            self.threshold if threshold is None else threshold,
        )
        result = self._diarizer.process(np.asarray(samples, dtype=np.float32))
        return [
            {
                "speaker": f"Speaker {s.speaker}",
                "speaker_id": int(s.speaker),
                "start": round(float(s.start), 2),
                "end": round(float(s.end), 2),
                "duration": round(float(s.duration), 2),
                "confidence": round(float(s.confidence), 3),
            }
            for s in result.sort_by_start_time()
        ]


class SherpaOnnxASR(BaseASREngine):
    """Speaker-aware ASR: diarization + Whisper ONNX, output tagged per speaker turn."""

    def __init__(
        self,
        model_id: str = "sherpa-onnx",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        segmentation_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        asr_type: Optional[str] = None,
        asr_encoder: Optional[str] = None,
        asr_decoder: Optional[str] = None,
        asr_tokens: Optional[str] = None,
        num_clusters: Optional[int] = None,
        clustering_threshold: Optional[float] = None,
        min_duration_on: float = 0.2,
        min_duration_off: float = 0.3,
        tag_speakers: bool = True,
        tag_format: Optional[str] = None,
        num_threads: Optional[int] = None,
    ):
        super().__init__(model_id=model_id, device=device, dtype=dtype)

        self.segmentation_model_path = segmentation_model or get_sherpa_segmentation_model()
        self.embedding_model_path = embedding_model or get_sherpa_embedding_model()
        self.asr_type = (asr_type or get_sherpa_asr_type()).lower().strip()
        self.asr_encoder_path = asr_encoder or get_sherpa_asr_encoder()
        self.asr_decoder_path = asr_decoder or get_sherpa_asr_decoder()
        self.asr_tokens_path = asr_tokens or get_sherpa_asr_tokens()
        self.num_clusters = get_sherpa_num_clusters() if num_clusters is None else num_clusters
        self.clustering_threshold = get_sherpa_threshold() if clustering_threshold is None else clustering_threshold
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.tag_speakers = tag_speakers
        self.tag_format = (tag_format or get_sherpa_tag_format()).lower().strip()
        self.num_threads = num_threads or get_sherpa_num_threads()
        self.provider = self._onnx_provider()

        self._tagger: Optional[SherpaSpeakerTagger] = None
        self._recognizer = None
        self._embedder: Optional[SpeakerEmbedder] = None
        self._decode_lock = threading.Lock()  # live sessions (e.g. mic + system audio) share the recognizer

    def _onnx_provider(self) -> str:
        if self.device != "cuda":
            return "cpu"
        try:
            import onnxruntime as ort
            return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def name(self) -> str:
        provider_name = "GPU/CUDA" if self.provider == "cuda" else "ONNX Runtime (CPU)"
        return f"Sherpa-ONNX Speech Tagging & Diarization ({provider_name})"

    @property
    def sampling_rate(self) -> int:
        return 16000

    @property
    def supported_languages(self) -> dict[str, Any]:
        return SHERPA_LANGUAGES

    @property
    def streaming_presets(self) -> dict[Union[int, str], dict[str, Any]]:
        return SHERPA_STREAMING_PRESETS

    def load(self) -> SherpaOnnxASR:
        if self._is_loaded:
            return self
        import sherpa_onnx

        if self.asr_type != "whisper":
            raise ValueError(f"Unsupported Sherpa ASR type '{self.asr_type}'. Supported: 'whisper'.")

        self._tagger = SherpaSpeakerTagger(
            segmentation_model=self.segmentation_model_path,
            embedding_model=self.embedding_model_path,
            num_clusters=self.num_clusters,
            threshold=self.clustering_threshold,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
            num_threads=self.num_threads,
            provider=self.provider,
        )
        self._tagger.load()
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=resolve_hf_file(self.asr_encoder_path, DEFAULT_WHISPER_REPO, DEFAULT_WHISPER_ENCODER),
            decoder=resolve_hf_file(self.asr_decoder_path, DEFAULT_WHISPER_REPO, DEFAULT_WHISPER_DECODER),
            tokens=resolve_hf_file(self.asr_tokens_path, DEFAULT_WHISPER_REPO, DEFAULT_WHISPER_TOKENS),
            language="en",
            provider=self.provider,
            num_threads=self.num_threads,
        )
        self._is_loaded = True
        logger.info("SherpaOnnxASR initialized.")
        return self

    @property
    def embedder(self) -> SpeakerEmbedder:
        """Shared speaker-embedding model for live classification (lazy)."""
        if self._embedder is None:
            self._embedder = SpeakerEmbedder(self.embedding_model_path, self.num_threads, self.provider)
        return self._embedder

    def diarize(
        self,
        audio: AudioInput,
        num_clusters: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        self._ensure_loaded()
        wav, _ = self._prepare_audio(audio)
        return self._tagger.diarize(wav, num_clusters=num_clusters, threshold=threshold)

    def _transcribe_batch(self, chunks: list[np.ndarray]) -> list[str]:
        """Decode many chunks in a single batched call (parallel across ONNX threads)."""
        min_len = int(self.sampling_rate * MIN_SEGMENT_SECONDS)
        idx = [i for i, c in enumerate(chunks) if len(c) >= min_len]
        streams = []
        for i in idx:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(self.sampling_rate, chunks[i])
            streams.append(stream)
        if streams:
            with self._decode_lock:
                self._recognizer.decode_streams(streams)
        texts = [""] * len(chunks)
        for i, stream in zip(idx, streams):
            texts[i] = stream.result.text.strip()
        return texts

    def transcribe(
        self,
        audio: AudioInput,
        language: str = "auto",
        skip_special_tokens: bool = True,
        tag_speakers: Optional[bool] = None,
        tag_format: Optional[str] = None,
        num_clusters: Optional[int] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe with automatic speaker detection.

        Args:
            tag_speakers: Prefix speaker tags in `text` (default: engine setting).
            tag_format: 'block' or 'inline'.
            num_clusters: Force the speaker count (<= 0 auto-detects).
            threshold: Diarization clustering threshold.
        """
        self._ensure_loaded()
        should_tag = self.tag_speakers if tag_speakers is None else tag_speakers
        style = (tag_format or self.tag_format).lower().strip()

        wav, duration = self._prepare_audio(audio)
        sr = self.sampling_rate
        start = time.perf_counter()

        segments = self._tagger.diarize(wav, num_clusters=num_clusters, threshold=threshold)
        if not segments:
            segments = [{"speaker": "Speaker 0", "speaker_id": 0, "start": 0.0, "end": round(duration, 2),
                         "duration": round(duration, 2), "confidence": 1.0}]

        chunks = [wav[max(0, int(s["start"] * sr)) : int(s["end"] * sr)] for s in segments]
        texts = self._transcribe_batch(chunks)
        segments = merge_speaker_segments(
            [{**seg, "text": text} for seg, text in zip(segments, texts) if text], max_gap=0.8
        )

        raw_text = " ".join(s["text"] for s in segments)
        text = format_speaker_tagged_transcript(segments, style) if should_tag else raw_text
        elapsed = time.perf_counter() - start

        return TranscriptionResult(
            text=text,
            raw_text=raw_text,
            detected_language="en" if language == "auto" else language,
            duration_seconds=duration,
            inference_seconds=elapsed,
            real_time_factor=rtf(elapsed, duration),
            segments=segments,
        )

    def stream_transcribe(
        self,
        audio: AudioInput,
        language: str = "auto",
        lookahead: int = 6,
        skip_special_tokens: bool = True,
        simulate_realtime: bool = False,
        tag_speakers: Optional[bool] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Diarize once, then yield one speaker-tagged line per turn."""
        should_tag = self.tag_speakers if tag_speakers is None else tag_speakers
        result = self.transcribe(audio, language, skip_special_tokens, tag_speakers=should_tag, **kwargs)
        for seg in result.segments or []:
            if simulate_realtime:
                time.sleep(max(0.1, min(1.0, (seg["end"] - seg["start"]) * 0.5)))
            yield (f"[{seg['speaker']}]: " if should_tag else "") + seg["text"] + "\n"

    def create_stream_session(
        self,
        tag_speakers: Optional[bool] = None,
        tag_format: Optional[str] = None,
        speaker_threshold: Optional[float] = None,
        max_speakers: Optional[int] = None,
        enroll: Optional[dict[str, AudioInput]] = None,
        classify: bool = True,
        **kwargs: Any,
    ) -> SherpaLiveStreamSession:
        """Live session with VAD + online speaker classification.

        Args:
            classify: False skips speaker classification (plain, untagged text).
            speaker_threshold: Cosine similarity needed to match a known speaker.
            max_speakers: Cap on discovered speakers (defaults to num_clusters when > 0).
            enroll: {name: audio} voice samples of known people, labelled by name.
            on_speaker: Callback for live speaker events (see SherpaLiveStreamSession).
        """
        self._ensure_loaded()
        classifier = None
        if classify:
            classifier = SpeakerClassifier(
                self.embedder,
                threshold=get_speaker_threshold() if speaker_threshold is None else speaker_threshold,
                max_speakers=max_speakers if max_speakers is not None else max(self.num_clusters, 0),
            )
            for label, audio in (enroll or {}).items():
                classifier.enroll(label, self._prepare_audio(audio)[0])
        return SherpaLiveStreamSession(
            engine=self,
            classifier=classifier,
            tag_speakers=self.tag_speakers if tag_speakers is None else tag_speakers,
            tag_format=(tag_format or self.tag_format).lower(),
            **kwargs,
        )


class SherpaLiveStreamSession(UtteranceStreamSession):
    """Live speaker-tagged transcription: VAD -> speaker-change detection -> Whisper per turn.

    `on_speaker(event)` receives {"label", "score", "provisional"} while someone is talking
    (provisional, from the last couple of seconds) and, with "start"/"end"/"text", once per
    finished turn.
    """

    engine: SherpaOnnxASR

    def __init__(
        self,
        *args,
        classifier: Optional[SpeakerClassifier],
        tag_speakers: bool = True,
        tag_format: str = "block",
        on_speaker: Optional[Callable[[dict[str, Any]], None]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.classifier = classifier
        self.tag_speakers = tag_speakers and classifier is not None
        self.tag_format = tag_format
        self.on_speaker = on_speaker
        self.segments: list[dict[str, Any]] = []
        self._last_speaker: Optional[str] = None
        self._preview = LiveSpeakerPreview(classifier, self._vad) if classifier else None

    def _on_audio(self) -> None:
        if self.on_speaker and self._preview and (match := self._preview.update()):
            self._notify({"label": match.label, "score": round(match.score, 3), "provisional": True})

    def _process_utterance(self, utt: Utterance) -> str:
        text = self.engine._transcribe_batch([utt.samples])[0]
        if not text:
            return ""
        if self.classifier is None:
            end = utt.start + len(utt.samples) / self.sr
            self.segments.append({"start": round(utt.start, 2), "end": round(end, 2), "text": text})
            return f" {text}" if self._transcript else text
        turns = self.classifier.segment(utt.samples)
        chunks = [utt.samples[int(t.start * self.sr) : int(t.end * self.sr)] for t in turns]
        texts = [text] if len(turns) == 1 else self.engine._transcribe_batch(chunks)

        out = []
        for turn, turn_text in zip(turns, texts):
            if not turn_text:
                continue
            segment = {"speaker": turn.match.label, "start": round(utt.start + turn.start, 2),
                       "end": round(utt.start + turn.end, 2), "score": round(turn.match.score, 3),
                       "text": turn_text}
            self.segments.append(segment)
            self._notify({"label": segment["speaker"], "score": segment["score"], "provisional": False,
                          "start": segment["start"], "end": segment["end"], "text": turn_text})
            out.append(self._format(turn.match.label, turn_text))
        for segment in self.segments:  # earlier turns of speakers merged since
            segment["speaker"] = self.classifier.resolve(segment["speaker"])
        return "".join(out)

    def _format(self, label: str, text: str) -> str:
        previous = self._last_speaker and self.classifier.resolve(self._last_speaker)
        self._last_speaker = label
        if previous is None:
            separator = ""
        elif not self.tag_speakers or previous == label:
            return f" {text}"
        else:
            separator = " " if self.tag_format == "inline" else "\n"

        if not self.tag_speakers:
            return text
        if self.tag_format == "inline":
            return f"{separator}[{label}] {text}"
        return f"{separator}[{label}]: {text}"

    def _notify(self, event: dict[str, Any]) -> None:
        if not self.on_speaker:
            return
        try:
            self.on_speaker(event)
        except Exception as err:
            logger.error("Error in on_speaker callback: %s", err)


# Backward-compatible alias
SherpaASR = SherpaOnnxASR
