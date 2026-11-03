"""Speck - model-agnostic speech recognition with real-time streaming and live speaker labels.

Package layout:
    speck.asr          engine interface, built-in engines, registry, transcript post-processing
    speck.speakers     speaker embeddings, live classification, offline diarization
    speck.streaming    VAD, push-based live sessions, mic + system audio mixing
    speck.capture      microphone and desktop (PulseAudio / PipeWire) audio sources
    speck.evaluation   speaker, transcript (WER/CER) and TTS metrics and pipelines
    speck.cli          the `speck` command

The names below are imported on first access, so `import speck` (and the CLI) stays fast
until a model is actually needed.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _version
from typing import TYPE_CHECKING, Any

_EXPORTS: dict[str, str] = {
    # ASR
    "BaseASREngine": "speck.asr",
    "TranscriptionResult": "speck.asr",
    "BenchmarkResult": "speck.asr",
    "NemotronASR": "speck.asr",
    "ParakeetASR": "speck.asr",
    "WhisperASR": "speck.asr",
    "VoxtralASR": "speck.asr",
    "VoxtralMiniTranscribe": "speck.asr",
    "get_engine": "speck.asr",
    "register_engine": "speck.asr",
    "list_available_models": "speck.asr",
    "LOOKAHEAD_CONFIGS": "speck.asr",
    "SUPPORTED_LANGUAGES": "speck.asr",
    "correct_vocabulary": "speck.asr",
    "rover": "speck.asr",
    # Speakers
    "SpeakerEmbedder": "speck.speakers",
    "SpeakerClassifier": "speck.speakers",
    "SpeakerTurn": "speck.speakers",
    "LiveSpeakerPreview": "speck.speakers",
    "PyannoteSpeakerRecognizer": "speck.speakers",
    # Streaming
    "VadSegmenter": "speck.streaming",
    "LiveAudioStreamSession": "speck.streaming",
    "SpeakerStreamSession": "speck.streaming",
    # Evaluation
    "compute_mcd": "speck.evaluation",
    "compute_f0_metrics": "speck.evaluation",
    "compute_speaker_similarity": "speck.evaluation",
    "compute_wer": "speck.evaluation",
    "compute_cer": "speck.evaluation",
    "compute_mer_and_wil": "speck.evaluation",
    "compute_speech_text_metrics": "speck.evaluation",
    "normalize_text": "speck.evaluation",
    "normalize_english_text": "speck.evaluation",
    "normalize_italian_text": "speck.evaluation",
    "TTSMetricsPipeline": "speck.evaluation",
    # Configuration
    "load_env_file": "speck.config",
}

__all__ = list(_EXPORTS)

try:
    __version__ = _version("speck")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0+unknown"


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'speck' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value  # cache: later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])


if TYPE_CHECKING:  # static view of the lazy exports above
    from speck.asr import (  # noqa: F401
        LOOKAHEAD_CONFIGS,
        SUPPORTED_LANGUAGES,
        BaseASREngine,
        BenchmarkResult,
        NemotronASR,
        ParakeetASR,
        TranscriptionResult,
        VoxtralASR,
        VoxtralMiniTranscribe,
        WhisperASR,
        correct_vocabulary,
        get_engine,
        list_available_models,
        register_engine,
        rover,
    )
    from speck.config import load_env_file  # noqa: F401
    from speck.evaluation import (  # noqa: F401
        TTSMetricsPipeline,
        compute_cer,
        compute_f0_metrics,
        compute_mcd,
        compute_mer_and_wil,
        compute_speaker_similarity,
        compute_speech_text_metrics,
        compute_wer,
        normalize_english_text,
        normalize_italian_text,
        normalize_text,
    )
    from speck.speakers import (  # noqa: F401
        LiveSpeakerPreview,
        PyannoteSpeakerRecognizer,
        SpeakerClassifier,
        SpeakerEmbedder,
        SpeakerTurn,
    )
    from speck.streaming import LiveAudioStreamSession, SpeakerStreamSession, VadSegmenter  # noqa: F401
