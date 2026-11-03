"""Speech recognition: the engine interface, the built-in engines and the alias registry."""

from __future__ import annotations

from speck.asr.base import BaseASREngine, BenchmarkResult, TranscriptionResult
from speck.asr.engines.nemotron import LOOKAHEAD_CONFIGS, NemotronASR
from speck.asr.engines.parakeet import ParakeetASR
from speck.asr.engines.voxtral import VoxtralASR, VoxtralMiniTranscribe
from speck.asr.engines.whisper import WhisperASR
from speck.asr.languages import SUPPORTED_LANGUAGES, LanguageTier
from speck.asr.postprocess import correct_vocabulary, rover
from speck.asr.registry import get_engine, list_available_models, register_engine

__all__ = [
    "BaseASREngine",
    "TranscriptionResult",
    "BenchmarkResult",
    "NemotronASR",
    "ParakeetASR",
    "WhisperASR",
    "VoxtralASR",
    "VoxtralMiniTranscribe",
    "LOOKAHEAD_CONFIGS",
    "SUPPORTED_LANGUAGES",
    "LanguageTier",
    "correct_vocabulary",
    "rover",
    "get_engine",
    "register_engine",
    "list_available_models",
]
