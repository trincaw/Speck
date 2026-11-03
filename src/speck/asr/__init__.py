"""Speech recognition: the engine interface, the built-in engines and the alias registry."""

from __future__ import annotations

from speck.asr.base import BaseASREngine, TranscriptionResult
from speck.asr.engines.parakeet import ParakeetASR
from speck.asr.engines.whisper import WhisperASR
from speck.asr.languages import LanguageTier
from speck.asr.postprocess import correct_vocabulary, rover
from speck.asr.registry import ENGINE_SPECS, EngineSpec, get_engine, get_spec, list_available_models, register_engine

__all__ = [
    "BaseASREngine",
    "TranscriptionResult",
    "ParakeetASR",
    "WhisperASR",
    "LanguageTier",
    "correct_vocabulary",
    "rover",
    "EngineSpec",
    "ENGINE_SPECS",
    "get_engine",
    "get_spec",
    "register_engine",
    "list_available_models",
]
