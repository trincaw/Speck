"""Speech recognition models and engine interfaces.

Main entry point for ASR inference engines in Speck.
"""

from __future__ import annotations

from speck.base import BaseASREngine, BenchmarkResult, TranscriptionResult
from speck.engines.nemotron import NemotronASR
from speck.engines.voxtral import VoxtralASR, VoxtralMiniTranscribe
from speck.factory import get_engine, list_available_models, register_engine

__all__ = [
    "BaseASREngine",
    "TranscriptionResult",
    "BenchmarkResult",
    "NemotronASR",
    "VoxtralASR",
    "VoxtralMiniTranscribe",
    "get_engine",
    "register_engine",
    "list_available_models",
]

