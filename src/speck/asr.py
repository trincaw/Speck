"""Speech recognition models and engine interfaces.

Main entry point for ASR inference engines in Spek.
"""

from __future__ import annotations

from spek.base import BaseASREngine, BenchmarkResult, TranscriptionResult
from spek.engines.nemotron import NemotronASR
from spek.engines.voxtral import VoxtralASR, VoxtralMiniTranscribe
from spek.factory import get_engine, list_available_models, register_engine

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

