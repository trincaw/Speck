"""Spek - Model-Agnostic ASR and Streaming Engine."""

from spek.asr import (
    BaseASREngine,
    BenchmarkResult,
    NemotronASR,
    TranscriptionResult,
    VoxtralASR,
    VoxtralMiniTranscribe,
    get_engine,
)
from spek.config import load_env_file
from spek.speakers import LiveSpeakerPreview, SpeakerClassifier, SpeakerEmbedder, SpeakerTurn, VadSegmenter
from spek.streamer import LiveAudioStreamSession
from spek.utils import LOOKAHEAD_CONFIGS, SUPPORTED_LANGUAGES

__all__ = [
    "BaseASREngine",
    "NemotronASR",
    "VoxtralASR",
    "VoxtralMiniTranscribe",
    "get_engine",
    "TranscriptionResult",
    "BenchmarkResult",
    "LiveAudioStreamSession",
    "LiveSpeakerPreview",
    "SpeakerClassifier",
    "SpeakerEmbedder",
    "SpeakerTurn",
    "VadSegmenter",
    "LOOKAHEAD_CONFIGS",
    "SUPPORTED_LANGUAGES",
    "load_env_file",
]

__version__ = "0.2.0"
