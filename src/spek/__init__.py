"""Spek - Model-Agnostic ASR and Streaming Engine."""

from importlib.metadata import PackageNotFoundError, version as _version

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
from spek.speakers import LiveSpeakerPreview, SpeakerClassifier, SpeakerEmbedder, SpeakerTurn
from spek.vad import VadSegmenter
from spek.streamer import LiveAudioStreamSession, SpeakerStreamSession
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
    "SpeakerStreamSession",
    "SpeakerTurn",
    "VadSegmenter",
    "LOOKAHEAD_CONFIGS",
    "SUPPORTED_LANGUAGES",
    "load_env_file",
]

try:
    __version__ = _version("spek")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0+unknown"
