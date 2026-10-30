"""Speck - Model-Agnostic ASR and Streaming Engine."""

from importlib.metadata import PackageNotFoundError, version as _version

from speck.asr import (
    BaseASREngine,
    BenchmarkResult,
    NemotronASR,
    ParakeetASR,
    TranscriptionResult,
    VoxtralASR,
    VoxtralMiniTranscribe,
    WhisperASR,
    get_engine,
)
from speck.config import load_env_file
from speck.speakers import LiveSpeakerPreview, SpeakerClassifier, SpeakerEmbedder, SpeakerTurn
from speck.pyannote import PyannoteSpeakerRecognizer
from speck.vad import VadSegmenter
from speck.streamer import LiveAudioStreamSession, SpeakerStreamSession
from speck.utils import LOOKAHEAD_CONFIGS, SUPPORTED_LANGUAGES
from speck.tts_metrics import (
    compute_mcd,
    compute_f0_metrics,
    compute_speaker_similarity,
    compute_wer,
    compute_cer,
    compute_mer_and_wil,
    compute_speech_text_metrics,
    normalize_text,
    normalize_english_text,
    normalize_italian_text,
    TTSMetricsPipeline,
)
from speck.postprocess import correct_vocabulary, rover

__all__ = [
    "BaseASREngine",
    "NemotronASR",
    "ParakeetASR",
    "WhisperASR",
    "VoxtralASR",
    "VoxtralMiniTranscribe",
    "get_engine",
    "TranscriptionResult",
    "BenchmarkResult",
    "LiveAudioStreamSession",
    "LiveSpeakerPreview",
    "PyannoteSpeakerRecognizer",
    "SpeakerClassifier",
    "SpeakerEmbedder",
    "SpeakerStreamSession",
    "SpeakerTurn",
    "VadSegmenter",
    "LOOKAHEAD_CONFIGS",
    "SUPPORTED_LANGUAGES",
    "load_env_file",
    "compute_mcd",
    "compute_f0_metrics",
    "compute_speaker_similarity",
    "compute_wer",
    "compute_cer",
    "compute_mer_and_wil",
    "compute_speech_text_metrics",
    "normalize_text",
    "normalize_english_text",
    "normalize_italian_text",
    "TTSMetricsPipeline",
    "correct_vocabulary",
    "rover",
]

try:
    __version__ = _version("speck")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0+unknown"
