"""Live audio pipelines: VAD segmentation, push-based ASR sessions and two-channel (mic + system) mixing."""

from speck.streaming.gate import TranscriptionGate
from speck.streaming.mixed import EchoGate, MixedSession, TranscriptMixer
from speck.streaming.sessions import (
    AudioBuffer,
    BaseAudioStreamSession,
    LiveAudioStreamSession,
    SpeakerStreamSession,
    UtteranceStreamSession,
)
from speck.streaming.vad import Utterance, VadSegmenter

__all__ = [
    "AudioBuffer",
    "BaseAudioStreamSession",
    "UtteranceStreamSession",
    "SpeakerStreamSession",
    "LiveAudioStreamSession",
    "MixedSession",
    "TranscriptMixer",
    "EchoGate",
    "TranscriptionGate",
    "VadSegmenter",
    "Utterance",
]
