"""Speaker recognition: embeddings, live online classification and offline diarization."""

from speck.speakers.classifier import LiveSpeakerPreview, SpeakerClassifier, SpeakerMatch, SpeakerTurn
from speck.speakers.embedding import SpeakerEmbedder
from speck.speakers.recognizer import PyannoteSpeakerRecognizer, RealTimeStreamingEngine

__all__ = [
    "SpeakerEmbedder",
    "SpeakerClassifier",
    "SpeakerMatch",
    "SpeakerTurn",
    "LiveSpeakerPreview",
    "PyannoteSpeakerRecognizer",
    "RealTimeStreamingEngine",
]
