"""Speech recognition engine implementations."""

from spek.engines.nemotron import NemotronASR
from spek.engines.sherpa import (
    SherpaASR,
    SherpaOnnxASR,
    SherpaSpeakerTagger,
    format_speaker_tagged_transcript,
    merge_speaker_segments,
)
from spek.engines.voxtral import VoxtralASR, VoxtralMiniTranscribe

__all__ = [
    "NemotronASR",
    "VoxtralASR",
    "VoxtralMiniTranscribe",
    "SherpaOnnxASR",
    "SherpaASR",
    "SherpaSpeakerTagger",
    "format_speaker_tagged_transcript",
    "merge_speaker_segments",
]

