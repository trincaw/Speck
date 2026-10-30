"""Speech recognition engine implementations."""

from speck.engines.nemotron import NemotronASR
from speck.engines.parakeet import ParakeetASR
from speck.engines.voxtral import VoxtralASR, VoxtralMiniTranscribe
from speck.engines.whisper import WhisperASR

__all__ = [
    "NemotronASR",
    "ParakeetASR",
    "WhisperASR",
    "VoxtralASR",
    "VoxtralMiniTranscribe",
]

