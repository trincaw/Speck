"""Speech recognition engine implementations."""

from speck.engines.nemotron import NemotronASR
from speck.engines.voxtral import VoxtralASR, VoxtralMiniTranscribe

__all__ = [
    "NemotronASR",
    "VoxtralASR",
    "VoxtralMiniTranscribe",
]

