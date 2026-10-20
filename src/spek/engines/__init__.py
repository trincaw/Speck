"""Speech recognition engine implementations."""

from spek.engines.nemotron import NemotronASR
from spek.engines.voxtral import VoxtralASR, VoxtralMiniTranscribe

__all__ = [
    "NemotronASR",
    "VoxtralASR",
    "VoxtralMiniTranscribe",
]

