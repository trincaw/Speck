"""Speech recognition engine implementations."""

from speck.asr.engines.parakeet import ParakeetASR
from speck.asr.engines.whisper import WhisperASR

__all__ = ["ParakeetASR", "WhisperASR"]
