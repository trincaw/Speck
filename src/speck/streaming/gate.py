"""Pause / resume switch for live transcription."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TranscriptionGate:
    """Thread-safe on/off switch shared by any number of live sessions.

    While off, a session gets silence of the same length instead of the real audio. The
    audio itself is discarded, but the session's timeline keeps running and its VAD sees
    the pause as a silence, so what was said just before the switch is still finished and
    transcribed, and speech on either side of a pause is never glued into one utterance.

    `on_change(active)` runs after every real change, on the thread that flipped the switch.
    """

    def __init__(self, active: bool = True, on_change: Optional[Callable[[bool], None]] = None):
        self._active = active
        self.on_change = on_change
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._active

    def set(self, active: bool) -> bool:
        """Turn transcription on or off; returns the new state."""
        with self._lock:
            changed = active != self._active
            self._active = active
        if changed:
            self._notify(active)
        return active

    def toggle(self) -> bool:
        """Flip the switch; returns the new state."""
        with self._lock:
            self._active = active = not self._active
        self._notify(active)
        return active

    def apply(self, chunk: np.ndarray) -> np.ndarray:
        """The chunk itself while on, silence of the same shape while off."""
        return chunk if self._active else np.zeros_like(chunk)

    def _notify(self, active: bool) -> None:
        if self.on_change:
            try:
                self.on_change(active)
            except Exception as err:
                logger.error("Error in on_change callback: %s", err)
