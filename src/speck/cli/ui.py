"""Terminal output shared by the commands (rich console, banners, transcript panels)."""

from __future__ import annotations

import os
import select
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Optional

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from speck.config import get_model_name

try:
    import termios
    import tty
except ImportError:  # not a POSIX terminal: no single-key input
    termios = tty = None

if TYPE_CHECKING:
    from speck.asr.base import BaseASREngine
    from speck.streaming.gate import TranscriptionGate

console = Console()

TOGGLE_KEY = " "  # pauses / resumes live transcription


def print_banner(engine: Optional[BaseASREngine] = None) -> None:
    if engine:
        title, sub = engine.name, f"Model ID: {engine.model_id} | Device: {engine.device}"
    else:
        title, sub = f"Speck ASR Engine ({get_model_name()})", "Model-Agnostic Real-Time Speech Recognition"
    console.print(Panel.fit(f"[bold green]{title}[/bold green]\n[dim]{sub}[/dim]", border_style="green"))


def print_token(text: str) -> None:
    """Write streamed transcript text as it arrives (plain stdout, no markup)."""
    sys.stdout.write(text)
    sys.stdout.flush()


def print_transcript(text: str, title: str = "Final Transcript") -> None:
    console.print(Panel(escape(text) or "[dim](no speech)[/dim]", title=title, border_style="green"))


def save_transcript(text: str, path: Path) -> None:
    path.write_text(text + "\n", encoding="utf-8")
    console.print(f"[dim]Saved transcript to {path}[/dim]")


def keys_available() -> bool:
    """True when single keypresses can be read from the terminal (stdin is an interactive TTY)."""
    return termios is not None and sys.stdin is not None and sys.stdin.isatty()


def stop_hint(gate: Optional[TranscriptionGate] = None) -> str:
    """"press ... to stop.", mentioning the pause key when `wait_for_interrupt` will listen for it."""
    keys = "Space to pause/resume, Ctrl+C to stop" if gate and keys_available() else "Ctrl+C to stop"
    return f"press {keys}."


@contextmanager
def _keypresses(enabled: bool = True) -> Iterator[Callable[[float], str]]:
    """Yield `read(timeout)`: the characters typed within `timeout` seconds (possibly none).

    The terminal is in cbreak mode meanwhile (no echo, no Enter needed, Ctrl+C still raises
    KeyboardInterrupt) and is always restored. Without a terminal `read` just sleeps.
    """
    if not (enabled and keys_available()):
        yield lambda timeout: time.sleep(timeout) or ""
        return

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)

    def read(timeout: float) -> str:
        ready, _, _ = select.select([fd], [], [], timeout)
        return os.read(fd, 1024).decode(errors="ignore") if ready else ""

    tty.setcbreak(fd)
    try:
        yield read
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def wait_for_interrupt(
    message: str,
    gate: Optional[TranscriptionGate] = None,
    until: Optional[Callable[[], bool]] = None,
) -> None:
    """Block until Ctrl+C (then print `message`) or until `until()` is true.

    With a `gate`, pressing Space meanwhile pauses or resumes transcription.
    """
    try:
        with _keypresses(enabled=gate is not None) as read:
            while not (until and until()):
                if gate and TOGGLE_KEY in read(0.1):
                    gate.toggle()
    except KeyboardInterrupt:
        console.print(f"\n[yellow]{message}[/yellow]")
