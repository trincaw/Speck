"""Terminal output shared by the commands (rich console, banners, transcript panels)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from speck.config import get_model_name

if TYPE_CHECKING:
    from speck.asr.base import BaseASREngine

console = Console()


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


def wait_for_interrupt(message: str) -> None:
    """Block until Ctrl+C, then print `message`."""
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]{message}[/yellow]")
