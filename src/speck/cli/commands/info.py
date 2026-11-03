"""`speck models` and `speck languages`: what is installed and configured."""

from __future__ import annotations

import argparse

from rich.table import Table

from speck.cli.options import Options
from speck.cli.ui import console, print_banner
from speck.config import get_mistral_api_key, get_model_name


def handle_models(args: argparse.Namespace) -> None:
    from speck.asr import list_available_models

    print_banner()
    table = Table(title="Available Speech Recognition Engines", border_style="cyan")
    for col, style in [("Status", None), ("Engine ID", "bold"), ("Display Name", "green"),
                       ("Model Checkpoint / Endpoint", "white"), ("Type", "yellow"), ("Description", "dim")]:
        table.add_column(col, style=style)
    for m in list_available_models():
        status = "[bold green]● ACTIVE[/bold green]" if m["is_active"] else "[dim]○[/dim]"
        table.add_row(status, m["id"], m["name"], m["model_id"], m["type"], m["description"])
    console.print(table)

    has_key = bool(get_mistral_api_key())
    console.print(f"\nSPECK_MODEL: [yellow]{get_model_name()}[/yellow]")
    console.print(f"MISTRAL_API_KEY: {'[green]set' if has_key else '[red]not set (needed for Voxtral API)'}[/]")
    console.print("[dim]Switch models via .env or --model <id>.[/dim]\n")


def handle_languages(args: argparse.Namespace) -> None:
    from speck.asr import get_engine

    engine = get_engine(model=args.model, device=args.device)
    print_banner(engine)
    if not engine.supported_languages:
        console.print("[yellow]Languages auto-detected or standard ISO codes supported.[/yellow]")
        return
    for tier in engine.supported_languages.values():
        table = Table(title=tier.name, title_style="bold yellow", border_style="dim")
        table.add_column("Locales / Codes", style="green")
        table.add_column("Description")
        table.add_row(", ".join(tier.locales), tier.description)
        console.print(table)


def register(sub: argparse._SubParsersAction, opts: Options) -> None:
    sub.add_parser("models", help="List engines and .env status").set_defaults(func=handle_models)
    sub.add_parser("languages", help="List supported languages", parents=[opts.engine]).set_defaults(
        func=handle_languages)
