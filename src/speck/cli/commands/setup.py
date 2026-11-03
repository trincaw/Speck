"""`speck setup`: one-shot onboarding, fetches the default models and checks they work."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from speck.cli.options import Options
from speck.cli.ui import console
from speck.config import load_env_file

SAMPLE = Path("samples/bcn_weather.mp3")


def handle_setup(args: argparse.Namespace) -> None:
    from speck.asr import ENGINE_SPECS, get_engine
    from speck.hub import default_assets

    env, example = Path(".env"), Path(".env.example")
    if not env.exists() and example.exists():
        shutil.copyfile(example, env)
        load_env_file(env)
        console.print(f"[green]✓[/green] Created {env} from {example}")

    with console.status("[bold green]Fetching speaker, segmentation and VAD models..."):
        assets = default_assets()
        for asset in assets:
            asset.path()  # cached after the first download
    console.print(f"[green]✓[/green] {', '.join(asset.name for asset in assets)}")

    models = [spec.id for spec in ENGINE_SPECS] if args.all else [args.model]  # None -> SPECK_MODEL
    engines = []
    for model in models:
        engine = get_engine(model=model, device=args.device)
        with console.status(f"[bold green]Fetching and loading {engine.name}..."):
            engine.load()  # downloads exactly what the engine needs into the HF cache
        console.print(f"[green]✓[/green] {engine.name} ({engine.model_id})")
        engines.append(engine)

    if args.no_verify or not SAMPLE.is_file():
        console.print("[dim]Skipped the transcription check.[/dim]")
    else:
        failed = []
        for engine in engines:
            try:
                result = engine.transcribe(str(SAMPLE))
            except Exception as err:  # e.g. missing CUDA libraries: report it, check the other engines
                console.print(f"[red]✗[/red] {engine.name}: {err}")
                failed.append(engine.name)
                continue
            console.print(f"[green]✓[/green] {engine.name}: [italic]{result.text[:80]}[/italic] "
                          f"(RTF {result.real_time_factor:.3f})")
        if failed:
            raise SystemExit(f"Check failed for {', '.join(failed)}; retry with --device cpu to rule out the GPU setup.")
    console.print("\n[bold green]Ready.[/bold green] Try: [cyan]speck mic[/cyan] or [cyan]speck transcribe AUDIO[/cyan]")


def register(sub: argparse._SubParsersAction, opts: Options) -> None:
    p = sub.add_parser("setup", help="Download the default models and verify the install", parents=[opts.engine])
    p.add_argument("--all", action="store_true", help="Fetch every built-in engine, not only SPECK_MODEL")
    p.add_argument("--no-verify", action="store_true", help=f"Skip transcribing {SAMPLE}")
    p.set_defaults(func=handle_setup)
