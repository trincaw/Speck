"""File transcription: `speck transcribe`, `speck stream` and `speck benchmark`."""

from __future__ import annotations

import argparse
import time

from rich.panel import Panel
from rich.table import Table

from speck.cli.options import Options, load_engine, preset_label
from speck.cli.ui import console, print_token


def handle_transcribe(args: argparse.Namespace) -> None:
    engine = load_engine(args)
    console.print(f"[bold]Audio:[/bold] [cyan]{args.audio}[/cyan]  [bold]Language:[/bold] [yellow]{args.language}[/yellow]")

    with console.status(f"[bold green]Transcribing with {engine.name}..."):
        result = engine.transcribe(args.audio, language=args.language, skip_special_tokens=not args.keep_tag)

    table = Table(title="Transcription Metrics", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")
    table.add_row("Engine", engine.name)
    table.add_row("Language", str(result.detected_language or "N/A"))
    table.add_row("Audio Duration", f"{result.duration_seconds:.2f}s")
    table.add_row("Inference Time", f"{result.inference_seconds:.2f}s")
    table.add_row("Real-Time Factor", f"{result.real_time_factor:.3f}x")

    speaker_segments = [s for s in result.segments or [] if "speaker" in s]
    if speaker_segments:
        speakers = sorted({s["speaker"] for s in speaker_segments})
        table.add_row("Detected Speakers", f"{len(speakers)} ({', '.join(speakers)})")
        table.add_row("Speech Segments", str(len(speaker_segments)))
    console.print(table)

    if len(speaker_segments) > 1:
        turns = Table(title="Speaker Turns", border_style="dim")
        turns.add_column("Speaker", style="bold magenta", justify="center")
        turns.add_column("Time", style="yellow", justify="center")
        turns.add_column("Speech")
        for s in speaker_segments:
            turns.add_row(s["speaker"], f"{s['start']:.2f}s - {s['end']:.2f}s", s.get("text", ""))
        console.print(turns)

    console.print(Panel(result.text, title="Transcript", border_style="green"))


def handle_stream(args: argparse.Namespace) -> None:
    engine = load_engine(args)
    console.print(f"[bold]Audio:[/bold] [cyan]{args.audio}[/cyan]  [bold]Preset:[/bold] [magenta]{preset_label(engine, args.lookahead)}[/magenta]")
    console.print("[dim]--- Begin Stream ---[/dim]")

    start, updates = time.perf_counter(), 0
    try:
        for chunk in engine.stream_transcribe(
            args.audio,
            language=args.language,
            lookahead=args.lookahead,
            skip_special_tokens=not args.keep_tag,
            simulate_realtime=args.simulate_realtime,
        ):
            print_token(chunk)
            updates += 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    console.print(f"\n[dim]--- End Stream: {updates} updates in {time.perf_counter() - start:.2f}s ---[/dim]")


def handle_benchmark(args: argparse.Namespace) -> None:
    engine = load_engine(args)
    with console.status(f"[bold green]Benchmarking {engine.name}..."):
        results = engine.benchmark(args.audio, language=args.language)

    table = Table(title=f"{engine.name} Streaming Benchmark", border_style="green")
    for col in ("Lookahead", "Algorithmic Latency", "Audio", "Wall Time", "RTF"):
        table.add_column(col, justify="right")
    for r in results:
        table.add_row(
            f"{r.lookahead_tokens} tokens",
            f"{r.configured_latency_ms:.0f} ms",
            f"{r.audio_duration_seconds:.2f} s",
            f"{r.elapsed_seconds:.2f} s",
            f"[bold green]{r.real_time_factor:.3f}x[/bold green]",
        )
    console.print(table)


def register(sub: argparse._SubParsersAction, opts: Options) -> None:
    p = sub.add_parser("transcribe", help="Offline transcription", parents=[opts.engine, opts.asr])
    p.add_argument("audio", help="Audio file (.wav, .mp3, .flac)")
    p.set_defaults(func=handle_transcribe)

    p = sub.add_parser("stream", help="Chunked streaming transcription of a file",
                       parents=[opts.engine, opts.asr, opts.lookahead])
    p.add_argument("audio")
    p.add_argument("--simulate-realtime", action="store_true", help="Pace input like live audio")
    p.set_defaults(func=handle_stream)

    p = sub.add_parser("benchmark", help="Benchmark streaming latency presets", parents=[opts.engine])
    p.add_argument("audio")
    p.add_argument("--language", default="auto")
    p.set_defaults(func=handle_benchmark)
