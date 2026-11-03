"""Live transcription: `speck mic`, `speck listen` (desktop audio) and `speck mixed` (mic + desktop)."""

from __future__ import annotations

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

from rich.table import Table

from speck.cli.options import Options, load_engine, speaker_factory
from speck.cli.ui import console, print_token, print_transcript, save_transcript, wait_for_interrupt


def _require_system_audio():
    from speck.capture import system_audio as sa

    try:
        sa.check_available()
    except sa.SystemAudioError as err:
        raise SystemExit(f"System audio unavailable: {err}")
    return sa


def handle_mic(args: argparse.Namespace) -> None:
    from speck.capture.microphone import open_microphone

    engine = load_engine(args)
    console.print(f"[bold]Language:[/bold] [yellow]{args.language}[/yellow]")
    new_classifier = speaker_factory(args, engine.sampling_rate) if args.speakers else None
    console.print("[bold green]Listening... press Ctrl+C to stop.[/bold green]\n")

    session = engine.create_stream_session(
        language=args.language,
        on_token=print_token,
        speakers=new_classifier() if new_classifier else None,
    ).start()
    try:
        with open_microphone(session.push_audio, engine.sampling_rate):
            wait_for_interrupt("Stopping microphone capture...")
    finally:
        session.finish()
        session.join(timeout=30)
    print_transcript(session.transcript)


def handle_listen(args: argparse.Namespace) -> None:
    """Transcribe desktop audio automatically: starts when a call/video plays, stops when quiet."""
    sa = _require_system_audio()
    if args.list:
        return print_audio_streams(sa)

    engine = load_engine(args)
    save_dir = Path(args.save_dir).expanduser() if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    new_classifier = speaker_factory(args, engine.sampling_rate) if args.speakers else None

    def new_session():
        return engine.create_stream_session(
            language=args.language,
            on_token=print_token,
            speakers=new_classifier() if new_classifier else None,  # speakers are per recording
        )

    def on_start(rec: sa.Recording) -> None:
        kind = "call" if rec.in_call else "audio"
        mic = f" + mic [dim]({rec.capture.mic_device})[/dim]" if rec.capture.mic_device else ""
        console.print(f"\n[bold green]● Recording {kind}:[/bold green] [cyan]{rec.label}[/cyan]{mic}")

    def on_end(rec: sa.Recording) -> None:
        duration = time.time() - rec.started_at
        console.print(f"\n[bold yellow]■ Stopped[/bold yellow] [cyan]{rec.label}[/cyan] [dim]after {duration:.0f}s[/dim]")
        if save_dir and rec.session.transcript:
            stamp = datetime.fromtimestamp(rec.started_at).strftime("%Y%m%d-%H%M%S")
            slug = "".join(c if c.isalnum() else "-" for c in rec.label.lower()).strip("-")[:40] or "audio"
            save_transcript(rec.session.transcript, save_dir / f"{stamp}-{slug}.txt")

    listener = sa.SystemAudioListener(
        new_session,
        detector=sa.ActivityDetector(apps=args.app or ()),
        sample_rate=engine.sampling_rate,
        mic=args.mic,
        idle_timeout=args.idle_timeout,
        min_active=args.min_active,
        device=args.source,
        on_start=on_start,
        on_end=on_end,
    )
    target = args.source or (", ".join(args.app) if args.app else "any application")
    console.print(f"[bold]Watching:[/bold] [cyan]{target}[/cyan]  [bold]Mic:[/bold] [yellow]{args.mic}[/yellow]  "
                  f"[bold]Idle timeout:[/bold] {args.idle_timeout:.0f}s")
    console.print("[bold green]Waiting for audio... press Ctrl+C to stop.[/bold green]")

    stop = threading.Event()
    worker = threading.Thread(target=listener.run, args=(stop,), daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            worker.join(0.2)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping system audio capture...[/yellow]")
        stop.set()
        worker.join(timeout=60)


def handle_mixed(args: argparse.Namespace) -> None:
    """Microphone + system audio as two channels in one transcript: you vs. remote speakers."""
    from speck.streaming.mixed import EchoGate, MixedSession

    sa = _require_system_audio()
    engine = load_engine(args)
    new_classifier = speaker_factory(args, engine.sampling_rate) if args.speakers else None
    session = MixedSession(
        engine,
        me_label=args.me,
        remote_label=args.remote,
        tag_speakers=args.tag_speakers is not False,
        tag_format=args.tag_format or "block",
        echo_gate=EchoGate(engine.sampling_rate) if args.echo_guard else None,
        on_token=print_token,
        speakers=new_classifier() if new_classifier else None,
        language=args.language,
    ).start()
    capture = sa.DualCapture(session.push_mic, session.push_system, mic_device=args.mic_source,
                             system_device=args.system_source, sample_rate=engine.sampling_rate)

    remote = "[Speaker 1, Speaker 2...]" if args.speakers else f"[{args.remote}]"
    console.print(f"[bold]Mic:[/bold] [cyan]{args.mic_source}[/cyan] → [{args.me}]  "
                  f"[bold]System:[/bold] [cyan]{args.system_source}[/cyan] → {remote}  "
                  f"[bold]Echo guard:[/bold] {'on' if args.echo_guard else 'off'}")
    console.print("[bold green]Listening... press Ctrl+C to stop.[/bold green]\n")
    capture.start()
    try:
        wait_for_interrupt("Stopping capture...")
    finally:
        capture.stop()
        session.finish()
        session.join(timeout=60)

    console.print()
    if session.echo_gate and session.echo_gate.muted_seconds:
        console.print(f"[dim]Echo guard muted {session.echo_gate.muted_seconds:.0f}s of mic audio.[/dim]")
    print_transcript(session.transcript)
    if args.save:
        save_transcript(session.transcript, Path(args.save).expanduser())


def print_audio_streams(sa) -> None:
    detector = sa.ActivityDetector()
    playback, recording = sa.playback_streams(), sa.recording_streams()
    activity = detector.evaluate(playback, recording)

    table = Table(title="Playback Streams", border_style="cyan")
    for col in ("Index", "Application", "Sink", "State"):
        table.add_column(col)
    for s in playback:
        state = "[dim]ignored[/dim]" if s.ignored else "[dim]paused[/dim]" if s.corked else "[green]playing[/green]"
        if any(c.same_app(s) for c in activity.call_streams):
            state += " [magenta](call)[/magenta]"
        table.add_row(str(s.index), s.label, str(s.device), state)
    console.print(table)

    table = Table(title="Capture Sources (use with --source)", border_style="cyan")
    table.add_column("Name", overflow="fold")
    table.add_column("Kind")
    table.add_column("Description")
    for src in sa.list_sources():
        kind = "monitor" if src["name"].endswith(".monitor") or src.get("monitor_of_sink") else "microphone"
        table.add_row(src["name"], kind, src.get("description", ""))
    console.print(table)


def register(sub: argparse._SubParsersAction, opts: Options) -> None:
    live = [opts.engine, opts.asr]
    speakers = [opts.enroll, opts.live_speakers]

    p = sub.add_parser("mic", help="Live microphone transcription", parents=[*live, opts.diarize, *speakers])
    p.set_defaults(func=handle_mic)

    p = sub.add_parser("listen", help="Auto-transcribe system audio (calls, videos) on Linux",
                       parents=[*live, opts.diarize, *speakers])
    p.add_argument("--app", action="append", metavar="NAME",
                   help="Only react to these apps (substring of name/binary, e.g. discord, firefox); repeatable")
    p.add_argument("--mic", choices=["auto", "on", "off"], default="auto",
                   help="Mix in the microphone: 'auto' only during calls (default)")
    p.add_argument("--idle-timeout", type=float, default=8.0, help="Seconds of quiet that end a recording")
    p.add_argument("--min-active", type=float, default=1.0, help="Seconds of playback before recording starts")
    p.add_argument("--source", default=None, help="Record this capture source continuously instead of auto-detecting")
    p.add_argument("--save-dir", default=None, help="Write each recording's transcript to this directory")
    p.add_argument("--list", action="store_true", help="Show playback streams and capture sources, then exit")
    p.set_defaults(func=handle_listen)

    p = sub.add_parser("mixed", help="Mic + system audio as separate channels: you vs. remote speakers (Linux)",
                       parents=[*live, opts.channel_tags, *speakers])
    p.add_argument("--speakers", action=argparse.BooleanOptionalAction, default=True,
                   help="Classify each remote speaker as Speaker 1, Speaker 2, etc. (default: on; use --no-speakers for generic [Remote])")
    p.add_argument("--me", default="You", help="Label for the microphone channel (default: %(default)s)")
    p.add_argument("--remote", default="Remote",
                   help="Label for the system audio channel when --no-speakers is used (default: %(default)s)")
    p.add_argument("--mic-source", default="@DEFAULT_SOURCE@", help="Microphone source (see `speck listen --list`)")
    p.add_argument("--system-source", default="@DEFAULT_MONITOR@", help="System audio source (a sink monitor)")
    p.add_argument("--echo-guard", action=argparse.BooleanOptionalAction, default=True,
                   help="Mute mic audio that is just the speakers' echo (default: on; off with headphones)")
    p.add_argument("--save", default=None, metavar="PATH", help="Write the final transcript to this file")
    p.set_defaults(func=handle_mixed)
