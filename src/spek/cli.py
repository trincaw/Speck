"""Command-line interface for Spek."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spek.base import BaseASREngine
from spek.config import get_mistral_api_key, get_model_name, get_speaker_threshold
from spek.factory import get_engine, list_available_models

console = Console()


def print_banner(engine: Optional[BaseASREngine] = None) -> None:
    if engine:
        title, sub = engine.name, f"Model ID: {engine.model_id} | Device: {engine.device}"
    else:
        title, sub = f"Spek ASR Engine ({get_model_name()})", "Model-Agnostic Real-Time Speech Recognition"
    console.print(Panel.fit(f"[bold green]{title}[/bold green]\n[dim]{sub}[/dim]", border_style="green"))


def load_engine(args: argparse.Namespace, **kwargs: Any) -> BaseASREngine:
    engine = get_engine(model=args.model, device=args.device, **kwargs)
    with console.status(f"[bold green]Loading {engine.name}..."):
        engine.load()
    print_banner(engine)
    return engine


def preset_label(engine: BaseASREngine, lookahead: int) -> str:
    cfg = engine.streaming_presets.get(lookahead, {"latency_ms": 0.0, "description": "Custom"})
    return f"{cfg.get('description', '')} ({cfg.get('latency_ms', 0)}ms)"


def speaker_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Speaker-tagging options set on the command line (only those given)."""
    kwargs = {"tag_speakers": args.tag_speakers, "tag_format": args.tag_format, "num_clusters": args.num_speakers}
    return {k: v for k, v in kwargs.items() if v is not None}


def parse_enroll(values: list[str]) -> dict[str, str]:
    enroll = {}
    for value in values or []:
        name, sep, path = value.partition("=")
        if not sep or not os.path.isfile(path):
            raise SystemExit(f"--enroll expects NAME=path/to/voice.wav, got '{value}'")
        enroll[name] = path
    return enroll


def stream_microphone(session, sample_rate: int) -> None:
    """Feed the default microphone into a session until Ctrl+C."""
    import sounddevice as sd

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=sample_rate // 10,
        callback=lambda indata, *_: session.push_audio(indata[:, 0].copy()),
    ):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping microphone capture...[/yellow]")


# --- commands -----------------------------------------------------------------


def handle_models(args):
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
    console.print(f"\nSPEK_MODEL: [yellow]{get_model_name()}[/yellow]")
    console.print(f"MISTRAL_API_KEY: {'[green]set' if has_key else '[red]not set (needed for Voxtral API)'}[/]")
    console.print("[dim]Switch models via .env or --model <id>.[/dim]\n")


def handle_languages(args):
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


def handle_transcribe(args):
    engine = load_engine(args)
    console.print(f"[bold]Audio:[/bold] [cyan]{args.audio}[/cyan]  [bold]Language:[/bold] [yellow]{args.language}[/yellow]")

    with console.status(f"[bold green]Transcribing with {engine.name}..."):
        result = engine.transcribe(
            args.audio, language=args.language, skip_special_tokens=not args.keep_tag, **speaker_kwargs(args)
        )

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


def handle_stream(args):
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
            sys.stdout.write(chunk)
            sys.stdout.flush()
            updates += 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    console.print(f"\n[dim]--- End Stream: {updates} updates in {time.perf_counter() - start:.2f}s ---[/dim]")


def handle_benchmark(args):
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


def live_session_kwargs(engine: BaseASREngine, args: argparse.Namespace) -> dict[str, Any]:
    """Live speaker-classification options for the session (sherpa-onnx only)."""
    from spek.engines.sherpa import SherpaOnnxASR

    if isinstance(engine, SherpaOnnxASR):
        return {
            "tag_speakers": args.tag_speakers,
            "tag_format": args.tag_format,
            "speaker_threshold": args.speaker_threshold,
            "max_speakers": args.num_speakers,
            "enroll": parse_enroll(args.enroll),
        }
    if args.enroll or args.num_speakers:
        console.print("[yellow]Speaker options are only supported with --model sherpa-onnx; ignoring.[/yellow]")
    return {}


def handle_mic(args):
    """Live microphone transcription; with sherpa-onnx, each utterance is speaker-tagged."""
    engine = load_engine(args)
    session_kwargs = live_session_kwargs(engine, args)

    console.print(f"[bold]Language:[/bold] [yellow]{args.language}[/yellow]  [bold]Preset:[/bold] [magenta]{preset_label(engine, args.lookahead)}[/magenta]")
    console.print("[bold green]Listening... press Ctrl+C to stop.[/bold green]\n")

    session = engine.create_stream_session(
        language=args.language,
        lookahead=args.lookahead,
        skip_special_tokens=not args.keep_tag,
        on_token=lambda t: (sys.stdout.write(t), sys.stdout.flush()),
        **session_kwargs,
    ).start()
    try:
        stream_microphone(session, engine.sampling_rate)
    finally:
        session.finish()
        session.join(timeout=30)
    console.print(Panel(session.transcript or "[dim](no speech)[/dim]", title="Final Transcript", border_style="green"))


def handle_listen(args):
    """Transcribe desktop audio automatically: starts when a call/video plays, stops when quiet."""
    import threading
    from datetime import datetime
    from pathlib import Path

    from spek import system_audio as sa

    try:
        sa.check_available()
        if args.list:
            return print_audio_streams(sa)
    except sa.SystemAudioError as err:
        raise SystemExit(f"System audio unavailable: {err}")

    engine = load_engine(args)
    session_kwargs = live_session_kwargs(engine, args)
    save_dir = Path(args.save_dir).expanduser() if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    def new_session():
        return engine.create_stream_session(
            language=args.language,
            lookahead=args.lookahead,
            skip_special_tokens=not args.keep_tag,
            on_token=lambda t: (sys.stdout.write(t), sys.stdout.flush()),
            **session_kwargs,
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
            path = save_dir / f"{stamp}-{slug}.txt"
            path.write_text(rec.session.transcript + "\n", encoding="utf-8")
            console.print(f"[dim]Saved transcript to {path}[/dim]")

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


def handle_speakers(args):
    """Live speaker classification only (no ASR): who is speaking, utterance by utterance."""
    from transformers.audio_utils import load_audio

    from spek.speakers import SpeakerClassifier, SpeakerEmbedder, VadSegmenter

    sr = 16000
    with console.status("[bold green]Loading VAD and speaker-embedding models..."):
        vad = VadSegmenter(sr, min_silence=args.min_silence)
        classifier = SpeakerClassifier(
            SpeakerEmbedder(), threshold=args.speaker_threshold, max_speakers=args.num_speakers or 0
        )
        for name, path in parse_enroll(args.enroll).items():
            classifier.enroll(name, load_audio(path, sampling_rate=sr))
    if classifier.speakers:
        console.print(f"[bold]Enrolled:[/bold] {', '.join(s.label for s in classifier.speakers)}")

    def report(utts) -> None:
        for utt in utts:
            match = classifier.classify(utt.samples)
            end = utt.start + len(utt.samples) / sr
            tag = " [green](new)[/green]" if match.is_new else ""
            console.print(
                f"[dim]{utt.start:7.2f}s – {end:7.2f}s[/dim]  [bold magenta]{match.label}[/bold magenta]"
                f"  [dim]sim={match.score:.2f}[/dim]{tag}"
            )

    if args.audio:
        wav = load_audio(args.audio, sampling_rate=sr).astype("float32")
        for i in range(0, len(wav), sr // 10):
            report(vad.accept(wav[i : i + sr // 10]))
        report(vad.flush())
    else:
        import queue
        import sounddevice as sd

        frames: queue.Queue = queue.Queue()
        console.print("[bold green]Listening... press Ctrl+C to stop.[/bold green]\n")
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=sr // 10,
                            callback=lambda indata, *_: frames.put(indata[:, 0].copy())):
            try:
                while True:
                    report(vad.accept(frames.get()))
            except KeyboardInterrupt:
                report(vad.flush())

    console.print(f"\n[bold]Speakers seen:[/bold] {len(classifier.speakers)} "
                  f"({', '.join(f'{s.label}×{s.count}' for s in classifier.speakers)})")


def handle_serve(args):
    if args.model:
        os.environ["SPEK_MODEL"] = args.model
    import uvicorn

    from spek.server import get_engine as server_engine

    engine = server_engine()
    print_banner(engine)
    console.print(f"[bold green]Serving at[/bold green] [cyan]http://{args.host}:{args.port}[/cyan]\n")
    uvicorn.run("spek.server:app", host=args.host, port=args.port, log_level="info")


# --- argument parsing -----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    engine_opts = argparse.ArgumentParser(add_help=False)
    engine_opts.add_argument("--model", default=None, help="Engine alias or checkpoint (overrides SPEK_MODEL)")
    engine_opts.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: auto)")

    asr_opts = argparse.ArgumentParser(add_help=False)
    asr_opts.add_argument("--language", default="auto", help="Locale (e.g. en-US) or 'auto'")
    asr_opts.add_argument("--keep-tag", action="store_true", help="Keep the <xx-XX> language tag in output")

    lookahead_opts = argparse.ArgumentParser(add_help=False)
    lookahead_opts.add_argument("--lookahead", type=int, default=6, help="Latency preset: 0, 3, 6, 13")

    speaker_opts = argparse.ArgumentParser(add_help=False)
    speaker_opts.add_argument("--tag-speakers", action=argparse.BooleanOptionalAction, default=None,
                              help="Prefix speaker tags in output (sherpa-onnx)")
    speaker_opts.add_argument("--tag-format", choices=["block", "inline"], default=None)
    speaker_opts.add_argument("--num-speakers", type=int, default=None, help="Fixed/max speaker count (omit to auto-detect)")

    live_speaker_opts = argparse.ArgumentParser(add_help=False)
    live_speaker_opts.add_argument("--enroll", action="append", metavar="NAME=WAV",
                                   help="Known person's voice sample; repeatable")
    live_speaker_opts.add_argument("--speaker-threshold", type=float, default=get_speaker_threshold(),
                                   help="Cosine similarity to match a known speaker (default: %(default)s)")

    parser = argparse.ArgumentParser(prog="spek", description="Spek - model-agnostic ASR and streaming engine")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help: str, parents: list) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help, parents=parents)
        p.set_defaults(func=func)
        return p

    add("models", handle_models, "List engines and .env status", [])
    add("languages", handle_languages, "List supported languages", [engine_opts])

    p = add("transcribe", handle_transcribe, "Offline transcription", [engine_opts, asr_opts, speaker_opts])
    p.add_argument("audio", help="Audio file (.wav, .mp3, .flac)")

    p = add("stream", handle_stream, "Chunked streaming transcription of a file", [engine_opts, asr_opts, lookahead_opts])
    p.add_argument("audio")
    p.add_argument("--simulate-realtime", action="store_true", help="Pace input like live audio")

    add("mic", handle_mic, "Live microphone transcription (speaker-tagged with sherpa-onnx)",
        [engine_opts, asr_opts, lookahead_opts, speaker_opts, live_speaker_opts])

    p = add("listen", handle_listen, "Auto-transcribe system audio (calls, videos) on Linux",
            [engine_opts, asr_opts, lookahead_opts, speaker_opts, live_speaker_opts])
    p.add_argument("--app", action="append", metavar="NAME",
                   help="Only react to these apps (substring of name/binary, e.g. discord, firefox); repeatable")
    p.add_argument("--mic", choices=["auto", "on", "off"], default="auto",
                   help="Mix in the microphone: 'auto' only during calls (default)")
    p.add_argument("--idle-timeout", type=float, default=8.0, help="Seconds of quiet that end a recording")
    p.add_argument("--min-active", type=float, default=1.0, help="Seconds of playback before recording starts")
    p.add_argument("--source", default=None, help="Record this capture source continuously instead of auto-detecting")
    p.add_argument("--save-dir", default=None, help="Write each recording's transcript to this directory")
    p.add_argument("--list", action="store_true", help="Show playback streams and capture sources, then exit")

    p = add("speakers", handle_speakers, "Live speaker classification from mic (or a file)",
            [live_speaker_opts])
    p.add_argument("audio", nargs="?", help="Classify a file instead of the microphone")
    p.add_argument("--num-speakers", type=int, default=None, help="Max speakers to discover")
    p.add_argument("--min-silence", type=float, default=0.4, help="Silence (s) that ends an utterance")

    p = add("benchmark", handle_benchmark, "Benchmark streaming latency presets", [engine_opts])
    p.add_argument("audio")
    p.add_argument("--language", default="auto")

    p = add("serve", handle_serve, "Run the web UI + REST/WebSocket API", [])
    p.add_argument("--model", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
