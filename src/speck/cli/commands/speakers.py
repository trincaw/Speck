"""Speaker recognition without transcription: `speck speakers` (live), `speck identify`, `speck diarize`."""

from __future__ import annotations

import argparse

from rich.table import Table

from speck.cli.options import SAMPLE_RATE, Options, load_recognizer, load_wav, speaker_factory
from speck.cli.ui import console
from speck.config import get_speaker_threshold


def handle_speakers(args: argparse.Namespace) -> None:
    """Live speaker classification only (no ASR): who is speaking, turn by turn."""
    from speck.speakers import LiveSpeakerPreview
    from speck.streaming.vad import VadSegmenter

    sr = SAMPLE_RATE
    vad = VadSegmenter(sr, min_silence=args.min_silence)
    classifier = speaker_factory(args, sr)()
    preview = LiveSpeakerPreview(classifier, vad)

    def report(utts, status=None) -> None:
        for utt in utts:
            for turn in classifier.segment(utt.samples):
                match = turn.match
                tag = " [green](new)[/green]" if match.is_new else "" if match.confident else " [yellow](?)[/yellow]"
                console.print(
                    f"[dim]{utt.start + turn.start:7.2f}s – {utt.start + turn.end:7.2f}s[/dim]  "
                    f"[bold magenta]{match.label}[/bold magenta]  [dim]sim={match.score:.2f}[/dim]{tag}"
                )
        if status and (live := preview.update()):
            who = "new voice" if live.label == "?" else live.label
            status.update(f"[bold green]Listening...[/bold green] speaking now: [magenta]{who}[/magenta] "
                          f"[dim]({live.score:.2f})[/dim]")

    if args.audio:
        wav = load_wav(args.audio, sr)
        for i in range(0, len(wav), sr // 10):
            report(vad.accept(wav[i : i + sr // 10]))
        report(vad.flush())
    else:
        import queue

        from speck.capture.microphone import open_microphone

        frames: queue.Queue = queue.Queue()
        with console.status("[bold green]Listening... press Ctrl+C to stop.[/bold green]") as status, \
                open_microphone(frames.put, sr):
            try:
                while True:
                    report(vad.accept(frames.get()), status)
            except KeyboardInterrupt:
                report(vad.flush())

    merged = f"; merged {', '.join(f'{a}→{b}' for a, b in classifier.aliases.items())}" if classifier.aliases else ""
    console.print(f"\n[bold]Speakers seen:[/bold] {len(classifier.speakers)} "
                  f"({', '.join(f'{s.label}×{s.count}' for s in classifier.speakers)}){merged}")


def handle_identify(args: argparse.Namespace) -> None:
    """Closed-set identification of a whole file, or chunk by chunk over a sliding window."""
    recognizer = load_recognizer(args)
    if not recognizer.database:
        raise SystemExit("No speakers enrolled: pass --enroll NAME=WAV or --enroll-dir DIR.")
    audio = load_wav(args.audio)
    if args.stream:
        return _identify_stream(args, recognizer, audio)

    scores, elapsed = recognizer.match(audio)
    table = Table(title=f"Identification ({elapsed * 1000:.1f} ms)", border_style="cyan")
    table.add_column("Speaker", style="bold magenta")
    table.add_column("Cosine Similarity", justify="right")
    best = max(scores, key=scores.get)
    for spk, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        table.add_row(spk, f"[bold green]{score:.4f}[/bold green]" if spk == best else f"{score:.4f}")
    console.print(table)
    verdict = best if scores[best] >= args.speaker_threshold else "Unknown"
    console.print(f"[bold]Predicted speaker:[/bold] [magenta]{verdict}[/magenta] [dim](score {scores[best]:.4f}, "
                  f"threshold {args.speaker_threshold})[/dim]")


def _identify_stream(args: argparse.Namespace, recognizer, audio) -> None:
    from speck.speakers import RealTimeStreamingEngine

    chunk_sec = args.chunk_ms / 1000.0
    engine = RealTimeStreamingEngine(recognizer, sample_rate=SAMPLE_RATE, chunk_sec=chunk_sec,
                                     window_sec=args.window_sec, threshold=args.speaker_threshold)
    console.print(f"[bold]Chunk:[/bold] {args.chunk_ms} ms  [bold]Window:[/bold] {args.window_sec} s  "
                  f"[bold]Threshold:[/bold] {args.speaker_threshold}")
    table = Table(border_style="cyan")
    for col in ("Time", "Active Speaker", "Confidence", "Latency", "RTF"):
        table.add_column(col, justify="left" if col == "Active Speaker" else "right")
    for i in range(0, len(audio) - engine.chunk_size, engine.chunk_size):
        t = i / SAMPLE_RATE
        spk, conf, lat, rtf = engine.process_chunk(audio[i : i + engine.chunk_size])
        table.add_row(f"{t:.2f}s – {t + chunk_sec:.2f}s", spk, f"{conf:.4f}", f"{lat * 1000:.2f} ms", f"{rtf:.4f}")
    console.print(table)


def handle_diarize(args: argparse.Namespace) -> None:
    """Offline diarization: who spoke when; turns are named after enrolled speakers when they match."""
    recognizer = load_recognizer(args)
    with console.status("[bold green]Diarizing..."):
        turns = recognizer.diarize(load_wav(args.audio), num_speakers=args.num_speakers or -1,
                                   threshold=args.cluster_threshold)
    table = Table(title=f"Diarization: {args.audio}", border_style="cyan")
    table.add_column("Time", style="yellow", justify="right")
    table.add_column("Speaker", style="bold magenta")
    table.add_column("Confidence", justify="right")
    for t in turns:
        table.add_row(f"{t.start:.2f}s – {t.end:.2f}s", t.match.label, f"{t.match.score:.3f}")
    console.print(table)
    console.print(f"[bold]Turns:[/bold] {len(turns)}  [bold]Speakers:[/bold] {len({t.match.label for t in turns})}")


def register(sub: argparse._SubParsersAction, opts: Options) -> None:
    from speck.speakers.recognizer import CLUSTER_THRESHOLD

    p = sub.add_parser("speakers", help="Live speaker classification from mic (or a file)",
                       parents=[opts.enroll, opts.live_speakers])
    p.add_argument("audio", nargs="?", help="Classify a file instead of the microphone")
    p.add_argument("--min-silence", type=float, default=0.4, help="Silence (s) that ends an utterance")
    p.set_defaults(func=handle_speakers)

    p = sub.add_parser("identify", help="Identify an enrolled speaker in a file (--stream: chunk by chunk)",
                       parents=[opts.enroll, opts.enroll_dir])
    p.add_argument("audio")
    p.add_argument("--speaker-threshold", type=float, default=get_speaker_threshold(),
                   help="Cosine similarity needed to name a speaker (default: %(default)s)")
    p.add_argument("--stream", action="store_true", help="Simulate real-time recognition over a sliding window")
    p.add_argument("--chunk-ms", type=int, default=500, help="Chunk size with --stream (default: %(default)s ms)")
    p.add_argument("--window-sec", type=float, default=1.5,
                   help="Sliding context window with --stream (default: %(default)s s)")
    p.set_defaults(func=handle_identify)

    p = sub.add_parser("diarize", help="Offline diarization of a file: who spoke when",
                       parents=[opts.enroll, opts.enroll_dir])
    p.add_argument("audio")
    p.add_argument("--num-speakers", type=int, default=None, help="Expected number of speakers (default: auto)")
    p.add_argument("--cluster-threshold", type=float, default=CLUSTER_THRESHOLD,
                   help="Clustering distance; lower finds more speakers (default: %(default)s)")
    p.set_defaults(func=handle_diarize)
