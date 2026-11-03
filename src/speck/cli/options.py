"""Option groups shared by several commands, and the objects built from them."""

from __future__ import annotations

import argparse
import glob
import os
from typing import TYPE_CHECKING, Any, Callable

from speck.cli.ui import console, print_banner
from speck.config import get_language, get_speaker_threshold

if TYPE_CHECKING:
    from speck.asr.base import BaseASREngine
    from speck.speakers import PyannoteSpeakerRecognizer, SpeakerClassifier

SAMPLE_RATE = 16000  # speaker models and VAD work at 16 kHz


def _parent() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(add_help=False)


class Options:
    """Reusable argparse parents; built after the .env is loaded so defaults reflect it."""

    def __init__(self) -> None:
        self.engine = _parent()
        self.engine.add_argument("--model", default=None, help="Engine alias or checkpoint (overrides SPECK_MODEL)")
        self.engine.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: auto)")

        self.asr = _parent()
        self.asr.add_argument("--language", default=get_language(),
                              help="Locale (e.g. en-US) or 'auto' (default: SPECK_LANGUAGE)")
        self.asr.add_argument("--keep-tag", action="store_true", help="Keep the <xx-XX> language tag in output")

        self.lookahead = _parent()
        self.lookahead.add_argument("--lookahead", type=int, default=6, help="Latency preset: 0, 3, 6, 13")

        self.channel_tags = _parent()
        self.channel_tags.add_argument("--tag-speakers", action=argparse.BooleanOptionalAction, default=None,
                                       help="Prefix channel tags in output")
        self.channel_tags.add_argument("--tag-format", choices=["block", "inline"], default=None)

        self.enroll = _parent()
        self.enroll.add_argument("--enroll", action="append", metavar="NAME=WAV",
                                 help="Known person's voice sample; repeatable")

        self.live_speakers = _parent()
        self.live_speakers.add_argument("--speaker-threshold", type=float, default=get_speaker_threshold(),
                                        help="Cosine similarity to match a known speaker (default: %(default)s)")
        self.live_speakers.add_argument("--num-speakers", type=int, default=None, help="Max speakers to discover")

        self.diarize = _parent()
        self.diarize.add_argument("--speakers", action="store_true",
                                  help="Label who is speaking: split each utterance into speaker turns, then transcribe")

        self.enroll_dir = _parent()
        self.enroll_dir.add_argument("--enroll-dir", default="data", metavar="DIR",
                                     help="Enroll the first WAV of each DIR/<speaker>/ folder (default: %(default)s; "
                                          "skipped when missing)")


# --- objects built from options ---------------------------------------------------


def load_engine(args: argparse.Namespace, **kwargs: Any) -> BaseASREngine:
    from speck.asr import get_engine

    engine = get_engine(model=args.model, device=args.device, **kwargs)
    with console.status(f"[bold green]Loading {engine.name}..."):
        engine.load()
    print_banner(engine)
    return engine


def preset_label(engine: BaseASREngine, lookahead: int) -> str:
    cfg = engine.streaming_presets.get(lookahead, {"latency_ms": 0.0, "description": "Custom"})
    return f"{cfg.get('description', '')} ({cfg.get('latency_ms', 0)}ms)"


def load_wav(path: str, sample_rate: int = SAMPLE_RATE):
    """Any audio file as mono float32 at `sample_rate`."""
    from transformers.audio_utils import load_audio

    return load_audio(path, sampling_rate=sample_rate).astype("float32")


def parse_enroll(values: list[str] | None) -> dict[str, str]:
    enroll = {}
    for value in values or []:
        name, sep, path = value.partition("=")
        if not sep or not os.path.isfile(path):
            raise SystemExit(f"--enroll expects NAME=path/to/voice.wav, got '{value}'")
        enroll[name] = path
    return enroll


def enrollment_samples(args: argparse.Namespace, sample_rate: int = SAMPLE_RATE) -> dict[str, Any]:
    """Voice samples from `--enroll-dir` (if the command has it) and `--enroll`; explicit ones win."""
    paths: dict[str, str] = {}
    root = getattr(args, "enroll_dir", None)
    if root and os.path.isdir(root):
        for spk_dir in sorted(glob.glob(os.path.join(root, "*", ""))):
            wavs = sorted(glob.glob(os.path.join(spk_dir, "*.wav")))
            if wavs:
                paths[os.path.basename(os.path.dirname(spk_dir))] = wavs[0]
    paths.update(parse_enroll(args.enroll))
    return {name: load_wav(path, sample_rate) for name, path in paths.items()}


def speaker_factory(args: argparse.Namespace, sample_rate: int) -> Callable[[], SpeakerClassifier]:
    """Load the embedding model and enrollment samples once; return a factory of fresh classifiers."""
    from speck.speakers import SpeakerClassifier, SpeakerEmbedder

    with console.status("[bold green]Loading speaker-embedding model..."):
        embedder = SpeakerEmbedder(sample_rate=sample_rate)
        voices = enrollment_samples(args, sample_rate)
    if voices:
        console.print(f"[bold]Enrolled:[/bold] {', '.join(voices)}")

    def new_classifier() -> SpeakerClassifier:
        classifier = SpeakerClassifier(
            embedder,
            threshold=args.speaker_threshold,
            max_speakers=args.num_speakers or 0,
            start_id=1,
        )
        for name, wav in voices.items():
            classifier.enroll(name, wav)
        return classifier

    return new_classifier


def load_recognizer(args: argparse.Namespace) -> PyannoteSpeakerRecognizer:
    """Offline recognizer (segmentation + embeddings) with the requested speakers enrolled."""
    from speck.speakers import PyannoteSpeakerRecognizer

    with console.status("[bold green]Loading speaker models..."):
        recognizer = PyannoteSpeakerRecognizer(sample_rate=SAMPLE_RATE)
        voices = enrollment_samples(args)
        for name, wav in voices.items():
            recognizer.enroll(name, wav)
    if voices:
        console.print(f"[bold]Enrolled:[/bold] {', '.join(voices)}")
    return recognizer
