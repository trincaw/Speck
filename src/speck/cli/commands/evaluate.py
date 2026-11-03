"""`speck evaluate`: speaker-recognition and TTS metric pipelines with exported reports."""

from __future__ import annotations

import argparse

from speck.cli.options import Options
from speck.cli.ui import console


def handle_evaluate(args: argparse.Namespace) -> None:
    if args.task in ("speaker", "all"):
        from speck.evaluation.speaker_pipeline import SpeakerMetricsPipeline

        SpeakerMetricsPipeline(data_dir=args.data_dir, output_dir=args.output_dir).execute_pipeline()
    if args.task in ("tts", "all"):
        from speck.evaluation.tts import TTSMetricsPipeline, synthetic_benchmark

        console.rule("[bold]TTS & linguistic metrics (synthetic smoke test)")
        pipeline = TTSMetricsPipeline(output_dir=args.output_dir)
        artifacts = pipeline.export_reports(pipeline.evaluate_dataset(synthetic_benchmark()))
        console.print(f"  -> TTS Summary JSON:    {artifacts['summary_path']}")
        console.print(f"  -> TTS Markdown Report: {artifacts['report_path']}")


def register(sub: argparse._SubParsersAction, opts: Options) -> None:
    p = sub.add_parser("evaluate", help="Run the metric pipelines and export reports")
    p.add_argument("--task", choices=["speaker", "tts", "all"], default="all",
                   help="Benchmark suite (default: %(default)s)")
    p.add_argument("--data-dir", default="data", help="Speaker dataset: one folder of WAVs per speaker (default: %(default)s)")
    p.add_argument("--output-dir", default="reports", help="Where reports are written (default: %(default)s)")
    p.set_defaults(func=handle_evaluate)
