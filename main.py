import argparse
import glob
import os

import numpy as np
import soundfile as sf

from speck.pyannote import PyannoteSpeakerRecognizer, RealTimeStreamingEngine


def build_parser():
    parser = argparse.ArgumentParser(description="Near Real-Time Speaker Recognition Tool (spek)")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    p = sub.add_parser("enroll", help="Enroll a speaker voice profile")
    p.add_argument("speaker_name", help="Name or ID of the speaker")
    p.add_argument("audio_path", help="Path to WAV audio file (16kHz recommended)")

    p = sub.add_parser("identify", help="Identify the speaker in an audio file")
    p.add_argument("audio_path", help="Path to query audio file")

    p = sub.add_parser("diarize", help="Diarize speakers in an audio file")
    p.add_argument("audio_path", help="Path to audio file")
    p.add_argument("--num-speakers", type=int, default=-1, help="Expected number of speakers (-1 for auto)")
    p.add_argument("--threshold", type=float, default=0.5, help="Clustering threshold (default: 0.5)")

    p = sub.add_parser("stream", help="Simulate real-time streaming recognition on an audio file")
    p.add_argument("audio_path", help="Path to query audio stream")
    p.add_argument("--chunk-ms", type=int, default=500, help="Chunk size in milliseconds (default: 500ms)")
    p.add_argument("--window-sec", type=float, default=1.5, help="Sliding context window in seconds (default: 1.5s)")
    p.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold (default: 0.75)")

    p = sub.add_parser("evaluate", help="Run full metrics pipeline and export report")
    p.add_argument("--task", choices=["speaker", "tts", "all"], default="all", help="Evaluation benchmark suite (speaker, tts, all; default: all)")
    p.add_argument("--data-dir", default="data", help="Path to speaker dataset directory (default: data)")
    p.add_argument("--output-dir", default="reports", help="Path to output reports directory (default: reports)")
    return parser


def banner(title):
    print("\n" + "=" * 68 + f"\n{title:^68}\n" + "=" * 68)


def run_tts_evaluation(output_dir):
    from speck.tts_metrics import TTSMetricsPipeline

    banner("STARTING TTS & LINGUISTIC METRICS PIPELINE")
    pipeline = TTSMetricsPipeline(output_dir=output_dir)

    # Synthetic reference / "synthesized" pairs for English and Italian
    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False, dtype=np.float32)

    def tone(f0, seed):
        ref = (0.5 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * 2 * f0 * t)).astype(np.float32)
        return ref, ref * 1.02 + (0.0005 * np.random.RandomState(seed).randn(len(t))).astype(np.float32)

    items = [  # (language, f0 Hz, noise seed, synthesis time s, text)
        ("en-US", 140, 1, 0.12, "The quick brown fox jumps over the lazy dog."),
        ("it-IT", 190, 2, 0.13, "L'intelligenza artificiale per l'elaborazione del linguaggio ed il parlato."),
    ]
    benchmarks = []
    for language, f0, seed, synthesis_time, text in items:
        ref, syn = tone(f0, seed)
        benchmarks.append({"ref_audio": ref, "syn_audio": syn, "text": text, "asr_text": text,
                           "language": language, "synthesis_time_sec": synthesis_time})
    artifacts = pipeline.export_reports(pipeline.evaluate_dataset(benchmarks))
    print(f"  -> TTS Summary JSON:   {artifacts['summary_path']}")
    print(f"  -> TTS Markdown Report: {artifacts['report_path']}")
    banner("TTS METRICS PIPELINE COMPLETED")


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    recognizer = PyannoteSpeakerRecognizer()
    # Pre-enroll demo speakers from data/ (first utterance of each) if available
    for spk_dir in sorted(glob.glob(os.path.join("data", "*", ""))):
        wavs = sorted(glob.glob(os.path.join(spk_dir, "*.wav")))
        if wavs:
            recognizer.enroll(os.path.basename(os.path.dirname(spk_dir)), sf.read(wavs[0])[0])

    if args.command == "evaluate":
        if args.task in ("speaker", "all"):
            from speck.metrics_pipeline import SpeakerMetricsPipeline
            SpeakerMetricsPipeline(recognizer, data_dir=args.data_dir, output_dir=args.output_dir).execute_pipeline()
        if args.task in ("tts", "all"):
            run_tts_evaluation(args.output_dir)
        return

    audio, sr = sf.read(args.audio_path)

    if args.command == "enroll":
        recognizer.enroll(args.speaker_name, audio)
        print(f"[OK] Enrolled speaker '{args.speaker_name}' successfully from {args.audio_path}.")

    elif args.command == "identify":
        scores, elapsed = recognizer.match(audio)
        print(f"\nIdentification Results (Latency: {elapsed*1000:.2f} ms):")
        print("-" * 45)
        for spk, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  Speaker: {spk:<15} | Cosine Similarity: {score:.4f}")
        print("-" * 45)
        if scores:
            best_spk = max(scores, key=scores.get)
            print(f"Predicted Speaker: {best_spk} (Score: {scores[best_spk]:.4f})\n")
        else:
            print("Predicted Speaker: None (no speakers enrolled)\n")

    elif args.command == "diarize":
        turns = recognizer.diarize(audio, num_speakers=args.num_speakers, threshold=args.threshold)
        print(f"\nDiarization Results for {args.audio_path}:")
        print("-" * 55)
        for t in turns:
            print(f"  [{t.start:6.2f}s - {t.end:6.2f}s] {t.match.label:<15} (confidence: {t.match.score:.3f})")
        print("-" * 55)
        print(f"Total turns detected: {len(turns)}\n")

    elif args.command == "stream":
        chunk_sec = args.chunk_ms / 1000.0
        engine = RealTimeStreamingEngine(recognizer, sample_rate=sr, chunk_sec=chunk_sec,
                                         window_sec=args.window_sec, threshold=args.threshold)
        print(f"\n--- Starting Streaming Simulation: {args.audio_path} ---")
        print(f"Chunk Duration: {args.chunk_ms} ms | Sliding Window: {args.window_sec} s | Threshold: {args.threshold}\n")
        print(f"{'Time':<12} | {'Active Speaker':<16} | {'Confidence':<12} | {'Latency':<10} | {'RTF'}")
        print("-" * 65)
        for i in range(0, len(audio) - engine.chunk_size, engine.chunk_size):
            t_sec = i / sr
            spk, conf, lat, rtf = engine.process_chunk(audio[i:i + engine.chunk_size])
            print(f"{t_sec:>6.2f}s - {t_sec+chunk_sec:>5.2f}s | {spk:<16} | {conf:>10.4f} | {lat*1000:>6.2f} ms | {rtf:>6.4f}")
        print("-" * 65)
        print("Streaming simulation complete.\n")


if __name__ == "__main__":
    main()
