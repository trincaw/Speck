import sys
import os
import argparse
import numpy as np
import soundfile as sf

from speck.pyannote import PyannoteSpeakerRecognizer
from src.streaming import RealTimeStreamingEngine

def main():
    parser = argparse.ArgumentParser(description="Near Real-Time Speaker Recognition Tool (spek)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Enroll
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a speaker voice profile")
    enroll_parser.add_argument("speaker_name", help="Name or ID of the speaker")
    enroll_parser.add_argument("audio_path", help="Path to WAV audio file (16kHz recommended)")
    
    # Identify
    ident_parser = subparsers.add_parser("identify", help="Identify the speaker in an audio file")
    ident_parser.add_argument("audio_path", help="Path to query audio file")
    
    # Diarize
    diar_parser = subparsers.add_parser("diarize", help="Diarize speakers in an audio file")
    diar_parser.add_argument("audio_path", help="Path to audio file")
    diar_parser.add_argument("--num-speakers", type=int, default=-1, help="Expected number of speakers (-1 for auto)")
    diar_parser.add_argument("--threshold", type=float, default=0.5, help="Clustering threshold (default: 0.5)")

    # Streaming Simulation
    stream_parser = subparsers.add_parser("stream", help="Simulate real-time streaming recognition on an audio file")
    stream_parser.add_argument("audio_path", help="Path to query audio stream")
    stream_parser.add_argument("--chunk-ms", type=int, default=500, help="Chunk size in milliseconds (default: 500ms)")
    stream_parser.add_argument("--window-sec", type=float, default=1.5, help="Sliding context window in seconds (default: 1.5s)")
    stream_parser.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold (default: 0.75)")
    
    # Evaluate Metrics Pipeline
    eval_parser = subparsers.add_parser("evaluate", help="Run full metrics pipeline and export report")
    eval_parser.add_argument("--data-dir", default="data", help="Path to speaker dataset directory (default: data)")
    eval_parser.add_argument("--output-dir", default="reports", help="Path to output reports directory (default: reports)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return

    recognizer = PyannoteSpeakerRecognizer()

    # Pre-enroll demo speakers from data/ if available
    if os.path.exists("data"):
        for spk in sorted(os.listdir("data")):
            spk_dir = os.path.join("data", spk)
            if os.path.isdir(spk_dir):
                wavs = sorted([os.path.join(spk_dir, f) for f in os.listdir(spk_dir) if f.endswith(".wav")])
                if wavs:
                    audio, _ = sf.read(wavs[0])
                    recognizer.enroll(spk, audio)

    if args.command == "enroll":
        audio, sr = sf.read(args.audio_path)
        recognizer.enroll(args.speaker_name, audio)
        print(f"[OK] Enrolled speaker '{args.speaker_name}' successfully from {args.audio_path}.")

    elif args.command == "identify":
        audio, sr = sf.read(args.audio_path)
        scores, elapsed = recognizer.match(audio)
        print(f"\nIdentification Results (Latency: {elapsed*1000:.2f} ms):")
        print("-" * 45)
        for spk, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  Speaker: {spk:<15} | Cosine Similarity: {score:.4f}")
        best_spk = max(scores, key=scores.get) if scores else "None"
        print("-" * 45)
        print(f"Predicted Speaker: {best_spk} (Score: {scores[best_spk]:.4f})\n")

    elif args.command == "diarize":
        audio, sr = sf.read(args.audio_path)
        turns = recognizer.diarize(audio, num_speakers=args.num_speakers, threshold=args.threshold)
        print(f"\nDiarization Results for {args.audio_path}:")
        print("-" * 55)
        for t in turns:
            print(f"  [{t.start:6.2f}s - {t.end:6.2f}s] {t.match.label:<15} (confidence: {t.match.score:.3f})")
        print("-" * 55)
        print(f"Total turns detected: {len(turns)}\n")

    elif args.command == "stream":
        audio, sr = sf.read(args.audio_path)
        chunk_sec = args.chunk_ms / 1000.0
        engine = RealTimeStreamingEngine(
            recognizer,
            sample_rate=sr,
            chunk_sec=chunk_sec,
            window_sec=args.window_sec,
            threshold=args.threshold
        )
        
        chunk_size = int(chunk_sec * sr)
        print(f"\n--- Starting Streaming Simulation: {args.audio_path} ---")
        print(f"Chunk Duration: {args.chunk_ms} ms | Sliding Window: {args.window_sec} s | Threshold: {args.threshold}\n")
        print(f"{'Time':<12} | {'Active Speaker':<16} | {'Confidence':<12} | {'Latency':<10} | {'RTF'}")
        print("-" * 65)
        
        for t_idx, i in enumerate(range(0, len(audio) - chunk_size, chunk_size)):
            t_sec = i / sr
            chunk = audio[i:i + chunk_size]
            spk, conf, lat, rtf = engine.process_chunk(chunk)
            print(f"{t_sec:>6.2f}s - {t_sec+chunk_sec:>5.2f}s | {spk:<16} | {conf:>10.4f} | {lat*1000:>6.2f} ms | {rtf:>6.4f}")
        print("-" * 65)
        print("Streaming simulation complete.\n")

    elif args.command == "evaluate":
        from src.metrics_pipeline import SpeakerMetricsPipeline
        pipeline = SpeakerMetricsPipeline(recognizer, data_dir=args.data_dir, output_dir=args.output_dir)
        pipeline.execute_pipeline()

if __name__ == "__main__":
    main()
