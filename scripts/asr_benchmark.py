#!/usr/bin/env python3
"""
ASR accuracy benchmark: corpus-level WER/CER per engine on a LibriSpeech-style folder
(`*.trans.txt` files next to the FLAC/WAV audio), optionally with a ROVER ensemble of all engines.

    python scripts/asr_benchmark.py --models parakeet whisper-large-v3-turbo --limit 100
"""

import argparse
import json
import os
import time
from pathlib import Path

from speck.asr import get_engine, rover
from speck.config import load_env_file
from speck.evaluation.text import compute_cer, compute_wer


def load_manifest(data_dir):
    items = []
    for trans in sorted(Path(data_dir).rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt_id, _, text = line.partition(" ")
            for ext in (".flac", ".wav"):
                audio = trans.parent / f"{utt_id}{ext}"
                if audio.is_file():
                    items.append({"id": utt_id, "audio": str(audio), "reference": text})
                    break
    return items


def score(references, hypotheses, language):
    """Corpus WER/CER: total errors over total reference words (not a mean of per-utterance ratios)."""
    w_err = w_tot = c_err = c_tot = 0
    for ref, hyp in zip(references, hypotheses):
        w = compute_wer(ref, hyp, language=language)
        c = compute_cer(ref, hyp, language=language)
        w_err += w["substitutions"] + w["deletions"] + w["insertions"]
        w_tot += w["ref_words"]
        c_err += c["substitutions"] + c["deletions"] + c["insertions"]
        c_tot += c["ref_chars"]
    return {"wer": w_err / max(w_tot, 1), "cer": c_err / max(c_tot, 1), "ref_words": w_tot}


def main():
    load_env_file()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help="Engine aliases (see `speck models`)")
    parser.add_argument("--data-dir", default="data/librispeech", help="Folder with *.trans.txt manifests")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N utterances (0 = all)")
    parser.add_argument("--language", default="en-US", help="Locale passed to engines and used for text normalization (default: en-US)")
    parser.add_argument("--ensemble", action="store_true", help="Also score a ROVER vote across all models")
    parser.add_argument("--output", default="reports/asr_benchmark.json")
    args = parser.parse_args()

    items = load_manifest(args.data_dir)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"No utterances under {args.data_dir}. Run: python scripts/download_assets.py --librispeech test-clean")
    refs = [i["reference"] for i in items]
    print(f"{len(items)} utterances, {sum(len(r.split()) for r in refs)} reference words")

    results, all_hyps = {}, {}
    for name in args.models:
        engine = get_engine(name).load()
        engine.transcribe(items[0]["audio"], language=args.language)  # warm-up (CUDA init, kernel compile)
        hyps, infer, audio_s = [], 0.0, 0.0
        for item in items:
            res = engine.transcribe(item["audio"], language=args.language)
            hyps.append(res.text)
            infer += res.inference_seconds
            audio_s += res.duration_seconds
        all_hyps[name] = hyps
        results[name] = {**score(refs, hyps, args.language), "rtf": infer / audio_s}
        print(f"{name:28s} WER {results[name]['wer']:6.2%}  CER {results[name]['cer']:6.2%}  RTF {results[name]['rtf']:.3f}")
        del engine
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    if args.ensemble and len(args.models) > 2:
        voted = [rover([all_hyps[m][k] for m in args.models]) for k in range(len(items))]
        results["rover(" + "+".join(args.models) + ")"] = score(refs, voted, args.language)
        r = results[list(results)[-1]]
        print(f"{'ROVER ensemble':28s} WER {r['wer']:6.2%}  CER {r['cer']:6.2%}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "utterances": len(items), "results": results, "hypotheses": all_hyps}, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
