#!/usr/bin/env python3
"""
Speaker-embedding benchmark: verification EER per embedding model on a LibriSpeech-style folder
(speaker = first-level directory), on whole utterances and on short crops like the live turns
the `speck` classifier embeds.

    python scripts/speaker_benchmark.py --models nemo_en_titanet_small.onnx wespeaker_en_voxceleb_resnet34_LM.onnx
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from speck.evaluation.speaker_metrics import compute_eer
from speck.hub import EMBEDDING_REPO, resolve_hf_file
from speck.speakers import SpeakerEmbedder


def load_items(data_dir, limit):
    files = sorted(Path(data_dir).rglob("*.flac")) + sorted(Path(data_dir).rglob("*.wav"))
    items = [{"speaker": f.relative_to(data_dir).parts[0], "audio": str(f)} for f in files]
    return items[:limit] if limit else items


def center_crop(wav, sr, seconds):
    n = int(seconds * sr)
    if len(wav) <= n:
        return wav
    start = (len(wav) - n) // 2
    return wav[start : start + n]


def verification(embs, speakers):
    """EER and threshold over every distinct pair (cosine on L2-normalized embeddings)."""
    sims = embs @ embs.T
    same = speakers[:, None] == speakers[None, :]
    upper = np.triu(np.ones_like(same, dtype=bool), k=1)
    eer, thr = compute_eer(sims[upper & same], sims[upper & ~same])
    return eer, thr


def identification(full, crops, speakers):
    """Top-1 accuracy: enroll each speaker's first whole utterance, query with crops of the rest."""
    enrolled = {}
    for i, spk in enumerate(speakers):
        enrolled.setdefault(spk, i)
    labels = list(enrolled)
    centroids = np.stack([full[enrolled[s]] for s in labels])
    queries = [i for i in range(len(speakers)) if i not in enrolled.values()]
    pred = np.argmax(crops[queries] @ centroids.T, axis=1)
    return float(np.mean([labels[p] == speakers[q] for p, q in zip(pred, queries)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help=f"ONNX files from {EMBEDDING_REPO}, or local paths")
    parser.add_argument("--data-dir", default="data/librispeech/test-clean", help="One folder per speaker")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N files (0 = all)")
    parser.add_argument("--crop", type=float, default=1.5, help="Short-crop length in seconds (live turns)")
    parser.add_argument("--threads", type=int, default=2, help="ONNX CPU threads (the live classifier uses 2)")
    parser.add_argument("--output", default="reports/speaker_benchmark.json")
    args = parser.parse_args()

    items = load_items(args.data_dir, args.limit)
    if not items:
        raise SystemExit(f"No audio under {args.data_dir}. Run: python scripts/download_assets.py --librispeech test-clean")
    speakers = np.array([i["speaker"] for i in items])
    audio = [sf.read(i["audio"], dtype="float32") for i in items]
    total_s = sum(len(w) / sr for w, sr in audio)
    print(f"{len(items)} files, {len(set(speakers))} speakers, {total_s / 60:.1f} min")

    results = {}
    for name in args.models:
        path = name if os.path.isfile(name) else resolve_hf_file(None, EMBEDDING_REPO, name)
        embedder = SpeakerEmbedder(model=path, num_threads=args.threads)
        start = time.perf_counter()
        full = np.stack([embedder.embed(w) for w, _ in audio])
        rtf = (time.perf_counter() - start) / total_s
        crops = np.stack([embedder.embed(center_crop(w, sr, args.crop)) for w, sr in audio])
        eer_full, thr_full = verification(full, speakers)
        eer_crop, thr_crop = verification(crops, speakers)
        results[name] = {
            "dim": embedder.dim,
            "eer_full": eer_full,
            "threshold_full": thr_full,
            "eer_crop": eer_crop,
            "threshold_crop": thr_crop,
            "top1_crop": identification(full, crops, speakers),
            "rtf_cpu": rtf,
        }
        r = results[name]
        print(f"{name:58s} EER {eer_full:6.2%} / {eer_crop:6.2%} @{args.crop:g}s  thr {thr_full:.2f}/{thr_crop:.2f}"
              f"  top1 {r['top1_crop']:6.2%}  RTF {rtf:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "files": len(items), "speakers": len(set(speakers)),
                   "crop_seconds": args.crop, "results": results}, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
