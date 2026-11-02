# Models Report

Observed performance of each model, taken from `reports/` (git-ignored; regenerate with `scripts/asr_benchmark.py` and `main.py evaluate`).
WER/CER are on normalized text. RTF = inference time / audio time (lower is faster; < 1 is real-time).

## ASR: LibriSpeech test-clean subset

200 utterances, 4456 reference words. Source: `reports/asr_benchmark.json` (2026-10-30).

| Model | WER | CER | RTF |
|---|---|---|---|
| `whisper-large-v3-turbo` | 2.09% | 0.96% | 0.0401 |
| `parakeet` | 2.18% | 0.75% | 0.0115 |
| `nemotron` | 3.43% | 1.32% | 0.0207 |
| ROVER (all three) | 2.18% | 0.89% | n/a |

- `parakeet` is the best accuracy/speed trade-off: within 0.1 pt of the best WER, the lowest CER, and ~3.5x faster than Whisper large-v3-turbo.
- ROVER did not beat the best single engine on WER.

## ASR: small-model check (3 clips)

RTX 4060, CUDA 12. Source: `reports/ASR_COMPARISON_REPORT.md` (2026-10-26). Three clips only, run before text normalization ignored fillers and number format, so the Nemotron WER is overstated. Use for relative speed, not accuracy.

| Model | WER | CER | RTF |
|---|---|---|---|
| `whisper-tiny` | 2.02% | 1.25% | 0.0111 |
| `whisper-base` | 0.00% | 0.00% | 0.0121 |
| `whisper-small` | 1.55% | 0.00% | 0.0188 |
| `nemotron` | 10.54% | 8.53% | 0.0392 |

Nemotron streaming RTF by latency preset (11.04 s clip):

| Lookahead | Latency | RTF |
|---|---|---|
| 0 | 80 ms | 0.523 |
| 3 | 320 ms | 0.143 |
| 6 | 560 ms | 0.089 |
| 13 | 1120 ms | 0.056 |

## Speaker recognition

pyannote segmentation-3.0 + WeSpeaker ResNet34, 12 test samples, 3 speakers (bdl, clb, slt). Source: `reports/EVALUATION_REPORT.md` (2026-10-26).

| Metric | Result |
|---|---|
| EER | 3.00% |
| AUC-ROC | 0.9987 |
| minDCF (P_target = 0.01) | 0.0667 |
| Top-1 identification | 100% |
| RTF | 0.0457 |
| P95 verification latency | 163 ms |
| Streaming chunk latency (500 ms chunk) | 35 ms |

Small test set; 100% top-1 on 12 samples says little about larger speaker pools.

## Not measured

- `voxtral`: no results recorded.
- `whisper` default alias: only `large-v3-turbo`, `tiny`, `base`, `small` have results.
