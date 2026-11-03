# Models Report

Measured performance of every model Speck ships with, the models selected as defaults, and **historic benchmarks** of earlier runs and replaced models (kept as reference data, not regenerated).
WER/CER are on normalized text. RTF = inference time / audio time (lower is faster; < 1 is real-time). Hardware: RTX 4060 (8 GB), CUDA; speaker models on CPU, 2 threads.
Raw results live in `reports/` (git-ignored); historic sources in `reports/historic/`.

## Selected models

| Role | Model | Used by | Why |
|---|---|---|---|
| ASR (default, `SPECK_MODEL`) | `parakeet` (Parakeet-TDT v3 0.6B) | every `speck` command, unrecognized `--model` values | Best WER, CER and RTF of all engines |
| ASR, multilingual control | `whisper` (large-v3-turbo) | `--model whisper` | Second-best WER; language lock and hotword biasing. Beats large-v3 on accuracy and speed |
| ASR, lowest latency | `nemotron` (Nemotron 3.5 streaming 0.6B) | `--model nemotron`, `speck benchmark` | Only native streaming engine (`--lookahead` 80-1120 ms) |
| Speaker embeddings | TitaNet-small | live `--speakers`, `identify` / `diarize`, SECS | Lowest EER of 9 candidates, 2nd-fastest |
| Diarization clustering threshold | 0.7 (TitaNet-small) | `speck diarize` | Speaker count right in 15/16 test conversations |

## ASR: LibriSpeech test-clean subset (2026-11-02)

723 utterances (18 speakers, ~93 min). Source: `reports/asr_benchmark.json`; regenerate with `scripts/asr_benchmark.py --models parakeet whisper-large-v3-turbo nemotron`.

| Model | WER | CER | RTF |
|---|---|---|---|
| `parakeet` | **1.98%** | **0.61%** | **0.0125** |
| `whisper-large-v3-turbo` | 2.26% | 0.87% | 0.0428 |
| `whisper-large-v3` | 2.49% | 1.19% | 0.0801 |
| `nemotron` | 3.37% | 1.16% | 0.0229 |

- `whisper-large-v3` is worse than turbo on every metric and ~2x slower; its alias was removed (still reachable by repo id, e.g. `--model Systran/faster-whisper-large-v3`).
- `whisper-medium`/`small`/`base`/`tiny` were not re-run (too slow to justify; the historic 3-clip run below already showed no accuracy edge); their aliases were removed.
- Whisper on GPU needs the CUDA 12 cuBLAS/cuDNN libraries (CTranslate2 4.8 is built for CUDA 12); see the README.

## Speaker embeddings: LibriSpeech test-clean subset (2026-11-02)

Same 723 files, 18 speakers. EER over all pairs, on whole utterances and on centered 1.5 s crops (the length of live turns); "thr" is the EER threshold (cosine). Top-1: enroll one whole utterance per speaker, query with 1.5 s crops of the rest. Source: `reports/speaker_benchmark.json`; regenerate with `scripts/speaker_benchmark.py`.

| Model | EER full | EER 1.5 s | thr full / 1.5 s | Top-1 (1.5 s) | RTF (CPU) |
|---|---|---|---|---|---|
| `nemo_en_titanet_small` (selected) | **0.19%** | **1.35%** | 0.41 / 0.31 | **100%** | 0.0102 |
| `nemo_en_titanet_large` | 0.26% | 1.39% | 0.38 / 0.28 | 100% | 0.0271 |
| `3dspeaker_campplus_sv_zh_en_16k-common_advanced` | 0.49% | 1.59% | 0.43 / 0.33 | 99.7% | 0.0094 |
| `3dspeaker_eres2net_sv_en_voxceleb_16k` | 1.28% | 3.05% | 0.38 / 0.27 | 98.4% | 0.0272 |
| `wespeaker_en_voxceleb_resnet34_LM` (previous speaker-tool model) | 9.16% | 18.73% | 0.73 / 0.60 | 68.5% | 0.0218 |
| `wespeaker_en_voxceleb_resnet152_LM` | 11.69% | 19.88% | 0.79 / 0.66 | 64.5% | 0.0721 |
| `wespeaker_en_voxceleb_resnet293_LM` | 15.19% | 22.20% | 0.84 / 0.72 | 56.6% | 0.1540 |
| `3dspeaker_campplus_sv_en_voxceleb_16k` | 36.70% | 12.74% | 0.51 / 0.52 | 46.5% | 0.0095 |
| `wespeaker_en_voxceleb_CAM++_LM` | 44.53% | 36.50% | 0.44 / 0.76 | 6.7% | 0.0095 |

- WeSpeaker and the VoxCeleb CAM++ exports score far below their published numbers through sherpa-onnx (likely a front-end mismatch); measured here as used by Speck.
- TitaNet-small's 1.5 s EER threshold (0.31) is consistent with the live classifier default `SPECK_SPEAKER_THRESHOLD=0.35`, which is also now the `speck identify` match threshold (was 0.65 / 0.75, tuned to ResNet34's higher score scale).

### Diarization (`speck diarize`)

16 synthetic conversations of 2-4 LibriSpeech speakers (2 turns each, 0.3 s gaps), `num_speakers=-1`. Frame accuracy after optimal label mapping.

| Embedding | Clustering threshold | Frame accuracy | Exact speaker count |
|---|---|---|---|
| WeSpeaker ResNet34 (previous) | 0.5 | 0.721 | 25% |
| TitaNet-small | 0.5 | 0.827 | 62% |
| TitaNet-small | 0.6 | 0.859 | 88% |
| **TitaNet-small** | **0.7** | **0.875** | **94%** |
| TitaNet-small | 0.8 | 0.847 | 81% |
| TitaNet-small | 0.9 | 0.816 | 62% |

### `speck evaluate --task speaker` (CMU Arctic, 2026-11-02)

pyannote segmentation-3.0 + TitaNet-small, 12 test samples, 3 speakers (bdl, clb, slt). Source: `reports/EVALUATION_REPORT.md`.

| Metric | Result |
|---|---|
| EER | 0.00% |
| minDCF (P_target = 0.01) | 0.0000 |
| Top-1 identification | 100% |
| RTF | 0.0194 |
| P95 verification latency | 69 ms |
| Streaming chunk latency (500 ms chunk) | 15 ms |

Small test set; use the LibriSpeech table above to compare models.

## Not measured

- `voxtral`: needs `MISTRAL_API_KEY` (API backend) or ~8 GB VRAM for the 4B local weights.

## Historic benchmarks

Earlier runs, kept unchanged as historic data. They were measured on smaller sets or with models that are no longer defaults; do not use them to choose a model.

### ASR: LibriSpeech test-clean, 200 utterances (2026-10-30)

4456 reference words. Source: `reports/historic/asr_benchmark_2026-10-30.json`.

| Model | WER | CER | RTF |
|---|---|---|---|
| `whisper-large-v3-turbo` | 2.09% | 0.96% | 0.0401 |
| `parakeet` | 2.18% | 0.75% | 0.0115 |
| `nemotron` | 3.43% | 1.32% | 0.0207 |
| ROVER (all three) | 2.18% | 0.89% | n/a |

ROVER did not beat the best single engine on WER.

### ASR: small-model check, 3 clips (2026-10-26)

Source: `reports/historic/ASR_COMPARISON_REPORT_2026-10-26.md` (one-off run, no script regenerates it). Run before text normalization ignored fillers and number format, so the Nemotron WER is overstated. Whisper tiny/base/small aliases have since been removed.

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

### Speaker recognition with WeSpeaker ResNet34 (2026-10-26)

Previous speaker-tool model (pyannote segmentation-3.0 + WeSpeaker ResNet34), CMU Arctic, 12 test samples, 3 speakers. Source: `reports/historic/EVALUATION_REPORT_2026-10-26_resnet34.md`.

| Metric | Result |
|---|---|
| EER | 3.00% |
| AUC-ROC | 0.9987 |
| minDCF (P_target = 0.01) | 0.0667 |
| Top-1 identification | 100% |
| RTF | 0.0457 |
| P95 verification latency | 163 ms |
| Streaming chunk latency (500 ms chunk) | 35 ms |
