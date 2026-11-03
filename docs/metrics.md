# Metrics & Evaluation

Two pipelines, run through `speck evaluate` (the speaker models are only loaded for `--task speaker` / `all`). Dataset layout and assets: [data_and_models.md](data_and_models.md). Observed results: [models_report.md](models_report.md).

```bash
python scripts/download_assets.py --all
speck evaluate --task speaker --data-dir data/ --output-dir reports/
speck evaluate --task tts --output-dir reports/
speck evaluate                      # both (--task all, default)
uv run pytest tests/test_tts_metrics.py -v
```

## Speaker recognition (`--task speaker`)

pyannote segmentation-3.0 + TitaNet-small (match threshold 0.35, diarization clustering threshold 0.7; calibration in [models_report.md](models_report.md)). Reads `--data-dir` (one folder per speaker, at least 2 WAVs each; layout in [data_and_models.md](data_and_models.md)): the first WAV of each speaker is enrolled, the rest are queries.

| Group | Metrics |
|---|---|
| Verification (1:1) | EER, NIST minDCF (P_target 0.01 and 0.001), AUC-ROC, target/non-target score mean and std |
| Identification (1:N) | Top-1 accuracy, macro precision / recall / F1, confusion matrix |
| Efficiency | Real-time factor, latency percentiles (P50/P90/P95/P99) |
| Streaming | Chunk latency and RTF on 500 ms chunks with a 1.5 s window, average time to first correct speaker |

## TTS & intelligibility (`--task tts`, `en-US` and `it-IT`)

`speck evaluate --task tts` does not read any dataset: it scores two built-in synthetic pairs (a sine-tone "reference" and a slightly scaled, noised copy, one per locale), with the ASR text set equal to the prompt. WER/CER are therefore trivially 0 and the numbers only confirm the pipeline runs. The pairs come from `speck.evaluation.tts.synthetic_benchmark()`. To score real synthesis, pass items to `TTSMetricsPipeline.evaluate_dataset` (keys: `ref_audio`, `syn_audio`, `text`, optional `asr_text`, `language`, `synthesis_time_sec`) and then `export_reports`.

| Group | Metrics |
|---|---|
| Acoustic | MCD (DTW, dB) |
| Prosody | F0 RMSE (Hz, cents), F0 PCC, VDE, GPE |
| Identity | SECS (cosine similarity of TitaNet-small speaker embeddings) |
| Linguistic | WER, CER, MER, WIL |

Text normalizers (also used by WER/CER everywhere, including `scripts/asr_benchmark.py`): `en-US` expands contractions (`it's` → `it is`), drops fillers and writes spelled-out numbers as digits; `it-IT` splits elisions (`l'albero` → `l' albero`), drops fillers and keeps accents.

## Output (`reports/`, git-ignored)

| File | Content |
|---|---|
| `EVALUATION_REPORT.md`, `metrics_summary.json` | Speaker report / JSON |
| `TTS_EVALUATION_REPORT.md`, `tts_metrics_summary.json` | TTS report / JSON |
| `roc_curve.csv` | Verification ROC points |
| `asr_benchmark.json` | ASR WER/CER/RTF and hypotheses, from `scripts/asr_benchmark.py` (default `--output`) |
| `speaker_benchmark.json` | Speaker-embedding EER/top-1/RTF per model, from `scripts/speaker_benchmark.py` |
| `historic/` | Earlier reports kept as historic data (see [models_report.md](models_report.md#historic-benchmarks)) |
