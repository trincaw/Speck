# Metrics & Evaluation

Two pipelines, run through `main.py evaluate`. Dataset layout and assets: [data_and_models.md](data_and_models.md). Observed results: [models_report.md](models_report.md).

```bash
python scripts/download_assets.py --all
python main.py evaluate --task speaker --data-dir data/ --output-dir reports/
python main.py evaluate --task tts --output-dir reports/
python main.py evaluate                      # both (--task all, default)
uv run pytest tests/test_tts_metrics.py -v
```

## Speaker recognition (`--task speaker`)

EER, NIST minDCF, AUC-ROC, real-time factor, streaming chunk latency.

## TTS & intelligibility (`--task tts`, `en-US` and `it-IT`)

| Group | Metrics |
|---|---|
| Acoustic | MCD (DTW, dB) |
| Prosody | F0 RMSE (Hz, cents), F0 PCC, VDE, GPE |
| Identity | SECS (speaker-embedding cosine similarity) |
| Linguistic | WER, CER, MER, WIL |

Text normalizers: `en-US` expands contractions (`it's` → `it is`); `it-IT` handles elisions (`l'albero`, `c'è`) and keeps accents.

## Output (`reports/`, git-ignored)

| File | Content |
|---|---|
| `EVALUATION_REPORT.md`, `metrics_summary.json` | Speaker report / JSON |
| `TTS_EVALUATION_REPORT.md`, `tts_metrics_summary.json` | TTS report / JSON |
| `roc_curve.csv` | Verification ROC points |
