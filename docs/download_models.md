# Downloading models

Nothing is bundled in the repo: every model is fetched from HuggingFace on first use and stored in the HF cache (`~/.cache/huggingface`, override with `HF_HOME`). The first run of a command is therefore slow; pre-fetch the models to avoid that, or to work offline afterwards.

```bash
uv run speck setup          # ONNX models + SPECK_MODEL engine (Parakeet), then a test transcription
uv run speck setup --all    # also Whisper
```

## Models

| Model | Used by | HuggingFace repo |
|---|---|---|
| Parakeet-TDT v3 0.6B (default ASR) | `--model parakeet` | `nvidia/parakeet-tdt-0.6b-v3` |
| Whisper large-v3-turbo | `--model whisper` | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` |
| TitaNet-small (speaker embeddings) | live labels, `identify`, `diarize`, SECS | `csukuangfj/speaker-embedding-models` |
| pyannote segmentation-3.0 (ONNX) | `diarize` | `csukuangfj/sherpa-onnx-pyannote-segmentation-3-0` |
| Silero VAD (ONNX) | streaming | `csukuangfj/vad` |

The ONNX repos and filenames are registered in [src/speck/hub.py](../src/speck/hub.py) (`EMBEDDING_MODELS`, `SEGMENTATION_MODELS`, `VAD_MODELS`).

## Pre-fetch manually

What `speck setup` does, step by step:

```bash
# Small ONNX models (speaker embeddings, VAD, segmentation)
uv run python scripts/download_assets.py --models

# ASR engines: download only the ones you plan to use
uv run hf download nvidia/parakeet-tdt-0.6b-v3
uv run hf download mobiuslabsgmbh/faster-whisper-large-v3-turbo
```

## Authentication and rate limits

Set `HF_TOKEN` (or run `uv run hf auth login`) if a download is rate-limited or a repo requires accepting its license.

## Offline use

After pre-fetching, set `HF_HUB_OFFLINE=1` to stop any network lookups.

## Verify

```bash
uv run speck transcribe samples/bcn_weather.mp3
```

Datasets (CMU Arctic, LibriSpeech) and the audio format: [data_and_models.md](data_and_models.md). Model choice and benchmarks: [models_report.md](models_report.md).
