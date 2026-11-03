# Downloading models

Nothing is bundled in the repo: every model is fetched from HuggingFace on first use and stored in the HF cache (`~/.cache/huggingface`, override with `HF_HOME`). The first run of a command is therefore slow; pre-fetch the models to avoid that, or to work offline afterwards.

## Models

| Model | Used by | HuggingFace repo |
|---|---|---|
| Parakeet-TDT v3 0.6B (default ASR) | `--model parakeet` | `nvidia/parakeet-tdt-0.6b-v3` |
| Whisper large-v3-turbo | `--model whisper` | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` |
| Nemotron 3.5 ASR Streaming 0.6B | `--model nemotron` | `nvidia/nemotron-3.5-asr-streaming-0.6b` |
| TitaNet-small (speaker embeddings) | live labels, `identify`, `diarize`, SECS | `csukuangfj/speaker-embedding-models` |
| pyannote segmentation-3.0 (ONNX) | `diarize` | `csukuangfj/sherpa-onnx-pyannote-segmentation-3-0` |
| Silero VAD (ONNX) | streaming | `csukuangfj/vad` |

The ONNX repos and filenames are defined in [src/speck/hub.py](../src/speck/hub.py). Voxtral runs through the Mistral API (`MISTRAL_API_KEY`), so there is nothing to download unless `VOXTRAL_BACKEND=local` (then `VOXTRAL_MODEL` is fetched from HuggingFace).

## Pre-fetch

```bash
# Small ONNX models (speaker embeddings, VAD, segmentation)
uv run python scripts/download_assets.py --models

# ASR engines: download only the ones you plan to use
uv run hf download nvidia/parakeet-tdt-0.6b-v3
uv run hf download mobiuslabsgmbh/faster-whisper-large-v3-turbo
uv run hf download nvidia/nemotron-3.5-asr-streaming-0.6b
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
