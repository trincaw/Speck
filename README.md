<p align="center">
  <img src="assets/speck.png" alt="Speck" width="200" />
</p>

# Speck

Linux speech recognition with real-time streaming and live speaker labels. Parakeet by default, Whisper as the alternative.

## Quick start

```bash
uv sync && uv run speck setup     # install, fetch the default models, verify on a sample
uv run speck mixed                # start live transcription
```

`speck setup` creates `.env` from [.env.example](.env.example) if missing, downloads the speaker, segmentation and VAD
models plus the `SPECK_MODEL` engine (Parakeet) into `~/.cache/huggingface`, then transcribes `samples/bcn_weather.mp3`.
`--all` also fetches Whisper, `--model whisper` fetches only Whisper, `--no-verify` skips the check.
Whisper on GPU needs the CUDA 12 cuBLAS/cuDNN libraries (CTranslate2 is built for CUDA 12); without them use `--device cpu`.

Python 3.12+; `pip install -e .` works too. The `speck` name on PyPI is a different project.

| Role | Model |
|---|---|
| ASR (default) | `parakeet`: NVIDIA Parakeet-TDT v3 0.6B, 25 European languages |
| ASR (alternative) | `whisper`: faster-whisper large-v3-turbo, 99 languages, language lock, hotwords |
| Speakers | TitaNet-small embeddings (best of 9 benchmarked), pyannote segmentation-3.0, Silero VAD |

Benchmarks: [docs/models_report.md](docs/models_report.md). Manual downloads, tokens, offline mode: [docs/download_models.md](docs/download_models.md).

## CLI

| Command | Does |
|---|---|
| `speck setup` | Fetch the default models and verify the install |
| `speck transcribe AUDIO` | Offline transcription |
| `speck stream AUDIO` | Streaming transcription of a file (`--simulate-realtime`) |
| `speck mic` | Live microphone transcription |
| `speck speakers [AUDIO]` | Live "who is speaking", no transcription |
| `speck identify AUDIO` | Which enrolled speaker is this (`--stream`: chunk by chunk over a sliding window) |
| `speck diarize AUDIO` | Offline "who spoke when" (pyannote segmentation-3.0 + TitaNet-small) |
| `speck listen` | Auto-transcribe calls and videos playing on this machine (Linux) |
| `speck mixed` | Mic + system audio: `[You]` vs. `[Remote]` (Linux) |
| `speck models` / `speck languages` | List engines / languages |
| `speck evaluate` | Speaker / TTS metric pipelines, reports in `reports/` ([docs](docs/metrics.md)) |

Common flags (`speck <command> --help` for the rest):

| Flag | Meaning |
|---|---|
| `--model NAME`, `--device cuda\|cpu` | Engine alias or checkpoint, and where to run it |
| `--language CODE` | Locale like `en-US`, or `auto`; default is `SPECK_LANGUAGE` (`auto`) |
| `--speakers` | Label speakers (`mic`, `listen`; already on in `mixed`, `--no-speakers` turns it off) |
| `--enroll NAME=WAV` | Known voice sample, repeatable (also on `speakers`, `identify`, `diarize`) |
| `--enroll-dir DIR` | `identify` / `diarize`: enroll the first WAV of each `DIR/<speaker>/` (default `data`) |
| `--speaker-threshold X` | Match strictness, default 0.35 (the speaker model's calibration). Raise if people get merged, lower if one is split |
| `--num-speakers N` | Cap on speakers discovered |

```bash
speck transcribe samples/bcn_weather.mp3
speck mic --speakers --enroll Alice=alice.wav
speck listen --app discord --save-dir ~/transcripts     # --list shows playing apps
speck mixed --speakers --save call.txt                  # --me/--remote rename labels, --no-echo-guard with headphones
```

`listen` and `mixed` need PulseAudio/PipeWire (`pactl`, `parec`). `mixed` labels the microphone `[You]` and, with speakers on, remote voices `Speaker 1`, `Speaker 2`...; `[Remote]` is used only with `--no-speakers`.

## Library

```python
from speck import get_engine, SpeakerClassifier, SpeakerEmbedder

engine = get_engine().load()                # SPECK_MODEL, default "parakeet"; or get_engine("whisper")
print(engine.transcribe("audio.mp3", language="en-US").text)

# live: push 16 kHz mono float32 chunks, read text as it arrives
session = engine.create_stream_session(
    on_token=lambda t: print(t, end="", flush=True),
    speakers=SpeakerClassifier(SpeakerEmbedder()),   # optional: label who is speaking
).start()
session.push_audio(chunk)
session.finish(); session.join()
```

Add an engine by subclassing `speck.asr.BaseASREngine` (`name`, `sampling_rate`, `load`, `transcribe`, `stream_transcribe`), then either call `speck.register_engine(alias, cls)` at runtime or, to ship it built in, add an `EngineSpec` to `ENGINE_SPECS` in [src/speck/asr/registry.py](src/speck/asr/registry.py). Live sessions, speaker labels, hotwords and `speck setup` work with any engine. An unrecognized model string falls back to Parakeet.

Speaker, segmentation and VAD models work the same way: one entry each in `EMBEDDING_MODELS`, `SEGMENTATION_MODELS` and `VAD_MODELS` in [src/speck/hub.py](src/speck/hub.py). An embedding model carries its own match and clustering thresholds (measure them with `scripts/speaker_benchmark.py`); select it with `SPECK_SPEAKER_MODEL`.

## Lowering the error rate

These settings can be activated to lower the error rate.

| Setting (`.env`) | Effect |
|---|---|
| `SPECK_LANGUAGE=en` | Locks Whisper to one language (no identification pass, no false switches); also the CLI `--language` default |
| `SPECK_HOTWORDS=Docker,Kubernetes` | Biases Whisper decoding and snaps near-miss words in every engine's output to this vocabulary |
| `SPECK_BEAM_SIZE=5` | Whisper beam width (default 5) |
| `SPECK_NORMALIZE_AUDIO=1` | DC removal and peak normalization of offline audio (`0` disables) |

`speck.rover([text_a, text_b, text_c])` merges transcripts from several engines by word-level majority vote (the first one is the pivot on ties).

WER is computed on normalized text: punctuation, casing and hesitation fillers ("uh", "um") are ignored in English and
Italian, and spelled-out numbers ("35" vs "thirty five") in English only (the script defaults to `--language en-US`). Measure on a real set, not three clips:

```bash
python scripts/download_assets.py --librispeech test-clean          # ~1 h subset
python scripts/asr_benchmark.py --models parakeet whisper-large-v3-turbo   # writes reports/asr_benchmark.json; --ensemble needs 3+ models (checkpoints count)
python scripts/speaker_benchmark.py --models nemo_en_titanet_small.onnx nemo_en_titanet_large.onnx   # speaker-embedding EER; writes reports/speaker_benchmark.json
```

## Project layout

```
src/speck/
├── asr/          BaseASREngine, engines/ (parakeet, whisper), registry, hotword + ROVER post-processing
├── speakers/     embeddings (TitaNet-small), live online classifier, offline recognizer / diarizer
├── streaming/    Silero VAD, push-based live sessions, mic + system audio mixing
├── capture/      microphone and PulseAudio / PipeWire sources
├── evaluation/   speaker, transcript (WER/CER) and TTS metrics and pipelines
├── cli/          `speck` command: app.py (entry point), options.py, commands/<domain>.py (setup.py: onboarding)
├── audio.py      array helpers    hub.py  speaker / VAD model registry on HuggingFace    config.py  .env settings
```

`scripts/` holds the dataset and benchmark tools: `download_assets.py`, `asr_benchmark.py` and `speaker_benchmark.py`.

## Tests

```bash
uv run pytest
```

## Docs

Read these before working on the related area:

- [docs/metrics.md](docs/metrics.md): speaker and TTS evaluation pipelines (`speck evaluate`), metrics, report files.
- [docs/download_models.md](docs/download_models.md): model repos, pre-fetch commands, HuggingFace token, offline use.
- [docs/data_and_models.md](docs/data_and_models.md): downloading models and datasets, audio format, dataset layout.
- [docs/models_report.md](docs/models_report.md): selected default models and their measured WER/RTF/EER, plus historic benchmarks of earlier runs and replaced models (kept as reference data).
