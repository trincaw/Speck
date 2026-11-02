<p align="center">
  <img src="assets/speck.png" alt="Speck" width="200" />
</p>

# Speck

Model-agnostic speech recognition with real-time streaming and live speaker labels.

## Install

```bash
uv sync                 # or: pip install -e .
uv run speck --help
```

Installs the `speck` command and the Python library from this checkout (Python 3.12+). The `speck` name on PyPI is a different project. Configure through environment variables
(see [.env.example](.env.example))
Speaker embeddings (TitaNet-small) and the other ONNX models are fetched from HuggingFace on first use.
Default models: Parakeet for ASR, Whisper large-v3-turbo and Nemotron (streaming) as alternatives, TitaNet-small for speakers ([benchmarks](docs/models_report.md)).

## CLI

| Command | Does |
|---|---|
| `speck transcribe AUDIO` | Offline transcription |
| `speck stream AUDIO` | Streaming transcription of a file (`--simulate-realtime`) |
| `speck mic` | Live microphone transcription |
| `speck speakers [AUDIO]` | Live "who is speaking", no transcription |
| `speck listen` | Auto-transcribe calls and videos playing on this machine (Linux) |
| `speck mixed` | Mic + system audio: `[You]` vs. `[Remote]` (Linux) |
| `speck benchmark AUDIO` | Real-time factor per latency preset |
| `speck models` / `speck languages` | List engines / languages |

Common flags (`speck <command> --help` for the rest):

| Flag | Meaning |
|---|---|
| `--model NAME`, `--device cuda\|cpu` | Engine alias or checkpoint, and where to run it |
| `--language CODE` | Locale like `en-US`, or `auto`; default is `SPECK_LANGUAGE` (`auto`) |
| `--lookahead 0\|3\|6\|13` | Latency preset: 80 / 320 / 560 (default) / 1120 ms. Lower is lower latency, higher is more accurate. Only Nemotron uses it; other engines ignore it |
| `--speakers` | Label speakers (`mic`, `listen`; already on in `mixed`, `--no-speakers` turns it off) |
| `--enroll NAME=WAV` | Known voice sample, repeatable (also on `speakers`) |
| `--speaker-threshold X` | Match strictness, default 0.35. Raise if people get merged, lower if one is split |
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

engine = get_engine("parakeet").load()      # defaults to SPECK_MODEL (parakeet)
print(engine.transcribe("audio.mp3", language="en-US").text)

# live: push 16 kHz mono float32 chunks, read text as it arrives
session = engine.create_stream_session(
    on_token=lambda t: print(t, end="", flush=True),
    speakers=SpeakerClassifier(SpeakerEmbedder()),   # optional: label who is speaking
).start()
session.push_audio(chunk)
session.finish(); session.join()
```

Add your own engine by subclassing `BaseASREngine` and calling `speck.factory.register_engine(alias, cls)`. An unrecognized model string falls back to Parakeet.

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
python scripts/asr_benchmark.py --models parakeet whisper-large-v3-turbo nemotron --ensemble   # --ensemble needs 3+ models; writes reports/asr_benchmark.json
python scripts/speaker_benchmark.py --models nemo_en_titanet_small.onnx nemo_en_titanet_large.onnx   # speaker-embedding EER; writes reports/speaker_benchmark.json
```

## Speaker tool and benchmarks

`main.py` is a separate speaker-recognition tool (pyannote segmentation-3.0 + TitaNet-small, the same embeddings as the live classifier used by the `speck` CLI):
`enroll`, `identify`, `diarize`, `stream` and `evaluate`. Enrolled profiles live in memory only, and `main.py` pre-enrolls the first WAV of each `data/<speaker>/` folder.
`scripts/` holds `download_assets.py`, `asr_benchmark.py` and `speaker_benchmark.py`.

## Tests

```bash
uv run pytest
```

## Docs

Read these before working on the related area:

- [docs/metrics.md](docs/metrics.md): speaker and TTS evaluation pipelines (`main.py evaluate`), metrics, report files.
- [docs/data_and_models.md](docs/data_and_models.md): downloading models and datasets, audio format, dataset layout.
- [docs/models_report.md](docs/models_report.md): selected default models and their measured WER/RTF/EER, plus historic benchmarks of earlier runs and replaced models (kept as reference data).
