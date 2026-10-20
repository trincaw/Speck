<p align="center">
  <img src="assets/spek.png" alt="Spek" width="200" />
</p>

# Spek

Model-agnostic speech recognition with real-time streaming and live speaker labels.

| Engine | `--model` | Notes |
|---|---|---|
| NVIDIA Nemotron 3.5 (0.6B) | `nemotron` | Local, streaming, 80–1120 ms latency |
| Mistral Voxtral Mini Transcribe 2 | `voxtral` | Cloud API (`MISTRAL_API_KEY`) or local weights |

## Install

```bash
pip install spek
```

Installs the `spek` command and the Python library. Python 3.12+. Configure through environment variables
(see [.env.example](.env.example)); the CLI loads a `.env` automatically, library users can call `spek.load_env_file()`.

## CLI

| Command | Does |
|---|---|
| `spek transcribe AUDIO` | Offline transcription |
| `spek stream AUDIO` | Streaming transcription of a file (`--simulate-realtime`) |
| `spek mic` | Live microphone transcription |
| `spek speakers [AUDIO]` | Live "who is speaking", no transcription |
| `spek listen` | Auto-transcribe calls and videos playing on this machine (Linux) |
| `spek mixed` | Mic + system audio: `[You]` vs. `[Remote]` (Linux) |
| `spek benchmark AUDIO` | Real-time factor per latency preset |
| `spek models` / `spek languages` | List engines / languages |

Common flags (`spek <command> --help` for the rest):

| Flag | Meaning |
|---|---|
| `--model NAME`, `--device cuda\|cpu` | Engine alias or checkpoint, and where to run it |
| `--language CODE` | Locale like `en-US`, or `auto` (default) |
| `--lookahead 0\|3\|6\|13` | Latency preset: 80 / 320 / 560 (default) / 1120 ms. Lower is faster, higher is more accurate |
| `--speakers` | Label speakers (`mic`, `listen`, `mixed`) |
| `--enroll NAME=WAV` | Known voice sample, repeatable |
| `--speaker-threshold X` | Match strictness, default 0.35. Raise if people get merged, lower if one is split |
| `--num-speakers N` | Cap on speakers discovered |

```bash
spek transcribe samples/bcn_weather.mp3
spek mic --speakers --enroll Alice=alice.wav
spek listen --app discord --save-dir ~/transcripts     # --list shows playing apps
spek mixed --speakers --save call.txt                  # --me/--remote rename labels, --no-echo-guard with headphones
```

`listen` and `mixed` need PulseAudio/PipeWire (`pactl`, `parec`).

## Library

```python
from spek import get_engine, SpeakerClassifier, SpeakerEmbedder

engine = get_engine("nemotron").load()      # defaults to SPEK_MODEL
print(engine.transcribe("audio.mp3", language="en-US").text)

# live: push 16 kHz mono float32 chunks, read text as it arrives
session = engine.create_stream_session(
    on_token=lambda t: print(t, end="", flush=True),
    speakers=SpeakerClassifier(SpeakerEmbedder()),   # optional: label who is speaking
).start()
session.push_audio(chunk)
session.finish(); session.join()
```

Add your own engine by subclassing `BaseASREngine` and calling `spek.factory.register_engine(alias, cls)`.

## Tests

```bash
uv run pytest
```
