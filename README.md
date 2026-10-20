<p align="center">
  <img src="assets/spek.png" alt="Spek" width="200" />
</p>

# Spek

Model-agnostic speech recognition with real-time streaming and live speaker classification.

| Engine | `--model` | Notes |
|---|---|---|
| NVIDIA Nemotron 3.5 (0.6B) | `nemotron` | Local, cache-aware streaming, 80–1120 ms latency |
| Mistral Voxtral Mini Transcribe 2 | `voxtral` | Cloud API (`MISTRAL_API_KEY`) or local weights |
| Sherpa-ONNX | `sherpa-onnx` | Speaker diarization + Whisper, live speaker tags |

## Setup

```bash
uv sync
cp .env.example .env   # pick SPEK_MODEL, add keys
```

## CLI

```bash
uv run spek transcribe samples/bcn_weather.mp3            # offline
uv run spek stream samples/obama.mp3 --lookahead 6        # streaming
uv run spek mic --language en-US                          # live mic
uv run spek mic --model sherpa-onnx                       # live mic, speaker-tagged
uv run spek listen                                        # auto-transcribe calls/videos (Linux)
uv run spek listen --app discord --save-dir ~/transcripts # only Discord, save each call
uv run spek listen --list                                 # show playing apps and capture sources
uv run spek mixed --model sherpa-onnx                     # mic + system audio: [You] vs. each remote voice
uv run spek mixed --no-classify --save call.txt           # [You] vs. [Remote], any engine
uv run spek speakers                                      # live "who is speaking"
uv run spek speakers --enroll Alice=alice.wav --enroll Bob=bob.wav
uv run spek serve --port 8000                             # web UI + API
uv run spek benchmark samples/bcn_weather.mp3
uv run spek models
```

Live speaker classification uses Silero VAD to cut utterances, then slides a TitaNet embedding
window over each one to catch hand-overs without a pause, and matches every turn against enrolled
people or speakers discovered on the fly (duplicates are merged as their voices converge). While
someone is still talking a provisional label is shown (`spek speakers`, the web UI, or
`on_speaker=` on a sherpa-onnx session). Tune with `--speaker-threshold` (or
`SPEK_SPEAKER_THRESHOLD`, default 0.35) and cap with `--num-speakers`.

### System audio (Linux)

`spek listen` watches PulseAudio/PipeWire (via `pactl`/`parec`; `libpulse` on Arch,
`pulseaudio-utils` on Debian). When an app starts playing, it records that sink's monitor
and transcribes it. If the same app is also recording a microphone (Discord, Zoom, Meet,
Teams...), it's treated as a call and your mic is mixed in (`--mic auto|on|off`). A recording
ends after `--idle-timeout` seconds of silence. Paused streams, notification sounds and volume
meters are ignored. Use `--source NAME` to record one source continuously instead, and
`--model sherpa-onnx` to tag call participants by speaker.

### Mixed mode (Linux)

`spek mixed` records the microphone and system audio as two separate channels and merges them
into one transcript. The channel itself says who is talking: the mic is `[You]` (`--me NAME`),
and system audio is either split by voice into `[Speaker N]` / enrolled names (`--classify`,
sherpa-onnx, default) or tagged `[Remote]` (`--no-classify`, works with every engine).
`--no-tag-speakers` gives plain merged text. Without headphones the mic hears the speakers too,
so `--echo-guard` (default on) mutes mic audio whose loudness follows what is playing; turn it
off with headphones. Pick devices with `--mic-source` / `--system-source` (`spek listen --list`).

## Python

```python
from spek import get_engine

engine = get_engine("sherpa-onnx").load()
print(engine.transcribe("samples/bcn_weather.mp3").text)

session = engine.create_stream_session(
    on_token=lambda t: print(t, end=""),
    enroll={"Alice": "alice.wav"},
).start()
session.push_audio(pcm_chunk)  # 16 kHz float32, from mic / socket
session.finish(); session.join()
```

## API

- `GET /` – browser mic UI
- `WS /ws/stream` – JSON config, then float32 PCM frames, then `{"type": "eof"}`
- `POST /transcribe` – audio file upload
- `GET /health`, `GET /models`

## Tests

```bash
uv run pytest
```
