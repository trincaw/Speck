<p align="center">
  <img src="assets/spek.png" alt="Spek" width="200" />
</p>

# Spek

Model-agnostic speech recognition with real-time streaming and live speaker classification.

| Engine | `--model` | Notes |
|---|---|---|
| NVIDIA Nemotron 3.5 (0.6B) | `nemotron` | Local, cache-aware streaming, 80–1120 ms latency |
| Mistral Voxtral Mini Transcribe 2 | `voxtral` | Cloud API (`MISTRAL_API_KEY`) or local weights |

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
uv run spek mic --speakers --enroll Alice=alice.wav       # live mic, speaker-labelled text
uv run spek listen                                        # auto-transcribe calls/videos (Linux)
uv run spek listen --app discord --save-dir ~/transcripts # only Discord, save each call
uv run spek listen --list                                 # show playing apps and capture sources
uv run spek mixed --save call.txt                         # mic + system audio: [You] vs. [Remote]
uv run spek mixed --speakers                              # [You] vs. [Speaker 0], [Speaker 1]...
uv run spek speakers                                      # live "who is speaking"
uv run spek speakers --enroll Alice=alice.wav --enroll Bob=bob.wav
uv run spek serve --port 8000                             # web UI + API
uv run spek benchmark samples/bcn_weather.mp3
uv run spek models
```

Live speaker classification uses Silero VAD to cut utterances, then slides a TitaNet embedding
window over each one to catch hand-overs without a pause, and matches every turn against enrolled
people or speakers discovered on the fly (duplicates are merged as their voices converge). While
someone is still talking a provisional label is shown (`spek speakers`). Tune with `--speaker-threshold` (or
`SPEK_SPEAKER_THRESHOLD`, default 0.35) and cap with `--num-speakers`.

### Speaker-labelled transcripts (any engine)

`--speakers` on `mic` / `listen` / `mixed` (or `"speakers": true` in the WebSocket config, or
`engine.create_stream_session(speakers=SpeakerClassifier(SpeakerEmbedder()))`) runs:

| Stage | Input | Output |
|---|---|---|
| VAD | audio stream | utterances (start/end) |
| Speaker turns | utterance | single-speaker turns + label |
| ASR | each turn | text |

Speaker turns are found *before* ASR, so each turn's text gets that turn's label. No ASR timestamps
and no merge step are needed, so it works with any engine that implements `transcribe()`. Text
appears about 1 s after a turn ends. The WebSocket also sends `speaker` events (provisional label
while someone is talking) and `segment` events. When two discovered speakers turn out to be the same
person, `session.segments` / `session.transcript` (and the final `done` message) use the merged
label for all their turns.

### System audio (Linux)

`spek listen` watches PulseAudio/PipeWire (via `pactl`/`parec`; `libpulse` on Arch,
`pulseaudio-utils` on Debian). When an app starts playing, it records that sink's monitor
and transcribes it. If the same app is also recording a microphone (Discord, Zoom, Meet,
Teams...), it's treated as a call and your mic is mixed in (`--mic auto|on|off`). A recording
ends after `--idle-timeout` seconds of silence. Paused streams, notification sounds and volume
meters are ignored. Use `--source NAME` to record one source continuously instead.

### Mixed mode (Linux)

`spek mixed` records the microphone and system audio as two separate channels and merges them
into one transcript. The channel itself says who is talking: the mic is `[You]` (`--me NAME`),
and system audio is tagged `[Remote]` (`--remote NAME`).
With `--speakers`, each remote voice gets its own label instead: an enrolled name (`--enroll`)
or `Speaker N`, numbered in order of appearance. `--no-tag-speakers` gives plain merged text. Without headphones the mic hears the speakers too,
so `--echo-guard` (default on) mutes mic audio whose loudness follows what is playing; turn it
off with headphones. Pick devices with `--mic-source` / `--system-source` (`spek listen --list`).

## Python

```python
from spek import get_engine

engine = get_engine("nemotron").load()
print(engine.transcribe("samples/bcn_weather.mp3").text)

session = engine.create_stream_session(on_token=lambda t: print(t, end="")).start()
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
