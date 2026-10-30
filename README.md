<p align="center">
  <img src="assets/speck.png" alt="Speck" width="200" />
</p>

# Speck

Model-agnostic speech recognition with real-time streaming and live speaker labels.

| Engine | `--model` | Notes |
|---|---|---|
| NVIDIA Nemotron 3.5 (0.6B) | `nemotron` | Local, streaming, 80–1120 ms latency |
| Mistral Voxtral Mini Transcribe 2 | `voxtral` | Cloud API (`MISTRAL_API_KEY`) or local weights |
| Whisper (faster-whisper) | `whisper`, `whisper-base`, `whisper-large-v3-turbo`, … | Local, offline; beam search, VAD filter, hotwords |
| NVIDIA Parakeet-TDT v3 (0.6B) | `parakeet` | **Default.** Local, offline, 25 European languages; live mode transcribes per utterance (VAD) |

## Install

```bash
pip install speck
```

Installs the `speck` command and the Python library. Python 3.12+. Configure through environment variables
(see [.env.example](.env.example)); the CLI loads a `.env` automatically, library users can call `speck.load_env_file()`.

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
| `--language CODE` | Locale like `en-US`, or `auto` (default) |
| `--lookahead 0\|3\|6\|13` | Latency preset: 80 / 320 / 560 (default) / 1120 ms. Lower is faster, higher is more accurate |
| `--speakers` | Label speakers (`mic`, `listen`, `mixed`) |
| `--enroll NAME=WAV` | Known voice sample, repeatable |
| `--speaker-threshold X` | Match strictness, default 0.35. Raise if people get merged, lower if one is split |
| `--num-speakers N` | Cap on speakers discovered |

```bash
speck transcribe samples/bcn_weather.mp3
speck mic --speakers --enroll Alice=alice.wav
speck listen --app discord --save-dir ~/transcripts     # --list shows playing apps
speck mixed --speakers --save call.txt                  # --me/--remote rename labels, --no-echo-guard with headphones
```

`listen` and `mixed` need PulseAudio/PipeWire (`pactl`, `parec`).

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

Add your own engine by subclassing `BaseASREngine` and calling `speck.factory.register_engine(alias, cls)`.

## Lowering the error rate

| Setting (`.env`) | Effect |
|---|---|
| `SPECK_LANGUAGE=en` | Locks Whisper to one language (no identification pass, no false switches); also the CLI `--language` default |
| `SPECK_HOTWORDS=Avvale,Kubernetes` | Biases Whisper decoding and snaps near-miss words in every engine's output to this vocabulary |
| `SPECK_BEAM_SIZE=5` | Whisper beam width |
| `SPECK_NORMALIZE_AUDIO=1` | DC removal and peak normalization of offline audio (`0` disables) |

`speck.rover([text_a, text_b, text_c])` merges transcripts from several engines by word-level majority vote.

WER is computed on normalized text: punctuation, casing, hesitation fillers ("uh", "um") and number format
("35" vs "thirty five") are ignored. Measure on a real set, not three clips:

```bash
python scripts/download_assets.py --librispeech test-clean          # ~1 h subset
python scripts/asr_benchmark.py --models nemotron whisper-large-v3-turbo parakeet --ensemble
```

## Tests

```bash
uv run pytest
```

## Metrics & Evaluation Pipelines

Speck provides standardized evaluation pipelines for both **Speaker Recognition** and **Text-to-Speech (TTS) / Speech Intelligibility**:

### 1. Speaker Recognition Pipeline
Measures **Equal Error Rate (EER)**, **NIST minDCF**, **AUC-ROC**, and **real-time streaming latency**:

```bash
# 1. Download models and benchmark audio WAV dataset
python scripts/download_assets.py --all

# 2. Run speaker verification & identification evaluation
python main.py evaluate --task speaker --data-dir data/ --output-dir reports/
```

### 2. Text-to-Speech (TTS) & Linguistic Pipeline (English & Italian)
Measures objective speech synthesis quality and round-trip intelligibility:
* **Acoustic Fidelity:** **Mel-Cepstral Distortion (MCD)** via Dynamic Time Warping (DTW) in dB.
* **Prosody Tracking:** **$F_0$ Pitch RMSE** (Hz and cents), **$F_0$ Pearson Correlation Coefficient (PCC)**, **Voicing Decision Error (VDE)**, and **Gross Pitch Error (GPE)**.
* **Speaker Identity (SECS):** Deep speaker embedding cosine similarity between reference and synthesized voices.
* **Linguistic Accuracy & Intelligibility:** Normalized **Word Error Rate (WER)**, **Character Error Rate (CER)**, **Match Error Rate (MER)**, and **Word Information Lost (WIL)** with specialized normalizers for:
  * **English (`en-US`):** Contraction expansion (`it's` -> `it is`, `don't` -> `do not`), punctuation normalization.
  * **Italian (`it-IT`):** Italian apostrophe elisions (`l'albero`, `un'altra`, `c'è`, `qual è`) and vowel accent preservation (`à`, `è`, `é`, `ì`, `ò`, `ù`).

```bash
# Run TTS evaluation suite across English and Italian
python main.py evaluate --task tts --output-dir reports/

# Or run the entire unified test suite via pytest
uv run pytest tests/test_tts_metrics.py -v
```

Generated evaluation artifacts are stored in `reports/`:
- `reports/EVALUATION_REPORT.md`: Speaker recognition executive report.
- `reports/metrics_summary.json`: Machine-readable speaker evaluation dump.
- `reports/TTS_EVALUATION_REPORT.md`: Multilingual TTS and linguistic evaluation report.
- `reports/tts_metrics_summary.json`: Machine-readable TTS evaluation dump.
- `reports/roc_curve.csv`: Coordinates for plotting verification ROC curves.

For detailed instructions on obtaining WAV files, converting your own voice recordings, and downloading model weights, see [docs/data_and_models.md](docs/data_and_models.md).

