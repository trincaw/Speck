# Data & Models

`models/*.onnx`, `data/*.wav` and `reports/` are git-ignored. Restore them with:

```bash
python scripts/download_assets.py --all        # or --models / --data
python scripts/download_assets.py --librispeech test-clean   # ASR WER set (~1 h)
```

The ONNX models are fetched from HuggingFace into the HF cache on first use: TitaNet-small (speaker embeddings for live labels, `identify` / `diarize` and SECS), pyannote segmentation-3.0 (`diarize`) and Silero VAD; their repos and filenames are listed in `src/speck/hub.py`. `--models` pre-fetches them, e.g. before going offline. Model choice and benchmarks: [models_report.md](models_report.md).

`--data` fetches CMU Arctic (speakers `slt`, `bdl`, `clb`, utterances `arctic_a0001`-`a0005`) into `data/<speaker>/`. `--librispeech` downloads into `data/librispeech/<split>` and exits without doing `--models`/`--data`; `--chapter-step N` keeps 1 chapter in N (default 5).

## Audio format

16 kHz, mono WAV (`speck evaluate` does not resample; every other `speck` command resamples any file it loads). No minimum length is enforced; 1.5-3 s per utterance is recommended for enrollment. Convert with:

```bash
ffmpeg -i in.mp3 -ar 16000 -ac 1 -c:a pcm_s16le data/alice/utterance_01.wav
```

## Dataset layout (speaker evaluation)

```
data/
├── speaker_1/{utterance_01,utterance_02}.wav
└── speaker_2/{utterance_01,utterance_02}.wav
```

One folder per identity, at least 2 WAVs each (the first, in sorted order, is the enrollment, the rest are queries). `speck identify` and `speck diarize` enroll the first WAV of every `data/*/` folder by default (`--enroll-dir DIR` to change it, plus any `--enroll NAME=WAV`); profiles are not saved between runs.

## Public corpora

[CMU Arctic](http://festvox.org/cmu_arctic/) (default download), [LibriSpeech](https://www.openslr.org/12/), [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/).
