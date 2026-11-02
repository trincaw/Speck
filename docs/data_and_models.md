# Data & Models

`models/*.onnx`, `data/*.wav` and `reports/` are git-ignored. Restore them with:

```bash
python scripts/download_assets.py --all        # or --models / --data
python scripts/download_assets.py --librispeech test-clean   # ASR WER set (~1 h)
```

The pyannote segmentation and WeSpeaker embedding models are fetched from HuggingFace on first use. `--models` downloads `models/campplus_cn_en_common_200k.onnx`.

## Audio format

16 kHz, mono, 16-bit PCM WAV, at least 1 s (1.5-3 s recommended for enrollment). Convert with:

```bash
ffmpeg -i in.mp3 -ar 16000 -ac 1 -c:a pcm_s16le data/alice/utterance_01.wav
```

## Dataset layout (speaker evaluation)

```
data/
├── speaker_1/{utterance_01,utterance_02}.wav
└── speaker_2/{utterance_01,utterance_02}.wav
```

One folder per identity, at least 2 utterances each (1 enrollment, 1+ queries).

## Public corpora

[CMU Arctic](http://festvox.org/cmu_arctic/) (default download), [LibriSpeech](https://www.openslr.org/12/), [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/).
