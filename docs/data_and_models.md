# Models & Audio Dataset Setup Guide

This guide explains what assets are required for Speck's speaker recognition and metrics pipelines, how to obtain them, and how to format custom audio.

---

## 1. Overview of Required Assets

To run speaker identification, real-time streaming, and the metrics benchmark, two types of assets are required:

1. **Pre-trained ONNX Speaker Diarization & Embedding Models**:
   - Pyannote Segmentation: `csukuangfj/sherpa-onnx-pyannote-segmentation-3-0` (neural speaker turn boundary & overlap detection).
   - Voice Recognition Backbone: `csukuangfj/speaker-embedding-models/wespeaker_en_voxceleb_resnet34_LM.onnx` (the official pyannote.audio 3.x voice embedding model).
2. **Audio WAV Files** (`data/`):
   - Multi-speaker audio recordings formatted into speaker subdirectories.
   - Used for enrollment, streaming inference, and 1:1 verification / 1:N identification benchmarks.

> [!NOTE]
> Large binary files (`models/*.onnx`, `data/*.wav`, and `reports/`) are intentionally excluded from git via `.gitignore` to keep the repository lightweight. Use the automated script below to download or restore them.

---

## 2. Quick Automated Download

Speck includes an automated asset downloader that fetches the pre-trained ONNX model from HuggingFace and standard phonetically balanced speech samples:

```bash
# Download both model and benchmark WAV dataset
python scripts/download_assets.py --all

# Download ONLY the model weights
python scripts/download_assets.py --models

# Download ONLY the speech WAV files
python scripts/download_assets.py --data
```

---

## 3. Audio Specifications & Requirements

For optimal speaker recognition accuracy and streaming performance, input audio must adhere to:

| Parameter | Specification | Notes |
| :--- | :--- | :--- |
| **Format** | WAV (Microsoft RIFF PCM) | 16-bit signed integer |
| **Sample Rate** | **16,000 Hz (16 kHz)** | Standard speech recognition frequency |
| **Channels** | **1 (Mono)** | Stereo audio will be downmixed to mono |
| **Minimum Duration** | $\ge 1.0\text{ s}$ | $1.5\text{ s} - 3.0\text{ s}$ per utterance recommended for enrollment |

---

## 4. How to Structure the Dataset

For the evaluation pipeline (`python main.py evaluate`), organize audio into speaker subfolders inside `data/`:

```
Speck/
├── models/
│   └── campplus_cn_en_common_200k.onnx
└── data/
    ├── speaker_1/
    │   ├── utterance_01.wav
    │   ├── utterance_02.wav
    │   └── utterance_03.wav
    ├── speaker_2/
    │   ├── utterance_01.wav
    │   └── utterance_02.wav
    └── speaker_3/
        ├── utterance_01.wav
        └── utterance_02.wav
```

* Each speaker directory represents an enrolled identity.
* Each speaker requires at least **2 distinct utterances** (1 for enrollment/reference, 1+ for evaluation queries).

---

## 5. Using Your Own Voice Recordings

If you want to evaluate custom voices or enroll yourself:

### Converting Existing Files (MP3, M4A, FLAC, WAV) with FFmpeg

Use `ffmpeg` to resample and downmix any recording to 16kHz mono WAV:

```bash
ffmpeg -i input_recording.mp3 -ar 16000 -ac 1 -c:a pcm_s16le data/alice/utterance_01.wav
```

### Recording Live from Microphone (Linux)

You can record a 5-second voice sample using `sox` or `parecord`/`arecord`:

```bash
# Using arecord (ALSA)
arecord -d 5 -r 16000 -f S16_LE -c 1 data/alice/sample1.wav

# Using sox
rec -r 16000 -c 1 -b 16 data/alice/sample1.wav trim 0 5
```

---

## 6. Public Speech Datasets for Benchmarking

If you want to benchmark against larger open-source corpora:

1. **CMU Arctic (Festvox)**:
   - Source: [Festvox CMU Arctic Speech Database](http://festvox.org/cmu_arctic/)
   - Clean, phonetically balanced sentences read by US, Canadian, and Scottish speakers.
   - Built into `scripts/download_assets.py`.
2. **LibriSpeech (test-clean)**:
   - Source: [OpenSLR Resource 12](https://www.openslr.org/12/)
   - Read speech from public domain audiobooks with standard speaker partitions.
3. **VoxCeleb 1 & 2**:
   - Source: [VoxCeleb Large-Scale Speaker Identification](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
   - In-the-wild conversational speech extracted from YouTube videos under various acoustic conditions.
