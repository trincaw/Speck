#!/usr/bin/env python3
"""
Asset Downloader for Speck:
Downloads pre-trained speaker recognition ONNX models and benchmark speech datasets (WAVs).
"""

import os
import argparse
import tarfile
import urllib.request
import sys
import zlib

def download_model(model_name="campplus", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    if model_name == "campplus":
        model_path = os.path.join(output_dir, "campplus_cn_en_common_200k.onnx")
        if os.path.exists(model_path):
            print(f"[OK] Model already exists at {model_path}")
            return model_path

        print("[...] Downloading CAM++ ONNX model from HuggingFace...")
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="welcomyou/campplus-3dspeaker-200k-onnx",
                filename="campplus_cn_en_common_200k.onnx",
                local_dir=output_dir
            )
            print(f"[OK] Model downloaded to {downloaded}")
            return downloaded
        except Exception as e:
            print(f"[!] Error downloading model: {e}")
            sys.exit(1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def download_cmu_arctic_dataset(output_dir="data", speakers=None, utterances_per_speaker=5):
    """
    Downloads phonetically balanced speech WAV files from CMU Arctic (Festvox).
    Default speakers:
      - slt (US female)
      - bdl (US male)
      - clb (US female)
    """
    if speakers is None:
        speakers = ["slt", "bdl", "clb"]

    os.makedirs(output_dir, exist_ok=True)
    print(f"[...] Downloading benchmark speech WAVs to '{output_dir}/'...")
    
    utterances = [f"arctic_a000{i}.wav" for i in range(1, utterances_per_speaker + 1)]
    total_downloaded = 0

    for spk in speakers:
        spk_dir = os.path.join(output_dir, spk)
        os.makedirs(spk_dir, exist_ok=True)
        for utt in utterances:
            dest = os.path.join(spk_dir, utt)
            if os.path.exists(dest):
                continue
            
            url = f"http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_{spk}_arctic/wav/{utt}"
            try:
                urllib.request.urlretrieve(url, dest)
                total_downloaded += 1
                print(f"  + Downloaded {spk}/{utt}")
            except Exception as e:
                print(f"  ! Warning: could not download {url}: {e}")

    print(f"[OK] Dataset ready in '{output_dir}'. ({total_downloaded} new files downloaded)")

def download_librispeech(output_dir="data/librispeech", split="test-clean", chapter_step=5):
    """
    Streams a LibriSpeech evaluation split from OpenSLR and keeps every `chapter_step`-th chapter
    (FLAC + transcripts), giving a speaker-diverse subset: step 5 on test-clean is ~1 hour of audio.
    """
    url = f"https://www.openslr.org/resources/12/{split}.tar.gz"
    dest = os.path.join(output_dir, split)
    if os.path.isdir(dest) and any(f.endswith(".trans.txt") for _, _, fs in os.walk(dest) for f in fs):
        print(f"[OK] LibriSpeech {split} already present in {dest}")
        return dest

    print(f"[...] Streaming {url} (keeping 1 chapter in {chapter_step})...")
    kept = 0
    with urllib.request.urlopen(url) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = member.name.split("/")  # LibriSpeech/<split>/<speaker>/<chapter>/<file>
            if len(parts) < 5 or zlib.crc32("/".join(parts[2:4]).encode()) % chapter_step:
                continue
            target = os.path.join(dest, *parts[2:])
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as out:
                out.write(tar.extractfile(member).read())
            kept += member.name.endswith(".flac")
    print(f"[OK] {kept} utterances written to {dest}")
    return dest

def main():
    parser = argparse.ArgumentParser(description="Download models and audio datasets for Speck")
    parser.add_argument("--models", action="store_true", help="Download ONNX speaker recognition models")
    parser.add_argument("--data", action="store_true", help="Download benchmark speech dataset (WAV files)")
    parser.add_argument("--all", action="store_true", help="Download both models and datasets (default)")
    parser.add_argument("--librispeech", choices=["test-clean", "test-other"], help="Download a LibriSpeech ASR evaluation subset")
    parser.add_argument("--chapter-step", type=int, default=5, help="Keep 1 LibriSpeech chapter in N (default: 5, ~1 h of test-clean)")
    parser.add_argument("--models-dir", default="models", help="Directory for models (default: models)")
    parser.add_argument("--data-dir", default="data", help="Directory for audio dataset (default: data)")
    
    args = parser.parse_args()
    if args.librispeech:
        download_librispeech(output_dir=os.path.join(args.data_dir, "librispeech"), split=args.librispeech, chapter_step=args.chapter_step)
        return
    if not (args.models or args.data or args.all):
        args.all = True

    if args.all or args.models:
        download_model(output_dir=args.models_dir)
        
    if args.all or args.data:
        download_cmu_arctic_dataset(output_dir=args.data_dir)

if __name__ == "__main__":
    main()
