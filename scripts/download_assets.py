#!/usr/bin/env python3
"""
Asset Downloader for Speck:
Downloads pre-trained speaker recognition ONNX models and benchmark speech datasets (WAVs).
"""

import os
import argparse
import urllib.request
import sys

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

def main():
    parser = argparse.ArgumentParser(description="Download models and audio datasets for Speck")
    parser.add_argument("--models", action="store_true", help="Download ONNX speaker recognition models")
    parser.add_argument("--data", action="store_true", help="Download benchmark speech dataset (WAV files)")
    parser.add_argument("--all", action="store_true", help="Download both models and datasets (default)")
    parser.add_argument("--models-dir", default="models", help="Directory for models (default: models)")
    parser.add_argument("--data-dir", default="data", help="Directory for audio dataset (default: data)")
    
    args = parser.parse_args()
    if not (args.models or args.data or args.all):
        args.all = True

    if args.all or args.models:
        download_model(output_dir=args.models_dir)
        
    if args.all or args.data:
        download_cmu_arctic_dataset(output_dir=args.data_dir)

if __name__ == "__main__":
    main()
