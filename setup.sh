#!/bin/bash
set -e
source ~/workspace/sparky/.venv/bin/activate
pip install --upgrade pip wheel setuptools

echo "=== torch/torchaudio ==="
pip install torch torchaudio torchvision

echo "=== core audio/speech ==="
pip install sounddevice webrtcvad-wheels faster-whisper openai numpy scipy

echo "=== TTS ==="
pip install piper-tts piper-phonemize-cross

echo "=== fairseq deps (skip numpy constraint) ==="
pip install bitarray hydra-core transformers huggingface_hub accelerate safetensors

echo "=== fairseq ==="
pip install fairseq --no-deps
pip install omegaconf

echo "=== RVC deps ==="
pip install soundfile praat-parselmouth pyworld torchcrepe einops
pip install faiss-cpu
pip install rvc-python --no-deps

echo "=== web/file tools ==="
pip install ddgs beautifulsoup4 lxml requests
pip install opencv-python-headless pillow qwen-vl-utils

echo "=== other ==="
pip install paramiko

echo "INSTALL COMPLETE"
