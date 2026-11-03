#!/bin/bash
set -e
source ~/workspace/sparky/.venv/bin/activate

echo "=== fairseq (needs python3-dev, now installed) ==="
pip install fairseq --no-deps
pip install omegaconf hydra-core bitarray

echo "=== RVC deps ==="
pip install soundfile praat-parselmouth pyworld torchcrepe einops faiss-cpu
pip install rvc-python --no-deps

echo "=== web + file ==="
pip install ddgs beautifulsoup4 lxml requests

echo "=== camera + vision ==="
pip install opencv-python-headless pillow qwen-vl-utils

echo "=== accelerate safetensors ==="
pip install accelerate safetensors

echo "INSTALL2 COMPLETE"
