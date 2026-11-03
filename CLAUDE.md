# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate venv (always required first)
source .venv/bin/activate

# Run the chatbot
python chatbot.py

# Install/update dependencies
pip install -r requirements.txt
```

## Architecture

Single-file voice pipeline (`chatbot.py`):

```
Mic (Brio 101 USB webcam)
  └─ VAD (webrtcvad, mode 3) — 30 ms frames, 1.2 s silence threshold
       └─ STT (faster-whisper, base model, CPU int8)
            └─ LLM (OpenAI-compatible API → vLLM on localhost:8000, same DGX Spark)
                 └─ TTS (piper-tts, en_US-lessac-medium voice)
                      └─ Speaker (HDA NVidia HDMI output)
```

**Key design points:**
- LLM responses stream sentence-by-sentence: each sentence is synthesised and spoken as it arrives, minimising time-to-first-audio.
- `<think>…</think>` blocks emitted by Qwen3's chain-of-thought are filtered out of the TTS stream in real time (never spoken aloud) and stripped from conversation history.
- Device auto-detection: searches for "brio"/"usb" for input, "nvidia"/"hdmi" for output. Falls back to system defaults.
- Piper voice model (`en_US-lessac-medium.onnx` + `.onnx.json`) is downloaded on first run to `~/.local/share/piper-voices/`.
- Conversation history is kept in memory for the session; there is no persistence between runs.

## Hardware

- **Compute:** everything runs locally on an NVIDIA DGX Spark mounted on the robot, with no internet connection — the vLLM server, the chatbot, Whisper, Piper/RVC, and the offline Wikipedia ZIM.
- **Input:** Logitech Brio 101 webcam (USB Audio, card 2)
- **Output:** HDA NVidia HDMI, device 3 (monitor "27E2UA")
- **LLM:** `Qwen/Qwen3.6-35B-A3B-FP8` served via vLLM on the same DGX Spark at `http://localhost:8000/v1/`. `_find_llm_url()` tries localhost first; `http://192.168.0.78:8000/v1/` is only a development fallback for running the chatbot on another machine against the Spark's server over the LAN.

## System dependencies

Requires `libportaudio2` / `portaudio19-dev`:
```bash
sudo apt-get install -y portaudio19-dev libportaudio2
```
