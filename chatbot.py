#!/usr/bin/env python3
"""Voice-interactive AI chatbot using a local LLM."""

import ctypes
import re
import sys
import queue
from pathlib import Path
from typing import Optional

# Pre-load portaudio + jack from the venv before sounddevice does its import-time
# dlopen, so the shared libraries are in the process's already-loaded set.
# (These are extracted from .deb packages into .venv/lib/ without needing sudo.)
_venv_lib = Path(__file__).parent / ".venv" / "lib"
for _so in ("libjack.so.0", "libportaudio.so.2"):
    _p = _venv_lib / _so
    if _p.exists():
        ctypes.CDLL(str(_p), mode=ctypes.RTLD_GLOBAL)

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────
LLM_BASE_URL  = "http://192.168.0.78:8000/v1/"
LLM_MODEL     = "Qwen/Qwen3.6-35B-A3B-FP8"
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Give concise, conversational responses suitable for speech. "
    "Avoid markdown, bullet points, code blocks, and special characters."
)

WHISPER_SIZE   = "base"     # tiny | base | small | medium | large-v3
SAMPLE_RATE    = 16_000     # Hz — webrtcvad requires 8k/16k/32k/48k
CHANNELS       = 1
FRAME_MS       = 30         # webrtcvad frame length in ms (must be 10, 20, or 30)
FRAME_SAMPLES  = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples per frame
VAD_MODE       = 3          # 0–3; 3 = most aggressive background-noise rejection
SILENCE_FRAMES = 40         # ~1.2 s of continuous silence ends an utterance
MIN_SPEECH_FRM = 8          # discard clips shorter than ~240 ms of speech frames

# Piper TTS voice (downloaded on first run)
VOICE_DIR  = Path.home() / ".local" / "share" / "piper-voices"
VOICE_NAME = "en_US-lessac-medium"
VOICE_ONNX = VOICE_DIR / f"{VOICE_NAME}.onnx"
VOICE_JSON = VOICE_DIR / f"{VOICE_NAME}.onnx.json"
VOICE_HF   = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
    f"/en/en_US/lessac/medium/{VOICE_NAME}"
)


# ── Voice model ───────────────────────────────────────────────────────────────
def ensure_voice_model() -> None:
    """Download piper voice model files if not already present."""
    if VOICE_ONNX.exists() and VOICE_JSON.exists():
        return
    import urllib.request
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("onnx", "onnx.json"):
        dest = VOICE_DIR / f"{VOICE_NAME}.{suffix}"
        if not dest.exists():
            url = f"{VOICE_HF}.{suffix}"
            print(f"  Downloading {dest.name} …", flush=True)
            urllib.request.urlretrieve(url, dest)
    print("  Voice model ready.")


_voice = None

def get_voice():
    global _voice
    if _voice is None:
        from piper.voice import PiperVoice
        _voice = PiperVoice.load(str(VOICE_ONNX), config_path=str(VOICE_JSON))
    return _voice


# ── TTS ───────────────────────────────────────────────────────────────────────
def tts_to_array(text: str) -> tuple[np.ndarray, int]:
    """Synthesise text → (float32 PCM array in [-1,1], sample_rate)."""
    chunks = list(get_voice().synthesize(text))
    if not chunks:
        return np.array([], dtype=np.float32), 22050
    audio = np.concatenate([c.audio_float_array for c in chunks])
    return audio, chunks[0].sample_rate


def speak(text: str, out_dev: Optional[int]) -> None:
    text = text.strip()
    if not text:
        return
    audio, sr = tts_to_array(text)
    sd.play(audio, samplerate=sr, device=out_dev, blocking=True)


# ── VAD + recording ───────────────────────────────────────────────────────────
def record_speech(in_dev: Optional[int], vad: webrtcvad.Vad) -> Optional[np.ndarray]:
    """
    Block until an utterance is detected via VAD, then return it as a
    float32 array at SAMPLE_RATE.  Returns None if no meaningful speech found.
    """
    q: queue.Queue[bytes] = queue.Queue()

    def cb(indata, frames, t, status):
        if status:
            print(f"  [audio status] {status}", file=sys.stderr)
        q.put(bytes(indata))

    frame_buf: list[bytes] = []
    speech_count  = 0
    silence_count = 0
    in_speech     = False

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="int16",
        channels=CHANNELS,
        device=in_dev,
        callback=cb,
    ):
        print("\n[Listening…]", end="", flush=True)
        while True:
            frame = q.get()
            if len(frame) != FRAME_SAMPLES * 2:
                continue

            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if is_speech:
                if not in_speech:
                    print(" [speech detected]", end="", flush=True)
                    in_speech = True
                speech_count  += 1
                silence_count  = 0
                frame_buf.append(frame)
            else:
                if in_speech:
                    silence_count += 1
                    frame_buf.append(frame)
                    if silence_count >= SILENCE_FRAMES:
                        print(" [end]", flush=True)
                        break

    if speech_count < MIN_SPEECH_FRM:
        return None

    raw = b"".join(frame_buf)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# ── STT ───────────────────────────────────────────────────────────────────────
def transcribe(audio: np.ndarray, model: WhisperModel) -> str:
    segments, _ = model.transcribe(audio, language="en", beam_size=5)
    return " ".join(s.text.strip() for s in segments).strip()


# ── LLM ───────────────────────────────────────────────────────────────────────
def _strip_think(text: str) -> str:
    """Remove <think>…</think> blocks emitted by Qwen3 chain-of-thought."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def llm_reply(
    client: OpenAI,
    history: list[dict],
    user_text: str,
    out_dev: Optional[int],
) -> str:
    """
    Stream the LLM reply, speaking each sentence as soon as it arrives.
    Skips <think> blocks so internal reasoning is never spoken aloud.
    Returns the clean full response (appended to history).
    """
    history.append({"role": "user", "content": user_text})

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    full_raw  = ""
    speak_buf = ""
    in_think  = False

    for chunk in stream:
        delta = ""
        if chunk.choices:
            delta = chunk.choices[0].delta.content or ""
        full_raw += delta

        # Filter out <think>…</think> spans as they stream in
        remaining = delta
        while remaining:
            if in_think:
                end = remaining.find("</think>")
                if end == -1:
                    remaining = ""  # still inside think block
                else:
                    in_think  = False
                    remaining = remaining[end + 8:]
            else:
                start = remaining.find("<think>")
                if start == -1:
                    speak_buf += remaining
                    remaining  = ""
                else:
                    speak_buf += remaining[:start]
                    in_think   = True
                    remaining  = remaining[start + 7:]

        # Flush completed sentences to TTS immediately (low latency)
        while True:
            best = -1
            for punct in ".!?":
                i = speak_buf.find(punct)
                if i != -1 and (best == -1 or i < best):
                    best = i
            if best == -1:
                break
            sentence  = speak_buf[: best + 1].strip()
            speak_buf = speak_buf[best + 1 :]
            if sentence:
                print(f"  Assistant: {sentence}")
                speak(sentence, out_dev)

    # Speak any trailing fragment
    if speak_buf.strip():
        print(f"  Assistant: {speak_buf.strip()}")
        speak(speak_buf.strip(), out_dev)

    clean = _strip_think(full_raw)
    history.append({"role": "assistant", "content": clean})
    return clean


# ── Device helpers ────────────────────────────────────────────────────────────
def _find_device(keyword: str, kind: str) -> Optional[int]:
    ch_key = f"max_{kind}_channels"
    for i, dev in enumerate(sd.query_devices()):
        if dev[ch_key] > 0 and keyword.lower() in dev["name"].lower():
            return i
    return None


def _dev_name(idx: Optional[int], kind: str) -> str:
    if idx is not None:
        return sd.query_devices(idx)["name"]
    return f"system default ({kind})"


def _pipewire_device(kind: str) -> Optional[int]:
    """
    Return the 'pipewire' virtual device index for the given kind.
    On this system PipeWire routes input to Brio 101 and output to HDMI 0
    (the monitor), so using the pipewire device is correct for both.
    Falls back to the 'default' virtual device, then None.
    """
    for name in ("pipewire", "default"):
        idx = _find_device(name, kind)
        if idx is not None:
            return idx
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Voice AI Chatbot ===")
    print(f"LLM : {LLM_MODEL}")
    print(f"URL : {LLM_BASE_URL}\n")

    # Show available devices
    print("Audio devices:")
    for i, dev in enumerate(sd.query_devices()):
        tags = ("" if dev["max_input_channels"]  == 0 else " [in]") + \
               ("" if dev["max_output_channels"] == 0 else " [out]")
        print(f"  {i:2d}: {dev['name']}{tags}")
    print()

    # Use PipeWire virtual device: it routes to Brio 101 mic (input) and
    # HDMI 0 monitor speaker (output) per the system's WirePlumber defaults.
    in_dev  = _pipewire_device("input")
    out_dev = _pipewire_device("output")

    print(f"Input  → {_dev_name(in_dev,  'input')}")
    print(f"Output → {_dev_name(out_dev, 'output')}")

    # Load models
    print("\nLoading models…")
    ensure_voice_model()

    print(f"  Whisper ({WHISPER_SIZE})…", end=" ", flush=True)
    whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    print("ready")

    print("  Piper TTS…", end=" ", flush=True)
    get_voice()
    print("ready")

    vad    = webrtcvad.Vad(VAD_MODE)
    client = OpenAI(base_url=LLM_BASE_URL, api_key="none")
    history: list[dict] = []

    print("\nReady!  Speak to chat — say 'quit' or press Ctrl-C to exit.\n")
    speak("Hello! I'm ready. How can I help you?", out_dev)

    while True:
        try:
            audio = record_speech(in_dev, vad)
        except KeyboardInterrupt:
            print("\nGoodbye.")
            speak("Goodbye!", out_dev)
            break

        if audio is None:
            continue

        print("[Transcribing…]", end=" ", flush=True)
        text = transcribe(audio, whisper)
        if not text:
            continue
        print(f"You: {text}")

        if text.lower().strip() in {"quit", "exit", "goodbye", "bye", "stop"}:
            speak("Goodbye!", out_dev)
            break

        print("[Thinking…]")
        try:
            llm_reply(client, history, text, out_dev)
        except Exception as exc:
            print(f"[LLM error] {exc}")
            speak("Sorry, I had trouble reaching the language model.", out_dev)


if __name__ == "__main__":
    main()
