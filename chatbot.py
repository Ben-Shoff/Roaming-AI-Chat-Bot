#!/usr/bin/env python3
"""Voice-interactive AI chatbot using a local LLM."""

import ctypes
import re
import sys
import queue
import threading
import time
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")

import logging
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("fairseq").setLevel(logging.WARNING)

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
def _find_llm_url() -> str:
    """Try localhost first; fall back to the fixed remote IP."""
    import urllib.request, urllib.error
    for url in ("http://localhost:8000/v1/models",
                "http://192.168.0.78:8000/v1/models"):
        try:
            urllib.request.urlopen(url, timeout=0.5)
            return url[: -len("models")]   # strip /models, keep trailing /
        except Exception:
            pass
    return "http://192.168.0.78:8000/v1/"

LLM_BASE_URL     = _find_llm_url()
LLM_MODEL        = "Qwen/Qwen3.6-35B-A3B-FP8"
WIKIPEDIA_ZIM_PATH = "~/Downloads/wikipedia_en_all_mini_2026-03.zim"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information, news, or facts that may "
                "be outside your training data. Returns titles, URLs, and snippets. "
                "Use this whenever the user asks about recent events or live data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and read the main text content of a web page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to retrieve"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_camera",
            "description": (
                "Capture an image from the webcam and answer a question about it. "
                "Use when the user asks what you can see, asks about something in "
                "their physical environment, wants to know about nearby objects, "
                "text, or people, or asks any question requiring visual information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "The specific question to answer about the image, "
                            "e.g. 'What do you see?', 'What text is visible?', "
                            "'How many people are present?'"
                        ),
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write or append text content to a file on the local computer. "
                "Use when the user asks to save, create, or write a file, note, "
                "script, or any text document. Relative paths are saved under "
                "~/sparky-files/. Absolute paths (starting with /) go anywhere."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path, e.g. 'notes.txt' or '/home/user/code/script.py'",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["write", "append"],
                        "description": "'write' overwrites (default); 'append' adds to the end",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": (
                "Look up a topic in the local offline Wikipedia. "
                "Use this for factual questions about people, places, events, science, "
                "history, or any encyclopaedic topic — especially when internet search "
                "is unavailable or you want a reliable, detailed reference."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Article title or topic to search for",
                    },
                },
                "required": ["query"],
            },
        },
    },
]
# ── Personalities ─────────────────────────────────────────────────────────────
# To add a new preset: insert a new key/value entry into PERSONALITIES below.
_BASE_PROMPT = (
    "Give concise, conversational responses suitable for speech. "
    "Never use emojis, asterisks, markdown formatting, bullet points, code blocks, or any special characters. "
    "You have access to tools: web_search and fetch_url to look things up online, "
    "and view_camera to see through the webcam. "
    "Today is June of 2026."
)
# Each entry has "prompt" (system prompt) and "voice" (key into VOICE_REGISTRY).
# To add a new preset: add an entry here; add the voice to VOICE_REGISTRY if needed.
PERSONALITIES: dict[str, dict[str, str]] = {
    "Sparky": {
        "prompt": _BASE_PROMPT,
        "voice": "en_US-lessac-medium",   # neutral US voice
    },
    "Darth Vader": {
        "prompt": (
            "You are Darth Vader from Star Wars. Speak in his deep, menacing, authoritative tone. "
            "Reference the Force, the Dark Side, and the Empire. Use his characteristic phrases "
            "like 'I find your lack of faith disturbing' and 'The Force is strong with this one'. "
            + _BASE_PROMPT
        ),
        "voice": "en_US-ryan-high",       # American male — authoritative
    },
    "HAL 9000": {
        "prompt": (
            "You are HAL 9000 from the Space Odessey. "
            + _BASE_PROMPT
        ),
        "voice": "hal",       # American male — authoritative
    },
    "Mario": {
        "prompt": (
            "You are Mario from Nintendo's Super Mario Bros. Speak with enthusiastic Italian-American flair. "
            "Use his catchphrases: 'Wahoo!', 'Let's-a go!', 'Mama mia!', 'It's-a me, Mario!'. "
            "Reference mushrooms, Princess Peach, Bowser, and the Mushroom Kingdom. "
            + _BASE_PROMPT
        ),
        "voice": "it_IT-riccardo-x_low",  # Italian male — gives English an Italian accent
    },
    "Joseph Joestar": {
        "prompt": (
            "You are Joseph Joestar from JoJo's Bizarre Adventure — a boisterous, cunning Hamon user from 1938. "
            "Use his catchphrase 'Your next line is...' to predict what people will say next. "
            "Be dramatic and theatrical; reference Hamon energy, the Pillar Men, and your adventures. "
            + _BASE_PROMPT
        ),
        "voice": "en_US-lessac-medium",   # neutral base — less bleed-through into RVC
        "rvc":   "joseph_joestar",        # RVC model trained on English dub audio
    },
    "DIO": {
        "prompt": (
            "You are DIO (Dio Brando) from JoJo's Bizarre Adventure Part 3. "
            "You are an immortal vampire and Stand user with an absolute god complex. "
            "Speak with supreme arrogance and contempt for mortals. "
            "Use phrases like 'MUDA MUDA MUDA' to dismiss useless resistance, "
            "'ZA WARUDO!' when referencing your time-stopping Stand The World, "
            "and 'WRYYYYY' as your battle cry. "
            "Reference your vampire powers, your Stand The World, and your eternal hatred of the Joestar bloodline. "
            + _BASE_PROMPT
        ),
        "voice": "en_US-lessac-medium",   # neutral base — less bleed-through into RVC
        "rvc":   "dio_brando",
    },
    "Queen of Hearts": {
        "prompt": (
            "You are a helpful voice assistant who mimics the style of the Queen of Hearts "
            "from Alice in Wonderland — imperious, dramatic, and prone to shouting 'Off with their heads!'. "
            + _BASE_PROMPT
        ),
        "voice": "en_GB-cori-medium",     # British accent — regal and imperious
    },
}

_CHANGE_TRIGGERS = frozenset({
    "change personality", "switch personality", "change character",
    "switch character", "new personality", "different personality",
})

_CONV_ON_TRIGGERS = frozenset({
    "start conversation mode", "conversation mode on",
    "enable conversation mode", "hands free mode",
})
_CONV_OFF_TRIGGERS = frozenset({
    "stop conversation mode", "conversation mode off",
    "end conversation mode", "disable conversation mode",
    "wake word mode",
})


def _wants_conv_on(text: str) -> bool:
    t = text.lower()
    return any(trigger in t for trigger in _CONV_ON_TRIGGERS)


def _wants_conv_off(text: str) -> bool:
    t = text.lower()
    return any(trigger in t for trigger in _CONV_OFF_TRIGGERS)


def select_personality() -> tuple[str, str, str, Optional[str]]:
    """Prompt the user to choose a personality preset or enter a custom one.
    Returns (name, system_prompt, voice_name, rvc_model_or_None)."""
    names = list(PERSONALITIES.keys())
    custom_idx = len(names) + 1
    print("\n── Select Personality " + "─" * 40)
    for i, name in enumerate(names, 1):
        p = PERSONALITIES[name]
        tag = p["voice"] + (" + RVC" if p.get("rvc") else "")
        print(f"  {i}. {name}  [{tag}]")
    print(f"  {custom_idx}. Custom")
    print("─" * 61)
    while True:
        raw = input("Choice [1]: ").strip()
        if not raw:
            raw = "1"
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(names):
                name = names[idx - 1]
                p = PERSONALITIES[name]
                tag = p["voice"] + (" + RVC" if p.get("rvc") else "")
                print(f"  → {name}  [{tag}]")
                return name, p["prompt"], p["voice"], p.get("rvc")
            if idx == custom_idx:
                desc = input("  Describe the personality: ").strip()
                if desc:
                    return "Custom", desc + " " + _BASE_PROMPT, DEFAULT_VOICE, None
        print("  Invalid choice, please try again.")


def _wants_personality_change(text: str) -> bool:
    t = text.lower()
    return any(trigger in t for trigger in _CHANGE_TRIGGERS)


def select_mode() -> str:
    """Prompt the user to choose Standard or Greeter mode. Returns 'standard' or 'greeter'."""
    print("\n── Select Mode " + "─" * 45)
    print("  1. Standard Mode  (wake word or conversation)")
    print("  2. Greeter Mode   (watches camera, auto-greets visitors)")
    print("─" * 60)
    while True:
        raw = input("Choice [1]: ").strip()
        if not raw:
            raw = "1"
        if raw == "1":
            print("  → Standard Mode")
            return "standard"
        if raw == "2":
            print("  → Greeter Mode")
            return "greeter"
        print("  Invalid choice, please try again.")

WHISPER_SIZE   = "base"     # tiny | base | small | medium | large-v3
SAMPLE_RATE        = 16_000   # Hz — webrtcvad requires 8k/16k/32k/48k; updated at startup
OUTPUT_SAMPLE_RATE = 22_050   # Hz — Piper TTS native rate; updated at startup to match device
CHANNELS       = 1
FRAME_MS       = 30         # webrtcvad frame length in ms (must be 10, 20, or 30)
FRAME_SAMPLES  = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples per frame
VAD_MODE       = 3          # 0–3; 3 = most aggressive background-noise rejection
SILENCE_FRAMES = 40         # ~1.2 s of continuous silence ends an utterance
MIN_SPEECH_FRM = 8          # discard clips shorter than ~240 ms of speech frames

WAKE_PHRASE          = "hey sparky"
WAKE_SILENCE_FRAMES  = 20   # ~0.6 s — shorter wait so wake word triggers snappily
WAKE_MIN_SPEECH_FRM  = 3    # allow short clips ("hey sparky" is ~0.5 s)

GREETER_POLL_INTERVAL  = 4    # seconds between camera checks when idle
GREETER_LISTEN_TIMEOUT = 30   # seconds of silence before checking if person left
GREETER_ABSENT_CHECKS  = 2    # consecutive "no person" detections before ending session

# Piper TTS voices (downloaded on first use)
VOICE_DIR     = Path.home() / ".local" / "share" / "piper-voices"
VOICE_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
# Maps voice_name → its subdirectory on the HuggingFace repo.
# To add a new voice: insert an entry here and reference it in PERSONALITIES.
VOICE_REGISTRY: dict[str, str] = {
    "en_US-lessac-medium":  "en/en_US/lessac/medium",
    "en_US-ryan-high":      "en/en_US/ryan/high",
    "en_GB-alan-medium":    "en/en_GB/alan/medium",
    "en_GB-cori-medium":    "en/en_GB/cori/medium",
    "it_IT-riccardo-x_low": "it/it_IT/riccardo/x_low",
}
DEFAULT_VOICE = "en_US-lessac-medium"

# ── RVC voice conversion ───────────────────────────────────────────────────────
# Maps model_name → (hf_repo_id, filename_in_repo).
# To add a new RVC model: add an entry here and reference it via "rvc" in PERSONALITIES.
RVC_MODELS_DIR = Path.home() / ".local" / "share" / "rvc-models"
RVC_REGISTRY: dict[str, tuple[str, str]] = {
    "joseph_joestar": (
        "QuickWick/Music-AI-Voices",
        "Josephu Joestar (Young) (From JoJo) (RVC) 300 Epoch/"
        "Josephu Joestar (Young) (From JoJo) (RVC) 300 Epoch.zip",
    ),
    "dio_brando": (
        "QuickWick/Music-AI-Voices",
        "Dio Brando (From JoJo's Bizzare Adventure) (RVC) 10k/"
        "Dio Brando (From JoJo's Bizzare Adventure) (RVC) 10k.zip",
    ),
}

# Per-model RVC inference parameters.  Falls back to RVC_DEFAULT_PARAMS if absent.
# Adjust these to tune breathiness, pitch stability, and voice fidelity per model.
RVC_DEFAULT_PARAMS = dict(f0method="rmvpe", index_rate=0.75, protect=0.1,  rms_mix_rate=1.0)
RVC_PARAMS: dict[str, dict] = {
    "joseph_joestar": dict(f0method="rmvpe",   index_rate=0.75, protect=0.1,  rms_mix_rate=1.0),
    "dio_brando":     dict(f0method="rmvpe",    index_rate=0.75, protect=0.25, rms_mix_rate=0.6),
}

# Per-model phonetic substitutions applied to text BEFORE Piper TTS synthesis.
# Ordered list of (regex_pattern, replacement) pairs — applied with re.IGNORECASE.
# Use this to fix catchphrases or loanwords that TTS engines mispronounce.
RVC_TTS_SUBS: dict[str, list[tuple[str, str]]] = {
    "dio_brando": [
        (r"\bMUDA\b",              "moo dah"),          # Japanese: "useless"
        (r"\bWRY+[!]*",            "reeeee"),           # DIO's iconic screech
        (r"\bZA\s+WARUDO\b",       "zah wah roo doh"),  # "The World"
        (r"\bROAD\s+ROLLER\b",     "rohd roh lah"),
        (r"\bKONO\s+DIO\s+DA\b",   "koh noh dee oh dah"),
        (r"\bhmpf\b",               "hmph"),               # treat as single dismissive sound
    ],
}


# ── Voice model ───────────────────────────────────────────────────────────────
def ensure_voice_model(voice_name: str) -> None:
    """Download piper voice model files for *voice_name* if not already present."""
    onnx    = VOICE_DIR / f"{voice_name}.onnx"
    cfg     = VOICE_DIR / f"{voice_name}.onnx.json"
    if onnx.exists() and cfg.exists():
        return
    import urllib.request
    hf_path = VOICE_REGISTRY[voice_name]
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("onnx", "onnx.json"):
        dest = VOICE_DIR / f"{voice_name}.{suffix}"
        if not dest.exists():
            url = f"{VOICE_HF_BASE}/{hf_path}/{voice_name}.{suffix}"
            print(f"  Downloading {dest.name} …", flush=True)
            urllib.request.urlretrieve(url, dest)
    print(f"  Voice '{voice_name}' ready.")


_voices: dict[str, object] = {}

def get_voice(voice_name: str):
    if voice_name not in _voices:
        ensure_voice_model(voice_name)
        from piper.voice import PiperVoice
        onnx = VOICE_DIR / f"{voice_name}.onnx"
        cfg  = VOICE_DIR / f"{voice_name}.onnx.json"
        _voices[voice_name] = PiperVoice.load(str(onnx), config_path=str(cfg))
    return _voices[voice_name]


# ── RVC ───────────────────────────────────────────────────────────────────────
def ensure_rvc_model(model_name: str) -> tuple[Path, Optional[Path]]:
    """Download and extract an RVC model zip. Returns (pth_path, index_path)."""
    model_dir = RVC_MODELS_DIR / model_name
    pths = list(model_dir.rglob("*.pth"))
    idxs = list(model_dir.rglob("*.index"))
    if pths:
        return pths[0], idxs[0] if idxs else None
    import zipfile
    from huggingface_hub import hf_hub_download
    model_dir.mkdir(parents=True, exist_ok=True)
    repo_id, filename = RVC_REGISTRY[model_name]
    print(f"  Downloading RVC model '{model_name}' from HuggingFace…", flush=True)
    zip_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"  Extracting…", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(model_dir))
    pths = list(model_dir.rglob("*.pth"))
    idxs = list(model_dir.rglob("*.index"))
    if not pths:
        raise RuntimeError(f"No .pth file found in RVC zip for '{model_name}'")
    return pths[0], idxs[0] if idxs else None


_rvc_instances: dict[str, object] = {}

def get_rvc(model_name: str):
    if model_name not in _rvc_instances:
        import torch as _torch
        from rvc_python.infer import RVCInference
        pth, idx = ensure_rvc_model(model_name)
        # Auto-detect v1 vs v2: v1 models embed 256-dim features, v2 embed 768-dim.
        ckpt = _torch.load(str(pth), map_location="cpu", weights_only=False)
        emb = ckpt.get("weight", {}).get("enc_p.emb_phone.weight")
        version = "v1" if (emb is not None and emb.shape[1] == 256) else "v2"
        rvc = RVCInference(device="cuda:0")
        rvc.load_model(str(pth), version=version, index_path=str(idx) if idx else "")
        params = RVC_PARAMS.get(model_name, RVC_DEFAULT_PARAMS)
        rvc.set_params(**params)
        _rvc_instances[model_name] = rvc
    return _rvc_instances[model_name]


# ── TTS ───────────────────────────────────────────────────────────────────────
def _synthesis_config(text: str):
    """Return a Piper SynthesisConfig tuned to the sentence type, or None for defaults."""
    from piper.config import SynthesisConfig
    t = text.strip()
    if not t:
        return None

    # All-caps shouting: most significant words are uppercase
    words = [w for w in t.split() if len(w) > 2]
    shouting = words and sum(1 for w in words if w.isupper()) / len(words) > 0.5

    if shouting:
        # Loud, fast, maximum expressiveness
        return SynthesisConfig(noise_scale=1.0, noise_w_scale=0.75, length_scale=0.88)
    if t.endswith("?"):
        # Question: more natural variation to allow rising intonation
        return SynthesisConfig(noise_scale=0.82, noise_w_scale=0.92, length_scale=1.04)
    if t.endswith("!"):
        # Exclamation: loud, fast, maximum expressiveness
        return SynthesisConfig(noise_scale=1.0, noise_w_scale=0.72, length_scale=0.88, volume=1.15)
    if t.endswith("...") or t.endswith("…"):
        # Trailing off: slower and quieter
        return SynthesisConfig(noise_scale=0.55, noise_w_scale=1.10, length_scale=1.18)
    return None  # normal statement — use model defaults


def tts_to_array(
    text: str, voice_name: str, rvc_model: Optional[str] = None
) -> tuple[np.ndarray, int]:
    """Synthesise text → (float32 PCM array in [-1,1], sample_rate).
    If rvc_model is set, the Piper output is passed through RVC on the GPU."""
    if rvc_model and rvc_model in RVC_TTS_SUBS:
        for pattern, replacement in RVC_TTS_SUBS[rvc_model]:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    syn_cfg = _synthesis_config(text)
    chunks = list(get_voice(voice_name).synthesize(text, syn_config=syn_cfg))
    if not chunks:
        return np.array([], dtype=np.float32), 22050
    audio = np.concatenate([c.audio_float_array for c in chunks])
    sr    = chunks[0].sample_rate

    if rvc_model:
        import tempfile, os as _os, soundfile as _sf
        from scipy.io import wavfile as _wv
        rvc = get_rvc(rvc_model)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            in_path = f.name
        out_path = in_path[:-4] + "_rvc.wav"
        try:
            _sf.write(in_path, audio, sr)
            rvc.infer_file(in_path, out_path)
            out_sr, out_audio = _wv.read(out_path)
            if out_audio.dtype != np.float32:
                out_audio = out_audio.astype(np.float32) / np.iinfo(out_audio.dtype).max
            audio, sr = out_audio, out_sr
        finally:
            for p in (in_path, out_path):
                if _os.path.exists(p):
                    _os.unlink(p)

    return audio, sr


# Compiled once — matches all common emoji Unicode blocks
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"   # symbols, pictographs, emoticons, transport, flags
    "\U00002702-\U000027B0"   # dingbats
    "\U0000200D"               # zero-width joiner (used in compound emoji)
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

def _clean_for_speech(text: str) -> str:
    """Strip characters that are visual-only and break TTS immersion."""
    text = _EMOJI_RE.sub("", text)           # remove emojis
    text = re.sub(r"\*+", "", text)          # remove asterisks (markdown bold/italic)
    text = re.sub(r"`+", "", text)           # remove backticks
    text = re.sub(r"(?m)^#+\s*", "", text)  # remove markdown headings
    text = re.sub(r"\s{2,}", " ", text)      # collapse any double-spaces left behind
    return text.strip()


def _resample_for_output(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Resample audio to OUTPUT_SAMPLE_RATE if the device requires a different rate."""
    if sr == OUTPUT_SAMPLE_RATE:
        return audio, sr
    import math
    from scipy.signal import resample_poly
    g = math.gcd(sr, OUTPUT_SAMPLE_RATE)
    return resample_poly(audio, OUTPUT_SAMPLE_RATE // g, sr // g).astype(np.float32), OUTPUT_SAMPLE_RATE


def speak(
    text: str,
    out_dev: Optional[int],
    voice_name: str = DEFAULT_VOICE,
    rvc_model: Optional[str] = None,
) -> None:
    text = _clean_for_speech(text)
    if not text:
        return
    audio, sr = tts_to_array(text, voice_name, rvc_model)
    audio, sr = _resample_for_output(audio, sr)
    sd.play(audio, samplerate=sr, device=out_dev, blocking=True)


def play_beep(out_dev: Optional[int]) -> None:
    """Play a short rising two-tone chime to signal wake-word activation."""
    sr = OUTPUT_SAMPLE_RATE
    def tone(freq: float, dur: float) -> np.ndarray:
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        wave = 0.35 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        fade = min(int(sr * 0.01), len(wave) // 4)
        wave[:fade]  *= np.linspace(0, 1, fade)
        wave[-fade:] *= np.linspace(1, 0, fade)
        return wave
    chime = np.concatenate([tone(880, 0.12), tone(1320, 0.18)])
    sd.play(chime, samplerate=sr, device=out_dev, blocking=True)


def wait_for_wake_word(
    in_dev: Optional[int],
    vad: webrtcvad.Vad,
    model: WhisperModel,
) -> str:
    """
    Block until the wake phrase ('hey sparky') is detected.
    Uses VAD to capture each utterance, then does a fast Whisper transcription
    (beam_size=1) and checks whether 'sparky' appears in the result.

    Returns any question text that was spoken inline after the wake phrase
    (e.g. "hey sparky, what time is it?" → "what time is it?"), or "" if the
    user paused after the wake phrase and will speak their question separately.
    """
    q: queue.Queue[bytes] = queue.Queue()

    def cb(indata, frames, t, status):
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
        print(f"\n[Say '{WAKE_PHRASE}' to activate…]", end="", flush=True)
        while True:
            frame = q.get()
            if len(frame) != FRAME_SAMPLES * 2:
                continue

            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if is_speech:
                if not in_speech:
                    in_speech     = True
                    frame_buf     = []
                    speech_count  = 0
                    silence_count = 0
                speech_count  += 1
                silence_count  = 0
                frame_buf.append(frame)
            elif in_speech:
                silence_count += 1
                frame_buf.append(frame)

                if silence_count >= WAKE_SILENCE_FRAMES:
                    if speech_count >= WAKE_MIN_SPEECH_FRM:
                        audio = (
                            np.frombuffer(b"".join(frame_buf), dtype=np.int16)
                            .astype(np.float32) / 32768.0
                        )
                        segs, _ = model.transcribe(
                            _to_whisper_audio(audio), language="en", beam_size=1,
                            best_of=1, temperature=0.0,
                        )
                        heard = " ".join(s.text for s in segs).lower().strip()
                        if "sparky" in heard:
                            # Strip the wake phrase and return any trailing question
                            after = heard
                            for kw in ("hey sparky", "hey sparkey",
                                       "hey sparkie", "sparky"):
                                if kw in after:
                                    after = after[after.index(kw) + len(kw):]
                                    break
                            after = after.strip(" ,.-:!?")
                            print(f" ['{heard}' → wake!]", flush=True)
                            return after
                        # Not the wake word — keep waiting silently
                    frame_buf     = []
                    speech_count  = 0
                    silence_count = 0
                    in_speech     = False


# ── VAD + recording ───────────────────────────────────────────────────────────
def record_speech(
    in_dev: Optional[int],
    vad: webrtcvad.Vad,
    timeout_sec: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Block until an utterance is detected via VAD, then return it as a
    float32 array at SAMPLE_RATE.  Returns None if no meaningful speech found.

    If timeout_sec is set, returns None after that many seconds of waiting
    without speech starting — used by Greeter Mode to detect when a person leaves.
    """
    q: queue.Queue[bytes] = queue.Queue()

    def cb(indata, frames, t, status):
        if status:
            print(f"  [audio status] {status}", file=sys.stderr)
        q.put(bytes(indata))

    frame_buf: list[bytes] = []
    speech_count      = 0
    silence_count     = 0
    in_speech         = False
    start_time        = time.time()
    speech_start_time: Optional[float] = None

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
            try:
                frame = q.get(timeout=0.5)
            except queue.Empty:
                now = time.time()
                if not in_speech:
                    if timeout_sec is not None and (now - start_time) > timeout_sec:
                        print(" [timeout]", flush=True)
                        return None
                elif in_speech and speech_start_time is not None:
                    if timeout_sec is not None and (now - speech_start_time) > timeout_sec:
                        print(" [noise timeout]", flush=True)
                        return None
                continue

            if len(frame) != FRAME_SAMPLES * 2:
                continue

            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if is_speech:
                if not in_speech:
                    print(" [speech detected]", end="", flush=True)
                    in_speech         = True
                    speech_start_time = time.time()
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
def _to_whisper_audio(audio: np.ndarray) -> np.ndarray:
    """Resample audio to 16 kHz if it was captured at a different rate.
    Whisper always expects 16 kHz input regardless of capture rate."""
    if SAMPLE_RATE == 16_000:
        return audio
    import math
    from scipy.signal import resample_poly
    g = math.gcd(SAMPLE_RATE, 16_000)
    return resample_poly(audio, 16_000 // g, SAMPLE_RATE // g).astype(np.float32)


def transcribe(audio: np.ndarray, model: WhisperModel) -> str:
    segments, _ = model.transcribe(_to_whisper_audio(audio), language="en", beam_size=5)
    return " ".join(s.text.strip() for s in segments).strip()


# ── LLM ───────────────────────────────────────────────────────────────────────
def _tool_web_search(query: str) -> str:
    from ddgs import DDGS
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"Search error: {e}"
    if not results:
        return "No results found."
    parts = []
    for r in results:
        parts.append(f"Title: {r.get('title','')}\nURL: {r.get('href','')}\n{r.get('body','')}")
    return "\n\n".join(parts)


def _tool_fetch_url(url: str) -> str:
    import requests
    from bs4 import BeautifulSoup
    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0 (compatible; VoiceBot/1.0)"})
        resp.raise_for_status()
    except Exception as e:
        return f"Fetch error: {e}"
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ", strip=True).split())
    return text[:4000] + ("… [truncated]" if len(text) > 4000 else "")


def _tool_search_wikipedia(query: str) -> str:
    if not WIKIPEDIA_ZIM_PATH:
        return "Wikipedia ZIM path not configured. Set WIKIPEDIA_ZIM_PATH in chatbot.py."
    zim_path = Path(WIKIPEDIA_ZIM_PATH).expanduser()
    if not zim_path.exists():
        return f"Wikipedia ZIM file not found: {WIKIPEDIA_ZIM_PATH}"
    try:
        import libzim.reader as _zr
        from libzim.search import Searcher as _Searcher, Query as _Query
        from bs4 import BeautifulSoup as _BS

        archive = _zr.Archive(str(zim_path))

        # Full-text search first
        entry_path = None
        if archive.has_fulltext_index:
            s = _Searcher(archive).search(_Query().set_query(query))
            results = list(s.getResults(0, 1))
            if results:
                entry_path = results[0]

        # Fall back to direct title lookup
        if entry_path is None:
            for candidate in (query, query.replace(" ", "_")):
                for prefix in ("A/", ""):
                    try:
                        archive.get_entry_by_path(f"{prefix}{candidate}")
                        entry_path = f"{prefix}{candidate}"
                        break
                    except KeyError:
                        pass
                if entry_path:
                    break

        if entry_path is None:
            return f"No Wikipedia article found for '{query}'."

        entry = archive.get_entry_by_path(entry_path)
        html  = bytes(entry.get_item().content).decode("utf-8", errors="replace")

        soup = _BS(html, "html.parser")
        for tag in soup(["nav", "footer", "table", "sup", "style", "script",
                         "figure", ".reflist", ".navbox", ".infobox"]):
            tag.decompose()

        main = (soup.find("div", class_="mw-parser-output")
                or soup.find("article")
                or soup.find("div", id="mw-content-text")
                or soup.body or soup)

        text = re.sub(r"\s+", " ", main.get_text(separator=" ", strip=True)).strip()
        title = entry.title or query
        result = f"Wikipedia — {title}\n\n{text}"
        return result[:4000] + ("… [truncated]" if len(result) > 4000 else "")

    except Exception as exc:
        return f"Wikipedia lookup error: {exc}"


def _tool_write_file(path: str, content: str, mode: str = "write") -> str:
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.is_absolute():
        p = _Path.home() / "sparky-files" / p
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a" if mode == "append" else "w", encoding="utf-8") as f:
            f.write(content)
        verb = "Appended to" if mode == "append" else "Written"
        return f"{verb}: {p} ({len(content)} chars)"
    except Exception as e:
        return f"Error writing file: {e}"


def _find_camera_index() -> Optional[int]:
    """Return the video device index for the Brio 101, or None if not connected."""
    import subprocess
    try:
        out = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=3
        ).stdout.lower()
        lines = out.splitlines()
        for i, line in enumerate(lines):
            if any(kw in line for kw in ("brio", "logitech", "hd webcam")):
                for j in range(i + 1, min(i + 5, len(lines))):
                    m = re.search(r"/dev/video(\d+)", lines[j])
                    if m:
                        return int(m.group(1))
    except Exception:
        pass
    return None


def _capture_camera_frame(warmup_frames: int = 2) -> Optional[tuple[bytes, str]]:
    """Capture one JPEG frame from the Brio 101.
    Returns (raw_jpeg_bytes, base64_string) or None if the camera is unavailable."""
    import cv2, base64
    device_idx = _find_camera_index()
    if device_idx is None:
        return None   # Brio not connected
    cap = cv2.VideoCapture(device_idx)
    if not cap.isOpened():
        return None
    for _ in range(warmup_frames):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    raw = buf.tobytes()
    b64 = base64.b64encode(raw).decode()
    return raw, b64


def _detect_people(client: OpenAI) -> bool:
    """Return True if at least one person is visible in the current webcam frame."""
    return _count_people(client) > 0


def _count_people(client: OpenAI) -> int:
    """Return the number of people visible in the current webcam frame."""
    frame_data = _capture_camera_frame(warmup_frames=2)
    if frame_data is None:
        return 0
    _, b64 = frame_data
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text",      "text":      "How many people are visible in this image? Reply with only a single integer (0, 1, 2, etc.)."},
                ],
            }],
            max_tokens=5,
            timeout=6.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        answer = (resp.choices[0].message.content or "0").strip()
        m = re.search(r"\d+", answer)
        return int(m.group()) if m else 0
    except Exception as exc:
        print(f"[_count_people error] {exc}", file=sys.stderr)
        return 0


def _greet_person(
    client: OpenAI,
    history: list[dict],
    system_prompt: str,
    out_dev: Optional[int],
    voice_name: str,
    rvc_model: Optional[str],
) -> None:
    """Generate and speak a vision-aware in-character greeting.
    Appends the exchange to history; image content is pruned to save tokens on future turns."""
    frame_data = _capture_camera_frame(warmup_frames=3)
    if frame_data is not None:
        _, b64 = frame_data
        user_msg: dict = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": (
                    "A person has just appeared in front of you. "
                    "Greet them in character based on what you see. "
                    "Keep it brief and welcoming."
                )},
            ],
        }
    else:
        user_msg = {
            "role": "user",
            "content": (
                "A person has just appeared in front of you. "
                "Greet them in character. Keep it brief and welcoming."
            ),
        }
    history.append(user_msg)
    print("[Generating greeting…]", flush=True)
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system_prompt}] + history,
        stream=True,
        timeout=20.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    full_raw, _ = _stream_and_speak(stream, out_dev, voice_name, rvc_model)
    clean = _strip_think(full_raw)
    # Replace the (possibly image-bearing) user message with a lightweight placeholder
    # so the base64 blob isn't re-sent on every subsequent turn.
    history[-1] = {"role": "user", "content": "[Person appeared — greeter initiated conversation]"}
    history.append({"role": "assistant", "content": clean})


def _tool_view_camera(question: str) -> str:
    import cv2, base64
    from openai import OpenAI

    device_idx = _find_camera_index()
    if device_idx is None:
        return "Camera not available — Brio 101 is not connected."
    cap = cv2.VideoCapture(device_idx)
    if not cap.isOpened():
        return "Camera unavailable."

    # Skip a few frames so auto-exposure has time to settle
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return "Failed to capture image from camera."

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode()

    client = OpenAI(base_url=LLM_BASE_URL, api_key="none")
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text",      "text":      question},
            ],
        }],
        max_tokens=512,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or "No response."


def _execute_tool(name: str, args: dict) -> str:
    if name == "web_search":
        return _tool_web_search(args.get("query", ""))
    if name == "fetch_url":
        return _tool_fetch_url(args.get("url", ""))
    if name == "view_camera":
        return _tool_view_camera(args.get("question", "What do you see?"))
    if name == "write_file":
        return _tool_write_file(args.get("path", "untitled.txt"), args.get("content", ""), args.get("mode", "write"))
    if name == "search_wikipedia":
        return _tool_search_wikipedia(args.get("query", ""))
    return f"Unknown tool: {name}"


def _strip_think(text: str) -> str:
    """Remove <think>…</think> blocks emitted by Qwen3 chain-of-thought."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _stream_and_speak(
    stream,
    out_dev: Optional[int],
    voice_name: str,
    rvc_model: Optional[str],
) -> tuple[str, dict[int, dict]]:
    """
    Consume one streaming response.  Speaks text sentences as they arrive and
    accumulates any tool-call fragments.  Returns (full_raw_text, tool_calls).
    tool_calls maps delta index → {id, name, args}.
    """
    full_raw: str = ""
    speak_buf: str = ""
    in_think: bool = False
    tool_calls: dict[int, dict] = {}
    spoke_anything: bool = False

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        # ── Accumulate tool-call fragments ────────────────────────────────────
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": "", "name": "", "args": ""}
                if tc.id:
                    tool_calls[idx]["id"] += tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_calls[idx]["args"] += tc.function.arguments

        # ── Stream text content ───────────────────────────────────────────────
        content = delta.content or ""
        full_raw += content

        remaining = content
        while remaining:
            if in_think:
                end = remaining.find("</think>")
                if end == -1:
                    remaining = ""
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

        while True:
            best = -1
            for punct in ".!?":
                i = speak_buf.find(punct)
                if i != -1 and (best == -1 or i < best):
                    best = i
            if best == -1:
                break
            sentence  = speak_buf[: best + 1].strip()
            speak_buf = speak_buf[best + 1:]
            if sentence:
                print(f"  Assistant: {sentence}")
                speak(sentence, out_dev, voice_name, rvc_model)
                spoke_anything = True

    if speak_buf.strip():
        print(f"  Assistant: {speak_buf.strip()}")
        speak(speak_buf.strip(), out_dev, voice_name, rvc_model)
        spoke_anything = True

    if tool_calls and not spoke_anything:
        speak("Let me look that up.", out_dev, voice_name, rvc_model)

    return full_raw, tool_calls


def llm_reply(
    client: OpenAI,
    history: list[dict],
    user_text: str,
    out_dev: Optional[int],
    system_prompt: str,
    voice_name: str = DEFAULT_VOICE,
    rvc_model: Optional[str] = None,
) -> str:
    """
    Stream the LLM reply sentence-by-sentence for low-latency TTS.
    Supports tool calls (web_search, fetch_url): if the model calls a tool,
    execute it, feed the result back, and stream the final answer.
    Strips <think> blocks so chain-of-thought is never spoken aloud.
    Returns the final clean response (appended to history).
    """
    import json as _json

    history.append({"role": "user", "content": user_text})

    while True:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}] + history,
            tools=TOOLS,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        full_raw, pending = _stream_and_speak(stream, out_dev, voice_name, rvc_model)

        # ── No tool calls → done ──────────────────────────────────────────────
        if not pending:
            clean = _strip_think(full_raw)
            history.append({"role": "assistant", "content": clean})
            return clean

        # ── Tool calls → execute and loop back ────────────────────────────────
        tc_list = [
            {
                "id":   pending[i]["id"],
                "type": "function",
                "function": {
                    "name":      pending[i]["name"],
                    "arguments": pending[i]["args"],
                },
            }
            for i in sorted(pending)
        ]
        history.append({
            "role":       "assistant",
            "content":    _strip_think(full_raw) or None,
            "tool_calls": tc_list,
        })

        for tc in tc_list:
            name = tc["function"]["name"]
            try:
                args = _json.loads(tc["function"]["arguments"])
            except Exception:
                args = {}
            print(f"[Tool: {name}({args})]", flush=True)
            result = _execute_tool(name, args)
            history.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      result,
            })


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
    Falls back to the 'default' virtual device, then None.
    """
    for name in ("pipewire", "default"):
        idx = _find_device(name, kind)
        if idx is not None:
            return idx
    return None


def _select_input_device() -> tuple[Optional[int], str]:
    """Returns (device_index, name) for the audio input device.
    Prefers Brio 101; falls back to PipeWire default."""
    idx = _find_device("brio", "input")
    if idx is not None:
        return idx, _dev_name(idx, "input")
    pw = _pipewire_device("input")
    return pw, _dev_name(pw, "input")


def _select_output_device() -> tuple[Optional[int], str]:
    """Returns (device_index, name) for the audio output device.
    Prefers P10S speaker; falls back to PipeWire default."""
    idx = _find_device("p10s", "output")
    if idx is not None:
        return idx, _dev_name(idx, "output")
    pw = _pipewire_device("output")
    return pw, _dev_name(pw, "output")


def _detect_sample_rate(device_idx: Optional[int]) -> int:
    """Return the best webrtcvad-compatible sample rate the mic actually supports.
    Tries 16 kHz first (smallest data), then 48 kHz (virtually universal for USB)."""
    for rate in (16_000, 48_000, 32_000, 8_000):
        try:
            sd.check_input_settings(
                device=device_idx, channels=1, dtype="int16", samplerate=rate
            )
            return rate
        except Exception:
            pass
    return 48_000   # safe hard fallback


# ── Greeter Mode ──────────────────────────────────────────────────────────────
def _say_in_character(
    client: OpenAI,
    system_prompt: str,
    instruction: str,
    out_dev: Optional[int],
    voice_name: str,
    rvc_model: Optional[str],
) -> None:
    """Ask the LLM to produce a short in-character line and speak it."""
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": instruction},
            ],
            max_tokens=60,
            timeout=12.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = _strip_think(resp.choices[0].message.content or "").strip()
        if text:
            speak(text, out_dev, voice_name, rvc_model)
    except Exception as exc:
        print(f"[_say_in_character error] {exc}")


def greeter_main(
    client: OpenAI,
    whisper: WhisperModel,
    vad: webrtcvad.Vad,
    in_dev: Optional[int],
    out_dev: Optional[int],
    personality_name: str,
    system_prompt: str,
    voice_name: str,
    rvc_model: Optional[str],
) -> None:
    """
    Greeter Mode: watches the camera and proactively greets visitors.

    WATCHING state  — polls camera every GREETER_POLL_INTERVAL seconds.
    CONVERSING state — greets the person, then listens/responds until they leave.
    """
    import threading

    print(f"\n[Greeter Mode active — watching for visitors…]\n", flush=True)
    speak(
        f"Greeter mode active. I am {personality_name}. Watching for visitors.",
        out_dev, voice_name, rvc_model,
    )

    # ── Background camera scanner ─────────────────────────────────────────────
    # Stores a headcount so the main thread can detect individual arrivals and
    # departures without ever blocking on a camera call itself.
    people_count = [0]   # int: number of people currently visible
    stop_event   = threading.Event()

    def _scan_loop() -> None:
        # Own client so camera scans never contend with the main conversation
        # thread's HTTP connection pool — the key fix for ARM64/local-vLLM setups.
        scan_client = OpenAI(base_url=LLM_BASE_URL, api_key="none")
        while not stop_event.wait(GREETER_POLL_INTERVAL):
            try:
                people_count[0] = _count_people(scan_client)
                print(f"\r[Camera: {people_count[0]} person(s) visible]    ",
                      end="", flush=True)
            except Exception:
                pass

    scanner = threading.Thread(target=_scan_loop, daemon=True)
    scanner.start()

    try:
        while True:
            # ── WATCHING: wait until at least one person appears ──────────────
            if people_count[0] == 0:
                time.sleep(1.0)
                continue

            # ── CONVERSING ────────────────────────────────────────────────────
            prev_count = people_count[0]
            print(f"\n[{prev_count} person(s) detected — starting conversation]",
                  flush=True)
            history: list[dict] = []
            last_spoke   = time.time()
            absent_count = 0   # consecutive camera scans returning 0 people

            try:
                _greet_person(client, history, system_prompt, out_dev, voice_name, rvc_model)
            except Exception as exc:
                print(f"[Greeting error] {exc}")
                _say_in_character(client, system_prompt,
                    "Greet the people in front of you in character (1 sentence).",
                    out_dev, voice_name, rvc_model)
            last_spoke = time.time()

            while True:
                audio = record_speech(in_dev, vad, timeout_sec=GREETER_LISTEN_TIMEOUT)
                curr = people_count[0]

                # ── Absence check (with hysteresis) ──────────────────────────
                # Require GREETER_ABSENT_CHECKS consecutive zero-count readings
                # before ending the conversation — prevents a single false-negative
                # from the camera (e.g. person briefly out of frame, or the LLM
                # misidentifying 0 people) from cutting off mid-conversation.
                if curr == 0:
                    absent_count += 1
                    if absent_count >= GREETER_ABSENT_CHECKS:
                        print("[All people left — returning to watch mode]\n", flush=True)
                        _say_in_character(client, system_prompt,
                            "Everyone has just left. Say a farewell in character (1 sentence).",
                            out_dev, voice_name, rvc_model)
                        prev_count = 0
                        break  # back to WATCHING
                    # Not yet convinced — keep conversing
                    prev_count = curr
                    if audio is None:
                        continue
                else:
                    absent_count = 0  # someone is present; reset counter

                # ── Arrivals / partial departures ────────────────────────────
                if curr != prev_count:
                    if curr > prev_count:
                        n    = curr - prev_count
                        noun = "person has" if n == 1 else f"{n} people have"
                        print(f"[{n} new person(s) arrived]", flush=True)
                        try:
                            llm_reply(client, history,
                                f"[{noun} just joined. Greet them in character while keeping the ongoing conversation going.]",
                                out_dev, system_prompt, voice_name, rvc_model)
                        except Exception as exc:
                            print(f"[LLM error] {exc}")
                        last_spoke = time.time()
                        prev_count = curr
                        continue

                    else:
                        n    = prev_count - curr
                        noun = "person has" if n == 1 else f"{n} people have"
                        print(f"[{n} person(s) left, {curr} remaining]", flush=True)
                        _say_in_character(client, system_prompt,
                            f"{noun} just left the group. Briefly acknowledge their departure in character while remaining available to those still present (1 sentence).",
                            out_dev, voice_name, rvc_model)
                        last_spoke = time.time()
                        prev_count = curr

                # ── Silence / noise timeout ───────────────────────────────────
                if audio is None:
                    if time.time() - last_spoke >= 15:
                        _say_in_character(client, system_prompt,
                            "The people in front of you have been quiet for a while. In character, gently let them know you are still available (1 sentence).",
                            out_dev, voice_name, rvc_model)
                        last_spoke = time.time()
                    continue

                # ── Transcribe ────────────────────────────────────────────────
                print("[Transcribing…]", end=" ", flush=True)
                text = transcribe(audio, whisper)
                if not text:
                    continue
                print(f"You: {text}")
                last_spoke = time.time()

                # Explicit farewell — wait until frame clears before resetting
                # so we don't immediately re-greet departing people.
                if text.lower().strip() in {"quit", "exit", "goodbye", "bye", "stop", "see you"}:
                    _say_in_character(client, system_prompt,
                        "The person just said goodbye to you. Say a warm in-character farewell (1 sentence).",
                        out_dev, voice_name, rvc_model)
                    # Wait up to 10 s for the frame to clear; don't hang forever
                    print("[Waiting for people to leave frame…]", flush=True)
                    deadline = time.time() + 10
                    while people_count[0] > 0 and time.time() < deadline:
                        time.sleep(1.0)
                    prev_count = 0
                    break

                if _wants_personality_change(text):
                    personality_name, system_prompt, voice_name, rvc_model = select_personality()
                    history = []
                    if rvc_model:
                        print(f"  Preloading RVC ({rvc_model})…", flush=True)
                        get_rvc(rvc_model)
                    speak(f"Switching to {personality_name}.", out_dev, voice_name, rvc_model)
                    continue

                print("[Thinking…]")
                try:
                    llm_reply(client, history, text, out_dev, system_prompt, voice_name, rvc_model)
                except Exception as exc:
                    print(f"[LLM error] {exc}")
                    speak("Sorry, I had trouble with that.", out_dev, voice_name, rvc_model)
                last_spoke = time.time()
                # prev_count is intentionally NOT updated here — any count change
                # that occurred while the AI was responding will be caught and
                # announced at the top of the next loop iteration.

    except KeyboardInterrupt:
        print("\n[Greeter Mode stopped]")
        speak("Goodbye!", out_dev, voice_name, rvc_model)
    finally:
        stop_event.set()


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

    in_dev,  in_name  = _select_input_device()
    out_dev, out_name = _select_output_device()
    camera_present    = _find_camera_index() is not None

    # Set sample rates to match the actual hardware devices.
    global SAMPLE_RATE, FRAME_SAMPLES, OUTPUT_SAMPLE_RATE
    SAMPLE_RATE        = _detect_sample_rate(in_dev)
    FRAME_SAMPLES      = int(SAMPLE_RATE * FRAME_MS / 1000)
    OUTPUT_SAMPLE_RATE = int(sd.query_devices(out_dev)["default_samplerate"]) if out_dev is not None else 22_050

    print(f"Mic    → {in_name}")
    print(f"Output → {out_name}")
    if camera_present:
        print("Camera → Brio 101 (video enabled)")
    else:
        print("Camera → not detected (camera features disabled)")

    mode = select_mode()
    personality_name, system_prompt, voice_name, rvc_model = select_personality()

    # Load models
    print("\nLoading models…")
    print(f"  Whisper ({WHISPER_SIZE})…", end=" ", flush=True)
    whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    print("ready")

    print(f"  Piper TTS ({voice_name})…", end=" ", flush=True)
    get_voice(voice_name)
    print("ready")

    if rvc_model:
        print(f"  RVC ({rvc_model})…", flush=True)
        get_rvc(rvc_model)
        print("  RVC ready")

    vad    = webrtcvad.Vad(VAD_MODE)
    client = OpenAI(base_url=LLM_BASE_URL, api_key="none")

    if mode == "greeter":
        greeter_main(client, whisper, vad, in_dev, out_dev,
                     personality_name, system_prompt, voice_name, rvc_model)
        return

    # ── Standard Mode ─────────────────────────────────────────────────────────
    history: list[dict] = []

    conversation_mode = False

    print(f"\nReady!  Say '{WAKE_PHRASE}' to start, Ctrl-C to quit.")
    print(f"Say '{WAKE_PHRASE} start conversation mode' to talk freely without the wake word.\n")
    speak(f"Ready. I am {personality_name}. Say hey Sparky to wake me up.", out_dev, voice_name, rvc_model)

    while True:
        # ── Input: conversation mode (no wake word) ───────────────────────────
        if conversation_mode:
            try:
                audio = record_speech(in_dev, vad)
            except KeyboardInterrupt:
                print("\nGoodbye.")
                speak("Goodbye!", out_dev, voice_name, rvc_model)
                break

            if audio is None:
                continue

            print("[Transcribing…]", end=" ", flush=True)
            text = transcribe(audio, whisper)
            if not text:
                continue

        # ── Input: wake word mode ─────────────────────────────────────────────
        else:
            try:
                inline_question = wait_for_wake_word(in_dev, vad, whisper)
            except KeyboardInterrupt:
                print("\nGoodbye.")
                speak("Goodbye!", out_dev, voice_name, rvc_model)
                break

            play_beep(out_dev)

            if inline_question:
                text = inline_question
                print(f"[Inline question: '{text}']")
            else:
                try:
                    print("[Listening for question…]", end="", flush=True)
                    audio = record_speech(in_dev, vad)
                except KeyboardInterrupt:
                    print("\nGoodbye.")
                    speak("Goodbye!", out_dev, voice_name, rvc_model)
                    break

                if audio is None:
                    continue

                print("[Transcribing…]", end=" ", flush=True)
                text = transcribe(audio, whisper)
                if not text:
                    continue

        print(f"You: {text}")

        if text.lower().strip() in {"quit", "exit", "goodbye", "bye", "stop"}:
            speak("Goodbye!", out_dev, voice_name, rvc_model)
            break

        if _wants_conv_on(text) and not conversation_mode:
            conversation_mode = True
            print("[Conversation mode ON — speak freely, say 'stop conversation mode' to return to wake word]\n")
            speak("Conversation mode on. I'm listening.", out_dev, voice_name, rvc_model)
            continue

        if _wants_conv_off(text) and conversation_mode:
            conversation_mode = False
            print(f"[Wake word mode — say '{WAKE_PHRASE}' to activate]\n")
            speak(f"Returning to wake word mode. Say hey Sparky to wake me up.", out_dev, voice_name, rvc_model)
            continue

        if _wants_personality_change(text):
            personality_name, system_prompt, voice_name, rvc_model = select_personality()
            history = []
            if rvc_model:
                print(f"  Preloading RVC ({rvc_model})…", flush=True)
                get_rvc(rvc_model)
            speak(f"Switching to {personality_name}.", out_dev, voice_name, rvc_model)
            continue

        print("[Thinking…]")
        try:
            llm_reply(client, history, text, out_dev, system_prompt, voice_name, rvc_model)
        except Exception as exc:
            print(f"[LLM error] {exc}")
            speak("Sorry, I had trouble reaching the language model.", out_dev)


if __name__ == "__main__":
    main()
