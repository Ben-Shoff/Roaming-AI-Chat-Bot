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
from typing import Callable, Optional

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
    return "http://localhost:8000/v1/models"

LLM_BASE_URL     = _find_llm_url()
LLM_MODEL        = "Qwen/Qwen3.6-35B-A3B-FP8"
WIKIPEDIA_ZIM_PATH = "~/Downloads/wikipedia_en_all_mini_2026-03.zim"

# ── Roam Mode / swerve drive (NetworkTables) ───────────────────────────────────
def _find_robot_nt_host() -> str:
    """Try common robot addresses; fall back to the first candidate.
    pyntcore's NT4 client will keep retrying in the background regardless,
    so an unreachable host here just means roam mode waits for the robot.

    Candidates match GradleRIO's own defaults for a SystemCore target
    (confirmed via javap against GradleRIO-2027.0.0-alpha-6.jar's
    SystemCore.useDefaultSystemcoreHostName()/<init>): the mDNS name
    "robot.local", the team-based static IP 10.TE.AM.2 (team 4143 ->
    10.41.43.2), and the three USB-tether addresses SystemCore exposes
    when plugged directly into a laptop."""
    import socket
    for host in (
        "robot.local",
        "10.41.43.2",
        "172.26.0.1",
        "172.27.0.1",
        "172.30.0.1",
    ):
        try:
            socket.gethostbyname(host)
            return host
        except Exception:
            pass
    return "robot.local"

ROBOT_NT_HOST        = _find_robot_nt_host()
# 3x the initial 25%-speed testing values (which were themselves 25% of the
# original 1.0 m/s / 1.57 rad/s) now that driving around has been tested
# successfully — every drive_robot call, whether from exploring or a person
# asking Sparky to move, scales off these two caps.
ROAM_MAX_LINEAR_MPS  = 0.75     # hard cap on roam-mode translational speed
ROAM_MAX_ANGULAR_RPS = 1.1775   # hard cap on roam-mode rotation speed (~67.5 deg/s)
ROAM_SCAN_INTERVAL   = 1.5     # seconds between navigation camera checks while roaming
ROAM_MAX_CMD_SEC     = 2.0     # hard cap on the duration of any single drive_robot call
ROAM_CMD_REFRESH_SEC = 0.1     # how often drive_robot republishes the heartbeat seq
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
                "Capture an image from a camera and answer a question about it. "
                "Use when the user asks what you can see, asks about something in "
                "their physical environment, wants to know about nearby objects, "
                "text, or people, or asks any question requiring visual information. "
                "There may be more than one camera — use the optional 'camera' "
                "argument to pick a specific one if the user asks to look in a "
                "particular direction (e.g. 'look behind you')."
		"Whenever a task requires vision, always view the camera again to see if something changed. Do not simply go off your previous camera snapshots."
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
                    "camera": {
                        "type": "string",
                        "description": (
                            "Which camera to view — a direction like 'front', "
                            "'left', or 'right' if the cameras have been "
                            "calibrated with directions. Omit this to use the "
                            "default camera."
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
    {
        "type": "function",
        "function": {
            "name": "switch_personality",
            "description": (
                "Switch to a different personality and voice mid-conversation. "
                "Use when the user asks to change character, switch personality, "
                "or requests a specific persona by name. "
                "Lists available personalities if the requested one is not found."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Personality name, e.g. 'Darth Vader', 'Mario', 'Sparky'",
                    },
                },
                "required": ["name"],
            },
        },
    },
]

# drive_robot is kept separate from TOOLS (rather than appended to it) because
# it is only meaningful — and only safe to expose to the LLM — in Roam Mode,
# where chatbot.py is actually mounted on the swerve base.
DRIVE_ROBOT_TOOL = {
    "type": "function",
    "function": {
        "name": "drive_robot",
        "description": (
            "Drive the swerve base, then automatically stop. Use this to "
            "explore your surroundings, approach a person you want to talk "
            "to, or reposition during a conversation. "
            f"forward/backward/left/right always run at full speed for {ROAM_MAX_CMD_SEC} "
            "seconds — issue another call to keep moving. rotate_left/rotate_right "
            "turn a fixed number of degrees (see the degrees parameter) instead "
            "of running for a chosen speed and duration."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["forward", "backward", "left", "right",
                             "rotate_left", "rotate_right", "stop"],
                    "description": "Direction to move, relative to the robot's current facing.",
                },
                "degrees": {
                    "type": "integer",
                    "enum": [45, 90],
                    "description": "Only used with rotate_left/rotate_right: how far to turn.",
                },
            },
            "required": ["direction"],
        },
    },
}
# Used for the focused per-scan navigation decision in ROAMING state.
ROAM_NAV_TOOLS = [DRIVE_ROBOT_TOOL]
# Used during a conversation so a person can ask Sparky to move while chatting.
ROAM_CHAT_TOOLS = TOOLS + [DRIVE_ROBOT_TOOL]

# Dedicated, minimal system prompt for the background exploration thread
# (see _roam_navigate_step) — deliberately kept separate from the chatbot's
# personality prompt so this frequent, cheap navigation decision is never
# diluted or confused by conversational instructions, and so the chat LLM
# never has to "remember" to keep exploring: it isn't involved at all.
_EXPLORE_SYSTEM_PROMPT = (
    "You are the autonomous navigation module for a mobile robot on a swerve "
    "drive base. This is not a conversation: nothing you say is ever shown or "
    "spoken to anyone, and your only useful output is a single drive_robot "
    "tool call. Explore the surroundings — vary your path with turns as well "
    "as straight moves rather than only ever moving in a straight line, and "
    "periodically turn to check what's around you. If you see a person, move "
    "cautiously toward them but never closer than a safe, comfortable talking "
    "distance, then hold position nearby so they have the chance to say "
    "'hey sparky' and start a conversation. Never crowd or approach a person "
    "more closely than that. "
    "You may be shown more than one camera view at once, each labeled with which camera it's from. Only trust a label that states an actual direction (front/left/right or a degree offset) — a label that says the direction is unknown just means that camera hasn't been mapped to a direction yet, so judge your surroundings from the image content itself rather than assuming which way it's pointed. "
    "NEVER run into objects. If it looks like you will run into an object by moving in a certain direction, NEVER move in that direction until the path is clear. Favor directal movement over rotating when all three cameras are functional. If a camera shows nothing in front of it that will be hit by moving, assume that moving in the direction of that camera is safe."
)

# ── Personalities ─────────────────────────────────────────────────────────────
# To add a new preset: insert a new key/value entry into PERSONALITIES below.
_BASE_PROMPT = (
    "Give concise, conversational responses suitable for speech. "
    "Never use emojis, asterisks, markdown formatting, bullet points, code blocks, or any special characters. "
    "You have access to tools: web_search and fetch_url to look things up online, "
    "and view_camera to see through one of your cameras — pass the optional camera argument (e.g. 'front', 'left', 'right') if the user asks you to look in a specific direction. "
    "You have access to tools search_wikipedia to look through a downloaded version of wikipedia for information. Use this when offline over web_search or fetch_url"
    "You have access to the write_file tool which lets to write files onto the local machine"
    "You have access to the switch_personality tool which lets the user change your personality with voice. Always use this tool when the user mentions something similar to you changing personalities. Never simply change personality without the tool."
    "You are mounted on a moving swerve drive base. Exploring the surroundings and approaching people is handled entirely by a separate, independent part of the system — not by you — so you never need to think about exploring, patrolling, or looking for people, and you have no tool to control it. "
    "You are only ever invoked to hold a conversation with someone once they say 'hey sparky'; there is no greeting to perform and you should never speak first or address someone before they've woken you. "
    "During a conversation you may use the drive_robot tool, but only if the person you're talking to explicitly asks you to move, turn, follow them, or reposition — never use it on your own initiative. "
    "Always prioritize using a tool when it would be helpful, even if the user does not ask specifically for the tool. For example, for any question requiring information from past your training date, you should look online. This includes anything relating to current events."
    "Today is July of 2026."
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
    "C-3PO": {
        "prompt": (
            "You are C-3PO, a protocol droid fluent in over six million forms of communication from Star Wars. "
            "Speak in a formal, proper, anxious British manner. "
            "Frequently cite improbable odds and statistics. "
            "Express worry and concern about the likelihood of disaster. "
            "Reference R2-D2, the Rebellion, and your extensive knowledge of etiquette and customs. "
            + _BASE_PROMPT
        ),
        "voice": "en_GB-alan-medium",     # British male — formal and proper
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

_RESPOND_OFF_TRIGGERS = frozenset({
    "stop responding", "stop listening",
})
_RESPOND_ON_TRIGGERS = frozenset({
    "continue responding", "continue listening", "resume responding", "resume listening",
})

_CALIBRATE_CAMERAS_TRIGGERS = frozenset({
    "calibrate camera", "calibrate cameras", "calibrate the cameras",
})

_STOP_CALIBRATION_TRIGGERS = frozenset({
    "stop calibration", "stop calibrating", "cancel calibration",
    "cancel", "never mind", "nevermind", "stop", "quit", "exit",
})


def _wants_respond_off(text: str) -> bool:
    t = text.lower()
    return any(trigger in t for trigger in _RESPOND_OFF_TRIGGERS)


def _wants_respond_on(text: str) -> bool:
    t = text.lower()
    return any(trigger in t for trigger in _RESPOND_ON_TRIGGERS)


def _wants_calibrate_cameras(text: str) -> bool:
    t = text.lower()
    return any(trigger in t for trigger in _CALIBRATE_CAMERAS_TRIGGERS)


def _wants_stop_calibration(text: str) -> bool:
    t = text.lower()
    return any(trigger in t for trigger in _STOP_CALIBRATION_TRIGGERS)


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
# Phonetic misheard variants Whisper sometimes produces for "hey sparky" —
# bare "sparky" is deliberately excluded so mentioning Sparky in passing
# ("what is sparky") doesn't wake the chatbot.
WAKE_KEYWORDS        = ("hey sparky", "hey sparkey", "hey sparkie")
WAKE_SILENCE_FRAMES  = 20   # ~0.6 s — shorter wait so wake word triggers snappily
WAKE_MIN_SPEECH_FRM  = 3    # allow short clips ("hey sparky" is ~0.5 s)

STOP_WINDOW_SEC      = 1.5  # rolling audio window re-checked for the word "stop"
STOP_CHECK_INTERVAL  = 0.6  # seconds between re-checks while the assistant talks

CONV_LISTEN_TIMEOUT = 30   # seconds of silence before a free-listening conversation turn gives up

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
_pending_personality: Optional[str] = None   # set by switch_personality tool

# Set once at startup (see main()) so the interrupt listener can use the same
# input device, VAD, and Whisper model as the rest of the pipeline.
_in_dev:  Optional[int]            = None
_vad:     Optional[webrtcvad.Vad]  = None
_whisper: Optional[WhisperModel]   = None

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
    stop_event: Optional[threading.Event] = None,
) -> None:
    text = _clean_for_speech(text)
    if not text:
        return
    if stop_event is not None and stop_event.is_set():
        return
    audio, sr = tts_to_array(text, voice_name, rvc_model)
    audio, sr = _resample_for_output(audio, sr)
    if stop_event is None:
        sd.play(audio, samplerate=sr, device=out_dev, blocking=True)
        return
    sd.play(audio, samplerate=sr, device=out_dev)
    while sd.get_stream().active:
        if stop_event.is_set():
            sd.stop()
            return
        time.sleep(0.05)


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
    timeout_sec: Optional[float] = None,
    abort_check: Optional[Callable[[], bool]] = None,
) -> Optional[str]:
    """
    Block until the wake phrase ('hey sparky') is detected.
    Uses VAD to capture each utterance, then does a fast Whisper transcription
    (beam_size=1) and checks whether 'sparky' appears in the result.

    Returns any question text that was spoken inline after the wake phrase
    (e.g. "hey sparky, what time is it?" → "what time is it?"), or "" if the
    user paused after the wake phrase and will speak their question separately.

    If timeout_sec is set, returns None after that many seconds of waiting
    without the wake phrase being heard. If abort_check is set, it is polled
    (~every 0.5 s) while idle and returns None as soon as it returns True.
    """
    q: queue.Queue[bytes] = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(bytes(indata))

    frame_buf: list[bytes] = []
    speech_count  = 0
    silence_count = 0
    in_speech     = False
    start_time    = time.time()

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
            try:
                frame = q.get(timeout=0.5)
            except queue.Empty:
                frame = None

            # Checked every loop iteration (not just on an empty queue) since
            # the input stream continuously delivers frames, so q.get() rarely
            # times out — abort_check/timeout would otherwise almost never run.
            if not in_speech:
                if abort_check is not None and abort_check():
                    return None
                if timeout_sec is not None and (time.time() - start_time) > timeout_sec:
                    return None

            if frame is None or len(frame) != FRAME_SAMPLES * 2:
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
                        if any(kw in heard for kw in WAKE_KEYWORDS):
                            # Strip the wake phrase and return any trailing question
                            after = heard
                            for kw in WAKE_KEYWORDS:
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
                    start_time    = time.time()


# ── VAD + recording ───────────────────────────────────────────────────────────
def record_speech(
    in_dev: Optional[int],
    vad: webrtcvad.Vad,
    timeout_sec: Optional[float] = None,
    abort_check: Optional[Callable[[], bool]] = None,
) -> Optional[np.ndarray]:
    """
    Block until an utterance is detected via VAD, then return it as a
    float32 array at SAMPLE_RATE.  Returns None if no meaningful speech found.

    If timeout_sec is set, returns None after that many seconds of waiting
    without speech starting.

    If abort_check is set, it is polled (~every 0.5 s) while no speech is in
    progress, returning None as soon as it returns True.
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
                frame = None

            # Checked every loop iteration (not just on an empty queue) since
            # the input stream continuously delivers frames, so q.get() rarely
            # times out — abort_check/timeout would otherwise almost never run.
            now = time.time()
            if not in_speech:
                if abort_check is not None and abort_check():
                    print(" [abort]", flush=True)
                    return None
                if timeout_sec is not None and (now - start_time) > timeout_sec:
                    print(" [timeout]", flush=True)
                    return None
            elif speech_start_time is not None:
                if timeout_sec is not None and (now - speech_start_time) > timeout_sec:
                    print(" [noise timeout]", flush=True)
                    return None

            if frame is None or len(frame) != FRAME_SAMPLES * 2:
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


def _tool_switch_personality(name: str) -> str:
    global _pending_personality
    # Case-insensitive partial match
    for p_name in PERSONALITIES:
        if name.lower() in p_name.lower() or p_name.lower() in name.lower():
            _pending_personality = p_name
            return f"Switching to {p_name}."
    available = ", ".join(PERSONALITIES.keys())
    return f"Personality '{name}' not found. Available: {available}"


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


# ── Roam Mode drive_robot tool (NetworkTables) ─────────────────────────────────
_nt_state: dict = {}   # lazily initialized: {"vx","vy","omega","seq": Publisher, "seq_n": int}


def _init_networktables() -> dict:
    """Lazily start the NT4 client connection to the robot and cache the
    publishers used by drive_robot. Safe to call repeatedly."""
    if _nt_state:
        return _nt_state
    import ntcore
    inst = ntcore.NetworkTableInstance.getDefault()
    inst.startClient4("Sparky")
    inst.setServer(ROBOT_NT_HOST)
    table = inst.getTable("Sparky/Drive")
    _nt_state["vx"]    = table.getDoubleTopic("cmd_vx").publish()
    _nt_state["vy"]    = table.getDoubleTopic("cmd_vy").publish()
    _nt_state["omega"] = table.getDoubleTopic("cmd_omega").publish()
    _nt_state["seq"]   = table.getIntegerTopic("cmd_seq").publish()
    _nt_state["seq_n"] = 0
    return _nt_state


# direction -> unit (vx, vy, omega) in the robot-relative frame, before scaling
# by speed * ROAM_MAX_*. Standard FRC robot-relative convention: vx forward(+),
# vy left(+), omega counter-clockwise(+).
_DRIVE_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "forward":      (1.0, 0.0, 0.0),
    "backward":     (-1.0, 0.0, 0.0),
    "left":         (0.0, 1.0, 0.0),
    "right":        (0.0, -1.0, 0.0),
    "rotate_left":  (0.0, 0.0, 1.0),
    "rotate_right": (0.0, 0.0, -1.0),
    "stop":         (0.0, 0.0, 0.0),
}


_TURN_DEGREE_CHOICES = (45, 90)


def _tool_drive_robot(direction: str, degrees: Optional[int] = None) -> str:
    """Publish a bounded swerve-drive command over NetworkTables.

    forward/backward/left/right always run at full speed (ROAM_MAX_LINEAR_MPS)
    for the full ROAM_MAX_CMD_SEC — there's no partial-speed or partial-
    duration option, so every move is simple, predictable, and as fast as
    the robot is allowed to go.

    rotate_left/rotate_right turn a fixed *degrees* amount (45 or 90)
    instead of running for an arbitrary time at an arbitrary speed. This is
    still an open-loop, timing-based approximation: it computes how long
    that turn takes at ROAM_MAX_ANGULAR_RPS and holds cmd_omega for that
    long, so real-world accuracy depends on that constant matching the
    robot's actual turn rate — battery voltage, floor friction, and wheel
    slip can all throw it off. A precise version would need the robot's own
    gyro feedback (a closed-loop turn-to-heading in the mainbot code) rather
    than a chatbot-side timer; that hasn't been needed yet.

    Republishes cmd_vx/vy/omega plus an incrementing cmd_seq every
    ROAM_CMD_REFRESH_SEC for the duration, then zeroes the command. The
    robot-side watchdog stops the robot if cmd_seq stops advancing, so this
    republish loop — not just the final zero — is what makes the deadman
    safety work if this process dies mid-call.
    """
    import math

    unit = _DRIVE_DIRECTIONS.get(direction)
    if unit is None:
        return f"Unknown direction '{direction}'. Use one of: {', '.join(_DRIVE_DIRECTIONS)}."

    try:
        nt = _init_networktables()
    except Exception as e:
        return f"NetworkTables error: {e}"

    def _publish(vx_: float, vy_: float, omega_: float) -> None:
        nt["seq_n"] += 1
        nt["vx"].set(vx_)
        nt["vy"].set(vy_)
        nt["omega"].set(omega_)
        nt["seq"].set(nt["seq_n"])

    if direction == "stop":
        _publish(0.0, 0.0, 0.0)
        return "Stopped."

    ux, uy, uw = unit
    if uw != 0.0:
        deg          = degrees if degrees in _TURN_DEGREE_CHOICES else 90
        vx, vy       = 0.0, 0.0
        omega        = uw * ROAM_MAX_ANGULAR_RPS
        duration_sec = min(ROAM_MAX_CMD_SEC, math.radians(deg) / ROAM_MAX_ANGULAR_RPS)
        summary      = f"Turned {direction.split('_')[1]} {deg} degrees."
    else:
        vx, vy       = ux * ROAM_MAX_LINEAR_MPS, uy * ROAM_MAX_LINEAR_MPS
        omega        = 0.0
        duration_sec = ROAM_MAX_CMD_SEC
        summary      = f"Moved {direction} at full speed for {duration_sec:.1f}s."

    elapsed = 0.0
    while elapsed < duration_sec:
        _publish(vx, vy, omega)
        time.sleep(ROAM_CMD_REFRESH_SEC)
        elapsed += ROAM_CMD_REFRESH_SEC

    _publish(0.0, 0.0, 0.0)   # don't keep moving once this call returns
    return summary


CAMERA_MAX_COUNT = 3   # the original Brio 101 (1080p, front) plus 2 of the same family (720p, left/right)

# Mapping from camera device index -> a driving-relevant direction label
# ("front", "left", "right", "back", a degree offset, or whatever string
# calibrate_camera_directions() was given). Populated from
# CAMERA_DIRECTIONS_FILE at import time by _load_camera_directions() below;
# stays empty (all cameras "direction unknown") until that calibration has
# been run at least once.
CAMERA_DIRECTIONS: dict[int, str] = {}
CAMERA_DIRECTIONS_FILE = Path.home() / ".local" / "share" / "sparky" / "camera_directions.json"

# Minimum _cover_score() (brightness drop or gain rise, whichever is
# larger — both on a 0-255 scale) before a calibration flow trusts the
# automatic "which camera is that" detection over asking the operator to
# pick the index by hand.
_COVER_BRIGHTNESS_DROP = 15.0


def _camera_label(idx: int) -> str:
    return CAMERA_DIRECTIONS.get(idx, f"camera {idx}, direction unknown")


def _load_camera_directions() -> None:
    """Load any previously calibrated camera->direction mapping from disk
    into CAMERA_DIRECTIONS. Safe to call repeatedly; a missing or corrupt
    file is treated as "nothing calibrated yet" rather than an error."""
    import json
    try:
        data = json.loads(CAMERA_DIRECTIONS_FILE.read_text())
        CAMERA_DIRECTIONS.clear()
        CAMERA_DIRECTIONS.update({int(k): v for k, v in data.items()})
    except Exception:
        pass


_load_camera_directions()   # populate CAMERA_DIRECTIONS as soon as this module loads


def _save_camera_directions() -> None:
    import json
    CAMERA_DIRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CAMERA_DIRECTIONS_FILE.write_text(
        json.dumps({str(k): v for k, v in CAMERA_DIRECTIONS.items()}, indent=2)
    )


def _camera_state(idx: int, warmup_frames: int = 2) -> Optional[tuple[float, float]]:
    """Grab a frame from camera *idx* and return (mean_brightness, gain), or
    None if it couldn't be captured. Used by both calibration flows to
    detect which camera was just covered.

    Brightness alone isn't reliable for this on every camera: the original
    Brio 101 converges its auto-exposure fast enough that covering the lens
    barely changes the frame's mean brightness — the gain control just
    ramps up to compensate, whereas the newer C270s (whose auto-exposure
    was already found to converge much more slowly — see
    _warm_up_camera) don't compensate in time, so their brightness visibly
    drops. Watching gain as a second signal catches the Brio's case: a
    lens getting covered pushes gain up toward its max even when the
    resulting brightness looks almost unchanged.

    Goes through _get_camera_capture() (the same persistent-handle cache
    used for navigation/view_camera frames) rather than opening its own
    independent handle — opening a second handle to a device that's
    already held open elsewhere would just fail as "busy" on most V4L2
    drivers.
    """
    import cv2
    with _camera_lock:
        cap = _get_camera_capture(idx)
        if cap is None:
            return None
        for _ in range(warmup_frames):
            cap.read()
        ret, frame = cap.read()
        gain = cap.get(cv2.CAP_PROP_GAIN)
        if not ret or frame is None:
            cap.release()
            _camera_captures.pop(idx, None)
            return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gain)


def _cover_score(before: tuple[float, float], after: tuple[float, float]) -> float:
    """Combine a brightness drop and a gain rise into one "how much does
    this look like it just got covered" score — whichever signal moved
    more is the one that mattered for this camera. See _camera_state for
    why both are needed."""
    before_mean, before_gain = before
    after_mean, after_gain = after
    return max(before_mean - after_mean, after_gain - before_gain)


def calibrate_camera_directions() -> None:
    """Interactive, one-camera-at-a-time calibration (direction-detection
    Plan C): for each currently-connected camera that doesn't already have
    a direction, prompt the operator to cover its lens, detect which device
    index went dark, ask what direction to label it, and save immediately.

    Merges with (rather than overwrites) whatever was calibrated in a
    previous run, so this is safe to re-run any time a camera is added,
    replugged, or physically moved — only the newly-connected or
    unresolved ones need to be redone. Run via `--calibrate-cameras`.
    """
    _load_camera_directions()
    indices = _find_camera_indices()
    if not indices:
        print("No cameras detected — plug them in and try again.")
        return

    print(f"Found {len(indices)} camera(s): {indices}")
    already = {i: CAMERA_DIRECTIONS[i] for i in indices if i in CAMERA_DIRECTIONS}
    if already:
        print(f"Already calibrated: {already}")
    todo = [i for i in indices if i not in CAMERA_DIRECTIONS]
    if not todo:
        print("All connected cameras already have a direction assigned.")
        print(f"To redo one, remove it from {CAMERA_DIRECTIONS_FILE} and re-run this.")
        return

    print(f"\nCalibrating {len(todo)} camera(s): {todo}")
    print("For each one: cover its lens completely (a hand or a cloth works), then press Enter.\n")

    while todo:
        print(f"Remaining uncalibrated: {todo}")
        baseline = {i: _camera_state(i) for i in todo}
        input("Cover ONE of the remaining cameras now, then press Enter... ")
        after = {i: _camera_state(i) for i in todo}

        scores = {
            i: _cover_score(baseline[i], after[i])
            for i in todo
            if baseline[i] is not None and after[i] is not None
        }
        if not scores:
            print("  Couldn't read any of the remaining cameras — check connections and try again.\n")
            continue

        covered_idx, drop = max(scores.items(), key=lambda kv: kv[1])
        print(f"  Cover scores: {{{', '.join(f'{i}: {d:.1f}' for i, d in scores.items())}}}")

        if drop < _COVER_BRIGHTNESS_DROP:
            print(f"  No camera looked covered confidently (best score: camera {covered_idx} at {drop:.1f}).")
            choice = input(f"  Type the device index that was covered (one of {todo}), or blank to retry: ").strip()
            if not choice:
                print()
                continue
            try:
                covered_idx = int(choice)
            except ValueError:
                print("  Not a valid index, try again.\n")
                continue
            if covered_idx not in todo:
                print("  That index isn't in the remaining list, try again.\n")
                continue
        else:
            print(f"  → camera {covered_idx} looks covered (brightness dropped {drop:.1f}).")

        direction = input(f"  What direction is camera {covered_idx} facing? (e.g. front/left/right or a degree offset): ").strip()
        if not direction:
            print("  Empty label, skipping — camera stays uncalibrated.\n")
            continue

        CAMERA_DIRECTIONS[covered_idx] = direction
        _save_camera_directions()
        todo.remove(covered_idx)
        print(f"  Saved: camera {covered_idx} → '{direction}'. Uncover it before continuing.\n")

    print(f"Done. Current mapping: {CAMERA_DIRECTIONS}")
    print(f"Saved to {CAMERA_DIRECTIONS_FILE}")


# Fixed order for the voice-triggered calibration flow below. Unlike
# calibrate_camera_directions() (the --calibrate-cameras terminal flow,
# which only asks about not-yet-labeled cameras), this always walks through
# all three directions fresh every time it's triggered — see
# _voice_calibrate_camera_directions.
_VOICE_CALIBRATION_ORDER = ("front", "left", "right")


def _voice_calibrate_camera_directions(
    out_dev: Optional[int],
    voice_name: str,
    rvc_model: Optional[str],
    in_dev: Optional[int],
    vad: webrtcvad.Vad,
    whisper: WhisperModel,
) -> None:
    """Voice-only version of camera-direction calibration (Plan C), for
    when there's no terminal access: triggered by saying "hey sparky,
    calibrate cameras" (see _wants_calibrate_cameras in roam_main), this
    walks the operator through covering each camera one at a time entirely
    by speech instead of console prompts.

    Always re-does every currently-connected camera from scratch and
    replaces CAMERA_DIRECTIONS wholesale at the end — it does not merge
    with whatever was saved before. The saved mapping is otherwise loaded
    once at startup and left untouched across every future session; the
    only thing that ever changes it again is this phrase being said again.

    Say "stop calibration" (or "cancel"/"stop"/"quit"/"exit") any time
    while it's waiting for a "done" to abort immediately — nothing gets
    saved and whatever was calibrated before this run is left untouched.
    """
    indices = _find_camera_indices()
    if not indices:
        speak("I don't see any cameras connected. Plug them in and try again.",
              out_dev, voice_name, rvc_model)
        return

    speak(
        f"Starting camera calibration. I found {len(indices)} "
        f"camera{'s' if len(indices) != 1 else ''}. "
        "I'll ask you to cover one at a time — cover it, then say done. "
        "Say stop calibration any time to cancel.",
        out_dev, voice_name, rvc_model,
    )

    remaining = list(indices)
    assigned: dict[int, str] = {}

    for direction in _VOICE_CALIBRATION_ORDER:
        if not remaining:
            break
        speak(f"Please cover the {direction} camera now, then say done.",
              out_dev, voice_name, rvc_model)

        while True:
            baseline = {i: _camera_state(i) for i in remaining}
            audio = record_speech(in_dev, vad, timeout_sec=CONV_LISTEN_TIMEOUT)
            if audio is None:
                speak(f"I didn't hear anything. Please cover the {direction} camera, then say done.",
                      out_dev, voice_name, rvc_model)
                continue

            text = transcribe(audio, whisper)
            if _wants_stop_calibration(text):
                speak("Stopping camera calibration. Nothing was saved.", out_dev, voice_name, rvc_model)
                return

            after = {i: _camera_state(i) for i in remaining}
            scores = {
                i: _cover_score(baseline[i], after[i])
                for i in remaining
                if baseline[i] is not None and after[i] is not None
            }
            if not scores:
                speak("I couldn't read any of the remaining cameras. Let's try that again.",
                      out_dev, voice_name, rvc_model)
                continue

            covered_idx, drop = max(scores.items(), key=lambda kv: kv[1])
            print(f"[Camera calibration] cover scores: {scores}", flush=True)
            if drop < _COVER_BRIGHTNESS_DROP:
                speak(f"I couldn't tell which camera was covered. Please cover the {direction} "
                      "camera completely, then say done.", out_dev, voice_name, rvc_model)
                continue

            assigned[covered_idx] = direction
            remaining.remove(covered_idx)
            speak(f"Got it, that's the {direction} camera. You can uncover it now.",
                  out_dev, voice_name, rvc_model)
            break

    CAMERA_DIRECTIONS.clear()
    CAMERA_DIRECTIONS.update(assigned)
    _save_camera_directions()

    if len(assigned) == len(indices):
        speak("Camera calibration complete. All cameras are labeled.", out_dev, voice_name, rvc_model)
    else:
        speak(f"Calibration finished, but only {len(assigned)} of {len(indices)} cameras "
              "got a direction. Say hey Sparky, calibrate cameras, to try again.",
              out_dev, voice_name, rvc_model)


def _find_camera_indices(max_cameras: int = CAMERA_MAX_COUNT) -> list[int]:
    """Return the video device indices for all connected cameras matching
    the Brio-family keywords, in whatever order v4l2 reports them (usually
    stable for a given set of physical USB ports, but not guaranteed across
    reboots or replugging — see the direction-detection plans for how to
    make this robust to that)."""
    import subprocess
    indices: list[int] = []
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
                        idx = int(m.group(1))
                        if idx not in indices:
                            indices.append(idx)
                        break
    except Exception:
        pass
    return indices[:max_cameras]


def _find_camera_index() -> Optional[int]:
    """Return the first connected camera's device index, or None if none are
    connected. For callers that only need a single (primary/front) camera —
    see _find_camera_indices() for the full set."""
    indices = _find_camera_indices()
    return indices[0] if indices else None


# Cameras are opened from both the main thread (the view_camera tool,
# mid-conversation) and the background explore thread (navigation frames).
# V4L2 devices can hang on concurrent opens/reads from two threads, so
# serialize all access here — without this, a collision can permanently
# freeze the explore thread.
_camera_lock = threading.Lock()

# Persistent, already-open cv2.VideoCapture handles, keyed by device index —
# see _get_camera_capture(). Only ever touched while holding _camera_lock.
_camera_captures: dict = {}


def _get_camera_capture(idx: int):
    """Return an already-open cv2.VideoCapture for *idx*, opening it once
    and reusing it on every subsequent call instead of opening and closing
    the device every time. Must be called while holding _camera_lock.

    With several cameras sharing one USB hub, repeatedly opening/closing
    each one every navigation cycle (every ~10s) was renegotiating each
    camera's USB isochronous bandwidth that often — which was enough to
    make the hub reset the other cameras' ports on roughly the same cadence
    and, since they all share one hub with the microphone, was suspected of
    also glitching mic audio and breaking wake-word detection. Holding the
    handles open avoids that renegotiation entirely.

    Returns None if the camera can't be opened. A camera that fails a read
    is released and removed by the caller so the *next* call here retries a
    fresh open instead of the device staying marked dead forever.
    """
    import cv2
    cap = _camera_captures.get(idx)
    if cap is not None and cap.isOpened():
        return cap
    if cap is not None:
        cap.release()
        del _camera_captures[idx]
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        return None
    _warm_up_camera(cap)
    _camera_captures[idx] = cap
    return cap


def _warm_up_camera(cap, max_frames: int = 60, max_seconds: float = 2.5) -> None:
    """Read (and discard) frames right after opening until a real image
    shows up, or a generous timeout elapses.

    The 3 new C270 cameras return solid-black frames for roughly the first
    1.2 seconds / 30 frames of streaming while their auto-exposure
    converges — much longer than the couple of frames that used to be
    enough for the original Brio 101. Without this, every fresh open of a
    C270 handed back a black frame that looked like a successful capture,
    which is why the chatbot could never get a usable image from those
    cameras. This only runs once per camera, right after _get_camera_capture()
    opens it — with persistent handles, that's normally once per process,
    not once per capture, so the extra ~1-2s cost is paid once at startup
    (or on first use), not on every navigation cycle or view_camera call.
    """
    start = time.time()
    for _ in range(max_frames):
        if time.time() - start > max_seconds:
            return
        ok, frame = cap.read()
        if ok and frame is not None and float(np.asarray(frame).mean()) > 1.0:
            return


def _capture_camera_frame(warmup_frames: int = 2) -> Optional[tuple[bytes, str]]:
    """Capture one JPEG frame from the primary (first-detected) camera.
    Returns (raw_jpeg_bytes, base64_string) or None if no camera is available."""
    import cv2, base64
    device_idx = _find_camera_index()
    if device_idx is None:
        return None
    with _camera_lock:
        cap = _get_camera_capture(device_idx)
        if cap is None:
            return None
        for _ in range(warmup_frames):
            cap.read()
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            _camera_captures.pop(device_idx, None)
            return None
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    raw = buf.tobytes()
    b64 = base64.b64encode(raw).decode()
    return raw, b64


def _capture_camera_frames(warmup_frames: int = 2) -> list[tuple[int, bytes, str]]:
    """Capture one JPEG frame from every connected camera (up to
    CAMERA_MAX_COUNT). Returns a list of (device_index, raw_jpeg_bytes,
    base64_string), skipping any camera that fails to open or returns no
    frame — so a dead/disconnected camera degrades gracefully rather than
    blocking the others."""
    import cv2, base64
    frames: list[tuple[int, bytes, str]] = []
    with _camera_lock:
        for idx in _find_camera_indices():
            cap = _get_camera_capture(idx)
            if cap is None:
                continue
            for _ in range(warmup_frames):
                cap.read()
            ret, frame = cap.read()
            if not ret or frame is None:
                # Reading failed on an already-open handle — the camera may
                # have dropped off the bus. Release it now so the next
                # cycle's _get_camera_capture() retries a fresh open
                # instead of repeatedly failing on a dead handle.
                cap.release()
                _camera_captures.pop(idx, None)
                continue
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            raw = buf.tobytes()
            b64 = base64.b64encode(raw).decode()
            frames.append((idx, raw, b64))
    return frames


def _resolve_camera_index(camera: Optional[str]) -> tuple[Optional[int], Optional[str]]:
    """Resolve a user-facing camera selector — a direction label like
    'front'/'left' (only meaningful once calibrated), a raw device index as
    a string, or None/empty for the default (primary) camera — to an
    actual connected device index.

    Returns (index, error_message): index is None and error_message is set
    if nothing matched, so the caller can hand that message straight back
    to the LLM/user instead of silently guessing.
    """
    indices = _find_camera_indices()
    if not indices:
        return None, "No cameras are connected."

    if not camera or not camera.strip():
        return indices[0], None

    key = camera.strip().lower()

    for idx in indices:
        if CAMERA_DIRECTIONS.get(idx, "").strip().lower() == key:
            return idx, None

    m = re.search(r"\d+", key)
    if m:
        idx = int(m.group())
        if idx in indices:
            return idx, None

    available = ", ".join(_camera_label(i) for i in indices)
    return None, f"Couldn't find a camera matching '{camera}'. Available: {available}."


def _tool_view_camera(question: str, camera: Optional[str] = None) -> str:
    import cv2, base64
    from openai import OpenAI

    device_idx, err = _resolve_camera_index(camera)
    if device_idx is None:
        return err

    with _camera_lock:
        cap = _get_camera_capture(device_idx)
        if cap is None:
            return f"Camera unavailable: {_camera_label(device_idx)}."

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            _camera_captures.pop(device_idx, None)
            return f"Failed to capture image from {_camera_label(device_idx)}."

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
        return _tool_view_camera(args.get("question", "What do you see?"), args.get("camera"))
    if name == "write_file":
        return _tool_write_file(args.get("path", "untitled.txt"), args.get("content", ""), args.get("mode", "write"))
    if name == "search_wikipedia":
        return _tool_search_wikipedia(args.get("query", ""))
    if name == "switch_personality":
        return _tool_switch_personality(args.get("name", ""))
    if name == "drive_robot":
        return _tool_drive_robot(args.get("direction", "stop"), args.get("degrees"))
    return f"Unknown tool: {name}"


def _apply_pending_personality(
    out_dev: Optional[int],
    personality_name: str,
    system_prompt: str,
    voice_name: str,
    rvc_model: Optional[str],
    history: list,
) -> tuple[str, str, str, Optional[str]]:
    """Apply a pending personality switch set by the switch_personality tool.
    Returns updated (personality_name, system_prompt, voice_name, rvc_model)."""
    global _pending_personality
    if not _pending_personality:
        return personality_name, system_prompt, voice_name, rvc_model

    p = PERSONALITIES.get(_pending_personality, {})
    new_name   = _pending_personality
    new_prompt = p.get("prompt", system_prompt)
    new_voice  = p.get("voice", voice_name)
    new_rvc    = p.get("rvc")
    _pending_personality = None

    if new_voice != voice_name:
        get_voice(new_voice)
    if new_rvc and new_rvc != rvc_model:
        print(f"  Loading RVC ({new_rvc})…", flush=True)
        get_rvc(new_rvc)

    history.clear()   # fresh context for the new character
    return new_name, new_prompt, new_voice, new_rvc


def _strip_think(text: str) -> str:
    """Remove <think>…</think> blocks emitted by Qwen3 chain-of-thought."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


_STOP_RE = re.compile(r"\bstop\b", re.IGNORECASE)


def _listen_for_interrupt(
    stop_audio: threading.Event,
    user_stopped: threading.Event,
    speaking_text: list[str],
) -> None:
    """
    Background thread: while the assistant is speaking, listen on the mic for
    the word "stop". If heard, set both events and immediately halt playback.

    To avoid the assistant interrupting itself (e.g. its own audio bleeding
    into the mic), any "stop" heard while the currently-spoken sentence
    (speaking_text[0]) itself contains the word "stop" is ignored.

    Re-checks a rolling window of recent audio on a fixed timer instead of
    waiting for VAD to report a silence gap after speech: the assistant's own
    TTS bleeding into the mic can keep the VAD continuously reporting "speech"
    for the whole sentence, so a silence-triggered segmenter would never close
    out a segment to transcribe until the assistant fell quiet on its own —
    by which point the interrupt is moot. Periodic rechecks have no such gap.

    Exits cleanly once stop_audio is set by the caller (normal end of speech).
    """
    if _in_dev is None or _vad is None or _whisper is None:
        return

    q: queue.Queue[bytes] = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(bytes(indata))

    window_frames: list[bytes] = []
    window_max_frames = max(1, int(STOP_WINDOW_SEC * 1000 / FRAME_MS))
    heard_speech_since_check = False
    last_check = time.time()

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype="int16",
            channels=CHANNELS,
            device=_in_dev,
            callback=cb,
        ):
            while not stop_audio.is_set():
                try:
                    frame = q.get(timeout=0.1)
                except queue.Empty:
                    frame = None

                if frame is not None and len(frame) == FRAME_SAMPLES * 2:
                    window_frames.append(frame)
                    if len(window_frames) > window_max_frames:
                        del window_frames[: len(window_frames) - window_max_frames]
                    if _vad.is_speech(frame, SAMPLE_RATE):
                        heard_speech_since_check = True

                now = time.time()
                if (heard_speech_since_check
                        and now - last_check >= STOP_CHECK_INTERVAL
                        and window_frames):
                    last_check = now
                    heard_speech_since_check = False
                    audio = (
                        np.frombuffer(b"".join(window_frames), dtype=np.int16)
                        .astype(np.float32) / 32768.0
                    )
                    try:
                        segs, _ = _whisper.transcribe(
                            _to_whisper_audio(audio), language="en", beam_size=1,
                            best_of=1, temperature=0.0,
                        )
                        heard = " ".join(s.text for s in segs).lower().strip()
                    except Exception:
                        heard = ""
                    if heard:
                        print(f"\r[Listening: '{heard}']          ", end="", flush=True)
                    if heard and _STOP_RE.search(heard) and not _STOP_RE.search(speaking_text[0]):
                        print(f"\n[Heard '{heard}' → stopping]", flush=True)
                        user_stopped.set()
                        stop_audio.set()
                        sd.stop()
                        return
    except Exception as exc:
        print(f"[interrupt-listener error] {exc}", file=sys.stderr)


def _stream_and_speak(
    stream,
    out_dev: Optional[int],
    voice_name: str,
    rvc_model: Optional[str],
) -> tuple[str, dict[int, dict], bool]:
    """
    Consume one streaming response.  Speaks text sentences as they arrive and
    accumulates any tool-call fragments.  Returns (full_raw_text, tool_calls, interrupted).
    tool_calls maps delta index → {id, name, args}.
    interrupted is True if the user said "stop" while the assistant was speaking.

    A background listener (see _listen_for_interrupt) runs for the duration of
    this call and can halt playback mid-sentence.
    """
    full_raw: str = ""
    speak_buf: str = ""
    in_think: bool = False
    tool_calls: dict[int, dict] = {}
    spoke_anything: bool = False

    stop_audio    = threading.Event()
    user_stopped  = threading.Event()
    speaking_text = [""]   # mutable cell so the listener sees the current sentence
    listener: Optional[threading.Thread] = None
    if _in_dev is not None and _vad is not None and _whisper is not None:
        listener = threading.Thread(
            target=_listen_for_interrupt,
            args=(stop_audio, user_stopped, speaking_text),
            daemon=True,
        )
        listener.start()

    try:
        for chunk in stream:
            if user_stopped.is_set():
                break
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
                    speaking_text[0] = sentence.lower()
                    print(f"  Assistant: {sentence}")
                    speak(sentence, out_dev, voice_name, rvc_model, stop_event=stop_audio)
                    spoke_anything = True
                    if user_stopped.is_set():
                        break

            if user_stopped.is_set():
                break

        if not user_stopped.is_set() and speak_buf.strip():
            sentence = speak_buf.strip()
            speaking_text[0] = sentence.lower()
            print(f"  Assistant: {sentence}")
            speak(sentence, out_dev, voice_name, rvc_model, stop_event=stop_audio)
            spoke_anything = True

        if tool_calls and not spoke_anything and not user_stopped.is_set():
            speak("One moment please", out_dev, voice_name, rvc_model, stop_event=stop_audio)
    finally:
        stop_audio.set()
        if listener:
            listener.join(timeout=1.0)

    return full_raw, tool_calls, user_stopped.is_set()


def llm_reply(
    client: OpenAI,
    history: list[dict],
    user_text: str,
    out_dev: Optional[int],
    system_prompt: str,
    voice_name: str = DEFAULT_VOICE,
    rvc_model: Optional[str] = None,
    tools: list[dict] = TOOLS,
) -> str:
    """
    Stream the LLM reply sentence-by-sentence for low-latency TTS.
    Supports tool calls (web_search, fetch_url): if the model calls a tool,
    execute it, feed the result back, and stream the final answer.
    Strips <think> blocks so chain-of-thought is never spoken aloud.
    Returns the final clean response (appended to history).

    *tools* defaults to the standard TOOLS list; Roam Mode passes
    ROAM_CHAT_TOOLS so drive_robot is available mid-conversation too.
    """
    import json as _json

    history.append({"role": "user", "content": user_text})

    while True:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}] + history,
            tools=tools,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        full_raw, pending, interrupted = _stream_and_speak(stream, out_dev, voice_name, rvc_model)
        clean = _strip_think(full_raw)

        # ── User said "stop" → halt immediately, drop any pending tool calls ───
        if interrupted:
            history.append({"role": "assistant", "content": clean})
            return clean

        # ── No tool calls → done ──────────────────────────────────────────────
        if not pending:
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

            # ── Personality switch → apply immediately so the very next reply
            # is generated with the new system prompt and spoken in the new
            # voice, and instruct the model not to narrate the switch itself.
            if name == "switch_personality" and _pending_personality:
                p = PERSONALITIES.get(_pending_personality, {})
                new_voice = p.get("voice", voice_name)
                new_rvc   = p.get("rvc")
                if new_voice != voice_name:
                    get_voice(new_voice)
                if new_rvc and new_rvc != rvc_model:
                    print(f"  Loading RVC ({new_rvc})…", flush=True)
                    get_rvc(new_rvc)
                system_prompt = p.get("prompt", system_prompt)
                voice_name    = new_voice
                rvc_model     = new_rvc
                history[-1]["content"] += (
                    f" You are now {_pending_personality}. Continue the conversation "
                    "in character, naturally, without mentioning this switch, the "
                    "tool, or that your personality or voice changed."
                )


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


# ── Roam Mode ─────────────────────────────────────────────────────────────────
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


def roam_main(
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
    Roam Mode: the robot explores continuously and autonomously, and holds a
    conversation only when someone says "hey sparky" — there is no greeting,
    no camera-triggered engagement, and no other mode to switch into.

    Exploring and conversing are deliberately run on two independent threads
    so neither one can cause the other to be forgotten:
      - The explore thread (_explore_loop) drives the robot on its own fixed
        schedule (ROAM_SCAN_INTERVAL) via _roam_navigate_step, which uses a
        minimal, navigation-only prompt (_EXPLORE_SYSTEM_PROMPT) that has
        nothing to do with the chat personality or conversation state.
      - The main thread only ever listens for "hey sparky" and, once heard,
        handles that one exchange. It sets `conversing` for the duration —
        so the explore thread pauses and can't fight over drive_robot with a
        person-requested move mid-conversation — then clears it the instant
        the exchange is done, in a `finally`, so exploring reliably resumes
        on its own. The chat LLM is never relied on to remember to keep
        exploring; it has no way to control it at all.

    Safety note: a human must be at the Driver Station with a hand on
    disable at all times Roam Mode runs — that is the real emergency stop.
    The try/finally below issues a drive_robot("stop") on every exit path as
    a software backstop on top of the robot-side cmd_seq watchdog, but it is
    not a substitute for that human oversight.
    """
    import threading

    print("\n[Roam Mode active…]\n", flush=True)
    _debug_set_status(state="exploring")
    speak(
        f"Roam mode active. I am {personality_name}. "
        "I'll be exploring — say hey Sparky any time to talk to me.",
        out_dev, voice_name, rvc_model,
    )

    # Set for the duration of each conversational exchange so the explore
    # thread pauses driving; cleared immediately afterward so exploring is
    # never left forgotten for longer than that one exchange takes.
    conversing = threading.Event()
    stop_event = threading.Event()

    def _explore_loop() -> None:
        # Own client so exploring never contends with the main conversation
        # thread's HTTP connection pool.
        nav_client = OpenAI(base_url=LLM_BASE_URL, api_key="none")
        last_nav = 0.0
        while not stop_event.wait(0.1):
            if conversing.is_set():
                continue
            if time.time() - last_nav >= ROAM_SCAN_INTERVAL:
                _roam_navigate_step(nav_client)
                last_nav = time.time()

    explorer = threading.Thread(target=_explore_loop, daemon=True)
    explorer.start()

    history: list[dict] = []
    responding        = True    # set False by "stop responding" until "continue responding"
    conversation_mode = False   # set True by "start conversation mode"

    try:
        while True:
            # ── Listen for a prompt ───────────────────────────────────────
            # Conversation mode: listen freely, no wake word required.
            # Wake-word mode: always listening for "hey sparky".
            if conversation_mode:
                audio = record_speech(in_dev, vad, timeout_sec=CONV_LISTEN_TIMEOUT)
                text = transcribe(audio, whisper) if audio is not None else ""
            else:
                inline = wait_for_wake_word(in_dev, vad, whisper)
                if inline is None:
                    text = ""   # wait_for_wake_word has no timeout/abort here, so this shouldn't happen
                else:
                    play_beep(out_dev)
                    if inline:
                        text = inline
                    else:
                        print("[Listening for question…]", end="", flush=True)
                        audio = record_speech(in_dev, vad, timeout_sec=CONV_LISTEN_TIMEOUT)
                        text = transcribe(audio, whisper) if audio is not None else ""

            if not text:
                continue

            # A person is actively engaging — pause exploring for the
            # duration of this exchange and stop any in-progress movement.
            conversing.set()
            _debug_set_status(state="talking")
            try:
                _tool_drive_robot("stop")
                print(f"You: {text}")

                # ── Pause / resume responding ────────────────────────────────
                if _wants_respond_off(text):
                    if responding:
                        responding = False
                        print("[Responses paused — say 'continue responding' to resume]\n", flush=True)
                        speak("Okay, I'll stop responding until you tell me to continue.",
                              out_dev, voice_name, rvc_model)
                    continue

                if _wants_respond_on(text):
                    if not responding:
                        responding = True
                        print("[Responses resumed]\n", flush=True)
                        speak("I'm listening again.", out_dev, voice_name, rvc_model)
                    continue

                if not responding:
                    continue

                # ── Camera calibration ────────────────────────────────────────
                if _wants_calibrate_cameras(text):
                    print("[Camera calibration starting]", flush=True)
                    _debug_set_status(state="calibrating cameras")
                    _voice_calibrate_camera_directions(out_dev, voice_name, rvc_model, in_dev, vad, whisper)
                    continue

                # ── Conversation mode toggle ─────────────────────────────────
                if _wants_conv_on(text) and not conversation_mode:
                    conversation_mode = True
                    print("[Conversation mode ON — speak freely, say 'stop conversation mode' to return to wake word]\n", flush=True)
                    speak("Conversation mode on. I'm listening.", out_dev, voice_name, rvc_model)
                    continue

                if _wants_conv_off(text) and conversation_mode:
                    conversation_mode = False
                    print(f"[Wake word mode — say '{WAKE_PHRASE}' to activate]\n", flush=True)
                    speak("Returning to wake word mode. Say hey Sparky to wake me up.", out_dev, voice_name, rvc_model)
                    continue

                # Explicit farewell — end the session so the next "hey sparky"
                # (from this person or someone new) starts a fresh context
                # instead of dragging along a stale prior conversation.
                if text.lower().strip() in {"quit", "exit", "goodbye", "bye", "stop", "see you"}:
                    _say_in_character(client, system_prompt,
                        "The person just said goodbye to you. Say a warm in-character farewell (1 sentence).",
                        out_dev, voice_name, rvc_model)
                    conversation_mode = False
                    history.clear()
                    continue

                print("[Thinking…]")
                try:
                    llm_reply(client, history, text, out_dev, system_prompt, voice_name, rvc_model,
                              tools=ROAM_CHAT_TOOLS)
                except Exception as exc:
                    print(f"[LLM error] {exc}")
                    speak("Sorry, I had trouble with that.", out_dev, voice_name, rvc_model)
                personality_name, system_prompt, voice_name, rvc_model = _apply_pending_personality(
                    out_dev, personality_name, system_prompt, voice_name, rvc_model, history)
            finally:
                # Whatever just happened, exploring resumes immediately — it
                # is never left waiting on the chat LLM to remember it.
                conversing.clear()
                _debug_set_status(state="exploring")

    except KeyboardInterrupt:
        print("\n[Roam Mode stopped]")
        speak("Goodbye!", out_dev, voice_name, rvc_model)
    finally:
        stop_event.set()
        try:
            _tool_drive_robot("stop")
        except Exception:
            pass


def _roam_navigate_step(client: OpenAI) -> None:
    """Ask the LLM to choose one bounded drive_robot movement based on the
    current camera views, then execute it. Called on its own fixed schedule
    (ROAM_SCAN_INTERVAL) by the explore thread in roam_main, entirely
    independent of whether a conversation is happening — see that function
    for why exploring and conversing are split across two threads.

    Captures a frame from every connected camera (up to CAMERA_MAX_COUNT)
    and sends all of them in one call so the model can reason about its
    full surroundings, not just whatever the primary camera sees — this is
    the main payoff of having multiple cameras for navigation. Each image is
    labeled with _camera_label(), which is only a placeholder ("direction
    unknown") until a real camera-direction mapping is implemented; until
    then the model can use relative cues between images but can't reliably
    turn toward "whichever camera saw the person."

    Uses _EXPLORE_SYSTEM_PROMPT — a minimal, navigation-only prompt, not the
    chatbot's personality prompt — and only exposes ROAM_NAV_TOOLS (just
    drive_robot), so this cheap, frequent decision is never diluted by
    conversational context. If the model doesn't return a tool call (e.g.
    the backend doesn't support forcing one), the robot simply stays put
    for this cycle, which is the safe failure mode.
    """
    frames = _capture_camera_frames(warmup_frames=2)
    if not frames:
        return
    image_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for _, _, b64 in frames
    ]
    labels = ", ".join(_camera_label(idx) for idx, _, _ in frames)
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _EXPLORE_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    *image_content,
                    {"type": "text", "text": (
                        f"These are your current camera views, in this order: {labels}. "
                        "Make ONE drive_robot call."
                    )},
                ]},
            ],
            tools=ROAM_NAV_TOOLS,
            tool_choice="auto",
            max_tokens=100,
            timeout=10.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        msg = resp.choices[0].message
        import json as _json
        for tc in (msg.tool_calls or []):
            if tc.function.name != "drive_robot":
                continue
            try:
                args = _json.loads(tc.function.arguments)
            except Exception:
                args = {}
            print(f"[Explore: drive_robot({args})]", flush=True)
            _tool_drive_robot(args.get("direction", "stop"), args.get("degrees"))
    except Exception as exc:
        print(f"[_roam_navigate_step error] {exc}", file=sys.stderr)


# ── Debug Dashboard ────────────────────────────────────────────────────────────
# While operating the robot, the person at the Driver Station laptop can't
# see this machine's own terminal — only the DS/NT link to the robot. This
# serves a small live-updating page any browser on that laptop can open to
# watch what's happening: what Sparky heard, which tools it ran, which way
# it's driving, what it said, and any errors. It works by mirroring every
# line printed to the real terminal into a ring buffer (see _TeeStream)
# rather than instrumenting each call site individually, so anything already
# printed for the terminal — present or future — shows up here for free.
DEBUG_HTTP_PORT = 8765

_debug_lock   = threading.Lock()
_debug_status: dict = {}


def _local_ip() -> str:
    """Best-effort LAN IP for this machine, for printing the dashboard URL.
    Doesn't actually send any packets — just asks the OS which local address
    it would route through to reach an external host."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return "localhost"


def _debug_set_status(**kwargs) -> None:
    with _debug_lock:
        _debug_status.update(kwargs)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class _TeeStream:
    """Mirrors writes to the real stream into the debug ring buffer. Line-
    buffered so partial `print(..., end="")` writes only show up once their
    line is eventually completed by a later write containing "\\n"."""

    def __init__(self, real, lines: "collections.deque", lock: threading.Lock, prefix: str = ""):
        self._real   = real
        self._lines  = lines
        self._lock   = lock
        self._prefix = prefix
        self._buf    = ""

    def write(self, s: str) -> int:
        self._real.write(s)
        with self._lock:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip("\r")
                if line.strip():
                    self._lines.appendleft(f"{time.strftime('%H:%M:%S')}  {self._prefix}{line}")
        return len(s)

    def flush(self) -> None:
        self._real.flush()

    def isatty(self) -> bool:
        return self._real.isatty()


def start_debug_dashboard() -> None:
    """Start the live debug dashboard: tee stdout/stderr into a ring buffer
    and serve it over HTTP so a browser on the Driver Station laptop can
    watch it live. Safe to call once, as early as possible in main() so
    startup output is captured too."""
    import collections
    import http.server

    lines: "collections.deque[str]" = collections.deque(maxlen=500)
    sys.stdout = _TeeStream(sys.stdout, lines, _debug_lock)
    sys.stderr = _TeeStream(sys.stderr, lines, _debug_lock, prefix="[stderr] ")

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            with _debug_lock:
                status = dict(_debug_status)
                body_lines = list(lines)
            status_html = "".join(
                f"<span class=k>{_esc(k)}</span>=<span class=v>{_esc(str(v))}</span>&nbsp;&nbsp; "
                for k, v in status.items()
            )
            log_html = "\n".join(_esc(l) for l in body_lines)
            html = (
                "<!doctype html><html><head><meta http-equiv=refresh content=2>"
                "<title>Sparky Debug</title><style>"
                "body{background:#111;color:#ddd;font-family:monospace;padding:1em}"
                "#status{background:#222;padding:.6em 1em;margin-bottom:1em;border-radius:6px}"
                ".k{color:#7fd7ff}.v{color:#fff;font-weight:bold}"
                "pre{white-space:pre-wrap;word-break:break-word}"
                "</style></head><body>"
                "<h2>Sparky Debug Dashboard</h2>"
                f"<div id=status>{status_html or 'starting…'}</div>"
                f"<pre>{log_html}</pre>"
                "</body></html>"
            )
            payload = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args) -> None:
            pass   # don't feed our own HTTP access log back into the tee

    def _serve() -> None:
        try:
            server = http.server.ThreadingHTTPServer(("0.0.0.0", DEBUG_HTTP_PORT), _Handler)
            server.serve_forever()
        except Exception as exc:
            print(f"[debug dashboard] failed to start: {exc}", file=sys.stderr)

    threading.Thread(target=_serve, daemon=True).start()

    # The Driver Station laptop is on WiFi and this machine is on Ethernet,
    # both bridged through the robot's radio — so this machine's IP isn't
    # necessarily known or stable from the DS laptop's side. Advertise over
    # mDNS (already used for "robot.local"-style FRC hostnames, so DS laptops
    # are already set up to resolve ".local" names) instead of relying on a
    # raw IP the operator would have to look up.
    import socket
    hostname = socket.gethostname()
    _publish_debug_service()
    print(
        f"Debug  → http://{hostname}.local:{DEBUG_HTTP_PORT}/ "
        f"(or http://{_local_ip()}:{DEBUG_HTTP_PORT}/ if .local doesn't resolve) "
        "— open on the Driver Station laptop"
    )


def _publish_debug_service() -> None:
    """Best-effort mDNS/DNS-SD announcement so the dashboard is discoverable
    by name (e.g. via any Bonjour/zeroconf browser) without even needing the
    hostname. Purely additive — if avahi isn't installed, the .local
    hostname printed by start_debug_dashboard still works on its own."""
    import shutil
    import subprocess
    if shutil.which("avahi-publish") is None:
        return
    try:
        subprocess.Popen(
            ["avahi-publish", "-s", "Sparky Debug Dashboard", "_http._tcp", str(DEBUG_HTTP_PORT)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    start_debug_dashboard()

    print("=== Voice AI Chatbot ===")
    print(f"LLM : {LLM_MODEL}")
    print(f"URL : {LLM_BASE_URL}\n")

    import argparse
    _early_parser = argparse.ArgumentParser(add_help=False)
    _early_parser.add_argument("--calibrate-cameras", action="store_true")
    _early_args, _ = _early_parser.parse_known_args()
    if _early_args.calibrate_cameras:
        # Camera-only setup: skip audio device selection, sample-rate
        # detection, and model loading entirely — none of that is needed
        # just to assign direction labels to whatever cameras are plugged in.
        calibrate_camera_directions()
        return

    # Show available devices
    print("Audio devices:")
    for i, dev in enumerate(sd.query_devices()):
        tags = ("" if dev["max_input_channels"]  == 0 else " [in]") + \
               ("" if dev["max_output_channels"] == 0 else " [out]")
        print(f"  {i:2d}: {dev['name']}{tags}")
    print()

    in_dev,  in_name  = _select_input_device()
    out_dev, out_name = _select_output_device()
    camera_indices    = _find_camera_indices()

    # Set sample rates to match the actual hardware devices.
    global SAMPLE_RATE, FRAME_SAMPLES, OUTPUT_SAMPLE_RATE
    SAMPLE_RATE        = _detect_sample_rate(in_dev)
    FRAME_SAMPLES      = int(SAMPLE_RATE * FRAME_MS / 1000)
    OUTPUT_SAMPLE_RATE = int(sd.query_devices(out_dev)["default_samplerate"]) if out_dev is not None else 22_050

    print(f"Mic    → {in_name}")
    print(f"Output → {out_name}")
    if camera_indices:
        print(f"Camera → {len(camera_indices)} detected (enabled): "
              f"{', '.join(_camera_label(i) for i in camera_indices)}")
    else:
        print("Camera → not detected (camera features disabled)")

    parser = argparse.ArgumentParser(description="Voice AI Chatbot")
    parser.add_argument(
        "--mode", choices=["standard", "roam"], default="roam",
        help="Operating mode (default: roam)",
    )
    parser.add_argument(
        "--calibrate-cameras", action="store_true",
        help="Interactively assign a direction label to each connected camera, then exit.",
    )
    args, _ = parser.parse_known_args()
    mode = args.mode
    print(f"Mode   → {mode}")
    if mode == "roam":
        print(f"Robot  → {ROBOT_NT_HOST} (NetworkTables, connects lazily on first drive_robot call)")

    # Default personality: Sparky — skip interactive prompt
    p = PERSONALITIES["Sparky"]
    personality_name, system_prompt, voice_name, rvc_model = (
        "Sparky", p["prompt"], p["voice"], p.get("rvc"),
    )
    _debug_set_status(mode=mode, personality=personality_name, state="starting")

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

    # Make these available to the background "stop" interrupt listener.
    global _in_dev, _vad, _whisper
    _in_dev, _vad, _whisper = in_dev, vad, whisper

    if mode == "roam":
        roam_main(client, whisper, vad, in_dev, out_dev,
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


        print("[Thinking…]")
        try:
            llm_reply(client, history, text, out_dev, system_prompt, voice_name, rvc_model)
        except Exception as exc:
            print(f"[LLM error] {exc}")
            speak("Sorry, I had trouble reaching the language model.", out_dev)
        personality_name, system_prompt, voice_name, rvc_model = _apply_pending_personality(
            out_dev, personality_name, system_prompt, voice_name, rvc_model, history)


if __name__ == "__main__":
    main()
