# ============================================================
# Prerequisites (run once):
#   pip install -U openai sounddevice numpy
#   OR
#   python -m pip install -U openai sounddevice numpy
# 
# Live view the output file (PowerShell):
#   Get-Content "C:\Users\numan\OneDrive\Masaüstü\speech.txt" -Wait
#
# Notes:
# - Lets you choose microphone input device.
# - Writes transcript to: C:\Users\numan\OneDrive\Masaüstü\speech.txt
# - File is cleared on start.
# - Stop with Ctrl+C.
# ============================================================

import os
import sys
import time
import queue
import io
import wave
import numpy as np
import sounddevice as sd

from openai import OpenAI

# ----------------------------
# API KEY (set this)
# ----------------------------
OPENAI_API_KEY = "sk-proj-xxx"

# ----------------------------
# Model choice
# ----------------------------
# OpenAI Speech-to-Text models include: gpt-4o-transcribe, gpt-4o-mini-transcribe, whisper-1, etc. :contentReference[oaicite:1]{index=1}
TRANSCRIBE_MODEL = "gpt-4o-transcribe"

# ----------------------------
# Output file
# ----------------------------
OUT_PATH = r"C:\Users\numan\OneDrive\Masaüstü\speech.txt"

# ----------------------------
# Audio settings
# ----------------------------
SAMPLE_RATE = 16000
CHANNELS = 1

# Chunk size: bigger = more context, less “works/rocks” randomness, but more latency & cost per request.
CHUNK_SEC = 4.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)

# Lock language to reduce weirdness and save work (set to "en" or "tr", or None for auto)
LANGUAGE = "en"

# Queue for microphone audio frames coming from the callback thread
audio_q: "queue.Queue[np.ndarray]" = queue.Queue()


def ensure_output_file_cleared(path: str) -> None:
    """Create directories if needed and clear the output file on startup."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def append_file(path: str, text: str) -> None:
    """Append text to the transcript file (space-separated)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + " ")


def mic_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    """Sounddevice callback: receives audio blocks and pushes them to a queue."""
    if status:
        print(status, file=sys.stderr)
    audio_q.put(indata.copy())


def choose_input_device() -> int | None:
    """
    List only input-capable devices and let the user pick one.
    Returns:
      - device index (int) if user chooses one
      - None if user presses Enter (use OS default)
    """
    devices = sd.query_devices()

    inputs: list[tuple[int, str, str, int]] = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            hostapi_name = sd.query_hostapis(d["hostapi"])["name"]
            inputs.append((i, d["name"], hostapi_name, d["max_input_channels"]))

    print("Available INPUT devices (microphones):")
    for i, name, hostapi, ch in inputs:
        print(f"  {i:>2}  {name} [{hostapi}] (in_ch={ch})")

    # sd.default.device is typically [input_index, output_index]
    default_in = None
    if isinstance(sd.default.device, (list, tuple)) and len(sd.default.device) >= 1:
        default_in = sd.default.device[0]

    print(f"Default input device index: {default_in}")
    print("-" * 60)

    choice = input("Choose input device index (Enter = default): ").strip()
    if choice == "":
        return None

    try:
        idx = int(choice)
    except ValueError:
        print("Invalid number. Using default input device.")
        return None

    if idx < 0 or idx >= len(devices) or devices[idx].get("max_input_channels", 0) <= 0:
        print("Invalid input device index. Using default input device.")
        return None

    return idx


def float32_to_wav_bytes(audio_f32: np.ndarray, sample_rate: int) -> bytes:
    """
    Convert float32 mono audio [-1..1] into WAV (PCM16) bytes in memory.
    OpenAI transcription endpoint accepts WAV. :contentReference[oaicite:2]{index=2}
    """
    # Clamp and convert to int16 PCM
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (audio_f32 * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def transcribe_chunk(client: OpenAI, wav_bytes: bytes) -> str:
    """
    Send a single WAV chunk to OpenAI and return recognized text.
    Uses /v1/audio/transcriptions with gpt-4o-transcribe. :contentReference[oaicite:3]{index=3}
    """
    # The OpenAI Python SDK accepts file-like uploads; we pass a (filename, bytes, mimetype) tuple.
    # response_format="text" returns plain text. :contentReference[oaicite:4]{index=4}
    kwargs = {
        "model": TRANSCRIBE_MODEL,
        "file": ("chunk.wav", wav_bytes, "audio/wav"),
        "response_format": "text",
    }
    if LANGUAGE:
        kwargs["language"] = LANGUAGE

    try:
        result = client.audio.transcriptions.create(**kwargs)
        # For response_format="text", result is typically a string-like object; str() is safe.
        return str(result).strip()
    except Exception as e:
        # Don’t crash the whole loop on one transient failure
        print(f"\n⚠️ Transcription error: {e}", file=sys.stderr)
        return ""


def main() -> None:
    if not OPENAI_API_KEY:
        print("ERROR: Set OPENAI_API_KEY at the top of the file.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    ensure_output_file_cleared(OUT_PATH)
    selected_input_device = choose_input_device()

    # Buffer to accumulate microphone samples until we have a chunk
    buf = np.zeros((0, CHANNELS), dtype=np.float32)

    print(f"\n🎙️ Realtime transcription started (API). Model={TRANSCRIBE_MODEL}")
    print(f"chunk_sec={CHUNK_SEC} | language={LANGUAGE} | sample_rate={SAMPLE_RATE}")
    print(f"Writing to: {OUT_PATH}")
    print("Press Ctrl+C to stop.\n")

    with sd.InputStream(
        device=selected_input_device,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=mic_callback,
    ):
        while True:
            block = audio_q.get()
            buf = np.concatenate([buf, block], axis=0)

            while len(buf) >= CHUNK_SAMPLES:
                chunk = buf[:CHUNK_SAMPLES]
                buf = buf[CHUNK_SAMPLES:]  # keep remainder

                audio_1d = chunk.squeeze(axis=1)
                rms = np.sqrt(np.mean(audio_1d ** 2))
                if rms < 0.01:   # tune this (0.005–0.02)
                    continue    # skip this chunk entirely
                wav_bytes = float32_to_wav_bytes(audio_1d, SAMPLE_RATE)
                text = transcribe_chunk(client, wav_bytes)

                if text:
                    print(text, flush=True)
                    append_file(OUT_PATH, text)

            time.sleep(0.001)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
