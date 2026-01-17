# ============================================================
# Prerequisites (run once):
#   pip install -U faster-whisper sounddevice numpy
#   OR
#   python -m pip install -U faster-whisper sounddevice numpy
#   
#   https://developer.nvidia.com/cuda-12-9-1-download-archive
#
# Live view the output file (PowerShell):
# Get-Content "C:\Users\numan\OneDrive\Masaüstü\speech.txt" -Wait
#
# Notes:
# - Lets you choose microphone input device.
# - Writes transcript to: C:\Users\numan\OneDrive\Masaüstü\speech.txt
# - File is cleared on start.
# - Stop with Ctrl+C.
# ============================================================

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Make CUDA DLLs visible to this Python process (Windows)
# (Keeps you from needing to mess with PATH.)
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin")

# Optional sanity check (can be commented out once you're confident)
import ctypes
ctypes.CDLL("cublas64_12.dll")

import sys
import time
import queue
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ------------------------------------------------------------
# ONE-LINE SWITCH:
#   - Use "4GB" for 4 GB VRAM machines (realtime-friendly)
#   - Use "8GB" for 8 GB VRAM machines (better quality)
#   - Use "12GB" for 12 GB VRAM machines (high quality)
#   - Use "16GB" for 16 GB VRAM machines (best quality)
# ------------------------------------------------------------
PRESET = "4GB"

# ------------------------------------------------------------
# Output text file path (Windows)
# ------------------------------------------------------------
OUT_PATH = r"C:\Users\numan\OneDrive\Masaüstü\speech.txt"

# ------------------------------------------------------------
# Audio settings (16 kHz mono is ideal for Whisper)
# ------------------------------------------------------------
SAMPLE_RATE = 16000
CHANNELS = 1

# ------------------------------------------------------------
# Choose your spoken language to avoid auto-detect overhead.
# Set to "en" or "tr". If you really want auto, set None.
# ------------------------------------------------------------
LANGUAGE = "en"  # or "tr" (IMPORTANT: must be a string like "en")

# ------------------------------------------------------------
# Model presets (model + compute)
# ------------------------------------------------------------
PRESETS = {
    "4GB": {
        "model_name": "large-v3-turbo",
        "device": "cuda",
        "compute_type": "float16",      # or "int8_float16" if you want more headroom
        "chunk_sec": 2.0,
        "beam_size": 3,
        "vad_filter": True,
    },
    "8GB": {
        "model_name": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
        "chunk_sec": 2.0,               # more context beats beam inflation
        "beam_size": 5,
        "vad_filter": True,
    },
    "12GB": {
        "model_name": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
        "chunk_sec": 2.0,
        "beam_size": 5,                 # 6 if you want, but 5 is the sweet spot
        "vad_filter": True,
    },
    "16GB": {
        "model_name": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
        "chunk_sec": 2.0,               # if latency truly isn't important
        "beam_size": 5,
        "vad_filter": True,
    },
}

# Queue for microphone audio frames coming from the callback thread
audio_q: "queue.Queue[np.ndarray]" = queue.Queue()


def ensure_output_file_cleared(path: str) -> None:
    """Create directories if needed and clear the output file on startup."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    # Empty the file at start
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def append_file(path: str, line: str) -> None:
    """Append a single line to the transcript file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + " ")


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

def choose_vram_preset(default_preset: str = "4GB") -> str:
    """
    Ask user to choose a VRAM preset (4/8/12/16).
    Default is 4GB and is shown with a '*' marker.
    Returns a preset key like "4GB".
    """
    order = ["4GB", "8GB", "12GB", "16GB"]

    print("Choose VRAM preset:")
    for p in order:
        star = " *" if p == default_preset else ""
        print(f"  {p.replace('GB',''):>2}{star}")

    choice = input(f"VRAM (Enter = {default_preset.replace('GB','')}): ").strip()

    if choice == "":
        return default_preset

    # Accept "4" or "4GB" etc.
    normalized = choice.upper().replace(" ", "")
    if normalized.isdigit():
        normalized = normalized + "GB"

    if normalized in PRESETS:
        return normalized

    print("Invalid choice. Using default preset.")
    return default_preset

def main() -> None:
    """Main realtime transcription loop."""
    selected_preset = choose_vram_preset(default_preset=PRESET)

    if selected_preset not in PRESETS:
        raise ValueError(f"Unknown PRESET='{selected_preset}'. Use one of: {list(PRESETS.keys())}")

    cfg = PRESETS[selected_preset]

    # Per-preset chunk sizing
    chunk_sec = float(cfg["chunk_sec"])
    chunk_samples = int(SAMPLE_RATE * chunk_sec)

    # Clear output file on startup
    ensure_output_file_cleared(OUT_PATH)

    # Let user pick mic
    selected_input_device = choose_input_device()

    # Load Whisper model
    model = WhisperModel(
        cfg["model_name"],
        device=cfg["device"],
        compute_type=cfg["compute_type"],
    )

    # Buffer to accumulate microphone samples until we have a chunk
    buf = np.zeros((0, CHANNELS), dtype=np.float32)

    print(f"\n🎙️ Realtime transcription started. PRESET={selected_preset} GB")
    print(f"Model: {cfg['model_name']} | device={cfg['device']} | compute_type={cfg['compute_type']}")
    print(f"chunk_sec={chunk_sec} | beam_size={cfg['beam_size']} | language={LANGUAGE} | vad_filter={cfg['vad_filter']}")
    print(f"Writing to: {OUT_PATH}")
    print("Press Ctrl+C to stop.\n")

    # Open microphone stream
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

            while len(buf) >= chunk_samples:
                chunk = buf[:chunk_samples]
                buf = buf[chunk_samples:]  # keep remainder

                # Convert shape (N,1) -> (N,)
                audio_1d = chunk.squeeze(axis=1)

                # Transcribe this chunk
                segments, info = model.transcribe(
                    audio_1d,
                    beam_size=int(cfg["beam_size"]),
                    language=LANGUAGE,
                    vad_filter=bool(cfg["vad_filter"]),
                )

                text = "".join(seg.text for seg in segments).strip()
                if text:
                    print(text, flush=True)
                    append_file(OUT_PATH, text)

            time.sleep(0.001)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
