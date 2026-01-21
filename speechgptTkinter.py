# ============================================================
# Realtime Speech Transcription – Desktop App (Python)
#
# Prerequisites (run once):
#   pip install -U openai sounddevice numpy
#   OR
#   python -m pip install -U openai sounddevice numpy
#
# What this app does:
# - Shows a desktop window (Tkinter).
# - Lets you choose a microphone from a combobox.
# - Streams microphone audio in chunks.
# - Sends chunks to OpenAI Speech-to-Text (API).
# - Continuously appends transcription to the text area.
# - Optionally writes transcript to:
#     C:\Users\numan\OneDrive\Masaüstü\speech.txt
#
# Notes:
# - The output file is cleared each time you press "Start".
# - Silence is skipped using an RMS threshold.
# - Stop recording with the "Stop" button or by closing the window.
# - Tkinter is included with Python (no extra GUI installs).
#
# ============================================================

import os
import sys
import time
import queue
import threading
import io
import wave

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox

from openai import OpenAI

# ----------------------------
# API KEY (set this)
# ----------------------------
OPENAI_API_KEY = "sk-proj-xxx"

# ----------------------------
# Model choice
# ----------------------------
TRANSCRIBE_MODEL = "gpt-4o-transcribe"

# ----------------------------
# Output file (optional)
# ----------------------------
OUT_PATH = r"C:\Users\numan\OneDrive\Masaüstü\speech.txt"
WRITE_TO_FILE = True  # set False if you only want the textarea

# ----------------------------
# Audio settings
# ----------------------------
SAMPLE_RATE = 16000
CHANNELS = 1

CHUNK_SEC = 4.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)

LANGUAGE = "en"  # "tr", "en", or None
RMS_THRESHOLD = 0.01  # tune (0.005–0.02)

# Thread-safe queues
audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
ui_q: "queue.Queue[str]" = queue.Queue()

stop_event = threading.Event()
    
def ensure_output_file_cleared(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def append_file(path: str, text: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + " ")


def float32_to_wav_bytes(audio_f32: np.ndarray, sample_rate: int) -> bytes:
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (audio_f32 * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def transcribe_chunk(client: OpenAI, wav_bytes: bytes) -> str:
    kwargs = {
        "model": TRANSCRIBE_MODEL,
        "file": ("chunk.wav", wav_bytes, "audio/wav"),
        "response_format": "text",
    }
    if LANGUAGE:
        kwargs["language"] = LANGUAGE

    try:
        result = client.audio.transcriptions.create(**kwargs)
        return str(result).strip()
    except Exception as e:
        ui_q.put(f"\n⚠️ Transcription error: {e}\n")
        return ""


def mic_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    if status:
        ui_q.put(f"\n⚠️ Audio status: {status}\n")
    # Copy out of callback thread
    audio_q.put(indata.copy())


def list_input_devices():
    devices = sd.query_devices()
    inputs = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            hostapi_name = sd.query_hostapis(d["hostapi"])["name"]
            label = f"{i:>2}  {d['name']} [{hostapi_name}] (in_ch={d['max_input_channels']})"
            inputs.append((i, label))
    return inputs


def worker_loop(device_index: int | None, api_key: str):
    """
    Background thread:
    - opens input stream
    - accumulates chunks
    - sends to OpenAI
    - pushes text to ui_q
    """
    client = OpenAI(api_key=api_key)

    # Clear output file at start (like your script)
    if WRITE_TO_FILE:
        ensure_output_file_cleared(OUT_PATH)

    buf = np.zeros((0, CHANNELS), dtype=np.float32)

    ui_q.put(f"🎙️ Started. Model={TRANSCRIBE_MODEL}, chunk_sec={CHUNK_SEC}, lang={LANGUAGE}\n")
    if WRITE_TO_FILE:
        ui_q.put(f"📝 Writing to: {OUT_PATH}\n")

    try:
        with sd.InputStream(
            device=device_index,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=mic_callback,
        ):
            while not stop_event.is_set():
                try:
                    block = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                buf = np.concatenate([buf, block], axis=0)

                while len(buf) >= CHUNK_SAMPLES and not stop_event.is_set():
                    chunk = buf[:CHUNK_SAMPLES]
                    buf = buf[CHUNK_SAMPLES:]

                    audio_1d = chunk.squeeze(axis=1)
                    rms = float(np.sqrt(np.mean(audio_1d ** 2)))
                    if rms < RMS_THRESHOLD:
                        continue

                    wav_bytes = float32_to_wav_bytes(audio_1d, SAMPLE_RATE)
                    text = transcribe_chunk(client, wav_bytes)

                    if text:
                        ui_q.put(text + " ")
                        if WRITE_TO_FILE:
                            append_file(OUT_PATH, text)

                time.sleep(0.001)

    except Exception as e:
        ui_q.put(f"\n❌ Worker crashed: {e}\n")

    ui_q.put("\n🛑 Stopped.\n")

def center_window(win: tk.Tk, width: int, height: int) -> None:
    win.update_idletasks()
    screen_w = win.winfo_screenwidth()
    screen_h = win.winfo_screenheight()
    x = (screen_w - width) // 2
    y = (screen_h - height) // 2
    win.geometry(f"{width}x{height}+{x}+{y}")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Realtime Transcribe (OpenAI)")
        center_window(self, 1280, 600)

        self.worker_thread: threading.Thread | None = None
        self.device_map: dict[str, int | None] = {}

        # Top bar
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(top, text="Microphone:").pack(side="left")

        self.device_combo = ttk.Combobox(top, state="readonly", width=70)
        self.device_combo.pack(side="left", padx=8)

        self.refresh_btn = ttk.Button(top, text="Refresh", command=self.refresh_devices)
        self.refresh_btn.pack(side="left", padx=6)

        self.start_btn = ttk.Button(top, text="Start", command=self.start)
        self.start_btn.pack(side="left", padx=6)

        self.stop_btn = ttk.Button(top, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)

        # Text area
        self.text = tk.Text(self, wrap="word")
        self.text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Init devices + queue polling
        self.refresh_devices()
        self.after(80, self.drain_ui_queue)

        # Close handling
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def ui_append(self, msg: str) -> None:
        self.text.insert("end", msg)
        self.text.see("end")
    
    def refresh_devices(self):
        inputs = list_input_devices()

        labels = ["(Default Input Device)"]
        self.device_map = {"(Default Input Device)": None}

        for idx, label in inputs:
            labels.append(label)
            self.device_map[label] = idx

        self.device_combo["values"] = labels
        self.device_combo.current(0)

    def start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return

        # Clear UI text each start? (optional)
        self.text.delete("1.0", "end")

        stop_event.clear()

        selected_label = self.device_combo.get()
        device_idx = self.device_map.get(selected_label, None)
        
        if device_idx is None:
            self.ui_append("🎛️ Input device: DEFAULT (OS default)\n")

            idx, name, hostapi = self.get_default_input_device_info()
            if name:
                self.ui_append(f"   Default mic: index={idx} name='{name}' [{hostapi}]\n")
            else:
                self.ui_append("⚠️ Could not resolve default input device details.\n")

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.refresh_btn.config(state="disabled")
        self.device_combo.config(state="disabled")

        self.worker_thread = threading.Thread(
            target=worker_loop,
            args=(device_idx, OPENAI_API_KEY),
            daemon=True,
        )
        self.worker_thread.start()
        
    def get_default_input_device_info(self):
        """
        Returns (index, name, hostapi) for the system default input device,
        or (None, None, None) if it can't be resolved.
        """
        try:
            d = sd.query_devices(None, "input")  # <--- THIS is the key
            idx = d.get("index", None)           # often present
            hostapi = sd.query_hostapis(d["hostapi"])["name"] if "hostapi" in d else "?"
            name = d.get("name", "?")
            return idx, name, hostapi
        except Exception:
            return None, None, None

    def stop(self):
        stop_event.set()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.refresh_btn.config(state="normal")
        self.device_combo.config(state="readonly")

    def on_close(self):
        stop_event.set()
        self.destroy()

    def drain_ui_queue(self):
        # Pull any pending text from worker and append to text widget
        try:
            while True:
                msg = ui_q.get_nowait()
                self.ui_append(msg)
        except queue.Empty:
            pass
        self.after(80, self.drain_ui_queue)


if __name__ == "__main__":
    # Quick sanity: show Python + sounddevice default
    try:
        App().mainloop()
    except KeyboardInterrupt:
        pass
