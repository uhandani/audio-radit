"""Shared helpers for the voice-template-extractor scripts."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CACHE_DIR = Path(os.environ.get("VOICE_TEMPLATE_CACHE", Path.home() / ".cache" / "voice-template"))
VAD_MODEL = CACHE_DIR / "silero_vad.onnx"
SHERPA_WHISPER_DIR = CACHE_DIR / "sherpa-onnx-whisper-small"


def log(msg: str) -> None:
    print(f"[voice-template] {msg}", file=sys.stderr, flush=True)


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg/ffprobe tidak ditemukan. Install dulu (apt install ffmpeg / brew install ffmpeg).")


def run(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=capture)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def find_vad_model(explicit: str | None = None) -> Path | None:
    for cand in [explicit, os.environ.get("VOICE_VAD_MODEL"), VAD_MODEL]:
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


def silero_segments(samples, sr: int, model: Path, min_silence: float = 0.25,
                    max_speech: float = 20.0):
    """Run Silero VAD (via sherpa-onnx). samples must be float32 mono at 16 kHz.
    Returns list of (start_sec, end_sec)."""
    import sherpa_onnx

    assert sr == 16000, "Silero VAD expects 16 kHz"
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(model)
    cfg.silero_vad.threshold = 0.45
    cfg.silero_vad.min_silence_duration = min_silence
    cfg.silero_vad.min_speech_duration = 0.2
    cfg.silero_vad.max_speech_duration = max_speech
    cfg.sample_rate = sr
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=int(len(samples) / sr) + 30)
    win = cfg.silero_vad.window_size
    out = []

    def drain():
        while not vad.empty():
            seg = vad.front
            out.append((seg.start / sr, (seg.start + len(seg.samples)) / sr))
            vad.pop()

    for i in range(0, len(samples), win):
        vad.accept_waveform(samples[i:i + win])
        drain()
    vad.flush()
    drain()
    return out


def energy_segments(samples, sr: int, top_db: float = 35, min_gap: float = 0.25):
    """Fallback VAD: energy-based split, merging gaps shorter than min_gap."""
    import librosa

    iv = librosa.effects.split(samples, top_db=top_db, frame_length=1024, hop_length=256)
    segs = []
    for s, e in iv:
        s, e = s / sr, e / sr
        if e - s < 0.15:
            continue
        if segs and s - segs[-1][1] < min_gap:
            segs[-1] = (segs[-1][0], e)
        else:
            segs.append((s, e))
    return segs


def split_long_segments(samples, sr: int, segs, max_len: float = 9.0, min_len: float = 3.0):
    """Split segments longer than max_len at the quietest 150 ms point so every
    segment is roughly one spoken phrase (better transcripts and per-phrase prosody)."""
    import numpy as np

    hop = int(0.01 * sr)
    out = []
    stack = list(segs)[::-1]
    while stack:
        s, e = stack.pop()
        if e - s <= max_len:
            out.append((s, e))
            continue
        a, b = int((s + min_len) * sr), int(min(e - min_len, s + max_len) * sr)
        x = samples[a:b]
        if len(x) < hop * 20:
            out.append((s, e))
            continue
        frames = len(x) // hop
        rms = np.sqrt(np.mean(x[:frames * hop].reshape(frames, hop) ** 2, axis=1))
        win = 15
        smooth = np.convolve(rms, np.ones(win) / win, mode="same")
        cut = s + min_len + int(np.argmin(smooth)) * hop / sr
        stack.append((cut, e))
        stack.append((s, cut))
    return out
