#!/usr/bin/env python3
"""Remove background music: isolate the voice with an MDX-Net ONNX model (Kim_Vocal_2).

Light-weight (onnxruntime + numpy, no torch). Used automatically by extract_voice.py
when background music is detected. Can also be run standalone:

  python separate_vocals.py INPUT.(wav|mp4) OUTPUT_vocals.wav [--model PATH]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, log, run  # noqa: E402

MDX_MODEL = CACHE_DIR / "Kim_Vocal_2.onnx"
MDX_URL = ("https://github.com/TRvlvr/model_repo/releases/download/"
           "all_public_uvr_models/Kim_Vocal_2.onnx")

# Kim_Vocal_2 parameters (from the UVR model data table)
N_FFT, HOP, DIM_F, DIM_T = 7680, 1024, 3072, 256
COMPENSATE = 1.009
SR = 44100


def find_model(explicit: str | None = None) -> Path | None:
    for c in [explicit, MDX_MODEL]:
        if c and Path(c).is_file():
            return Path(c)
    return None


def _stft(x: np.ndarray) -> np.ndarray:
    """x: [B, 2, L] -> model input [B, 4, DIM_F, DIM_T]."""
    import librosa
    B = x.shape[0]
    spec = librosa.stft(x.reshape(-1, x.shape[-1]), n_fft=N_FFT, hop_length=HOP,
                        window="hann", center=True, pad_mode="reflect")  # [B*2, F, T]
    ri = np.stack([spec.real, spec.imag], axis=1)  # [B*2, 2, F, T]
    ri = ri.reshape(B, 4, ri.shape[-2], ri.shape[-1])
    return ri[:, :, :DIM_F].astype(np.float32)


def _istft(s: np.ndarray, length: int) -> np.ndarray:
    import librosa
    B = s.shape[0]
    n_bins = N_FFT // 2 + 1
    pad = np.zeros((B, 4, n_bins - DIM_F, s.shape[-1]), np.float32)
    s = np.concatenate([s, pad], axis=2).reshape(B * 2, 2, n_bins, s.shape[-1])
    spec = s[:, 0] + 1j * s[:, 1]
    wav = librosa.istft(spec, hop_length=HOP, window="hann", center=True, length=length)
    return wav.reshape(B, 2, -1)


def separate_array(mix: np.ndarray, model: Path) -> np.ndarray:
    """mix: [2, L] float32 at 44.1 kHz -> vocals [2, L]."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    chunk = HOP * (DIM_T - 1)
    trim = N_FFT // 2
    gen = chunk - 2 * trim
    n = mix.shape[1]
    pad = gen - n % gen
    mp = np.concatenate([np.zeros((2, trim)), mix, np.zeros((2, pad)), np.zeros((2, trim))], 1)
    outs = []
    total = (n + pad) // gen
    for k, i in enumerate(range(0, n + pad, gen)):
        w = mp[:, i:i + chunk][None].astype(np.float32)
        spec = sess.run(None, {"input": _stft(w)})[0]
        wav = _istft(spec, chunk)[0][:, trim:-trim]
        outs.append(wav)
        if k % 5 == 0:
            log(f"   separasi vokal {k + 1}/{total}")
    voc = np.concatenate(outs, 1)[:, :n] * COMPENSATE
    return voc.astype(np.float32)


def separate_file(src: Path, dst: Path, model: Path) -> None:
    tmp = Path(tempfile.mkdtemp()) / "stereo.wav"
    run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", str(SR),
         "-c:a", "pcm_f32le", str(tmp)])
    mix, _ = sf.read(str(tmp), dtype="float32", always_2d=True)
    voc = separate_array(mix.T, model)
    mono = voc.mean(0)
    peak = float(np.max(np.abs(mono)) + 1e-9)
    if peak > 0.99:
        mono = mono / peak * 0.98
    sf.write(str(dst), mono, SR, subtype="PCM_16")
    tmp.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    model = find_model(a.model)
    if not model:
        sys.exit(f"Model tidak ditemukan. Jalankan setup_env.py --separator (atau download {MDX_URL})")
    separate_file(Path(a.input), Path(a.output), model)
    print("OK", a.output)


if __name__ == "__main__":
    main()
