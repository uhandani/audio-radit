#!/usr/bin/env python3
"""Check/install dependencies and download small models used by the pipeline.

Usage:
  python setup_env.py                 # check + install python deps + VAD model
  python setup_env.py --separator     # also fetch vocal/music separator model (~65 MB)
  python setup_env.py --tts           # offline Indonesian TTS engine (speak.py): own venv with
                                      # torch + Coqui TTS + 83-voice Indonesian VITS (~1-6 GB)
  python setup_env.py --sherpa-whisper  # also fetch Whisper-small ONNX (GitHub mirror,
                                        # for machines where HuggingFace is blocked)

Everything is downloaded into ~/.cache/voice-template (override with VOICE_TEMPLATE_CACHE).
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, SHERPA_WHISPER_DIR, VAD_MODEL, log, require_ffmpeg  # noqa: E402
from separate_vocals import MDX_MODEL, MDX_URL  # noqa: E402

PY_DEPS = {  # import name -> pip name
    "numpy": "numpy",
    "scipy": "scipy",
    "soundfile": "soundfile",
    "librosa": "librosa",
    "parselmouth": "praat-parselmouth",
    "sherpa_onnx": "sherpa-onnx",
    "matplotlib": "matplotlib",
    "onnxruntime": "onnxruntime",
}
OPTIONAL = {"faster_whisper": "faster-whisper", "demucs": "demucs"}

GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"


def ensure_pkgs(pkgs: dict[str, str]) -> list[str]:
    missing = []
    for mod, pipname in pkgs.items():
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pipname)
    if missing:
        log(f"pip install {' '.join(missing)}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)
    still = []
    for mod, pipname in pkgs.items():
        try:
            importlib.import_module(mod)
        except Exception:
            still.append(pipname)
    return still


def download(url: str, dest: Path) -> None:
    log(f"download {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


TTS_DIR = CACHE_DIR / "id-tts-vits"
TTS_URL = "https://github.com/Wikidepia/indonesian-tts/releases/download/v1.2"
TTS_FILES = ["config.json", "speakers.pth", "checkpoint_1260000-inference.pth"]
TTS_PKGS = ["coqui-tts", "torchcodec", "six", "g2p-id-py", "praat-parselmouth", "sherpa-onnx",
            "soundfile", "librosa"]


def setup_tts() -> None:
    """Separate venv: torch/Coqui pin many packages that would clash with the main env."""
    venv = CACHE_DIR / "venv-tts"
    py = venv / "bin" / "python"
    if not py.exists():
        log("membuat venv TTS")
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        subprocess.run([str(py), "-m", "pip", "install", "-q", "-U", "pip", "setuptools", "wheel"], check=True)
    ok = subprocess.run([str(py), "-c", "import torch, torchaudio"], capture_output=True).returncode == 0
    if not ok:
        log("install torch (CPU) — bisa beberapa menit")
        r = subprocess.run([str(py), "-m", "pip", "install", "-q", "torch", "torchaudio",
                            "--index-url", "https://download.pytorch.org/whl/cpu"])
        if r.returncode != 0:  # CPU index unreachable -> default PyPI wheels (bigger)
            subprocess.run([str(py), "-m", "pip", "install", "-q", "torch", "torchaudio"], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "-q", *TTS_PKGS], check=True)
    for f in TTS_FILES:
        if not (TTS_DIR / f).exists():
            download(f"{TTS_URL}/{f}", TTS_DIR / f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sherpa-whisper", action="store_true",
                    help="download Whisper-small ONNX from GitHub (fallback transcriber)")
    ap.add_argument("--separator", action="store_true",
                    help="download MDX-Net vocal model (~65 MB) to strip background music")
    ap.add_argument("--tts", action="store_true",
                    help="install the offline TTS engine used by speak.py (separate venv)")
    ap.add_argument("--with-faster-whisper", action="store_true",
                    help="pip install faster-whisper (best transcriber, needs HuggingFace access)")
    ap.add_argument("--with-demucs", action="store_true",
                    help="pip install demucs (vocal/music separation, ~1 GB with torch)")
    args = ap.parse_args()

    require_ffmpeg()
    bad = ensure_pkgs(PY_DEPS)
    if bad:
        sys.exit(f"Gagal install: {bad}")
    if args.with_faster_whisper:
        ensure_pkgs({"faster_whisper": "faster-whisper"})
    if args.with_demucs:
        ensure_pkgs({"demucs": "demucs"})

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not VAD_MODEL.exists():
        download(f"{GH}/silero_vad.onnx", VAD_MODEL)

    if args.separator and not MDX_MODEL.exists():
        download(MDX_URL, MDX_MODEL)

    if args.sherpa_whisper and not (SHERPA_WHISPER_DIR / "small-tokens.txt").exists():
        tarpath = CACHE_DIR / "sherpa-onnx-whisper-small.tar.bz2"
        download(f"{GH}/sherpa-onnx-whisper-small.tar.bz2", tarpath)
        with tarfile.open(tarpath) as t:
            t.extractall(CACHE_DIR, filter="data")
        tarpath.unlink()
        # keep only int8 weights to save disk
        for f in SHERPA_WHISPER_DIR.glob("small-*coder.onnx"):
            f.unlink()

    if args.tts:
        setup_tts()
        args.sherpa_whisper = True  # speak.py uses it to pick the clearest take

    status = {m: "ok" for m in PY_DEPS}
    for mod, pipname in OPTIONAL.items():
        try:
            importlib.import_module(mod)
            status[mod] = "ok"
        except Exception:
            status[mod] = f"not installed (optional: pip install {pipname})"
    status["vad_model"] = str(VAD_MODEL) if VAD_MODEL.exists() else "missing"
    status["vocal_separator"] = str(MDX_MODEL) if MDX_MODEL.exists() else "missing (--separator)"
    if args.sherpa_whisper and not (SHERPA_WHISPER_DIR / "small-tokens.txt").exists():
        tarpath = CACHE_DIR / "sherpa-onnx-whisper-small.tar.bz2"
        download(f"{GH}/sherpa-onnx-whisper-small.tar.bz2", tarpath)
        with tarfile.open(tarpath) as t:
            t.extractall(CACHE_DIR, filter="data")
        tarpath.unlink()
    venv_py = CACHE_DIR / "venv-tts" / "bin" / "python"
    status["tts_engine"] = "ok" if (venv_py.exists() and (TTS_DIR / TTS_FILES[-1]).exists()) else "missing (--tts)"
    status["sherpa_whisper"] = (str(SHERPA_WHISPER_DIR)
                                if (SHERPA_WHISPER_DIR / "small-tokens.txt").exists() else "missing")
    for k, v in status.items():
        print(f"{k:16s} {v}")


if __name__ == "__main__":
    main()
