#!/usr/bin/env python3
"""Check a voice-over made by speak.py against its style, and test that it's intelligible.

Usage:
  python check_tts.py OUT.wav [--style pria-jakarta-santai]

Prints: pitch median/range vs style, articulation rate vs style, per-sentence pitch & rate, and an
ASR round-trip (Whisper transcribes the audio; low word-match = mispronounced or mushy words).
Use it after every generation — you can't listen, this is how you find what to fix
(add a lexicon entry, rephrase, or adjust the style).
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SHERPA_WHISPER_DIR  # noqa: E402

STYLES = Path(__file__).resolve().parent.parent / "styles"


def syl(t: str) -> int:
    n = sum(max(1, len(re.findall(r"[aiueo]+", re.sub(r"(ai|au|oi)", "a", w))))
            for w in re.findall(r"[a-z]+", t.lower()))
    return n + sum(min(8, 2 * len(x) + 1) for x in re.findall(r"\d+", t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--style", default=None)
    a = ap.parse_args()
    import librosa
    import parselmouth

    wav = Path(a.wav)
    tm = json.loads(wav.with_suffix(".timing.json").read_text())
    style = json.loads((STYLES / f"{a.style or tm['style']}.json").read_text())
    y, sr = librosa.load(str(wav), sr=None)

    def f0(x):
        f = parselmouth.Sound(x.astype(float), sr).to_pitch_ac(time_step=0.01, pitch_floor=60,
                                                               pitch_ceiling=400).selected_array["frequency"]
        return f[f > 0]

    f = f0(y)
    s = 12 * np.log2(f)
    sp = sum(x["end"] - x["start"] for x in tm["sentences"])
    rate = sum(syl(x["text"]) for x in tm["sentences"]) / sp
    print(f"pitch median {np.median(f):.0f} Hz (style {style['pitch']['median_hz']:.0f})  "
          f"range p5-95 {np.percentile(s, 95) - np.percentile(s, 5):.1f} st (style {style['pitch']['range_global_st']})")
    print(f"tempo {rate:.2f} syl/s (style {style['tempo']['articulation_syl_s']})  "
          f"durasi {tm['duration']:.1f} s")

    import sherpa_onnx
    d = SHERPA_WHISPER_DIR
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(d / "small-encoder.int8.onnx"), decoder=str(d / "small-decoder.int8.onnx"),
        tokens=str(d / "small-tokens.txt"), language="id", task="transcribe", num_threads=4)
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    scores = []
    for x in tm["sentences"]:
        a_, b_ = int(x["start"] * sr), int(x["end"] * sr)
        ff = f0(y[a_:b_])
        st = rec.create_stream()
        try:
            st.set_option("max_tokens_per_second", "12")
        except Exception:
            pass
        st.accept_waveform(16000, y16[int(x["start"] * 16000):int(x["end"] * 16000)])
        rec.decode_stream(st)
        norm = lambda t: re.sub(r"[^\w\s]", "", t.lower()).split()  # noqa: E731
        r = difflib.SequenceMatcher(None, norm(x["text"]), norm(st.result.text)).ratio()
        scores.append(r)
        flag = "  <-- cek" if r < 0.7 else ""
        print(f"{x['emotion'][:14]:14s} f0 {np.median(ff) if len(ff) else 0:4.0f}  "
              f"rate {syl(x['text']) / max(0.1, x['end'] - x['start']):4.1f}  asr {r:.2f}  {x['text'][:55]}{flag}")
        if r < 0.7:
            print(f"{'':14s} ASR dengar: {st.result.text}")
    print(f"rata-rata kecocokan ASR {np.mean(scores):.2f} (>=0.8 bagus; ASR juga salah dengar nama asing)")


if __name__ == "__main__":
    main()
