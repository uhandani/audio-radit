#!/usr/bin/env python3
"""Step 2 — transcribe the cleaned voice with timestamps.

Usage:
  python transcribe.py OUTDIR [--lang id] [--backend auto|faster-whisper|sherpa]
                       [--model small|medium|large-v3]

Reads OUTDIR/audio/clean_full.wav (+ OUTDIR/segments.json) and writes OUTDIR/transcript.json:
  {"backend": ..., "language": ..., "segments": [{"start", "end", "text", "words": [...]}]}

Backends:
  faster-whisper  best: word-level timestamps (needs model download from HuggingFace)
  sherpa          fallback: Whisper-small ONNX from GitHub, phrase-level timestamps only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SHERPA_WHISPER_DIR, log, read_json, write_json  # noqa: E402


def load16(path: Path) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=16000, mono=True)
    return y.astype(np.float32)


def with_faster_whisper(wav: Path, lang: str, model: str):
    from faster_whisper import WhisperModel
    m = WhisperModel(model, device="auto", compute_type="int8")
    segs, info = m.transcribe(str(wav), language=lang or None, word_timestamps=True,
                              vad_filter=True, condition_on_previous_text=False)
    out = []
    for s in segs:
        out.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                    "words": [{"start": round(w.start, 2), "end": round(w.end, 2),
                               "word": w.word.strip(), "prob": round(w.probability, 3)}
                              for w in (s.words or [])]})
    return out, info.language


def split_long(segments, max_len=12.0):
    """Whisper ONNX handles <=30 s; keep phrases short so prosody stays per-phrase."""
    out = []
    for s, e in segments:
        while e - s > max_len:
            out.append((s, s + max_len))
            s += max_len
        out.append((s, e))
    return out


def with_sherpa(wav: Path, lang: str, segments, model_dir: Path):
    import sherpa_onnx
    enc = next(model_dir.glob("*encoder.int8.onnx"), None) or next(model_dir.glob("*encoder.onnx"))
    dec = next(model_dir.glob("*decoder.int8.onnx"), None) or next(model_dir.glob("*decoder.onnx"))
    tok = next(model_dir.glob("*tokens.txt"))
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(enc), decoder=str(dec), tokens=str(tok), language=lang or "",
        task="transcribe", num_threads=4, tail_paddings=1000)
    y = load16(wav)
    out = []
    segs = split_long(segments)
    for i, (s, e) in enumerate(segs):
        st = rec.create_stream()
        try:  # fast Indonesian speech exceeds the default 6 tokens/s cap -> truncated text
            st.set_option("max_tokens_per_second", "12")
        except Exception:
            pass
        st.accept_waveform(16000, y[int(s * 16000):int(e * 16000)])
        rec.decode_stream(st)
        text = st.result.text.strip()
        if text:
            out.append({"start": round(s, 2), "end": round(e, 2), "text": text, "words": []})
        if i % 5 == 0:
            log(f"   transkripsi {i + 1}/{len(segs)}")
    return out, lang


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--lang", default="id", help="kode bahasa (id, en, ms, jv...) atau '' untuk auto")
    ap.add_argument("--backend", default="auto", choices=["auto", "faster-whisper", "sherpa"])
    ap.add_argument("--model", default="small", help="ukuran model faster-whisper")
    ap.add_argument("--sherpa-dir", default=str(SHERPA_WHISPER_DIR))
    a = ap.parse_args()

    out = Path(a.outdir)
    wav = out / "audio" / "clean_full.wav"
    segs = [(d["start"], d["end"]) for d in read_json(out / "segments.json")["segments"]]

    result, backend, lang = None, None, a.lang
    if a.backend in ("auto", "faster-whisper"):
        try:
            log("transkripsi dengan faster-whisper")
            result, lang = with_faster_whisper(wav, a.lang, a.model)
            backend = f"faster-whisper/{a.model}"
        except Exception as e:
            if a.backend == "faster-whisper":
                raise
            log(f"faster-whisper tidak bisa dipakai ({type(e).__name__}); fallback ke sherpa-onnx")
    if result is None:
        mdir = Path(a.sherpa_dir)
        if not mdir.is_dir():
            sys.exit("Tidak ada backend transkripsi. Jalankan: python setup_env.py --sherpa-whisper")
        result, lang = with_sherpa(wav, a.lang, segs, mdir)
        backend = "sherpa-onnx/whisper-small"

    write_json(out / "transcript.json", {"backend": backend, "language": lang,
                                          "word_timestamps": any(s["words"] for s in result),
                                          "segments": result})
    print(f"OK backend={backend} segments={len(result)}")
    for s in result:
        print(f"[{s['start']:6.2f}-{s['end']:6.2f}] {s['text']}")


if __name__ == "__main__":
    main()
