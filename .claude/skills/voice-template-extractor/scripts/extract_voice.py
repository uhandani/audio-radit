#!/usr/bin/env python3
"""Step 1 — pull a clean, cloning-ready voice track out of a video (or audio) file.

Usage:
  python extract_voice.py INPUT.mp4 --out OUTDIR [--separate auto|yes|no]
                          [--max-sample 180] [--no-denoise]

Writes into OUTDIR:
  audio/original.wav        mono 44.1 kHz straight from the video (untouched)
  audio/vocals.wav          vocals only (only when music separation ran)
  audio/clean_full.wav      cleaned + loudness-normalised full track (timing preserved)
  audio/cloning_sample.wav  speech only, long pauses shortened -> upload this to TTS cloning
  audio/cloning_sample.mp3  same, 192 kbps MP3 (ElevenLabs accepts both)
  audio/clips/clip_XX.wav   5–25 s phrase clips (for picking the best takes / PVC)
  segments.json             speech segments (seconds) on clean_full.wav timeline
  quality.json              loudness, background-noise/music estimate, warnings
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (energy_segments, find_vad_model, log, require_ffmpeg, run,  # noqa: E402
                    silero_segments, split_long_segments, write_json)

SR = 44100


def ffmpeg_extract(src: Path, dst: Path) -> None:
    run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR),
         "-c:a", "pcm_s16le", str(dst)])


def load_16k(path: Path) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=16000, mono=True)
    return y.astype(np.float32)


def detect_speech(y16: np.ndarray, vad_model: Path | None):
    if vad_model:
        try:
            return silero_segments(y16, 16000, vad_model), "silero"
        except Exception as e:  # pragma: no cover
            log(f"Silero VAD gagal ({e}); pakai energy VAD")
    return energy_segments(y16, 16000), "energy"


def background_report(y16: np.ndarray, segs) -> dict:
    """Estimate how loud and how 'musical' the non-speech parts are."""
    import librosa
    sr = 16000
    mask = np.zeros(len(y16), bool)
    for s, e in segs:
        mask[int(s * sr):int(e * sr)] = True
    speech, gaps = y16[mask], y16[~mask]
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)) + 1e-9)  # noqa: E731
    sp_db = 20 * np.log10(rms(speech)) if len(speech) else -120.0
    if len(gaps) < sr * 0.5:
        return {"speech_rms_db": round(sp_db, 1), "background_rel_db": None,
                "gap_seconds": round(len(gaps) / sr, 1), "music_likely": False,
                "note": "hampir tidak ada jeda; latar belakang tidak bisa diukur"}
    bg_db = 20 * np.log10(rms(gaps))
    flat = float(np.median(librosa.feature.spectral_flatness(y=gaps, n_fft=2048)))
    # tonal (low flatness) + clearly audible background between phrases -> music bed
    rel = bg_db - sp_db
    music = bool(rel > -32 and flat < 0.08)
    return {"speech_rms_db": round(sp_db, 1), "background_rel_db": round(rel, 1),
            "background_flatness": round(flat, 4), "gap_seconds": round(len(gaps) / sr, 1),
            "music_likely": music}


def run_separation(src: Path, out_wav: Path) -> str | None:
    """Try MDX-Net ONNX (light, no torch) first, then demucs. Returns method name or None."""
    from separate_vocals import find_model, separate_file
    model = find_model()
    if model:
        try:
            import onnxruntime  # noqa: F401
            separate_file(src, out_wav, model)
            return "mdx-net (Kim_Vocal_2)"
        except Exception as e:
            log(f"MDX-Net gagal: {e}")
    if run_demucs(src, out_wav):
        return "demucs htdemucs"
    return None


def run_demucs(src: Path, out_wav: Path) -> bool:
    try:
        import demucs  # noqa: F401
    except Exception:
        log("separator belum tersedia (jalankan setup_env.py --separator)")
        return False
    tmp = Path(tempfile.mkdtemp())
    try:
        run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
             "-o", str(tmp), str(src)])
        voc = next(tmp.rglob("vocals.wav"))
        run(["ffmpeg", "-v", "error", "-y", "-i", str(voc), "-ac", "1", "-ar", str(SR),
             "-c:a", "pcm_s16le", str(out_wav)])
        return True
    except Exception as e:
        log(f"demucs gagal: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def loudnorm_two_pass(src: Path, dst: Path, pre_filters: str, target=-20.0, tp=-3.0) -> dict:
    """Linear (not dynamic) loudness normalisation so emotional dynamics survive."""
    chain = (pre_filters + "," if pre_filters else "")
    p = run(["ffmpeg", "-hide_banner", "-y", "-i", str(src), "-af",
             f"{chain}loudnorm=I={target}:TP={tp}:LRA=20:print_format=json", "-f", "null", "-"],
            capture=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr, re.S)
    import json
    meas = json.loads(m.group(0))
    ln = (f"loudnorm=I={target}:TP={tp}:LRA=20:measured_I={meas['input_i']}:"
          f"measured_TP={meas['input_tp']}:measured_LRA={meas['input_lra']}:"
          f"measured_thresh={meas['input_thresh']}:offset={meas['target_offset']}:linear=true")
    run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-af", chain + ln, "-ar", str(SR),
         "-c:a", "pcm_s16le", str(dst)])
    return {"input_lufs": float(meas["input_i"]), "input_true_peak": float(meas["input_tp"]),
            "input_lra": float(meas["input_lra"]), "target_lufs": target}


def group_clips(segs, min_len=5.0, max_len=25.0, pad=0.15):
    clips, cur = [], None
    for s, e in segs:
        if cur is None:
            cur = [s, e]
        elif e - cur[0] <= max_len and s - cur[1] < 1.2:
            cur[1] = e
        else:
            clips.append(cur)
            cur = [s, e]
    if cur:
        clips.append(cur)
    # merge too-short clips into neighbour when possible
    merged = []
    for c in clips:
        if merged and (c[1] - c[0] < min_len or merged[-1][1] - merged[-1][0] < min_len) \
                and c[1] - merged[-1][0] <= max_len + 5:
            merged[-1][1] = c[1]
        else:
            merged.append(c)
    return [(max(0.0, s - pad), e + pad) for s, e in merged]


def build_cloning_sample(y: np.ndarray, segs, max_pause=0.6, max_total=180.0) -> np.ndarray:
    parts, total, prev_end = [], 0.0, None
    for s, e in segs:
        if prev_end is not None:
            gap = min(s - prev_end, max_pause)
            parts.append(np.zeros(int(max(gap, 0.12) * SR), np.float32))
            total += gap
        a, b = int(max(0, s - 0.05) * SR), int(min(len(y) / SR, e + 0.08) * SR)
        chunk = y[a:b].copy()
        fade = min(len(chunk) // 4, int(0.01 * SR))
        if fade > 0:
            chunk[:fade] *= np.linspace(0, 1, fade)
            chunk[-fade:] *= np.linspace(1, 0, fade)
        parts.append(chunk)
        total += e - s
        prev_end = e
        if total >= max_total:
            break
    return np.concatenate(parts) if parts else np.zeros(1, np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--separate", choices=["auto", "yes", "no"], default="auto",
                    help="pisahkan vokal dari musik latar (MDX-Net ONNX, fallback demucs)")
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--max-sample", type=float, default=180.0,
                    help="durasi maksimal cloning_sample (detik)")
    ap.add_argument("--vad-model", default=None)
    args = ap.parse_args()

    require_ffmpeg()
    src = Path(args.input).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    adir = out / "audio"
    (adir / "clips").mkdir(parents=True, exist_ok=True)
    vad_model = find_vad_model(args.vad_model)
    warnings: list[str] = []

    log("1/5 ekstrak audio dari video")
    original = adir / "original.wav"
    ffmpeg_extract(src, original)
    duration = sf.info(str(original)).duration

    log("2/5 deteksi suara & cek musik/noise latar")
    y16 = load_16k(original)
    segs0, vad_kind = detect_speech(y16, vad_model)
    bg = background_report(y16, segs0)

    work = original
    separated: str | bool = False
    if args.separate == "yes" or (args.separate == "auto" and bg.get("music_likely")):
        log("   musik latar terdeteksi -> pisahkan vokal dari musik")
        method = run_separation(original, adir / "vocals.wav")
        if method:
            work, separated = adir / "vocals.wav", method
            y16 = load_16k(work)
            segs0, vad_kind = detect_speech(y16, vad_model)
            bg_after = background_report(y16, segs0)
            bg["after_separation"] = bg_after
        else:
            warnings.append("Musik latar kemungkinan ada tapi pemisah vokal tidak tersedia. Hasil clone "
                            "bisa ikut 'menyerap' musik. Jalankan setup_env.py --separator lalu ulangi "
                            "dengan --separate yes.")

    log("3/5 bersihkan (high-pass, denoise ringan) + normalisasi loudness linear")
    filters = ["highpass=f=70"]
    rel = (bg.get("after_separation") or bg).get("background_rel_db")
    if not args.no_denoise and rel is not None and rel > -45:
        # gentle: heavy denoise makes clones sound metallic/robotic
        filters.append("afftdn=nr=10:nf=-42:tn=1")
    clean = adir / "clean_full.wav"
    loud = loudnorm_two_pass(work, clean, ",".join(filters))

    log("4/5 segmentasi kalimat & potong klip")
    y, _ = sf.read(str(clean), dtype="float32")
    y16c = load_16k(clean)
    segs, vad_kind = detect_speech(y16c, vad_model)
    segs = split_long_segments(y16c, 16000, segs)
    segs = [(round(s, 3), round(e, 3)) for s, e in segs if e - s >= 0.25]
    speech_total = sum(e - s for s, e in segs)

    clips_meta = []
    for i, (s, e) in enumerate(group_clips(segs), 1):
        a, b = int(s * SR), int(min(e * SR, len(y)))
        chunk = y[a:b]
        peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
        clip_ratio = float(np.mean(np.abs(chunk) > 0.985)) if len(chunk) else 0.0
        name = f"clip_{i:02d}.wav"
        sf.write(str(adir / "clips" / name), chunk, SR, subtype="PCM_16")
        clips_meta.append({"file": f"audio/clips/{name}", "start": round(s, 2), "end": round(e, 2),
                           "duration": round(e - s, 2), "peak": round(peak, 3),
                           "clipping_ratio": round(clip_ratio, 5)})

    log("5/5 buat sampel untuk voice cloning")
    sample = build_cloning_sample(y, segs, max_total=args.max_sample)
    sf.write(str(adir / "cloning_sample.wav"), sample, SR, subtype="PCM_16")
    run(["ffmpeg", "-v", "error", "-y", "-i", str(adir / "cloning_sample.wav"), "-b:a", "192k",
         str(adir / "cloning_sample.mp3")])
    sample_dur = len(sample) / SR

    # warnings that matter for clone quality
    if speech_total < 30:
        warnings.append(f"Total ucapan hanya {speech_total:.0f} detik. Instant Voice Clone butuh minimal "
                        "~1 menit (ideal 1–3 menit) agar mirip. Gabungkan beberapa video dari orang yang sama.")
    elif speech_total < 60:
        warnings.append(f"Total ucapan {speech_total:.0f} detik — cukup untuk mencoba, tapi 1–3 menit "
                        "biasanya lebih stabil.")
    if any(c["clipping_ratio"] > 0.001 for c in clips_meta):
        warnings.append("Ada klip yang clipping/pecah. Buang klip dengan clipping_ratio tinggi sebelum upload.")
    if rel is not None and rel > -25 and not separated:
        warnings.append("Latar belakang cukup keras dibanding suara (≤25 dB). Clone bisa ikut meniru noise; "
                        "pilih klip yang paling bersih atau pakai --separate yes.")
    if separated and rel is not None and rel > -30:
        warnings.append(f"Musik sudah dipisah, tapi sisa musik/noise samar masih ada ({rel:.0f} dB di bawah "
                        "suara). Dengarkan klip dan buang yang paling kotor; pakai Similarity ~0.7.")
    if loud["input_lufs"] < -35:
        warnings.append("Audio asli sangat pelan; normalisasi menaikkan noise juga.")

    seg_out = {"audio": "audio/clean_full.wav", "vad": vad_kind,
               "segments": [{"start": s, "end": e} for s, e in segs]}
    write_json(out / "segments.json", seg_out)
    quality = {
        "source": src.name, "source_duration_sec": round(duration, 2),
        "speech_total_sec": round(speech_total, 1), "speech_segments": len(segs),
        "cloning_sample_sec": round(sample_dur, 1), "vad": vad_kind,
        "music_separated": separated, "filters": filters, "loudness": loud,
        "background": bg, "clips": clips_meta, "warnings": warnings,
    }
    write_json(out / "quality.json", quality)
    print(f"OK  speech={speech_total:.1f}s  sample={sample_dur:.1f}s  clips={len(clips_meta)}  "
          f"music_likely={bg.get('music_likely')}  separated={separated}")
    for w in warnings:
        print("WARN", w)


if __name__ == "__main__":
    main()
