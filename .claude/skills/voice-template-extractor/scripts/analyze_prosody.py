#!/usr/bin/env python3
"""Step 3 — measure pitch, intonation, tempo, pauses, energy and emotional arousal.

Usage:
  python analyze_prosody.py OUTDIR

Reads  OUTDIR/audio/clean_full.wav, OUTDIR/transcript.json, OUTDIR/quality.json
Writes OUTDIR/prosody.json        per-phrase measurements + global voice profile
       OUTDIR/voice_profile.json  reusable template (settings + style), feed this back later
       OUTDIR/annotated_draft.md  transcript with automatic emotion/pause tags (draft;
                                  Claude refines it into VOICE_TEMPLATE.md)
       OUTDIR/prosody_plot.png    pitch + loudness over time

All measurements are relative to the speaker's own baseline, because "excited" for a calm
narrator is a different absolute pitch than for a hype creator.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import log, read_json, write_json  # noqa: E402

# --- text helpers -----------------------------------------------------------------

DIPHTHONG = re.compile(r"(ai|au|oi|ei)", re.I)
VOWELS = re.compile(r"[aiueoéèy]+", re.I)

# colloquial / discourse markers common in Indonesian content-creator speech
STYLE_MARKERS = ["gue", "gua", "gw", "lo", "lu", "aku", "kamu", "kalian", "guys", "teman-teman",
                 "banget", "nah", "jadi", "kayak", "sih", "dong", "deh", "loh", "lho", "kok", "ya",
                 "yah", "tuh", "nih", "kan", "terus", "pokoknya", "bener-bener", "beneran", "gitu",
                 "gimana", "oke", "wah", "wow", "anjir", "parah", "gila", "asli", "literally",
                 "basically", "actually", "so", "btw", "emang", "udah", "aja", "gak", "nggak", "ga"]


def count_syllables(text: str) -> int:
    n = 0
    for w in re.findall(r"[A-Za-zÀ-ÿ]+", text.lower()):
        w2 = DIPHTHONG.sub("a", w)
        n += max(1, len(VOWELS.findall(w2)))
    for num in re.findall(r"\d+", text):  # rough: "42" -> empat puluh dua ~ 5
        n += min(8, 2 * len(num) + 1)
    return n


# --- acoustic helpers -------------------------------------------------------------

def st(f, ref=100.0):
    return 12 * np.log2(np.asarray(f) / ref)


def pitch_track(snd):
    """Two-pass pitch range estimation (Hirst 2011) for robust F0 on any voice."""
    p1 = snd.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=700)
    f = p1.selected_array["frequency"]
    f = f[f > 0]
    if len(f) < 50:
        floor, ceil = 75, 500
    else:
        q15, q65 = np.percentile(f, [15, 65])
        floor, ceil = max(50, 0.75 * q15), min(800, 1.75 * q65)
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceil)
    return p.xs(), p.selected_array["frequency"], (round(floor), round(ceil))


def voice_quality(snd, floor, ceil):
    import parselmouth
    from parselmouth.praat import call
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", floor, ceil)
        jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harm = call(snd, "To Harmonicity (cc)", 0.01, floor, 0.1, 1.0)
        hnr = call(harm, "Get mean", 0, 0)
    except parselmouth.PraatError:
        return {}
    return {"jitter_pct": round(100 * jitter, 2), "shimmer_pct": round(100 * shimmer, 2),
            "hnr_db": round(hnr, 1)}


def classify(value, cuts, labels):
    for c, l in zip(cuts, labels):
        if value < c:
            return l
    return labels[-1]


def main() -> None:
    import librosa
    import parselmouth
    import soundfile as sf

    out = Path(sys.argv[1])
    wav = out / "audio" / "clean_full.wav"
    tr = read_json(out / "transcript.json")
    quality = read_json(out / "quality.json") if (out / "quality.json").exists() else {}
    y, sr = sf.read(str(wav), dtype="float32")
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)

    log("pitch & intensitas")
    t_f0, f0, (floor, ceil) = pitch_track(snd)
    inten = snd.to_intensity(minimum_pitch=max(60, floor), time_step=0.01)
    t_in, db = inten.xs(), inten.values[0]
    voiced = f0 > 0
    f0_st_all = st(f0[voiced])
    spk_med_st = float(np.median(f0_st_all))

    # silence = frames 18 dB below typical speech loudness; pauses = runs >= 150 ms
    speech_level = np.percentile(db, 70)
    quiet = db < speech_level - 18
    pauses = []  # (start, end)
    k = 0
    while k < len(quiet):
        if quiet[k]:
            j = k
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - k) * 0.01 >= 0.15:
                pauses.append((float(t_in[k]), float(t_in[min(j, len(t_in) - 1)])))
            k = j
        else:
            k += 1
    pauses_arr = np.array(pauses) if pauses else np.zeros((0, 2))

    def pause_time_in(a, b):
        if not len(pauses_arr):
            return 0.0, 0
        ov = np.clip(np.minimum(pauses_arr[:, 1], b) - np.maximum(pauses_arr[:, 0], a), 0, None)
        return float(ov.sum()), int((ov > 0.15).sum())

    def pause_at(t):
        """Length of the silence around a phrase boundary at time t."""
        if not len(pauses_arr):
            return 0.0
        near = pauses_arr[(pauses_arr[:, 1] >= t - 0.3) & (pauses_arr[:, 0] <= t + 0.3)]
        return float((near[:, 1] - near[:, 0]).max()) if len(near) else 0.0

    phrases = []
    segs = tr["segments"]
    for i, s in enumerate(segs):
        a, b = s["start"], s["end"]
        m = (t_f0 >= a) & (t_f0 <= b) & voiced
        fseg = f0[m]
        tseg = t_f0[m]
        mi = (t_in >= a) & (t_in <= b)
        dseg = db[mi]
        dur = max(0.2, b - a)
        syl = count_syllables(s["text"])
        inner_pause, n_inner = pause_time_in(a + 0.15, b - 0.15)
        phon = max(0.3 * dur, dur - inner_pause)
        loud = dseg > speech_level - 18 if len(dseg) else np.array([])
        quiet_runs = [None] * n_inner
        rec = {"i": i, "start": a, "end": b, "text": s["text"], "duration": round(dur, 2),
               "syllables": syl, "rate_syl_s": round(syl / dur, 2),
               "articulation_syl_s": round(syl / phon, 2),
               "internal_pauses": len(quiet_runs),
               "pause_after": (round(max(segs[i + 1]["start"] - b, pause_at(b)), 2)
                               if i + 1 < len(segs) else None)}
        if len(fseg) >= 10:
            sst = st(fseg)
            rec["f0_median_hz"] = round(float(np.median(fseg)), 1)
            rec["pitch_rel_st"] = round(float(np.median(sst)) - spk_med_st, 2)
            rec["pitch_range_st"] = round(float(np.percentile(sst, 90) - np.percentile(sst, 10)), 2)
            # terminal contour: last 0.35 s of voicing vs the 0.6 s before it
            tend = tseg[-1]
            last = sst[tseg >= tend - 0.35]
            prev = sst[(tseg < tend - 0.35) & (tseg >= tend - 0.95)]
            delta = float(np.median(last) - np.median(prev)) if len(prev) >= 5 and len(last) >= 5 else 0.0
            rec["ending_delta_st"] = round(delta, 2)
            rec["ending"] = "naik" if delta > 1.5 else "turun" if delta < -1.5 else "datar"
            # where in the phrase is the pitch peak (0=start, 1=end): emphasis placement
            rec["peak_position"] = round(float((tseg[np.argmax(sst)] - a) / dur), 2)
        else:
            rec.update({"f0_median_hz": None, "pitch_rel_st": 0.0, "pitch_range_st": 0.0,
                        "ending_delta_st": 0.0, "ending": "datar", "peak_position": None})
        rec["loudness_db"] = round(float(np.mean(dseg[loud])) if len(loud) and loud.any() else
                                   float(np.mean(dseg)) if len(dseg) else 0.0, 2)
        # emphasised words (needs word timestamps)
        emph = []
        for w in s.get("words", []):
            wm = (t_in >= w["start"]) & (t_in <= w["end"])
            fm = (t_f0 >= w["start"]) & (t_f0 <= w["end"]) & voiced
            if wm.any() and fm.sum() >= 3 and len(fseg) >= 10:
                if (db[wm].max() > rec["loudness_db"] + 3.5 and
                        np.median(st(f0[fm])) > np.median(st(fseg)) + 1.5):
                    emph.append(w["word"])
        rec["emphasis_words"] = emph
        phrases.append(rec)

    # --- speaker-relative arousal (energy/pitch/range/tempo z-scores) -------------
    # minimum spreads stop tiny, inaudible differences (e.g. in heavily compressed ads)
    # from being blown up into "very excited" vs "very calm"
    MIN_SD = {"loudness_db": 1.5, "pitch_rel_st": 1.0, "pitch_range_st": 1.5, "articulation_syl_s": 0.5}

    def z(key):
        v = np.array([p[key] for p in phrases], float)
        sd = max(v.std(), MIN_SD[key])
        return (v - v.mean()) / sd

    if len(phrases) >= 3:
        ar = (1.2 * z("loudness_db") + 1.0 * z("pitch_rel_st") + 0.8 * z("pitch_range_st")
              + 0.6 * z("articulation_syl_s")) / 3.6
    else:
        ar = np.zeros(len(phrases))
    for p, a in zip(phrases, ar):
        p["arousal_z"] = round(float(a), 2)
        p["energy_level"] = classify(a, [-0.8, -0.25, 0.35, 0.9],
                                     ["tenang", "santai", "netral", "antusias", "sangat antusias"])
        txt = p["text"].strip()
        tags = []
        if a >= 0.9:
            tags.append("[excited]")
        elif a >= 0.35:
            tags.append("[enthusiastic]")
        elif a <= -0.8:
            tags.append("[calm]" if p["articulation_syl_s"] < np.median([q["articulation_syl_s"] for q in phrases]) else "[serious]")
        if txt.endswith("?") or (p["ending"] == "naik" and a > -0.25):
            tags.append("[curious]" if txt.endswith("?") else "")
        p["suggested_tags"] = [t for t in tags if t]

    # --- global profile ---------------------------------------------------------
    med_hz = float(np.median(f0[voiced]))
    rng_global = float(np.percentile(f0_st_all, 95) - np.percentile(f0_st_all, 5))
    rng_phrase = float(np.median([p["pitch_range_st"] for p in phrases])) if phrases else 0
    rates = np.array([p["rate_syl_s"] for p in phrases])
    arts = np.array([p["articulation_syl_s"] for p in phrases])
    gaps = np.array([p["pause_after"] for p in phrases if p["pause_after"] is not None])
    all_pause_d = pauses_arr[:, 1] - pauses_arr[:, 0] if len(pauses_arr) else np.zeros(0)
    total_min = max(1e-6, (segs[-1]["end"] - segs[0]["start"]) / 60) if segs else 1
    endings = Counter(p["ending"] for p in phrases)
    levels = Counter(p["energy_level"] for p in phrases)
    loud_sd = float(np.std([p["loudness_db"] for p in phrases])) if phrases else 0
    vq = voice_quality(snd, floor, ceil)

    if med_hz < 165:
        register = classify(med_hz, [100, 125, 145], ["pria - sangat rendah/berat", "pria - rendah",
                                                      "pria - sedang", "pria - agak tinggi / tenor"])
    else:
        register = classify(med_hz, [190, 225, 260], ["wanita - rendah/alto", "wanita - sedang",
                                                      "wanita - tinggi", "wanita - sangat tinggi / anak"])
    expressiveness = classify(rng_global, [6, 10, 14], ["monoton", "agak datar", "ekspresif", "sangat ekspresif"])
    art_mean = float(np.mean(arts)) if len(arts) else 0
    tempo = classify(art_mean, [4.8, 5.8, 6.8, 7.8], ["lambat", "sedang", "agak cepat", "cepat", "sangat cepat"])
    spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0][
        librosa.feature.rms(y=y)[0] > np.percentile(librosa.feature.rms(y=y)[0], 40)]))
    timbre = classify(spec_cent, [1400, 2200, 3000], ["gelap/hangat", "hangat-netral", "cerah", "sangat cerah/tajam"])
    hnr = vq.get("hnr_db", 15)
    texture = classify(hnr, [7, 12, 18], ["serak/berat napas", "sedikit serak/breathy", "jernih", "sangat jernih"])

    # transcript style
    words = re.findall(r"[a-zA-Z\-]+", " ".join(p["text"] for p in phrases).lower())
    wc = Counter(words)
    markers = {m: wc[m] for m in STYLE_MARKERS if wc[m] > 0}
    markers = dict(sorted(markers.items(), key=lambda kv: -kv[1]))
    sent_lens = [len(re.findall(r"\w+", x)) for x in re.split(r"[.!?]+", " ".join(p["text"] for p in phrases)) if x.strip()]

    # --- TTS recommendations ----------------------------------------------------
    bg = quality.get("background", {})
    rel = (bg.get("after_separation") or bg).get("background_rel_db")
    clean = rel is None or rel < -30
    if rng_global >= 10:
        stab, style, v3 = 0.35, 0.35, "Natural (Creative kalau mau lebih dramatis)"
    elif rng_global >= 6:
        stab, style, v3 = 0.45, 0.2, "Natural"
    else:
        stab, style, v3 = 0.6, 0.05, "Robust"
    design_speed = round(float(np.clip(art_mean / 6.0, 0.85, 1.15)), 2)
    gender_en = "male" if med_hz < 165 else "female"
    tempo_en = {"lambat": "slow", "sedang": "moderate", "agak cepat": "slightly fast",
                "cepat": "fast", "sangat cepat": "very fast"}[tempo]
    expr_en = {"monoton": "flat, even", "agak datar": "restrained", "ekspresif": "expressive",
               "sangat ekspresif": "very animated"}[expressiveness]
    timbre_en = {"gelap/hangat": "dark, warm", "hangat-netral": "warm", "cerah": "bright",
                 "sangat cerah/tajam": "very bright, crisp"}[timbre]
    reg_en = ("deep" if med_hz < 125 or 165 <= med_hz < 190 else
              "mid-pitched" if med_hz < 145 or med_hz < 225 else "high-pitched")
    lang = tr.get("language") or "id"
    lang_en = {"id": "Indonesian (Jakarta casual accent)", "en": "English", "ms": "Malay"}.get(lang, lang)
    design_prompt = (f"A {reg_en} {gender_en} voice speaking {lang_en}, {timbre_en} timbre, "
                     f"{expr_en} intonation with a {tempo_en} conversational pace. Sounds like a "
                     f"friendly content creator talking directly to camera, natural and relaxed, "
                     f"close-mic studio quality.")

    profile = {
        "pitch": {"median_hz": round(med_hz, 1), "register": register,
                  "range_global_st": round(rng_global, 1), "range_per_phrase_st": round(rng_phrase, 1),
                  "expressiveness": expressiveness, "pitch_floor_ceiling_hz": [floor, ceil]},
        "tempo": {"speech_rate_syl_s": round(float(rates.mean()), 2) if len(rates) else None,
                  "articulation_rate_syl_s": round(art_mean, 2), "class": tempo},
        "pauses": {"pauses_per_minute": round(len(all_pause_d) / total_min, 1),
                   "pause_median_s": round(float(np.median(all_pause_d)), 2) if len(all_pause_d) else None,
                   "pause_p90_s": round(float(np.percentile(all_pause_d, 90)), 2) if len(all_pause_d) else None,
                   "between_phrases_median_s": round(float(np.median(gaps)), 2) if len(gaps) else None,
                   "between_phrases_p90_s": round(float(np.percentile(gaps, 90)), 2) if len(gaps) else None,
                   "share_long_pauses_gt_0_5s": round(float(np.mean(gaps > 0.5)), 2) if len(gaps) else None},
        "intonation": {"phrase_endings": dict(endings),
                       "peak_position_median": (round(float(np.median([p["peak_position"] for p in phrases if p["peak_position"] is not None])), 2)
                                                if phrases else None)},
        "energy": {"loudness_sd_db": round(loud_sd, 2), "levels": dict(levels)},
        "timbre": {"spectral_centroid_hz": round(spec_cent), "brightness": timbre, "texture": texture, **vq},
        "style": {"language": lang, "markers": markers,
                  "avg_sentence_words": round(float(np.mean(sent_lens)), 1) if sent_lens else None},
    }
    recommend = {
        "elevenlabs": {
            "clone_type": "Instant Voice Clone (upload audio/cloning_sample.mp3)",
            "model_for_emotion": "eleven_v3 (pakai audio tags seperti [excited], [laughs])",
            "model_for_stable_narration": "eleven_multilingual_v2",
            "v2_settings": {"stability": stab, "similarity_boost": 0.8 if clean else 0.7,
                            "style": style, "use_speaker_boost": True, "speed": 1.0},
            "v3_stability_mode": v3,
            "note_speed": ("Voice clone sudah mewarisi tempo asli; biarkan speed 1.0. "
                           f"Untuk voice lain/Voice Design, mulai dari speed {design_speed}."),
        },
        "voice_design_prompt": design_prompt,
    }
    # --- reference clips (for zero-shot TTS that wants audio + its exact text) ----
    refdir = out / "audio" / "reference"
    refdir.mkdir(parents=True, exist_ok=True)
    references = {}
    ok = [p for p in phrases if 4.0 <= p["duration"] <= 15.0 and len(p["text"]) > 20]

    def complete(p):  # whole sentence: starts after a sentence end, ends with punctuation
        prev_ok = p["i"] == 0 or phrases[p["i"] - 1]["text"].rstrip().endswith((".", "!", "?"))
        t = p["text"].strip()
        return prev_ok and t[:1].isupper() and t.endswith((".", "!", "?"))

    ok = [p for p in ok if complete(p)] or ok
    if ok:
        picks = {"neutral": min(ok, key=lambda p: abs(p["arousal_z"])),
                 "energetic": max(ok, key=lambda p: p["arousal_z"]),
                 "calm": min(ok, key=lambda p: p["arousal_z"])}
        for name, p in picks.items():
            a_, b_ = int(max(0, p["start"] - 0.05) * sr), int(min(len(y), (p["end"] + 0.1) * sr))
            sf.write(str(refdir / f"{name}.wav"), y[a_:b_], sr, subtype="PCM_16")
            (refdir / f"{name}.txt").write_text(p["text"].strip() + "\n")
            references[name] = {"wav": f"audio/reference/{name}.wav", "text": p["text"].strip(),
                                "phrase": p["i"], "energy_level": p["energy_level"]}

    write_json(out / "prosody.json", {"profile": profile, "phrases": phrases})
    write_json(out / "voice_profile.json", {
        "name": out.name, "source": quality.get("source"),
        "cloning_sample": "audio/cloning_sample.mp3", "reference_clips": references,
        "profile": profile, "recommend": recommend})

    # --- draft annotated transcript --------------------------------------------
    lines = [f"# Draft anotasi — {out.name}", "",
             "Otomatis dari analisis akustik. Tag emosi = saran awal; perbaiki dengan konteks makna.", "",
             "| # | waktu | energi | nada vs rata2 | akhir | tempo | jeda sesudah | teks |",
             "|---|---|---|---|---|---|---|---|"]
    for p in phrases:
        lines.append(f"| {p['i']} | {p['start']:.1f}-{p['end']:.1f} | {p['energy_level']} | "
                     f"{p['pitch_rel_st']:+.1f} st | {p['ending']} | {p['articulation_syl_s']} | "
                     f"{p['pause_after'] if p['pause_after'] is not None else '-'} | {p['text']} |")
    lines += ["", "## Draft naskah bertag (ElevenLabs v3)", ""]
    for p in phrases:
        tag = " ".join(p["suggested_tags"])
        txt = p["text"]
        for w in p["emphasis_words"]:
            txt = re.sub(rf"\b{re.escape(w)}\b", w.upper(), txt, count=1)
        pause = p["pause_after"] or 0
        sep = " ..." if pause >= 0.7 else ""
        lines.append(f"{tag + ' ' if tag else ''}{txt}{sep}")
    (out / "annotated_draft.md").write_text("\n".join(lines) + "\n")

    # --- plot -------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
        ff = np.where(voiced, f0, np.nan)
        ax[0].plot(t_f0, ff, ".", ms=1.5, color="#1f6feb")
        ax[0].set_ylabel("Pitch (Hz)")
        ax[0].axhline(med_hz, color="gray", lw=0.8, ls="--")
        ax[1].plot(t_in, db, lw=0.6, color="#d1242f")
        ax[1].set_ylabel("Loudness (dB)")
        ax[1].set_xlabel("Waktu (detik)")
        for p in phrases:
            for a_ in ax:
                a_.axvspan(p["start"], p["end"], alpha=0.06 + 0.1 * max(0, p["arousal_z"]), color="orange")
            ax[0].text(p["start"], ceil * 0.95, str(p["i"]), fontsize=7)
        ax[0].set_ylim(floor, ceil)
        fig.suptitle(f"Kontur nada & energi — {out.name} (oranye lebih pekat = lebih antusias)")
        fig.tight_layout()
        fig.savefig(out / "prosody_plot.png", dpi=110)
    except Exception as e:  # plotting is optional
        log(f"plot gagal: {e}")

    print(f"OK phrases={len(phrases)} median_f0={med_hz:.0f}Hz register='{register}' "
          f"range={rng_global:.1f}st ({expressiveness}) tempo={art_mean:.1f} syl/s ({tempo})")


if __name__ == "__main__":
    main()
