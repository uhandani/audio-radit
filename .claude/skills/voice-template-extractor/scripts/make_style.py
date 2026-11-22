#!/usr/bin/env python3
"""Turn an analysed voice (prosody.json) into a reusable *speaking style* for the offline TTS.

Usage:
  python make_style.py OUTDIR --name pria-jakarta-santai [--description "..."]

Writes <skill>/styles/<name>.json. Only numbers are stored: pitch level and range, tempo,
pauses, intonation shape and how each emotion level moves pitch/tempo/loudness. No audio and
no voice identity is kept — speak.py applies this style to one of the built-in TTS voices.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_json, write_json  # noqa: E402

STYLES_DIR = Path(__file__).resolve().parent.parent / "styles"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--description", default="")
    a = ap.parse_args()

    pj = read_json(Path(a.outdir) / "prosody.json")
    prof, phrases = pj["profile"], pj["phrases"]
    med = prof["pitch"]["median_hz"]
    gender = "male" if med < 165 else "female"
    art = prof["tempo"]["articulation_rate_syl_s"]
    loud_mean = float(np.mean([p["loudness_db"] for p in phrases]))

    levels = {}
    for lvl in ["tenang", "santai", "netral", "antusias", "sangat antusias"]:
        ps = [p for p in phrases if p["energy_level"] == lvl]
        if ps:
            levels[lvl] = {
                "n": len(ps),
                "pitch_rel_st": round(float(np.mean([p["pitch_rel_st"] for p in ps])), 2),
                "range_st": round(float(np.mean([p["pitch_range_st"] for p in ps])), 2),
                "speed_rel": round(float(np.mean([p["articulation_syl_s"] for p in ps])) / art, 3),
                "loudness_rel_db": round(float(np.mean([p["loudness_db"] for p in ps])) - loud_mean, 2),
            }
    endings = prof["intonation"]["phrase_endings"]
    tot = max(1, sum(endings.values()))
    pauses = prof["pauses"]
    sent_pause = pauses.get("pause_median_s") or pauses.get("between_phrases_median_s") or 0.3
    p90 = pauses.get("pause_p90_s") or pauses.get("between_phrases_p90_s") or 0.5

    style = {
        "name": a.name,
        "description": a.description or (
            f"{'Pria' if gender == 'male' else 'Wanita'}, nada {prof['pitch']['register']}, "
            f"{prof['pitch']['expressiveness']}, tempo {prof['tempo']['class']}"),
        "gender": gender,
        "pitch": {"median_hz": med, "range_global_st": prof["pitch"]["range_global_st"],
                  "range_phrase_st": prof["pitch"]["range_per_phrase_st"],
                  "peak_position": prof["intonation"].get("peak_position_median") or 0.3},
        "tempo": {"articulation_syl_s": art},
        "pauses": {"sentence_s": round(max(0.15, sent_pause), 2),
                   "paragraph_s": round(max(0.4, p90 * 1.6), 2),
                   "ellipsis_s": round(max(0.5, p90 * 2.0), 2)},
        "endings": {k: round(v / tot, 2) for k, v in endings.items()},
        "levels": levels,
        "style_words": prof.get("style", {}).get("markers", {}),
        "avg_sentence_words": prof.get("style", {}).get("avg_sentence_words"),
    }
    STYLES_DIR.mkdir(exist_ok=True)
    dst = STYLES_DIR / f"{a.name}.json"
    write_json(dst, style)
    print(f"OK style saved: {dst}")


if __name__ == "__main__":
    main()
