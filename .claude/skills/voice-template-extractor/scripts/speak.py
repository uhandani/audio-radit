#!/usr/bin/env python3
"""Offline Indonesian text-to-speech that speaks in a stored *style* (no ElevenLabs needed).

Usage:
  python speak.py SCRIPT.(md|txt) --out OUT.wav [--style pria-jakarta-santai]
                  [--speaker JV-01932] [--lexicon extra.json] [--seed 7]

SCRIPT format (same as Mode B scripts):
  - plain narration; paragraphs separated by blank lines. Markdown headings, bullet/numbered
    lists, tables, quotes, **label** lines and ``` fence markers are skipped, so keep notes in
    those forms or (better) give speak.py a file containing only the narration
  - [tag] sets the emotion for the following text: excited, enthusiastic, curious, playful,
    serious, matter-of-fact, calm, thoughtful, warm, sincere, sad, whispers, quickly, slowly ...
    (non-verbal tags like [laughs] are skipped)
  - '...' = a longer pause, '?' = rising ending, WORD IN CAPS = stressed word

Outputs: OUT.wav (44.1 kHz, -16 LUFS), OUT.mp3, OUT.timing.json (sentence + word timings),
OUT.srt (subtitles).

How it works: a multi-speaker Indonesian VITS model (83 voices) synthesises each sentence;
then the stored style is imposed with Praat PSOLA — pitch level and range, the speaker's phrase
shape (start high, glide down), ending contours, per-emotion pitch/tempo/loudness taken from the
analysed recording, and its pause rhythm.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("VOICE_TEMPLATE_CACHE", Path.home() / ".cache" / "voice-template"))
VENV_PY = CACHE / "venv-tts" / "bin" / "python"

try:  # heavy deps live in a dedicated venv (created by setup_env.py --tts)
    import torch  # noqa: F401
    import TTS  # noqa: F401
except Exception:
    if VENV_PY.exists() and Path(sys.prefix).resolve() != VENV_PY.parent.parent.resolve():
        os.execv(str(VENV_PY), [str(VENV_PY), "-I", str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.exit("TTS engine belum terpasang. Jalankan: python setup_env.py --tts")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import unicodedata  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

MODEL_DIR = CACHE / "id-tts-vits"
STYLES = HERE.parent / "styles"
SR = 22050
HOP = 256
DEFAULT_SPEAKER = {"male": "JV-01932", "female": "JV-01519"}

# emotion tag -> (pitch shift st, range x, speed x, gain dB, pause x, analysed level)
EMOTIONS = {
    "neutral": (0.0, 1.00, 1.00, 0.0, 1.0, "netral"),
    "conversational tone": (0.0, 1.00, 1.00, 0.0, 1.0, "netral"),
    "conversational": (0.0, 1.00, 1.00, 0.0, 1.0, "netral"),
    "excited": (2.5, 1.35, 1.07, 2.0, 0.8, "sangat antusias"),
    "surprised": (3.0, 1.40, 1.04, 1.5, 1.0, "sangat antusias"),
    "enthusiastic": (1.5, 1.20, 1.04, 1.0, 0.9, "antusias"),
    "happy": (1.5, 1.20, 1.03, 1.0, 0.9, "antusias"),
    "playful": (1.5, 1.25, 1.02, 0.5, 0.9, "antusias"),
    "confident": (0.5, 1.05, 1.00, 1.0, 0.9, "netral"),
    "curious": (1.0, 1.15, 1.00, 0.0, 1.0, "netral"),
    "serious": (-1.0, 0.85, 0.97, 0.0, 1.1, "santai"),
    "matter-of-fact": (-0.5, 0.85, 1.04, 0.0, 0.9, "santai"),
    "warm": (-0.5, 0.95, 0.95, 0.0, 1.1, "santai"),
    "sincere": (-0.8, 0.90, 0.93, -0.5, 1.15, "santai"),
    "reassuring": (-0.5, 0.90, 0.95, 0.0, 1.1, "santai"),
    "calm": (-1.5, 0.80, 0.92, -1.5, 1.2, "tenang"),
    "thoughtful": (-1.0, 0.90, 0.90, -1.0, 1.3, "tenang"),
    "sad": (-2.5, 0.70, 0.85, -2.5, 1.4, "tenang"),
    "whispers": (-0.5, 0.80, 0.90, -6.0, 1.1, "tenang"),
    "softly": (-0.5, 0.85, 0.93, -4.0, 1.1, "tenang"),
    "quickly": (0.5, 1.00, 1.12, 0.0, 0.8, "netral"),
    "slowly": (-0.5, 0.95, 0.88, 0.0, 1.2, "santai"),
    "emphasized": (1.0, 1.15, 0.97, 1.5, 1.0, "antusias"),
}

# spoken forms: "/..." = raw phonemes, otherwise a respelling read by the G2P
DEFAULT_LEXICON = {
    "gue": "/ɡuə", "gua": "/ɡua", "gw": "/ɡuə", "lu": "/lu", "lo": "/lo",
    "gak": "/ɡaʔ", "nggak": "/ŋɡaʔ", "enggak": "/əŋɡaʔ", "ga": "/ɡa", "kok": "/koʔ",
    "banget": "/baŋət", "dapet": "/dapət", "bener": "/bənər", "pake": "/pake",
    "deh": "/deh", "dong": "/doŋ", "sih": "/sih", "nih": "/nih", "tuh": "/tuh",
}


def log(m: str) -> None:
    print(f"[speak] {m}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- text parsing -------
def parse_script(raw: str):
    """-> list of paragraphs; each paragraph = list of (emotion, text) pieces."""
    lines, fence = [], False
    for ln in raw.splitlines():
        s = ln.strip()
        if s.startswith("```"):
            fence = not fence
            continue
        if (s.startswith(("#", ">", "- ", "* ", "|", "<!--")) or re.match(r"^\d+\.\s", s)
                or (s.startswith("**") and s.endswith("**"))):
            lines.append("")
            continue
        lines.append(s)
    paras = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(lines)) if p.strip()]
    out = []
    for p in paras:
        p = re.sub(r"\s+", " ", p).replace("…", "...")
        pieces, emo = [], "neutral"
        for tok in re.split(r"(\[[^\]]+\])", p):
            if tok.startswith("["):
                tag = tok[1:-1].strip().lower()
                if tag in EMOTIONS:
                    emo = tag
                else:
                    log(f"tag [{tag}] dilewati (tidak didukung TTS offline)")
                continue
            if tok.strip():
                pieces.append((emo, tok.strip()))
        if pieces:
            out.append(pieces)
    return out


def split_units(text: str):
    """Split a piece into sentence-ish units. Returns (text, end_kind)."""
    parts = re.split(r"(?<=[.!?])\s+|(?<=\.\.\.)\s*", text)
    units = []
    for u in parts:
        u = u.strip()
        if not u:
            continue
        if u.endswith("..."):
            units.append((u[:-3].rstrip(" ,") + ",", "ellipsis"))
        elif u[-1] in ".!?":
            units.append((u, u[-1]))
        else:
            units.append((u.rstrip(",") + ",", "continue"))
    return units


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


class Phonemizer:
    ALLOWED = set("abdefhijklmnoprstuwxzŋɔəɛɡɪɲʃʊʒʔˈ")
    FIX = {"v": "f", "g": "ɡ", "c": "tʃ", "y": "j", "q": "k"}

    def __init__(self, lexicon: dict[str, str]):
        from g2p_id import G2p
        self.g2p = G2p()
        self.lex = {k.lower(): v for k, v in lexicon.items()}

    def word(self, w: str) -> str:
        key = re.sub(r"[^\w\-']", "", w.lower())
        spoken = self.lex.get(key, w)
        if spoken.startswith("/"):
            ph = spoken[1:]
        else:
            ph = " ".join("".join(x) for x in self.g2p(strip_accents(spoken).lower())
                          if "".join(x) not in ",.?!")
        ph = "".join(self.FIX.get(c, c) for c in ph)
        # g2p-id puts a glottal stop between vowels (dua -> duʔa); this voice renders it as a
        # hard "k" (diketahui -> "diketahuki"), so drop it inside words
        ph = re.sub(r"(?<=[aiueoəɛɔɪʊ])ʔ(?=[aiueoəɛɔɪʊ])", "", ph)
        return "".join(c for c in ph if c in self.ALLOWED or c == " ").strip()

    def sentence(self, text: str):
        """-> phoneme string, list of (display_word, char_start, char_end) in phoneme string."""
        words = re.findall(r"[^\s]+", text)
        out, spans = "", []
        for w in words:
            punct = "".join(c for c in w if c in ",.?!")[-1:] if re.search(r"[,.?!]$", w) else ""
            ph = self.word(w)
            if not ph:
                continue
            if out:
                out += " "
            start = len(out)
            out += ph
            spans.append((w.strip(",.?!\"'()"), start, len(out)))
            out += punct
        return out, spans


# ---------------------------------------------------------------- model -------------
def load_model():
    import torch
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.models.vits import Vits
    cfg = VitsConfig()
    cfg.load_json(str(MODEL_DIR / "config.json"))
    cfg.model_args.speakers_file = str(MODEL_DIR / "speakers.pth")
    cfg.speakers_file = str(MODEL_DIR / "speakers.pth")
    model = Vits.init_from_config(cfg)
    state = torch.load(str(MODEL_DIR / "checkpoint_1260000-inference.pth"), weights_only=True,
                       map_location="cpu")["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def synth(model, phon: str, spans, speaker_id: int, length_scale: float, noise: float):
    """-> audio, word timings [(word, t0, t1)] in seconds."""
    import torch
    model.length_scale = length_scale
    model.inference_noise_scale = noise
    ids = model.tokenizer.text_to_ids(phon)
    with torch.no_grad():
        o = model.inference(torch.LongTensor(ids).unsqueeze(0),
                            aux_input={"speaker_ids": torch.LongTensor([speaker_id])})
    y = o["model_outputs"][0, 0].cpu().numpy().astype(np.float32)
    dur = o["durations"][0, 0].cpu().numpy()  # frames per token
    cum = np.concatenate([[0], np.cumsum(dur)]) * HOP / SR
    words = []
    for w, a, b in spans:
        ta = len(model.tokenizer.text_to_ids(phon[:a])) if a else 0
        tb = len(model.tokenizer.text_to_ids(phon[:b]))
        ta = min(max(ta - 1, 0), len(cum) - 1)  # eos/blank offsets: approximate is fine
        tb = min(tb, len(cum) - 1)
        words.append((w, float(cum[ta]), float(cum[tb])))
    return y, words


def count_syllables(text: str) -> int:
    n = 0
    for w in re.findall(r"[A-Za-zÀ-ÿ]+", text.lower()):
        n += max(1, len(re.findall(r"[aiueoy]+", re.sub(r"(ai|au|oi|ei)", "a", w))))
    n += sum(min(8, 2 * len(x) + 1) for x in re.findall(r"\d+", text))
    return n


def trim(y: np.ndarray, words, top_db=45):
    import librosa
    _, (a, b) = librosa.effects.trim(y, top_db=top_db, frame_length=512, hop_length=128)
    a = max(0, a - int(0.06 * SR))
    b = min(len(y), b + int(0.08 * SR))
    off = a / SR
    return y[a:b], [(w, max(0.0, t0 - off), max(0.0, t1 - off)) for w, t0, t1 in words]


class Judge:
    """Whisper (sherpa-onnx) listens to each take; we keep the clearest one."""

    def __init__(self, lexicon):
        import sherpa_onnx
        d = CACHE / "sherpa-onnx-whisper-small"
        self.rec = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(d / "small-encoder.int8.onnx"), decoder=str(d / "small-decoder.int8.onnx"),
            tokens=str(d / "small-tokens.txt"), language="id", task="transcribe", num_threads=4)
        self.skip = {k.lower() for k in lexicon}  # respelled names: ASR spells them its own way

    @staticmethod
    def norm(t):
        t = re.sub(r"\bx\b", "eks", t.lower())
        return re.sub(r"[^\w\s]", " ", strip_accents(t)).split()

    def score(self, y, text):
        import librosa
        st = self.rec.create_stream()
        try:
            st.set_option("max_tokens_per_second", "12")
        except Exception:
            pass
        st.accept_waveform(16000, librosa.resample(y, orig_sr=SR, target_sr=16000))
        self.rec.decode_stream(st)
        ref = [w for w in self.norm(text) if w not in self.skip]
        hyp = self.norm(st.result.text)
        if not ref:  # only respelled names: nothing reliable to compare against
            return 1.0 if hyp else 0.0
        import difflib
        return difflib.SequenceMatcher(None, ref, hyp).ratio()


# ---------------------------------------------------------------- prosody -----------
def f0_stats(y: np.ndarray):
    import parselmouth
    p = parselmouth.Sound(y.astype(np.float64), SR).to_pitch_ac(time_step=0.01, pitch_floor=60,
                                                                pitch_ceiling=500)
    f = p.selected_array["frequency"]
    return f[f > 0]


def restyle(y, style, emo, end_kind, para_final, emph_spans, base_med, k):
    """PSOLA: impose pitch level/range, phrase shape, ending and emotion; returns audio."""
    import parselmouth
    from parselmouth.praat import call

    shift, rng_x, _, gain, _, _ = emo
    snd = parselmouth.Sound(y.astype(np.float64), SR)
    manip = call(snd, "To Manipulation", 0.01, 60, 400)
    pt = call(manip, "Extract pitch tier")
    n = call(pt, "Get number of points")
    if n < 5:
        return y * 10 ** (gain / 20)
    ts = np.array([call(pt, "Get time from index", i) for i in range(1, n + 1)])
    fs = np.array([call(pt, "Get value at index", i) for i in range(1, n + 1)])
    t0, t1 = ts[0], ts[-1]
    u = (ts - t0) / max(1e-3, t1 - t0)

    target_med = style["pitch"]["median_hz"]
    level_shift = float(np.clip(12 * np.log2(target_med / base_med), -4, 4))
    rel = 12 * np.log2(fs / base_med)
    rel = rel - np.median(rel) * 0.5  # keep half of the sentence's own register offset
    range_x = rng_x * style["_range_ratio"]

    # phrase shape: begin fairly high, peak early, glide down (measured peak position)
    pk = float(np.clip(style["pitch"].get("peak_position", 0.3), 0.1, 0.7))
    D = 1.2 * k
    shape = np.where(u < pk, D * (0.4 + 0.6 * u / pk), D * (1 - 2 * (u - pk) / (1 - pk)))

    # endings (last 0.35 s)
    tail = np.clip((ts - (t1 - 0.35)) / 0.35, 0, 1)
    cont = style["endings"].get("datar", 0) + style["endings"].get("naik", 0)
    if end_kind == "?":
        ending = 3.5 * tail
    elif end_kind in (".", "!") and para_final:
        ending = -1.0 * tail
    elif end_kind in ("continue", "ellipsis") or (end_kind == "." and cont > 0.6):
        ending = (1.5 * k) * tail  # keep it "hanging": the speaker rarely drops mid-story
    else:
        ending = 0 * tail

    emph = np.zeros_like(ts)
    for a, b in emph_spans:
        emph += 2.0 * ((ts >= a) & (ts <= b))

    new = base_med * 2 ** ((rel * range_x + shape + ending + emph + shift + level_shift) / 12)
    new = np.clip(new, 55, 450)
    npt = call("Create PitchTier", "new", snd.xmin, snd.xmax)
    for t, f in zip(ts, new):
        call(npt, "Add point", float(t), float(f))
    call([npt, manip], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)")
    yy = out.values[0].astype(np.float32)
    # loudness: emotion gain + stressed words
    g = np.full(len(yy), 10 ** (gain / 20), np.float32)
    for a, b in emph_spans:
        g[int(a * SR):int(b * SR)] *= 10 ** (2.0 / 20)
    return yy * g


# ---------------------------------------------------------------- main --------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script")
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", default="pria-jakarta-santai", help="nama di styles/ atau path .json")
    ap.add_argument("--speaker", default=None, help="ID suara (lihat --list-speakers)")
    ap.add_argument("--lexicon", default=None, help="JSON kata->ucapan tambahan")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--takes", type=int, default=4,
                    help="maks. take per kalimat; take terjelas (menurut ASR) yang dipakai")
    ap.add_argument("--list-speakers", action="store_true")
    a = ap.parse_args()

    import torch
    torch.manual_seed(a.seed)
    model = load_model()
    spk_map = model.speaker_manager.name_to_id
    if a.list_speakers:
        print(json.dumps(sorted(spk_map), indent=0))
        return

    sp = Path(a.style)
    style = json.loads((sp if sp.suffix == ".json" else STYLES / f"{a.style}.json").read_text())
    speaker = a.speaker or style.get("default_speaker") or DEFAULT_SPEAKER[style.get("gender", "male")]
    sid = spk_map[speaker]
    k = float(np.clip(style["pitch"]["range_global_st"] / 10.0, 0.6, 1.4))  # expressiveness
    noise = 0.667 + 0.1 * (k - 1)

    lex = dict(DEFAULT_LEXICON)
    if a.lexicon:
        lex.update(json.loads(Path(a.lexicon).read_text()))
    ph = Phonemizer(lex)
    judge = None
    if a.takes > 1:
        try:
            judge = Judge(lex)
        except Exception as e:
            log(f"ASR judge tidak tersedia ({e}); 1 take per kalimat")
    paras = parse_script(Path(a.script).read_text())
    log(f"style={style['name']} speaker={speaker} paragraf={len(paras)}")

    P_ELLIPSIS = style["pauses"]["ellipsis_s"]
    # build unit list
    units = []
    for pi, para in enumerate(paras):
        flat = [(emo, t, kind) for emo, piece in para for t, kind in split_units(piece)]
        for ui, (emo, t, kind) in enumerate(flat):
            units.append({"para": pi, "emo": emo, "text": t, "kind": kind,
                          "para_final": ui == len(flat) - 1})

    merged = []
    for u in units:
        prev = merged[-1] if merged else None
        if (prev and prev["kind"] in ("ellipsis", "continue") and prev["para"] == u["para"]
                and len(prev["text"].split()) < 4 and not prev.get("joined")):
            gap = P_ELLIPSIS if prev["kind"] == "ellipsis" else 0.08
            u = dict(u, text=prev["text"] + " " + u["text"], emo=prev["emo"],
                     insert_pause=(len(prev["text"].split()) - 1, gap), joined=True)
            merged[-1] = u
        else:
            merged.append(u)
    units = merged

    def emo_params(name):
        s, r, v, g, p, lvl = EMOTIONS[name]
        s, r = s * k, 1 + (r - 1) * k * 0.7
        L = style.get("levels", {}).get(lvl)
        if L and name not in ("neutral", "conversational tone", "conversational"):
            # blend generic emotion with what this speaker actually did at that energy level
            s = 0.5 * s + 0.5 * L["pitch_rel_st"]
            v = 0.5 * v + 0.5 * float(np.clip(L["speed_rel"], 0.85, 1.15))
            g = 0.5 * g + 0.5 * L["loudness_rel_db"] * 2
        return (s, r, v, g, p, lvl)

    # calibrate base speed & pitch of the chosen voice on the first units
    cal_txt = " ".join(u["text"] for u in units[:4])
    phon, spans = ph.sentence(cal_txt)
    y, _ = synth(model, phon, spans, sid, 1.0, noise)
    y, _ = trim(y, [])
    f = f0_stats(y)
    base_med = float(np.median(f))
    base_rng = float(np.percentile(12 * np.log2(f), 90) - np.percentile(12 * np.log2(f), 10))
    rate1 = count_syllables(cal_txt) / (len(y) / SR)
    target = style["tempo"]["articulation_syl_s"]
    base_ls = float(np.clip(rate1 / target, 0.6, 1.4))
    style["_range_ratio"] = float(np.clip(style["pitch"]["range_phrase_st"] / max(3, base_rng), 0.7, 1.25))
    log(f"suara dasar {base_med:.0f} Hz, {rate1:.1f} suku kata/dtk -> length_scale {base_ls:.2f}, "
        f"range x{style['_range_ratio']:.2f}")

    # keep the script's overall median pitch on the style's median: emotions move pitch
    # around it instead of lifting the whole voice-over
    wts = np.array([len(u["text"]) for u in units], float)
    mean_shift = float(np.average([emo_params(u["emo"])[0] for u in units], weights=wts))

    chunks, timing, t = [], [], 0.0
    P = style["pauses"]
    for i, u in enumerate(units):
        emo = emo_params(u["emo"])
        text = u["text"]
        emph_words = [w.strip(",.?!") for w in text.split() if len(w.strip(",.?!")) > 1
                      and w.strip(",.?!").isupper() and not w.strip(",.?!").isdigit()]
        phon, spans = ph.sentence(text)
        if not phon:
            continue
        want = target * emo[2]
        ls = float(np.clip(base_ls / emo[2], 0.55, 1.5))
        y, words = synth(model, phon, spans, sid, ls, noise)
        y, words = trim(y, words)
        got = count_syllables(text) / max(0.3, len(y) / SR)
        if abs(got / want - 1) > 0.08 and count_syllables(text) >= 6:
            ls = float(np.clip(ls * got / want, 0.55, 1.5))
            y, words = synth(model, phon, spans, sid, ls, noise)
            y, words = trim(y, words)
        best = (judge.score(y, text) if judge else 1.0, y, words)
        for take in range(1, a.takes if judge else 1):
            if best[0] >= 0.9:
                break
            torch.manual_seed(a.seed + 1000 * take + i)
            y2, w2 = synth(model, phon, spans, sid, ls, noise)
            y2, w2 = trim(y2, w2)
            sc = judge.score(y2, text)
            if sc > best[0]:
                best = (sc, y2, w2)
        clarity, y, words = best
        if u.get("insert_pause"):
            wi, gap = u["insert_pause"]
            if wi < len(words):
                cut = int(words[wi][2] * SR)
                y = np.concatenate([y[:cut], np.zeros(int(gap * SR), np.float32), y[cut:]])
                words = [(w, t0 + (gap if j > wi else 0), t1 + (gap if j > wi else 0))
                         for j, (w, t0, t1) in enumerate(words)]
        emph_spans = [(t0, t1) for w, t0, t1 in words if w in emph_words]
        emo = (emo[0] - mean_shift,) + emo[1:]
        y = restyle(y, style, emo, u["kind"], u["para_final"], emph_spans, base_med, k)
        chunks.append(y)
        timing.append({"text": text.rstrip(","), "emotion": u["emo"], "clarity": round(clarity, 2),
                       "start": round(t, 3),
                       "end": round(t + len(y) / SR, 3), "paragraph": u["para"],
                       "words": [{"word": w, "start": round(t + t0, 3), "end": round(t + t1, 3)}
                                 for w, t0, t1 in words]})
        t += len(y) / SR
        # pause after
        nxt = units[i + 1] if i + 1 < len(units) else None
        if nxt is None:
            pause = 0.4
        elif nxt["para"] != u["para"]:
            pause = P["paragraph_s"]
        elif u["kind"] == "ellipsis":
            pause = P["ellipsis_s"]
        elif u["kind"] == "continue":
            pause = 0.08
        else:
            pause = P["sentence_s"] * emo[4] * (1.4 if u["kind"] == "?" else 1.0)
        chunks.append(np.zeros(int(pause * SR), np.float32))
        t += pause
        log(f"{i + 1}/{len(units)} [{u['emo']}] jelas={clarity:.2f} {text[:55]}")

    audio = np.concatenate(chunks)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".raw.wav")
    sf.write(str(raw), audio, SR, subtype="FLOAT")
    # linear loudness normalisation to -16 LUFS (social video), 44.1 kHz
    meas = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(raw), "-af",
                           "loudnorm=I=-16:TP=-1.5:LRA=20:print_format=json", "-f", "null", "-"],
                          capture_output=True, text=True).stderr
    m = json.loads(re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", meas, re.S).group(0))
    ln = (f"loudnorm=I=-16:TP=-1.5:LRA=20:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
          f"offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af", ln, "-ar", "44100",
                    "-c:a", "pcm_s16le", str(out)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out), "-b:a", "192k",
                    str(out.with_suffix(".mp3"))], check=True)
    raw.unlink()

    meta = {"style": style["name"], "speaker": speaker, "duration": round(t, 2), "sentences": timing}
    out.with_suffix(".timing.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    # subtitles: <= ~42 chars per cue using word timings
    def fmt(x):
        h, r = divmod(x, 3600)
        mnt, s = divmod(r, 60)
        return f"{int(h):02d}:{int(mnt):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"
    cues = []
    for s in timing:
        cur = []
        for w in s["words"]:
            cur.append(w)
            if len(" ".join(x["word"] for x in cur)) > 36 or w is s["words"][-1]:
                cues.append((cur[0]["start"], cur[-1]["end"], " ".join(x["word"] for x in cur)))
                cur = []
    srt = "\n".join(f"{i}\n{fmt(a_)} --> {fmt(b_)}\n{txt}\n" for i, (a_, b_, txt) in enumerate(cues, 1))
    out.with_suffix(".srt").write_text(srt)
    low = [x["text"][:50] for x in timing if x["clarity"] < 0.7]
    print(f"OK {out} ({t:.1f} s, {len(timing)} kalimat)")
    for x in low:
        print(f"WARN kurang jelas: {x}  -> ubah kata/ejaan atau tambah --lexicon")


if __name__ == "__main__":
    main()
