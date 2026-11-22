# ElevenLabs (and other TTS) — practical guide for natural delivery

Product details change; if the user reports a setting or tag that no longer exists, trust them and
adapt. The principles (clean sample, looser stability, write like speech) hold for any TTS.

## 1. Making the clone

| | Instant Voice Clone (IVC) | Professional Voice Clone (PVC) |
|---|---|---|
| Audio needed | ~1–3 min clean speech (more rarely helps) | 30 min minimum, 1–3 h ideal |
| Who can make it | anyone with rights to the voice (checkbox) | voice owner must verify by reading a prompt |
| Use | quick, good enough for most content | closest likeness, most consistent |

Sample rules (the extractor already handles most): one speaker, no music/SFX, little room echo,
consistent mic distance, no clipping, natural pauses kept short. **The clone copies the sample's
energy** — a sample that is all hype produces a hype voice even for calm text. If the user wants
a versatile voice, include both calm and energetic phrases (the default `cloning_sample` keeps the
whole performance in order, which is usually right).

Upload: Voices → Add voice → Instant Voice Clone → upload `audio/cloning_sample.mp3` (or several
`audio/clips/*.wav`). Name it, add a description (use the template's one-line voice description).

## 2. Models

- **Eleven v3** — most expressive; understands **audio tags** in square brackets. Best for
  content-creator style, emotion shifts, laughs. Slightly less predictable; generate 2–3 takes.
- **Multilingual v2** — very stable long-form narration, supports Indonesian; emotion comes only
  from the text + settings; supports SSML `<break>`.
- **Flash / Turbo** — low latency, less nuance. Avoid for final voice-overs.

## 3. Settings

Multilingual v2 sliders (0–1):
- **Stability** — lower = more variation and emotion, higher = monotone and consistent.
  Expressive speakers: 0.30–0.40. Moderate: 0.45–0.55. Flat/announcer: 0.60+. Below ~0.3 risks
  odd jumps.
- **Similarity** — how hard it sticks to the sample. 0.75–0.85 for clean samples; drop to
  ~0.65–0.7 if the sample has noise/music residue (high similarity reproduces artifacts).
- **Style exaggeration** — amplifies the sample's style. 0–0.45; above that it gets theatrical and
  slower to generate.
- **Speaker boost** — on.
- **Speed** — 0.7–1.2. A clone already has the speaker's pace: keep 1.0, adjust ±0.05.

Eleven v3: Stability has three modes — **Creative** (most emotional, can hallucinate), **Natural**
(balanced; default for clones of expressive speakers), **Robust** (stable, less responsive to tags).

API extras for repeated content: `seed` (same seed + same text → near-identical take),
`previous_text` / `next_text` (keeps intonation continuous across chunks).

## 4. Audio tags (v3 only)

Write tags in English even when the text is Indonesian. Place the tag right before the words it
should colour; it lasts roughly until the next sentence or tag.

- Emotion / attitude: `[excited]` `[happy]` `[curious]` `[sarcastic]` `[serious]` `[calm]`
  `[warm]` `[sincere]` `[thoughtful]` `[confident]` `[playful]` `[surprised]` `[sad]`
  `[frustrated]` `[nervous]` `[reassuring]`
- Delivery: `[whispers]` `[softly]` `[shouts]` `[slowly]` `[quickly]` `[emphasized]`
  `[matter-of-fact]` `[conversational tone]`
- Non-verbal: `[laughs]` `[chuckles]` `[light laugh]` `[sighs]` `[exhales]` `[clears throat]`
  `[gasps]` `[hmm]`

Rules of thumb:
- One tag per emotional turn, not per sentence. Over-tagging sounds acted.
- A tag must fit the voice: a calm narrator clone can't really `[shouts]`.
- Pauses in v3: punctuation and ellipses (`...`), line breaks. v3 does **not** support SSML
  `<break>` — it may read it out or ignore it.
- Emphasis: CAPITALS on one word (`ini CAKEP banget`), sparingly.
- Rising question intonation: end with `?`. Trailing thought: `...`. Energy pop: `!` (once or twice
  per paragraph at most).

## 5. Multilingual v2 markup

- Pause: `<break time="0.4s" />` (max 3 s). More than a few per generation can cause instability —
  prefer commas/periods for short pauses and breaks only for deliberate beats.
- No audio tags (they'd be read aloud). Emotion comes from wording: exclamations, rhetorical
  questions, short punchy sentences after long ones.

## 6. Writing so TTS doesn't sound "kaku"

- Write how the person talks, not how they'd write: contractions and colloquial forms
  (*udah, gak, banget, kayak, gimana*) at the frequency the template shows.
- Vary sentence length: long explanatory run-ons broken by short punchlines.
- Numbers and abbreviations: spell how they're said (`24 jam` fine; `ABPM` → `A-B-P-M` if spelled
  out; `D3` → `D-tiga` if TTS mispronounces). Test brand names once and adjust spelling.
- Start new topics with the speaker's own transition words (*nah, jadi, terus, apalagi*).
- Keep each generation to one coherent paragraph so intonation arcs feel complete.

## 7. Other TTS / open models

- Zero-shot cloners (F5-TTS, Fish Speech/OpenAudio, CosyVoice, XTTS…) want a short reference clip +
  its **exact** transcript → use `audio/reference/<neutral|energetic|calm>.wav/.txt`. Choose the
  reference whose energy matches the line you're generating — this is how those models "copy
  emotion".
- Platforms with emotion dropdowns (MiniMax, etc.) → map template energy levels: tenang/santai →
  neutral/calm, antusias → happy, sangat antusias → excited/surprised.
