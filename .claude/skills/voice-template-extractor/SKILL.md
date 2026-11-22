---
name: voice-template-extractor
description: Extract a speaker's speaking style from a video or audio file into a reusable text-to-speech template — pitch, intonation, tempo, pauses and emotion — and turn scripts into natural Indonesian voice-overs with a built-in offline TTS (no ElevenLabs needed), or prepare clean samples and settings for ElevenLabs. Also writes new scripts in that style with emotion tags. Use this whenever the user uploads a video/audio (mp4, mov, mp3, wav, Instagram/TikTok/YouTube download) and wants a voice template, "ambil suaranya", "ekstrak audio buat TTS", "voice clone", "biar suaranya natural / tidak kaku", to copy a speaking style, tone, intonation or emotion, or when they want a voice-over / "isi suara" / "text to speech" / "bikin VO" for a video or script, especially using a style or template made earlier (styles/*.json, voice_profile.json, VOICE_TEMPLATE.md).
---

# Voice Template Extractor

Turns a talking video into reusable pieces for every future video (and, with Mode C, speaks new
scripts in that style offline):

1. **The voice** — a cleaned, music-free, loudness-normalised speech sample ready to upload as an
   ElevenLabs Instant Voice Clone (works for other cloning TTS too).
2. **The delivery** — how this person *speaks*: pitch level and range, how phrases rise and fall,
   tempo, pause habits, energy changes, emphasis, and their colloquial word choices. Cloning copies
   timbre; delivery is what makes output sound human. TTS sounds "kaku" mostly because scripts are
   written like formal text and settings are too stable — the template fixes both.

Talk to the user in their language (most users of this skill write Indonesian). Keep technical terms
light; explain a number only when it changes what they should do.

## Consent first (one short line, not a lecture)

Cloning someone's voice needs the owner's permission, and ElevenLabs makes the uploader confirm they
have the rights. If the source looks like someone else's content (an influencer/ad download, a
celebrity), say once that the clone sample should only be uploaded with that person's permission,
and that the *style* part of the template (intonation, pace, emotion, script style) can still be
applied to their own cloned voice or a library/Voice-Design voice. Then continue — the analysis
itself is fine either way.

## Setup (once per machine)

```bash
S=<this skill dir>/scripts
python $S/setup_env.py --separator --sherpa-whisper
```

Installs Python deps (librosa, praat-parselmouth, sherpa-onnx, onnxruntime…) and downloads into
`~/.cache/voice-template/`: Silero VAD, an MDX-Net vocal separator (~65 MB) and a Whisper-small ONNX
fallback transcriber (all from GitHub, so it works where HuggingFace is blocked). If HuggingFace is
reachable, `--with-faster-whisper` gives better transcripts with word timestamps (enables emphasis
detection). Needs `ffmpeg`.

## Mode A — build a template from a video

Pick an output folder named after the speaker/source, e.g. `voice-output/<name>/`. Run the three
steps in order (each is idempotent; rerun any step alone):

```bash
python $S/extract_voice.py INPUT.mp4 --out OUT        # ~1–2 min per 3 min of video if music is removed
python $S/transcribe.py OUT --lang id                 # auto: faster-whisper, else sherpa-onnx
python $S/analyze_prosody.py OUT
```

What each step does and why:
- **extract_voice.py** — pulls mono 44.1 kHz audio, detects background music (tonal energy in the
  gaps between phrases) and, if found, strips it with the vocal separator. Then a gentle clean-up:
  high-pass, *light* denoise only when needed (heavy denoise makes clones metallic), and **linear**
  loudness normalisation to −20 LUFS — dynamic normalisation would flatten the very emotion we want
  to keep. Writes `audio/cloning_sample.{wav,mp3}` (speech only, long silences trimmed to ≤0.6 s,
  max 3 min), phrase clips, `segments.json`, `quality.json`. Use `--separate yes|no` to force.
- **transcribe.py** — phrase-level transcript with timestamps.
- **analyze_prosody.py** — per phrase: pitch relative to the speaker's own baseline (semitones),
  pitch range, ending contour (naik/turun/datar), articulation rate (syllables/s), pauses, loudness,
  a speaker-relative arousal score → energy level (tenang…sangat antusias) and draft ElevenLabs tags.
  Globally: register, expressiveness, tempo class, pause habits, timbre, colloquial markers, and
  recommended TTS settings. Writes `prosody.json`, `voice_profile.json`, `annotated_draft.md`,
  `prosody_plot.png`, and `audio/reference/{neutral,energetic,calm}.{wav,txt}`.

### Then do the part scripts can't: interpret and write `OUT/VOICE_TEMPLATE.md`

Read `quality.json`, `prosody.json` (profile + phrases), `annotated_draft.md`, and look at
`prosody_plot.png`. Then write the template following `references/template-format.md`. The key
judgement calls:

- **Fix the transcript.** ASR mangles brand names, English loanwords and slang ("Amajon" →
  "Menurut gue", "penuaat" → "penuaan"). Correct obvious errors from context in the template and in
  `audio/reference/*.txt` (zero-shot TTS needs the reference text to match the audio exactly). Never
  invent content you can't infer.
- **Emotion needs meaning, not just acoustics.** The arousal score only says "more/less energetic
  than this speaker's average". Combine it with what's being said: a high-energy line about a
  product feature is *excited/enthusiastic*; a low-energy personal line is *reflective/sincere*; a
  rising ending on a question is *curious*; a closing CTA often turns *warm*. Pick ElevenLabs v3 tags
  accordingly (see `references/elevenlabs-guide.md` for the tag list).
- **Describe the delivery pattern in plain words** a script writer can follow: e.g. "each sentence
  starts high and falls to the end (peak at ~30% of the phrase)", "barely pauses — ~0.25 s between
  phrases, run-on sentences joined with *jadi / terus / nah*", "energy lifts on product benefits,
  drops for personal story".
- **Quality caveats.** If music was separated, voice-quality numbers (HNR, jitter) are less
  reliable — say "may sound slightly breathy" rather than stating it as fact. Surface every warning
  from `quality.json` that affects clone quality (too little speech, clipping, residual music) with
  the concrete fix.

Finish by telling the user: which file to upload, the exact settings to use, and the 3–5 style
rules that matter most. Offer Mode B.

## Mode B — write / convert a script in the template's style

When the user gives a topic or a plain script and has a template (`voice_profile.json` and/or
`VOICE_TEMPLATE.md`), read both and produce TTS-ready text:

1. Rewrite into the speaker's register: their pronouns (gue/aku/saya), discourse markers at the
   frequency the profile shows, average sentence length, and how they link sentences. Don't overdo
   markers — a natural rate is what's in the profile, not one per sentence.
2. Mark delivery: v3 audio tags at emotional turns (not every line — every 2–4 sentences is
   plenty), ellipses or commas where they pause, CAPS sparingly for the words they'd stress,
   question marks for rising endings. For multilingual_v2 use `<break time="0.4s" />` instead
   (rules in `references/elevenlabs-guide.md`).
3. Split long scripts into chunks of ~2–4 sentences (≤ ~800 characters) at natural paragraph
   breaks; generate chunk by chunk with the same voice + settings (and the same seed via API) so
   repeated content stays consistent.
4. Output: the tagged script, the settings block, and a one-line note on what energy arc you
   followed.

Read `references/elevenlabs-guide.md` before writing tags or settings — the rules differ by model.

## Mode C — make the voice-over offline (no ElevenLabs)

`speak.py` turns a narration script into a finished voice-over. A multi-speaker Indonesian VITS
model (83 voices, trained on open Javanese/Sundanese speech datasets) does the speaking. The
stored **style** then reshapes it with Praat PSOLA:
- pitch level and range;
- the speaker's phrase shape (starts high, glides down);
- endings: rising on `?`, held up mid-story, falling at the end of a paragraph;
- each emotion tag's pitch/tempo/loudness, blended with what the analysed speaker actually did
  at that energy level;
- the pause rhythm.

Whisper listens to every sentence. Up to 4 takes are made, and the clearest one is kept.

1. **Save the style once** after Mode A: `python $S/make_style.py OUT --name <style-name>`. This
   writes `styles/<style-name>.json`. It holds only numbers (no audio, no voice identity), so
   it's the way to reuse a creator's *delivery* without cloning their voice.
   `styles/pria-jakarta-santai.json` ships with the skill.
2. **Write the narration** with Mode B. Save it as a plain text file that contains **only the
   spoken lines**, with v3-style tags. Keep notes and settings out of it; markdown headings and
   bullets are skipped anyway.
3. **Respell hard words** in a lexicon JSON. Foreign names, acronyms and brand names usually
   need it. Plain respelling (`"Descartes": "Dekart"`) or raw phonemes with a leading slash
   (`"Descartes": "/dekart"`). The phoneme set is `a b d e f h i j k l m n o p r s t u w x z ŋ ɔ ə ɛ ɡ ɪ ɲ ʃ ʊ ʒ ʔ`,
   with `tʃ` = c and `dʒ` = j. Common slang (gue, gak, banget…) is built in.
4. **Generate and check:**
   ```bash
   python $S/speak.py narasi.txt --out OUT/voiceover.wav --style pria-jakarta-santai --lexicon lexicon.json
   python $S/check_tts.py OUT/voiceover.wav
   ```
   This writes `voiceover.wav/.mp3` (−16 LUFS, ready for social video), `voiceover.timing.json`
   (sentence and word timings, for animating and captions) and `voiceover.srt`.
5. **Read the check, since you can't listen.** Compare pitch/tempo with the style, then look at
   the per-sentence clarity scores. For any line flagged `kurang jelas` or `<-- cek`, rephrase
   it or add a lexicon entry, then regenerate. ASR also mishears respelled foreign names, so a
   low score on a name alone is not a defect. Expect about 0.8 average clarity.
6. **Options:**
   - `--speaker` picks another voice: `--list-speakers` shows them all. Clear male voices:
     JV-01932 (default), JV-00027, SU-01552. A female style defaults to JV-01519.
   - `--seed` gives repeatable output for the same text.
   - `--takes 1` is faster but has no clarity selection.

Supported emotion tags: excited, surprised, enthusiastic, happy, playful, confident, curious,
serious, matter-of-fact, warm, sincere, reassuring, calm, thoughtful, sad, whispers, softly,
quickly, slowly, emphasized, conversational tone. Non-verbal tags ([laughs], [sighs]) are
skipped. Tell the user the honest trade-off: this is clearly a synthetic voice. It keeps the
style's rhythm and emotion, but it is not as lifelike as ElevenLabs.

Setup for Mode C: `python $S/setup_env.py --tts`. This makes its own venv with
torch + Coqui TTS (several GB) and downloads the model from GitHub. `speak.py` re-launches
itself inside that venv automatically. The model weights come from a community GitHub release,
so for commercial use, check its license and the licenses of the underlying datasets.

## Troubleshooting

- *Music still audible in the sample* → the separator leaves faint bleed on dense mixes; tell the
  user to listen to `audio/clips/` and drop bad ones, or combine more videos with cleaner audio.
- *Only a short clip of speech (<60 s)* → the clone will be unstable; ask for more videos of the
  same speaker and run extract on each, then concatenate the `cloning_sample.wav` files with ffmpeg.
- *Multiple speakers in the video* → the pipeline assumes one speaker. Ask which person, then cut
  the relevant time ranges with ffmpeg before running Mode A.
- *Transcription falls back to sherpa-onnx* → fine; emphasis words just won't be detected
  automatically — infer stress from the plot and meaning instead.
