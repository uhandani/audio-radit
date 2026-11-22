---
name: voice-template-extractor
description: Extract a speaker's voice from a video or audio file into a reusable text-to-speech voice template — a clean cloning sample for ElevenLabs (or any voice-clone TTS) plus a measured profile of their pitch, intonation, tempo, pauses and emotion, so generated voice-overs sound natural instead of flat. Also writes new scripts in that same speaking style with ElevenLabs audio tags. Use this whenever the user uploads a video/audio (mp4, mov, mp3, wav, Instagram/TikTok/YouTube download) and wants to clone the voice, make a voice template, "ambil suaranya", "ekstrak audio buat TTS", "voice clone", "biar suaranya natural / tidak kaku", copy someone's speaking style, tone, intonation or emotion, or when they want to turn a script into TTS-ready text that matches a voice template they made earlier (voice_profile.json / VOICE_TEMPLATE.md).
---

# Voice Template Extractor

Turns a talking video into two things the user can reuse for every future video:

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

## Troubleshooting

- *Music still audible in the sample* → the separator leaves faint bleed on dense mixes; tell the
  user to listen to `audio/clips/` and drop bad ones, or combine more videos with cleaner audio.
- *Only a short clip of speech (<60 s)* → the clone will be unstable; ask for more videos of the
  same speaker and run extract on each, then concatenate the `cloning_sample.wav` files with ffmpeg.
- *Multiple speakers in the video* → the pipeline assumes one speaker. Ask which person, then cut
  the relevant time ranges with ffmpeg before running Mode A.
- *Transcription falls back to sherpa-onnx* → fine; emphasis words just won't be detected
  automatically — infer stress from the plot and meaning instead.
