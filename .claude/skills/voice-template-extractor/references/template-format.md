# VOICE_TEMPLATE.md format

Write in the user's language. Keep it scannable: the user will reopen this file every time they
make a new video, so the most-used parts (settings, style rules, blank script template) must be
easy to find. Numbers come from `prosody.json` / `voice_profile.json`; interpretation is yours.

```markdown
# Template Suara — <nama/sumber>

> <one-sentence description of the voice, e.g. "Suara pria rendah-hangat, gaya ngobrol santai ala
> Jakarta, cepat dan mengalir, antusias saat bahas fitur.">

## 1. File untuk di-upload
- Voice clone: `audio/cloning_sample.mp3` (<durasi> ucapan bersih, musik latar sudah dibuang?)
- Alternatif/klip terbaik: <list 2–4 clip files that are cleanest/most representative>
- Referensi zero-shot: `audio/reference/neutral.wav` + `.txt` (and energetic/calm)
- Catatan kualitas: <warnings that matter + fix>

## 2. Setting ElevenLabs
| Model | Setting |
|---|---|
| Eleven v3 (emosi) | Stability: <mode> |
| Multilingual v2 (narasi stabil) | Stability <x> · Similarity <x> · Style <x> · Speaker boost ON · Speed 1.0 |
Voice Design prompt (kalau tidak meng-clone): "<voice_design_prompt, refined>"

## 3. Karakter suara (hasil ukur)
| Aspek | Nilai | Artinya |
|---|---|---|
| Nada dasar | <Hz> | <register> |
| Rentang nada | <st> | <monoton/ekspresif + what it means> |
| Tempo | <syl/s> | <class; compare to normal ~5–6 syl/s> |
| Jeda | <per minute, median> | <how they pause> |
| Akhir kalimat | naik x% / turun y% / datar z% | <pattern> |
| Warna suara | <brightness, texture> | |

## 4. Pola intonasi & emosi
- Pola kalimat: <e.g. starts high, falls; peak position>
- Peta energi video: <arc: opening → story → product → CTA with energy level of each part>
- Kapan naik energinya: <topics/moments>
- Kapan turun/tenang: <topics/moments>
- Penekanan: <what kinds of words get stressed>

## 5. Gaya bahasa (untuk nulis naskah baru)
- Kata ganti: <gue/aku/...>
- Kata khas & frekuensi: <markers>
- Panjang kalimat: <avg words>, <how they chain sentences>
- Code-mixing: <English terms they use>
- 5 aturan utama: <numbered, concrete>

## 6. Transkrip beranotasi (contoh hasil)
<the corrected transcript, tagged for v3: tags at emotional turns, ellipses at real pauses, CAPS for
stressed words, grouped into paragraphs by section of the video>

## 7. Template naskah kosong
<a fill-in-the-blanks skeleton that follows the same energy arc and structure, with tag positions
already placed, e.g.:
[warm] <Pembuka personal: konteks diri, 1–2 kalimat panjang dengan "jadi"/"terus">...
[curious] <Masalah/pertanyaan>?
[excited] Nah, <produk/inti> ini <manfaat utama> — <detail>!
...
[sincere] <Penutup + CTA hangat>.>

## 8. Cara pakai berulang
1. Tulis naskah baru → minta Claude "ubah ke gaya template <nama>" (Mode B).
2. Generate per paragraf, voice & setting sama (API: seed sama).
3. Dengar ulang; kalau terlalu datar turunkan stability 0.05, kalau aneh/berlebihan naikkan.
```
