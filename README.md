# audio-radit

Skill Claude Code untuk membuat **template suara text-to-speech** dari video. Skill ini mengekstrak suara
yang bersih untuk voice clone (ElevenLabs dan lainnya), lalu menganalisis **nada, intonasi, tempo, jeda,
dan emosi** pembicara supaya hasil TTS terdengar natural dan tidak kaku.

Skill: [`.claude/skills/voice-template-extractor`](.claude/skills/voice-template-extractor/SKILL.md)

## Pakai
- Buka repo ini di Claude Code (skill otomatis terbaca), atau salin foldernya ke `~/.claude/skills/`.
- Upload video, lalu tulis misalnya: *"ekstrak suara dari video ini jadi template TTS ElevenLabs"*.
- Untuk konten berikutnya: *"ubah naskah ini ke gaya template <nama>"*.

## Hasil per video (`voice-output/<nama>/`)
| File | Isi |
|---|---|
| `audio/cloning_sample.mp3` | Sampel suara bersih (musik dibuang) untuk Instant Voice Clone |
| `audio/clips/`, `audio/reference/` | Klip per frasa, dan klip referensi + teks untuk TTS zero-shot |
| `VOICE_TEMPLATE.md` | Setting ElevenLabs, karakter suara, pola intonasi & emosi, gaya bahasa, transkrip bertag, template naskah |
| `voice_profile.json`, `prosody.json` | Data ukur (dipakai ulang untuk menulis naskah baru) |
| `prosody_plot.png` | Grafik kontur nada & energi |

## Setup manual
```bash
python .claude/skills/voice-template-extractor/scripts/setup_env.py --separator --sherpa-whisper
```
Butuh `ffmpeg` dan Python 3.10+.

Gunakan hanya untuk suara milik sendiri atau suara yang pemiliknya sudah memberi izin.
