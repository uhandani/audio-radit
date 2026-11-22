# audio-radit

Skill Claude Code untuk membuat **template suara text-to-speech** dari video. Skill ini mengekstrak suara
yang bersih untuk voice clone (ElevenLabs dan lainnya), lalu menganalisis **nada, intonasi, tempo, jeda,
dan emosi** pembicara supaya hasil TTS terdengar natural dan tidak kaku.

Skill: [`.claude/skills/voice-template-extractor`](.claude/skills/voice-template-extractor/SKILL.md)

## Text-to-speech offline (tanpa ElevenLabs)
Gaya bicara yang sudah dianalisis disimpan di `styles/` (hanya angka: nada, intonasi, tempo, jeda, emosi).
Gaya itu dipakai untuk membacakan naskah baru dengan suara TTS bahasa Indonesia yang berjalan di komputer sendiri:

```bash
S=.claude/skills/voice-template-extractor/scripts
python $S/setup_env.py --tts                       # sekali saja
python $S/speak.py narasi.txt --out vo.wav --style pria-jakarta-santai --lexicon lexicon.json
python $S/check_tts.py vo.wav                      # cek nada/tempo + kejelasan
```
Hasilnya `vo.wav`/`vo.mp3`, subtitle `vo.srt`, dan `vo.timing.json` (waktu per kata untuk animasi video).

## Pakai
- Buka repo ini di Claude Code (skill otomatis terbaca), atau salin foldernya ke `~/.claude/skills/`.
- Upload video, lalu tulis misalnya: *"ekstrak suara dari video ini jadi template TTS ElevenLabs"*.
- Untuk konten berikutnya: *"ubah naskah ini ke gaya template <nama>"*, atau *"bikin voice-over naskah ini pakai gaya pria-jakarta-santai"*.

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
