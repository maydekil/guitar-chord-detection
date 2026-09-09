# 07 — Desktop Application

## Stack

```text
Electron
React
TypeScript
Vite
```

Package manager preferred:

```text
pnpm
```

---

## MVP Screens

Satu screen utama sudah cukup.

Layout:

```text
┌──────────────────────────────────────┐
│ Guitar Chord Detector                │
├──────────────────────────────────────┤
│ [ Open Audio ]                       │
│                                      │
│ song-name.mp3                        │
│                                      │
│              Am                      │
│                                      │
│ ──────────────────────────────────── │
│ C       G       Am       F           │
│                                      │
│ ▶  01:21 / 03:42                     │
│ ─────────────●────────────────────── │
└──────────────────────────────────────┘
```

---

## Required UI State

```text
idle
loading-file
analyzing
ready
playing
paused
error
```

---

## Open Audio

User click:

```text
Open Audio
```

Native dialog.

Allowed:

```text
.mp3
.wav
```

---

## Analysis Behavior

Setelah file dipilih:

```text
select
↓
load metadata
↓
check analysis cache
↓
if cache missing:
    run engine
↓
display timeline
```

Normal Open Audio flow boleh menggunakan cache.

### Re-analyze (force refresh)

Desktop shell harus menyediakan tombol:

```text
Re-analyze
```

Aturan:

- hanya tampil/aktif jika file audio valid sudah dipilih;
- bypass cache lookup;
- jalankan engine lagi untuk file yang sedang dipilih;
- replace cache entry untuk cache key file tersebut dengan hasil baru jika success;
- clear analysis/timeline yang sedang tampil segera saat re-analysis dimulai;
- set state ke `analyzing` saat request berjalan;
- stale/older response tidak boleh menimpa hasil request terbaru;
- jika success, state menjadi `ready`;
- jika failure, stale result tetap cleared dan UI masuk controlled `error`.

### Cache Identity

Cache key wajib mencakup:

```text
SHA-256 file content
analysis contract version
analysis algorithm version
```

Tidak boleh menggunakan filename saja sebagai cache identity.

Jika pipeline algorithm direvisi (contoh sesudah Task 7.1A), identifier/version algorithm harus di-bump agar cache namespace lama tidak dipakai ulang secara diam-diam.

---

## Playback

Browser audio element atau equivalent boleh digunakan.

Requirements:

- play
- pause
- seek
- duration
- currentTime

---

## Active Chord

Gunakan playback current time.

Cari segment:

```text
segment.start <= currentTime < segment.end
```

Chord tersebut menjadi:

```text
activeChord
```

---

## Error UI

Jangan tampilkan stack trace.

Tampilkan:

```text
Could not analyze this audio file.
```

Debug information tetap di log.

---

## Multi-Genre Evaluation

Desktop app boleh menyediakan workflow evaluasi corpus ground-truth lokal untuk
quality tuning lintas genre.

Aturan:

- user memilih manifest `.json` lewat native dialog;
- main process menjalankan Python engine command `evaluate-genre-corpus`;
- renderer hanya menampilkan report JSON/summary, bukan menjalankan DSP;
- report harus memperlihatkan confusion pairs item-level jika engine
  mengembalikannya, supaya tuning bisa membedakan salah root, salah quality,
  dan salah timing;
- workflow ini tidak boleh mengubah lagu aktif, timeline aktif, cache analysis,
  lyric, transition chord, atau saved library record;
- jika API mode aktif, workflow ini tetap lokal kecuali API contract khusus
  evaluasi corpus dibuat di masa depan.

---

## Security

Electron preload menggunakan:

```text
contextBridge
```

Jangan expose raw Node API ke renderer.

Jangan enable:

```text
nodeIntegration: true
```

kecuali ada alasan kuat dan terdokumentasi.
