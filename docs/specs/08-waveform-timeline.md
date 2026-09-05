# 08 — Chord Timeline

## MVP Requirement

Waveform visual sebenarnya optional untuk first functional milestone.

Chord timeline wajib.

---

## Timeline Representation

Example:

```text
0s       4s       8s       12s

| C      | G      | Am     | F      |
```

Segment width dihitung berdasarkan duration.

Formula:

```text
leftPercent = start / songDuration * 100

widthPercent =
(end - start) / songDuration * 100
```

---

## Chord Block

Setiap chord segment memiliki:

- chord name
- start
- duration
- confidence

UI tidak wajib menampilkan confidence selalu.

Confidence dapat muncul di tooltip.

---

## Active Segment

Chord aktif harus memiliki visual emphasis.

Do not depend hanya pada warna.

Gunakan juga:

- border
- scale
- font weight
- marker

agar tetap accessible.

---

## Seeking

User click timeline:

```text
time =
clickPosition / timelineWidth * duration
```

Playback harus seek ke waktu tersebut.

---

## Timeline Scroll

Untuk MVP:

seluruh lagu boleh fit-to-width.

Future:

zoom + horizontal scroll.

---

## Waveform

Waveform dapat ditambahkan setelah chord timeline stabil.

Jika dibuat:

- generate peaks sekali
- jangan render raw samples
- gunakan decimated amplitude peaks
