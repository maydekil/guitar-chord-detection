# 01 — Product Scope

## Product Name

Working title:

`Guitar Chord Detector`

Nama dapat diubah kemudian.

---

## Problem

User ingin mengetahui progression chord dari sebuah lagu tanpa harus mencari chord manual.

Input utama:

- file audio lokal
- WAV
- MP3

Output utama:

- chord
- start time
- end time
- confidence

Contoh:

```json
[
  {
    "start": 0.0,
    "end": 2.13,
    "chord": "C",
    "confidence": 0.89
  },
  {
    "start": 2.13,
    "end": 4.29,
    "chord": "G",
    "confidence": 0.84
  }
]
```

---

## MVP Scope

MVP wajib memiliki:

### Audio

- Load WAV
- Load MP3
- Playback
- Pause
- Seek
- Current playback position

### Analysis

- Convert audio ke mono jika perlu
- Standardize sample rate
- Generate harmonic feature
- Detect major chord
- Detect minor chord
- Detect no-chord
- Temporal smoothing
- Merge consecutive identical chord segments

### UI

- Open audio button
- Song filename
- Play/pause
- Seek bar
- Timeline chord
- Highlight active chord
- Analysis progress state
- Error state

### Storage

Analysis disimpan lokal berdasarkan fingerprint file.

---

## Supported Chords MVP

Major:

```text
C
C#
D
D#
E
F
F#
G
G#
A
A#
B
```

Minor:

```text
Cm
C#m
Dm
D#m
Em
Fm
F#m
Gm
G#m
Am
A#m
Bm
```

Special:

```text
N
```

Total:

```text
25 classes
```

---

## Explicitly Out of Scope MVP

Jangan implementasikan pada MVP:

- realtime microphone detection
- YouTube download
- streaming URL
- Spotify integration
- cloud processing
- account/login
- source separation
- stem extraction
- guitar tablature
- automatic capo recommendation
- slash chord
- seventh chord
- maj7
- m7
- diminished
- augmented
- sus2
- sus4
- key transposition UI
- MIDI output

---

## Future Scope

Setelah MVP stabil:

### Phase 2

- Beat detection
- BPM detection
- Bar alignment
- Key detection
- Guitar fingering diagram

### Phase 3

Chord extensions:

- 7
- maj7
- min7
- sus2
- sus4
- dim
- aug

### Phase 4

- Realtime microphone mode
- system audio capture
- chord streaming inference

### Phase 5

Optional ML model:

- CNN
- CRNN
- Transformer
- ONNX runtime inference
