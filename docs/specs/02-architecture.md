# 02 — Architecture

## High-Level Architecture

```text
┌──────────────────────────────┐
│ Electron Desktop Application │
│ React + TypeScript           │
└──────────────┬───────────────┘
               │
               │ IPC / local process
               ▼
┌──────────────────────────────┐
│ Python Audio Engine          │
│                              │
│ Decode                       │
│ Preprocess                   │
│ Chroma / CQT                 │
│ Chord Detection              │
│ Temporal Smoothing           │
└──────────────┬───────────────┘
               │
               ▼
          Analysis JSON
```

---

## Desktop Responsibilities

Electron/React hanya bertanggung jawab untuk:

- file selection
- audio playback
- app state
- timeline rendering
- invoking analysis engine
- caching analysis response
- error presentation

Desktop layer tidak melakukan chord inference.

---

## Engine Responsibilities

Python engine bertanggung jawab untuk:

- load audio
- decode MP3/WAV
- preprocessing
- feature extraction
- chord inference
- smoothing
- segment merging
- returning normalized JSON

---

## Why Separate Engine

Keuntungan:

- audio research lebih cepat dilakukan di Python
- dependency DSP tersedia luas
- algorithm dapat diganti tanpa mengubah UI
- ML migration lebih mudah
- memungkinkan export ONNX kemudian

---

## Analysis Flow

```text
Input audio
   ↓
Decode
   ↓
Resample
   ↓
Mono conversion
   ↓
Harmonic-focused preprocessing
   ↓
Beat-synchronous harmonic representation
   ↓
Multi-resolution contextual harmonic representation
   ↓
Harmonic novelty / change-point detection
   ↓
Beat-aware harmonic boundary consolidation
   ↓
Stable harmonic-region construction
   ↓
Region chord evidence scoring (root-aware + quality + template + soft key prior)
   ↓
Global sequence decoding (deterministic DP / Viterbi-style)
   ↓
Chord segments JSON

### Architectural Rule for Task 7.1A Subtask

Detector architecture must prioritize:

```text
determine WHEN harmony changes
before
determine WHAT chord is active
```

The pipeline must not primarily rely on classifying many short regions and repairing transitions afterward.

Boundary decisions and global decoding must be deterministic and must not encode one expected song progression.

Task 7.1A refinement rule:

- boundary consolidation operates on novelty/change-point candidates before final region classification;
- do not repair unstable labels afterward as primary strategy;
- protect genuine rapid harmonic changes when independent evidence is strong.

Task 7.1A additional refinement rule (multi-resolution context):

- harmonic boundary acceptance must use contextual support over multiple neighboring beats, not only adjacent-beat local novelty;
- retain at least short-context and medium-context harmonic evidence;
- short context preserves genuine rapid changes, medium context verifies persistence beyond transient melody/bass movement;
- boundary confidence must combine local novelty and multi-resolution context support deterministically.
```

---

## Dependency Direction

Allowed:

```text
UI
 ↓
Electron IPC
 ↓
Engine Adapter
 ↓
Python Engine
```

Not allowed:

```text
Python engine → Electron
Python engine → UI
DSP module → React
```

---

## Architecture Rule

Core chord detection logic harus dapat dijalankan tanpa Electron.

Contoh:

```bash
python -m chord_engine.cli input.mp3
```

harus menghasilkan JSON.

Ini penting agar algorithm dapat diuji secara independen.
