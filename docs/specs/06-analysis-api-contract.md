# 06 — Analysis API Contract

## Goal

Contract antara Python engine dan Electron harus stabil.

---

## Analyze Command

CLI:

```bash
python -m chord_engine.cli analyze "/path/song.mp3"
```

Output hanya JSON ke stdout.

Debug log harus ke stderr.

---

## Success Response

```json
{
  "version": "1",
  "source": {
    "path": "/path/song.mp3",
    "duration": 185.421,
    "sampleRate": 22050
  },
  "analysis": {
    "algorithm": "chroma-template-v1",
    "chords": [
      {
        "start": 0.0,
        "end": 2.18,
        "chord": "C",
        "confidence": 0.87
      }
    ]
  }
}
```

---

## Failure Response

```json
{
  "version": "1",
  "error": {
    "code": "AUDIO_DECODE_FAILED",
    "message": "Unable to decode audio file"
  }
}
```

Exit code harus non-zero untuk failure.

---

## Chord Segment

Fields:

```text
start       number seconds
end         number seconds
chord       string
confidence  number 0..1
```

Rules:

```text
start >= 0
end > start
confidence >= 0
confidence <= 1
```

---

## Ordering

Segments wajib:

```text
sorted ascending by start
```

Tidak boleh overlap.

---

## Versioning

Contract selalu memiliki:

```json
{
  "version": "1"
}
```

Jangan mengubah response shape tanpa bump version.

---

## TypeScript Contract

Representasi:

```ts
export interface ChordSegment {
  start: number;
  end: number;
  chord: string;
  confidence: number;
}

export interface ChordAnalysisResult {
  version: string;
  source: {
    path: string;
    duration: number;
    sampleRate: number;
  };
  analysis: {
    algorithm: string;
    chords: ChordSegment[];
  };
}
```
