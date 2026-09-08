# 04 — Audio Engine

## Goal

Membuat audio preprocessing pipeline yang deterministic.

---

## Recommended Dependencies

Python 3.11+

Recommended:

```text
numpy
scipy
librosa
soundfile
pydantic
```

Optional experimental backend:

```text
essentia
```

Essentia may be used only as an explicit alternative analysis backend or
benchmark comparator. It must not become a hidden dependency of the default
engine path, and absence of the optional package must produce a controlled
error when the Essentia backend is explicitly requested.

MP3 decoding dapat menggunakan backend yang tersedia melalui librosa/audioread atau ffmpeg.

---

## Internal Audio Representation

Setelah decode:

```python
AudioBuffer(
    samples=np.ndarray,
    sample_rate=22050,
    duration=...
)
```

Samples harus:

```text
mono
float32
range sekitar -1.0 .. 1.0
```

---

## Target Sample Rate

Gunakan:

```text
22050 Hz
```

untuk MVP.

---

## Processing Steps

### Step 1

Load source audio.

### Step 2

Convert stereo ke mono.

Suggested:

```text
mean(left, right)
```

### Step 3

Resample ke:

```text
22050 Hz
```

### Step 4

Normalize hanya jika diperlukan.

Hindari aggressive normalization yang mengubah karakter audio.

### Step 5 (Task 7.1A)

Tambahkan harmonic-focused preprocessing sebelum chroma extraction untuk full-mix song.

Direkomendasikan menggunakan stack yang sudah disetujui:

```text
librosa.effects.hpss
atau
librosa.decompose.hpss
```

Aturan implementasi:

- pipeline tetap deterministic;
- hasil analisis chord utama menggunakan komponen harmonic;
- komponen percussive tidak dipakai untuk keputusan chord utama;
- fallback ke sinyal mono asli diperbolehkan jika HPSS gagal, dengan error handling terkontrol.

Tidak boleh menambah service eksternal, cloud processing, atau source stem extraction service.

### Step 6 (Task 7.1A)

Tambahkan beat-synchronous harmonic aggregation untuk membentuk region analisis musikal yang lebih stabil.

Aturan implementasi:

- beat/boundary dihitung dari audio lokal dengan librosa;
- agregasi chroma dilakukan per region beat-synchronous;
- keputusan chord tidak hanya berdasarkan frame pendek individual;
- detail agregasi harus terdokumentasi di implementation notes dan terukur di benchmark.

### Step 7 (Task 7.1A Subtask: Musical-Time Harmonic Persistence)

Tambahkan strategi deterministik musical-time persistence agar transisi chord merepresentasikan perubahan harmoni yang benar-benar bertahan, bukan perubahan chroma pendek akibat melodi, vokal, transien, passing note, atau pergerakan bass sesaat.

Aturan:

- gunakan agregasi musical-time berbasis beat lokal (tanpa hardcode BPM);
- gunakan evidence multi-beat untuk keputusan transisi;
- gunakan hysteresis: evidence untuk SWITCH chord harus lebih kuat dibanding evidence untuk KEEP chord saat ini;
- kandidat chord tidak boleh mengganti chord aktif hanya karena menang satu region pendek;
- tetap izinkan genuine short chord jika evidence kuat;
- jangan menyelesaikan masalah ini hanya dengan memperbesar global smoothing window atau minimum segment duration secara arbitrer.

Faktor evaluasi minimal:

- consecutive musical-time support;
- candidate confidence;
- score advantage dibanding chord saat ini;
- beat duration / musical timing;
- harmonic plausibility (soft);
- persistence pada observasi tetangga.

### Step 8 (Task 7.1A Subtask: Root-Aware Harmonic Scoring)

Tambahkan scoring harmonik root-aware pada observasi musical-time untuk meningkatkan akurasi identitas chord (root + kualitas mayor/minor) di full-mix music.

Aturan:

- evaluasi evidence root dipisahkan dari evidence kualitas mayor/minor;
- estimasi root harus memakai evidence harmonik region musical-time (bukan hanya satu cosine template penuh);
- gunakan evidence deterministik dari stack yang disetujui, termasuk jika relevan:
  - harmonic chroma;
  - pitch-class energy;
  - root/fifth support;
  - bass/low-frequency chroma evidence;
  - beat-synchronous aggregation;
- bass evidence bersifat supporting evidence saja, bukan hard rule bahwa nada terendah selalu root;
- kualitas mayor/minor dievaluasi sesudah kandidat root ditentukan, dengan fokus pada major third, minor third, root/fifth support;
- jika evidence third lemah, kualitas boleh dinilai ambigu (jangan memaksa confidence tinggi).

Integrasi wajib:

- tetap kompatibel dengan key soft prior;
- tetap kompatibel dengan persistence + transition acceptance refinement;
- template scorer existing tidak dibuang buta, tetapi dikombinasikan deterministik dengan evidence root-aware.

Larangan:

- jangan menambah smoothing global untuk mengejar segment count;
- jangan menambah output chord di luar 12 major + 12 minor + N;
- jangan menambah deep learning atau service eksternal.

### Step 9 (Task 7.1A Subtask: Harmonic Change-Point Segmentation and Global Chord Decoding)

Ubah arsitektur keputusan chord menjadi boundary-first, bukan short-region-first.

Urutan wajib:

```text
harmonic features
-> beat-synchronous representation
-> harmonic change-point detection
-> stable harmonic regions
-> region-level chord evidence
-> global sequence decoding
```

Aturan boundary detection:

- gunakan metode deterministic signal processing dari NumPy/librosa stack;
- novelty dihitung dari representasi harmonik beat-synchronous (misalnya adjacent chroma distance atau self-similarity based novelty);
- boundary harus merepresentasikan perubahan harmoni yang sustained, bukan noise melodi sesaat;
- tidak hardcode BPM;
- tidak mengasumsikan jumlah chord tetap per bar;
- tidak mengasumsikan perubahan hanya di bar boundary;
- keputusan boundary harus deterministic.

Aturan region decoding:

- setelah boundary dipilih, evidence chord diagregasi di seluruh region stabil;
- gunakan root-aware scoring + major/minor quality evidence;
- template evidence dipakai sebagai supporting evidence;
- key tetap soft prior saja;
- hasil akhir region harus dipilih melalui deterministic global sequence decoding
  (Viterbi/HMM-style dynamic programming atau optimizer deterministik ekuivalen).

Aturan decoding global:

- transition cost adalah soft penalty, bukan hard prohibition;
- rapid genuine chord changes tetap boleh jika evidence kuat;
- non-diatonic dan secondary-dominant outcomes tetap mungkin;
- tidak boleh encode progression lagu tertentu.

### Step 10 (Task 7.1A Subtask: Beat-Aware Harmonic Boundary Consolidation)

Tambahkan tahap konsolidasi boundary pada kandidat novelty/change-point sebelum klasifikasi chord region final.

Urutan wajib:

```text
harmonic beat-synchronous features
-> novelty/change-point candidates
-> beat-aware boundary consolidation
-> stable harmonic regions
-> region chord classification
-> global chord decoding
```

Aturan konsolidasi:

- deteksi cluster boundary yang terlalu rapat dalam musical-time lokal;
- jika beat tracking reliable, proximity dievaluasi utama dalam beat interval, bukan fixed milliseconds saja;
- jangan memaksa satu chord per N beat secara buta;
- evaluasi cluster menggunakan strength/prominence novelty, beat distance, similarity region sebelum-tengah-sesudah, durasi region tengah, dan evidence region tengah;
- region tengah pendek dan harmonically lemah boleh di-collapse;
- region tengah pendek tapi evidence independen kuat harus dipertahankan;
- jangan merge hanya karena region pendek;
- jangan optimasi terhadap target chord-per-minute tertentu.

Larangan:

- jangan menaikkan global novelty threshold sebagai solusi utama;
- jangan menambah smoothing global;
- jangan menambah patch heuristik perbaikan label setelah decoding final sebagai solusi utama.

### Step 11 (Task 7.1A Subtask: Multi-Resolution Harmonic Context Representation)

Tambahkan tahap representasi konteks harmonik multi-resolution sebelum novelty/change-point final diputuskan.

Urutan wajib:

```text
harmonic beat-synchronous features
-> multi-resolution contextual harmonic representation
-> novelty/change-point scoring
-> boundary consolidation
```

Aturan inti:

- boundary candidate harus dievaluasi terhadap konteks beberapa beat sebelum dan sesudah boundary;
- jangan mengandalkan perbedaan dua beat bertetangga saja sebagai keputusan final;
- minimal ada dua skala konteks:
  - short context: menjaga perubahan harmoni cepat yang genuine;
  - medium context: memvalidasi bahwa perubahan bertahan dan bukan transient melody/bass/voicing;
- gunakan agregasi robust deterministik (misalnya median/robust mean beat-synchronous chroma) untuk mengurangi pengaruh single-note excursion;
- parameter jendela konteks harus tersentralisasi dan beat-relative (boleh memakai adaptasi berbasis local beat duration), tidak hardcode satu pola meter/genre;
- boundary confidence akhir harus menggabungkan local novelty + support short context + support medium context secara deterministik.

Diagnostics minimum per candidate boundary:

- local novelty;
- short-context before/after distance;
- medium-context before/after distance;
- context agreement score;
- persistence score of harmonic change;
- short/medium agreement flag;
- final boundary confidence;
- acceptance/rejection reason.

Larangan:

- jangan sekadar blur semua fitur dengan satu smoothing window besar;
- jangan menaikkan global threshold sebagai workaround utama;
- jangan menambah post-boundary merge/correction sebagai solusi utama subtask ini.

---

## Validation

Reject jika:

- file tidak ditemukan
- format tidak bisa dibaca
- duration = 0
- samples kosong

---

## Error Contract

Engine harus menghasilkan error yang dapat dibaca manusia.

Contoh:

```json
{
  "error": {
    "code": "AUDIO_DECODE_FAILED",
    "message": "Unable to decode audio file"
  }
}
```

---

## Acceptance Criteria

Audio layer dianggap selesai jika:

- WAV mono berhasil dibaca
- WAV stereo berhasil diubah mono
- MP3 berhasil dibaca
- sample rate output selalu 22050 Hz
- duration terhitung benar dengan toleransi kecil
- invalid audio menghasilkan controlled error
- tidak ada uncaught exception di CLI

Tambahan untuk Task 7.1A:

- harmonic preprocessing tersedia dan aktif pada jalur analisis real-song;
- beat-synchronous aggregation tersedia sebelum scoring final chord;
- kontrak API JSON tidak berubah bentuknya;
- synthetic regression yang sudah ada tetap lulus.
