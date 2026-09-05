# 10 — Acceptance Criteria

Dokumen ini adalah source of truth Definition of Done.

---

# AC-01 — Project Bootstrap

Given fresh checkout.

When developer mengikuti setup.

Then:

```text
pnpm install
```

berhasil.

Dan Python environment dapat di-install.

---

# AC-02 — Engine CLI

Command:

```bash
python -m chord_engine.cli analyze fixture.wav
```

harus menghasilkan valid JSON.

Tidak boleh ada log non-JSON di stdout.

---

# AC-03 — WAV Support

Given valid WAV.

When analyzed.

Then engine berhasil menghasilkan chord segments.

---

# AC-04 — MP3 Support

Given valid MP3.

When analyzed.

Then engine berhasil decode dan menghasilkan response.

---

# AC-05 — Major Chord Recognition

Synthetic C major fixture.

Expected dominant detected chord:

```text
C
```

Synthetic G major:

```text
G
```

---

# AC-06 — Minor Chord Recognition

Synthetic A minor:

```text
Am
```

Synthetic E minor:

```text
Em
```

---

# AC-07 — Silence

Given silence fixture.

Expected majority result:

```text
N
```

---

# AC-08 — Segment Integrity

Setiap segment:

```text
start >= 0
end > start
confidence 0..1
```

Segments:

```text
sorted
non-overlapping
```

---

# AC-09 — Smoothing

Given chord prediction dengan isolated one-frame jitter.

Expected:

jitter tidak menghasilkan chord segment singkat yang tidak stabil.

---

# AC-10 — Desktop Load Audio

User dapat memilih:

```text
.wav
.mp3
```

Selected filename tampil pada UI.

---

# AC-11 — Analyze Audio

Setelah file dipilih:

UI menampilkan status:

```text
Analyzing
```

dan kemudian chord timeline.

---

# AC-12 — Playback

User dapat:

```text
play
pause
seek
```

Playback position harus bergerak sesuai audio.

---

# AC-13 — Active Chord

Jika playback berada pada:

```text
segment.start <= currentTime < segment.end
```

chord tersebut ditampilkan sebagai chord aktif.

---

# AC-14 — Timeline Seek

Given user click posisi tertentu di timeline.

Playback seek ke posisi ekuivalen dengan toleransi:

```text
±250 ms
```

---

# AC-15 — Cache

Setelah sebuah file dianalisis.

Jika file yang sama dibuka kembali tanpa perubahan.

Engine tidak perlu dijalankan kembali.

---

# AC-16 — File Change Detection

Jika content file berubah.

Cache lama tidak boleh digunakan.

Fingerprint harus berubah.

---

# AC-17 — Error Handling

Invalid audio tidak menyebabkan desktop crash.

UI menampilkan human readable error.

---

# AC-18 — Offline

Setelah dependency terinstall.

App tidak membutuhkan internet untuk:

```text
open audio
analyze
play
show chord
```

---

# AC-19 — Security

Renderer tidak mendapat unrestricted Node access.

Preload hanya expose method yang dibutuhkan.

---

# AC-20 — Test Suite

Semua automated test wajib pass sebelum MVP dinyatakan selesai.

---

# MVP Exit Criteria

MVP selesai hanya jika:

```text
AC-01 sampai AC-20 = PASS
```

Tidak boleh menyatakan MVP selesai jika acceptance criteria critical masih gagal.


---

# AC-21 — Development Toolchain

Development environment memiliki:

```text
Node.js
pnpm
Python 3.11
FFmpeg
```

dan `.venv` dapat dibuat.

---

# AC-22 — No Absolute Development Path

Source code tidak boleh bergantung pada path spesifik seperti:

```text
/Users/<developer>/
```

atau absolute path FFmpeg/Python milik developer.

---

# AC-23 — Production Python Independence

Packaged desktop application tidak boleh mewajibkan end-user menginstall Python.

---

# AC-24 — Production FFmpeg Independence

Packaged desktop application tidak boleh mewajibkan end-user menginstall FFmpeg.

---

# AC-25 — macOS Apple Silicon Packaging

Build target:

```text
darwin-arm64
```

berhasil menghasilkan artifact desktop yang dapat dijalankan.

---

# Revised MVP / Distribution Exit Criteria

Functional MVP:

```text
AC-01 sampai AC-20 = PASS
```

Distribution-ready macOS build:

```text
AC-01 sampai AC-25 = PASS
```

---

# AC-26 — Repository Structure Compliance

Repository contains no forbidden top-level directories and implementation files follow `03-project-structure.md` and `03b-file-ownership.md`.

# AC-27 — Dependency Boundary Compliance

Renderer does not directly access filesystem, child processes, Python, or production engine paths. Python engine has no Electron/React dependency.

# AC-28 — Agent Scope Compliance

Implementation tasks do not create architecture outside the allowed paths without a documented human-approved spec change.

## Governance Gate

Every phase must satisfy AC-26 through AC-28 where applicable. Distribution-ready completion requires AC-01 through AC-28 PASS.

---

# AC-29 — Synthetic Regression Preservation After 7.1A

Setelah implementasi Task 7.1A, seluruh regression synthetic yang sebelumnya wajib tetap PASS, termasuk progression:

```text
C → G → Am → F
```

tanpa regression behavior mayor/minor/silence.

# AC-30 — Harmonic-Focused Preprocessing Enabled

Pipeline real-song menggunakan harmonic-focused preprocessing (HPSS atau ekuivalen deterministic lokal) sebelum keputusan chord utama.

# AC-31 — Beat-Synchronous Harmonic Aggregation

Pipeline improved menggunakan region beat-synchronous untuk agregasi harmonic evidence sebelum scoring final chord.

# AC-32 — Key Prior Is Soft (Non-Diatonic Still Possible)

Estimasi key global dipakai sebagai soft prior saja.

Non-diatonic/borrowed chord tidak boleh menjadi impossible outcome jika evidence audio kuat.

# AC-33 — Real-Song Flicker Reduction With Baseline Comparison

Pada benchmark file lokal yang sama, hasil improved harus menunjukkan pengurangan material rapid/implausible chord switching dibanding baseline.

Minimal harus ditunjukkan melalui BEFORE/AFTER metrics, termasuk:

- jumlah segmen;
- median/mean durasi segmen;
- excessive short-segment rate.

# AC-34 — No API Contract Regression

Output contract tetap kompatibel dengan `06-analysis-api-contract.md` (stdout JSON shape tidak regression).

# AC-35 — Local Benchmark Audio Governance

Workflow benchmark real-song menggunakan file lokal user-owned yang tidak di-commit ke repository.

Repository tidak boleh berisi commercial song fixtures.

# AC-41 — Deterministic Benchmark Diagnostics Schema

Task 7.1A benchmark output harus menyediakan diagnostic schema deterministik untuk baseline dan improved, minimal berisi:

- algorithm/version;
- global key + key confidence/score (jika tersedia);
- segment count + song duration;
- mean/median/min/max segment duration;
- short-segment buckets (<250ms, <500ms, <1s) dalam count + percentage;
- chord occurrence count;
- chord total duration;
- chord duration percentage of song;
- diatonic vs non-diatonic count + duration relatif terhadap detected key;
- suspicious short non-diatonic list (chord/start/end/duration/confidence).

# AC-42 — Diagnostic-Only Diatonicity

Informasi diatonic/non-diatonic pada Task 7.1A benchmark bersifat diagnosis saja.

Pipeline tidak boleh otomatis menghapus atau mengganti non-diatonic chord hanya karena status keanggotaan key.

# AC-43 — Context-Aware Short-Segment Correction Quality

Task 7.1A subtask context-aware correction harus memperbaiki short isolated weak anomalies berdasarkan konteks temporal/harmonik, bukan aturan single-factor.

Minimal faktor evaluasi mencakup: prev/candidate/next chord, segment duration, confidence, dan harmonic plausibility.

# AC-44 — No Blind Short-Segment Removal

Short duration saja tidak boleh menjadi alasan otomatis untuk menghapus atau mengganti chord.

# AC-45 — No Blind Non-Diatonic Removal

Status non-diatonic saja tidak boleh menjadi alasan otomatis untuk menghapus atau mengganti chord.

Borrowed chord/secondary function harus tetap mungkin bertahan bila evidence mendukung.

# AC-46 — Genuine Change Preservation

Sustained chord transitions yang genuine harus tetap bertahan setelah context-aware correction.

Short genuine chord dengan confidence/evidence kuat tidak boleh dihilangkan secara otomatis.

# AC-47 — Benchmark Improvement Evidence Without Segment-Count Bias

Penerimaan subtask tidak boleh hanya mengejar segment count lebih kecil.

BEFORE/AFTER benchmark harus menunjukkan quality-oriented evidence melalui kombinasi metrik durasi, short-segment profile, dan stability diagnostics.

# AC-48 — Musical-Time Harmonic Persistence

Task 7.1A subtask persistence harus menggunakan agregasi musical-time deterministik berbasis beat-aware evidence (tanpa hardcode BPM).

Transisi chord tidak boleh dipicu hanya oleh satu observasi pendek yang menang sesaat.

# AC-49 — Asymmetric Transition Evidence

Evidence untuk SWITCH chord harus lebih kuat daripada evidence untuk KEEP chord saat ini.

Implementasi harus membedakan kriteria KEEP vs SWITCH secara eksplisit.

# AC-50 — Genuine Transition Preservation

Peningkatan persistence tidak boleh menghancurkan genuine sustained chord transitions.

Short genuine chord dengan evidence kuat tetap harus mungkin bertahan.

# AC-51 — Transition-Density Benchmark Metrics

Benchmark Task 7.1A harus melaporkan transition-density metrics sebelum/sesudah:

- segment count;
- chord transitions per minute;
- mean/median segment duration;
- <500ms rate;
- <1s rate;
- optional tempo estimate;
- transitions per beat jika beat reliability memadai.

# AC-52 — Ordinary Override Must Respect Evidence Ordering

Pada Task 7.1A Transition Acceptance Refinement, ordinary override tidak boleh menerima candidate switch ketika `switchScore <= keepScore`.

# AC-53 — Exceptional Override Safety + Diagnosis

Jika acceptance tetap terjadi saat `switchScore <= keepScore`, maka wajib:

1. memenuhi syarat explicit stronger multi-beat/context support; dan
2. reason acceptance tercatat eksplisit pada diagnostics.

# AC-54 — Root-Preserving Quality Hysteresis

Root-preserving quality switch (major <-> minor pada root sama) harus memakai kriteria acceptance lebih ketat daripada continuation biasa untuk mencegah flicker karena kontaminasi melody/vocal/bass.

# AC-55 — Genuine Quality-Change Preservation

Refinement tidak boleh melarang genuine major/minor quality change yang didukung evidence kuat dan sustained dalam observasi musical-time.

# AC-56 — Transition Acceptance Diagnostic Coverage

Benchmark diagnostics Task 7.1A wajib memuat:

- accepted transition reason counts;
- count root-preserving major/minor accepted switches;
- count accepted overrides dengan `switchScore <= keepScore`;
- before/after transition count dan transitions per minute.

# AC-57 — Root/Quality Evidence Separation

Pada Task 7.1A Root-Aware Harmonic Scoring, evidence root dan evidence quality (major/minor) harus dihitung terpisah sebelum keputusan chord identity final.

# AC-58 — Root Estimation Must Use Musical-Time Harmonic Evidence

Estimasi root harus memanfaatkan evidence harmonik regional musical-time (pitch-class energy, root/fifth support, dan supporting bass/low-frequency evidence), bukan hanya template cosine penuh secara tunggal.

# AC-59 — Bass Evidence Is Supporting, Not Absolute

Evidence bass tidak boleh dijadikan hard rule bahwa pitch terendah selalu root; kondisi inversion-like tidak otomatis mengubah root.

# AC-60 — Quality Ambiguity Handling

Jika evidence major-third vs minor-third lemah, kualitas boleh ditandai ambigu dan tidak boleh dipaksa menjadi keputusan berkepastian tinggi.

# AC-61 — Root-Aware + Template Deterministic Combination

Template scorer existing harus tetap digunakan dan dikombinasikan deterministik dengan root-aware evidence; tidak boleh dibuang buta.

# AC-62 — Root-Aware Benchmark Diagnostic Coverage

Benchmark diagnostics Task 7.1A wajib memuat:

- root candidate scores;
- selected root confidence;
- major-quality evidence;
- minor-quality evidence;
- quality margin;
- template score;
- root-aware combined score;
- root-change count;
- quality-change count;
- ambiguous-quality decision count.

# AC-63 — Identity-Oriented Success Gate

Keberhasilan subtask root-aware tidak boleh didefinisikan hanya oleh pengurangan segment count.

Hasil harus menunjukkan:

1. ground-truth synthetic root/quality tetap benar;
2. unsupported chord-identity changes berkurang;
3. genuine root dan genuine major/minor changes tetap terdeteksi;
4. transition-density improvements sebelumnya tetap terjaga;
5. API publik tetap kompatibel.

# AC-64 — Ground-Truth Annotation Format Support

Task 7.1A Ground-Truth Chord Evaluation harus mendukung format anotasi lokal sederhana dengan field minimal:

- start;
- end;
- chord.

Workflow juga harus mendukung evaluasi partial-song.

# AC-65 — Ground-Truth Evaluation CLI

Engine CLI harus menyediakan command evaluasi yang membandingkan output chord engine dengan file anotasi ground truth lokal secara deterministik.

# AC-66 — Ground-Truth Evaluation Metrics Coverage

Output evaluasi minimal memuat:

- time-weighted chord accuracy;
- exact chord match percentage;
- root accuracy;
- major/minor quality accuracy;
- false transition count;
- missed transition count;
- boundary timing error;
- confusion pairs.

# AC-67 — No Detector Retuning in Evaluation Subtask

Subtask Ground-Truth Chord Evaluation tidak boleh mengubah detector thresholds/heuristics.

# AC-68 — Local Annotation Governance

Anotasi user-owned song tetap lokal dan tidak wajib di-commit kecuali diminta eksplisit.

# AC-69 — Regression and API Preservation

Public engine API existing tetap kompatibel dan seluruh regression tests yang sudah ada tetap PASS setelah workflow evaluasi ditambahkan.

# AC-70 — Boundary-First Harmonic Segmentation

Pipeline improved Task 7.1A harus menentukan boundary perubahan harmoni terlebih dahulu (change-point/novelty deterministic), lalu melakukan keputusan chord per region stabil.

Pipeline tidak boleh utama bergantung pada classify short region lalu patch transisi setelahnya.

# AC-71 — Deterministic Harmonic Novelty Boundary Selection

Boundary detection harus deterministic dan berbasis harmonic beat-synchronous evidence.

Novelty dari passing tone atau kontaminasi melody singkat harus bisa ditolak ketika tidak menunjukkan perubahan harmoni yang sustained.

# AC-72 — Region-Level Chord Decision

Keputusan chord harus dibuat per harmonic region stabil dengan agregasi evidence region penuh, termasuk root-aware evidence separation + quality evidence + template support + key soft prior.

# AC-73 — Global Sequence Decoding

Pipeline improved wajib memiliki deterministic global decoding stage (Viterbi/HMM-style DP atau ekuivalen) yang mempertimbangkan:

- region evidence;
- self-transition/stability preference;
- soft transition-change penalty;
- harmonic relationship antar kandidat;
- confidence/evidence margin;
- key soft prior.

# AC-74 — Soft Transition Cost (No Hard Ban)

Transition cost harus soft penalty, bukan larangan.

Rapid genuine chord changes tetap boleh saat evidence kuat, dan non-diatonic outcomes tetap mungkin.

# AC-75 — Superseded-Heuristic Discipline

Jika boundary-first + global decoding mensupersede heuristic transition-correction lama, stage redundant harus dibypass/remove pada jalur improved dan didokumentasikan.

Tidak boleh menumpuk banyak layer corrective heuristics tanpa justifikasi arsitektural.

# AC-76 — Harmonic Segmentation Diagnostics Coverage

Benchmark diagnostics improved wajib memuat:

- harmonic boundary count;
- harmonic regions per minute;
- mean/median harmonic region duration;
- rejected novelty peaks;
- accepted boundary timestamps;
- per-region selected chord;
- top-3 chord candidates + score per region;
- global decoder path score;
- local-best vs globally-decoded chord;
- number of decisions changed by global decoding.

# AC-77 — Quality-Oriented Success Gate

Keberhasilan subtask ini tidak boleh didefinisikan hanya sebagai segment-count reduction.

Minimal harus memenuhi:

1. synthetic ground-truth boundaries/chords tetap benar;
2. sustained-harmony contamination cases tetap stabil;
3. genuine chord changes tetap preserved;
4. real-song regionasi terasa lebih tidak terfragmentasi;
5. public API tidak regression.

# AC-78 — Beat-Aware Boundary Consolidation Stage

Pipeline improved harus memiliki tahap boundary consolidation pada kandidat harmonic change-point sebelum klasifikasi chord region final.

Tahap ini tidak boleh dipindahkan menjadi post-label repair utama.

# AC-79 — Musical-Time Cluster Handling

Boundary yang terlalu rapat harus dievaluasi dalam konteks musical time lokal.

Jika beat tracking reliable, proximity utama diekspresikan dalam beat interval.

# AC-80 — Weak Intermediate Region Collapse Rule

Dua boundary boleh di-collapse jika region intermediate pendek dan harmonically lemah terhadap konteks sekitar.

Collapse tidak boleh dipicu oleh durasi pendek saja.

# AC-81 — Strong Intermediate Region Preservation

Jika region intermediate pendek tetapi evidence harmonik independennya kuat, boundary pair tidak boleh di-collapse.

# AC-82 — Rapid Genuine Change Preservation

Konsolidasi cluster tidak boleh menghapus genuine rapid harmonic changes yang didukung evidence kuat.

# AC-83 — Consolidation Diagnostics Coverage

Benchmark diagnostics wajib memuat:

- candidateBoundaryCount;
- acceptedBoundaryCountBeforeConsolidation;
- acceptedBoundaryCountAfterConsolidation;
- consolidatedBoundaryCount;
- boundaryClusterCount;
- mean/median beatsBetweenAcceptedBoundaries;
- shortHarmonicRegionCount;
- consolidated cluster examples;
- harmonic region duration distribution (<1 beat, 1-2 beats, 2-4 beats, >=4 beats).

# AC-84 — No Blanket Threshold/Smoothing Workaround

Subtask ini tidak boleh diselesaikan terutama dengan menaikkan global novelty threshold atau menambah global smoothing sebagai workaround umum.

# AC-85 — Multi-Resolution Harmonic Context Stage

Pipeline improved harus menambahkan stage multi-resolution contextual harmonic representation sebelum keputusan harmonic novelty/change-point final.

Stage ini wajib menggunakan minimal short-context dan medium-context evidence.

# AC-86 — Contextual Boundary Evidence Requirement

Boundary tidak boleh diterima hanya karena local adjacent-beat novelty.

Boundary final harus memiliki dukungan konteks harmonik lintas beberapa beat sebelum/sesudah boundary.

# AC-87 — Local-Only Novelty Rejection

Kasus local novelty tinggi tetapi medium-context support rendah harus umumnya ditolak sebagai harmonic boundary.

# AC-88 — Genuine Rapid Change Preservation Under Context Agreement

Perubahan harmoni cepat yang genuine harus tetap bisa diterima ketika short-context dan medium-context sama-sama mendukung.

# AC-89 — Multi-Resolution Diagnostics Coverage

Benchmark diagnostics wajib memuat:

- localNoveltyCandidateCount;
- contextualBoundaryCandidateCount;
- multiResolutionAcceptedBoundaryCount;
- rejectedLocalOnlyBoundaryCount;
- shortMediumAgreementRate;
- meanShortContextDistance;
- meanMediumContextDistance;
- acceptedBoundaryExamples;
- rejectedLocalOnlyExamples.

# AC-90 — No Additional Post-Boundary Merge/Persistence Workaround

Subtask ini tidak boleh diselesaikan dengan menambah layer baru post-boundary merging, blanket persistence, atau post-label repair sebagai strategi utama.

# AC-36 — Re-analyze Cache Bypass

Jika user menekan `Re-analyze` untuk file terpilih yang valid:

- cache lookup harus di-bypass;
- engine harus dijalankan ulang;
- UI harus clear analysis lama dan masuk state `analyzing` segera;
- hasil success harus menjadi hasil tampilan baru dan menggantikan cache untuk key tersebut.

# AC-37 — Re-analyze Error Behavior

Jika `Re-analyze` gagal:

- stale result lama tidak boleh dipulihkan ke UI;
- UI menampilkan controlled `error` state/message.

# AC-38 — Re-analyze Stale Response Guard

Jika request lama selesai setelah request terbaru, response lama tidak boleh overwrite hasil request terbaru.

# AC-39 — Cache Identity Scope

Cache identity harus mencakup:

```text
file-content SHA-256
analysis contract version
analysis algorithm version
```

Filename saja tidak valid sebagai cache identity.

# AC-40 — Algorithm Revision Cache Namespace

Jika algorithm analisis direvisi, identifier/version algorithm harus di-bump sehingga cache lama tidak dipakai secara silent untuk algorithm baru.
