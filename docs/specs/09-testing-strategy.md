# 09 — Testing Strategy

Use pytest for Python engine and Vitest for TypeScript. Never commit full commercial copyrighted songs.

## Established coverage from Tasks 1.1–2.4
Audio: mono/stereo WAV, real MP3 decode path, mono conversion, 22050 resampling, duration, invalid/empty/nonfinite.
Features: 12×N finite CQT chroma, deterministic silence, C-major C/E/G dominance, A-minor A/C/E dominance, timing, input immutability.
Templates: all 24 generated templates and rotations.
Detector: all 24 exact templates, synthetic audio C/Am, silence/near-silence N, invalid frames, confidence bounds/determinism.
Smoothing: stable labels, isolated jitter, real transitions, sustained N, output-length preservation, anti-over-smoothing.
Segmentation: run merge, short jitter, sustained N, confidence aggregation, sorted/contiguous/non-overlap, determinism.

## Task 3.1 mandatory integration tests
Use actual file pipeline; no mocks or bypass:
1. synthetic C-major file → dominant/final C;
2. synthetic A-minor file → dominant/final Am;
3. valid silence WAV → meaningful N;
4. synthetic `C → G → Am → F`, each region about 1–2 seconds so it survives 534ms smoothing and 250ms minimum segment duration;
5. invalid/corrupt audio → controlled failure;
6. same deterministic fixture analyzed twice → same labels/boundaries/confidence within float tolerance.

For progression, preserve chord order. Small boundary shifts are acceptable; do not assert sample-exact transition times.

Result integrity: sorted, non-overlapping, confidence 0..1, final public end <= source duration within small tolerance.

Do not weaken a failing integration test merely to obtain PASS.

## Task 7.1A real-song accuracy test strategy

### Scope

Task 7.1A menambah evaluasi real-song terkontrol tanpa menghapus test synthetic yang sudah ada.

### Preserve existing regression

Semua test engine yang sudah ada wajib tetap jalan dan tetap PASS, termasuk:

- major/minor synthetic recognition;
- silence behavior (`N`);
- progression `C → G → Am → F`;
- determinism dan segment integrity.

### Local real-song benchmark fixture (not committed)

Gunakan satu file lagu milik user yang disimpan lokal dan tidak di-commit ke repository.

Aturan:

- file path benchmark diambil dari environment variable lokal (contoh: `GCD_REAL_SONG_PATH`);
- benchmark harus skip terkontrol jika env var tidak tersedia;
- test/benchmark report boleh menyimpan ringkasan numerik, tetapi tidak boleh menyimpan audio copyrighted.

### Baseline vs improved comparison requirement

Benchmark Task 7.1A harus membandingkan baseline saat ini dengan pipeline improved pada file lokal yang sama, bukan hanya menampilkan hasil final improved.

### Required comparison metrics

Laporkan metrik berikut untuk baseline dan improved:

1. jumlah chord segments;
2. median segment duration;
3. mean segment duration;
4. excessive short-segment rate (contoh: proporsi segment durasi < 250ms);
5. detected global key;
6. chord-family distribution (major/minor/N counts atau proporsi);
7. synthetic regression accuracy/pass status.

Tambahan metrik minimum wajib:

1. key confidence/score jika tersedia;
2. min dan max segment duration;
3. count + percentage untuk segment durasi <250ms, <500ms, <1s;
4. chord occurrence count per chord;
5. total duration per chord;
6. percentage of song duration per chord;
7. diatonic vs non-diatonic segment count terhadap detected key;
8. diatonic vs non-diatonic duration terhadap detected key;
9. daftar suspicious short non-diatonic segments: chord, start, end, duration, confidence.

### Deterministic diagnostic calculation tests

Tambahkan unit test deterministik berbasis synthetic analysis result (tanpa audio copyrighted) untuk memverifikasi perhitungan metrik benchmark.

Minimal validasi:

- duration statistics (mean/median/min/max);
- short-segment buckets dan persentase;
- chord occurrence + duration aggregation;
- diatonic/non-diatonic counting and duration;
- suspicious short non-diatonic list.

### Anti-overfitting guardrails

- jangan hardcode key/progression dari lagu benchmark ke production code;
- jangan tuning konstanta hanya untuk satu lagu;
- perubahan harus mempertahankan general behavior pada synthetic suite.

### Task 7.1A Subtask tests: Context-Aware Short-Segment Correction

Tambahkan test deterministik untuk memastikan koreksi short-segment memakai konteks, bukan aturan buta.

Minimal cakupan:

- short low-confidence isolated anomaly di antara harmoni stabil dapat dikoreksi jika context evidence lebih kuat;
- short genuine chord dengan evidence kuat tidak otomatis dihapus;
- sustained genuine transitions tetap survive;
- non-diatonic/secondary-dominant yang musically valid tetap bisa survive;
- silence/N behavior tetap benar;
- semua regression synthetic lama tetap PASS.

Evaluasi benchmark harus melaporkan BEFORE/AFTER dengan metrik objektif (bukan visual saja) menggunakan sistem benchmark yang sama.

### Task 7.1A Subtask tests: Musical-Time Harmonic Persistence

Tambahkan test deterministik untuk membuktikan persistence musical-time bekerja tanpa merusak ground-truth synthetic.

Minimal test wajib:

- stable chord dengan moving melody tidak memunculkan false transitions berlebihan;
- stable chord dengan moving bass tidak memunculkan false transitions berlebihan;
- passing note pendek tidak memicu switch chord;
- genuine C -> G transition tetap terdeteksi;
- progression C -> G -> Am -> F tetap benar;
- genuine short chord dengan evidence kuat tetap bisa survive;
- repeated analysis tetap deterministik.

Benchmark comparison wajib memuat transition-density metrics:

- segment count;
- transitions per minute;
- mean/median segment duration;
- <500ms rate;
- <1s rate;
- optional estimated tempo;
- transitions per beat jika beat reliability memadai.

### Task 7.1A Subtask tests: Transition Acceptance Refinement

Tambahkan test deterministik untuk memastikan acceptance policy mengurangi weak/ambiguous accepted switches tanpa merusak genuine transitions.

Minimal test wajib:

- stable E major dengan moving melody tidak flicker ke Em;
- stable A major tidak flicker ke Am;
- genuine E -> Em transition dengan sustained strong evidence tetap terdeteksi;
- genuine C#m -> C# quality change dapat survive jika evidence kuat;
- switchScore lebih lemah dari keepScore tidak boleh diterima oleh ordinary override;
- explicit multi-beat/context-supported override boleh switch hanya jika syarat override refinement terpenuhi dan reason terdiagnosis;
- progression C -> G -> Am -> F tetap benar;
- repeated analysis tetap deterministik.

Benchmark diagnostics wajib ditambah:

- accepted transition reason counts;
- count root-preserving major/minor accepted switches;
- count accepted overrides dengan switchScore <= keepScore;
- before/after transition count dan transitions per minute.

Kriteria keberhasilan subtask ini bukan hanya pengurangan segment count, tetapi:

- penurunan accepted weak/ambiguous switches;
- genuine sustained transitions tidak regress;
- synthetic ground-truth regression tetap PASS.

### Task 7.1A Subtask tests: Root-Aware Harmonic Scoring

Tambahkan test deterministik untuk memastikan pemisahan evidence root vs quality meningkatkan identity stability tanpa merusak genuine perubahan harmonik.

Minimal test wajib:

- semua 12 major roots terdeteksi benar pada fixture synthetic terkait;
- semua 12 minor roots terdeteksi benar pada fixture synthetic terkait;
- C major vs C minor quality discrimination benar;
- A major vs A minor quality discrimination benar;
- E major dengan melody contamination tetap E;
- A major dengan moving bass tetap rooted sebagai A jika harmonic evidence mendukung A;
- inversion-like bass condition tidak otomatis mengubah root;
- weak third evidence tidak memicu major/minor flicker tidak stabil;
- genuine major -> minor quality change tetap terdeteksi;
- regression C -> G -> Am -> F tetap benar;
- silence/N behavior tetap benar;
- repeated analysis tetap deterministik.

Benchmark diagnostics wajib ditambah:

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

Benchmark BEFORE/AFTER minimal melaporkan:

- segment count;
- transitions per minute;
- root-change count;
- quality-change count;
- root-preserving quality switches;
- weak override count;
- chord distribution;
- mean/median segment duration.

Kriteria lulus subtask ini bukan segment-count-only reduction, tetapi ketepatan identity dan stabilitas evidence.

### Task 7.1A Subtask tests: Ground-Truth Chord Evaluation

Tambahkan workflow evaluasi deterministik yang membandingkan output engine terhadap anotasi chord ground truth lokal.

Format anotasi lokal minimal wajib memuat:

- start;
- end;
- chord.

Workflow harus mendukung anotasi partial-song (misal hanya 00:30-01:00).

Tambahkan test synthetic fixture untuk memastikan evaluasi menghasilkan metrik yang konsisten dan tidak bergantung pada lagu komersial.

Metrik evaluasi minimum wajib:

- time-weighted chord accuracy;
- exact chord match percentage;
- root accuracy;
- major/minor quality accuracy;
- false transition count;
- missed transition count;
- boundary timing error;
- confusion pairs (misal E->Em, A->F#m, dst).

Aturan scope:

- subtask ini tidak boleh mengubah tuning detector thresholds/heuristics;
- tidak boleh hardcode path/song/progression tertentu;
- file anotasi user-owned bersifat lokal dan tidak wajib di-commit;
- public engine API dan regression tests existing harus tetap terjaga.

### Task 7.1A Subtask tests: Multi-Genre Ground-Truth Corpus Evaluation

Tambahkan workflow evaluasi corpus lokal supaya tuning chord detector dapat
dibandingkan lintas jenis lagu, tempo, dan style tanpa hardcode lagu tertentu.

Format manifest corpus lokal minimal wajib memuat:

- `items`, array non-empty;
- `id`, nama stabil sample lokal;
- `genre`, label bebas seperti `rock`, `ska`, `reggae`, `pop`, `jazz`, `folk`;
- `audioPath`, path audio lokal;
- `annotationPath`, path anotasi ground-truth lokal.

`audioPath` dan `annotationPath` boleh relatif terhadap lokasi manifest,
atau absolute/local-env path. Manifest boleh menambahkan `clipStart` dan
`clipEnd` per item untuk evaluasi partial-song.

Workflow harus:

- menjalankan evaluasi ground-truth yang sama untuk setiap item;
- mengagregasi metrik per genre dan seluruh corpus;
- melaporkan item gagal secara terkontrol tanpa menyembunyikan item lain;
- menghasilkan JSON deterministik dari CLI;
- tidak mengubah chord detector threshold/heuristic hanya karena manifest ada;
- tidak menyimpan audio user-owned atau anotasi privat ke repository.

Metrik agregat minimum wajib:

- item count, evaluated item count, failed item count;
- weighted time-weighted chord accuracy;
- weighted root accuracy;
- weighted quality accuracy;
- total false transition count;
- total missed transition count;
- mean boundary timing error jika tersedia.

Aturan scope:

- subtask ini adalah alat evaluasi dan guardrail tuning, bukan perubahan
  musical decision engine;
- tidak boleh hardcode genre-specific progression;
- tidak boleh menjadikan lyric sebagai sumber kebenaran chord/timing.

### Task 7.1A Subtask tests: Harmonic Change-Point Segmentation and Global Chord Decoding

Tambahkan test deterministik yang memverifikasi boundary-first harmonic segmentation dan global decoding.

Minimal test wajib:

1. satu sustained A chord + moving melody tetap satu harmonic region;
2. satu sustained A chord + moving bass tidak menghasilkan false boundary;
3. passing tones tidak memicu harmonic boundary;
4. genuine A -> E transition menghasilkan boundary;
5. progression C -> G -> Am -> F menghasilkan empat region/chord benar;
6. dua genuine rapid chord tetap bisa lolos jika evidence kuat;
7. major/minor quality change pada root sama bisa menjadi boundary jika sustained;
8. silence / N tetap valid;
9. repeated runs deterministik.

Benchmark diagnostics wajib ditambah:

- harmonic boundary count;
- harmonic regions per minute;
- mean/median harmonic region duration;
- rejected novelty peaks;
- accepted boundary timestamps;
- per-region selected chord;
- top-3 candidate + score per region;
- global decoder path score;
- local-best vs global-decoded chord;
- number of region decisions changed by global decoding.

Kriteria lulus subtask ini tidak boleh didefinisikan hanya sebagai pengurangan segment count.

### Task 7.1A Subtask tests: Beat-Aware Harmonic Boundary Consolidation

Tambahkan test deterministik untuk konsolidasi cluster boundary sebelum klasifikasi region final.

Minimal test wajib:

1. satu perubahan harmoni kuat yang memunculkan beberapa novelty peak berdekatan -> terkonsolidasi menjadi satu boundary;
2. transient/passing-note peak di dekat boundary nyata -> transient boundary dihapus;
3. short weak intermediate harmonic region di antara harmoni sekitar yang serupa -> collapse;
4. dua genuine rapid changes dengan evidence independen kuat -> keduanya tetap;
5. C -> G -> Am -> F tetap empat region benar;
6. moving melody tidak membuat cluster boundary berlebih;
7. moving bass tidak membuat cluster boundary berlebih;
8. same-root major/minor genuine change tetap mungkin;
9. silence/N tetap benar;
10. repeated analysis tetap deterministik.

Benchmark diagnostics wajib ditambah:

- candidateBoundaryCount;
- acceptedBoundaryCountBeforeConsolidation;
- acceptedBoundaryCountAfterConsolidation;
- consolidatedBoundaryCount;
- boundaryClusterCount;
- mean/median beatsBetweenAcceptedBoundaries;
- shortHarmonicRegionCount;
- consolidated cluster examples berisi timestamp, beat distance, novelty strengths, prominences, harmonic similarities, selected surviving boundary, rejection reason;
- harmonic region duration distribution: <1 beat, 1-2 beats, 2-4 beats, >=4 beats.

### Task 7.1A Subtask tests: Multi-Resolution Harmonic Context Representation

Tambahkan test deterministik untuk memverifikasi boundary evidence berbasis short+medium context sebelum keputusan harmonic boundary final.

Minimal test wajib:

1. sustained A major + changing melody notes -> tidak menghasilkan false harmonic boundary;
2. sustained A major + moving/walking bass -> tidak menghasilkan false harmonic boundary;
3. sustained chord + arpeggiated/voicing changes -> tidak menghasilkan false harmonic boundary;
4. passing-tone sequence menghasilkan local novelty tetapi gagal medium-context support -> boundary ditolak;
5. genuine A -> E change tetap menghasilkan boundary;
6. progression C -> G -> Am -> F tetap tepat;
7. dua genuine relatively rapid changes tetap preserved saat short+medium evidence kuat;
8. same-root quality change A -> Am tetap terdeteksi saat sustained;
9. silence/N tetap benar;
10. repeated analysis tetap deterministik.

Benchmark diagnostics wajib ditambah:

- localNoveltyCandidateCount;
- contextualBoundaryCandidateCount;
- multiResolutionAcceptedBoundaryCount;
- rejectedLocalOnlyBoundaryCount;
- shortMediumAgreementRate;
- meanShortContextDistance;
- meanMediumContextDistance;
- acceptedBoundaryExamples;
- rejectedLocalOnlyExamples;
- segment count;
- transitions per minute;
- harmonic region duration distribution.
