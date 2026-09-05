# 11 — AI Agent Task Plan

## Global execution rule
Execute ONE task only. Read its REQUIRED SPECS, modify only ALLOWED FILES/PATHS, run REQUIRED TESTS/VERIFY, validate listed acceptance criteria, report, then STOP. Never auto-continue.

Earlier completed tasks remain governed by feature specs and acceptance criteria. Do not rewrite earlier modules merely for convenience.

---

## Phase 3 — Task 3.1: Analysis Orchestrator

REQUIRED SPECS:
`03a-repository-rules.md`, `03b-file-ownership.md`, `03c-dependency-rules.md`, `04-audio-engine.md`, `05-chord-detection.md`, `06-analysis-api-contract.md`, `09-testing-strategy.md`, `10-acceptance-criteria.md`.

ALLOWED FILES:
`engine/chord_engine/analyze.py`, `engine/chord_engine/models.py`, `engine/tests/test_analyze.py`, `docs/implementation-notes.md`.

IMPLEMENT:
Compose only `load audio → extract chroma → detect every frame → smooth → segment`. Provide an engine API such as `analyze_audio(path)`. Preserve source path, duration, sample rate, algorithm id `chroma-template-v1`, and segments. Use existing feature timing. Do not duplicate DSP/detector/smoothing/segmentation logic. Final public segment end must not exceed source duration beyond floating tolerance. If this needs an upstream contract change, STOP and report.

REQUIRED TESTS:
Actual full pipeline, no mocks/bypass: synthetic C major file → C; A minor → Am; valid silence → N; progression `C → G → Am → F` with ~1–2 seconds per chord; invalid audio controlled failure; deterministic repeated analysis; segment sorted/non-overlap/confidence bounds/end within source duration.

VERIFY:
Task tests + complete engine pytest + governance checks. Validate AC-03..09, AC-17, AC-20, AC-26..28 as applicable.

DO NOT:
CLI JSON/process behavior, Electron, cache, playback, UI, beat/key detection, source separation, ML.

STOP after Task 3.1.

---

## Phase 3 — Task 3.2: CLI Contract

REQUIRED SPECS:
`03a`, `03b`, `03c`, `06-analysis-api-contract.md`, `09-testing-strategy.md`, `10-acceptance-criteria.md`.

ALLOWED FILES:
`engine/chord_engine/cli.py`, `engine/chord_engine/models.py`, `engine/tests/test_cli.py`, `docs/implementation-notes.md`.

IMPLEMENT:
`python -m chord_engine.cli analyze <path>`. Serialize Task 3.1 result only. Contract version `1`; stdout JSON only; diagnostics stderr; non-zero exit on failure.

REQUIRED TESTS:
WAV success, MP3 success, JSON parsing, no stdout logs, missing/corrupt file controlled JSON error/nonzero, required fields/types, deterministic serialization.

VERIFY:
Full engine pytest. AC-02,03,04,17,20,26..28.

STOP.

---

## Phase 4 — Task 4.1: Desktop Shell

REQUIRED SPECS:
`00`, `02`, `03`, `03a`, `03b`, `03c`, `07-desktop-app.md`, `10`.

ALLOWED PATHS:
`apps/desktop/**`, `packages/shared/**`, root `package.json`, `pnpm-workspace.yaml`, `pnpm-lock.yaml`, `docs/implementation-notes.md`.

IMPLEMENT:
Electron main/preload/React renderer shell, shared typed analysis contract, dev/build/test scripts, secure defaults. No engine spawn.

VERIFY:
typecheck, relevant Vitest, build/dev bootstrap; AC-19,26..28.

STOP.

---

## Phase 4 — Task 4.2: Secure IPC & File Selection

REQUIRED SPECS:
`03b`, `03c`, `06`, `07`, `10`.

ALLOWED:
desktop main/preload/renderer, shared package, implementation notes.

IMPLEMENT:
Native WAV/MP3 dialog, cancel behavior, minimal typed preload API, selected filename display. No unrestricted Node exposure and no engine process yet.

TEST/VERIFY:
filters, cancel, bridge/security; AC-10,19,26..28.

STOP.

---

## Phase 5 — Task 5.1: Engine Process Integration

REQUIRED SPECS:
`00`, `03b`, `03c`, `06`, `07`, `10`.

ALLOWED:
desktop main/preload, shared package, implementation notes.

IMPLEMENT:
Development engine resolver using repository `.venv` without developer absolute paths; spawn CLI; parse JSON stdout; stderr diagnostics; handle spawn failure, timeout, nonzero exit, malformed JSON; expose typed result through IPC.

TEST:
process-boundary success/error tests and development integration where practical.

STOP.

---

## Phase 5 — Task 5.2: Analysis Cache

REQUIRED SPECS:
`03b`, `03c`, `06`, `07`, `10`.

ALLOWED:
desktop main, shared package, implementation notes.

IMPLEMENT:
Local cache key = SHA-256 file content + analysis algorithm/contract version. Same content/version hit; content/version change miss. Python engine remains cache-free.

Additional cache behavior:

- support force bypass mode for explicit re-analysis requests from desktop flow;
- bypass mode must run engine even when cache entry exists;
- successful force re-analysis must replace cached value for computed key.

TEST/VERIFY:
hit, miss, changed content, version invalidation, cache corruption/error; AC-15,16.

STOP.

---

## Phase 6 — Task 6.1: Playback

REQUIRED SPECS:
`07`, `10`.

ALLOWED:
desktop renderer, shared package, implementation notes.

IMPLEMENT:
Audio load/play/pause/seek/currentTime/duration; idle/analyzing/ready/playing/paused/error states. No chord timeline implementation yet.

TEST/VERIFY:
player state/time logic; AC-11,12.

STOP.

---

## Phase 7 — Task 7.1: Chord Timeline

REQUIRED SPECS:
`06`, `07`, `08-waveform-timeline.md`, `10`.

ALLOWED:
desktop renderer, shared package, implementation notes.

IMPLEMENT:
Render ordered chord segments. `left%=start/duration*100`, `width%=(end-start)/duration*100`. Handle N/empty result. Accessible active-state styling infrastructure.

TEST:
percentage math, ordering, bounds, N/empty.

STOP.

---

## Phase 7 — Task 7.1A: Real-Song Accuracy Improvement

REQUIRED SPECS:
`03a-repository-rules.md`, `03b-file-ownership.md`, `03c-dependency-rules.md`, `04-audio-engine.md`, `05-chord-detection.md`, `06-analysis-api-contract.md`, `09-testing-strategy.md`, `10-acceptance-criteria.md`.

ALLOWED FILES/PATHS:
`engine/chord_engine/analyze.py`, `engine/chord_engine/audio.py`, `engine/chord_engine/features.py`, `engine/chord_engine/detector.py`, `engine/chord_engine/smoothing.py`, `engine/tests/**`, `docs/implementation-notes.md`.

IMPLEMENT:
Improve real-song chord stability in a controlled way while preserving existing synthetic behavior and API contract.

Required implementation direction:

1. Add harmonic-focused preprocessing before chroma analysis (HPSS or equivalent deterministic harmonic-first method using approved stack).
2. Add beat-synchronous harmonic aggregation so decisions are based on musically meaningful regions, not only short independent frames.
3. Add global key estimation as a soft prior to plausibility/confidence scoring.

Key-prior constraints:

- prior influences ranking/confidence but never hard-forbids non-diatonic chords;
- borrowed chords remain possible when audio evidence supports them.

Flicker policy:

- reduce implausible rapid chord switching through better harmonic evidence and region-level scoring;
- do not hide flicker primarily by an excessively large smoothing window.

Must preserve:

- existing chord vocabulary (12 major + 12 minor + N);
- existing analysis API contract;
- existing synthetic regression expectations.

REQUIRED TESTS:

1. Run all previous engine tests and ensure all remain PASS.
2. Keep synthetic `C → G → Am → F` fixture correct.
3. Keep valid silence behavior correct (`N`).
4. Add/maintain tests for key-prior soft behavior where non-diatonic evidence can still win.
5. Add/maintain tests ensuring no contract-shape regression.

REAL-SONG BENCHMARK PROCEDURE:

1. Use one local user-owned real-song audio file via local environment variable (for example `GCD_REAL_SONG_PATH`).
2. Do not commit the song file to repository.
3. Run baseline pipeline and improved pipeline on the same file.
4. Record BEFORE/AFTER summary metrics in benchmark output and implementation notes.
5. If local song path is unavailable, benchmark step must skip with controlled message; automated CI must not require copyrighted audio.

BEFORE/AFTER METRICS:

For baseline and improved, report:

1. number of chord segments;
2. median segment duration;
3. mean segment duration;
4. excessive short-segment rate;
5. detected global key;
6. chord-family distribution (major/minor/N);
7. synthetic regression pass/fail summary.

Expanded deterministic diagnostics are mandatory for Task 7.1A benchmark output:

1. analysis algorithm/version;
2. detected global key;
3. key confidence/score if available;
4. song duration;
5. min/max segment duration;
6. short-segment bucket counts + percentages for <250ms, <500ms, <1s;
7. chord occurrence count by label;
8. total duration by chord label;
9. percentage-of-song duration by chord label;
10. diatonic vs non-diatonic segment count relative to detected key;
11. diatonic vs non-diatonic total duration relative to detected key;
12. suspicious short non-diatonic segments list with chord/start/end/duration/confidence.

The diagnostic diatonicity output is reporting-only and must not auto-rewrite detected non-diatonic chords.

ACCEPTANCE CRITERIA:

Task 7.1A must satisfy AC-29, AC-30, AC-31, AC-32, AC-33, AC-34, AC-35 plus AC-20 and governance AC-26..28.

DO NOT:

- do not modify Electron main/preload/renderer for this task;
- do not add ML/deep-learning model/runtime;
- do not add cloud processing or external stem extraction service;
- do not hardcode key/progression for any specific song;
- do not tune constants specifically for one benchmark file;
- do not replace baseline blindly without BEFORE/AFTER comparison evidence.

STOP RULE:

After implementing Task 7.1A, report changed files, test results, benchmark metrics (or controlled skip reason), AC status, deviations/blockers, then STOP.

### Task 7.1A Subtask — Context-Aware Short-Segment Correction

REQUIRED SPECS:
Same as Task 7.1A.

ALLOWED FILES/PATHS:
Same as Task 7.1A.

GOAL:
Correct short, weak, isolated chord anomalies using temporal + harmonic context while preserving genuine musical changes and valid non-diatonic functions.

BEFORE-STATE REFERENCE SNAPSHOT (local benchmark):

1. segment count: 213;
2. mean segment duration: about 1.326 s;
3. median segment duration: about 1.068 s;
4. segments <500 ms: about 21.13%;
5. segments <1 s: about 45.54%;
6. non-diatonic segments: 13;
7. non-diatonic duration: about 7.22 s;
8. detected global key: F#m;
9. key confidence: about 0.751.

These values are comparison context only and must not be used as hardcoded target behavior.

IMPLEMENT:

1. Add deterministic context-aware short-segment correction logic using prev/candidate/next chord context, duration, confidence, and harmonic plausibility.
2. Do not auto-remove solely by short duration.
3. Do not auto-remove solely by non-diatonic status.
4. Keep key as soft prior only.
5. Preserve musically valid borrowed/secondary-function chords when evidence is strong.
6. Do not increase global smoothing window solely to hide detection errors.

REQUIRED TESTS:

1. Full engine pytest remains PASS.
2. Synthetic progression `C -> G -> Am -> F` remains correct.
3. Silence/N remains correct.
4. Add deterministic tests showing:
	- weak isolated short anomaly can be corrected;
	- strong short genuine chord is preserved;
	- sustained genuine transitions are preserved;
	- valid non-diatonic/secondary-dominant cases can survive.
5. API contract shape remains unchanged.

BENCHMARK:

1. Run BEFORE/AFTER comparison on same local song using existing benchmark system.
2. Report quality-oriented metrics, not segment-count-only claims.

STOP RULE:

Report specification files changed, implementation files changed, correction strategy, thresholds/configuration, test results, benchmark before/after metrics, examples corrected/preserved, acceptance status, blockers; then STOP.

### Task 7.1A Subtask — Musical-Time Harmonic Persistence

REQUIRED SPECS:
Same as Task 7.1A.

ALLOWED FILES/PATHS:
Same as Task 7.1A.

GOAL:
Represent sustained harmonic changes in musical time and reduce false transition density caused by short-term chroma perturbations.

IMPLEMENT:

1. Add deterministic beat-aware musical-time aggregation strategy.
2. Add persistence/hysteresis where SWITCH requires stronger evidence than KEEP.
3. Candidate chord must not replace current chord by a single short-region win only.
4. Use factors: consecutive support, confidence, score advantage, beat timing, harmonic plausibility, neighbor persistence.
5. Preserve genuine sustained transitions and strong-evidence short genuine chords.
6. Do not solve by merely enlarging global smoothing window or arbitrary large min segment duration.

REQUIRED TESTS:

1. Full engine regression suite PASS.
2. Stable-chord moving-melody case does not create false transitions excessively.
3. Stable-chord moving-bass case does not create false transitions excessively.
4. Passing-note short event does not trigger chord switch.
5. Genuine C -> G transition preserved.
6. C -> G -> Am -> F regression preserved.
7. Genuine short strong chord can survive.
8. Repeated analysis deterministic.

BENCHMARK:

1. Run BEFORE/AFTER on same local real-song file.
2. Report transition-density metrics: segment count, transitions/min, mean/median duration, <500ms rate, <1s rate, optional tempo estimate, transitions/beat when reliable.
3. Success is quality-oriented (lower false transition density + preserved genuine transitions), not segment-count-only.

STOP RULE:

Report strategy, beat behavior, keep/switch criteria, thresholds, before/after metrics, synthetic regression results, examples rejected false transitions, examples preserved genuine transitions, AC status, blockers; then STOP.

### Task 7.1A Subtask — Transition Acceptance Refinement

REQUIRED SPECS:
Same as Task 7.1A.

ALLOWED FILES/PATHS:
Same as Task 7.1A.

GOAL:
Refine transition acceptance to reduce weak/ambiguous accepted switches while preserving genuine sustained harmonic changes, with extra safeguards for root-preserving quality switches.

IMPLEMENT:

1. Focus only on transition acceptance logic inside musical-time persistence stage.
2. Ordinary override must not accept switch when switch evidence is weaker than keep evidence.
3. If acceptance may occur despite `switchScore <= keepScore`, enforce explicit stronger multi-beat/context support and record a diagnosable reason.
4. Root-preserving quality switches (same-root major/minor) must use stronger hysteresis than ordinary continuation.
5. Add harmonic-change evidence comparing current-harmony profile vs candidate-harmony profile across musical-time observations.
6. Preserve legitimate major/minor changes when sustained strong evidence exists.
7. Do not add global smoothing inflation as a substitute for acceptance quality.

MUST PRESERVE:

- beat-aware aggregation and existing persistence system;
- key as soft prior;
- secondary dominants;
- non-diatonic chords;
- N/silence behavior;
- public API contract;
- all previous synthetic tests.

REQUIRED TESTS:

1. Full engine regression suite PASS.
2. Stable E major with moving melody does not flicker to Em.
3. Stable A major does not flicker to Am.
4. Genuine E -> Em sustained quality change remains detectable.
5. Genuine C#m -> C# quality change can survive when strongly supported.
6. Ordinary override never accepts when `switchScore <= keepScore`.
7. Explicit multi-beat/context-supported override may accept with diagnosable reason.
8. C -> G -> Am -> F regression remains correct.
9. Repeat analysis deterministic.

BENCHMARK:

1. Run BEFORE/AFTER on same local real-song file.
2. Report transition acceptance diagnostics:
	- accepted transition reason counts;
	- root-preserving major/minor accepted-switch count;
	- accepted override count where `switchScore <= keepScore`;
	- transition count + transitions per minute before/after.
3. Success criteria are quality-based:
	- fewer weak/ambiguous accepted transitions;
	- no regression on genuine sustained changes;
	- synthetic ground-truth tests remain PASS.

STOP RULE:

Report transition acceptance rules, quality-change hysteresis rules, override safety rules, before/after metrics, root-preserving switch count, weak override count, examples rejected/preserved, regression results, AC status, blockers; then STOP.

### Task 7.1A Subtask — Root-Aware Harmonic Scoring

REQUIRED SPECS:
Same as Task 7.1A.

ALLOWED FILES/PATHS:
Same as Task 7.1A.

GOAL:
Improve chord identity accuracy on full-mix music by separating root evidence from major/minor quality evidence while preserving the MVP vocabulary (12 major, 12 minor, N).

IMPLEMENT:

1. Evaluate root evidence and quality evidence separately.
2. Estimate root from harmonic musical-time evidence (not only full-template cosine).
3. Use deterministic evidence where relevant:
	- harmonic chroma,
	- pitch-class energy,
	- root/fifth support,
	- bass/low-frequency chroma support,
	- beat-synchronous aggregation.
4. Treat bass evidence as supporting evidence only.
5. Evaluate major/minor quality using third evidence (major/minor third) with root/fifth support.
6. Allow quality ambiguity when third evidence is weak; do not manufacture high-confidence quality decisions from weak evidence.
7. Combine root-aware evidence with existing template scorer deterministically.
8. Integrate with existing key soft prior, persistence, and transition-acceptance refinement.
9. Do not increase global smoothing/persistence simply to reduce segment count.

DO NOT:

- add chord outputs beyond 12 major + 12 minor + N;
- hardcode song/key/progression;
- tune thresholds for one song only;
- change Electron/UI;
- break N/silence behavior;
- add ML/deep learning.

REQUIRED TESTS:

1. Full engine regression PASS.
2. 12 major roots synthetic coverage PASS.
3. 12 minor roots synthetic coverage PASS.
4. C major vs C minor quality discrimination PASS.
5. A major vs A minor quality discrimination PASS.
6. Melody contamination test: stable E major remains E.
7. Moving-bass test: stable A major remains rooted A when harmonic evidence supports A.
8. Inversion-like bass condition does not auto-change root.
9. Weak-third evidence does not cause unstable quality flicker.
10. Genuine major -> minor quality change remains detectable.
11. C -> G -> Am -> F regression remains correct.
12. Silence/N remains correct.
13. Deterministic repeated analysis remains correct.

BENCHMARK:

1. Run BEFORE/AFTER on same local real-song file.
2. Report at minimum:
	- segment count,
	- transitions/minute,
	- root-change count,
	- quality-change count,
	- root-preserving quality switches,
	- weak override count,
	- chord distribution,
	- mean/median segment duration.
3. Include diagnostics:
	- root candidate scores,
	- selected root confidence,
	- major/minor quality evidence,
	- quality margin,
	- template score,
	- root-aware combined score,
	- ambiguous-quality decision count.

SUCCESS GATE:

- quality/identity oriented, not segment-count-only;
- unsupported identity changes reduced;
- genuine root and quality changes preserved;
- synthetic ground truth preserved;
- API contract unchanged.

STOP RULE:

Report root scoring strategy, bass strategy, quality strategy, combination logic, thresholds, regression results, benchmark before/after metrics, corrected ambiguous examples, inversion-safe examples, ambiguous-quality cases, AC status, blockers/deviations; then STOP.

### Task 7.1A Subtask — Ground-Truth Chord Evaluation

REQUIRED SPECS:
Same as Task 7.1A.

ALLOWED FILES/PATHS:
`engine/chord_engine/cli.py`, `engine/chord_engine/analyze.py`, `engine/chord_engine/models.py`, `engine/chord_engine/evaluation.py`, `engine/tests/**`, `docs/implementation-notes.md`.

GOAL:
Provide deterministic workflow to evaluate engine chord output against manual chord ground truth annotations on local audio excerpts.

IMPLEMENT:

1. Add simple local annotation format with required fields: `start`, `end`, `chord`.
2. Support partial-song annotation/evaluation windows.
3. Add CLI evaluation command comparing engine output vs manual annotation.
4. Report at minimum:
	- time-weighted chord accuracy,
	- exact chord match percentage,
	- root accuracy,
	- major/minor quality accuracy,
	- false transition count,
	- missed transition count,
	- boundary timing error,
	- confusion pairs.
5. Keep evaluation deterministic and song-agnostic (no hardcoded local song path/progression).
6. Do not modify detector thresholds or heuristics in this subtask.

GOVERNANCE:

- local user-owned annotation files remain local unless explicitly intended to be committed;
- preserve public engine API and existing regression behavior.

REQUIRED TESTS:

1. Add deterministic tests using synthetic annotation fixtures.
2. Full engine regression suite remains PASS.
3. Existing synthetic progression/silence/determinism tests remain PASS.

STOP RULE:

Report annotation format, evaluation CLI command, metrics produced, tests run/results, files changed, AC status, blockers/deviations; then STOP.

### Task 7.1A Subtask — Harmonic Change-Point Segmentation and Global Chord Decoding

REQUIRED SPECS:
Same as Task 7.1A.

ALLOWED FILES/PATHS:
`engine/chord_engine/analyze.py`, `engine/chord_engine/features.py`, `engine/chord_engine/detector.py`, `engine/tests/**`, `docs/implementation-notes.md`.

GOAL:
Refine architecture to boundary-first harmonic segmentation and deterministic global chord decoding.

IMPLEMENT:

1. Build harmonic beat-synchronous representation from existing harmonic preprocessing pipeline.
2. Compute deterministic harmonic novelty / change-point evidence.
3. Select sustained harmonic boundaries (reject short novelty from passing tones/melody contamination).
4. Build stable harmonic regions that may span multiple beats.
5. Classify per-region chord evidence using root-aware scoring + quality evidence + template support + key soft prior.
6. Add deterministic global sequence decoder (Viterbi/HMM-style DP or equivalent) over regions.
7. Use soft transition costs only; do not hard-ban non-diatonic chords or rapid genuine changes.
8. If prior persistence/correction stage is superseded in improved path, bypass/remove redundant stage and document it.

DO NOT:

- do not continue stacking heuristic transition repair over old short-region classifier;
- do not hardcode BPM/chord-count-per-bar/song key/progression;
- do not add chord vocabulary beyond 12 major + 12 minor + N;
- do not touch Electron/UI.

REQUIRED TESTS:

1. Full engine regression suite PASS.
2. Synthetic harmonic-boundary tests:
	- sustained A + moving melody remains one harmonic region;
	- sustained A + moving bass does not create false boundary;
	- passing tones do not create boundary;
	- genuine A -> E creates boundary;
	- C -> G -> Am -> F yields four correct regions/chords;
	- rapid genuine two-chord case survives with strong evidence;
	- sustained same-root quality change boundary remains possible;
	- silence/N remains valid;
	- repeated runs deterministic.
3. C -> G -> Am -> F regression PASS.

BENCHMARK:

1. Compare previous pipeline baseline vs new pipeline on same local song.
2. Report:
	- boundary count before/after;
	- transitions per minute before/after;
	- mean/median harmonic region duration;
	- rejected novelty peaks;
	- accepted boundary timestamps;
	- per-region selected chord + top-3 candidates/scores;
	- global decoder path score;
	- local-best vs globally-decoded differences.
3. Success gate is quality-oriented; segment count reduction alone is insufficient.

STOP RULE:

Report method, novelty calculation, boundary selection, region representation, global decoding strategy, transition-cost model, replaced/retained old stages, synthetic results, real-song BEFORE/AFTER metrics, preserved/blocked examples, AC status, blockers/deviations; then STOP.

### Task 7.1A Subtask — Beat-Aware Harmonic Boundary Consolidation

REQUIRED SPECS:
Same as Task 7.1A.

ALLOWED FILES/PATHS:
`engine/chord_engine/analyze.py`, `engine/chord_engine/features.py`, `engine/tests/**`, `docs/implementation-notes.md`.

GOAL:
Consolidate locally clustered harmonic boundaries in musical time before final region classification/global decoding.

IMPLEMENT:

1. Detect clusters of nearby candidate/accepted boundaries.
2. Use beat distance as primary proximity measure when beat tracking is reliable.
3. Evaluate collapse decision with novelty strength/prominence, harmonic similarity pre/mid/post, intermediate-region duration, and intermediate chord/root evidence.
4. Collapse weak short intermediate regions only when evidence supports one larger transition.
5. Preserve short intermediate regions with strong independent evidence.
6. Preserve genuine rapid changes with strong evidence.
7. Avoid post-label repair stacking; consolidation must happen at boundary-candidate stage.

DO NOT:

- do not blanket-raise global novelty threshold;
- do not add global smoothing;
- do not add another post-label correction heuristic;
- do not hardcode song-specific progression/timestamps.

REQUIRED TESTS:

1. Full engine regression suite PASS.
2. Deterministic boundary-consolidation synthetic tests:
	- clustered novelty peaks around one true change collapse to one;
	- transient peak near true boundary removed;
	- short weak intermediate region collapses;
	- rapid genuine dual-change preserved;
	- C -> G -> Am -> F remains correct;
	- moving melody/bass stability preserved;
	- same-root quality genuine change preserved;
	- silence/N preserved;
	- repeated runs deterministic.

BENCHMARK:

1. Compare boundary diagnostics before/after consolidation on local song.
2. Report candidate/accepted before/after counts, consolidated count, cluster count, beat-distance stats, short-region count, region-duration distribution bins, and cluster examples.
3. Success gate is quality-oriented, not segment-count-only.

STOP RULE:

Report consolidation strategy, beat-distance rules, similarity rules, rapid-change protection, boundary counts before/after, cluster/consolidated counts, region-duration distribution, collapsed/preserved examples, regression results, AC status, blockers/deviations; then STOP.

### Task 7.1A Subtask — Multi-Resolution Harmonic Context Representation

REQUIRED SPECS:
Same as Task 7.1A.

ALLOWED FILES/PATHS:
`engine/chord_engine/analyze.py`, `engine/chord_engine/features.py`, `engine/tests/**`, `docs/implementation-notes.md`.

GOAL:
Improve harmonic representation for change-point detection so melody/bass/voicing/passing-tone excursions are less likely to become accepted harmonic boundaries.

IMPLEMENT:

1. Add deterministic multi-resolution contextual harmonic representation before final boundary acceptance.
2. Keep at least two context scales:
	- short context for relatively fast genuine changes,
	- medium context to validate sustained harmonic separation.
3. Evaluate candidate boundaries using local novelty + short-context before/after distance + medium-context before/after distance + persistence/context agreement signals.
4. Reject local-only novelty events that fail medium-context support.
5. Preserve genuine rapid changes when both context scales support change.
6. Keep boundary-first architecture and retain existing boundary consolidation stage.

DO NOT:

- do not add extra post-boundary merging as primary strategy;
- do not raise global threshold as blanket fix;
- do not add new persistence/label-repair layer as workaround;
- do not hardcode one song/key/style/timestamps.

REQUIRED TESTS:

1. Full engine regression suite PASS.
2. Deterministic multi-resolution synthetic tests:
	- sustained A + melody contamination -> no false boundary,
	- sustained A + moving bass -> no false boundary,
	- sustained chord + arpeggiated/voicing change -> no false boundary,
	- passing-tone local novelty without medium support -> rejected,
	- genuine A -> E boundary preserved,
	- C -> G -> Am -> F preserved exactly,
	- two genuine rapid changes preserved when short+medium agree,
	- same-root A -> Am sustained quality change preserved,
	- silence/N preserved,
	- repeated runs deterministic.
3. Dedicated C -> G -> Am -> F regression PASS.

BENCHMARK:

1. Run BEFORE/AFTER on same local real-song file.
2. Report:
	- localNoveltyCandidateCount,
	- contextualBoundaryCandidateCount,
	- multiResolutionAcceptedBoundaryCount,
	- rejectedLocalOnlyBoundaryCount,
	- shortMediumAgreementRate,
	- meanShortContextDistance,
	- meanMediumContextDistance,
	- acceptedBoundaryExamples,
	- rejectedLocalOnlyExamples,
	- segmentCount before/after,
	- transitionsPerMinute before/after,
	- harmonic region duration distribution.

STOP RULE:

Report contextual feature strategy, short-context definition, medium-context definition, novelty/context combination rule, self-similarity usage (if any), local candidates, contextual accepted boundaries, rejected local-only boundaries, agreement statistics, before/after transition density and segment counts, false-local novelty rejected examples, genuine-boundary preserved examples, regression results, AC status, blockers/deviations; then STOP.

---

## Phase 7 — Task 7.1B: Re-analyze Cache Bypass Flow

REQUIRED SPECS:
`03b-file-ownership.md`, `03c-dependency-rules.md`, `06-analysis-api-contract.md`, `07-desktop-app.md`, `10-acceptance-criteria.md`.

ALLOWED FILES/PATHS:
`apps/desktop/src/main/**`, `apps/desktop/src/preload/**`, `apps/desktop/src/renderer/**`, `packages/shared/**`, `docs/implementation-notes.md`.

IMPLEMENT:

1. Add `Re-analyze` button in renderer; show/enable only when selected file path valid.
2. Normal open behavior remains cache-enabled.
3. Re-analyze behavior must bypass cache lookup and force engine run for selected file.
4. On re-analyze start: clear current analysis immediately and set `analyzing`.
5. On success: update UI to `ready` and overwrite cached result for current cache key.
6. On failure: keep stale result cleared and set controlled `error` state.
7. Protect against stale async responses: older request must not overwrite newer result.
8. Do not clear whole application cache to implement this flow.
9. If analysis algorithm revision exists, cache scope must use updated algorithm identifier.

REQUIRED TESTS:

1. normal open uses cache hit behavior;
2. re-analyze bypasses cache;
3. re-analyze invokes engine even when cache exists;
4. successful re-analyze overwrites cache value;
5. algorithm-version change causes cache miss;
6. stale prior response cannot overwrite newer re-analysis;
7. re-analysis failure does not restore stale result.

VERIFY:
Relevant desktop/main/preload/renderer tests + typecheck/build for desktop workspace. Validate AC-15, AC-16, AC-36..40, AC-19, AC-20, AC-26..28.

DO NOT:

- do not modify chord-detection tuning;
- do not modify engine algorithm behavior for this task;
- do not change analysis API response shape.

STOP RULE:

Report root cause findings about stale-cache behavior, cache key/version behavior, files changed, tests run, and PASS/FAIL/BLOCKED; then STOP.

---

## Phase 7 — Task 7.2: Active Chord

Same REQUIRED SPECS and ALLOWED paths as Task 7.1.

IMPLEMENT:
Active iff `segment.start <= currentTime < segment.end`; update during playback and seek; boundary/N behavior deterministic.

VERIFY:
AC-13.

STOP.

---

## Phase 7 — Task 7.3: Timeline Seek

Same REQUIRED SPECS and ALLOWED paths as Task 7.1.

IMPLEMENT:
Click position → playback time, clamp to duration, update audio.

TEST/VERIFY:
seek target within ±250ms; AC-14.

STOP.

---

## Phase 8 — Task 8.1: Quality Gate

REQUIRED SPECS:
`09`, `10`, governance specs.

ALLOWED:
tests/config/implementation notes only. If a failure requires implementation outside this scope, STOP and identify the owning earlier task.

RUN:
full Python tests, TS tests, lint, typecheck, build, local WAV/MP3 end-to-end.

REPORT:
AC-01..20 and AC-26..28 as PASS/FAIL/BLOCKED/NOT-YET-APPLICABLE.

No new features. STOP.

---

## Phase 9 — Task 9.1: Functional MVP Validation

Validate actual flow:
`select → analyze → timeline → play → active chord → seek → cache → reopen`.

Validate offline behavior. Do not add new features. Any failure must be traced to owning task. STOP.

---

## Phase 10 — Task 10.1: Standalone Engine

REQUIRED SPECS:
`00`, `03a`, `03b`, `06`, `10`.

ALLOWED:
engine packaging files, `resources/engine/**`, desktop main engine resolver/package config, notes.

IMPLEMENT:
PyInstaller standalone engine. Production must not call system Python. Test packaged executable against same JSON contract.

VERIFY:
AC-23. STOP.

---

## Phase 10 — Task 10.2: Bundled FFmpeg

REQUIRED SPECS:
`00`, `03a`, `07`, `10`.

ALLOWED:
`resources/ffmpeg/**`, desktop main resolver/package config, notes.

IMPLEMENT:
Bundle platform FFmpeg if required by production decode path; no dependency on end-user PATH.

VERIFY:
AC-24. STOP.

---

## Phase 10 — Task 10.3: macOS Apple Silicon Package

REQUIRED SPECS:
`00`, `07`, `10`.

ALLOWED:
desktop/package/resources/root packaging config/notes.

IMPLEMENT:
electron-builder `darwin-arm64`, `.app` and `.dmg`; production engine/resource resolution; smoke test without relying on development `.venv` or global FFmpeg.

VERIFY:
AC-18,19,22..28. STOP.
