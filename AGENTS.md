# AGENTS.md

## Purpose

This repository is an AI-assisted development project for a desktop Guitar Chord Detector.

The application analyzes local audio files, detects chord progressions, and displays the results in an Electron desktop UI with playback and timeline visualization.

This file defines repository-wide instructions for Codex and other coding agents.

Treat this file as mandatory project guidance.

---

## Source of Truth

Before making changes, read the relevant specifications under:

```text
docs/specs/
```

The specification files are the primary source of truth for architecture, scope, task boundaries, acceptance criteria, ownership, and implementation order.

At minimum, understand these files when relevant:

```text
docs/specs/README.md
docs/specs/00-development-environment.md
docs/specs/01-product-scope.md
docs/specs/02-architecture.md
docs/specs/03-project-structure.md
docs/specs/03a-repository-rules.md
docs/specs/03b-file-ownership.md
docs/specs/03c-dependency-rules.md
docs/specs/04-audio-engine.md
docs/specs/05-chord-detection.md
docs/specs/09-testing-strategy.md
docs/specs/10-acceptance-criteria.md
docs/specs/11-agent-task-plan.md
```

If another spec is referenced by the active task, read it too.

Do not assume a requirement from memory when the current specification can be read directly.

If this file conflicts with a more specific active-task instruction in `docs/specs/`, follow the more specific task instruction unless it violates repository governance or architecture rules.

---

## Existing Copilot Instructions

This repository may also contain:

```text
.github/copilot-instructions.md
```

Codex must read and follow it as supplementary project guidance.

However:

- `AGENTS.md` is the repository-wide agent entry point.
- `docs/specs/` remains the authoritative functional and architectural source of truth.
- `.github/copilot-instructions.md` remains valid and must not be ignored merely because Codex is being used instead of Copilot.

Do not duplicate or rewrite the full specifications into implementation code comments.

---

## Development Environment

Respect the versions and tooling defined in:

```text
docs/specs/00-development-environment.md
```

Current project assumptions include:

```text
Node.js: 24.15.0
Python: 3.11 virtual environment
Desktop: Electron + React + TypeScript
Audio engine: Python
```

Use the repository's existing virtual environment and package configuration.

Do not silently upgrade Python, Node.js, Electron, React, TypeScript, librosa, NumPy, SciPy, or soundfile unless the active task explicitly requires it.

Do not introduce a new package when the existing stack can implement the requirement cleanly.

---

## Repository Architecture

Maintain strict separation between the desktop application and the Python audio engine.

Conceptually:

```text
Desktop UI / Renderer
        |
        v
Electron preload / secure IPC
        |
        v
Electron main process
        |
        v
Python CLI / process boundary
        |
        v
Audio analysis engine
```

The renderer must not directly import Python code.

The Python engine must not depend on Electron, React, DOM APIs, browser APIs, or desktop UI code.

Do not collapse architectural boundaries for convenience.

---

## Python Audio Engine Responsibilities

The Python engine owns audio and chord analysis.

Major responsibilities include:

```text
audio decode / preprocessing
harmonic feature extraction
chroma representation
beat-aware processing
harmonic change detection
harmonic region segmentation
root-aware chord scoring
major/minor quality scoring
key estimation as a soft prior
global chord sequence decoding
temporal stabilization where specified
final chord segmentation
benchmark diagnostics
CLI analysis contract
```

Keep implementation ownership aligned with the existing file ownership rules.

Do not move DSP logic into orchestration files merely because it is easier.

---

## Chord Vocabulary

Unless a specification explicitly changes this, the public MVP chord vocabulary is:

```text
12 major chords
12 minor chords
N
```

Do not add 7, maj7, m7, sus2, sus4, dim, aug, slash chords, extensions, or alternate chord notation unless explicitly requested by a future task.

Internal diagnostics may reason about harmonic ambiguity, but the public contract must remain within the approved vocabulary.

---

## Pitch-Class Order

When pitch-class arrays are used, preserve the canonical order:

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

Do not change this ordering in one layer without updating every contract and test that depends on it.

---

## Current Analysis Philosophy

The project has evolved beyond simple frame-by-frame template matching.

Do not revert the architecture to:

```text
classify many short frames
-> aggressively smooth labels
-> merge afterward
```

The intended real-song direction is harmonic-boundary-first analysis.

Conceptually:

```text
audio
  ↓
harmonic-focused preprocessing
  ↓
beat-synchronous harmonic features
  ↓
multi-resolution harmonic context
  ↓
harmonic novelty / change-point detection
  ↓
beat-aware boundary consolidation
  ↓
stable harmonic regions
  ↓
root-aware chord scoring
  ↓
major/minor quality evaluation
  ↓
key soft prior
  ↓
global sequence decoding
  ↓
final chord segments
```

Do not replace this with a simpler legacy pipeline unless the active task explicitly calls for an architectural experiment.

---

## Important Real-Song Accuracy Rules

Real-song accuracy is more important than merely reducing segment count.

Do not optimize directly for fewer segments, fewer transitions, or longer average chord duration. Those are diagnostic metrics, not truth.

A change is only an improvement when it reduces incorrect harmonic decisions while preserving genuine chord changes.

Never hardcode `Album Lama`, any specific song title, a specific key such as F#m or A major, a known progression, timestamps from a benchmark song, a genre-specific assumption, or expected chord counts.

A local user-owned song may be used for benchmarking, but production code must remain song-independent.

---

## Key Estimation

Global key estimation is a soft prior only.

Do not treat the detected key as a hard constraint.

Valid music may contain secondary dominants, borrowed chords, modal mixture, non-diatonic chords, and inversion-like bass behavior.

Do not automatically reject a chord simply because it is not diatonic to the estimated key.

Relative major/minor ambiguity is expected and must not be treated as a trivial classification error.

---

## Root-Aware Scoring

Where root-aware scoring is active:

- evaluate root evidence separately from major/minor quality evidence;
- use harmonic evidence across the region;
- treat bass evidence as supporting evidence only;
- do not assume the lowest note is always the chord root;
- allow inversion-like conditions without forcing a root change;
- use third evidence to distinguish major/minor quality;
- allow ambiguous quality diagnostics when third evidence is weak.

Do not force confident major/minor quality from weak evidence.

---

## Harmonic Boundary Rules

A harmonic boundary should represent a meaningful harmonic change, not simply a local spectral event.

Boundary logic may use beat-synchronous harmonic chroma, novelty, prominence, short-context harmonic distance, medium-context harmonic distance, multi-resolution agreement, recurrence/self-similarity where specified, harmonic region similarity, contextual persistence, and chord/root evidence.

Do not solve local fragmentation by blindly increasing one global threshold, adding large smoothing windows, imposing a fixed minimum chord duration, forcing one chord every fixed number of beats, or forcing one chord per bar.

Genuine rapid chord changes must remain possible when evidence is strong.

---

## Multi-Resolution Harmonic Context

When implemented by the active specs, preserve at least two musical context scales:

```text
short context
medium context
```

The short context helps preserve genuine faster harmonic motion.

The medium context helps reject temporary novelty caused by melody, passing tones, bass movement, voicing changes, and transient events.

Do not replace multi-resolution context with a single large moving average.

---

## Boundary Consolidation

Boundary consolidation is allowed only at the harmonic-boundary candidate stage.

It must not become a generic final-label cleanup mechanism.

When nearby boundary candidates exist, evaluate musical evidence before collapsing them.

A short intermediate region must not be removed only because it is short.

Preserve a short region when it has strong independent harmonic support.

---

## Global Sequence Decoding

If the active pipeline uses a global decoder, preserve its role.

The decoder may consider region chord evidence, root-aware scores, quality evidence, template evidence, key soft prior, transition costs, self-transition preference, and harmonic relationships between neighboring candidates.

Transition costs are soft penalties. They must never make a musically valid transition impossible.

Do not encode an expected song progression in the decoder.

---

## Legacy Heuristic Accumulation

Avoid stacking more corrective heuristics on top of old heuristics.

Before adding another stabilization layer, determine whether an earlier stage should be replaced or simplified.

Prefer better representation, better segmentation, better evidence, and better sequence decoding over more exceptions, more magic thresholds, and more post-hoc replacements.

If a new stage supersedes an old stage, document which stage remains active and which one is bypassed or removed.

---

## Benchmarking

The project contains benchmark/diagnostic tooling for comparing baseline and improved analysis.

Benchmark metrics may include:

```text
segmentCount
transitionCount
transitionsPerMinute
transitionsPerBeat
meanSegmentDuration
medianSegmentDuration
shortSegmentRates
globalKey
globalKeyConfidence
rootChangeCount
qualityChangeCount
weakOverrideCount
ambiguousQualityDecisionCount
candidateBoundaryCount
acceptedBoundaryCountBeforeConsolidation
acceptedBoundaryCountAfterConsolidation
consolidatedBoundaryCount
boundaryClusterCount
harmonicRegionsPerMinute
short/medium context statistics
local novelty metrics
global decoder diagnostics
```

These are diagnostics. Do not claim that lower values automatically mean better chord accuracy.

---

## Ground-Truth Evaluation

When a manually annotated ground-truth workflow exists, treat it as the primary accuracy evaluation source.

Preferred metrics include:

```text
time-weighted exact chord accuracy
root accuracy
major/minor quality accuracy
false transition count
missed transition count
boundary timing error
confusion pairs
```

Ground-truth annotations for user-owned songs are local development data unless explicitly approved for repository inclusion.

Do not commit copyrighted/user-owned audio or private annotation data.

---

## Local Real-Song Benchmark Input

Real-song benchmark execution may use:

```text
GCD_REAL_SONG_PATH
```

When that environment variable is absent or invalid, controlled skip behavior is expected where defined by the tests.

Do not replace controlled skip behavior with a hard failure unless the specifications explicitly require it.

Do not add absolute developer-specific paths into source code or tests.

---

## Audio Files and Fixtures

Synthetic fixtures intended for automated tests may live in approved fixture paths.

User-owned real songs must not be committed unless explicitly authorized.

Never copy a user-owned MP3 into a tracked test fixture path without instruction, hardcode its local path, or make CI depend on a private local song.

---

## Desktop Application Responsibilities

The desktop app owns native file selection, secure IPC, analysis process invocation, cache, playback, timeline visualization, active chord visualization, and user interaction state.

The desktop layer must not implement its own chord detection.

---

## Electron Security

Maintain secure Electron defaults.

Unless a specification explicitly states otherwise:

- keep `contextIsolation` enabled;
- do not expose Node APIs directly to the renderer;
- use a narrow preload API;
- use explicit IPC channels;
- validate IPC inputs;
- do not expose arbitrary filesystem or process execution to the renderer.

Do not weaken security to make a task easier.

---

## Renderer State Rules

When a new audio file is selected:

- clear stale analysis immediately;
- reset previous playback where required;
- show the correct loading/analyzing state;
- never temporarily display analysis belonging to the previously selected file;
- prevent older asynchronous analysis responses from overwriting newer file state.

Treat stale analysis as a correctness bug.

---

## Analysis Cache

Normal file selection may use cached analysis.

Re-analysis must be able to bypass cache when that workflow exists.

Cache identity should include the relevant approved identity fields, such as file content hash, contract version, and algorithm version.

Do not use filename alone as cache identity.

When the algorithm changes materially, ensure stale analysis does not silently survive under the same cache namespace.

Do not delete the entire cache as a shortcut for implementing correct invalidation.

---

## Playback

Playback is independent from chord-analysis readiness unless the specification explicitly couples them.

A valid selected audio source should be playable when the playback contract allows it.

Do not make playback availability depend on an unrelated analysis state.

Handle rejected `HTMLMediaElement.play()` calls correctly.

Do not show `playing` state until playback actually succeeds.

Selecting a new file must stop/reset playback of the previous file where specified.

---

## Timeline

The current timeline uses multi-row time windows for readability.

Do not force the full song into one single compressed bar.

Preserve original analysis segment timing.

Renderer code may split a segment visually across row boundaries, but must not mutate the original analysis result.

Narrow visual fragments may hide text labels, but the underlying chord segment must remain intact.

---

## Active Chord

When Task 7.2 or later active-chord behavior is being implemented, active chord selection must be derived from playback time and the current analysis result.

Do not duplicate chord inference in the renderer.

The UI must highlight the segment that owns the current playback timestamp according to the agreed boundary convention.

---

## Task Execution Discipline

When asked to execute a task such as:

```text
Execute ONLY Phase 7 — Task 7.1A ...
```

you must:

1. Read `docs/specs/11-agent-task-plan.md`.
2. Locate the exact task/subtask.
3. Read every spec listed under REQUIRED SPECS.
4. Respect ALLOWED FILES/PATHS.
5. Implement only that task.
6. Run the required tests.
7. Validate the listed acceptance criteria.
8. Report the result.
9. STOP.

Do not automatically continue to the next task.

---

## Scope Discipline

Do not modify unrelated files.

Do not perform opportunistic refactors outside the active task.

Do not rename modules, folders, APIs, or contracts unless required.

Do not introduce formatting-only churn across unrelated files.

If a required change appears to violate the task's allowed paths, stop and report the conflict rather than silently expanding scope.

---

## Specification-First Changes

If the user asks for behavior that is not yet defined in the specification:

1. update the relevant `docs/specs/` files first;
2. ensure `docs/specs/11-agent-task-plan.md` reflects the new task/subtask;
3. then implement the newly specified behavior if the request includes implementation.

If the request explicitly says "update specs only", do not implement code.

---

## Testing

Run the smallest relevant test first, then the broader required suites.

Examples may include:

```bash
.venv/bin/python -m pytest engine/tests/test_<area>.py -q
.venv/bin/python -m pytest engine/tests -q
```

Use the repository's actual paths and commands if they differ.

For desktop work, run the scripts required by the specs, such as typecheck, unit tests, and build.

Do not report PASS if a required test did not run.

If a test is intentionally skipped because a local external input is missing, report it explicitly as controlled skip.

---

## Regression Protection

Any real-song accuracy improvement must preserve existing synthetic ground truth.

Important regression cases include, where defined:

```text
C -> G -> Am -> F
major/minor exact templates
silence -> N
stable chord under melody movement
stable chord under moving bass
stable chord under voicing/arpeggio variation
genuine root transition
genuine major/minor quality change
genuine rapid changes
deterministic repeated analysis
```

Do not accept an apparent real-song improvement that breaks known synthetic ground truth.

---

## Determinism

Analysis must remain deterministic.

Given identical audio, configuration, and algorithm version, the engine should return identical results within the project's defined numeric tolerances.

Do not introduce random behavior without an explicit seeded design requirement.

---

## Error Handling

Use controlled error types defined by the owning layer.

Do not leak arbitrary stack traces through the public analysis contract.

Do not hide programmer errors behind generic success output.

Preserve existing error codes and public error contracts unless the active task explicitly changes them.

---

## Public Contract Stability

Do not change the public analysis JSON shape unless the task explicitly requires a contract revision.

Internal benchmark diagnostics may be richer than the normal desktop analysis response.

Do not pollute the normal user-facing analysis response with development-only benchmark data unless specified.

---

## Algorithm Versioning

When a material analysis algorithm change affects cached results, increment or update the appropriate algorithm version/identifier according to the existing specification.

Do not reuse a stale algorithm cache namespace after a materially different pipeline becomes active.

---

## Dependencies

Prefer existing dependencies.

Before adding a package:

1. verify the active task truly requires it;
2. check whether NumPy/SciPy/librosa/current TypeScript stack already supports the need;
3. document why the new dependency is necessary.

Do not add heavyweight ML frameworks for the MVP chord detector.

---

## No Cloud Dependency

The audio-analysis workflow is local.

Do not add cloud inference, remote audio upload, external transcription services, external stem separation services, or SaaS dependencies unless the user explicitly changes product scope.

---

## No ML Expansion Without Approval

Do not introduce PyTorch, TensorFlow, JAX, ONNX runtime, pretrained chord models, or remote AI inference as a shortcut for current DSP tasks unless a future specification explicitly authorizes it.

---

## Code Quality

Prefer small focused functions, typed public boundaries, deterministic logic, centralized constants, clear ownership, testable pure functions, and explicit data models.

Avoid giant orchestration functions, hidden global mutable state, duplicated DSP logic, magic numbers scattered through code, circular imports, and UI/business/DSP mixing.

---

## Constants and Thresholds

Thresholds must be centralized in the owning module or approved configuration layer.

Do not scatter hardcoded thresholds across functions.

Each non-obvious threshold should have a clear name, documented meaning, and deterministic tests around important boundary behavior.

Do not tune a threshold solely until one benchmark song "looks right".

---

## Diagnostics

Development diagnostics are encouraged when they explain algorithm decisions.

Useful diagnostics include why a boundary was accepted/rejected, why a chord candidate won, root evidence, quality evidence, context agreement, novelty strength, decoder-local vs decoder-global choice, cache hit/miss, and analysis source/version.

Diagnostics should help explain behavior, not change production output semantics.

---

## Comments

Comments should explain why a non-obvious algorithmic choice exists, what a threshold means, or what invariant a section protects.

Do not add comments that simply restate obvious code.

Do not place large copies of specs inside source files.

---

## Documentation Updates

When implementation changes an architecture or contract already described in specs, update the relevant specs only when the active task permits or requires documentation changes.

Keep implementation notes concise and evidence-based.

Do not claim acceptance criteria are PASS without test evidence.

---

## Reporting Format

After completing a task, report only information relevant to that task.

A good report includes:

```text
Task
Files created
Files modified
Implementation summary
Tests executed
Test results
Acceptance criteria
Architecture compliance
Deviations
Blockers
Overall status
```

For algorithm work also include relevant BEFORE/AFTER metrics and diagnostic examples.

Do not report a task as PASS merely because code compiles.

---

## Stop Rule

When the user or task says STOP:

STOP.

Do not continue to the next task, unrelated cleanup, additional refactors, or future phase work.

You may report what the next task is, but do not implement it.

---

## Git

Do not assume the workspace is a Git repository.

If Git is unavailable, continue with allowed file operations, report Git-dependent checks as unavailable, and do not treat missing Git metadata as an implementation failure unless Git is required by the active task.

Do not initialize a repository unless explicitly requested.

---

## Absolute Paths

Do not commit developer-specific absolute paths such as:

```text
/Users/...
/home/<developer>/...
```

Use repository-relative paths, environment variables, temporary paths supplied by tests, or existing configuration mechanisms.

---

## Generated / Temporary Files

Do not add generated artifacts, caches, temporary benchmark data, logs, or local audio files to tracked source paths unless explicitly required.

Respect existing `.gitignore` and repository rules.

---

## User Intent

The user is building a practical guitar chord detector, not a research demo whose metrics only look better.

When evaluating an algorithmic change, prioritize:

1. chord identity correctness;
2. musically sensible change boundaries;
3. robustness on real full-mix songs;
4. preservation of genuine fast changes;
5. reproducibility;
6. architectural cleanliness.

Do not optimize vanity metrics at the expense of musical correctness.

---

## Before Every Change

Before editing code, ask:

```text
What exact task am I executing?
Which specs govern it?
Which files am I allowed to change?
What behavior is currently wrong?
What evidence proves it is wrong?
What is the smallest architectural change that addresses the actual cause?
What tests prove improvement without regression?
```

If those questions cannot be answered from the active task and repository specifications, inspect the specs before coding.

---

## Final Rule

Do not guess the architecture. Read it.

Do not guess the task scope. Read it.

Do not guess whether a metric means the detector is better. Test it.

Do not continue to the next task until explicitly instructed.
