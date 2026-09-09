# 05 — Chord Detection Contract

Canonical pitch order: `C C# D D# E F F# G G# A A# B`.

## Templates — Task 2.1 established
Exactly 24 generated templates. Major intervals `0,4,7`; minor `0,3,7`. No N template or extended chords.

## Detector — Task 2.2 established
Single finite 12-bin frame. Cosine similarity against 24 templates. Deterministic tie-break.
Baseline constants: `MIN_FRAME_ENERGY=1e-6`, `MIN_TEMPLATE_SIMILARITY=0.35`.
N for insufficient energy/reliability.
Confidence is deterministic certainty-style `[0,1]` based on best score and best-vs-second margin; not calibrated probability. Silence/near-silence N may be confidence 1.0.

## Smoothing — Task 2.3 established
Centered sliding majority vote; default 23 frames ≈534.06ms at 22050 Hz / hop 512. One output per input. Truncated boundaries. Tie: center label → aggregate confidence → lexical. Confidence = mean winning-label confidence. Sustained N survives.

## Segmentation — Task 2.4 established
Run-length encode then iteratively merge segments under 250ms.
Timing: `start=start_frame*512/22050`; `end=end_frame_exclusive*512/22050`.
Sorted, contiguous, non-overlap.
Final segment confidence = mean contained frame confidence.
Short merge: sole neighbor; same-label neighbors merge all; otherwise stronger aggregate confidence sum → longer duration → lexical tie.
Sustained N remains valid.

These are baseline tuning values. Do not tune to individual tests. Real-song evaluation may later justify a documented change.

## Task 7.1A — Real-Song Accuracy Improvement

Objective: improve robustness on real full-mix songs while preserving existing synthetic behavior, vocabulary, and API contract.

### Locked modern engine order

Shared presentation uses `analysis.chords` as the single playable sequence from
the engine or user edits. Timeline, lyrics, save, and export must not infer keys,
apply fixed progressions, retime chords from lyric lines, or invent confidence.
Lyric markers only intersect the source timestamps with each lyric time window.
Keep `detectedChords` as an original snapshot and `leadSheetChords` as a legacy
compatibility mirror. Preserve stored timelines, including old manual edits;
re-analysis explicitly replaces old generated chord arrangements.

Instrumental/intro presentation must preserve the engine's segment labels,
timestamps, repetitions, no-chord regions, and confidence. Section lyric markers
only locate those segments within the lyric time window. Do not infer an intro
progression, estimate repeat counts from section duration, or apply a second
musical-start offset in shared presentation code. Audio preceding the first
timestamped lyric remains visible from the source timeline.

Beat tracker frames and densified analysis-window boundaries have different
meanings. Keep both: artificial window subdivisions must never increase beat
count or provide a downbeat origin. Harmonic-window construction retains the
harmonic signal as its timing source; switching it directly to full-mix beats
failed moving-bass and real-song regression checks. A future full-mix metrical
tracker must be kept separate from harmonic analysis-window construction.
Experimental bar/phrase diagnostics are opt-in and must not trigger a second
audio analysis during regression evaluation or run on ordinary playback requests.

Lyrics are presentation data only. Adding, generating, editing, syncing, saving,
loading, previewing, or exporting lyrics must not rewrite `analysis.chords`,
change chord labels, retime chord transitions, apply learned phrase patterns, or
select a different playable progression. Lyric views may only intersect the
current chord timeline with lyric time windows for display/export.

For further real-song accuracy work, do not tune the final chord labels first.
The engine must improve the musical foundation in this order:

1. Detect audio preprocessing quality and stable harmonic content.
2. Detect tempo and beat timing.
3. Detect downbeat/bar phase.
4. Detect musical start time so drum/pickup-only intros do not shift the chord grid.
5. Detect phrase/section repetition from harmonic recurrence.
6. Score chord evidence per beat/bar/phrase using harmonic chroma.
7. Decode a playable guitar progression using key as a soft prior.
8. Render/export the playable timeline from the musical grid.

Phrase repetition consistency must be inferred from the audio chord/root
sequence, not lyrics. If repeated root patterns contain weak same-root
major/minor disagreements, the engine may normalize the weaker occurrence to
the duration/confidence-supported quality. Strong sustained quality changes must
remain valid.

Major-key intro/pickup stabilization may align a weak opening substitute or
dominant chord to the tonic when the song has sufficient early tonic support.
This is an audio-derived playable-timeline correction only; it must preserve
strong relative-minor, substitute, or dominant chords and must not read lyrics
or hardcode a song progression.

When the stabilized major-key opening tonic becomes unusually long immediately
before a predominant/subdominant chord, the playable timeline may split only
that opening tonic into tonic then mediant-minor. This correction is limited to
the initial pickup/intro position and must not become a global fixed progression
rule.

For major-key body phrases, a short weak predominant chord between tonic and
mediant-minor may be treated as a passing full-mix artifact and absorbed into
the mediant-minor when the mediant has stable duration support. This correction
must preserve strong predominant chords and must preserve predominant-to-dominant
motions such as `ii -> V`.

The final timeline should be guitar-playable. Raw detailed detector segments may
remain available as diagnostics, but they must not be treated as the preferred
musician-facing chord sheet when they conflict with stronger beat/bar/phrase
evidence.

Do not stack threshold patches on the final output until the timing foundation
has been inspected first. If intro, verse, or post-instrumental chords are late,
early, or repeated incorrectly, inspect musical start/downbeat/bar alignment
before changing chord-template thresholds.

### Non-negotiable constraints

- Keep supported vocabulary exactly: 12 major, 12 minor, `N`.
- Keep JSON contract shape from `06-analysis-api-contract.md`.
- Keep all existing synthetic regression tests and baseline expectations.
- No ML/deep-learning runtime.
- No cloud processing or external stem extraction service.

### Harmonic-focused analysis path

Before chroma scoring for real-song pipeline, use harmonic-focused preprocessing (for example HPSS harmonic component) from approved librosa stack.

### Beat-synchronous harmonic aggregation

After frame-level features are available, aggregate harmonic evidence over beat-synchronous regions.

Requirements:

- region boundaries must come from deterministic local beat analysis;
- aggregated region evidence becomes the primary basis for chord decisions;
- frame-level confidence may still inform within-region weighting.

### Global key as soft prior

Add global key estimation (major/minor key candidates) as a soft prior in scoring.

Requirements:

- key prior may nudge plausibility/confidence;
- key prior must not hard-filter or forbid non-diatonic chords;
- when audio evidence strongly supports a borrowed/non-diatonic chord, that chord must remain selectable.

Example scoring concept (implementation may vary):

```text
adjusted_score = raw_template_score + prior_weight * key_compatibility_bonus
```

with finite prior weight and no impossible-score masking for non-diatonic candidates.

### Flicker reduction policy

Reduce implausible rapid chord changes on real songs by improving upstream evidence quality and region-level decisions.

Do not solve flicker primarily by excessively increasing smoothing window. Any smoothing retuning must be justified by before/after metrics and synthetic-regression stability.

### Baseline comparison requirement

Task 7.1A must include controlled BEFORE/AFTER comparison against current baseline behavior, not blind replacement.

Do not hardcode any specific song key, progression, or hand-tuned constants tied to one test song.

### Deterministic real-song diagnostics requirement

Task 7.1A must provide deterministic benchmark diagnostics for any local analyzed song so baseline vs improved can be compared with objective metrics.

The diagnostic output must include, at minimum, for both baseline and improved pipelines:

1. analysis algorithm/version identifier;
2. detected global key;
3. detected key confidence/score if available;
4. total chord segment count;
5. song duration;
6. mean segment duration;
7. median segment duration;
8. minimum segment duration;
9. maximum segment duration;
10. count and percentage of segments shorter than 250 ms;
11. count and percentage of segments shorter than 500 ms;
12. count and percentage of segments shorter than 1 second;
13. chord occurrence count per chord label;
14. total duration occupied by each chord label;
15. percentage of song duration occupied by each chord label;
16. diatonic vs non-diatonic segment count relative to detected key;
17. diatonic vs non-diatonic total duration relative to detected key;
18. suspicious short non-diatonic segment list with chord, start, end, duration, confidence.

Rules:

- key membership is diagnostic-only information;
- non-diatonic segments must not be auto-deleted or auto-replaced by the diagnostic flow;
- benchmark must not hardcode any specific key/song/path;
- benchmark must not tune detector constants specifically for one benchmark song.

### Task 7.1A Subtask — Context-Aware Short-Segment Correction

Objective: reduce implausible short isolated chord anomalies using local temporal/harmonic context while preserving genuine musical changes.

Before-state reference metrics from latest benchmark snapshot (local song only):

- segment count: 213;
- mean segment duration: about 1.326 s;
- median segment duration: about 1.068 s;
- segments <500 ms: about 21.13%;
- segments <1 s: about 45.54%;
- non-diatonic segments: 13;
- non-diatonic duration: about 7.22 s;
- detected global key: F#m;
- key confidence: about 0.751.

These values are a before-state benchmark reference only, not a target to overfit.

Implementation principles:

- correction must consider previous chord, candidate chord, next chord, duration, confidence, and harmonic plausibility context;
- short duration alone must not auto-remove a chord;
- non-diatonic status alone must not auto-remove a chord;
- key context remains soft and must not hard-forbid borrowed/secondary-function chords;
- do not assume chords such as C# major are automatically invalid in F# minor contexts;
- do not increase global smoothing window merely to hide errors;
- no hardcoded song/progression/key constants.

Scope constraints:

- preserve chord vocabulary (12 major + 12 minor + N);
- preserve existing public analysis contract shape;
- no desktop/electron/ui changes;
- no ML/deep-learning.

### Task 7.1A Subtask — Musical-Time Harmonic Persistence

Objective: reduce false chord switching by enforcing musical-time harmonic persistence, while preserving genuine sustained chord changes and valid short strong chords.

Core requirements:

- chord switching decisions must be made on beat-aware musical-time observations, not isolated short-frame wins;
- transition decision must use asymmetric evidence:
	- KEEP current chord threshold;
	- SWITCH to candidate chord threshold (stronger than KEEP threshold);
- candidate chord must persist across neighboring observations (or equivalent deterministic persistence rule) before replacing current chord, except when evidence is very strong;
- short duration alone must not trigger correction;
- non-diatonic status alone must not trigger correction;
- key remains soft prior only.

Guardrails:

- do not hardcode BPM;
- do not hardcode fixed chord count per bar;
- do not assume chord changes only on bar boundaries;
- do not hardcode specific song/key/progression/genre;
- secondary dominants and valid borrowed non-diatonic chords must remain possible outcomes.

### Experimental Subtask — Optional Essentia Backend Spike

Objective: allow a direct third-party Python chord detector to be tested against
the current built-in engine without replacing the production default.

Rules:

- Essentia backend selection must be explicit, for example CLI `--backend essentia`
  or an equivalent development environment override.
- Default analysis remains the repository's built-in deterministic engine.
- The public JSON contract shape remains unchanged.
- Output chord labels must be normalized to the MVP vocabulary: 12 major,
  12 minor, and `N`.
- Missing optional dependency must return a controlled engine error.
- Essentia output is an experiment/benchmark source, not proof of accuracy until
  compared against local ground truth or user-reviewed real-song results.

Benchmark extension requirements:

- total segment count;
- chord transitions per minute;
- mean segment duration;
- median segment duration;
- segment rate <500ms;
- segment rate <1s;
- optional estimated tempo;
- transitions per beat if beat estimation is reliable.

### Task 7.1A Subtask — Transition Acceptance Refinement

Objective: refine transition acceptance so sustained genuine harmonic changes remain detectable while weak or ambiguous accepted switches are reduced, especially for root-preserving major/minor quality changes.

Scope focus:

- only transition acceptance policy in the musical-time persistence stage;
- do not add global smoothing-window inflation or unrelated segment-count hacks;
- preserve existing beat-aware aggregation and persistence structure.

Core acceptance rules:

1. Ordinary override must not accept a candidate when instantaneous switch evidence is weaker than keep evidence.
2. If an override may accept while switch evidence is lower than keep evidence, it must require explicit stronger multi-beat/contextual support and must emit a diagnosable reason.
3. Root-preserving quality switches must require stronger acceptance evidence than ordinary sustained continuation:
	- E <-> Em
	- A <-> Am
	- C# <-> C#m
	- F# <-> F#m
4. Add harmonic-change evidence comparing the current harmonic profile against candidate profile across musical-time observations.
5. Do not ban legitimate major/minor quality changes; genuine sustained quality transitions must remain selectable when evidence is strong.

Preserve all of the following:

- key as soft prior only (no hard filtering);
- secondary dominants and other valid non-diatonic outcomes;
- N/silence behavior;
- public analysis API contract and output shape;
- previous synthetic behavior and tests.

Benchmark diagnostic extension requirements:

- accepted transition reason counts;
- root-preserving major/minor accepted-switch count;
- accepted override count where switchScore <= keepScore;
- before/after transition count and transitions per minute.

### Task 7.1A Subtask — Root-Aware Harmonic Scoring

Objective: improve chord identity accuracy under full-mix ambiguity by separating root evidence from major/minor quality evidence, while preserving the current MVP output vocabulary.

Vocabulary constraints:

- output tetap hanya 12 major + 12 minor + N;
- tidak menambah 7th/maj7/m7/sus/dim/aug/slash-chord output.

Core requirements:

1. Root evidence dan quality evidence harus dievaluasi terpisah.
2. Estimasi root harus berbasis harmonic evidence regional musical-time, bukan hanya cosine similarity template penuh.
3. Root evidence harus menggabungkan sinyal deterministik yang relevan:
	- harmonic chroma,
	- pitch-class energy,
	- root/fifth support,
	- bass/low-frequency chroma support,
	- beat-synchronous aggregation.
4. Bass evidence hanya supporting signal, tidak boleh diperlakukan sebagai hard root rule.
5. Setelah root kandidat dipilih, kualitas major/minor dievaluasi via discriminating tones:
	- major third,
	- minor third,
	- root/fifth support.
6. Jika third evidence lemah, quality boleh ambigu; jangan memaksa keputusan berkepastian tinggi dari evidence lemah.
7. Integrasikan root-aware scoring dengan key soft prior, beat-aware aggregation, persistence, dan transition acceptance refinement.
8. Template scorer existing harus tetap dipakai sebagai baseline evidence dan dikombinasikan deterministik dengan root-aware evidence.

Guardrails:

- tidak boleh hardcode lagu/key/progression tertentu;
- tidak boleh tuning threshold khusus satu benchmark song;
- tidak boleh menambah smoothing global demi menurunkan segment count;
- tidak boleh mengubah API publik atau behavior N/silence.

Benchmark diagnostic extension requirements:

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

Success criteria bersifat quality-oriented:

- synthetic root/quality ground-truth tetap benar;
- unsupported identity changes karena kontaminasi melody/bass berkurang;
- genuine root dan genuine quality changes tetap terdeteksi;
- transition-density improvement sebelumnya tetap terjaga;
- tidak ada public API regression.

### Task 7.1A Subtask — Harmonic Change-Point Segmentation and Global Chord Decoding

Objective: improve perceived harmonic progression quality by detecting stable harmonic-change boundaries first, then decoding the chord sequence globally.

Core architecture:

```text
harmonic features
-> beat-synchronous representation
-> harmonic change-point / novelty detection
-> stable harmonic regions
-> region-level chord evidence scoring
-> deterministic global sequence decoding
```

Boundary detection requirements:

- operate on harmonic-focused beat-synchronous features;
- detect sustained harmonic changes, not short melodic contamination;
- suppress novelty from passing tones or vocal melody;
- no fixed chord-count-per-bar assumption;
- no bar-boundary-only assumption;
- no hardcoded BPM;
- deterministic output.

Region classification requirements:

- region evidence must be aggregated across the full stable region;
- use existing root-aware evidence separation;
- use major/minor quality evidence;
- use template scoring as supporting evidence;
- key remains a soft prior only.

Global decoding requirements:

- use deterministic global optimization (prefer Viterbi/HMM-style DP or equivalent);
- include region evidence, stability preference, soft transition-cost model, key soft prior, and harmonic relationship signals;
- transition cost must be soft only;
- genuine rapid changes must remain possible when evidence is strong;
- non-diatonic results remain possible;
- must not encode one expected progression or one specific song.

Replacement discipline:

- do not stack previous persistence/transition heuristics blindly after the global decoder;
- if a previous heuristic is superseded by boundary-first + global decoding, bypass/remove the redundant stage in the improved path and document it.

### Task 7.1A Subtask — Beat-Aware Harmonic Boundary Consolidation

Objective: reduce locally clustered accepted harmonic boundaries by consolidating nearby novelty peaks before final region classification/global decoding.

Stage location (mandatory):

```text
novelty/change-point candidates
-> beat-aware boundary consolidation
-> stable harmonic regions
```

Core requirements:

- identify boundary clusters that occur unusually close in musical time;
- use beat intervals as primary proximity metric when beat tracking is reliable;
- do not enforce fixed one-chord-per-N-beat rule;
- decide cluster merges using deterministic evidence:
	- novelty strength and prominence,
	- beat distance,
	- harmonic similarity pre/mid/post,
	- intermediate-region duration,
	- intermediate-region chord/root evidence,
	- confidence/evidence margin;
- collapse only when intermediate region is both short and harmonically weak versus surrounding context;
- preserve short intermediate regions that have strong independent harmonic evidence;
- preserve genuine rapid changes when evidence is strong.

Guardrails:

- do not hardcode any song/key/timestamp/style;
- do not increase global novelty threshold as blanket fix;
- do not add global smoothing as workaround;
- do not add post-label correction heuristics as primary strategy.

### Task 7.1A Subtask — Multi-Resolution Harmonic Context Representation

Objective: improve harmonic-change evidence quality so melody, bass movement, voicing change, and passing tones are less likely to become false harmonic boundaries.

Mandatory stage location:

```text
harmonic features
-> beat-synchronous representation
-> multi-resolution contextual harmonic representation
-> harmonic novelty/change-point detection
-> boundary consolidation
-> harmonic regions
-> root-aware region classification
-> global decoding
```

Core requirements:

- boundary acceptance must require contextual harmonic separation across multiple neighboring beats, not adjacent-beat difference only;
- keep at least two context scales:
	- short context: preserve real relatively fast harmonic changes;
	- medium context: ensure apparent change persists beyond transient motion;
- use deterministic robust context profiles from approved NumPy/librosa stack;
- compare several beats before vs after candidate boundary for each context scale;
- compute deterministic context support metrics (for example cosine/correlation distance between context profiles);
- produce final boundary confidence from local novelty + multi-resolution context support.

Diagnostics requirements (candidate level):

- local novelty;
- short-context distance;
- medium-context distance;
- context agreement score;
- persistence score;
- short/medium agreement boolean;
- final boundary confidence;
- accepted/rejected decision and reason.

Expected behavior:

- local novelty high but medium-context support low -> usually reject as local-only boundary;
- genuine sustained chord change -> accepted due to stable short+medium contextual separation.

Guardrails:

- no hardcoded song/timestamps/key/progression/style;
- no one-chord-per-bar rule;
- no target-segment-count optimization;
- no ML/deep learning;
- no chord-vocabulary expansion;
- no Electron/UI changes;
- no public API contract break;
- do not solve by adding more post-boundary merging/smoothing/persistence/label-repair layers.
