# Timing foundation evaluation, 2026-09-08

Scope: separate tracker beats from analysis-window subdivisions and evaluate
existing correction stages within the user-authorized limit of five iterations.

## Accuracy evidence

The reference is the existing local regression manifest, with 16 chord anchors
per song. These are sparse reference checks, not verified full-song ground truth
or a measurement of guitar performance quality. Thresholds and anchors were not
changed. No additional manual listening is claimed.

| Configuration | Album Lama | Foto Kita Berdua | Decision |
| --- | --- | --- | --- |
| Starting v27 result | 13/16 | 11/16 | Reference |
| 1. Full-mix tracker for harmonic windows | 6/16 | 5/16 | Reject; moving-bass test also failed |
| 2. Restore harmonic timing, preserve actual beats separately | 13/16 | 11/16 | Retain foundation fixes |
| 3. Disable weak-fragment cleanup | 13/16 | 9/16 | Reject |
| 4. Disable short-segment context correction | 13/16 | 11/16 | No demonstrated accuracy gain; retain existing stage |
| 5. Disable phrase-fragment absorption | 10/16 | 7/16 | Reject |

Iterations 3-5 used process-local function substitution with identical decoded
audio and extracted features reused within each song. Those substitutions were
not applied to production source. They isolate the contribution of each stage;
they do not establish the effect of removing multiple stages together.

## Retained implementation

- `BeatTiming.beat_frames` preserves actual tracker positions. Densified
  `boundaries` remain harmonic analysis windows, not additional beats.
- Beat count and reliability use actual beats. Bar origins use the shared
  tracker-frame conversion, including a genuine beat at frame zero.
- Empty or failed tracking supplies analysis windows but no metrical beats.
- The one-frame subdivision case consistently returns a `BeatTiming` object.
- `analyze_audio_run` exposes result and diagnostics from one analysis pass.
  Regression evaluation previously executed production analysis plus a separate
  baseline/improved comparison, for three pipeline passes per song.
- Ordinary `analyze_audio` omits experimental bar/phrase diagnostics.
- The retained full-song evaluation measured 14.257 seconds for Album Lama and
  13.333 seconds for Foto Kita Berdua on this local run. There is no comparable
  before-time measurement, so no wall-clock speedup factor is asserted.
- Cache algorithm identifiers advance to v28. The public result shape is unchanged.

## Remaining limitation

Some reference mismatches originate before correction, while others are introduced
by correction stages. Removing an entire stage does not consistently fix them.
Experimental phrase-pattern inference remains diagnostic-only; it is not promoted
to the musician-facing output. This work does not complete a new bar-aware arranger
or demonstrate improved chord identity accuracy. Tuning stopped at five iterations.

## Final validation

- Recursive TypeScript typecheck: passed for shared, API, and desktop.
- Complete engine suite: 164 passed, 1 skipped, 57 warnings in 28.72 seconds.
- Retained configuration regression evaluation: passed existing manifest gates;
  exact anchor matches remain 13/16 and 11/16. Passing those gates is not evidence
  that the target accuracy has been reached.
