"""Output-only confidence calibration for final chord segments."""

from __future__ import annotations

import numpy as np

from chord_engine.segmentation import ChordSegment

OUTPUT_CONFIDENCE_RAW_FLOOR = 0.12
OUTPUT_CONFIDENCE_RAW_CEILING = 0.28
OUTPUT_CONFIDENCE_MIN = 0.52
OUTPUT_CONFIDENCE_RANGE = 0.32
OUTPUT_CONFIDENCE_DURATION_BONUS_MAX = 0.10
OUTPUT_CONFIDENCE_STABLE_DURATION_SECONDS = 6.0
OUTPUT_CONFIDENCE_SHORT_SEGMENT_SECONDS = 0.65
OUTPUT_CONFIDENCE_SHORT_SEGMENT_PENALTY = 0.08
OUTPUT_CONFIDENCE_MAX = 0.94
OUTPUT_CONFIDENCE_N_MAX = 0.72


def calibrate_output_confidence(segments: list[ChordSegment]) -> list[ChordSegment]:
    """Map internal DSP evidence scores to a stable user-facing confidence scale.

    The detector uses conservative raw margins internally so low numeric values are
    still useful for sequence decisions. Public confidence should communicate how
    actionable the final segment is without feeding back into refinement logic.
    """
    calibrated: list[ChordSegment] = []
    raw_span = max(1e-6, OUTPUT_CONFIDENCE_RAW_CEILING - OUTPUT_CONFIDENCE_RAW_FLOOR)
    for segment in segments:
        raw_confidence = float(segment.confidence) if np.isfinite(segment.confidence) else 0.0
        raw_position = float(np.clip((raw_confidence - OUTPUT_CONFIDENCE_RAW_FLOOR) / raw_span, 0.0, 1.0))
        duration = max(0.0, float(segment.end - segment.start))
        stability_position = float(
            np.clip(
                (duration - OUTPUT_CONFIDENCE_SHORT_SEGMENT_SECONDS)
                / max(1e-6, OUTPUT_CONFIDENCE_STABLE_DURATION_SECONDS - OUTPUT_CONFIDENCE_SHORT_SEGMENT_SECONDS),
                0.0,
                1.0,
            )
        )
        short_penalty = OUTPUT_CONFIDENCE_SHORT_SEGMENT_PENALTY if duration < OUTPUT_CONFIDENCE_SHORT_SEGMENT_SECONDS else 0.0
        output_confidence = (
            OUTPUT_CONFIDENCE_MIN
            + OUTPUT_CONFIDENCE_RANGE * raw_position
            + OUTPUT_CONFIDENCE_DURATION_BONUS_MAX * stability_position
            - short_penalty
        )
        if segment.chord == "N":
            output_confidence = min(output_confidence, OUTPUT_CONFIDENCE_N_MAX)
        output_confidence = float(np.clip(output_confidence, 0.0, OUTPUT_CONFIDENCE_MAX))
        calibrated.append(
            ChordSegment(
                start=segment.start,
                end=segment.end,
                chord=segment.chord,
                confidence=output_confidence,
            )
        )
    return calibrated
