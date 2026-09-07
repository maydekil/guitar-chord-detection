"""Reusable chord-segment utilities for analysis and refinement stages."""

from __future__ import annotations

import numpy as np

from chord_engine.analysis_config import CONTEXT_NEIGHBOR_MIN_DURATION_SECONDS, END_TOLERANCE_SECONDS
from chord_engine.detector import KeyEstimate
from chord_engine.music_theory import _harmonic_plausibility
from chord_engine.segmentation import ChordSegment


def _clamp_final_segment_end(segments: list[ChordSegment], *, source_duration: float) -> list[ChordSegment]:
    if not segments:
        return segments

    last = segments[-1]
    if last.end <= source_duration + END_TOLERANCE_SECONDS:
        return segments

    new_end = max(last.start + 1e-9, source_duration)
    segments[-1] = ChordSegment(
        start=last.start,
        end=new_end,
        chord=last.chord,
        confidence=last.confidence,
    )
    return segments


def _merge_adjacent_same_chord_segments(segments: list[ChordSegment]) -> list[ChordSegment]:
    if not segments:
        return []

    merged: list[ChordSegment] = []
    for segment in segments:
        if merged and merged[-1].chord == segment.chord:
            previous = merged[-1]
            merged[-1] = ChordSegment(
                start=previous.start,
                end=segment.end,
                chord=previous.chord,
                confidence=_weighted_confidence(previous, segment),
            )
        else:
            merged.append(segment)
    return merged


def _weighted_confidence(left: ChordSegment, right: ChordSegment) -> float:
    left_duration = _segment_duration(left)
    right_duration = _segment_duration(right)
    total = left_duration + right_duration
    if total <= 0.0:
        return float(np.clip(max(left.confidence, right.confidence), 0.0, 1.0))
    return float(np.clip(((left.confidence * left_duration) + (right.confidence * right_duration)) / total, 0.0, 1.0))


def _chord_duration_map(segments: list[ChordSegment]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for segment in segments:
        durations[segment.chord] = durations.get(segment.chord, 0.0) + _segment_duration(segment)
    return durations


def _segment_duration(segment: ChordSegment) -> float:
    return float(max(0.0, segment.end - segment.start))


def _neighbor_support(segment: ChordSegment, detected_key: KeyEstimate | None) -> float:
    duration_norm = min(1.0, _segment_duration(segment) / CONTEXT_NEIGHBOR_MIN_DURATION_SECONDS)
    plausibility = _harmonic_plausibility(segment.chord, detected_key)
    support = 0.52 * segment.confidence + 0.28 * duration_norm + 0.20 * plausibility
    return float(np.clip(support, 0.0, 1.0))


def _consensus_chord_with_ratio(chords: list[str], *, start: int, end: int) -> tuple[str | None, float]:
    left = max(0, int(start))
    right = min(len(chords), int(end))
    if right <= left:
        return None, 0.0

    counts: dict[str, int] = {}
    for chord in chords[left:right]:
        counts[chord] = counts.get(chord, 0) + 1
    if not counts:
        return None, 0.0

    label, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    window_len = max(1, right - left)
    ratio = float(count / window_len)
    return label, ratio
