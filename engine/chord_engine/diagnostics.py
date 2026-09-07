"""Benchmark and diagnostic summary helpers for chord analysis runs."""

from __future__ import annotations

from statistics import mean, median

from chord_engine.analysis_models import AnalysisResult
from chord_engine.detector import KeyEstimate
from chord_engine.features import BeatTiming
from chord_engine.music_theory import _is_diatonic_chord, _parse_chord_quality
from chord_engine.segmentation import ChordSegment

SHORT_SEGMENT_SECONDS = 0.25
SUSPICIOUS_SHORT_NON_DIATONIC_SECONDS = 1.0
def summarize_analysis(
    result: AnalysisResult,
    detected_key: KeyEstimate | None,
    beat_timing: BeatTiming | None,
) -> dict[str, object]:
    segments = result.analysis.chords
    song_duration = float(max(0.0, result.source.duration))
    durations = [max(0.0, seg.end - seg.start) for seg in segments]
    if durations:
        mean_duration = float(mean(durations))
        median_duration = float(median(durations))
        min_duration = float(min(durations))
        max_duration = float(max(durations))
        short_rate = float(sum(1 for value in durations if value < SHORT_SEGMENT_SECONDS) / len(durations))
    else:
        mean_duration = 0.0
        median_duration = 0.0
        min_duration = 0.0
        max_duration = 0.0
        short_rate = 0.0

    short_buckets = {
        "lt250ms": _short_bucket(durations, threshold_seconds=0.25),
        "lt500ms": _short_bucket(durations, threshold_seconds=0.50),
        "lt1s": _short_bucket(durations, threshold_seconds=1.0),
    }

    family_counts = {"major": 0, "minor": 0, "N": 0}
    chord_occurrence_count: dict[str, int] = {}
    chord_duration_seconds: dict[str, float] = {}
    for segment in segments:
        label = segment.chord
        duration = max(0.0, segment.end - segment.start)
        chord_occurrence_count[label] = chord_occurrence_count.get(label, 0) + 1
        chord_duration_seconds[label] = chord_duration_seconds.get(label, 0.0) + duration

        if segment.chord == "N":
            family_counts["N"] += 1
        elif segment.chord.endswith("m"):
            family_counts["minor"] += 1
        else:
            family_counts["major"] += 1

    denominator = song_duration if song_duration > 0.0 else 1.0
    chord_duration_percentage = {
        label: float((duration / denominator) * 100.0)
        for label, duration in chord_duration_seconds.items()
    }

    chord_breakdown = [
        {
            "chord": label,
            "count": chord_occurrence_count[label],
            "durationSeconds": float(chord_duration_seconds[label]),
            "durationPercentageOfSong": float(chord_duration_percentage[label]),
        }
        for label in sorted(chord_occurrence_count)
    ]

    diatonic_stats = _diatonicity_stats(segments, detected_key)
    suspicious_short_non_diatonic = _suspicious_short_non_diatonic_segments(segments, detected_key)

    global_key = {
        "label": detected_key.label if detected_key is not None else None,
        "confidence": float(detected_key.confidence) if detected_key is not None else None,
    }
    transition_count = _count_chord_transitions(segments)
    root_change_count = _count_root_changes(segments)
    quality_change_count = _count_quality_changes(segments)
    ambiguous_quality_decision_count = 0
    transitions_per_minute = float(0.0 if song_duration <= 0.0 else transition_count / (song_duration / 60.0))
    transitions_per_beat = None
    estimated_tempo = None
    beat_reliable = False
    beat_count = None
    if beat_timing is not None:
        estimated_tempo = beat_timing.tempo_bpm
        beat_reliable = beat_timing.is_reliable
        beat_count = beat_timing.beat_count
        if beat_timing.is_reliable and beat_timing.beat_count > 0:
            transitions_per_beat = float(transition_count / beat_timing.beat_count)

    return {
        "algorithm": result.analysis.algorithm,
        "contractVersion": result.version,
        "segmentCount": len(segments),
        "songDuration": song_duration,
        "meanSegmentDuration": mean_duration,
        "medianSegmentDuration": median_duration,
        "minSegmentDuration": min_duration,
        "maxSegmentDuration": max_duration,
        "excessiveShortSegmentRate": short_rate,
        "shortSegments": short_buckets,
        "transitionCount": transition_count,
        "rootChangeCount": root_change_count,
        "qualityChangeCount": quality_change_count,
        "ambiguousQualityDecisionCount": ambiguous_quality_decision_count,
        "transitionsPerMinute": transitions_per_minute,
        "estimatedTempoBpm": estimated_tempo,
        "beatCount": beat_count,
        "beatReliable": beat_reliable,
        "transitionsPerBeat": transitions_per_beat,
        "globalKey": global_key["label"],
        "globalKeyConfidence": global_key["confidence"],
        "chordFamilyDistribution": family_counts,
        "chordOccurrenceCount": chord_occurrence_count,
        "chordDurationSeconds": chord_duration_seconds,
        "chordDurationPercentageOfSong": chord_duration_percentage,
        "chordBreakdown": chord_breakdown,
        "diatonicity": diatonic_stats,
        "suspiciousShortNonDiatonicSegments": suspicious_short_non_diatonic,
    }


def _count_root_changes(segments: list[ChordSegment]) -> int:
    if len(segments) <= 1:
        return 0
    count = 0
    for prev, current in zip(segments, segments[1:], strict=False):
        prev_root, _ = _parse_chord_quality(prev.chord)
        curr_root, _ = _parse_chord_quality(current.chord)
        if prev_root is None or curr_root is None:
            continue
        if prev_root != curr_root:
            count += 1
    return count


def _count_quality_changes(segments: list[ChordSegment]) -> int:
    if len(segments) <= 1:
        return 0
    count = 0
    for prev, current in zip(segments, segments[1:], strict=False):
        prev_root, prev_quality = _parse_chord_quality(prev.chord)
        curr_root, curr_quality = _parse_chord_quality(current.chord)
        if prev_root is None or curr_root is None:
            continue
        if prev_root == curr_root and prev_quality != curr_quality:
            count += 1
    return count


def _count_chord_transitions(segments: list[ChordSegment]) -> int:
    if len(segments) <= 1:
        return 0

    count = 0
    for prev, current in zip(segments, segments[1:], strict=False):
        if prev.chord != current.chord:
            count += 1
    return count


def _short_bucket(durations: list[float], *, threshold_seconds: float) -> dict[str, float]:
    if not durations:
        return {"count": 0, "percentage": 0.0}

    count = int(sum(1 for duration in durations if duration < threshold_seconds))
    percentage = float((count / len(durations)) * 100.0)
    return {"count": count, "percentage": percentage}


def _diatonicity_stats(segments: list[ChordSegment], detected_key: KeyEstimate | None) -> dict[str, object]:
    diatonic_count = 0
    non_diatonic_count = 0
    diatonic_duration = 0.0
    non_diatonic_duration = 0.0
    no_chord_count = 0
    no_chord_duration = 0.0

    for seg in segments:
        duration = max(0.0, seg.end - seg.start)
        if seg.chord == "N":
            no_chord_count += 1
            no_chord_duration += duration
            continue

        is_diatonic = _is_diatonic_chord(seg.chord, detected_key)
        if is_diatonic:
            diatonic_count += 1
            diatonic_duration += duration
        else:
            non_diatonic_count += 1
            non_diatonic_duration += duration

    return {
        "referenceKey": detected_key.label if detected_key is not None else None,
        "diatonicSegmentCount": diatonic_count,
        "nonDiatonicSegmentCount": non_diatonic_count,
        "diatonicDuration": float(diatonic_duration),
        "nonDiatonicDuration": float(non_diatonic_duration),
        "noChordSegmentCount": no_chord_count,
        "noChordDuration": float(no_chord_duration),
    }


def _suspicious_short_non_diatonic_segments(
    segments: list[ChordSegment],
    detected_key: KeyEstimate | None,
) -> list[dict[str, float | str]]:
    suspicious: list[dict[str, float | str]] = []
    for seg in segments:
        if seg.chord == "N":
            continue

        duration = max(0.0, seg.end - seg.start)
        if duration >= SUSPICIOUS_SHORT_NON_DIATONIC_SECONDS:
            continue

        if _is_diatonic_chord(seg.chord, detected_key):
            continue

        suspicious.append(
            {
                "chord": seg.chord,
                "start": float(seg.start),
                "end": float(seg.end),
                "duration": float(duration),
                "confidence": float(seg.confidence),
            }
        )

    return suspicious

