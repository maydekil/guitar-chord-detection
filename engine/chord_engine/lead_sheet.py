"""Audio-based lead-sheet timeline arrangement.

This module turns detailed detector segments into phrase-level chords that are
more suitable for guitar playback. It is intentionally generic: no song title,
known timestamp, or fixed progression is encoded here.
"""

from __future__ import annotations

import numpy as np

from chord_engine.detector import KeyEstimate
from chord_engine.music_theory import _chord_root_pc, _is_diatonic_chord, _root_name_from_pc
from chord_engine.segment_utils import _merge_adjacent_same_chord_segments
from chord_engine.segmentation import ChordSegment

PHRASE_TARGET_SECONDS = 5.6
PHRASE_MIN_SECONDS = 3.2
PHRASE_SNAP_LOOKAHEAD_SECONDS = 1.2
MAX_CHORDS_PER_PHRASE = 3
MIN_MARKER_OVERLAP_SECONDS = 1.35
MIN_MARKER_OVERLAP_RATIO = 0.24
LEADSHEET_CONFIDENCE_FLOOR = 0.72
LEADSHEET_CONFIDENCE_CEILING = 0.94


def arrange_audio_lead_sheet(
	segments: list[ChordSegment],
	*,
	detected_key: KeyEstimate | None,
	duration_seconds: float,
) -> list[ChordSegment]:
	if detected_key is None or detected_key.mode != "major" or not segments or duration_seconds <= 0.0:
		return segments

	windows = _phrase_windows(segments, duration_seconds)
	arranged: list[ChordSegment] = []
	for start_time, end_time in windows:
		markers = _phrase_markers(segments, start_time, end_time)
		markers = _normalize_markers_to_key(markers, detected_key)
		markers = _limit_marker_density(markers)
		for idx, marker in enumerate(markers):
			next_left = markers[idx + 1][1] if idx + 1 < len(markers) else 1.0
			seg_start = start_time + ((end_time - start_time) * marker[1])
			seg_end = start_time + ((end_time - start_time) * next_left)
			if seg_end <= seg_start:
				continue
			arranged.append(
				ChordSegment(
					start=float(seg_start),
					end=float(seg_end),
					chord=marker[0],
					confidence=_phrase_confidence(segments, seg_start, seg_end),
				)
			)
	return _merge_adjacent_same_chord_segments(arranged) if arranged else segments


def _phrase_windows(segments: list[ChordSegment], duration_seconds: float) -> list[tuple[float, float]]:
	windows: list[tuple[float, float]] = []
	start = 0.0
	while start < duration_seconds:
		target = min(duration_seconds, start + PHRASE_TARGET_SECONDS)
		end = _nearest_boundary(segments, target, start + PHRASE_MIN_SECONDS, duration_seconds)
		end = max(start + 0.25, end)
		windows.append((start, end))
		start = end
	return windows


def _nearest_boundary(segments: list[ChordSegment], target: float, minimum: float, duration_seconds: float) -> float:
	maximum = min(duration_seconds, target + PHRASE_SNAP_LOOKAHEAD_SECONDS)
	candidates = [seg.start for seg in segments if minimum <= seg.start <= maximum]
	if not candidates:
		return min(duration_seconds, target)
	return min(candidates, key=lambda value: abs(value - target))


def _phrase_markers(segments: list[ChordSegment], start_time: float, end_time: float) -> list[tuple[str, float]]:
	duration = max(0.25, end_time - start_time)
	overlaps: list[tuple[str, float, float]] = []
	for segment in segments:
		overlap_start = max(start_time, segment.start)
		overlap_end = min(end_time, segment.end)
		overlap = max(0.0, overlap_end - overlap_start)
		if overlap <= 0.0:
			continue
		if overlap >= MIN_MARKER_OVERLAP_SECONDS or overlap / duration >= MIN_MARKER_OVERLAP_RATIO:
			overlaps.append((segment.chord, (overlap_start - start_time) / duration, overlap))
	if not overlaps:
		return []
	markers: list[tuple[str, float]] = []
	for chord, left, _ in overlaps:
		if markers and markers[-1][0] == chord:
			continue
		markers.append((chord, float(np.clip(left, 0.0, 0.92))))
	return markers


def _normalize_markers_to_key(markers: list[tuple[str, float]], detected_key: KeyEstimate) -> list[tuple[str, float]]:
	out: list[tuple[str, float]] = []
	for chord, left in markers:
		normalized = chord if _is_diatonic_chord(chord, detected_key) else _nearest_key_chord(chord, detected_key)
		if normalized is None:
			continue
		if out and out[-1][0] == normalized:
			continue
		out.append((normalized, left))
	return out


def _nearest_key_chord(chord: str, detected_key: KeyEstimate) -> str | None:
	root_pc = _chord_root_pc(chord)
	if root_pc is None:
		return None
	best_degree = min((0, 2, 4, 5, 7, 9), key=lambda degree: min((root_pc - (detected_key.tonic_pc + degree)) % 12, ((detected_key.tonic_pc + degree) - root_pc) % 12))
	suffix = "m" if best_degree in {2, 4, 9} else ""
	return f"{_root_name_from_pc(detected_key.tonic_pc + best_degree)}{suffix}"


def _limit_marker_density(markers: list[tuple[str, float]]) -> list[tuple[str, float]]:
	if len(markers) <= MAX_CHORDS_PER_PHRASE:
		return markers
	first = markers[0]
	middle = markers[len(markers) // 2]
	last = markers[-1]
	out: list[tuple[str, float]] = []
	for marker in (first, middle, last):
		if out and out[-1][0] == marker[0]:
			continue
		out.append(marker)
	return out


def _phrase_confidence(segments: list[ChordSegment], start_time: float, end_time: float) -> float:
	total = 0.0
	weighted = 0.0
	for segment in segments:
		overlap = max(0.0, min(end_time, segment.end) - max(start_time, segment.start))
		if overlap <= 0.0:
			continue
		total += overlap
		weighted += overlap * segment.confidence
	if total <= 0.0:
		return 0.88
	return float(np.clip(weighted / total, LEADSHEET_CONFIDENCE_FLOOR, LEADSHEET_CONFIDENCE_CEILING))
