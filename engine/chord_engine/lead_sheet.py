"""Audio-based lead-sheet timeline arrangement.

This module turns detailed detector segments into phrase-level chords that are
more suitable for guitar playback. It is intentionally generic: no song title,
known timestamp, or fixed progression is encoded here.
"""

from __future__ import annotations

import numpy as np

from chord_engine.detector import KeyEstimate
from chord_engine.features import BeatTiming, beat_times_seconds
from chord_engine.music_theory import _chord_root_pc, _is_diatonic_chord, _root_name_from_pc
from chord_engine.segment_utils import _merge_adjacent_same_chord_segments
from chord_engine.segmentation import ChordSegment

BAR_BEATS = 4
MIN_BAR_COUNT = 2
DOWNBEAT_ALIGNMENT_WINDOW_SECONDS = 0.42
DOWNBEAT_PARTIAL_BAR_PENALTY = 0.45
PHRASE_TARGET_SECONDS = 5.6
PHRASE_MIN_SECONDS = 3.2
PHRASE_SNAP_LOOKAHEAD_SECONDS = 1.2
MAX_CHORDS_PER_PHRASE = 3
MAX_CHORDS_PER_BAR = 2
MIN_MARKER_OVERLAP_SECONDS = 1.35
MIN_MARKER_OVERLAP_RATIO = 0.24
LEADSHEET_CONFIDENCE_FLOOR = 0.72
LEADSHEET_CONFIDENCE_CEILING = 0.94


def arrange_audio_lead_sheet(
	segments: list[ChordSegment],
	*,
	detected_key: KeyEstimate | None,
	duration_seconds: float,
	beat_timing: BeatTiming | None = None,
	hop_length: int | None = None,
	sample_rate: int | None = None,
) -> list[ChordSegment]:
	if detected_key is None or detected_key.mode != "major" or not segments or duration_seconds <= 0.0:
		return segments

	windows = _bar_windows(
		segments,
		duration_seconds,
		beat_timing=beat_timing,
		hop_length=hop_length,
		sample_rate=sample_rate,
	)
	max_chords = MAX_CHORDS_PER_BAR if windows.source == "bar" else MAX_CHORDS_PER_PHRASE
	arranged: list[ChordSegment] = []
	for start_time, end_time in windows.values:
		markers = _phrase_markers(segments, start_time, end_time)
		markers = _normalize_markers_to_key(markers, detected_key)
		markers = _limit_marker_density(markers, max_chords=max_chords)
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


def build_bar_windows_for_segments(
	segments: list[ChordSegment],
	duration_seconds: float,
	*,
	beat_timing: BeatTiming | None,
	hop_length: int | None,
	sample_rate: int | None,
) -> list[tuple[float, float]]:
	windows = _bar_windows(
		segments,
		duration_seconds,
		beat_timing=beat_timing,
		hop_length=hop_length,
		sample_rate=sample_rate,
	)
	return windows.values if windows.source == "bar" else []


class LeadSheetWindows:
	def __init__(self, values: list[tuple[float, float]], source: str) -> None:
		self.values = values
		self.source = source


def _bar_windows(
	segments: list[ChordSegment],
	duration_seconds: float,
	*,
	beat_timing: BeatTiming | None,
	hop_length: int | None,
	sample_rate: int | None,
) -> LeadSheetWindows:
	beat_interval = _beat_interval_seconds(beat_timing)
	if beat_interval is None:
		return LeadSheetWindows(_phrase_windows(segments, duration_seconds), "phrase")

	first_beat = _first_beat_seconds(
		beat_timing=beat_timing,
		hop_length=hop_length,
		sample_rate=sample_rate,
	)
	bar_length = beat_interval * BAR_BEATS
	downbeat = _infer_downbeat_phase(segments, first_beat=first_beat, beat_interval=beat_interval)
	windows = _build_bar_windows(downbeat, bar_length=bar_length, duration_seconds=duration_seconds)
	if len(windows) < MIN_BAR_COUNT:
		return LeadSheetWindows(_phrase_windows(segments, duration_seconds), "phrase")
	return LeadSheetWindows(windows, "bar")


def _beat_interval_seconds(beat_timing: BeatTiming | None) -> float | None:
	if beat_timing is None or not beat_timing.is_reliable or beat_timing.tempo_bpm is None:
		return None
	tempo = float(beat_timing.tempo_bpm)
	if not np.isfinite(tempo) or tempo <= 0.0:
		return None
	return 60.0 / tempo


def _first_beat_seconds(
	*,
	beat_timing: BeatTiming | None,
	hop_length: int | None,
	sample_rate: int | None,
) -> float:
	if beat_timing is None or hop_length is None or sample_rate is None:
		return 0.0
	values = beat_times_seconds(beat_timing, hop_length=hop_length, sample_rate=sample_rate)
	return values[0] if values else 0.0


def _infer_downbeat_phase(segments: list[ChordSegment], *, first_beat: float, beat_interval: float) -> float:
	best_phase = 0
	best_score = None
	for phase in range(BAR_BEATS):
		candidate = first_beat + (phase * beat_interval)
		score = _downbeat_alignment_score(segments, downbeat=candidate, bar_length=beat_interval * BAR_BEATS)
		if best_score is None or score > best_score:
			best_score = score
			best_phase = phase
	return first_beat + (best_phase * beat_interval)


def _downbeat_alignment_score(segments: list[ChordSegment], *, downbeat: float, bar_length: float) -> float:
	score = _initial_partial_bar_score(downbeat, bar_length=bar_length)
	first_chord = segments[0].chord if segments else None
	for segment in segments:
		if segment.chord == "N":
			continue
		distance = _distance_to_grid(segment.start, downbeat=downbeat, interval=bar_length)
		if distance <= DOWNBEAT_ALIGNMENT_WINDOW_SECONDS:
			score += 1.0 - (distance / DOWNBEAT_ALIGNMENT_WINDOW_SECONDS)
			if first_chord is not None and segment.chord == first_chord:
				score += 0.55
		score += 0.15 * _segment_grid_weight(segment, downbeat=downbeat, bar_length=bar_length)
	return score


def _initial_partial_bar_score(downbeat: float, *, bar_length: float) -> float:
	if bar_length <= 0.0:
		return 0.0
	previous = downbeat
	while previous > 0.0:
		previous -= bar_length
	first_grid = previous + bar_length
	partial = first_grid if first_grid > 0.0 else 0.0
	if partial <= 0.0:
		return 0.0
	return -DOWNBEAT_PARTIAL_BAR_PENALTY * (partial / bar_length)


def _distance_to_grid(value: float, *, downbeat: float, interval: float) -> float:
	if interval <= 0.0:
		return abs(value - downbeat)
	position = (value - downbeat) / interval
	nearest = downbeat + (round(position) * interval)
	return abs(value - nearest)


def _segment_grid_weight(segment: ChordSegment, *, downbeat: float, bar_length: float) -> float:
	mid = (segment.start + segment.end) * 0.5
	distance = _distance_to_grid(mid, downbeat=downbeat, interval=bar_length)
	return max(0.0, 1.0 - (distance / max(bar_length * 0.5, 1e-6)))


def _build_bar_windows(downbeat: float, *, bar_length: float, duration_seconds: float) -> list[tuple[float, float]]:
	if bar_length <= 0.0:
		return []
	while downbeat > 0.0:
		downbeat -= bar_length
	while downbeat + bar_length <= 0.0:
		downbeat += bar_length
	edges = [0.0]
	cursor = downbeat
	while cursor < duration_seconds:
		if cursor > 0.0:
			edges.append(float(cursor))
		cursor += bar_length
	if edges[-1] < duration_seconds:
		edges.append(duration_seconds)
	windows: list[tuple[float, float]] = []
	for start, end in zip(edges[:-1], edges[1:], strict=False):
		if end - start > 0.25:
			windows.append((start, end))
	return windows


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


def _limit_marker_density(markers: list[tuple[str, float]], *, max_chords: int) -> list[tuple[str, float]]:
	if len(markers) <= max_chords:
		return markers
	if max_chords <= 2:
		return [markers[0], markers[-1]] if markers[0][0] != markers[-1][0] else [markers[0]]
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
