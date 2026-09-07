"""Final playable guitar progression cleanup.

This stage is intentionally song-independent. It favors sustained harmonic
functions inside a confident global key and suppresses short, low-evidence
full-mix flashes that are usually not useful as guitar chord changes.
"""

from __future__ import annotations

import numpy as np

from chord_engine.analysis_config import (
	PLAYABLE_PROGRESSION_FRAGMENT_SECONDS,
	PLAYABLE_PROGRESSION_KEY_CONFIDENCE_MIN,
	PLAYABLE_PROGRESSION_NEIGHBOR_MIN_SECONDS,
	PLAYABLE_PROGRESSION_NEIGHBOR_SCORE_MIN,
	PLAYABLE_PROGRESSION_SAME_ROOT_CONFIDENCE_MAX,
	PLAYABLE_PROGRESSION_SAME_ROOT_SECONDS_MAX,
	PLAYABLE_PROGRESSION_SHORT_DIATONIC_SECONDS,
	PLAYABLE_PROGRESSION_SHORT_NON_DIATONIC_SECONDS,
	PLAYABLE_PROGRESSION_WEAK_CONFIDENCE,
)
from chord_engine.analysis_models import CorrectionEvent
from chord_engine.detector import KeyEstimate
from chord_engine.music_theory import (
	_chord_root_pc,
	_diatonic_chord_labels_for_key,
	_harmonic_plausibility,
	_harmonic_relationship_strength,
	_is_diatonic_chord,
	_parse_chord_quality,
)
from chord_engine.segment_utils import (
	_merge_adjacent_same_chord_segments,
	_neighbor_support,
	_segment_duration,
	_weighted_confidence,
)
from chord_engine.segmentation import ChordSegment


def _apply_playable_progression_refinement(
	segments: list[ChordSegment],
	detected_key: KeyEstimate | None,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if detected_key is None or detected_key.confidence < PLAYABLE_PROGRESSION_KEY_CONFIDENCE_MIN or len(segments) < 2:
		return segments, []

	key_chords = _diatonic_chord_labels_for_key(detected_key)
	if not key_chords:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	for _ in range(2):
		working, same_root_events = _prefer_same_root_diatonic_variants(working, detected_key, key_chords)
		events.extend(same_root_events)
		working, fragment_events = _absorb_unplayable_fragments(working, detected_key)
		events.extend(fragment_events)
		working = _merge_adjacent_same_chord_segments(working)

	return _merge_adjacent_same_chord_segments(working), events


def _prefer_same_root_diatonic_variants(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
	key_chords: set[str],
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	working = list(segments)
	events: list[CorrectionEvent] = []
	for idx, segment in enumerate(working):
		if segment.chord == "N" or _is_diatonic_chord(segment.chord, detected_key):
			continue
		if _segment_duration(segment) > PLAYABLE_PROGRESSION_SAME_ROOT_SECONDS_MAX:
			continue
		if segment.confidence > PLAYABLE_PROGRESSION_SAME_ROOT_CONFIDENCE_MAX:
			continue
		replacement = _same_root_key_chord(segment.chord, key_chords)
		if replacement is None:
			continue
		working[idx] = ChordSegment(
			start=segment.start,
			end=segment.end,
			chord=replacement,
			confidence=float(np.clip(segment.confidence * 0.96, 0.0, 1.0)),
		)
		events.append(_event(idx, segment, working[idx], detected_key, score=0.52))
	return _merge_adjacent_same_chord_segments(working), events


def _absorb_unplayable_fragments(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if len(segments) < 3:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	for idx in range(1, len(working) - 1):
		current = working[idx]
		if not _is_unplayable_fragment(current, detected_key):
			continue
		candidate = _best_stable_neighbor(working[idx - 1], current, working[idx + 1], detected_key)
		if candidate is None or candidate.chord == current.chord:
			continue
		working[idx] = ChordSegment(
			start=current.start,
			end=current.end,
			chord=candidate.chord,
			confidence=float(np.clip(_weighted_confidence(current, candidate) * 0.98, 0.0, 1.0)),
		)
		events.append(_event(idx, current, working[idx], detected_key, score=_neighbor_support(candidate, detected_key)))
	return _merge_adjacent_same_chord_segments(working), events


def _is_unplayable_fragment(segment: ChordSegment, detected_key: KeyEstimate) -> bool:
	duration = _segment_duration(segment)
	if segment.chord == "N":
		return False
	if not _is_diatonic_chord(segment.chord, detected_key):
		return duration <= PLAYABLE_PROGRESSION_SHORT_NON_DIATONIC_SECONDS or segment.confidence <= PLAYABLE_PROGRESSION_SAME_ROOT_CONFIDENCE_MAX
	if duration <= PLAYABLE_PROGRESSION_SHORT_DIATONIC_SECONDS and segment.confidence <= PLAYABLE_PROGRESSION_SAME_ROOT_CONFIDENCE_MAX:
		return True
	return duration <= PLAYABLE_PROGRESSION_FRAGMENT_SECONDS and segment.confidence <= PLAYABLE_PROGRESSION_WEAK_CONFIDENCE


def _best_stable_neighbor(
	left: ChordSegment,
	current: ChordSegment,
	right: ChordSegment,
	detected_key: KeyEstimate,
) -> ChordSegment | None:
	candidates = [segment for segment in (left, right) if _segment_duration(segment) >= PLAYABLE_PROGRESSION_NEIGHBOR_MIN_SECONDS]
	if not candidates:
		candidates = [left, right]
	scored: list[tuple[float, ChordSegment]] = []
	for candidate in candidates:
		if candidate.chord == "N":
			continue
		score = _neighbor_support(candidate, detected_key)
		score += 0.12 * _harmonic_relationship_strength(current.chord, candidate.chord, detected_key)
		if _is_diatonic_chord(candidate.chord, detected_key):
			score += 0.10
		if candidate.chord == left.chord == right.chord:
			score += 0.18
		if score >= PLAYABLE_PROGRESSION_NEIGHBOR_SCORE_MIN:
			scored.append((float(score), candidate))
	if not scored:
		return None
	return sorted(scored, key=lambda item: (-item[0], item[1].chord))[0][1]


def _same_root_key_chord(chord: str, key_chords: set[str]) -> str | None:
	root_pc = _chord_root_pc(chord)
	if root_pc is None:
		return None
	for candidate in sorted(key_chords):
		if _chord_root_pc(candidate) == root_pc and _parse_chord_quality(candidate)[1] != "diminished":
			return candidate
	return None


def _event(
	index: int,
	before: ChordSegment,
	after: ChordSegment,
	detected_key: KeyEstimate,
	*,
	score: float,
) -> CorrectionEvent:
	return CorrectionEvent(
		index=index,
		replaced_chord=before.chord,
		new_chord=after.chord,
		start=float(before.start),
		end=float(before.end),
		duration=float(_segment_duration(before)),
		original_confidence=float(before.confidence),
		new_confidence=float(after.confidence),
		score=float(score),
		harmonic_plausibility_before=float(_harmonic_plausibility(before.chord, detected_key)),
		harmonic_plausibility_after=float(_harmonic_plausibility(after.chord, detected_key)),
	)
