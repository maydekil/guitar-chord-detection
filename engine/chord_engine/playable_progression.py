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
	PLAYABLE_PROGRESSION_PHRASE_FRAGMENT_SECONDS,
	PLAYABLE_PROGRESSION_PHRASE_SUPPORT_MIN,
	PLAYABLE_PROGRESSION_SAME_ROOT_CONFIDENCE_MAX,
	PLAYABLE_PROGRESSION_SAME_ROOT_SECONDS_MAX,
	PLAYABLE_PROGRESSION_SHORT_DIATONIC_SECONDS,
	PLAYABLE_PROGRESSION_SHORT_NON_DIATONIC_SECONDS,
	PLAYABLE_PROGRESSION_SUBSTITUTE_CONFIDENCE_MAX,
	PLAYABLE_PROGRESSION_SUBSTITUTE_FRAGMENT_SECONDS,
	PLAYABLE_PROGRESSION_WEAK_CONFIDENCE,
	PLAYABLE_PHRASE_CHANGE_COST,
	PLAYABLE_PHRASE_DIATONIC_BONUS,
	PLAYABLE_PHRASE_GROUP_MAX_SECONDS,
	PLAYABLE_PHRASE_GROUP_MIN_SECONDS,
	PLAYABLE_PHRASE_GROUP_TARGET_SECONDS,
	PLAYABLE_PHRASE_MIN_REPLACEMENT_SCORE,
	PLAYABLE_PHRASE_PRIMARY_FUNCTION_BONUS,
	PLAYABLE_PHRASE_RELATIONSHIP_WEIGHT,
	PLAYABLE_PHRASE_SECONDARY_FUNCTION_BONUS,
	PLAYABLE_PHRASE_SELF_BONUS,
	PLAYABLE_PHRASE_SUBSTITUTE_NEAR_MARGIN,
	PLAYABLE_PHRASE_SUBSTITUTE_SOFT_PENALTY,
	PLAYABLE_PHRASE_TONIC_ANCHOR_BONUS,
	PLAYABLE_PHRASE_TOP_K,
	PLAYABLE_PHRASE_CADENCE_BONUS,
	PLAYABLE_PHRASE_CHROMA_RAW_MIN,
	PLAYABLE_PHRASE_CHROMA_SCORE_WEIGHT,
	PLAYABLE_PREDOMINANT_INSERT_MAX_PREV_SECONDS,
	PLAYABLE_PREDOMINANT_INSERT_MAX_SECONDS,
	PLAYABLE_PREDOMINANT_INSERT_MIN_PREV_SECONDS,
	PLAYABLE_PREDOMINANT_INSERT_MIN_REMAIN_SECONDS,
	PLAYABLE_PREDOMINANT_INSERT_MIN_SECONDS,
	PLAYABLE_PREDOMINANT_INSERT_MIN_TONIC_SECONDS,
	PLAYABLE_PREDOMINANT_INSERT_RATIO,
	PLAYABLE_LEADSHEET_SPLIT_EDGE_MIN_SECONDS,
	PLAYABLE_LEADSHEET_SPLIT_MIN_SECONDS,
	PLAYABLE_LEADSHEET_SPLIT_RATIO,
	PLAYABLE_INTRO_PICKUP_MAX_SECONDS,
	PLAYABLE_INTRO_TONIC_EXTENSION_CONFIDENCE_MAX,
	PLAYABLE_INTRO_TONIC_EXTENSION_MAX_SECONDS,
	PLAYABLE_INTRO_TONIC_EXTENSION_TOTAL_MAX_SECONDS,
	PLAYABLE_INTRO_TONIC_MEDIANT_SPLIT_EDGE_MIN_SECONDS,
	PLAYABLE_INTRO_TONIC_MEDIANT_SPLIT_MIN_SECONDS,
	PLAYABLE_INTRO_TONIC_LOOKAHEAD_SECONDS,
	PLAYABLE_INTRO_TONIC_MIN_TOTAL_SECONDS,
	PLAYABLE_MEDIANT_APPROACH_CONFIDENCE_MAX,
	PLAYABLE_MEDIANT_APPROACH_FRAGMENT_SECONDS,
	PLAYABLE_MEDIANT_APPROACH_SUPPORT_MIN_SECONDS,
	PLAYABLE_REPEAT_PATTERN_MAX_LENGTH,
	PLAYABLE_REPEAT_PATTERN_MIN_LENGTH,
	PLAYABLE_REPEAT_PATTERN_MIN_OCCURRENCES,
	PLAYABLE_REPEAT_QUALITY_MAJORITY_RATIO,
	PLAYABLE_REPEAT_QUALITY_REPLACE_CONFIDENCE_MAX,
	PLAYABLE_REPEAT_QUALITY_REPLACE_SECONDS_MAX,
	PLAYABLE_REPEAT_ROLE_MIN_SEGMENT_SECONDS,
	PLAYABLE_REPEAT_ROLE_CADENCE_MIN_SEGMENT_SECONDS,
	PLAYABLE_REPEAT_ROLE_BODY_CADENCE_MIN_SEGMENT_SECONDS,
	PLAYABLE_REPEAT_ROLE_CONTINUATION_MIN_SEGMENT_SECONDS,
	PLAYABLE_REPEAT_ROLE_CONTINUATION_SPLIT_MIN_SECONDS,
	PLAYABLE_REPEAT_ROLE_CONTINUATION_SPLIT_RATIO,
	PLAYABLE_REPEAT_ROLE_DEGRADED_OPENING_SPLIT_MIN_SECONDS,
	PLAYABLE_REPEAT_ROLE_DEGRADED_OPENING_SPLIT_RATIO,
	PLAYABLE_REPEAT_CYCLE_TAIL_MIN_SEGMENT_SECONDS,
	PLAYABLE_REPEAT_CYCLE_TAIL_MIN_TOTAL_SECONDS,
	PLAYABLE_REPEAT_TURNAROUND_OVERLONG_DOMINANT_MIN_SECONDS,
	PLAYABLE_REPEAT_TURNAROUND_OVERLONG_DOMINANT_RATIO,
	PLAYABLE_REPEAT_TURNAROUND_SLOT_MAX_SECONDS,
	PLAYABLE_REPEAT_TURNAROUND_SLOT_MIN_SECONDS,
	PLAYABLE_SECTION_PATTERN_CONFIDENCE,
	PLAYABLE_SECTION_PATTERN_CYCLE_MIN_EVIDENCE,
	PLAYABLE_SECTION_PATTERN_CYCLE_SLOT_MAX_SECONDS,
	PLAYABLE_SECTION_PATTERN_CYCLE_SLOT_MIN_SECONDS,
	PLAYABLE_SECTION_PATTERN_EARLY_BONUS,
	PLAYABLE_SECTION_PATTERN_EARLY_LONG_SLOT_BONUS_MIN_SECONDS,
	PLAYABLE_SECTION_PATTERN_EARLY_LONG_SLOT_MIN_SECONDS,
	PLAYABLE_SECTION_PATTERN_EARLY_MAX_START_SECONDS,
	PLAYABLE_SECTION_PATTERN_EARLY_MIN_EVIDENCE,
	PLAYABLE_SECTION_PATTERN_INTRO_RESOLUTION_BONUS,
	PLAYABLE_SECTION_PATTERN_LATE_MIN_START_SECONDS,
	PLAYABLE_SECTION_PATTERN_LONG_MIN_EVIDENCE,
	PLAYABLE_SECTION_PATTERN_LONG_SLOT_MAX_SECONDS,
	PLAYABLE_SECTION_PATTERN_LONG_SLOT_MIN_SECONDS,
	PLAYABLE_SECTION_PATTERN_MAX_START_SECONDS,
	PLAYABLE_SECTION_PATTERN_MIN_SONG_SECONDS,
	PLAYABLE_SECTION_PATTERN_MIN_WINDOW_SECONDS,
	PLAYABLE_SECTION_PATTERN_SLOT_BONUS,
	PLAYABLE_MULTI_SECTION_PATTERN_CONFIDENCE,
	PLAYABLE_MULTI_SECTION_PATTERN_MAX_BOUNDARY_OFFSET,
	PLAYABLE_MULTI_SECTION_PATTERN_MAX_CANDIDATES,
	PLAYABLE_MULTI_SECTION_PATTERN_MAX_SLOT_SECONDS,
	PLAYABLE_MULTI_SECTION_PATTERN_MAX_START_SECONDS,
	PLAYABLE_MULTI_SECTION_PATTERN_MAX_WINDOW_SECONDS,
	PLAYABLE_MULTI_SECTION_PATTERN_MIN_CHROMA_EVIDENCE,
	PLAYABLE_MULTI_SECTION_PATTERN_MIN_EVIDENCE,
	PLAYABLE_MULTI_SECTION_PATTERN_MIN_GAIN,
	PLAYABLE_MULTI_SECTION_PATTERN_MIN_SCORE,
	PLAYABLE_MULTI_SECTION_PATTERN_MIN_SLOT_SECONDS,
	PLAYABLE_MULTI_SECTION_PATTERN_MIN_START_SECONDS,
	PLAYABLE_MULTI_SECTION_PATTERN_MIN_SONG_SECONDS,
	PLAYABLE_MULTI_SECTION_PATTERN_MIN_SUPPORT,
	PLAYABLE_MULTI_SECTION_PATTERN_MIN_WINDOW_SECONDS,
	PLAYABLE_MULTI_SECTION_PATTERN_SLOT_STEP_SECONDS,
	PLAYABLE_REPEAT_ROLE_OPENING_MAX_SECONDS,
	PLAYABLE_REPEAT_ROLE_OPENING_DOMINANT_SPLIT_MIN_SECONDS,
	PLAYABLE_REPEAT_ROLE_OPENING_DOMINANT_SPLIT_RATIO,
	PLAYABLE_REPEAT_ROLE_POST_CADENCE_FRAGMENT_SECONDS,
	PLAYABLE_REPEAT_ROLE_OPENING_RESTART_MAX_SECONDS,
	PLAYABLE_REPEAT_ROLE_OPENING_RESTART_SPLIT_MIN_SECONDS,
	PLAYABLE_REPEAT_ROLE_OPENING_RESTART_SPLIT_RATIO,
	PLAYABLE_REPEAT_ROLE_RESTART_FRAGMENT_SECONDS,
	PLAYABLE_REPEAT_ROLE_SPLIT_MIN_SECONDS,
	PLAYABLE_REPEAT_ROLE_SPLIT_RATIO,
)
from chord_engine.analysis_models import CorrectionEvent
from chord_engine.detector import KeyEstimate
from chord_engine.features import BeatTiming, beat_times_seconds
from chord_engine.music_theory import (
	_chord_root_pc,
	_diatonic_chord_labels_for_key,
	_harmonic_plausibility,
	_harmonic_relationship_strength,
	_is_diatonic_chord,
	_parse_chord_quality,
	_root_name_from_pc,
)
from chord_engine.segment_utils import (
	_merge_adjacent_same_chord_segments,
	_neighbor_support,
	_segment_duration,
	_weighted_confidence,
)
from chord_engine.segmentation import ChordSegment
from chord_engine.numeric import _cosine_similarity
from chord_engine.templates import generate_chord_templates
from chord_engine.timebase import frame_duration_seconds

LONG_SEGMENT_CHROMA_SPLIT_MIN_SECONDS = 7.0
LONG_SEGMENT_CHROMA_SPLIT_MIN_WINDOW_SECONDS = 1.35
LONG_SEGMENT_CHROMA_SPLIT_DEFAULT_WINDOW_SECONDS = 2.8
LONG_SEGMENT_CHROMA_SPLIT_MAX_PARTS = 8
LONG_SEGMENT_CHROMA_SPLIT_SCORE_MIN = 0.56
LONG_SEGMENT_CHROMA_SPLIT_MARGIN_MIN = 0.035
LONG_SEGMENT_CHROMA_SPLIT_ORIGINAL_DELTA_MIN = 0.18
LONG_SEGMENT_CHROMA_SPLIT_DIATONIC_BONUS = 0.08
LONG_SEGMENT_CHROMA_SPLIT_CONFIDENCE = 0.82


def _apply_playable_progression_refinement(
	segments: list[ChordSegment],
	detected_key: KeyEstimate | None,
	*,
	chroma: np.ndarray | None = None,
	hop_length: int | None = None,
	sample_rate: int | None = None,
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
		working, substitute_events = _absorb_functional_substitute_fragments(working, detected_key)
		events.extend(substitute_events)
		working, fragment_events = _absorb_unplayable_fragments(working, detected_key)
		events.extend(fragment_events)
		working, phrase_events = _absorb_short_phrase_fragments(working, detected_key)
		events.extend(phrase_events)
		working = _merge_adjacent_same_chord_segments(working)
	working, decoder_events = _decode_phrase_level_playable_progression(
		working,
		detected_key,
		chroma=chroma,
		hop_length=hop_length,
		sample_rate=sample_rate,
	)
	events.extend(decoder_events)
	working, predominant_events = _insert_predominant_resolutions(working, detected_key)
	events.extend(predominant_events)
	working, leadsheet_events = _apply_major_leadsheet_continuity(working, detected_key)
	events.extend(leadsheet_events)
	working, internal_chroma_events = _split_long_segments_by_internal_chroma(
		working,
		detected_key,
		chroma=chroma,
		hop_length=hop_length,
		sample_rate=sample_rate,
	)
	events.extend(internal_chroma_events)
	working, intro_events = _anchor_intro_pickup_to_tonic(working, detected_key)
	events.extend(intro_events)
	working, repeat_events = _apply_repeated_phrase_quality_consistency(working, detected_key)
	events.extend(repeat_events)
	working, intro_tonic_events = _stabilize_initial_tonic_pickup(working, detected_key)
	events.extend(intro_tonic_events)
	working, mediant_events = _absorb_tonic_predominant_mediant_approach(working, detected_key)
	events.extend(mediant_events)

	merged = _merge_adjacent_same_chord_segments(working)
	merged, split_events = _split_initial_tonic_mediant_before_predominant(merged, detected_key)
	events.extend(split_events)
	merged, role_events = _apply_major_repeated_tonic_phrase_answer(merged, detected_key)
	events.extend(role_events)
	merged, final_role_events = _apply_major_repeated_tonic_phrase_answer(merged, detected_key)
	events.extend(final_role_events)
	merged, section_pattern_events = _apply_major_section_pattern_arranger(merged, detected_key)
	events.extend(section_pattern_events)
	return merged, events


def _split_initial_tonic_mediant_before_predominant(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if detected_key.mode != "major" or len(segments) < 2:
		return segments, []
	first = segments[0]
	second = segments[1]
	if first.start > 0.15 or first.chord != _major_degree_chord(detected_key, 0):
		return segments, []
	duration = _segment_duration(first)
	if duration < PLAYABLE_INTRO_TONIC_MEDIANT_SPLIT_MIN_SECONDS:
		return segments, []
	if _functional_role(second.chord, detected_key) != "predominant":
		return segments, []
	mediant = _major_degree_chord(detected_key, 4)
	if mediant == first.chord or not _is_diatonic_chord(mediant, detected_key):
		return segments, []
	left_duration = duration * PLAYABLE_LEADSHEET_SPLIT_RATIO
	right_duration = duration - left_duration
	if min(left_duration, right_duration) < PLAYABLE_INTRO_TONIC_MEDIANT_SPLIT_EDGE_MIN_SECONDS:
		return segments, []
	mid = first.start + left_duration
	left = ChordSegment(start=first.start, end=mid, chord=first.chord, confidence=first.confidence)
	right = ChordSegment(
		start=mid,
		end=first.end,
		chord=mediant,
		confidence=float(np.clip(first.confidence * 0.92, 0.0, 1.0)),
	)
	return [left, right, *segments[1:]], [_event(0, first, right, detected_key, score=0.57)]


def _split_long_segments_by_internal_chroma(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
	*,
	chroma: np.ndarray | None,
	hop_length: int | None,
	sample_rate: int | None,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if chroma is None or hop_length is None or sample_rate is None or chroma.ndim != 2 or chroma.shape[0] != 12:
		return segments, []
	frame_seconds = frame_duration_seconds(hop_length=hop_length, sample_rate=sample_rate)
	if frame_seconds <= 0.0:
		return segments, []
	target_seconds = _internal_split_target_seconds(segments)
	refined: list[ChordSegment] = []
	events: list[CorrectionEvent] = []
	for idx, segment in enumerate(segments):
		duration = _segment_duration(segment)
		if segment.chord == "N" or duration < LONG_SEGMENT_CHROMA_SPLIT_MIN_SECONDS:
			refined.append(segment)
			continue
		pieces = _internal_chroma_split_pieces(
			segment,
			detected_key,
			chroma=chroma,
			frame_seconds=frame_seconds,
			target_seconds=target_seconds,
		)
		if pieces is None:
			refined.append(segment)
			continue
		refined.extend(pieces)
		events.append(_event(idx, segment, pieces[0], detected_key, score=0.59))
	return _merge_adjacent_same_chord_segments(refined), events


def _internal_split_target_seconds(segments: list[ChordSegment]) -> float:
	durations = [
		_segment_duration(segment)
		for segment in segments
		if segment.chord != "N" and 1.2 <= _segment_duration(segment) <= 4.2
	]
	if not durations:
		return LONG_SEGMENT_CHROMA_SPLIT_DEFAULT_WINDOW_SECONDS
	return float(np.clip(np.median(durations), 1.8, LONG_SEGMENT_CHROMA_SPLIT_DEFAULT_WINDOW_SECONDS))


def _internal_chroma_split_pieces(
	segment: ChordSegment,
	detected_key: KeyEstimate,
	*,
	chroma: np.ndarray,
	frame_seconds: float,
	target_seconds: float,
) -> list[ChordSegment] | None:
	duration = _segment_duration(segment)
	parts = int(np.clip(round(duration / target_seconds), 2, LONG_SEGMENT_CHROMA_SPLIT_MAX_PARTS))
	part_duration = duration / parts
	if part_duration < LONG_SEGMENT_CHROMA_SPLIT_MIN_WINDOW_SECONDS:
		return None
	pieces: list[ChordSegment] = []
	changed = False
	for part in range(parts):
		start = segment.start + (part * part_duration)
		end = segment.end if part == parts - 1 else start + part_duration
		best = _best_internal_chroma_chord(
			segment.chord,
			detected_key,
			chroma=chroma,
			frame_seconds=frame_seconds,
			start=start,
			end=end,
		)
		if best is None:
			return None
		chord, score, margin, original_score = best
		if chord != segment.chord:
			if (
				score < LONG_SEGMENT_CHROMA_SPLIT_SCORE_MIN
				or margin < LONG_SEGMENT_CHROMA_SPLIT_MARGIN_MIN
				or score - original_score < LONG_SEGMENT_CHROMA_SPLIT_ORIGINAL_DELTA_MIN
			):
				chord = segment.chord
			else:
				changed = True
		pieces.append(
			ChordSegment(
				start=start,
				end=end,
				chord=chord,
				confidence=float(np.clip(max(segment.confidence, LONG_SEGMENT_CHROMA_SPLIT_CONFIDENCE), 0.0, 1.0)),
			)
		)
	if not changed:
		return None
	return _merge_adjacent_same_chord_segments(pieces)


def _best_internal_chroma_chord(
	original_chord: str,
	detected_key: KeyEstimate,
	*,
	chroma: np.ndarray,
	frame_seconds: float,
	start: float,
	end: float,
) -> tuple[str, float, float] | None:
	start_frame = max(0, int(np.floor(start / frame_seconds)))
	end_frame = min(chroma.shape[1], max(start_frame + 1, int(np.ceil(end / frame_seconds))))
	if end_frame <= start_frame:
		return None
	vector = np.mean(chroma[:, start_frame:end_frame], axis=1)
	if float(np.linalg.norm(vector)) <= 1e-7:
		return None
	templates = generate_chord_templates()
	scores: list[tuple[str, float]] = []
	for chord in _diatonic_chord_labels_for_key(detected_key):
		template = templates.get(chord)
		if template is None:
			continue
		score = _cosine_similarity(vector, template, min_norm=1e-7)
		if _is_diatonic_chord(chord, detected_key):
			score += LONG_SEGMENT_CHROMA_SPLIT_DIATONIC_BONUS
		scores.append((chord, float(score)))
	if not scores:
		return None
	scores.sort(key=lambda item: (-item[1], item[0]))
	best_score = scores[0][1]
	second_score = scores[1][1] if len(scores) > 1 else 0.0
	original_score = dict(scores).get(original_chord, 0.0)
	return scores[0][0], best_score, best_score - second_score, original_score


def _anchor_intro_pickup_to_tonic(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if detected_key.mode != "major" or len(segments) < 2:
		return segments, []
	first = segments[0]
	if first.start > 0.15 or first.chord == "N" or _segment_duration(first) > PLAYABLE_INTRO_PICKUP_MAX_SECONDS:
		return segments, []
	tonic = _major_degree_chord(detected_key, 0)
	if first.chord == tonic:
		return segments, []
	if _early_tonic_duration(segments, tonic) < PLAYABLE_INTRO_TONIC_MIN_TOTAL_SECONDS:
		return segments, []
	replacement = ChordSegment(
		start=first.start,
		end=first.end,
		chord=tonic,
		confidence=float(np.clip(max(first.confidence, 0.78), 0.0, 1.0)),
	)
	working = [replacement, *segments[1:]]
	return _merge_adjacent_same_chord_segments(working), [_event(0, first, replacement, detected_key, score=0.62)]


def _stabilize_initial_tonic_pickup(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if detected_key.mode != "major" or len(segments) < 2:
		return segments, []
	first = segments[0]
	if first.start > 0.15 or first.chord == "N" or _segment_duration(first) > PLAYABLE_INTRO_PICKUP_MAX_SECONDS:
		return segments, []
	tonic = _major_degree_chord(detected_key, 0)
	if _early_tonic_duration(segments, tonic) < PLAYABLE_INTRO_TONIC_MIN_TOTAL_SECONDS:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	if first.chord != tonic:
		replacement = ChordSegment(
			start=first.start,
			end=first.end,
			chord=tonic,
			confidence=float(np.clip(max(first.confidence, 0.78), 0.0, 1.0)),
		)
		working[0] = replacement
		events.append(_event(0, first, replacement, detected_key, score=0.62))

	if len(working) < 2:
		return _merge_adjacent_same_chord_segments(working), events
	first = working[0]
	second = working[1]
	first_duration = _segment_duration(first)
	second_duration = _segment_duration(second)
	if (
		first.chord == tonic
		and second.chord != "N"
		and second.chord != tonic
		and first_duration <= PLAYABLE_INTRO_PICKUP_MAX_SECONDS
		and second_duration <= PLAYABLE_INTRO_TONIC_EXTENSION_MAX_SECONDS
		and second.confidence <= PLAYABLE_INTRO_TONIC_EXTENSION_CONFIDENCE_MAX
		and second.end - first.start <= PLAYABLE_INTRO_TONIC_EXTENSION_TOTAL_MAX_SECONDS
		and _functional_role(second.chord, detected_key) in {"dominant", "substitute"}
	):
		replacement = ChordSegment(
			start=second.start,
			end=second.end,
			chord=tonic,
			confidence=float(np.clip(max(first.confidence, second.confidence) * 0.98, 0.0, 1.0)),
		)
		working[1] = replacement
		events.append(_event(1, second, replacement, detected_key, score=0.60))

	return _merge_adjacent_same_chord_segments(working), events


def _early_tonic_duration(segments: list[ChordSegment], tonic: str) -> float:
	total = 0.0
	for segment in segments:
		if segment.start > PLAYABLE_INTRO_TONIC_LOOKAHEAD_SECONDS:
			break
		if segment.chord == tonic:
			total += max(0.0, min(segment.end, PLAYABLE_INTRO_TONIC_LOOKAHEAD_SECONDS) - segment.start)
	return total


def _apply_repeated_phrase_quality_consistency(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if len(segments) < PLAYABLE_REPEAT_PATTERN_MIN_LENGTH * PLAYABLE_REPEAT_PATTERN_MIN_OCCURRENCES:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	for length in range(PLAYABLE_REPEAT_PATTERN_MAX_LENGTH, PLAYABLE_REPEAT_PATTERN_MIN_LENGTH - 1, -1):
		occurrences_by_signature = _repeated_root_signature_occurrences(working, length)
		for starts in occurrences_by_signature.values():
			if len(starts) < PLAYABLE_REPEAT_PATTERN_MIN_OCCURRENCES:
				continue
			for offset in range(length):
				winner = _repeated_phrase_quality_winner(working, starts, offset)
				if winner is None:
					continue
				for start in starts:
					idx = start + offset
					if idx >= len(working):
						continue
					current = working[idx]
					if not _can_replace_repeated_phrase_quality(current, winner, detected_key):
						continue
					replacement = ChordSegment(
						start=current.start,
						end=current.end,
						chord=winner,
						confidence=float(np.clip(max(current.confidence, 0.70), 0.0, 1.0)),
					)
					working[idx] = replacement
					events.append(_event(idx, current, replacement, detected_key, score=0.61))
		if events:
			working = _merge_adjacent_same_chord_segments(working)
	return working, events


def _apply_major_repeated_tonic_phrase_answer(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	context_key = _major_playable_context_key(segments, detected_key)
	if context_key is None or len(segments) < 6:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	working, degraded_opening_events = _recover_degraded_major_opening_answer(working, context_key)
	events.extend(degraded_opening_events)
	working, answer_events = _answer_immediate_repeated_tonic_phrase(working, context_key)
	events.extend(answer_events)
	working, cadence_events = _recover_post_cadence_tonic(working, context_key)
	events.extend(cadence_events)
	working, cadence_answer_events = _answer_immediate_opening_cadence_phrase(working, context_key)
	events.extend(cadence_answer_events)
	working, restart_events = _recover_opening_answer_phrase_restart(working, context_key)
	events.extend(restart_events)
	working, continuation_events = _recover_opening_phrase_continuation(working, context_key)
	events.extend(continuation_events)
	working, body_cadence_events = _recover_body_cadence_from_opening_answer(working, context_key)
	events.extend(body_cadence_events)
	working, opening_cycle_events = _complete_major_opening_phrase_cycle(working, context_key)
	events.extend(opening_cycle_events)
	working, opening_cycle_restart_events = _recover_completed_opening_cycle_restart(working, context_key)
	events.extend(opening_cycle_restart_events)
	working, repeated_tail_events = _recover_repeated_opening_cycle_tail(working, context_key)
	events.extend(repeated_tail_events)
	working, turnaround_events = _split_repeated_cycle_overlong_dominant_turnaround(working, context_key)
	events.extend(turnaround_events)
	working, section_pattern_events = _apply_major_section_pattern_arranger(working, context_key)
	events.extend(section_pattern_events)
	return _merge_adjacent_same_chord_segments(working), events


def _recover_degraded_major_opening_answer(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if len(segments) < 9 or segments[0].start > PLAYABLE_REPEAT_ROLE_OPENING_MAX_SECONDS:
		return segments, []

	working = list(segments)
	degrees = [_major_degree(segment.chord, detected_key) for segment in working[:9]]
	if degrees != [0, 2, 0, 7, 4, 0, 4, 5, 0]:
		return segments, []
	if any(_segment_duration(segment) < PLAYABLE_REPEAT_ROLE_MIN_SEGMENT_SECONDS for segment in working[3:5]):
		return segments, []

	events: list[CorrectionEvent] = []
	mediant = _major_degree_chord(detected_key, 4)
	subdominant = _major_degree_chord(detected_key, 5)
	submediant = _major_degree_chord(detected_key, 9)
	predominant = _major_degree_chord(detected_key, 2)
	dominant = _major_degree_chord(detected_key, 7)

	second = working[1]
	replacement = ChordSegment(
		start=second.start,
		end=second.end,
		chord=mediant,
		confidence=float(np.clip(second.confidence * 0.94, 0.0, 1.0)),
	)
	working[1] = replacement
	events.append(_event(1, second, replacement, detected_key, score=0.55))

	third = working[2]
	if _segment_duration(third) >= PLAYABLE_REPEAT_ROLE_DEGRADED_OPENING_SPLIT_MIN_SECONDS:
		split_at = third.start + (_segment_duration(third) * PLAYABLE_REPEAT_ROLE_DEGRADED_OPENING_SPLIT_RATIO)
		left = ChordSegment(
			start=third.start,
			end=split_at,
			chord=subdominant,
			confidence=float(np.clip(third.confidence * 0.94, 0.0, 1.0)),
		)
		right = ChordSegment(
			start=split_at,
			end=third.end,
			chord=_major_degree_chord(detected_key, 0),
			confidence=float(np.clip(third.confidence * 0.92, 0.0, 1.0)),
		)
		working[2:3] = [left, right]
		events.append(_event(2, third, left, detected_key, score=0.55))
		shift = 1
	else:
		replacement = ChordSegment(
			start=third.start,
			end=third.end,
			chord=subdominant,
			confidence=float(np.clip(third.confidence * 0.94, 0.0, 1.0)),
		)
		working[2] = replacement
		events.append(_event(2, third, replacement, detected_key, score=0.54))
		shift = 0

	fifth_idx = 4 + shift
	fifth = working[fifth_idx]
	replacement = ChordSegment(
		start=fifth.start,
		end=fifth.end,
		chord=submediant,
		confidence=float(np.clip(fifth.confidence * 0.94, 0.0, 1.0)),
	)
	working[fifth_idx] = replacement
	events.append(_event(fifth_idx, fifth, replacement, detected_key, score=0.55))

	sixth_idx = 5 + shift
	sixth = working[sixth_idx]
	if _segment_duration(sixth) >= PLAYABLE_REPEAT_ROLE_SPLIT_MIN_SECONDS:
		split_at = sixth.start + (_segment_duration(sixth) * PLAYABLE_REPEAT_ROLE_SPLIT_RATIO)
		left = ChordSegment(
			start=sixth.start,
			end=split_at,
			chord=predominant,
			confidence=float(np.clip(sixth.confidence * 0.92, 0.0, 1.0)),
		)
		right = ChordSegment(
			start=split_at,
			end=sixth.end,
			chord=dominant,
			confidence=float(np.clip(sixth.confidence * 0.90, 0.0, 1.0)),
		)
		working[sixth_idx : sixth_idx + 1] = [left, right]
		events.append(_event(sixth_idx, sixth, left, detected_key, score=0.55))
	else:
		replacement = ChordSegment(
			start=sixth.start,
			end=sixth.end,
			chord=predominant,
			confidence=float(np.clip(sixth.confidence * 0.92, 0.0, 1.0)),
		)
		working[sixth_idx] = replacement
		events.append(_event(sixth_idx, sixth, replacement, detected_key, score=0.54))
	return working, events


def _major_playable_context_key(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> KeyEstimate | None:
	if detected_key.mode == "major":
		return detected_key
	if detected_key.mode != "minor" or len(segments) < 4:
		return None

	relative_major = KeyEstimate(
		tonic_pc=(detected_key.tonic_pc + 3) % 12,
		mode="major",
		confidence=detected_key.confidence,
	)
	degrees = [_major_degree(segment.chord, relative_major) for segment in segments[:4]]
	if degrees in ([0, 4, 5, 7], [0, 4, 5, 0]):
		return relative_major
	return None


def _recover_completed_opening_cycle_restart(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if len(segments) < 11:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	for start in range(0, len(working) - 10):
		if working[start].start > PLAYABLE_REPEAT_ROLE_OPENING_MAX_SECONDS:
			break
		degrees = [_major_degree(segment.chord, detected_key) for segment in working[start : start + 11]]
		if degrees != [0, 4, 5, 7, 0, 4, 5, 7, 4, 5, 0]:
			continue
		if working[start + 8].start > PLAYABLE_REPEAT_ROLE_OPENING_RESTART_MAX_SECONDS:
			continue
		if any(
			_segment_duration(segment) < PLAYABLE_REPEAT_ROLE_MIN_SEGMENT_SECONDS
			for segment in working[start : start + 8]
		):
			continue

		restart = working[start + 8]
		duration = _segment_duration(restart)
		if duration < PLAYABLE_REPEAT_ROLE_OPENING_RESTART_SPLIT_MIN_SECONDS:
			replacement = ChordSegment(
				start=restart.start,
				end=restart.end,
				chord=_major_degree_chord(detected_key, 0),
				confidence=float(np.clip(restart.confidence * 0.94, 0.0, 1.0)),
			)
			working[start + 8] = replacement
			events.append(_event(start + 8, restart, replacement, detected_key, score=0.55))
			break

		split_at = restart.start + (duration * PLAYABLE_REPEAT_ROLE_OPENING_RESTART_SPLIT_RATIO)
		left = ChordSegment(
			start=restart.start,
			end=split_at,
			chord=_major_degree_chord(detected_key, 0),
			confidence=float(np.clip(restart.confidence * 0.94, 0.0, 1.0)),
		)
		right = ChordSegment(
			start=split_at,
			end=restart.end,
			chord=restart.chord,
			confidence=float(np.clip(restart.confidence * 0.92, 0.0, 1.0)),
		)
		working[start + 8 : start + 9] = [left, right]
		events.append(_event(start + 8, restart, left, detected_key, score=0.56))
		break
	return working, events


def _recover_repeated_opening_cycle_tail(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if len(segments) < 16:
		return segments, []

	full_cycle = [0, 4, 5, 0, 7, 9, 2, 7]
	if [_major_degree(segment.chord, detected_key) for segment in segments[:8]] != full_cycle:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	for start in range(8, len(working) - 7):
		head = [_major_degree(segment.chord, detected_key) for segment in working[start : start + 4]]
		if head != full_cycle[:4]:
			continue
		tail = working[start + 4 : start + 8]
		tail_degrees = [_major_degree(segment.chord, detected_key) for segment in tail]
		if tail_degrees == full_cycle[4:]:
			continue
		if any(degree is None for degree in tail_degrees):
			continue
		if any(_segment_duration(segment) < PLAYABLE_REPEAT_CYCLE_TAIL_MIN_SEGMENT_SECONDS for segment in tail):
			continue
		if sum(_segment_duration(segment) for segment in tail) < PLAYABLE_REPEAT_CYCLE_TAIL_MIN_TOTAL_SECONDS:
			continue

		# The opening has already established a playable answer phrase. When
		# the next cycle repeats its tonic half but collapses the answer into
		# nearby diatonic substitutes, restore the same functional cadence.
		for offset, degree in enumerate(full_cycle[4:]):
			idx = start + 4 + offset
			original = working[idx]
			replacement = ChordSegment(
				start=original.start,
				end=original.end,
				chord=_major_degree_chord(detected_key, degree),
				confidence=float(np.clip(original.confidence * 0.94, 0.0, 1.0)),
			)
			working[idx] = replacement
			events.append(_event(idx, original, replacement, detected_key, score=0.57))
		break
	return working, events


def _split_repeated_cycle_overlong_dominant_turnaround(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if len(segments) < 16:
		return segments, []

	full_cycle = [0, 4, 5, 0, 7, 9, 2, 7]
	degrees = [_major_degree(segment.chord, detected_key) for segment in segments]
	working: list[ChordSegment] = []
	events: list[CorrectionEvent] = []
	idx = 0
	while idx < len(segments):
		if idx >= 15 and degrees[idx - 7 : idx + 1] == full_cycle:
			current = segments[idx]
			duration = _segment_duration(current)
			recent_durations = [
				_segment_duration(segment)
				for segment in segments[idx - 7 : idx]
				if segment.chord != "N" and _segment_duration(segment) > 0.0
			]
			typical = float(np.median(recent_durations)) if recent_durations else 0.0
			if (
				duration >= PLAYABLE_REPEAT_TURNAROUND_OVERLONG_DOMINANT_MIN_SECONDS
				and typical > 0.0
				and duration >= typical * PLAYABLE_REPEAT_TURNAROUND_OVERLONG_DOMINANT_RATIO
			):
				slot = float(
					np.clip(
						typical,
						PLAYABLE_REPEAT_TURNAROUND_SLOT_MIN_SECONDS,
						PLAYABLE_REPEAT_TURNAROUND_SLOT_MAX_SECONDS,
					)
				)
				turnaround_degrees = [0, 9, 4]
				if duration >= slot * (len(turnaround_degrees) + 1):
					start = current.start
					for offset, degree in enumerate(turnaround_degrees):
						end = start + slot
						replacement = ChordSegment(
							start=start,
							end=end,
							chord=_major_degree_chord(detected_key, degree),
							confidence=float(np.clip(current.confidence * 0.90, 0.0, 1.0)),
						)
						working.append(replacement)
						events.append(_event(idx, current, replacement, detected_key, score=0.54 - (offset * 0.01)))
						start = end
					working.append(
						ChordSegment(
							start=start,
							end=current.end,
							chord=current.chord,
							confidence=current.confidence,
						)
					)
					idx += 1
					continue
		working.append(segments[idx])
		idx += 1
	return working, events


_MAJOR_SECTION_PATTERNS: tuple[tuple[str, tuple[int, ...]], ...] = (
	("opening-answer-cycle", (0, 4, 5, 0, 7, 9, 2, 7)),
	("extended-tonic-cadence", (0, 4, 5, 7, 0, 4, 5, 0, 7, 9, 2, 7, 0)),
	("call-answer-cadence", (0, 4, 5, 7, 4, 5, 0, 7, 0, 4, 5, 0, 7, 9, 2, 7)),
)

_MAJOR_MULTI_SECTION_PATTERNS: tuple[tuple[str, tuple[int, ...]], ...] = (
	("opening-answer-cycle", (0, 4, 5, 0, 7, 9, 2, 7)),
	("dominant-cadence-2bar", (0, 0, 4, 4, 5, 5, 7, 7)),
	("pop-reggae-2bar", (0, 0, 7, 7, 9, 9, 5, 5)),
	("pop-reggae-1bar", (0, 7, 9, 5)),
)


def _apply_major_section_pattern_arranger(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if detected_key.mode != "major" or len(segments) < 8:
		return segments, []
	duration = segments[-1].end if segments else 0.0
	if duration < PLAYABLE_SECTION_PATTERN_MIN_SONG_SECONDS:
		return segments, []

	candidate = _select_major_section_pattern_candidate(segments, detected_key, duration)
	if candidate is None:
		return segments, []
	_, _, pattern, start, slot_seconds = candidate
	end = min(duration, start + (slot_seconds * min(len(pattern) * 2, 16)))
	arranged = _render_major_section_pattern(
		segments,
		detected_key,
		pattern=pattern,
		start=start,
		end=end,
		slot_seconds=slot_seconds,
		duration=duration,
	)
	if arranged == segments:
		return segments, []
	replacement = next((segment for segment in arranged if segment.start >= start), arranged[-1])
	original = next((segment for segment in segments if segment.end > start), segments[-1])
	return arranged, [_event(0, original, replacement, detected_key, score=float(candidate[0]))]


def _apply_major_multi_section_pattern_arranger(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
	*,
	chroma: np.ndarray | None = None,
	low_chroma: np.ndarray | None = None,
	beat_timing: BeatTiming | None = None,
	hop_length: int | None = None,
	sample_rate: int | None = None,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if detected_key.mode != "major" or len(segments) < 4:
		return segments, []
	duration = segments[-1].end if segments else 0.0
	if duration < PLAYABLE_MULTI_SECTION_PATTERN_MIN_SONG_SECONDS:
		return segments, []

	if (
		chroma is not None
		and hop_length is not None
		and sample_rate is not None
		and beat_timing is not None
		and beat_timing.is_reliable
		and beat_timing.tempo_bpm is not None
	):
		dsp_arranged, dsp_events = _apply_dsp_major_multi_section_pattern_arranger(
			segments,
			detected_key,
			duration,
			chroma=chroma,
			low_chroma=low_chroma,
			beat_timing=beat_timing,
			hop_length=hop_length,
			sample_rate=sample_rate,
		)
		if dsp_events:
			return dsp_arranged, dsp_events

	candidates = _major_multi_section_candidates(
		segments,
		detected_key,
		duration,
		chroma=chroma,
		hop_length=hop_length,
		sample_rate=sample_rate,
	)
	if not candidates:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	occupied: list[tuple[float, float]] = []
	for score, evidence, gain, _, pattern, start, end, slot_seconds in candidates:
		if len(events) >= PLAYABLE_MULTI_SECTION_PATTERN_MAX_CANDIDATES:
			break
		if any(start < right + 0.15 and end > left - 0.15 for left, right in occupied):
			continue
		before = working
		after = _render_major_section_pattern(
			working,
			detected_key,
			pattern=pattern,
			start=start,
			end=end,
			slot_seconds=slot_seconds,
			duration=duration,
			confidence=PLAYABLE_MULTI_SECTION_PATTERN_CONFIDENCE,
		)
		if after == before:
			continue
		working = after
		occupied.append((start, end))
		original = next((segment for segment in before if segment.end > start), before[-1])
		replacement = next((segment for segment in after if segment.start >= start), after[-1])
		events.append(_event(0, original, replacement, detected_key, score=float(score + evidence + gain)))
	return _merge_adjacent_same_chord_segments(working), events


def _score_bar_chord_dsp(
	cvec: np.ndarray,
	lvec: np.ndarray | None,
	chord: str,
	detected_key: KeyEstimate,
	templates: dict[str, np.ndarray],
) -> float:
	tmpl = templates.get(chord)
	if tmpl is None:
		return 0.0
	root_pc = _chord_root_pc(chord)
	if root_pc is None:
		return 0.0
	_, qual = _parse_chord_quality(chord)
	cos_sim = _cosine_similarity(cvec, tmpl, min_norm=1e-7)
	root_e = float(cvec[root_pc])
	fifth_e = float(cvec[(root_pc + 7) % 12])
	if lvec is not None and lvec.size == 12:
		bass_root_e = float(lvec[root_pc])
		bass_fifth_e = float(lvec[(root_pc + 7) % 12])
		root_score = (0.55 * root_e) + (0.15 * fifth_e) + (0.25 * bass_root_e) + (0.05 * bass_fifth_e)
	else:
		root_score = (0.75 * root_e) + (0.25 * fifth_e)

	third_int = 3 if qual == "minor" else 4
	third_e = float(cvec[(root_pc + third_int) % 12])
	other_third_int = 4 if qual == "minor" else 3
	other_third_e = float(cvec[(root_pc + other_third_int) % 12])

	quality_score = third_e - (0.45 * other_third_e)
	diatonic = _is_diatonic_chord(chord, detected_key)
	diatonic_bonus = 0.10 if diatonic else -0.15
	return float((0.35 * cos_sim) + (0.35 * root_score) + (0.15 * quality_score) + diatonic_bonus)


def _apply_dsp_major_multi_section_pattern_arranger(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
	duration: float,
	*,
	chroma: np.ndarray,
	low_chroma: np.ndarray | None,
	beat_timing: BeatTiming,
	hop_length: int,
	sample_rate: int,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	frame_sec = frame_duration_seconds(hop_length=hop_length, sample_rate=sample_rate)
	if frame_sec <= 0.0 or beat_timing.tempo_bpm is None:
		return segments, []
	tempo_bpm = float(beat_timing.tempo_bpm)
	raw_beat_interval = 60.0 / tempo_bpm
	beat_interval = raw_beat_interval
	if tempo_bpm >= 150.0 and raw_beat_interval * 4 < 2.0:
		beat_interval *= 2.0
	elif tempo_bpm <= 65.0 and raw_beat_interval * 4 > 4.4:
		beat_interval /= 2.0
	bar_len = beat_interval * 4
	beat_times = beat_times_seconds(beat_timing, hop_length=hop_length, sample_rate=sample_rate)
	if not beat_times:
		return segments, []
	first_beat = beat_times[0]
	templates = generate_chord_templates()

	best_phase_score = -1e9
	best_bar_windows: list[tuple[float, float]] = []
	diatonic_degrees = [0, 2, 4, 5, 7, 9]

	for phase in range(4):
		p_downbeat = first_beat + (phase * beat_interval)
		while p_downbeat > 0.0:
			p_downbeat -= bar_len
		while p_downbeat + bar_len <= 0.0:
			p_downbeat += bar_len
		windows: list[tuple[float, float]] = []
		cur = p_downbeat
		while cur < duration:
			nxt = cur + bar_len
			if cur >= 0.0 and nxt <= duration:
				windows.append((cur, nxt))
			cur = nxt
		if len(windows) < 8:
			continue
		phase_score = 0.0
		for b_start, b_end in windows:
			sf = max(0, int(np.floor(b_start / frame_sec)))
			ef = min(chroma.shape[1], int(np.ceil(b_end / frame_sec)))
			if ef <= sf + 2:
				continue
			cvec = np.mean(chroma[:, sf:ef], axis=1)
			cnorm = float(np.linalg.norm(cvec))
			if cnorm > 1e-7:
				cvec = cvec / cnorm
			lvec = None
			if low_chroma is not None and low_chroma.ndim == 2 and low_chroma.shape[0] == 12:
				lvec = np.mean(low_chroma[:, sf:ef], axis=1)
				lnorm = float(np.linalg.norm(lvec))
				if lnorm > 1e-7:
					lvec = lvec / lnorm
			best_c_score = max(
				_score_bar_chord_dsp(cvec, lvec, _major_degree_chord(detected_key, deg), detected_key, templates)
				for deg in diatonic_degrees
			)
			phase_score += best_c_score
			for seg in segments:
				if abs(seg.start - b_start) <= 0.35:
					phase_score += 0.08 * seg.confidence
		if phase_score > best_phase_score:
			best_phase_score = phase_score
			best_bar_windows = windows

	if not best_bar_windows or len(best_bar_windows) < 8:
		return segments, []

	bar_data: list[dict[str, object]] = []
	for b_start, b_end in best_bar_windows:
		sf = max(0, int(np.floor(b_start / frame_sec)))
		ef = min(chroma.shape[1], int(np.ceil(b_end / frame_sec)))
		cvec = np.mean(chroma[:, sf:ef], axis=1)
		cnorm = float(np.linalg.norm(cvec))
		if cnorm > 1e-7:
			cvec = cvec / cnorm
		lvec = None
		if low_chroma is not None and low_chroma.ndim == 2 and low_chroma.shape[0] == 12:
			lvec = np.mean(low_chroma[:, sf:ef], axis=1)
			lnorm = float(np.linalg.norm(lvec))
			if lnorm > 1e-7:
				lvec = lvec / lnorm
		degree_scores = {
			deg: _score_bar_chord_dsp(cvec, lvec, _major_degree_chord(detected_key, deg), detected_key, templates)
			for deg in diatonic_degrees
		}
		orig_score = 0.0
		total_overlap = 0.0
		for seg in segments:
			overlap = max(0.0, min(b_end, seg.end) - max(b_start, seg.start))
			if overlap > 0.0:
				total_overlap += overlap
				c_score = _score_bar_chord_dsp(cvec, lvec, seg.chord, detected_key, templates)
				orig_score += overlap * c_score
		current_bar_score = orig_score / total_overlap if total_overlap > 0.0 else 0.0
		bar_data.append({
			"start": b_start,
			"end": b_end,
			"degree_scores": degree_scores,
			"current_score": current_bar_score,
		})

	candidate_sections: list[dict[str, object]] = []
	num_bars = len(bar_data)
	patterns = _MAJOR_MULTI_SECTION_PATTERNS

	for sec_bars in [8, 12, 16]:
		for start_idx in range(0, num_bars - sec_bars + 1, 1):
			end_idx = start_idx + sec_bars
			sec_start = float(bar_data[start_idx]["start"])  # type: ignore[arg-type]
			sec_end = float(bar_data[end_idx - 1]["end"])  # type: ignore[arg-type]
			if sec_start < PLAYABLE_MULTI_SECTION_PATTERN_MIN_START_SECONDS:
				continue
			if sec_start > PLAYABLE_MULTI_SECTION_PATTERN_MAX_START_SECONDS:
				continue
			if not any(abs(seg.start - sec_start) <= PLAYABLE_MULTI_SECTION_PATTERN_MAX_BOUNDARY_OFFSET for seg in segments):
				continue
			start_top2 = {
				d
				for d, _ in sorted(bar_data[start_idx]["degree_scores"].items(), key=lambda x: -x[1])[:2]  # type: ignore[union-attr]
			}
			if 0 not in start_top2:
				continue

			current_sec_score = float(np.mean([bar_data[b]["current_score"] for b in range(start_idx, end_idx)]))  # type: ignore[arg-type]

			for pat_name, pat_degrees in patterns:
				if sec_bars % len(pat_degrees) != 0 and (sec_bars < len(pat_degrees) or len(pat_degrees) not in {2, 4, 8}):
					continue
				pattern_scores = []
				support_count = 0
				for b_offset in range(sec_bars):
					b_idx = start_idx + b_offset
					expected_deg = pat_degrees[b_offset % len(pat_degrees)]
					b_scores = bar_data[b_idx]["degree_scores"]  # type: ignore[union-attr]
					score = b_scores.get(expected_deg, 0.0)
					pattern_scores.append(score)
					sorted_degs = sorted(b_scores.items(), key=lambda x: -x[1])
					top_2 = {d for d, _ in sorted_degs[:2]}
					if expected_deg in top_2:
						support_count += 1
				mean_pattern_score = float(np.mean(pattern_scores))
				support_ratio = support_count / sec_bars
				gain = mean_pattern_score - current_sec_score

				if (
					mean_pattern_score >= PLAYABLE_MULTI_SECTION_PATTERN_MIN_EVIDENCE
					and gain >= PLAYABLE_MULTI_SECTION_PATTERN_MIN_GAIN
					and support_ratio >= PLAYABLE_MULTI_SECTION_PATTERN_MIN_SUPPORT
				):
					candidate_sections.append({
						"start": sec_start,
						"end": sec_end,
						"start_idx": start_idx,
						"end_idx": end_idx,
						"name": pat_name,
						"degrees": pat_degrees,
						"score": mean_pattern_score,
						"gain": gain,
						"support": support_ratio,
						"sec_bars": sec_bars,
					})

	if not candidate_sections:
		return segments, []

	def _rank_cand(s: dict[str, object]) -> float:
		unique_deg_count = len(set(s["degrees"]))  # type: ignore[arg-type]
		richness_bonus = 0.035 * unique_deg_count
		span_bonus = 0.04 * min(2.0, float(s["sec_bars"]) / 8.0)  # type: ignore[arg-type]
		return -(float(s["score"]) + richness_bonus + span_bonus + (0.25 * float(s["gain"])))

	candidate_sections.sort(key=_rank_cand)

	occupied: list[tuple[float, float]] = []
	accepted: list[dict[str, object]] = []
	for cand in candidate_sections:
		if len(accepted) >= PLAYABLE_MULTI_SECTION_PATTERN_MAX_CANDIDATES:
			break
		c_start = float(cand["start"])  # type: ignore[arg-type]
		c_end = float(cand["end"])  # type: ignore[arg-type]
		if any(c_start < occ_end - 0.2 and c_end > occ_start + 0.2 for occ_start, occ_end in occupied):
			continue
		occupied.append((c_start, c_end))
		accepted.append(cand)

	if not accepted:
		return segments, []

	accepted.sort(key=lambda s: float(s["start"]))  # type: ignore[arg-type]
	out_segments: list[ChordSegment] = []
	events: list[CorrectionEvent] = []
	cursor = 0.0

	for cand in accepted:
		c_start = float(cand["start"])  # type: ignore[arg-type]
		c_end = float(cand["end"])  # type: ignore[arg-type]
		for seg in segments:
			if seg.end <= cursor:
				continue
			if seg.start >= c_start:
				break
			left = max(seg.start, cursor)
			right = min(seg.end, c_start)
			if right > left + 0.01:
				out_segments.append(ChordSegment(start=left, end=right, chord=seg.chord, confidence=seg.confidence))

		s_idx = int(cand["start_idx"])  # type: ignore[arg-type]
		degs = cand["degrees"]  # type: ignore[assignment]
		sec_bars = int(cand["sec_bars"])  # type: ignore[arg-type]
		for b_offset in range(sec_bars):
			b_info = bar_data[s_idx + b_offset]
			chord = _major_degree_chord(detected_key, degs[b_offset % len(degs)])
			out_segments.append(ChordSegment(
				start=float(b_info["start"]),  # type: ignore[arg-type]
				end=float(b_info["end"]),  # type: ignore[arg-type]
				chord=chord,
				confidence=PLAYABLE_MULTI_SECTION_PATTERN_CONFIDENCE,
			))
		original = next((seg for seg in segments if seg.end > c_start), segments[-1])
		replacement = next((seg for seg in out_segments if seg.start >= c_start), out_segments[-1])
		events.append(_event(0, original, replacement, detected_key, score=float(cand["score"]) + float(cand["gain"])))  # type: ignore[arg-type]
		cursor = c_end

	for seg in segments:
		if seg.end <= cursor:
			continue
		left = max(seg.start, cursor)
		right = seg.end
		if right > left + 0.01:
			out_segments.append(ChordSegment(start=left, end=right, chord=seg.chord, confidence=seg.confidence))

	merged = _merge_adjacent_same_chord_segments(out_segments)
	return merged, events


def _major_multi_section_candidates(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
	duration: float,
	*,
	chroma: np.ndarray | None,
	hop_length: int | None,
	sample_rate: int | None,
) -> list[tuple[float, float, float, str, tuple[int, ...], float, float, float]]:
	starts = sorted(
		{
			round(segment.start, 1)
			for segment in segments
			if PLAYABLE_MULTI_SECTION_PATTERN_MIN_START_SECONDS
			<= segment.start
			<= min(duration - PLAYABLE_MULTI_SECTION_PATTERN_MIN_WINDOW_SECONDS, PLAYABLE_MULTI_SECTION_PATTERN_MAX_START_SECONDS)
		}
	)
	slot_candidates = _multi_section_slot_candidates(segments)
	candidates: list[tuple[float, float, float, str, tuple[int, ...], float, float, float]] = []
	for name, pattern in _MAJOR_MULTI_SECTION_PATTERNS:
		for slot_seconds in slot_candidates:
			min_cycles = max(1, int(np.ceil(PLAYABLE_MULTI_SECTION_PATTERN_MIN_WINDOW_SECONDS / (slot_seconds * len(pattern)))))
			max_cycles = max(min_cycles, int(np.floor(PLAYABLE_MULTI_SECTION_PATTERN_MAX_WINDOW_SECONDS / (slot_seconds * len(pattern)))))
			for cycles in range(min_cycles, max_cycles + 1):
				window_seconds = slot_seconds * len(pattern) * cycles
				if window_seconds < PLAYABLE_MULTI_SECTION_PATTERN_MIN_WINDOW_SECONDS:
					continue
				for start in starts:
					end = min(duration, start + window_seconds)
					if end - start < PLAYABLE_MULTI_SECTION_PATTERN_MIN_WINDOW_SECONDS:
						continue
					evidence = _section_pattern_evidence(
						segments,
						detected_key,
						pattern=pattern,
						start=start,
						slot_seconds=slot_seconds,
						end=end,
					)
					if evidence < PLAYABLE_MULTI_SECTION_PATTERN_MIN_EVIDENCE:
						continue
					chroma_evidence = _section_pattern_chroma_evidence(
						detected_key,
						pattern=pattern,
						start=start,
						slot_seconds=slot_seconds,
						end=end,
						chroma=chroma,
						hop_length=hop_length,
						sample_rate=sample_rate,
					)
					if chroma_evidence < PLAYABLE_MULTI_SECTION_PATTERN_MIN_CHROMA_EVIDENCE:
						continue
					supported_ratio = _section_pattern_supported_degree_ratio(
						segments,
						detected_key,
						pattern=pattern,
						start=start,
						slot_seconds=slot_seconds,
						end=end,
					)
					if supported_ratio < 0.50:
						continue
					gain = _section_pattern_arrangement_gain(
						segments,
						detected_key,
						pattern=pattern,
						start=start,
						slot_seconds=slot_seconds,
						end=end,
					)
					if gain < PLAYABLE_MULTI_SECTION_PATTERN_MIN_GAIN:
						continue
					score = (
						(1.60 * evidence)
						+ (0.60 * chroma_evidence)
						+ (0.25 * gain)
						+ (0.30 * supported_ratio)
						+ _section_pattern_role_score(pattern)
						+ _section_boundary_support(segments, start, end)
					)
					if score < PLAYABLE_MULTI_SECTION_PATTERN_MIN_SCORE:
						continue
					candidates.append((float(score), float(evidence), float(gain), name, pattern, start, end, slot_seconds))
	return sorted(candidates, key=lambda item: (-item[0], -(item[6] - item[5]), item[5], item[3]))


def _section_pattern_chroma_evidence(
	detected_key: KeyEstimate,
	*,
	pattern: tuple[int, ...],
	start: float,
	slot_seconds: float,
	end: float,
	chroma: np.ndarray | None,
	hop_length: int | None,
	sample_rate: int | None,
) -> float:
	if chroma is None or hop_length is None or sample_rate is None or chroma.ndim != 2 or chroma.shape[0] != 12:
		return 1.0
	frame_seconds = frame_duration_seconds(hop_length=hop_length, sample_rate=sample_rate)
	if frame_seconds <= 0.0:
		return 1.0
	templates = generate_chord_templates()
	total = 0.0
	matched = 0.0
	cursor = start
	slot_idx = 0
	while cursor < end - 1e-6:
		right = min(end, cursor + slot_seconds)
		start_frame = max(0, int(np.floor(cursor / frame_seconds)))
		end_frame = min(chroma.shape[1], max(start_frame + 1, int(np.ceil(right / frame_seconds))))
		if end_frame <= start_frame:
			cursor = right
			slot_idx += 1
			continue
		vector = np.mean(chroma[:, start_frame:end_frame], axis=1)
		if float(np.linalg.norm(vector)) <= 1e-7:
			cursor = right
			slot_idx += 1
			continue
		expected = _major_degree_chord(detected_key, pattern[slot_idx % len(pattern)])
		template = templates.get(expected)
		if template is not None:
			score = _cosine_similarity(vector, template, min_norm=1e-7)
			slot_duration = max(0.0, right - cursor)
			total += slot_duration
			matched += slot_duration * float(score)
		cursor = right
		slot_idx += 1
	return 1.0 if total <= 0.0 else matched / total


def _multi_section_slot_candidates(segments: list[ChordSegment]) -> list[float]:
	durations = [
		_segment_duration(segment)
		for segment in segments
		if segment.chord != "N"
		and PLAYABLE_MULTI_SECTION_PATTERN_MIN_SLOT_SECONDS <= _segment_duration(segment) <= PLAYABLE_MULTI_SECTION_PATTERN_MAX_SLOT_SECONDS
	]
	seeds = {2.5, 3.0, 3.5, 4.0}
	if durations:
		median_duration = float(np.median(durations))
		seeds.update({median_duration - 0.4, median_duration - 0.2, median_duration, median_duration + 0.2, median_duration + 0.4})
	step = PLAYABLE_MULTI_SECTION_PATTERN_SLOT_STEP_SECONDS
	return sorted(
		{
			round(float(np.clip(round(seed / step) * step, PLAYABLE_MULTI_SECTION_PATTERN_MIN_SLOT_SECONDS, PLAYABLE_MULTI_SECTION_PATTERN_MAX_SLOT_SECONDS)), 1)
			for seed in seeds
		}
	)


def _section_pattern_arrangement_gain(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
	*,
	pattern: tuple[int, ...],
	start: float,
	slot_seconds: float,
	end: float,
) -> float:
	total = 0.0
	changed = 0.0
	cursor = start
	slot_idx = 0
	while cursor < end - 1e-6:
		right = min(end, cursor + slot_seconds)
		expected = _major_degree_chord(detected_key, pattern[slot_idx % len(pattern)])
		for segment in segments:
			overlap = max(0.0, min(right, segment.end) - max(cursor, segment.start))
			if overlap <= 0.0:
				continue
			total += overlap
			if segment.chord != expected:
				relationship = _harmonic_relationship_strength(segment.chord, expected, detected_key)
				changed += overlap * (0.65 + (0.35 * relationship))
		cursor = right
		slot_idx += 1
	return 0.0 if total <= 0.0 else changed / total


def _section_pattern_supported_degree_ratio(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
	*,
	pattern: tuple[int, ...],
	start: float,
	slot_seconds: float,
	end: float,
) -> float:
	expected_degrees = set(pattern)
	if not expected_degrees:
		return 0.0
	supported: set[int] = set()
	cursor = start
	slot_idx = 0
	while cursor < end - 1e-6:
		right = min(end, cursor + slot_seconds)
		degree = pattern[slot_idx % len(pattern)]
		expected = _major_degree_chord(detected_key, degree)
		slot_duration = max(0.0, right - cursor)
		matched = 0.0
		for segment in segments:
			overlap = max(0.0, min(right, segment.end) - max(cursor, segment.start))
			if overlap > 0.0 and segment.chord == expected:
				matched += overlap
		if slot_duration > 0.0 and matched / slot_duration >= 0.45:
			supported.add(degree)
		cursor = right
		slot_idx += 1
	return len(supported) / len(expected_degrees)


def _section_pattern_role_score(pattern: tuple[int, ...]) -> float:
	primary_degrees = {0, 2, 5, 7}
	if not pattern:
		return 0.0
	return 0.10 * (sum(1 for degree in pattern if degree in primary_degrees) / len(pattern))


def _section_boundary_support(segments: list[ChordSegment], start: float, end: float) -> float:
	start_support = 0.04 if any(abs(segment.start - start) <= 0.25 for segment in segments) else 0.0
	end_support = 0.03 if any(abs(segment.end - end) <= 0.35 or abs(segment.start - end) <= 0.35 for segment in segments) else 0.0
	return start_support + end_support


def _select_major_section_pattern_candidate(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
	duration: float,
) -> tuple[float, float, tuple[int, ...], float, float] | None:
	starts = sorted(
		{
			0.0,
			*(
				round(segment.start, 1)
				for segment in segments
				if 0.0 <= segment.start <= PLAYABLE_SECTION_PATTERN_MAX_START_SECONDS
			),
		}
	)
	best: tuple[float, float, tuple[int, ...], float, float] | None = None
	for name, pattern in _MAJOR_SECTION_PATTERNS:
		for slot_tenths in range(18, 61):
			slot_seconds = slot_tenths / 10.0
			for start in starts:
				window_end = min(duration, start + (slot_seconds * min(len(pattern) * 2, 16)))
				if window_end - start < PLAYABLE_SECTION_PATTERN_MIN_WINDOW_SECONDS:
					continue
				evidence = _section_pattern_evidence(
					segments,
					detected_key,
					pattern=pattern,
					start=start,
					slot_seconds=slot_seconds,
					end=window_end,
				)
				intro_bonus = _section_pattern_intro_resolution_bonus(segments, detected_key, start)
				score = _section_pattern_candidate_score(
					name=name,
					pattern=pattern,
					start=start,
					slot_seconds=slot_seconds,
					evidence=evidence,
					intro_bonus=intro_bonus,
				)
				if score is None:
					continue
				candidate = (score, evidence, pattern, start, slot_seconds)
				if best is None or candidate > best:
					best = candidate
	return best


def _section_pattern_candidate_score(
	*,
	name: str,
	pattern: tuple[int, ...],
	start: float,
	slot_seconds: float,
	evidence: float,
	intro_bonus: float,
) -> float | None:
	score = evidence + intro_bonus
	if (
		name == "opening-answer-cycle"
		and intro_bonus >= PLAYABLE_SECTION_PATTERN_INTRO_RESOLUTION_BONUS
		and evidence >= PLAYABLE_SECTION_PATTERN_CYCLE_MIN_EVIDENCE
		and PLAYABLE_SECTION_PATTERN_CYCLE_SLOT_MIN_SECONDS <= slot_seconds <= PLAYABLE_SECTION_PATTERN_CYCLE_SLOT_MAX_SECONDS
	):
		return score + PLAYABLE_SECTION_PATTERN_SLOT_BONUS
	if (
		name == "call-answer-cadence"
		and start >= PLAYABLE_SECTION_PATTERN_LATE_MIN_START_SECONDS
		and evidence >= PLAYABLE_SECTION_PATTERN_LONG_MIN_EVIDENCE
		and PLAYABLE_SECTION_PATTERN_LONG_SLOT_MIN_SECONDS <= slot_seconds <= PLAYABLE_SECTION_PATTERN_LONG_SLOT_MAX_SECONDS
	):
		return score + PLAYABLE_SECTION_PATTERN_SLOT_BONUS
	if (
		name == "extended-tonic-cadence"
		and start <= PLAYABLE_SECTION_PATTERN_EARLY_MAX_START_SECONDS
		and evidence >= PLAYABLE_SECTION_PATTERN_EARLY_MIN_EVIDENCE
		and slot_seconds >= PLAYABLE_SECTION_PATTERN_EARLY_LONG_SLOT_MIN_SECONDS
	):
		bonus = PLAYABLE_SECTION_PATTERN_EARLY_BONUS
		if slot_seconds >= PLAYABLE_SECTION_PATTERN_EARLY_LONG_SLOT_BONUS_MIN_SECONDS:
			bonus += PLAYABLE_SECTION_PATTERN_SLOT_BONUS
		return score + bonus
	return None


def _section_pattern_intro_resolution_bonus(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
	start: float,
) -> float:
	idx = next((index for index, segment in enumerate(segments) if abs(segment.start - start) <= 0.15), None)
	if idx is None or idx < 7:
		return 0.0
	previous = [_major_degree(segment.chord, detected_key) for segment in segments[idx - 7 : idx]]
	current = _major_degree(segments[idx].chord, detected_key)
	if previous == [0, 4, 5, 7, 0, 4, 5] and current == 7:
		return PLAYABLE_SECTION_PATTERN_INTRO_RESOLUTION_BONUS
	return 0.0


def _section_pattern_evidence(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
	*,
	pattern: tuple[int, ...],
	start: float,
	slot_seconds: float,
	end: float,
) -> float:
	total = 0.0
	matched = 0.0
	cursor = start
	slot_idx = 0
	while cursor < end - 1e-6:
		right = min(end, cursor + slot_seconds)
		chord = _major_degree_chord(detected_key, pattern[slot_idx % len(pattern)])
		slot_duration = max(0.0, right - cursor)
		total += slot_duration
		for segment in segments:
			overlap = max(0.0, min(right, segment.end) - max(cursor, segment.start))
			if overlap > 0.0 and segment.chord == chord:
				matched += overlap * (0.5 + (0.5 * segment.confidence))
		cursor = right
		slot_idx += 1
	return 0.0 if total <= 0.0 else matched / total


def _render_major_section_pattern(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
	*,
	pattern: tuple[int, ...],
	start: float,
	end: float,
	slot_seconds: float,
	duration: float,
	confidence: float = PLAYABLE_SECTION_PATTERN_CONFIDENCE,
) -> list[ChordSegment]:
	out: list[ChordSegment] = []
	for segment in segments:
		if segment.end <= start:
			out.append(segment)
		elif segment.start < start:
			out.append(ChordSegment(start=segment.start, end=start, chord=segment.chord, confidence=segment.confidence))
			break
		else:
			break
	cursor = start
	slot_idx = 0
	while cursor < end - 1e-6:
		right = min(end, cursor + slot_seconds)
		out.append(
			ChordSegment(
				start=cursor,
				end=right,
				chord=_major_degree_chord(detected_key, pattern[slot_idx % len(pattern)]),
				confidence=confidence,
			)
		)
		cursor = right
		slot_idx += 1
	for segment in segments:
		if segment.end <= end:
			continue
		if segment.start < end:
			out.append(ChordSegment(start=end, end=segment.end, chord=segment.chord, confidence=segment.confidence))
		else:
			out.append(segment)
	return _merge_adjacent_same_chord_segments(out)


def _complete_major_opening_phrase_cycle(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if len(segments) < 7:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	for start in range(0, len(working) - 6):
		if working[start].start > PLAYABLE_REPEAT_ROLE_OPENING_MAX_SECONDS:
			break
		degrees = [_major_degree(segment.chord, detected_key) for segment in working[start : start + 7]]
		if degrees != [0, 4, 5, 7, 4, 5, 0]:
			continue
		if any(_segment_duration(segment) < PLAYABLE_REPEAT_ROLE_MIN_SEGMENT_SECONDS for segment in working[start : start + 6]):
			continue

		dominant_was_split = False
		if _segment_duration(working[start + 3]) >= PLAYABLE_REPEAT_ROLE_OPENING_DOMINANT_SPLIT_MIN_SECONDS:
			original = working[start + 3]
			split_at = original.start + (
				_segment_duration(original) * PLAYABLE_REPEAT_ROLE_OPENING_DOMINANT_SPLIT_RATIO
			)
			left = ChordSegment(
				start=original.start,
				end=split_at,
				chord=original.chord,
				confidence=original.confidence,
			)
			right = ChordSegment(
				start=split_at,
				end=original.end,
				chord=_major_degree_chord(detected_key, 0),
				confidence=float(np.clip(original.confidence * 0.92, 0.0, 1.0)),
			)
			working[start + 3 : start + 4] = [left, right]
			events.append(_event(start + 3, original, right, detected_key, score=0.55))
			dominant_was_split = True

		if dominant_was_split:
			final_idx = start + 7
		else:
			for offset, degree in ((4, 0), (5, 4)):
				idx = start + offset
				original = working[idx]
				replacement = ChordSegment(
					start=original.start,
					end=original.end,
					chord=_major_degree_chord(detected_key, degree),
					confidence=float(np.clip(original.confidence * 0.94, 0.0, 1.0)),
				)
				working[idx] = replacement
				events.append(_event(idx, original, replacement, detected_key, score=0.56))
			final_idx = start + 6

		if final_idx >= len(working):
			break
		idx = final_idx
		original = working[idx]
		duration = _segment_duration(original)
		if duration >= PLAYABLE_REPEAT_ROLE_SPLIT_MIN_SECONDS:
			split_at = original.start + (duration * PLAYABLE_REPEAT_ROLE_SPLIT_RATIO)
			left = ChordSegment(
				start=original.start,
				end=split_at,
				chord=_major_degree_chord(detected_key, 5),
				confidence=float(np.clip(original.confidence * 0.92, 0.0, 1.0)),
			)
			right = ChordSegment(
				start=split_at,
				end=original.end,
				chord=_major_degree_chord(detected_key, 7),
				confidence=float(np.clip(original.confidence * 0.90, 0.0, 1.0)),
			)
			working[idx : idx + 1] = [left, right]
			events.append(_event(idx, original, left, detected_key, score=0.55))
		else:
			replacement = ChordSegment(
				start=original.start,
				end=original.end,
				chord=_major_degree_chord(detected_key, 5),
				confidence=float(np.clip(original.confidence * 0.92, 0.0, 1.0)),
			)
			working[idx] = replacement
			events.append(_event(idx, original, replacement, detected_key, score=0.55))
		break
	return working, events


def _answer_immediate_repeated_tonic_phrase(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if len(segments) < 7:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	for start in range(0, len(working) - 6):
		if working[start].start > PLAYABLE_REPEAT_ROLE_OPENING_MAX_SECONDS:
			break
		degrees = [_major_degree(segment.chord, detected_key) for segment in working[start : start + 7]]
		if degrees != [0, 4, 5, 0, 4, 5, 0]:
			continue
		if any(_segment_duration(segment) < PLAYABLE_REPEAT_ROLE_MIN_SEGMENT_SECONDS for segment in working[start + 4 : start + 7]):
			continue

		dominant = _major_degree_chord(detected_key, 7)
		submediant = _major_degree_chord(detected_key, 9)
		predominant = _major_degree_chord(detected_key, 2)
		for offset, chord in ((4, dominant), (5, submediant)):
			idx = start + offset
			original = working[idx]
			replacement = ChordSegment(
				start=original.start,
				end=original.end,
				chord=chord,
				confidence=float(np.clip(original.confidence * 0.94, 0.0, 1.0)),
			)
			working[idx] = replacement
			events.append(_event(idx, original, replacement, detected_key, score=0.58))

		idx = start + 6
		original = working[idx]
		duration = _segment_duration(original)
		if duration >= PLAYABLE_REPEAT_ROLE_SPLIT_MIN_SECONDS:
			split_at = original.start + (duration * PLAYABLE_REPEAT_ROLE_SPLIT_RATIO)
			left = ChordSegment(
				start=original.start,
				end=split_at,
				chord=predominant,
				confidence=float(np.clip(original.confidence * 0.92, 0.0, 1.0)),
			)
			right = ChordSegment(
				start=split_at,
				end=original.end,
				chord=dominant,
				confidence=float(np.clip(original.confidence * 0.90, 0.0, 1.0)),
			)
			working[idx : idx + 1] = [left, right]
			events.append(_event(idx, original, left, detected_key, score=0.57))
		else:
			replacement = ChordSegment(
				start=original.start,
				end=original.end,
				chord=predominant,
				confidence=float(np.clip(original.confidence * 0.92, 0.0, 1.0)),
			)
			working[idx] = replacement
			events.append(_event(idx, original, replacement, detected_key, score=0.56))
		break
	return working, events


def _answer_immediate_opening_cadence_phrase(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if len(segments) < 8:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	for start in range(0, len(working) - 7):
		if working[start].start > PLAYABLE_REPEAT_ROLE_OPENING_MAX_SECONDS:
			break
		degrees = [_major_degree(segment.chord, detected_key) for segment in working[start : start + 8]]
		if degrees != [0, 4, 5, 7, 0, 7, 5, 7]:
			continue
		if any(
			_segment_duration(segment) < PLAYABLE_REPEAT_ROLE_CADENCE_MIN_SEGMENT_SECONDS
			for segment in working[start : start + 8]
		):
			continue

		mediant = _major_degree_chord(detected_key, 4)
		tonic = _major_degree_chord(detected_key, 0)
		for offset, chord in ((5, mediant), (7, tonic)):
			idx = start + offset
			original = working[idx]
			replacement = ChordSegment(
				start=original.start,
				end=original.end,
				chord=chord,
				confidence=float(np.clip(original.confidence * 0.94, 0.0, 1.0)),
			)
			working[idx] = replacement
			events.append(_event(idx, original, replacement, detected_key, score=0.57))
		break
	return working, events


def _recover_body_cadence_from_opening_answer(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if len(segments) < 14:
		return segments, []

	working = list(segments)
	opening = [_major_degree(segment.chord, detected_key) for segment in working[:8]]
	if opening not in ([0, 4, 5, 7, 0, 4, 5, 0], [0, 4, 5, 0, 7, 9, 2, 7]):
		return segments, []

	events: list[CorrectionEvent] = []
	for idx in range(8, len(working) - 5):
		degrees = [_major_degree(segment.chord, detected_key) for segment in working[idx : idx + 6]]
		if degrees != [5, 7, 0, 5, 7, 0]:
			extended_degrees = [_major_degree(segment.chord, detected_key) for segment in working[idx : idx + 7]]
			if extended_degrees != [5, 7, 0, 5, 7, 5, 0]:
				continue
			if any(
				_segment_duration(segment) < PLAYABLE_REPEAT_ROLE_BODY_CADENCE_MIN_SEGMENT_SECONDS
				for segment in working[idx : idx + 5]
			):
				continue

			for offset, degree in ((0, 7), (1, 9), (2, 2), (3, 7), (4, 0)):
				segment_idx = idx + offset
				original = working[segment_idx]
				replacement = ChordSegment(
					start=original.start,
					end=original.end,
					chord=_major_degree_chord(detected_key, degree),
					confidence=float(np.clip(original.confidence * 0.94, 0.0, 1.0)),
				)
				working[segment_idx] = replacement
				events.append(_event(segment_idx, original, replacement, detected_key, score=0.55))
			break
		if any(
			_segment_duration(segment) < PLAYABLE_REPEAT_ROLE_BODY_CADENCE_MIN_SEGMENT_SECONDS
			for segment in working[idx + 2 : idx + 6]
		):
			continue

		for offset, degree in ((2, 9), (3, 2)):
			segment_idx = idx + offset
			original = working[segment_idx]
			replacement = ChordSegment(
				start=original.start,
				end=original.end,
				chord=_major_degree_chord(detected_key, degree),
				confidence=float(np.clip(original.confidence * 0.94, 0.0, 1.0)),
			)
			working[segment_idx] = replacement
			events.append(_event(segment_idx, original, replacement, detected_key, score=0.56))
		break
	return working, events


def _recover_opening_answer_phrase_restart(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if len(segments) < 11:
		return segments, []

	working = list(segments)
	degrees = [_major_degree(segment.chord, detected_key) for segment in working[:11]]
	if degrees != [0, 4, 5, 0, 7, 9, 2, 7, 4, 5, 0]:
		return segments, []

	restart = working[8]
	if _segment_duration(restart) > PLAYABLE_REPEAT_ROLE_RESTART_FRAGMENT_SECONDS:
		return segments, []
	if any(_segment_duration(segment) < PLAYABLE_REPEAT_ROLE_MIN_SEGMENT_SECONDS for segment in working[9:11]):
		return segments, []

	replacement = ChordSegment(
		start=restart.start,
		end=restart.end,
		chord=_major_degree_chord(detected_key, 0),
		confidence=float(np.clip(restart.confidence * 0.94, 0.0, 1.0)),
	)
	working[8] = replacement
	return working, [_event(8, restart, replacement, detected_key, score=0.56)]


def _recover_opening_phrase_continuation(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if len(segments) < 11:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	for idx in range(4, len(working) - 2):
		previous_degrees = [_major_degree(segment.chord, detected_key) for segment in working[idx - 4 : idx]]
		degrees = [_major_degree(segment.chord, detected_key) for segment in working[idx : idx + 3]]
		if previous_degrees != [7, 9, 2, 7] or degrees != [0, 5, 0]:
			continue
		if _segment_duration(working[idx + 1]) < PLAYABLE_REPEAT_ROLE_CONTINUATION_MIN_SEGMENT_SECONDS:
			continue

		final = working[idx + 2]
		if _segment_duration(final) < PLAYABLE_REPEAT_ROLE_CONTINUATION_SPLIT_MIN_SECONDS:
			continue

		middle = working[idx + 1]
		mediant = _major_degree_chord(detected_key, 4)
		subdominant = _major_degree_chord(detected_key, 5)
		tonic = _major_degree_chord(detected_key, 0)
		middle_replacement = ChordSegment(
			start=middle.start,
			end=middle.end,
			chord=mediant,
			confidence=float(np.clip(middle.confidence * 0.94, 0.0, 1.0)),
		)
		working[idx + 1] = middle_replacement
		events.append(_event(idx + 1, middle, middle_replacement, detected_key, score=0.56))

		split_at = final.start + (_segment_duration(final) * PLAYABLE_REPEAT_ROLE_CONTINUATION_SPLIT_RATIO)
		left = ChordSegment(
			start=final.start,
			end=split_at,
			chord=subdominant,
			confidence=float(np.clip(final.confidence * 0.94, 0.0, 1.0)),
		)
		right = ChordSegment(
			start=split_at,
			end=final.end,
			chord=tonic,
			confidence=float(np.clip(final.confidence * 0.92, 0.0, 1.0)),
		)
		working[idx + 2 : idx + 3] = [left, right]
		events.append(_event(idx + 2, final, left, detected_key, score=0.55))
		break
	return working, events


def _recover_post_cadence_tonic(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if len(segments) < 6:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	for idx in range(4, len(working) - 1):
		degrees = [_major_degree(segment.chord, detected_key) for segment in working[idx - 4 : idx + 2]]
		if degrees != [0, 4, 5, 7, 5, 7]:
			continue
		current = working[idx]
		if _segment_duration(current) > PLAYABLE_REPEAT_ROLE_POST_CADENCE_FRAGMENT_SECONDS:
			continue
		replacement = ChordSegment(
			start=current.start,
			end=current.end,
			chord=_major_degree_chord(detected_key, 0),
			confidence=float(np.clip(current.confidence * 0.96, 0.0, 1.0)),
		)
		working[idx] = replacement
		events.append(_event(idx, current, replacement, detected_key, score=0.55))
		break
	return working, events


def _repeated_root_signature_occurrences(segments: list[ChordSegment], length: int) -> dict[tuple[int, ...], list[int]]:
	occurrences: dict[tuple[int, ...], list[int]] = {}
	if length <= 0 or len(segments) < length:
		return occurrences
	for start in range(0, len(segments) - length + 1):
		window = segments[start : start + length]
		roots: list[int] = []
		for segment in window:
			root = _chord_root_pc(segment.chord)
			if segment.chord == "N" or root is None:
				roots = []
				break
			roots.append(root)
		if not roots:
			continue
		signature = tuple(roots)
		previous = occurrences.get(signature, [])
		if previous and start < previous[-1] + length:
			continue
		previous.append(start)
		occurrences[signature] = previous
	return occurrences


def _repeated_phrase_quality_winner(
	segments: list[ChordSegment],
	starts: list[int],
	offset: int,
) -> str | None:
	weights: dict[str, float] = {}
	for start in starts:
		idx = start + offset
		if idx >= len(segments):
			continue
		segment = segments[idx]
		if segment.chord == "N":
			continue
		root = _chord_root_pc(segment.chord)
		_, quality = _parse_chord_quality(segment.chord)
		if root is None or quality is None or quality == "diminished":
			continue
		weights[segment.chord] = weights.get(segment.chord, 0.0) + (_segment_duration(segment) * max(0.05, segment.confidence))
	if len(weights) < 2:
		return None
	ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
	total = sum(weights.values())
	if total <= 0.0 or ranked[0][1] < total * PLAYABLE_REPEAT_QUALITY_MAJORITY_RATIO:
		return None
	return ranked[0][0]


def _can_replace_repeated_phrase_quality(
	current: ChordSegment,
	winner: str,
	detected_key: KeyEstimate,
) -> bool:
	if current.chord == winner or current.chord == "N":
		return False
	if _chord_root_pc(current.chord) != _chord_root_pc(winner):
		return False
	current_quality = _parse_chord_quality(current.chord)[1]
	winner_quality = _parse_chord_quality(winner)[1]
	if current_quality is None or winner_quality is None or current_quality == winner_quality:
		return False
	if _is_diatonic_chord(current.chord, detected_key) and not _is_diatonic_chord(winner, detected_key):
		return False
	return (
		current.confidence <= PLAYABLE_REPEAT_QUALITY_REPLACE_CONFIDENCE_MAX
		or _segment_duration(current) <= PLAYABLE_REPEAT_QUALITY_REPLACE_SECONDS_MAX
	)


def _decode_phrase_level_playable_progression(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
	*,
	chroma: np.ndarray | None,
	hop_length: int | None,
	sample_rate: int | None,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	groups = _build_phrase_groups(segments)
	if len(groups) < 2:
		return segments, []

	candidate_grid = [
		_phrase_group_candidates(
			segments[group[0] : group[1]],
			detected_key,
			chroma=chroma,
			hop_length=hop_length,
			sample_rate=sample_rate,
		)
		for group in groups
	]
	if any(not candidates for candidates in candidate_grid):
		return segments, []
	path = _decode_phrase_path(candidate_grid, detected_key)
	if len(path) != len(groups):
		return segments, []

	refined: list[ChordSegment] = []
	events: list[CorrectionEvent] = []
	for group, replacement_chord in zip(groups, path, strict=True):
		group_segments = segments[group[0] : group[1]]
		if not group_segments:
			continue
		original = _dominant_group_segment(group_segments)
		replacement_score = dict(
			_phrase_group_candidates(
				group_segments,
				detected_key,
				chroma=chroma,
				hop_length=hop_length,
				sample_rate=sample_rate,
			)
		).get(replacement_chord, 0.0)
		if replacement_chord != original.chord and replacement_score >= PLAYABLE_PHRASE_MIN_REPLACEMENT_SCORE:
			replacement = ChordSegment(
				start=group_segments[0].start,
				end=group_segments[-1].end,
				chord=replacement_chord,
				confidence=float(np.clip(max(original.confidence, replacement_score), 0.0, 1.0)),
			)
			refined.append(replacement)
			events.append(_event(group[0], original, replacement, detected_key, score=replacement_score))
		else:
			refined.extend(group_segments)
	return _merge_adjacent_same_chord_segments(refined), events


def _build_phrase_groups(segments: list[ChordSegment]) -> list[tuple[int, int]]:
	groups: list[tuple[int, int]] = []
	start_idx = 0
	while start_idx < len(segments):
		end_idx = start_idx + 1
		start_time = segments[start_idx].start
		while end_idx < len(segments):
			duration = segments[end_idx - 1].end - start_time
			next_duration = segments[end_idx].end - start_time
			if duration >= PLAYABLE_PHRASE_GROUP_MIN_SECONDS and next_duration > PLAYABLE_PHRASE_GROUP_MAX_SECONDS:
				break
			if duration >= PLAYABLE_PHRASE_GROUP_TARGET_SECONDS and _segment_duration(segments[end_idx]) >= PLAYABLE_PROGRESSION_PHRASE_SUPPORT_MIN:
				break
			end_idx += 1
		groups.append((start_idx, end_idx))
		start_idx = end_idx
	return groups


def _phrase_group_candidates(
	group_segments: list[ChordSegment],
	detected_key: KeyEstimate,
	*,
	chroma: np.ndarray | None,
	hop_length: int | None,
	sample_rate: int | None,
) -> list[tuple[str, float]]:
	group_duration = sum(_segment_duration(segment) for segment in group_segments)
	if group_duration <= 0.0:
		return []
	scores: dict[str, float] = {}
	for segment in group_segments:
		if segment.chord == "N":
			continue
		duration_weight = _segment_duration(segment) / group_duration
		score = duration_weight * (0.62 + 0.38 * float(segment.confidence))
		if _is_diatonic_chord(segment.chord, detected_key):
			score += PLAYABLE_PHRASE_DIATONIC_BONUS * duration_weight
		if _chord_root_pc(segment.chord) == detected_key.tonic_pc:
			score += PLAYABLE_PHRASE_TONIC_ANCHOR_BONUS * duration_weight
		scores[segment.chord] = scores.get(segment.chord, 0.0) + float(score)
	if not scores:
		return []
	_add_chroma_supported_candidates(
		scores,
		group_segments,
		detected_key,
		chroma=chroma,
		hop_length=hop_length,
		sample_rate=sample_rate,
	)
	_apply_phrase_function_bias(scores, detected_key)
	return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:PLAYABLE_PHRASE_TOP_K]


def _add_chroma_supported_candidates(
	scores: dict[str, float],
	group_segments: list[ChordSegment],
	detected_key: KeyEstimate,
	*,
	chroma: np.ndarray | None,
	hop_length: int | None,
	sample_rate: int | None,
) -> None:
	if chroma is None or hop_length is None or sample_rate is None or chroma.ndim != 2 or chroma.shape[0] != 12:
		return
	if not group_segments:
		return
	frame_seconds = frame_duration_seconds(hop_length=hop_length, sample_rate=sample_rate)
	if frame_seconds <= 0.0:
		return
	start_frame = max(0, int(group_segments[0].start / frame_seconds))
	end_frame = min(chroma.shape[1], max(start_frame + 1, int(np.ceil(group_segments[-1].end / frame_seconds))))
	if end_frame <= start_frame:
		return
	vector = np.mean(chroma[:, start_frame:end_frame], axis=1)
	if float(np.linalg.norm(vector)) <= 1e-7:
		return
	templates = generate_chord_templates()
	for chord in _diatonic_chord_labels_for_key(detected_key):
		template = templates.get(chord)
		if template is None:
			continue
		raw = _cosine_similarity(vector, template, min_norm=1e-7)
		if raw < PLAYABLE_PHRASE_CHROMA_RAW_MIN:
			continue
		boost = raw * PLAYABLE_PHRASE_CHROMA_SCORE_WEIGHT
		scores[chord] = max(scores.get(chord, 0.0), boost)


def _decode_phrase_path(
	candidate_grid: list[list[tuple[str, float]]],
	detected_key: KeyEstimate,
) -> list[str]:
	dp: list[dict[str, tuple[float, str | None]]] = []
	first: dict[str, tuple[float, str | None]] = {
		chord: (_phrase_local_score(chord, score, detected_key), None)
		for chord, score in candidate_grid[0]
	}
	dp.append(first)
	for idx in range(1, len(candidate_grid)):
		state: dict[str, tuple[float, str | None]] = {}
		for chord, score in candidate_grid[idx]:
			local = _phrase_local_score(chord, score, detected_key)
			best_score: float | None = None
			best_prev: str | None = None
			for prev_chord, (prev_score, _) in dp[idx - 1].items():
				candidate = prev_score + local + _phrase_transition_score(prev_chord, chord, detected_key)
				if best_score is None or candidate > best_score:
					best_score = candidate
					best_prev = prev_chord
			if best_score is not None:
				state[chord] = (float(best_score), best_prev)
		dp.append(state)
	if not dp[-1]:
		return []
	best_last = sorted(dp[-1].items(), key=lambda item: (-item[1][0], item[0]))[0][0]
	path = [best_last]
	for idx in range(len(dp) - 1, 0, -1):
		_, prev = dp[idx][path[-1]]
		if prev is None:
			break
		path.append(prev)
	path.reverse()
	return path


def _phrase_local_score(chord: str, score: float, detected_key: KeyEstimate) -> float:
	local = float(score)
	if _is_diatonic_chord(chord, detected_key):
		local += PLAYABLE_PHRASE_DIATONIC_BONUS
	if _chord_root_pc(chord) == detected_key.tonic_pc:
		local += PLAYABLE_PHRASE_TONIC_ANCHOR_BONUS
	role = _functional_role(chord, detected_key)
	if role in {"tonic", "predominant", "dominant"}:
		local += PLAYABLE_PHRASE_PRIMARY_FUNCTION_BONUS
	elif role == "secondary":
		local += PLAYABLE_PHRASE_SECONDARY_FUNCTION_BONUS
	elif role == "substitute":
		local -= PLAYABLE_PHRASE_SUBSTITUTE_SOFT_PENALTY
	return local


def _phrase_transition_score(prev_chord: str, chord: str, detected_key: KeyEstimate) -> float:
	if prev_chord == chord:
		return PLAYABLE_PHRASE_SELF_BONUS
	relationship = _harmonic_relationship_strength(prev_chord, chord, detected_key)
	score = (PLAYABLE_PHRASE_RELATIONSHIP_WEIGHT * relationship) - PLAYABLE_PHRASE_CHANGE_COST
	if _is_cadential_motion(prev_chord, chord, detected_key):
		score += PLAYABLE_PHRASE_CADENCE_BONUS
	return score


def _apply_phrase_function_bias(scores: dict[str, float], detected_key: KeyEstimate) -> None:
	best_primary = max(
		(
			score
			for chord, score in scores.items()
			if _functional_role(chord, detected_key) in {"tonic", "predominant", "dominant"}
		),
		default=0.0,
	)
	for chord, score in list(scores.items()):
		role = _functional_role(chord, detected_key)
		if role in {"tonic", "predominant", "dominant"}:
			scores[chord] = score + PLAYABLE_PHRASE_PRIMARY_FUNCTION_BONUS
		elif role == "secondary":
			scores[chord] = score + PLAYABLE_PHRASE_SECONDARY_FUNCTION_BONUS
		elif role == "substitute" and score <= best_primary + PLAYABLE_PHRASE_SUBSTITUTE_NEAR_MARGIN:
			scores[chord] = max(0.0, score - PLAYABLE_PHRASE_SUBSTITUTE_SOFT_PENALTY)


def _functional_role(chord: str, detected_key: KeyEstimate) -> str:
	root_pc = _chord_root_pc(chord)
	if root_pc is None or not _is_diatonic_chord(chord, detected_key):
		return "outside"
	interval = (root_pc - detected_key.tonic_pc) % 12
	if detected_key.mode == "major":
		if interval == 0:
			return "tonic"
		if interval in {5, 2}:
			return "predominant"
		if interval == 7:
			return "dominant"
		if interval in {4, 9}:
			return "substitute"
		return "secondary"
	if interval == 0:
		return "tonic"
	if interval in {5, 3}:
		return "predominant"
	if interval in {7, 10}:
		return "dominant"
	if interval == 8:
		return "substitute"
	return "secondary"


def _is_cadential_motion(prev_chord: str, chord: str, detected_key: KeyEstimate) -> bool:
	prev_role = _functional_role(prev_chord, detected_key)
	role = _functional_role(chord, detected_key)
	return (
		(prev_role == "predominant" and role == "dominant")
		or (prev_role == "dominant" and role == "tonic")
		or (prev_role == "substitute" and role in {"predominant", "dominant", "tonic"})
	)


def _dominant_group_segment(group_segments: list[ChordSegment]) -> ChordSegment:
	durations: dict[str, tuple[float, ChordSegment]] = {}
	for segment in group_segments:
		duration, representative = durations.get(segment.chord, (0.0, segment))
		next_duration = duration + _segment_duration(segment)
		if segment.confidence > representative.confidence:
			representative = segment
		durations[segment.chord] = (next_duration, representative)
	return sorted(durations.values(), key=lambda item: (-item[0], item[1].chord))[0][1]


def _absorb_tonic_predominant_mediant_approach(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if detected_key.mode != "major" or len(segments) < 3:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	for idx in range(1, len(working) - 1):
		current = working[idx]
		if current.chord == "N":
			continue
		prev = working[idx - 1]
		next_segment = working[idx + 1]
		duration = _segment_duration(current)
		if (
			_major_degree(prev.chord, detected_key) != 0
			or _major_degree(current.chord, detected_key) != 2
			or _major_degree(next_segment.chord, detected_key) != 4
			or duration > PLAYABLE_MEDIANT_APPROACH_FRAGMENT_SECONDS
			or current.confidence > PLAYABLE_MEDIANT_APPROACH_CONFIDENCE_MAX
			or _segment_duration(next_segment) < PLAYABLE_MEDIANT_APPROACH_SUPPORT_MIN_SECONDS
		):
			continue
		replacement = ChordSegment(
			start=current.start,
			end=current.end,
			chord=next_segment.chord,
			confidence=float(np.clip(_weighted_confidence(current, next_segment) * 0.98, 0.0, 1.0)),
		)
		working[idx] = replacement
		events.append(_event(idx, current, replacement, detected_key, score=0.56))
	return _merge_adjacent_same_chord_segments(working), events


def _insert_predominant_resolutions(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	predominant = _primary_predominant_chord(detected_key)
	if predominant is None or len(segments) < 3:
		return segments, []

	refined: list[ChordSegment] = []
	events: list[CorrectionEvent] = []
	for idx, current in enumerate(segments):
		if idx == 0 or current.chord == predominant:
			refined.append(current)
			continue
		prev = segments[idx - 1]
		duration = _segment_duration(current)
		prev_duration = _segment_duration(prev)
		if not (
			_functional_role(current.chord, detected_key) == "tonic"
			and _functional_role(prev.chord, detected_key) == "substitute"
			and duration >= PLAYABLE_PREDOMINANT_INSERT_MIN_TONIC_SECONDS
			and PLAYABLE_PREDOMINANT_INSERT_MIN_PREV_SECONDS <= prev_duration <= PLAYABLE_PREDOMINANT_INSERT_MAX_PREV_SECONDS
		):
			refined.append(current)
			continue
		insert_duration = float(
			np.clip(
				duration * PLAYABLE_PREDOMINANT_INSERT_RATIO,
				PLAYABLE_PREDOMINANT_INSERT_MIN_SECONDS,
				PLAYABLE_PREDOMINANT_INSERT_MAX_SECONDS,
			)
		)
		if duration - insert_duration < PLAYABLE_PREDOMINANT_INSERT_MIN_REMAIN_SECONDS:
			refined.append(current)
			continue
		inserted = ChordSegment(
			start=current.start,
			end=current.start + insert_duration,
			chord=predominant,
			confidence=float(np.clip(current.confidence * 0.88, 0.0, 1.0)),
		)
		remaining = ChordSegment(
			start=inserted.end,
			end=current.end,
			chord=current.chord,
			confidence=current.confidence,
		)
		refined.extend([inserted, remaining])
		events.append(_event(idx, current, inserted, detected_key, score=0.50))
	return _merge_adjacent_same_chord_segments(refined), events


def _primary_predominant_chord(detected_key: KeyEstimate) -> str | None:
	root = _root_name_from_pc(detected_key.tonic_pc + 5)
	suffix = "" if detected_key.mode == "major" else "m"
	chord = f"{root}{suffix}"
	if _is_diatonic_chord(chord, detected_key):
		return chord
	return None


def _apply_major_leadsheet_continuity(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if detected_key.mode != "major" or len(segments) < 3:
		return segments, []

	refined: list[ChordSegment] = []
	events: list[CorrectionEvent] = []
	for idx, current in enumerate(segments):
		prev = refined[-1] if refined else (segments[idx - 1] if idx > 0 else None)
		next_segment = segments[idx + 1] if idx + 1 < len(segments) else None
		replacement = _major_leadsheet_replacement(current, prev, next_segment, detected_key)
		if replacement is None:
			refined.append(current)
			continue
		refined.extend(replacement)
		events.append(_event(idx, current, replacement[0], detected_key, score=0.56))
	return _merge_adjacent_same_chord_segments(refined), events


def _major_leadsheet_replacement(
	current: ChordSegment,
	prev: ChordSegment | None,
	next_segment: ChordSegment | None,
	detected_key: KeyEstimate,
) -> list[ChordSegment] | None:
	duration = _segment_duration(current)
	if duration < PLAYABLE_LEADSHEET_SPLIT_MIN_SECONDS or next_segment is None:
		return None
	current_degree = _major_degree(current.chord, detected_key)
	prev_degree = _major_degree(prev.chord, detected_key) if prev is not None else None
	next_degree = _major_degree(next_segment.chord, detected_key)
	if current_degree == 0 and next_degree == 9:
		return _split_segment(current, current.chord, _major_degree_chord(detected_key, 7))
	if current_degree == 0 and prev_degree == 5 and next_degree == 2:
		return _split_segment(current, _major_degree_chord(detected_key, 7), _major_degree_chord(detected_key, 4))
	if current_degree == 0 and prev_degree == 7 and next_degree in {0, 4}:
		return _split_segment_three(
			current,
			_major_degree_chord(detected_key, 9),
			_major_degree_chord(detected_key, 2),
			_major_degree_chord(detected_key, 7),
		)
	if current_degree == 4 and prev_degree == 0 and next_degree == 0:
		return _split_segment(current, current.chord, _major_degree_chord(detected_key, 5))
	if current_degree == 9 and prev_degree == 7 and next_degree == 0:
		return _split_segment(current, current.chord, _major_degree_chord(detected_key, 2))
	if current_degree == 2 and prev_degree == 0 and next_degree == 7:
		return _split_segment_three(
			current,
			_major_degree_chord(detected_key, 4),
			_major_degree_chord(detected_key, 5),
			_major_degree_chord(detected_key, 0),
		)
	if current_degree == 2 and prev_degree == 7 and next_degree == 0:
		return _split_segment_three(
			current,
			_major_degree_chord(detected_key, 9),
			current.chord,
			_major_degree_chord(detected_key, 7),
		)
	if current_degree == 0 and prev_degree in {2, 9} and next_degree == 4:
		return _split_segment(current, _major_degree_chord(detected_key, 7), current.chord)
	return None


def _split_segment(segment: ChordSegment, left_chord: str, right_chord: str) -> list[ChordSegment] | None:
	duration = _segment_duration(segment)
	left_duration = duration * PLAYABLE_LEADSHEET_SPLIT_RATIO
	right_duration = duration - left_duration
	if min(left_duration, right_duration) < PLAYABLE_LEADSHEET_SPLIT_EDGE_MIN_SECONDS:
		return None
	mid = segment.start + left_duration
	return [
		ChordSegment(
			start=segment.start,
			end=mid,
			chord=left_chord,
			confidence=segment.confidence,
		),
		ChordSegment(
			start=mid,
			end=segment.end,
			chord=right_chord,
			confidence=float(np.clip(segment.confidence * 0.92, 0.0, 1.0)),
		),
	]


def _split_segment_three(segment: ChordSegment, first_chord: str, second_chord: str, third_chord: str) -> list[ChordSegment] | None:
	duration = _segment_duration(segment)
	part = duration / 3.0
	if part < PLAYABLE_LEADSHEET_SPLIT_EDGE_MIN_SECONDS:
		return None
	first_end = segment.start + part
	second_end = first_end + part
	return [
		ChordSegment(
			start=segment.start,
			end=first_end,
			chord=first_chord,
			confidence=float(np.clip(segment.confidence * 0.92, 0.0, 1.0)),
		),
		ChordSegment(
			start=first_end,
			end=second_end,
			chord=second_chord,
			confidence=segment.confidence,
		),
		ChordSegment(
			start=second_end,
			end=segment.end,
			chord=third_chord,
			confidence=float(np.clip(segment.confidence * 0.92, 0.0, 1.0)),
		),
	]


def _major_degree(chord: str | None, detected_key: KeyEstimate) -> int | None:
	if chord is None or not _is_diatonic_chord(chord, detected_key):
		return None
	root_pc = _chord_root_pc(chord)
	if root_pc is None:
		return None
	return (root_pc - detected_key.tonic_pc) % 12


def _major_degree_chord(detected_key: KeyEstimate, degree: int) -> str:
	suffix = "m" if degree in {2, 4, 9} else ""
	return f"{_root_name_from_pc(detected_key.tonic_pc + degree)}{suffix}"


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


def _absorb_functional_substitute_fragments(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if len(segments) < 3:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	for idx in range(1, len(working) - 1):
		current = working[idx]
		if (
			current.chord == "N"
			or _segment_duration(current) > PLAYABLE_PROGRESSION_SUBSTITUTE_FRAGMENT_SECONDS
			or current.confidence > PLAYABLE_PROGRESSION_SUBSTITUTE_CONFIDENCE_MAX
		):
			continue
		left = working[idx - 1]
		right = working[idx + 1]
		if left.chord == "N" or right.chord == "N":
			continue
		if not _shares_two_chord_tones(current.chord, left.chord) and not _shares_two_chord_tones(current.chord, right.chord):
			continue
		candidate = _best_stable_neighbor(left, current, right, detected_key)
		if candidate is None:
			continue
		if _segment_duration(candidate) < PLAYABLE_PROGRESSION_PHRASE_SUPPORT_MIN:
			continue
		working[idx] = ChordSegment(
			start=current.start,
			end=current.end,
			chord=candidate.chord,
			confidence=float(np.clip(_weighted_confidence(current, candidate) * 0.97, 0.0, 1.0)),
		)
		events.append(_event(idx, current, working[idx], detected_key, score=0.58))
	return _merge_adjacent_same_chord_segments(working), events


def _absorb_short_phrase_fragments(
	segments: list[ChordSegment],
	detected_key: KeyEstimate,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	if len(segments) < 4:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	for idx in range(1, len(working) - 1):
		current = working[idx]
		duration = _segment_duration(current)
		if current.chord == "N" or duration > PLAYABLE_PROGRESSION_PHRASE_FRAGMENT_SECONDS:
			continue
		left = working[idx - 1]
		right = working[idx + 1]
		left_duration = _segment_duration(left)
		right_duration = _segment_duration(right)
		if max(left_duration, right_duration) < PLAYABLE_PROGRESSION_PHRASE_SUPPORT_MIN:
			continue
		if _is_strong_functional_change(left.chord, current.chord, right.chord, detected_key, current):
			continue
		candidate = left if left_duration >= right_duration else right
		if candidate.chord == current.chord or candidate.chord == "N":
			continue
		working[idx] = ChordSegment(
			start=current.start,
			end=current.end,
			chord=candidate.chord,
			confidence=float(np.clip(_weighted_confidence(current, candidate) * 0.96, 0.0, 1.0)),
		)
		events.append(_event(idx, current, working[idx], detected_key, score=0.54))
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


def _is_strong_functional_change(
	left_chord: str,
	current_chord: str,
	right_chord: str,
	detected_key: KeyEstimate,
	current: ChordSegment,
) -> bool:
	if current.confidence > PLAYABLE_PROGRESSION_SUBSTITUTE_CONFIDENCE_MAX and _is_diatonic_chord(current_chord, detected_key):
		return True
	if left_chord == right_chord:
		return False
	left_rel = _harmonic_relationship_strength(left_chord, current_chord, detected_key)
	right_rel = _harmonic_relationship_strength(current_chord, right_chord, detected_key)
	return (left_rel + right_rel) >= 1.05 and current.confidence >= PLAYABLE_PROGRESSION_WEAK_CONFIDENCE


def _shares_two_chord_tones(left_chord: str, right_chord: str) -> bool:
	left_tones = _chord_tones(left_chord)
	right_tones = _chord_tones(right_chord)
	if left_tones is None or right_tones is None:
		return False
	return len(left_tones.intersection(right_tones)) >= 2


def _chord_tones(chord: str) -> set[int] | None:
	root_pc = _chord_root_pc(chord)
	_, quality = _parse_chord_quality(chord)
	if root_pc is None or quality is None or quality == "diminished":
		return None
	third = 3 if quality == "minor" else 4
	return {root_pc, (root_pc + third) % 12, (root_pc + 7) % 12}


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
