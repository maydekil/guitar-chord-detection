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
	PLAYABLE_INTRO_TONIC_LOOKAHEAD_SECONDS,
	PLAYABLE_INTRO_TONIC_MIN_TOTAL_SECONDS,
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

	return _merge_adjacent_same_chord_segments(working), events


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


def _early_tonic_duration(segments: list[ChordSegment], tonic: str) -> float:
	total = 0.0
	for segment in segments:
		if segment.start > PLAYABLE_INTRO_TONIC_LOOKAHEAD_SECONDS:
			break
		if segment.chord == tonic:
			total += max(0.0, min(segment.end, PLAYABLE_INTRO_TONIC_LOOKAHEAD_SECONDS) - segment.start)
	return total


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
