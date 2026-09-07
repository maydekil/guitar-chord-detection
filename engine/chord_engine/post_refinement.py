"""Post-decoder chord segment refinement stages."""

from __future__ import annotations

from statistics import median

import numpy as np

from chord_engine.analysis_config import *  # noqa: F403 - shared refinement tuning constants
from chord_engine.analysis_models import CorrectionEvent
from chord_engine.detector import KeyEstimate
from chord_engine.music_theory import (
    _diatonic_chord_labels_for_key,
    _harmonic_plausibility,
    _harmonic_relationship_strength,
    _is_diatonic_chord,
    _is_root_preserving_quality_switch,
    _parse_chord_quality,
)
from chord_engine.mini_sequence_refinement import _refine_long_holds_with_mini_harmonic_sequence_decoder
from chord_engine.root_aware import _estimate_region_root_aware_identity
from chord_engine.segment_utils import (
    _merge_adjacent_same_chord_segments,
    _neighbor_support,
    _segment_duration,
    _weighted_confidence,
)
from chord_engine.segmentation import ChordSegment

def _apply_context_aware_short_segment_correction(
	segments: list[ChordSegment],
	detected_key: KeyEstimate | None,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	"""Correct weak short isolated anomalies using temporal+harmony context."""
	if len(segments) < 3:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []

	for idx in range(1, len(working) - 1):
		left = working[idx - 1]
		current = working[idx]
		right = working[idx + 1]

		if current.chord == left.chord and current.chord == right.chord:
			continue
		if current.chord == "N":
			continue

		current_duration = _segment_duration(current)
		if current_duration > CONTEXT_SHORT_SEGMENT_MAX_SECONDS:
			continue
		if current.confidence >= CONTEXT_SHORT_STRONG_CONFIDENCE_MIN:
			continue
		if current.confidence > CONTEXT_SHORT_LOW_CONFIDENCE_MAX:
			continue
		if _segment_duration(left) < CONTEXT_NEIGHBOR_MIN_DURATION_SECONDS:
			continue
		if _segment_duration(right) < CONTEXT_NEIGHBOR_MIN_DURATION_SECONDS:
			continue
		if left.confidence < CONTEXT_NEIGHBOR_MIN_CONFIDENCE:
			continue
		if right.confidence < CONTEXT_NEIGHBOR_MIN_CONFIDENCE:
			continue

		left_support = _neighbor_support(left, detected_key)
		right_support = _neighbor_support(right, detected_key)
		candidate = left if left_support >= right_support else right
		candidate_support = max(left_support, right_support)
		other_support = min(left_support, right_support)

		if candidate.chord == current.chord:
			continue
		if candidate_support < CONTEXT_CANDIDATE_SUPPORT_MIN:
			continue

		neighbors_same = left.chord == right.chord
		if not neighbors_same and (candidate_support - other_support) < CONTEXT_SUPPORT_GAP_MIN:
			continue

		current_plausibility = _harmonic_plausibility(current.chord, detected_key)
		replacement_plausibility = _harmonic_plausibility(candidate.chord, detected_key)
		current_is_diatonic = _is_diatonic_chord(current.chord, detected_key)
		candidate_is_diatonic = _is_diatonic_chord(candidate.chord, detected_key)

		if current_plausibility >= HARMONIC_PLAUSIBILITY_DOMINANT_MAJOR_IN_MINOR and current.confidence >= 0.68:
			continue

		neighbor_confidence = max(left.confidence, right.confidence)
		confidence_gap = max(0.0, neighbor_confidence - current.confidence)
		shortness = max(0.0, (CONTEXT_SHORT_SEGMENT_MAX_SECONDS - current_duration) / CONTEXT_SHORT_SEGMENT_MAX_SECONDS)
		plausibility_gap = max(0.0, replacement_plausibility - current_plausibility)
		support_gap = max(0.0, candidate_support - other_support)

		score = (
			0.44 * confidence_gap
			+ 0.22 * shortness
			+ 0.18 * plausibility_gap
			+ 0.16 * support_gap
			+ (CONTEXT_SAME_CHORD_STABILITY_BONUS if neighbors_same else 0.0)
			+ (CONTEXT_NON_DIATONIC_SHORT_BONUS if candidate_is_diatonic and not current_is_diatonic else 0.0)
		)

		if score < CONTEXT_CORRECTION_SCORE_MIN:
			continue

		new_confidence = float(np.clip(candidate.confidence * 0.95, 0.0, 1.0))
		working[idx] = ChordSegment(
			start=current.start,
			end=current.end,
			chord=candidate.chord,
			confidence=new_confidence,
		)
		events.append(
			CorrectionEvent(
				index=idx,
				replaced_chord=current.chord,
				new_chord=candidate.chord,
				start=float(current.start),
				end=float(current.end),
				duration=float(current_duration),
				original_confidence=float(current.confidence),
				new_confidence=float(new_confidence),
				score=float(score),
				harmonic_plausibility_before=float(current_plausibility),
				harmonic_plausibility_after=float(replacement_plausibility),
			)
		)

	return _merge_adjacent_same_chord_segments(working), events


def _suppress_weak_diminished_passing_segments(
	segments: list[ChordSegment],
	detected_key: KeyEstimate | None,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	"""Absorb very short weak diminished flashes into stable neighboring chords."""
	if len(segments) < 3:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []
	for idx in range(1, len(working) - 1):
		current = working[idx]
		if not current.chord.endswith("dim"):
			continue

		duration = _segment_duration(current)
		if duration > WEAK_DIMINISHED_MAX_SECONDS or current.confidence > WEAK_DIMINISHED_MAX_CONFIDENCE:
			continue

		left = working[idx - 1]
		right = working[idx + 1]
		left_quality = _parse_chord_quality(left.chord)[1]
		right_quality = _parse_chord_quality(right.chord)[1]
		left_ok = left.chord != "N" and left_quality != "diminished" and _segment_duration(left) >= CONTEXT_NEIGHBOR_MIN_DURATION_SECONDS
		right_ok = right.chord != "N" and right_quality != "diminished" and _segment_duration(right) >= CONTEXT_NEIGHBOR_MIN_DURATION_SECONDS
		if not left_ok and not right_ok:
			continue

		left_support = _neighbor_support(left, detected_key) if left_ok else -1.0
		right_support = _neighbor_support(right, detected_key) if right_ok else -1.0
		candidate = right if right_support > left_support else left
		candidate_support = max(left_support, right_support)
		if candidate_support < CONTEXT_CANDIDATE_SUPPORT_MIN:
			continue

		before = _harmonic_plausibility(current.chord, detected_key)
		after = _harmonic_plausibility(candidate.chord, detected_key)
		new_confidence = float(np.clip(candidate.confidence * 0.94, 0.0, 1.0))
		working[idx] = ChordSegment(
			start=current.start,
			end=current.end,
			chord=candidate.chord,
			confidence=new_confidence,
		)
		events.append(
			CorrectionEvent(
				index=idx,
				replaced_chord=current.chord,
				new_chord=candidate.chord,
				start=float(current.start),
				end=float(current.end),
				duration=float(duration),
				original_confidence=float(current.confidence),
				new_confidence=float(new_confidence),
				score=float(candidate_support),
				harmonic_plausibility_before=float(before),
				harmonic_plausibility_after=float(after),
			)
		)

	return _merge_adjacent_same_chord_segments(working), events


def _suppress_weak_timeline_fragments(
	segments: list[ChordSegment],
	detected_key: KeyEstimate | None,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	"""Absorb implausibly brief low-confidence chord fragments in the final timeline."""
	if len(segments) < 2:
		return segments, []

	working = list(segments)
	events: list[CorrectionEvent] = []

	for idx, current in enumerate(list(working)):
		duration = _segment_duration(current)
		if current.chord == "N":
			continue

		is_first = idx == 0
		is_last = idx == len(working) - 1
		is_edge = is_first or is_last
		if is_edge:
			if duration > WEAK_EDGE_SEGMENT_MAX_SECONDS or current.confidence > WEAK_EDGE_SEGMENT_MAX_CONFIDENCE:
				continue
			neighbor_idx = 1 if is_first else idx - 1
			if neighbor_idx < 0 or neighbor_idx >= len(working):
				continue
			candidate = working[neighbor_idx]
			if _segment_duration(candidate) < 2.0:
				continue
		else:
			if duration > WEAK_MICRO_SEGMENT_MAX_SECONDS or current.confidence > WEAK_MICRO_SEGMENT_MAX_CONFIDENCE:
				continue
			left = working[idx - 1]
			right = working[idx + 1]
			if left.chord == current.chord or right.chord == current.chord:
				continue
			left_support = _neighbor_support(left, detected_key)
			right_support = _neighbor_support(right, detected_key)
			candidate = right if right_support > left_support else left
			if max(left_support, right_support) < CONTEXT_CANDIDATE_SUPPORT_MIN:
				continue

		before = _harmonic_plausibility(current.chord, detected_key)
		after = _harmonic_plausibility(candidate.chord, detected_key)
		new_confidence = float(np.clip(candidate.confidence * 0.94, 0.0, 1.0))
		working[idx] = ChordSegment(
			start=current.start,
			end=current.end,
			chord=candidate.chord,
			confidence=new_confidence,
		)
		events.append(
			CorrectionEvent(
				index=idx,
				replaced_chord=current.chord,
				new_chord=candidate.chord,
				start=float(current.start),
				end=float(current.end),
				duration=float(duration),
				original_confidence=float(current.confidence),
				new_confidence=float(new_confidence),
				score=float(max(_neighbor_support(candidate, detected_key), 0.0)),
				harmonic_plausibility_before=float(before),
				harmonic_plausibility_after=float(after),
			)
		)

	return _merge_adjacent_same_chord_segments(working), events


def _refine_long_segments_with_harmonic_evidence(
	segments: list[ChordSegment],
	*,
	chroma: np.ndarray,
	low_chroma: np.ndarray,
	boundaries: np.ndarray,
	hop_length: int,
	sample_rate: int,
	detected_key: KeyEstimate | None,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	"""Split long chord holds when internal beat-level harmony consistently disagrees."""
	if len(segments) == 0 or boundaries.size < 2:
		return segments, []

	refined: list[ChordSegment] = []
	events: list[CorrectionEvent] = []
	for segment_idx, segment in enumerate(segments):
		duration = _segment_duration(segment)
		if duration < LONG_SEGMENT_REFINE_MIN_SECONDS or segment.chord == "N":
			refined.append(segment)
			continue

		start_frame = int(max(0, round(segment.start * sample_rate / hop_length)))
		end_frame = int(min(chroma.shape[1], round(segment.end * sample_rate / hop_length)))
		subsegments = _long_segment_candidate_subsegments(
			segment,
			start_frame=start_frame,
			end_frame=end_frame,
			chroma=chroma,
			low_chroma=low_chroma,
			boundaries=boundaries,
			hop_length=hop_length,
			sample_rate=sample_rate,
			detected_key=detected_key,
		)
		if len(subsegments) <= 1:
			refined.append(segment)
			continue

		refined.extend(subsegments)
		for sub in subsegments:
			if sub.chord == segment.chord:
				continue
			events.append(
				CorrectionEvent(
					index=segment_idx,
					replaced_chord=segment.chord,
					new_chord=sub.chord,
					start=float(sub.start),
					end=float(sub.end),
					duration=float(_segment_duration(sub)),
					original_confidence=float(segment.confidence),
					new_confidence=float(sub.confidence),
					score=float(sub.confidence),
					harmonic_plausibility_before=float(_harmonic_plausibility(segment.chord, detected_key)),
					harmonic_plausibility_after=float(_harmonic_plausibility(sub.chord, detected_key)),
				)
			)

	return _merge_adjacent_same_chord_segments(refined), events


def _long_segment_candidate_subsegments(
	segment: ChordSegment,
	*,
	start_frame: int,
	end_frame: int,
	chroma: np.ndarray,
	low_chroma: np.ndarray,
	boundaries: np.ndarray,
	hop_length: int,
	sample_rate: int,
	detected_key: KeyEstimate | None,
) -> list[ChordSegment]:
	frame_edges = [start_frame]
	for boundary in boundaries.tolist():
		frame = int(boundary)
		if start_frame < frame < end_frame:
			frame_edges.append(frame)
	refine_step_frames = max(1, int(round(LONG_SEGMENT_REFINE_WINDOW_SECONDS * sample_rate / hop_length)))
	for frame in range(start_frame + refine_step_frames, end_frame, refine_step_frames):
		frame_edges.append(int(frame))
	frame_edges.append(end_frame)
	frame_edges = sorted(set(frame_edges))
	if len(frame_edges) < 3:
		return [segment]

	beat_segments: list[ChordSegment] = []
	for left, right in zip(frame_edges[:-1], frame_edges[1:], strict=False):
		if right <= left:
			continue
		region_chroma = np.asarray(chroma[:, left:right], dtype=np.float32)
		region_low = np.asarray(low_chroma[:, left:right], dtype=np.float32)
		root_aware = _estimate_region_root_aware_identity(region_chroma, region_low, key_estimate=detected_key)
		scores = dict(root_aware["combined_scores"])
		winner = str(root_aware["winner_label"])
		winner_score = float(scores.get(winner, 0.0))
		parent_score = float(scores.get(segment.chord, 0.0))
		if (
			winner != segment.chord
			and winner_score >= LONG_SEGMENT_REFINE_MIN_SCORE
			and (winner_score - parent_score) >= LONG_SEGMENT_REFINE_SCORE_MARGIN
		):
			label = winner
			confidence = winner_score
		else:
			label = segment.chord
			confidence = max(float(segment.confidence), parent_score)

		beat_segments.append(
			ChordSegment(
				start=float(left * hop_length / sample_rate),
				end=float(right * hop_length / sample_rate),
				chord=label,
				confidence=float(np.clip(confidence, 0.0, 1.0)),
			)
		)

	collapsed = _merge_adjacent_same_chord_segments(beat_segments)
	cleaned: list[ChordSegment] = []
	for child in collapsed:
		child_duration = _segment_duration(child)
		if child.chord == segment.chord or child_duration >= LONG_SEGMENT_REFINE_MIN_CHILD_SECONDS:
			cleaned.append(child)
			continue
		if cleaned:
			prev = cleaned[-1]
			cleaned[-1] = ChordSegment(
				start=prev.start,
				end=child.end,
				chord=prev.chord,
				confidence=prev.confidence,
			)
		else:
			cleaned.append(
				ChordSegment(
					start=child.start,
					end=child.end,
					chord=segment.chord,
					confidence=segment.confidence,
				)
			)

	if len(cleaned) > 1:
		first = cleaned[0]
		last = cleaned[-1]
		cleaned[0] = ChordSegment(start=segment.start, end=first.end, chord=first.chord, confidence=first.confidence)
		cleaned[-1] = ChordSegment(start=last.start, end=segment.end, chord=last.chord, confidence=last.confidence)
		return _merge_adjacent_same_chord_segments(cleaned)

	return [segment]


def _refine_long_segments_with_sustained_harmonic_drift(
	segments: list[ChordSegment],
	*,
	chroma: np.ndarray,
	low_chroma: np.ndarray,
	boundaries: np.ndarray,
	hop_length: int,
	sample_rate: int,
	detected_key: KeyEstimate | None,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	"""Split long holds only when an alternate chord persists for about one beat."""
	if len(segments) == 0:
		return segments, []

	refined: list[ChordSegment] = []
	events: list[CorrectionEvent] = []
	for segment_idx, segment in enumerate(segments):
		if segment.chord == "N" or _segment_duration(segment) < HARMONIC_DRIFT_REFINE_MIN_SECONDS:
			refined.append(segment)
			continue

		subsegments = _sustained_harmonic_drift_subsegments(
			segment,
			chroma=chroma,
			low_chroma=low_chroma,
			boundaries=boundaries,
			hop_length=hop_length,
			sample_rate=sample_rate,
			detected_key=detected_key,
		)
		if len(subsegments) <= 1:
			refined.append(segment)
			continue

		refined.extend(subsegments)
		for sub in subsegments:
			if sub.chord == segment.chord:
				continue
			events.append(
				CorrectionEvent(
					index=segment_idx,
					replaced_chord=segment.chord,
					new_chord=sub.chord,
					start=float(sub.start),
					end=float(sub.end),
					duration=float(_segment_duration(sub)),
					original_confidence=float(segment.confidence),
					new_confidence=float(sub.confidence),
					score=float(sub.confidence),
					harmonic_plausibility_before=float(_harmonic_plausibility(segment.chord, detected_key)),
					harmonic_plausibility_after=float(_harmonic_plausibility(sub.chord, detected_key)),
				)
			)

	return _merge_adjacent_same_chord_segments(refined), events


def _sustained_harmonic_drift_subsegments(
	segment: ChordSegment,
	*,
	chroma: np.ndarray,
	low_chroma: np.ndarray,
	boundaries: np.ndarray,
	hop_length: int,
	sample_rate: int,
	detected_key: KeyEstimate | None,
) -> list[ChordSegment]:
	start_frame = int(max(0, round(segment.start * sample_rate / hop_length)))
	end_frame = int(min(chroma.shape[1], round(segment.end * sample_rate / hop_length)))
	if end_frame <= start_frame:
		return [segment]

	frame_edges = [start_frame]
	for boundary in boundaries.tolist():
		frame = int(boundary)
		if start_frame < frame < end_frame:
			frame_edges.append(frame)
	step_frames = max(1, int(round(HARMONIC_DRIFT_WINDOW_SECONDS * sample_rate / hop_length)))
	for frame in range(start_frame + step_frames, end_frame, step_frames):
		frame_edges.append(int(frame))
	frame_edges.append(end_frame)
	frame_edges = sorted(set(frame_edges))
	if len(frame_edges) < 3:
		return [segment]

	parent_plausibility = _harmonic_plausibility(segment.chord, detected_key)
	raw: list[ChordSegment] = []
	for left, right in zip(frame_edges[:-1], frame_edges[1:], strict=False):
		if right <= left:
			continue
		region_chroma = np.asarray(chroma[:, left:right], dtype=np.float32)
		region_low = np.asarray(low_chroma[:, left:right], dtype=np.float32)
		root_aware = _estimate_region_root_aware_identity(region_chroma, region_low, key_estimate=detected_key)
		scores = dict(root_aware["combined_scores"])
		winner = str(root_aware["winner_label"])
		winner_score = float(scores.get(winner, 0.0))
		parent_score = float(scores.get(segment.chord, 0.0))
		winner_plausibility = _harmonic_plausibility(winner, detected_key)
		required_score = HARMONIC_DRIFT_MIN_SCORE
		required_advantage = HARMONIC_DRIFT_MIN_ADVANTAGE
		if not _is_harmonic_drift_candidate_allowed(segment.chord, winner, detected_key):
			required_score += HARMONIC_DRIFT_NON_DIATONIC_SCORE_BONUS
			required_advantage += HARMONIC_DRIFT_NON_DIATONIC_ADVANTAGE_BONUS
		if (
			winner != segment.chord
			and winner_score >= required_score
			and (winner_score - parent_score) >= required_advantage
			and winner_plausibility >= parent_plausibility - HARMONIC_DRIFT_MAX_PLAUSIBILITY_DROP
		):
			label = winner
			confidence = winner_score
		else:
			label = segment.chord
			confidence = max(float(segment.confidence), parent_score)

		raw.append(
			ChordSegment(
				start=float(left * hop_length / sample_rate),
				end=float(right * hop_length / sample_rate),
				chord=label,
				confidence=float(np.clip(confidence, 0.0, 1.0)),
			)
		)

	collapsed = _merge_adjacent_same_chord_segments(raw)
	min_sustained_seconds = _sustained_harmonic_drift_min_seconds(
		boundaries,
		hop_length=hop_length,
		sample_rate=sample_rate,
	)
	cleaned: list[ChordSegment] = []
	for child in collapsed:
		child_duration = _segment_duration(child)
		if child.chord == segment.chord or child_duration >= min_sustained_seconds:
			cleaned.append(child)
			continue
		if cleaned:
			prev = cleaned[-1]
			cleaned[-1] = ChordSegment(
				start=prev.start,
				end=child.end,
				chord=prev.chord,
				confidence=_weighted_confidence(prev, child),
			)
		else:
			cleaned.append(
				ChordSegment(
					start=child.start,
					end=child.end,
					chord=segment.chord,
					confidence=segment.confidence,
				)
			)

	cleaned = _merge_adjacent_same_chord_segments(cleaned)
	if len(cleaned) <= 1:
		return [segment]

	first = cleaned[0]
	last = cleaned[-1]
	cleaned[0] = ChordSegment(start=segment.start, end=first.end, chord=first.chord, confidence=first.confidence)
	cleaned[-1] = ChordSegment(start=last.start, end=segment.end, chord=last.chord, confidence=last.confidence)
	return _merge_adjacent_same_chord_segments(cleaned)


def _is_harmonic_drift_candidate_allowed(
	parent_chord: str,
	candidate_chord: str,
	detected_key: KeyEstimate | None,
) -> bool:
	if candidate_chord == parent_chord:
		return True
	if candidate_chord == "N":
		return False
	if _is_root_preserving_quality_switch(parent_chord, candidate_chord):
		return True
	if detected_key is None:
		return True
	if _is_diatonic_chord(candidate_chord, detected_key):
		return True
	return False


def _sustained_harmonic_drift_min_seconds(
	boundaries: np.ndarray,
	*,
	hop_length: int,
	sample_rate: int,
) -> float:
	if boundaries.size < 2:
		return HARMONIC_DRIFT_MIN_CHILD_SECONDS
	durations = [
		float(max(1, int(right) - int(left)) * hop_length / sample_rate)
		for left, right in zip(boundaries[:-1], boundaries[1:], strict=False)
		if int(right) > int(left)
	]
	if not durations:
		return HARMONIC_DRIFT_MIN_CHILD_SECONDS
	one_beat = float(median(durations) * HARMONIC_DRIFT_MIN_BEAT_RATIO)
	return float(np.clip(one_beat, HARMONIC_DRIFT_MIN_CHILD_SECONDS, 1.10))


def _refine_low_confidence_long_holds_with_key_candidates(
	segments: list[ChordSegment],
	*,
	chroma: np.ndarray,
	low_chroma: np.ndarray,
	boundaries: np.ndarray,
	hop_length: int,
	sample_rate: int,
	detected_key: KeyEstimate | None,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	"""Target only suspicious long, low-confidence holds using chords from the active key."""
	if len(segments) == 0 or detected_key is None:
		return segments, []

	refined: list[ChordSegment] = []
	events: list[CorrectionEvent] = []
	key_candidates = _diatonic_chord_labels_for_key(detected_key)
	if not key_candidates:
		return segments, []

	for segment_idx, segment in enumerate(segments):
		duration = _segment_duration(segment)
		if (
			segment.chord == "N"
			or duration < TARGETED_LONG_HOLD_MIN_SECONDS
			or segment.confidence > TARGETED_LONG_HOLD_MAX_CONFIDENCE
		):
			refined.append(segment)
			continue

		subsegments = _targeted_key_candidate_subsegments(
			segment,
			chroma=chroma,
			low_chroma=low_chroma,
			boundaries=boundaries,
			hop_length=hop_length,
			sample_rate=sample_rate,
			detected_key=detected_key,
			key_candidates=key_candidates,
		)
		if len(subsegments) <= 1:
			refined.append(segment)
			continue

		refined.extend(subsegments)
		for sub in subsegments:
			if sub.chord == segment.chord:
				continue
			events.append(
				CorrectionEvent(
					index=segment_idx,
					replaced_chord=segment.chord,
					new_chord=sub.chord,
					start=float(sub.start),
					end=float(sub.end),
					duration=float(_segment_duration(sub)),
					original_confidence=float(segment.confidence),
					new_confidence=float(sub.confidence),
					score=float(sub.confidence),
					harmonic_plausibility_before=float(_harmonic_plausibility(segment.chord, detected_key)),
					harmonic_plausibility_after=float(_harmonic_plausibility(sub.chord, detected_key)),
				)
			)

	return _merge_adjacent_same_chord_segments(refined), events


def _targeted_key_candidate_subsegments(
	segment: ChordSegment,
	*,
	chroma: np.ndarray,
	low_chroma: np.ndarray,
	boundaries: np.ndarray,
	hop_length: int,
	sample_rate: int,
	detected_key: KeyEstimate,
	key_candidates: set[str],
) -> list[ChordSegment]:
	start_frame = int(max(0, round(segment.start * sample_rate / hop_length)))
	end_frame = int(min(chroma.shape[1], round(segment.end * sample_rate / hop_length)))
	if end_frame <= start_frame:
		return [segment]

	frame_edges = _targeted_long_hold_grid_edges(
		start_frame,
		end_frame,
		boundaries,
		sample_rate=sample_rate,
		hop_length=hop_length,
	)
	if len(frame_edges) < 3:
		return [segment]

	raw: list[ChordSegment] = []
	for left, right in zip(frame_edges[:-1], frame_edges[1:], strict=False):
		if right <= left:
			continue
		region_chroma = np.asarray(chroma[:, left:right], dtype=np.float32)
		region_low = np.asarray(low_chroma[:, left:right], dtype=np.float32)
		root_aware = _estimate_region_root_aware_identity(region_chroma, region_low, key_estimate=detected_key)
		scores = {
			label: float(score)
			for label, score in dict(root_aware["combined_scores"]).items()
			if label in key_candidates
		}
		if segment.chord in dict(root_aware["combined_scores"]):
			scores[segment.chord] = max(scores.get(segment.chord, 0.0), float(dict(root_aware["combined_scores"])[segment.chord]))
		if not scores:
			label = segment.chord
			confidence = segment.confidence
		else:
			ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
			winner, winner_score = ranked[0]
			parent_score = float(scores.get(segment.chord, 0.0))
			if (
				winner != segment.chord
				and winner_score >= TARGETED_LONG_HOLD_MIN_SCORE - TARGETED_LONG_HOLD_GRID_SCORE_RELAX
				and (winner_score - parent_score) >= TARGETED_LONG_HOLD_MIN_ADVANTAGE - TARGETED_LONG_HOLD_GRID_ADVANTAGE_RELAX
			):
				label = winner
				confidence = winner_score
			else:
				label = segment.chord
				confidence = max(segment.confidence, parent_score)

		raw.append(
			ChordSegment(
				start=float(left * hop_length / sample_rate),
				end=float(right * hop_length / sample_rate),
				chord=label,
				confidence=float(np.clip(confidence, 0.0, 1.0)),
			)
		)

	collapsed = _merge_adjacent_same_chord_segments(raw)
	cleaned = _keep_only_sustained_targeted_alternates(collapsed, parent=segment)
	if len(cleaned) <= 1:
		return [segment]

	first = cleaned[0]
	last = cleaned[-1]
	cleaned[0] = ChordSegment(start=segment.start, end=first.end, chord=first.chord, confidence=first.confidence)
	cleaned[-1] = ChordSegment(start=last.start, end=segment.end, chord=last.chord, confidence=last.confidence)
	return _merge_adjacent_same_chord_segments(cleaned)


def _keep_only_sustained_targeted_alternates(
	segments: list[ChordSegment],
	*,
	parent: ChordSegment,
) -> list[ChordSegment]:
	cleaned: list[ChordSegment] = []
	min_child_seconds = max(TARGETED_LONG_HOLD_MIN_CHILD_SECONDS, TARGETED_LONG_HOLD_MIN_ALTERNATE_RUN_SECONDS)
	for child in segments:
		child_duration = _segment_duration(child)
		if child.chord == parent.chord or child_duration >= min_child_seconds:
			cleaned.append(child)
			continue
		if cleaned:
			prev = cleaned[-1]
			cleaned[-1] = ChordSegment(
				start=prev.start,
				end=child.end,
				chord=prev.chord,
				confidence=_weighted_confidence(prev, child),
			)
		else:
			cleaned.append(
				ChordSegment(
					start=child.start,
					end=child.end,
					chord=parent.chord,
					confidence=parent.confidence,
				)
			)
	return _merge_adjacent_same_chord_segments(cleaned)


def _targeted_long_hold_grid_edges(
	start_frame: int,
	end_frame: int,
	boundaries: np.ndarray,
	*,
	sample_rate: int,
	hop_length: int,
) -> list[int]:
	inside = [
		int(frame)
		for frame in boundaries.tolist()
		if start_frame < int(frame) < end_frame
	]
	if len(inside) >= TARGETED_LONG_HOLD_GRID_BEATS:
		grid_edges = [start_frame]
		for idx in range(TARGETED_LONG_HOLD_GRID_BEATS - 1, len(inside), TARGETED_LONG_HOLD_GRID_BEATS):
			grid_edges.append(inside[idx])
		grid_edges.append(end_frame)
		grid_edges = sorted(set(grid_edges))
		if len(grid_edges) >= 3:
			return grid_edges

	step_frames = max(1, int(round(TARGETED_LONG_HOLD_WINDOW_SECONDS * sample_rate / hop_length)))
	frame_edges = list(range(start_frame, end_frame, step_frames))
	if not frame_edges or frame_edges[0] != start_frame:
		frame_edges.insert(0, start_frame)
	if frame_edges[-1] != end_frame:
		frame_edges.append(end_frame)
	return sorted(set(frame_edges))
