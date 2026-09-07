"""Mini-sequence decoder for refining suspicious long chord holds."""

from __future__ import annotations

import numpy as np

from chord_engine.analysis_config import *  # noqa: F403 - shared mini-sequence tuning constants
from chord_engine.analysis_models import CorrectionEvent
from chord_engine.detector import KeyEstimate
from chord_engine.music_theory import (
    _diatonic_chord_labels_for_key,
    _harmonic_plausibility,
    _harmonic_relationship_strength,
)
from chord_engine.root_aware import _estimate_region_root_aware_identity
from chord_engine.segment_utils import (
    _merge_adjacent_same_chord_segments,
    _segment_duration,
    _weighted_confidence,
)
from chord_engine.segmentation import ChordSegment

def _refine_long_holds_with_mini_harmonic_sequence_decoder(
	segments: list[ChordSegment],
	*,
	chroma: np.ndarray,
	low_chroma: np.ndarray,
	boundaries: np.ndarray,
	hop_length: int,
	sample_rate: int,
	detected_key: KeyEstimate | None,
) -> tuple[list[ChordSegment], list[CorrectionEvent]]:
	"""Decode a small diatonic chord sequence inside suspicious long holds."""
	if len(segments) == 0 or detected_key is None:
		return segments, []

	key_candidates = _diatonic_chord_labels_for_key(detected_key)
	if not key_candidates:
		return segments, []

	refined: list[ChordSegment] = []
	events: list[CorrectionEvent] = []
	for segment_idx, segment in enumerate(segments):
		if (
			segment.chord == "N"
			or _segment_duration(segment) < MINI_SEQUENCE_DECODER_MIN_SECONDS
			or segment.confidence > MINI_SEQUENCE_DECODER_MAX_CONFIDENCE
		):
			refined.append(segment)
			continue

		subsegments = _mini_sequence_decode_long_hold(
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

def _mini_sequence_decode_long_hold(
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

	frame_edges = _mini_sequence_grid_edges(start_frame, end_frame, boundaries)
	if len(frame_edges) < 4:
		return [segment]

	candidate_grid: list[list[tuple[str, float]]] = []
	for left, right in zip(frame_edges[:-1], frame_edges[1:], strict=False):
		region_chroma = np.asarray(chroma[:, left:right], dtype=np.float32)
		region_low = np.asarray(low_chroma[:, left:right], dtype=np.float32)
		root_aware = _estimate_region_root_aware_identity(region_chroma, region_low, key_estimate=detected_key)
		all_scores = dict(root_aware["combined_scores"])
		filtered = {
			label: float(score)
			for label, score in all_scores.items()
			if label in key_candidates or label == segment.chord
		}
		if segment.chord in all_scores:
			filtered[segment.chord] = max(filtered.get(segment.chord, 0.0), float(all_scores[segment.chord]) + MINI_SEQUENCE_DECODER_PARENT_BIAS)
		if not filtered:
			filtered = {segment.chord: segment.confidence}
		ranked = sorted(filtered.items(), key=lambda item: (-item[1], item[0]))[:MINI_SEQUENCE_DECODER_TOP_K]
		if segment.chord not in {label for label, _ in ranked}:
			ranked.append((segment.chord, float(filtered.get(segment.chord, segment.confidence))))
		candidate_grid.append(ranked)

	path = _decode_mini_sequence_path(candidate_grid, detected_key)
	if len(path) != len(candidate_grid):
		return [segment]

	raw: list[ChordSegment] = []
	for (left, right), (label, score) in zip(zip(frame_edges[:-1], frame_edges[1:], strict=False), path, strict=True):
		raw.append(
			ChordSegment(
				start=float(left * hop_length / sample_rate),
				end=float(right * hop_length / sample_rate),
				chord=label,
				confidence=float(np.clip(score, 0.0, 1.0)),
			)
		)

	collapsed = _merge_adjacent_same_chord_segments(raw)
	cleaned = _keep_only_sustained_mini_sequence_alternates(collapsed, parent=segment)
	if len(cleaned) <= 1:
		return [segment]

	total_alternate = sum(_segment_duration(child) for child in cleaned if child.chord != segment.chord)
	if total_alternate < MINI_SEQUENCE_DECODER_MIN_TOTAL_ALTERNATE_SECONDS:
		return [segment]

	first = cleaned[0]
	last = cleaned[-1]
	cleaned[0] = ChordSegment(start=segment.start, end=first.end, chord=first.chord, confidence=first.confidence)
	cleaned[-1] = ChordSegment(start=last.start, end=segment.end, chord=last.chord, confidence=last.confidence)
	return _merge_adjacent_same_chord_segments(cleaned)

def _decode_mini_sequence_path(
	candidate_grid: list[list[tuple[str, float]]],
	detected_key: KeyEstimate,
) -> list[tuple[str, float]]:
	if not candidate_grid:
		return []

	dp: list[dict[str, tuple[float, str | None, float]]] = []
	first_state: dict[str, tuple[float, str | None, float]] = {}
	for chord, score in candidate_grid[0]:
		first_state[chord] = (float(score), None, float(score))
	dp.append(first_state)

	for idx in range(1, len(candidate_grid)):
		state: dict[str, tuple[float, str | None, float]] = {}
		for chord, score in candidate_grid[idx]:
			best_score: float | None = None
			best_prev: str | None = None
			for prev_chord, (prev_score, _, _) in dp[idx - 1].items():
				transition = _mini_sequence_transition_score(prev_chord, chord, detected_key)
				candidate_score = prev_score + float(score) + transition
				if best_score is None or candidate_score > best_score:
					best_score = candidate_score
					best_prev = prev_chord
			if best_score is not None:
				state[chord] = (float(best_score), best_prev, float(score))
		dp.append(state)

	if not dp[-1]:
		return []
	best_last = sorted(dp[-1].items(), key=lambda item: (-item[1][0], item[0]))[0][0]
	labels = [best_last]
	for idx in range(len(dp) - 1, 0, -1):
		prev = dp[idx][labels[-1]][1]
		if prev is None:
			break
		labels.append(prev)
	labels.reverse()
	if len(labels) != len(candidate_grid):
		return []
	return [(label, dp[idx][label][2]) for idx, label in enumerate(labels)]

def _mini_sequence_transition_score(prev_chord: str, curr_chord: str, detected_key: KeyEstimate) -> float:
	if prev_chord == curr_chord:
		return 0.018
	relationship = _harmonic_relationship_strength(prev_chord, curr_chord, detected_key)
	return float((MINI_SEQUENCE_DECODER_RELATIONSHIP_WEIGHT * relationship) - MINI_SEQUENCE_DECODER_CHANGE_COST)

def _mini_sequence_grid_edges(start_frame: int, end_frame: int, boundaries: np.ndarray) -> list[int]:
	inside = [
		int(frame)
		for frame in boundaries.tolist()
		if start_frame < int(frame) < end_frame
	]
	if len(inside) >= MINI_SEQUENCE_DECODER_GRID_BEATS:
		edges = [start_frame]
		for idx in range(MINI_SEQUENCE_DECODER_GRID_BEATS - 1, len(inside), MINI_SEQUENCE_DECODER_GRID_BEATS):
			edges.append(inside[idx])
		edges.append(end_frame)
		return sorted(set(edges))
	return [start_frame, end_frame]

def _keep_only_sustained_mini_sequence_alternates(
	segments: list[ChordSegment],
	*,
	parent: ChordSegment,
) -> list[ChordSegment]:
	cleaned: list[ChordSegment] = []
	for child in segments:
		if child.chord == parent.chord or _segment_duration(child) >= MINI_SEQUENCE_DECODER_MIN_ALTERNATE_SECONDS:
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
