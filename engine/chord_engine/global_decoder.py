"""Global chord sequence decoding for harmonic regions."""

from __future__ import annotations

import numpy as np

from chord_engine.analysis_config import *  # noqa: F403 - shared decoder tuning constants
from chord_engine.analysis_models import MusicalRegionObservation
from chord_engine.detector import KeyEstimate
from chord_engine.music_theory import (
    _harmonic_relationship_strength,
    _is_diatonic_chord,
    _is_root_preserving_quality_switch,
)
from chord_engine.segment_utils import _merge_adjacent_same_chord_segments
from chord_engine.segmentation import ChordSegment

def _decode_global_chord_sequence(
	regions: list[MusicalRegionObservation],
	*,
	detected_key: KeyEstimate | None,
) -> tuple[list[MusicalRegionObservation], float, int]:
	if len(regions) == 0:
		return [], 0.0, 0

	region_candidates: list[list[tuple[str, float]]] = []
	for obs in regions:
		ranked = sorted(obs.scores.items(), key=lambda item: (-item[1], item[0]))
		candidates = ranked[:GLOBAL_DECODE_TOP_K]
		if obs.local_best_chord is not None and obs.local_best_chord not in {name for name, _ in candidates}:
			candidates.append((obs.local_best_chord, obs.local_best_score))
		region_candidates.append(candidates)

	dp: list[dict[str, tuple[float, str | None]]] = []
	first_state: dict[str, tuple[float, str | None]] = {}
	for chord, score in region_candidates[0]:
		first_state[chord] = (_local_evidence_score(regions[0], chord, score, detected_key), None)
	dp.append(first_state)

	for idx in range(1, len(regions)):
		state: dict[str, tuple[float, str | None]] = {}
		for chord, score in region_candidates[idx]:
			local_score = _local_evidence_score(regions[idx], chord, score, detected_key)
			best_prev_score = None
			best_prev_chord: str | None = None
			for prev_chord, (prev_score, _) in dp[idx - 1].items():
				candidate_score = prev_score + local_score + _transition_score(prev_chord, chord, regions[idx], detected_key)
				if best_prev_score is None or candidate_score > best_prev_score:
					best_prev_score = candidate_score
					best_prev_chord = prev_chord
			if best_prev_score is not None:
				state[chord] = (float(best_prev_score), best_prev_chord)
		dp.append(state)

	last_state = dp[-1]
	best_last_chord = sorted(last_state.items(), key=lambda item: (-item[1][0], item[0]))[0][0]
	best_path_score = float(last_state[best_last_chord][0])

	decoded = [best_last_chord]
	for idx in range(len(regions) - 1, 0, -1):
		_, prev_chord = dp[idx][decoded[-1]]
		if prev_chord is None:
			break
		decoded.append(prev_chord)
	decoded.reverse()

	if len(decoded) != len(regions):
		decoded = [obs.local_best_chord or obs.winner_chord for obs in regions]

	out: list[MusicalRegionObservation] = []
	changed = 0
	for obs, label in zip(regions, decoded, strict=True):
		if label != obs.local_best_chord:
			changed += 1
		out.append(
			MusicalRegionObservation(
				start_frame=obs.start_frame,
				end_frame=obs.end_frame,
				start_seconds=obs.start_seconds,
				end_seconds=obs.end_seconds,
				duration_seconds=obs.duration_seconds,
				winner_chord=obs.winner_chord,
				winner_confidence=obs.winner_confidence,
				scores=obs.scores,
				advantage_over_runner_up=obs.advantage_over_runner_up,
				root_candidate_scores=obs.root_candidate_scores,
				selected_root=obs.selected_root,
				selected_root_confidence=obs.selected_root_confidence,
				major_quality_evidence=obs.major_quality_evidence,
				minor_quality_evidence=obs.minor_quality_evidence,
				quality_margin=obs.quality_margin,
				template_score=obs.template_score,
				template_top_chord=obs.template_top_chord,
				template_top_score=obs.template_top_score,
				combined_score=obs.combined_score,
				is_quality_ambiguous=obs.is_quality_ambiguous,
				top_candidates=obs.top_candidates,
				local_best_chord=obs.local_best_chord,
				local_best_score=obs.local_best_score,
				decoded_chord=label,
				beat_length=obs.beat_length,
			)
		)

	return out, best_path_score, changed


def _local_evidence_score(
	obs: MusicalRegionObservation,
	chord: str,
	base_score: float,
	detected_key: KeyEstimate | None,
) -> float:
	ranked = sorted(obs.scores.items(), key=lambda item: (-item[1], item[0]))
	best = float(ranked[0][1]) if ranked else 0.0
	second = float(ranked[1][1]) if len(ranked) > 1 else 0.0
	margin = max(0.0, best - second)
	deviation_penalty = max(0.0, (obs.local_best_score - base_score) * 1.15)
	score = float(base_score + (GLOBAL_DECODE_EVIDENCE_MARGIN_WEIGHT * margin) - deviation_penalty)
	if (
		detected_key is not None
		and chord != "N"
		and obs.duration_seconds < 1.0
		and not _is_diatonic_chord(chord, detected_key)
	):
		# Short non-diatonic regions are often ornamental tones; dampen switch pressure.
		short_factor = float(np.clip((1.0 - obs.duration_seconds) / 1.0, 0.0, 1.0))
		score -= (0.16 + (0.06 if obs.is_quality_ambiguous else 0.0)) * short_factor
	if chord == "N" and obs.duration_seconds < 0.30:
		score -= 0.06
	return score


def _transition_score(
	prev_chord: str,
	curr_chord: str,
	obs: MusicalRegionObservation,
	detected_key: KeyEstimate | None,
) -> float:
	if prev_chord == curr_chord:
		return GLOBAL_DECODE_SELF_STABILITY_BONUS

	if (
		_is_root_preserving_quality_switch(prev_chord, curr_chord)
		and obs.local_best_chord == curr_chord
		and obs.winner_chord == curr_chord
		and obs.quality_margin >= GLOBAL_DECODE_QUALITY_SWITCH_STRONG_QMARGIN_MIN
		and obs.advantage_over_runner_up >= GLOBAL_DECODE_QUALITY_SWITCH_STRONG_MARGIN_MIN
	):
		return GLOBAL_DECODE_QUALITY_SWITCH_POSITIVE_TRANSITION

	relationship = _harmonic_relationship_strength(prev_chord, curr_chord, detected_key)
	confidence_bonus = GLOBAL_DECODE_LOCAL_CONFIDENCE_WEIGHT * obs.winner_confidence
	switch_bonus = 0.0
	if obs.local_best_chord == curr_chord and obs.advantage_over_runner_up >= GLOBAL_DECODE_STRONG_SWITCH_ADVANTAGE:
		switch_bonus += GLOBAL_DECODE_STRONG_SWITCH_BONUS
	if _is_root_preserving_quality_switch(prev_chord, curr_chord) and obs.quality_margin >= 0.12:
		switch_bonus += GLOBAL_DECODE_QUALITY_SWITCH_BONUS
	penalty = GLOBAL_DECODE_CHANGE_BASE_COST - (GLOBAL_DECODE_RELATIONSHIP_WEIGHT * relationship) - confidence_bonus - switch_bonus
	if (
		detected_key is not None
		and curr_chord != "N"
		and obs.duration_seconds < CONTEXT_SHORT_SEGMENT_MAX_SECONDS
		and not _is_diatonic_chord(curr_chord, detected_key)
	):
		penalty += 0.12 * float(np.clip((CONTEXT_SHORT_SEGMENT_MAX_SECONDS - obs.duration_seconds) / CONTEXT_SHORT_SEGMENT_MAX_SECONDS, 0.0, 1.0))
	penalty = float(max(0.02, penalty))
	return -penalty


def _regions_to_segments(
	regions: list[MusicalRegionObservation],
	*,
	hop_length: int,
	sample_rate: int,
	source_duration: float,
) -> list[ChordSegment]:
	if not regions:
		return [ChordSegment(start=0.0, end=max(1e-9, source_duration), chord="N", confidence=1.0)]

	segments: list[ChordSegment] = []
	for obs in regions:
		chord = obs.decoded_chord or obs.local_best_chord or obs.winner_chord
		confidence = float(np.clip(obs.scores.get(chord, obs.winner_confidence), 0.0, 1.0))
		segments.append(
			ChordSegment(
				start=float(obs.start_frame * hop_length / sample_rate),
				end=float(obs.end_frame * hop_length / sample_rate),
				chord=chord,
				confidence=confidence,
			)
		)

	merged = _merge_adjacent_same_chord_segments(segments)
	if merged:
		first = merged[0]
		if first.start > 0.0:
			merged[0] = ChordSegment(start=0.0, end=first.end, chord=first.chord, confidence=first.confidence)
		last = merged[-1]
		merged[-1] = ChordSegment(start=last.start, end=max(last.start + 1e-9, source_duration), chord=last.chord, confidence=last.confidence)
	return merged


def _collapse_regions_by_decoded_chord(regions: list[MusicalRegionObservation]) -> list[MusicalRegionObservation]:
	if len(regions) <= 1:
		return regions

	collapsed: list[MusicalRegionObservation] = [regions[0]]
	for obs in regions[1:]:
		prev = collapsed[-1]
		prev_label = prev.decoded_chord or prev.local_best_chord or prev.winner_chord
		curr_label = obs.decoded_chord or obs.local_best_chord or obs.winner_chord
		if prev_label != curr_label:
			collapsed.append(obs)
			continue

		merged_duration = prev.duration_seconds + obs.duration_seconds
		merged_confidence = prev.winner_confidence if merged_duration <= 0 else (
			(prev.winner_confidence * prev.duration_seconds + obs.winner_confidence * obs.duration_seconds) / merged_duration
		)
		collapsed[-1] = MusicalRegionObservation(
			start_frame=prev.start_frame,
			end_frame=obs.end_frame,
			start_seconds=prev.start_seconds,
			end_seconds=obs.end_seconds,
			duration_seconds=merged_duration,
			winner_chord=prev.winner_chord,
			winner_confidence=float(np.clip(merged_confidence, 0.0, 1.0)),
			scores=prev.scores,
			advantage_over_runner_up=prev.advantage_over_runner_up,
			root_candidate_scores=prev.root_candidate_scores,
			selected_root=prev.selected_root,
			selected_root_confidence=prev.selected_root_confidence,
			major_quality_evidence=prev.major_quality_evidence,
			minor_quality_evidence=prev.minor_quality_evidence,
			quality_margin=prev.quality_margin,
			template_score=prev.template_score,
			template_top_chord=prev.template_top_chord,
			template_top_score=prev.template_top_score,
			combined_score=prev.combined_score,
			is_quality_ambiguous=prev.is_quality_ambiguous,
			top_candidates=prev.top_candidates,
			local_best_chord=prev.local_best_chord,
			local_best_score=prev.local_best_score,
			decoded_chord=prev_label,
			beat_length=prev.beat_length + obs.beat_length,
		)

	return collapsed
