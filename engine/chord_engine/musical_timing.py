"""Musical timing diagnostics for beat, downbeat, and pickup alignment.

This module is diagnostic-only. It explains whether the analysis grid is likely
musically aligned before any arranger is allowed to use the grid as output.
"""

from __future__ import annotations

from itertools import product
from statistics import mean

import numpy as np

from chord_engine.detector import KeyEstimate
from chord_engine.features import BeatTiming, beat_times_seconds
from chord_engine.music_theory import (
	_chord_root_pc,
	_diatonic_chord_labels_for_key,
	_harmonic_relationship_strength,
	_is_diatonic_chord,
)
from chord_engine.numeric import _cosine_similarity
from chord_engine.segmentation import ChordSegment
from chord_engine.templates import generate_chord_templates
from chord_engine.timebase import frame_duration_seconds

BAR_BEATS = 4
MUSICAL_START_SCAN_SECONDS = 12.0
MUSICAL_START_MIN_SEGMENT_SECONDS = 0.85
MUSICAL_START_MIN_CONFIDENCE = 0.48
GRID_ALIGNMENT_WINDOW_SECONDS = 0.55
DIAGNOSTIC_WINDOW_COUNT = 8
BAR_CANDIDATE_COUNT = 16
BAR_TOP_CHORD_COUNT = 5
PHASE_CANDIDATE_COUNT = 4
SEQUENCE_SELF_BONUS = 0.10
SEQUENCE_CHANGE_COST = 0.10
SEQUENCE_RELATIONSHIP_WEIGHT = 0.34
SEQUENCE_DIATONIC_BONUS = 0.12
SEQUENCE_REPEAT_PATTERN_BONUS = 0.16
SEQUENCE_OVERLONG_REPEAT_PENALTY = 0.08
PHRASE_PATTERN_MIN_LENGTH = 3
PHRASE_PATTERN_MAX_LENGTH = 6
PHRASE_PATTERN_ANALYSIS_BARS = 12
PHRASE_PATTERN_SLOT_TOP_K = 4
PHRASE_PATTERN_MIN_SCORE = 0.47
PHRASE_PATTERN_MIN_MARGIN = 0.035
PHRASE_PATTERN_LENGTH_FOUR_BONUS = 0.11
PHRASE_PATTERN_DIVERSITY_BONUS = 0.075
PHRASE_PATTERN_STAGNATION_PENALTY = 0.16
PHRASE_PATTERN_EVIDENCE_WEIGHT = 0.72
PHRASE_PATTERN_RELATIONSHIP_WEIGHT = 0.18
PHRASE_PATTERN_KEY_WEIGHT = 0.10
PHRASE_PATTERN_PARTIAL_BAR_RATIO = 0.55
PHRASE_PATTERN_FUNCTION_COVERAGE_BONUS = 0.18
PHRASE_PATTERN_MISSING_DOMINANT_PENALTY = 0.14
PHRASE_PATTERN_FIRST_SLOT_ANCHOR_BONUS = 0.12
PHRASE_PATTERN_FIRST_SLOT_MISMATCH_PENALTY = 0.34
PHASE_EARLY_STAGNATION_PENALTY = 0.72
PHASE_EARLY_DIVERSITY_BONUS = 0.42
MULTIRES_FULL_CHROMA_WEIGHT = 0.34
MULTIRES_OVERLAP_WEIGHT = 0.20
MULTIRES_SUBWINDOW_WEIGHT = 0.26
MULTIRES_KEY_WEIGHT = 0.20
LEADSHEET_FULL_CHROMA_WEIGHT = 0.32
LEADSHEET_OVERLAP_WEIGHT = 0.08
LEADSHEET_SUBWINDOW_WEIGHT = 0.38
LEADSHEET_KEY_WEIGHT = 0.22
DOUBLE_TIME_TEMPO_BPM = 150.0
HALF_TIME_TEMPO_BPM = 65.0
MIN_NORMALIZED_BAR_SECONDS = 2.0
MAX_NORMALIZED_BAR_SECONDS = 4.4
HARMONIC_START_MIN_DOMINANCE = 0.28
HARMONIC_START_MIN_AGREEMENT = 0.56
HARMONIC_START_WINDOW_SECONDS = 0.90
HARMONIC_START_STEP_SECONDS = 0.25
HARMONIC_START_CONSECUTIVE_WINDOWS = 2


def summarize_musical_timing(
	segments: list[ChordSegment],
	*,
	beat_timing: BeatTiming | None,
	duration_seconds: float,
	hop_length: int,
	sample_rate: int,
	chroma: np.ndarray | None = None,
	detected_key: KeyEstimate | None = None,
) -> dict[str, object]:
	"""Return deterministic timing diagnostics without mutating chord output."""

	musical_start = estimate_musical_start_time(
		segments,
		chroma=chroma,
		hop_length=hop_length,
		sample_rate=sample_rate,
	)
	beat_seconds = _beat_seconds(beat_timing, hop_length=hop_length, sample_rate=sample_rate)
	raw_beat_interval = _beat_interval_seconds(beat_timing)
	tempo = _normalized_tempo(beat_timing, raw_beat_interval=raw_beat_interval)
	beat_interval = tempo["beatIntervalSeconds"]
	phase_candidates = _downbeat_phase_candidates(
		segments,
		beat_seconds=beat_seconds,
		beat_interval=beat_interval,
		musical_start=musical_start,
		duration_seconds=duration_seconds,
		chroma=chroma,
		hop_length=hop_length,
		sample_rate=sample_rate,
		detected_key=detected_key,
	)
	downbeat = phase_candidates[0]["downbeat"] if phase_candidates else None
	bar_length = beat_interval * BAR_BEATS if beat_interval is not None else None
	bar_windows = _bar_windows(downbeat, bar_length=bar_length, duration_seconds=duration_seconds)
	alignment_errors = _grid_alignment_errors(segments, downbeat=downbeat, bar_length=bar_length)

	return {
		"diagnosticOnly": True,
		"tempoBpm": tempo["normalizedTempoBpm"],
		"rawTempoBpm": tempo["rawTempoBpm"],
		"tempoNormalization": tempo["normalization"],
		"beatReliable": bool(beat_timing.is_reliable) if beat_timing is not None else False,
		"beatCount": int(beat_timing.beat_count) if beat_timing is not None else 0,
		"musicalStartSeconds": round(musical_start, 3),
		"firstBeatSeconds": round(beat_seconds[0], 3) if beat_seconds else None,
		"downbeatSeconds": round(downbeat, 3) if downbeat is not None else None,
		"downbeatPhaseCandidates": [
			{
				"phase": candidate["phase"],
				"downbeat": round(float(candidate["downbeat"]), 3),
				"score": round(float(candidate["score"]), 3),
				"winners": candidate["winners"],
			}
			for candidate in phase_candidates
		],
		"barLengthSeconds": round(bar_length, 3) if bar_length is not None else None,
		"rawBarLengthSeconds": round(raw_beat_interval * BAR_BEATS, 3) if raw_beat_interval is not None else None,
		"initialPickupSeconds": round(max(0.0, musical_start), 3),
		"meanGridAlignmentErrorSeconds": round(float(mean(alignment_errors)), 3) if alignment_errors else None,
		"maxGridAlignmentErrorSeconds": round(max(alignment_errors), 3) if alignment_errors else None,
		"firstBarWindows": [
			{"start": round(start, 3), "end": round(end, 3)}
			for start, end in bar_windows[:DIAGNOSTIC_WINDOW_COUNT]
		],
		"barPhraseCandidates": _bar_phrase_candidates(
			segments,
			bar_windows=bar_windows,
			chroma=chroma,
			hop_length=hop_length,
			sample_rate=sample_rate,
			detected_key=detected_key,
		),
		"leadSheetBarCandidates": _lead_sheet_bar_candidates(
			segments,
			bar_windows=bar_windows,
			chroma=chroma,
			hop_length=hop_length,
			sample_rate=sample_rate,
			detected_key=detected_key,
		),
		"firstChordStarts": [
			{"time": round(segment.start, 3), "chord": segment.chord}
			for segment in segments
			if segment.chord != "N"
		][:DIAGNOSTIC_WINDOW_COUNT],
		"harmonicStartCandidates": _harmonic_start_candidates(
			chroma=chroma,
			hop_length=hop_length,
			sample_rate=sample_rate,
		),
	}


def estimate_musical_start_time(
	segments: list[ChordSegment],
	*,
	chroma: np.ndarray | None = None,
	hop_length: int | None = None,
	sample_rate: int | None = None,
) -> float:
	if not segments:
		return 0.0
	chroma_start = _estimate_chroma_musical_start(
		chroma=chroma,
		hop_length=hop_length,
		sample_rate=sample_rate,
	)
	if chroma_start is not None:
		return chroma_start
	first = segments[0]
	if _is_stable_harmonic_segment(first) and first.start <= 0.15 and _has_stable_harmonic_chroma(
		first,
		chroma=chroma,
		hop_length=hop_length,
		sample_rate=sample_rate,
	):
		return 0.0
	for segment in segments:
		if segment.start > MUSICAL_START_SCAN_SECONDS:
			break
		if _is_stable_harmonic_segment(segment) and _has_stable_harmonic_chroma(
			segment,
			chroma=chroma,
			hop_length=hop_length,
			sample_rate=sample_rate,
		):
			return float(segment.start)
	return 0.0


def _estimate_chroma_musical_start(
	*,
	chroma: np.ndarray | None,
	hop_length: int | None,
	sample_rate: int | None,
) -> float | None:
	if chroma is None or hop_length is None or sample_rate is None:
		return None
	if chroma.ndim != 2 or chroma.shape[0] != 12 or chroma.shape[1] <= 0:
		return None
	frame_seconds = frame_duration_seconds(hop_length=hop_length, sample_rate=sample_rate)
	if frame_seconds <= 0.0:
		return None
	duration = chroma.shape[1] * frame_seconds
	latest_start = min(MUSICAL_START_SCAN_SECONDS, duration)
	window_frames = max(2, int(round(HARMONIC_START_WINDOW_SECONDS / frame_seconds)))
	step_frames = max(1, int(round(HARMONIC_START_STEP_SECONDS / frame_seconds)))
	max_start_frame = min(chroma.shape[1] - 1, int(round(latest_start / frame_seconds)))
	consecutive = 0
	first_stable_start: float | None = None
	for start_frame in range(0, max_start_frame + 1, step_frames):
		end_frame = min(chroma.shape[1], start_frame + window_frames)
		if end_frame - start_frame < window_frames // 2:
			break
		start_seconds = start_frame * frame_seconds
		stats = _window_harmonic_stats(chroma[:, start_frame:end_frame])
		if stats is not None and stats["stable"]:
			if consecutive == 0:
				first_stable_start = start_seconds
			consecutive += 1
			if consecutive >= HARMONIC_START_CONSECUTIVE_WINDOWS:
				return 0.0 if first_stable_start is not None and first_stable_start <= 0.15 else float(first_stable_start or 0.0)
		else:
			consecutive = 0
			first_stable_start = None
	return None


def _is_stable_harmonic_segment(segment: ChordSegment) -> bool:
	return (
		segment.chord != "N"
		and segment.confidence >= MUSICAL_START_MIN_CONFIDENCE
		and segment.end - segment.start >= MUSICAL_START_MIN_SEGMENT_SECONDS
	)


def _has_stable_harmonic_chroma(
	segment: ChordSegment,
	*,
	chroma: np.ndarray | None,
	hop_length: int | None,
	sample_rate: int | None,
) -> bool:
	if chroma is None or hop_length is None or sample_rate is None:
		return True
	if chroma.ndim != 2 or chroma.shape[0] != 12 or chroma.shape[1] <= 0:
		return True
	frame_seconds = frame_duration_seconds(hop_length=hop_length, sample_rate=sample_rate)
	if frame_seconds <= 0.0:
		return True
	start_idx = max(0, int(np.floor(segment.start / frame_seconds)))
	end_idx = min(chroma.shape[1], int(np.ceil(segment.end / frame_seconds)))
	if end_idx <= start_idx:
		return True
	return _window_is_stable_harmonic(chroma[:, start_idx:end_idx])


def _window_is_stable_harmonic(window: np.ndarray) -> bool:
	stats = _window_harmonic_stats(window)
	return bool(stats is not None and stats["stable"])


def _window_harmonic_stats(window: np.ndarray) -> dict[str, float | bool] | None:
	window = np.asarray(window, dtype=np.float32)
	if window.ndim != 2 or window.shape[0] != 12 or window.shape[1] <= 0:
		return None
	mean_vector = np.mean(window, axis=1)
	norm = float(np.linalg.norm(mean_vector))
	if not np.isfinite(norm) or norm <= 1e-9:
		return None
	mean_vector = mean_vector / norm
	dominance = float(np.max(mean_vector))
	frame_norms = np.linalg.norm(window, axis=0)
	valid = frame_norms > 1e-9
	if not np.any(valid):
		return None
	similarities = np.dot(mean_vector, window[:, valid] / frame_norms[valid])
	agreement = float(np.mean(similarities))
	return {
		"dominance": dominance,
		"agreement": agreement,
		"stable": dominance >= HARMONIC_START_MIN_DOMINANCE and agreement >= HARMONIC_START_MIN_AGREEMENT,
	}


def _harmonic_start_candidates(
	*,
	chroma: np.ndarray | None,
	hop_length: int,
	sample_rate: int,
) -> list[dict[str, object]]:
	if chroma is None or chroma.ndim != 2 or chroma.shape[0] != 12 or chroma.shape[1] <= 0:
		return []
	frame_seconds = frame_duration_seconds(hop_length=hop_length, sample_rate=sample_rate)
	if frame_seconds <= 0.0:
		return []
	window_frames = max(2, int(round(HARMONIC_START_WINDOW_SECONDS / frame_seconds)))
	step_frames = max(1, int(round(HARMONIC_START_STEP_SECONDS / frame_seconds)))
	latest_start = min(MUSICAL_START_SCAN_SECONDS, chroma.shape[1] * frame_seconds)
	max_start_frame = min(chroma.shape[1] - 1, int(round(latest_start / frame_seconds)))
	candidates: list[dict[str, object]] = []
	for start_frame in range(0, max_start_frame + 1, step_frames):
		end_frame = min(chroma.shape[1], start_frame + window_frames)
		if end_frame - start_frame < window_frames // 2:
			break
		stats = _window_harmonic_stats(chroma[:, start_frame:end_frame])
		if stats is None:
			continue
		candidates.append(
			{
				"start": round(start_frame * frame_seconds, 3),
				"end": round(end_frame * frame_seconds, 3),
				"dominance": round(float(stats["dominance"]), 3),
				"agreement": round(float(stats["agreement"]), 3),
				"stable": bool(stats["stable"]),
			}
		)
		if len(candidates) >= DIAGNOSTIC_WINDOW_COUNT:
			break
	return candidates


def _bar_phrase_candidates(
	segments: list[ChordSegment],
	*,
	bar_windows: list[tuple[float, float]],
	chroma: np.ndarray | None,
	hop_length: int,
	sample_rate: int,
	detected_key: KeyEstimate | None,
) -> list[dict[str, object]]:
	if not bar_windows:
		return []
	raw_candidates = _raw_bar_phrase_candidates(
		segments,
		bar_windows=bar_windows,
		chroma=chroma,
		hop_length=hop_length,
		sample_rate=sample_rate,
		detected_key=detected_key,
	)
	return _decode_bar_candidate_sequence(raw_candidates, detected_key=detected_key)


def _lead_sheet_bar_candidates(
	segments: list[ChordSegment],
	*,
	bar_windows: list[tuple[float, float]],
	chroma: np.ndarray | None,
	hop_length: int,
	sample_rate: int,
	detected_key: KeyEstimate | None,
) -> list[dict[str, object]]:
	if not bar_windows:
		return []
	raw_candidates = [
		_bar_candidate(
			segments,
			start=start,
			end=end,
			chroma=chroma,
			hop_length=hop_length,
			sample_rate=sample_rate,
			detected_key=detected_key,
			weights={
				"overlap": LEADSHEET_OVERLAP_WEIGHT,
				"full": LEADSHEET_FULL_CHROMA_WEIGHT,
				"subwindow": LEADSHEET_SUBWINDOW_WEIGHT,
				"key": LEADSHEET_KEY_WEIGHT,
			},
		)
		for start, end in bar_windows[:BAR_CANDIDATE_COUNT]
	]
	decoded = _decode_bar_candidate_sequence(raw_candidates, detected_key=detected_key)
	return _infer_repeating_phrase_sequence(decoded, detected_key=detected_key)


def _raw_bar_phrase_candidates(
	segments: list[ChordSegment],
	*,
	bar_windows: list[tuple[float, float]],
	chroma: np.ndarray | None,
	hop_length: int,
	sample_rate: int,
	detected_key: KeyEstimate | None,
) -> list[dict[str, object]]:
	return [
		_bar_candidate(
			segments,
			start=start,
			end=end,
			chroma=chroma,
			hop_length=hop_length,
			sample_rate=sample_rate,
			detected_key=detected_key,
			weights={
				"overlap": MULTIRES_OVERLAP_WEIGHT,
				"full": MULTIRES_FULL_CHROMA_WEIGHT,
				"subwindow": MULTIRES_SUBWINDOW_WEIGHT,
				"key": MULTIRES_KEY_WEIGHT,
			},
		)
		for start, end in bar_windows[:BAR_CANDIDATE_COUNT]
	]


def _bar_candidate(
	segments: list[ChordSegment],
	*,
	start: float,
	end: float,
	chroma: np.ndarray | None,
	hop_length: int,
	sample_rate: int,
	detected_key: KeyEstimate | None,
	weights: dict[str, float],
) -> dict[str, object]:
	overlap_scores = _bar_overlap_scores(segments, start=start, end=end)
	chroma_scores = _bar_chroma_scores(
		chroma,
		start=start,
		end=end,
		hop_length=hop_length,
		sample_rate=sample_rate,
	)
	subwindow_scores = _bar_subwindow_scores(
		segments,
		chroma=chroma,
		start=start,
		end=end,
		hop_length=hop_length,
		sample_rate=sample_rate,
	)
	labels = set(overlap_scores) | set(chroma_scores) | set(subwindow_scores)
	key_scores = _bar_key_scores(labels, detected_key=detected_key)
	combined = {
		label: (
			(weights["overlap"] * overlap_scores.get(label, 0.0))
			+ (weights["full"] * chroma_scores.get(label, 0.0))
			+ (weights["subwindow"] * subwindow_scores.get(label, 0.0))
			+ (weights["key"] * key_scores.get(label, 0.0))
		)
		for label in labels
	}
	top = sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:BAR_TOP_CHORD_COUNT]
	return {
		"start": round(start, 3),
		"end": round(end, 3),
		"winner": top[0][0] if top else None,
		"top": [
			{
				"chord": chord,
				"score": round(score, 3),
				"overlap": round(overlap_scores.get(chord, 0.0), 3),
				"chroma": round(chroma_scores.get(chord, 0.0), 3),
				"subwindow": round(subwindow_scores.get(chord, 0.0), 3),
				"key": round(key_scores.get(chord, 0.0), 3),
			}
			for chord, score in top
		],
	}


def _bar_subwindow_scores(
	segments: list[ChordSegment],
	*,
	chroma: np.ndarray | None,
	start: float,
	end: float,
	hop_length: int,
	sample_rate: int,
) -> dict[str, float]:
	windows = _subdivision_windows(start, end, parts=2) + _subdivision_windows(start, end, parts=4)
	scores: dict[str, float] = {}
	for left, right in windows:
		if right <= left:
			continue
		chroma_scores = _bar_chroma_scores(
			chroma,
			start=left,
			end=right,
			hop_length=hop_length,
			sample_rate=sample_rate,
		)
		overlap_scores = _bar_overlap_scores(segments, start=left, end=right)
		labels = set(chroma_scores) | set(overlap_scores)
		for label in labels:
			score = (0.72 * chroma_scores.get(label, 0.0)) + (0.28 * overlap_scores.get(label, 0.0))
			scores[label] = max(scores.get(label, 0.0), score)
	return scores


def _subdivision_windows(start: float, end: float, *, parts: int) -> list[tuple[float, float]]:
	if parts <= 1 or end <= start:
		return []
	duration = end - start
	return [
		(start + (duration * index / parts), start + (duration * (index + 1) / parts))
		for index in range(parts)
	]


def _bar_key_scores(labels: set[str], *, detected_key: KeyEstimate | None) -> dict[str, float]:
	if detected_key is None:
		return {}
	diatonic = set(_diatonic_chord_labels_for_key(detected_key))
	return {label: 1.0 if label in diatonic else 0.0 for label in labels}


def _decode_bar_candidate_sequence(
	candidates: list[dict[str, object]],
	*,
	detected_key: KeyEstimate | None,
) -> list[dict[str, object]]:
	if len(candidates) <= 1:
		return candidates
	states = [_candidate_states(candidate) for candidate in candidates]
	if any(not state for state in states):
		return candidates

	dp: list[dict[str, tuple[float, str | None]]] = []
	first = {chord: (_sequence_local_score(chord, score, detected_key), None) for chord, score in states[0].items()}
	dp.append(first)
	for index in range(1, len(states)):
		row: dict[str, tuple[float, str | None]] = {}
		for chord, local_score in states[index].items():
			best: tuple[float, str | None] | None = None
			for prev_chord, (prev_score, _) in dp[index - 1].items():
				score = prev_score + _sequence_local_score(chord, local_score, detected_key)
				score += _sequence_transition_score(prev_chord, chord, detected_key)
				score += _sequence_repeat_pattern_bonus(chord, index, states)
				if best is None or score > best[0]:
					best = (score, prev_chord)
			if best is not None:
				row[chord] = best
		dp.append(row)
	if not dp[-1]:
		return candidates

	last = sorted(dp[-1].items(), key=lambda item: (-item[1][0], item[0]))[0][0]
	path = [last]
	for index in range(len(dp) - 1, 0, -1):
		_, prev = dp[index][path[-1]]
		if prev is None:
			break
		path.append(prev)
	path.reverse()
	if len(path) != len(candidates):
		return candidates

	return [_with_sequence_winner(candidate, winner) for candidate, winner in zip(candidates, path, strict=True)]


def _candidate_states(candidate: dict[str, object]) -> dict[str, float]:
	top = candidate.get("top")
	if not isinstance(top, list):
		return {}
	states: dict[str, float] = {}
	for item in top:
		if not isinstance(item, dict):
			continue
		chord = item.get("chord")
		score = item.get("score")
		if not isinstance(chord, str):
			continue
		try:
			states[chord] = float(score)
		except (TypeError, ValueError):
			continue
	return states


def _sequence_local_score(chord: str, score: float, detected_key: KeyEstimate | None) -> float:
	value = float(score)
	if detected_key is not None and _is_diatonic_chord(chord, detected_key):
		value += SEQUENCE_DIATONIC_BONUS
	return value


def _sequence_transition_score(prev_chord: str, chord: str, detected_key: KeyEstimate | None) -> float:
	if prev_chord == chord:
		return SEQUENCE_SELF_BONUS
	if detected_key is None:
		return -SEQUENCE_CHANGE_COST
	return (SEQUENCE_RELATIONSHIP_WEIGHT * _harmonic_relationship_strength(prev_chord, chord, detected_key)) - SEQUENCE_CHANGE_COST


def _sequence_repeat_pattern_bonus(chord: str, index: int, states: list[dict[str, float]]) -> float:
	score = 0.0
	if index >= 4 and chord in states[index - 4]:
		score += SEQUENCE_REPEAT_PATTERN_BONUS
	if index >= 2:
		prev_states = states[index - 2 : index]
		if all(chord in state and state[chord] >= 0.50 for state in prev_states):
			score -= SEQUENCE_OVERLONG_REPEAT_PENALTY
	return score


def _with_sequence_winner(candidate: dict[str, object], winner: str) -> dict[str, object]:
	if candidate.get("winner") == winner:
		return candidate
	updated = dict(candidate)
	updated["rawWinner"] = candidate.get("winner")
	updated["winner"] = winner
	updated["sequenceDecoded"] = True
	return updated


def _infer_repeating_phrase_sequence(
	candidates: list[dict[str, object]],
	*,
	detected_key: KeyEstimate | None,
) -> list[dict[str, object]]:
	if len(candidates) < PHRASE_PATTERN_MIN_LENGTH * 2:
		return candidates
	states = [_candidate_states(candidate) for candidate in candidates]
	if any(not state for state in states):
		return candidates
	pattern_start = _phrase_pattern_start_index(candidates)
	limit = min(len(states), pattern_start + PHRASE_PATTERN_ANALYSIS_BARS)
	best = _best_phrase_pattern(states[pattern_start:limit], detected_key=detected_key)
	if best is None:
		return candidates
	pattern, score, margin = best
	if score < PHRASE_PATTERN_MIN_SCORE or margin < PHRASE_PATTERN_MIN_MARGIN:
		return candidates

	refined: list[dict[str, object]] = []
	for index, candidate in enumerate(candidates):
		if index < pattern_start:
			refined.append(candidate)
			continue
		winner = pattern[(index - pattern_start) % len(pattern)]
		if winner not in states[index]:
			refined.append(candidate)
			continue
		refined.append(_with_pattern_winner(candidate, winner, pattern=pattern, score=score, margin=margin))
	return refined


def _phrase_pattern_start_index(candidates: list[dict[str, object]]) -> int:
	durations = [
		float(candidate["end"]) - float(candidate["start"])
		for candidate in candidates
		if isinstance(candidate.get("start"), int | float) and isinstance(candidate.get("end"), int | float)
	]
	if len(durations) < PHRASE_PATTERN_MIN_LENGTH:
		return 0
	typical = float(np.median(durations))
	first = durations[0]
	if typical <= 0.0 or first >= typical * PHRASE_PATTERN_PARTIAL_BAR_RATIO:
		return 0
	return 1


def _best_phrase_pattern(
	states: list[dict[str, float]],
	*,
	detected_key: KeyEstimate | None,
) -> tuple[list[str], float, float] | None:
	scored: list[tuple[float, list[str]]] = []
	for length in range(PHRASE_PATTERN_MIN_LENGTH, PHRASE_PATTERN_MAX_LENGTH + 1):
		if len(states) < length * 2:
			continue
		pattern = _best_pattern_for_length(states, length=length, detected_key=detected_key)
		if pattern is None:
			continue
		score = _phrase_pattern_score(pattern, states, detected_key=detected_key)
		if length == BAR_BEATS:
			score += PHRASE_PATTERN_LENGTH_FOUR_BONUS
		scored.append((score, pattern))
	if not scored:
		return None
	scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
	best_score, best_pattern = scored[0]
	second_score = scored[1][0] if len(scored) > 1 else 0.0
	return best_pattern, float(best_score), float(best_score - second_score)


def _best_pattern_for_length(
	states: list[dict[str, float]],
	*,
	length: int,
	detected_key: KeyEstimate | None,
) -> list[str] | None:
	slot_options = [_slot_pattern_options(states, slot=slot, length=length) for slot in range(length)]
	if any(not options for options in slot_options):
		return None
	patterns = [list(candidate) for candidate in product(*slot_options)]
	if not patterns:
		return None
	return sorted(
		patterns,
		key=lambda pattern: (-_phrase_pattern_score(pattern, states, detected_key=detected_key), pattern),
	)[0]


def _slot_pattern_options(states: list[dict[str, float]], *, slot: int, length: int) -> list[str]:
	aggregate: dict[str, float] = {}
	for index in range(slot, len(states), length):
		for chord, score in states[index].items():
			aggregate[chord] = aggregate.get(chord, 0.0) + score
	return [
		chord
		for chord, _ in sorted(aggregate.items(), key=lambda item: (-item[1], item[0]))[:PHRASE_PATTERN_SLOT_TOP_K]
	]


def _slot_pattern_score(
	chord: str,
	states: list[dict[str, float]],
	*,
	slot: int,
	length: int,
	detected_key: KeyEstimate | None,
) -> float:
	values = [states[index].get(chord, 0.0) for index in range(slot, len(states), length)]
	if not values:
		return 0.0
	score = PHRASE_PATTERN_EVIDENCE_WEIGHT * float(mean(values))
	if detected_key is not None and _is_diatonic_chord(chord, detected_key):
		score += PHRASE_PATTERN_KEY_WEIGHT
	return score


def _phrase_pattern_score(
	pattern: list[str],
	states: list[dict[str, float]],
	*,
	detected_key: KeyEstimate | None,
) -> float:
	evidence = [states[index].get(pattern[index % len(pattern)], 0.0) for index in range(len(states))]
	score = PHRASE_PATTERN_EVIDENCE_WEIGHT * float(mean(evidence))
	score += _phrase_first_slot_anchor_score(pattern, states)
	if detected_key is not None:
		diatonic_ratio = sum(1 for chord in pattern if _is_diatonic_chord(chord, detected_key)) / len(pattern)
		score += PHRASE_PATTERN_KEY_WEIGHT * diatonic_ratio
		score += _phrase_function_coverage_score(pattern, detected_key)
		relations = [
			_harmonic_relationship_strength(left, right, detected_key)
			for left, right in zip(pattern, [*pattern[1:], pattern[0]], strict=False)
			if left != right
		]
		if relations:
			score += PHRASE_PATTERN_RELATIONSHIP_WEIGHT * float(mean(relations))
	unique_ratio = len(set(pattern)) / len(pattern)
	score += PHRASE_PATTERN_DIVERSITY_BONUS * unique_ratio
	score -= PHRASE_PATTERN_STAGNATION_PENALTY * max(0, _longest_run_length(pattern) - 1)
	return float(score)


def _phrase_first_slot_anchor_score(pattern: list[str], states: list[dict[str, float]]) -> float:
	if not pattern or not states or not states[0]:
		return 0.0
	first_winner = sorted(states[0].items(), key=lambda item: (-item[1], item[0]))[0][0]
	if pattern[0] == first_winner:
		return PHRASE_PATTERN_FIRST_SLOT_ANCHOR_BONUS
	return -PHRASE_PATTERN_FIRST_SLOT_MISMATCH_PENALTY


def _phrase_function_coverage_score(pattern: list[str], detected_key: KeyEstimate) -> float:
	roles = {_phrase_function_role(chord, detected_key) for chord in pattern}
	score = 0.0
	if {"tonic", "predominant", "dominant"}.issubset(roles):
		score += PHRASE_PATTERN_FUNCTION_COVERAGE_BONUS
	if "dominant" not in roles and len(set(pattern)) >= 3:
		score -= PHRASE_PATTERN_MISSING_DOMINANT_PENALTY
	return score


def _phrase_function_role(chord: str, detected_key: KeyEstimate) -> str:
	if not _is_diatonic_chord(chord, detected_key):
		return "outside"
	root_pc = _chord_root_pc(chord)
	if root_pc is None:
		return "outside"
	interval = (root_pc - detected_key.tonic_pc) % 12
	if interval == 0:
		return "tonic"
	if detected_key.mode == "major":
		if interval in {2, 5}:
			return "predominant"
		if interval == 7:
			return "dominant"
		return "secondary"
	if interval in {3, 5}:
		return "predominant"
	if interval in {7, 10}:
		return "dominant"
	return "secondary"


def _with_pattern_winner(
	candidate: dict[str, object],
	winner: str,
	*,
	pattern: list[str],
	score: float,
	margin: float,
) -> dict[str, object]:
	updated = dict(candidate)
	if candidate.get("winner") != winner:
		updated["rawWinner"] = candidate.get("winner")
		updated["winner"] = winner
	updated["phrasePatternDecoded"] = True
	updated["phrasePattern"] = pattern
	updated["phrasePatternScore"] = round(score, 3)
	updated["phrasePatternMargin"] = round(margin, 3)
	return updated


def _bar_overlap_scores(segments: list[ChordSegment], *, start: float, end: float) -> dict[str, float]:
	duration = max(1e-9, end - start)
	scores: dict[str, float] = {}
	for segment in segments:
		if segment.chord == "N":
			continue
		overlap = max(0.0, min(end, segment.end) - max(start, segment.start))
		if overlap <= 0.0:
			continue
		weight = overlap / duration
		scores[segment.chord] = scores.get(segment.chord, 0.0) + weight * (0.5 + 0.5 * segment.confidence)
	return scores


def _bar_chroma_scores(
	chroma: np.ndarray | None,
	*,
	start: float,
	end: float,
	hop_length: int,
	sample_rate: int,
) -> dict[str, float]:
	vector = _window_chroma_vector(chroma, start=start, end=end, hop_length=hop_length, sample_rate=sample_rate)
	if vector is None:
		return {}
	templates = generate_chord_templates()
	return {
		label: max(0.0, _cosine_similarity(vector, template, min_norm=1e-7))
		for label, template in templates.items()
	}


def _window_chroma_vector(
	chroma: np.ndarray | None,
	*,
	start: float,
	end: float,
	hop_length: int,
	sample_rate: int,
) -> np.ndarray | None:
	if chroma is None or chroma.ndim != 2 or chroma.shape[0] != 12 or chroma.shape[1] <= 0:
		return None
	frame_seconds = frame_duration_seconds(hop_length=hop_length, sample_rate=sample_rate)
	if frame_seconds <= 0.0:
		return None
	start_frame = max(0, int(np.floor(start / frame_seconds)))
	end_frame = min(chroma.shape[1], int(np.ceil(end / frame_seconds)))
	if end_frame <= start_frame:
		return None
	vector = np.mean(chroma[:, start_frame:end_frame], axis=1)
	norm = float(np.linalg.norm(vector))
	if not np.isfinite(norm) or norm <= 1e-9:
		return None
	return vector / norm


def _beat_seconds(beat_timing: BeatTiming | None, *, hop_length: int, sample_rate: int) -> list[float]:
	return beat_times_seconds(beat_timing, hop_length=hop_length, sample_rate=sample_rate)


def _beat_interval_seconds(beat_timing: BeatTiming | None) -> float | None:
	if beat_timing is None or not beat_timing.is_reliable or beat_timing.tempo_bpm is None:
		return None
	tempo = float(beat_timing.tempo_bpm)
	if not np.isfinite(tempo) or tempo <= 0.0:
		return None
	return 60.0 / tempo


def _normalized_tempo(
	beat_timing: BeatTiming | None,
	*,
	raw_beat_interval: float | None,
) -> dict[str, float | str | None]:
	raw_tempo = float(beat_timing.tempo_bpm) if beat_timing is not None and beat_timing.tempo_bpm is not None else None
	if raw_tempo is None or raw_beat_interval is None:
		return {
			"rawTempoBpm": raw_tempo,
			"normalizedTempoBpm": raw_tempo,
			"beatIntervalSeconds": raw_beat_interval,
			"normalization": "none",
		}
	raw_bar_length = raw_beat_interval * BAR_BEATS
	if raw_tempo >= DOUBLE_TIME_TEMPO_BPM and raw_bar_length < MIN_NORMALIZED_BAR_SECONDS:
		return {
			"rawTempoBpm": raw_tempo,
			"normalizedTempoBpm": raw_tempo / 2.0,
			"beatIntervalSeconds": raw_beat_interval * 2.0,
			"normalization": "half-time",
		}
	if raw_tempo <= HALF_TIME_TEMPO_BPM and raw_bar_length > MAX_NORMALIZED_BAR_SECONDS:
		return {
			"rawTempoBpm": raw_tempo,
			"normalizedTempoBpm": raw_tempo * 2.0,
			"beatIntervalSeconds": raw_beat_interval / 2.0,
			"normalization": "double-time",
		}
	return {
		"rawTempoBpm": raw_tempo,
		"normalizedTempoBpm": raw_tempo,
		"beatIntervalSeconds": raw_beat_interval,
		"normalization": "none",
	}


def _downbeat_phase_candidates(
	segments: list[ChordSegment],
	*,
	beat_seconds: list[float],
	beat_interval: float | None,
	musical_start: float,
	duration_seconds: float,
	chroma: np.ndarray | None,
	hop_length: int,
	sample_rate: int,
	detected_key: KeyEstimate | None,
) -> list[dict[str, object]]:
	if beat_interval is None or not beat_seconds:
		return []
	first_beat = beat_seconds[0]
	bar_length = beat_interval * BAR_BEATS
	candidates: list[dict[str, object]] = []
	for phase in range(BAR_BEATS):
		candidate = _align_to_musical_start(first_beat + (phase * beat_interval), bar_length=bar_length, musical_start=musical_start)
		windows = _bar_windows(candidate, bar_length=bar_length, duration_seconds=duration_seconds)
		bar_candidates = _decode_bar_candidate_sequence(
			_raw_bar_phrase_candidates(
				segments,
				bar_windows=windows,
				chroma=chroma,
				hop_length=hop_length,
				sample_rate=sample_rate,
				detected_key=detected_key,
			),
			detected_key=detected_key,
		)
		bar_candidates = _infer_repeating_phrase_sequence(bar_candidates, detected_key=detected_key)
		winners = [str(item["winner"]) for item in bar_candidates[:BAR_CANDIDATE_COUNT] if item.get("winner")]
		score = _downbeat_score(segments, downbeat=candidate, bar_length=bar_length, musical_start=musical_start)
		score += _progression_grid_score(winners, detected_key=detected_key)
		candidates.append(
			{
				"phase": phase,
				"downbeat": candidate,
				"score": score,
				"winners": winners[:DIAGNOSTIC_WINDOW_COUNT],
			}
		)
	return sorted(candidates, key=lambda item: (-float(item["score"]), int(item["phase"])))[:PHASE_CANDIDATE_COUNT]


def _progression_grid_score(winners: list[str], *, detected_key: KeyEstimate | None) -> float:
	if not winners:
		return 0.0
	score = 0.0
	if detected_key is not None:
		score += sum(0.45 for chord in winners if _is_diatonic_chord(chord, detected_key))
		for left, right in zip(winners[:-1], winners[1:], strict=False):
			if left == right:
				score -= 0.18
			else:
				score += 0.34 * _harmonic_relationship_strength(left, right, detected_key)
	repeat_count = sum(1 for left, right in zip(winners[:-1], winners[1:], strict=False) if left == right)
	score -= 0.06 * repeat_count
	unique_count = len(set(winners))
	if unique_count >= 3:
		score += 0.35
	score += _early_phrase_diversity_score(winners)
	return float(score)


def _early_phrase_diversity_score(winners: list[str]) -> float:
	early = [winner for winner in winners[:4] if winner]
	if len(early) < 3:
		return 0.0
	unique_count = len(set(early))
	score = PHASE_EARLY_DIVERSITY_BONUS * max(0, unique_count - 2)
	longest_run = _longest_run_length(early)
	if longest_run >= 3:
		score -= PHASE_EARLY_STAGNATION_PENALTY * (longest_run - 2)
	return float(score)


def _longest_run_length(labels: list[str]) -> int:
	longest = 0
	current = 0
	previous: str | None = None
	for label in labels:
		if label == previous:
			current += 1
		else:
			current = 1
			previous = label
		longest = max(longest, current)
	return longest


def _align_to_musical_start(downbeat: float, *, bar_length: float, musical_start: float) -> float:
	if bar_length <= 0.0:
		return downbeat
	while downbeat + bar_length <= musical_start:
		downbeat += bar_length
	while downbeat > musical_start:
		downbeat -= bar_length
	return downbeat


def _downbeat_score(segments: list[ChordSegment], *, downbeat: float, bar_length: float, musical_start: float) -> float:
	score = 0.0
	if musical_start > 0.0:
		score += 1.5 * _grid_weight(musical_start, downbeat=downbeat, interval=bar_length)
	for segment in segments:
		if segment.chord == "N":
			continue
		score += _grid_weight(segment.start, downbeat=downbeat, interval=bar_length) * (0.4 + 0.6 * segment.confidence)
	return float(score)


def _bar_windows(downbeat: float | None, *, bar_length: float | None, duration_seconds: float) -> list[tuple[float, float]]:
	if downbeat is None or bar_length is None or bar_length <= 0.0 or duration_seconds <= 0.0:
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
	return [(start, end) for start, end in zip(edges[:-1], edges[1:], strict=False) if end - start > 0.25]


def _grid_alignment_errors(
	segments: list[ChordSegment],
	*,
	downbeat: float | None,
	bar_length: float | None,
) -> list[float]:
	if downbeat is None or bar_length is None or bar_length <= 0.0:
		return []
	errors: list[float] = []
	for segment in segments:
		if segment.chord == "N":
			continue
		error = _distance_to_grid(segment.start, downbeat=downbeat, interval=bar_length)
		if error <= GRID_ALIGNMENT_WINDOW_SECONDS:
			errors.append(error)
	return errors


def _grid_weight(value: float, *, downbeat: float, interval: float) -> float:
	distance = _distance_to_grid(value, downbeat=downbeat, interval=interval)
	return max(0.0, 1.0 - (distance / max(GRID_ALIGNMENT_WINDOW_SECONDS, 1e-6)))


def _distance_to_grid(value: float, *, downbeat: float, interval: float) -> float:
	if interval <= 0.0:
		return abs(value - downbeat)
	position = (value - downbeat) / interval
	nearest = downbeat + (round(position) * interval)
	return abs(value - nearest)
