"""Harmonic novelty and boundary selection helpers."""

from __future__ import annotations

from statistics import mean, median

import numpy as np

from chord_engine.analysis_config import *  # noqa: F403 - shared boundary tuning constants
from chord_engine.analysis_models import MusicalRegionObservation
from chord_engine.detector import KeyEstimate
from chord_engine.music_theory import (
    _chord_root_pc,
    _is_root_preserving_quality_switch,
    _parse_chord_quality,
)
from chord_engine.numeric import _cosine_similarity
from chord_engine.root_aware import (
    _estimate_region_root_aware_identity,
    _normalize_nonnegative,
)
from chord_engine.region_observation import _build_region_observation
from chord_engine.segment_utils import _consensus_chord_with_ratio

def _compute_harmonic_novelty(beat_profiles: list[np.ndarray]) -> list[float]:
	if len(beat_profiles) <= 1:
		return []
	values: list[float] = []
	for idx in range(len(beat_profiles) - 1):
		cross = 1.0 - _cosine_similarity(beat_profiles[idx], beat_profiles[idx + 1])
		values.append(float(np.clip(cross, 0.0, 1.0)))
	return values

def _select_harmonic_boundaries(
	novelty: list[float],
	beat_profiles: list[np.ndarray],
	beat_ranges: list[tuple[int, int]],
	beat_local_chords: list[str],
	*,
	hop_length: int,
	sample_rate: int,
) -> tuple[list[int], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
	if len(novelty) == 0:
		return [], [], [], {
			"localNoveltyCandidateCount": 0,
			"contextualBoundaryCandidateCount": 0,
			"multiResolutionAcceptedBoundaryCount": 0,
			"rejectedLocalOnlyBoundaryCount": 0,
			"shortMediumAgreementRate": 0.0,
			"meanShortContextDistance": 0.0,
			"meanMediumContextDistance": 0.0,
			"acceptedBoundaryExamples": [],
			"rejectedLocalOnlyExamples": [],
		}

	arr = np.asarray(novelty, dtype=np.float32)
	med = float(np.median(arr))
	mad = float(np.median(np.abs(arr - med)))
	threshold = float(max(NOVELTY_MIN_PEAK, med + NOVELTY_MAD_SCALE * mad))

	selected: list[int] = []
	rejected: list[dict[str, object]] = []
	candidates: list[dict[str, object]] = []
	accepted_examples: list[dict[str, object]] = []
	rejected_local_only_examples: list[dict[str, object]] = []
	short_distances: list[float] = []
	medium_distances: list[float] = []
	agreement_flags: list[bool] = []
	contextual_candidate_count = 0
	rejected_local_only_count = 0
	for idx, value in enumerate(novelty):
		left = novelty[idx - 1] if idx - 1 >= 0 else -1.0
		right = novelty[idx + 1] if idx + 1 < len(novelty) else -1.0
		is_local_peak = value >= left and value >= right

		prominence = float(value - max(left, right, 0.0))
		forward_persistence = 0.0
		if idx + 2 < len(beat_profiles):
			forward_persistence = float(np.clip(1.0 - _cosine_similarity(beat_profiles[idx], beat_profiles[idx + 2]), 0.0, 1.0))
		backward_persistence = 0.0
		if idx - 1 >= 0:
			backward_persistence = float(np.clip(1.0 - _cosine_similarity(beat_profiles[idx - 1], beat_profiles[idx + 1]), 0.0, 1.0))
		persistence = max(forward_persistence, backward_persistence)

		timestamp = float(beat_ranges[idx][1] * hop_length / sample_rate)
		left_chord = beat_local_chords[idx]
		right_chord = beat_local_chords[idx + 1]
		chord_change_support = left_chord != right_chord
		near_peak_rapid_change = bool(
			chord_change_support
			and value >= (threshold * 0.72)
			and value >= (max(left, right, 0.0) * 0.64)
			and persistence >= 0.20
		)
		if not (is_local_peak or near_peak_rapid_change):
			continue
		evidence = _multi_resolution_boundary_evidence(
			beat_profiles,
			beat_ranges,
			idx,
			hop_length=hop_length,
			sample_rate=sample_rate,
		)
		short_distance = float(evidence["shortDistance"])
		medium_distance = float(evidence["mediumDistance"])
		context_agreement = float(evidence["contextAgreement"])
		persistence_support = float(evidence["persistenceSupport"])
		short_medium_agree = bool(evidence["shortMediumAgree"])
		short_window_beats = int(evidence["shortWindowBeats"])
		medium_window_beats = int(evidence["mediumWindowBeats"])
		left_consensus, left_consensus_ratio = _consensus_chord_with_ratio(
			beat_local_chords,
			start=max(0, idx - short_window_beats + 1),
			end=idx + 1,
		)
		right_consensus, right_consensus_ratio = _consensus_chord_with_ratio(
			beat_local_chords,
			start=idx + 1,
			end=min(len(beat_local_chords), idx + 1 + short_window_beats),
		)
		left_short_profile = _robust_context_profile(
			beat_profiles,
			start=max(0, idx - short_window_beats + 1),
			end=idx + 1,
		)
		right_short_profile = _robust_context_profile(
			beat_profiles,
			start=idx + 1,
			end=min(len(beat_profiles), idx + 1 + short_window_beats),
		)
		short_distances.append(short_distance)
		medium_distances.append(medium_distance)
		agreement_flags.append(short_medium_agree)

		candidates.append(
			{
				"index": idx,
				"timestamp": timestamp,
				"novelty": float(value),
				"prominence": float(prominence),
				"persistence": float(persistence),
				"shortContextDistance": short_distance,
				"mediumContextDistance": medium_distance,
				"contextAgreementScore": context_agreement,
				"persistenceSupport": persistence_support,
				"shortMediumAgree": short_medium_agree,
				"contextShortWindowBeats": short_window_beats,
				"contextMediumWindowBeats": medium_window_beats,
				"leftConsensusChord": left_consensus,
				"leftConsensusRatio": left_consensus_ratio,
				"rightConsensusChord": right_consensus,
				"rightConsensusRatio": right_consensus_ratio,
				"leftLocalChord": left_chord,
				"rightLocalChord": right_chord,
			}
		)

		left_root, _ = _parse_chord_quality(left_chord)
		right_root, _ = _parse_chord_quality(right_chord)
		same_root_candidate = bool(
			left_root is not None
			and right_root is not None
			and left_root == right_root
		)
		same_root_quality_shift = False
		if same_root_candidate:
			root_pc = _chord_root_pc(left_root or "")
			if root_pc is not None:
				left_quality_balance = _quality_balance_for_root(left_short_profile, root_pc)
				right_quality_balance = _quality_balance_for_root(right_short_profile, root_pc)
				same_root_quality_shift = bool(
					left_quality_balance * right_quality_balance < 0.0
					and max(abs(left_quality_balance), abs(right_quality_balance)) >= MULTIRES_SAME_ROOT_QUALITY_SHIFT_ABS_MIN
					and abs(left_quality_balance - right_quality_balance) >= MULTIRES_SAME_ROOT_QUALITY_SHIFT_DELTA_MIN
				)
		quality_change_support = _is_root_preserving_quality_switch(left_chord, right_chord) or same_root_quality_shift
		persistence_floor = NOVELTY_PERSISTENCE_DISTANCE * (0.5 if quality_change_support else 1.0)
		threshold_floor = max(NOVELTY_MIN_PEAK * (0.9 if quality_change_support else 1.0), threshold * 0.82)
		passes_local_gate = (
			value >= threshold
			and prominence >= NOVELTY_MIN_PROMINENCE
			and persistence >= persistence_floor
		) or (
			chord_change_support
			and value >= threshold_floor
			and prominence >= (NOVELTY_MIN_PROMINENCE * 0.75)
			and persistence >= (persistence_floor * 0.7)
		) or (
			chord_change_support
			and value >= threshold
			and persistence >= max(0.25, persistence_floor)
			and value >= 0.22
		)

		is_contextual_candidate = bool(
			short_distance >= (MULTIRES_SHORT_DISTANCE_MIN * 0.78)
			and medium_distance >= (MULTIRES_MEDIUM_DISTANCE_FLOOR * 0.90)
		)
		if is_contextual_candidate:
			contextual_candidate_count += 1

		strong_rapid_context = bool(
			short_distance >= (MULTIRES_SHORT_DISTANCE_MIN * 1.25)
			and medium_distance >= MULTIRES_MEDIUM_DISTANCE_FLOOR
			and context_agreement >= (MULTIRES_CONTEXT_AGREEMENT_MIN * 0.92)
			and persistence_support >= 0.42
		)
		progressive_context_change = bool(
			chord_change_support
			and short_distance >= MULTIRES_PROGRESSIVE_SHORT_MIN
			and medium_distance >= MULTIRES_PROGRESSIVE_MEDIUM_MIN
			and persistence_support >= MULTIRES_PROGRESSIVE_PERSISTENCE_MIN
		)
		required_medium_distance = MULTIRES_MEDIUM_DISTANCE_MIN
		required_persistence_support = MULTIRES_PERSISTENCE_MIN
		if left_consensus == right_consensus and left_consensus is not None and left_consensus != "N":
			required_medium_distance = max(required_medium_distance, MULTIRES_SAME_CONTEXT_MEDIUM_MIN)
		if left_chord == right_chord and left_chord != "N":
			required_medium_distance = max(required_medium_distance, MULTIRES_SAME_LOCAL_MEDIUM_MIN)
		if not short_medium_agree:
			required_medium_distance = max(required_medium_distance, MULTIRES_DISAGREE_MEDIUM_MIN)
			required_persistence_support = max(required_persistence_support, MULTIRES_DISAGREE_PERSISTENCE_MIN)
		medium_dominant_true_change = bool(
			chord_change_support
			and medium_distance >= MULTIRES_MEDIUM_DOMINANT_DISTANCE_MIN
			and short_distance >= MULTIRES_MEDIUM_DOMINANT_SHORT_MIN
			and persistence_support >= MULTIRES_DISAGREE_PERSISTENCE_MIN
			and value >= (threshold_floor * MULTIRES_MEDIUM_DOMINANT_NOVELTY_SCALE)
			and left_consensus is not None
			and right_consensus is not None
			and left_consensus != right_consensus
		)
		consensus_shift_true_change = bool(
			chord_change_support
			and left_consensus is not None
			and right_consensus is not None
			and left_consensus != right_consensus
			and left_consensus_ratio >= MULTIRES_CONSENSUS_SHIFT_RATIO_MIN
			and right_consensus_ratio >= MULTIRES_CONSENSUS_SHIFT_RATIO_MIN
			and medium_distance >= MULTIRES_CONSENSUS_SHIFT_MEDIUM_MIN
			and value >= (threshold_floor * MULTIRES_CONSENSUS_SHIFT_NOVELTY_SCALE)
		)
		same_root_quality_true_change = bool(
			same_root_quality_shift
			and value >= (threshold_floor * MULTIRES_SAME_ROOT_QUALITY_SHIFT_NOVELTY_SCALE)
			and medium_distance >= MULTIRES_SAME_ROOT_QUALITY_SHIFT_MEDIUM_MIN
			and persistence_support >= MULTIRES_SAME_ROOT_QUALITY_SHIFT_PERSISTENCE_MIN
			and context_agreement >= MULTIRES_SAME_ROOT_QUALITY_SHIFT_AGREEMENT_MIN
		)
		salient_novelty_true_change = bool(
			chord_change_support
			and value >= (threshold_floor * MULTIRES_SALIENT_NOVELTY_SCALE)
			and prominence >= MULTIRES_SALIENT_PROMINENCE_MIN
			and short_distance >= MULTIRES_SALIENT_SHORT_DISTANCE_MIN
			and medium_distance >= MULTIRES_SALIENT_MEDIUM_DISTANCE_MIN
			and persistence_support >= MULTIRES_SALIENT_PERSISTENCE_SUPPORT_MIN
			and context_agreement >= MULTIRES_SALIENT_CONTEXT_AGREEMENT_MIN
			and left_consensus is not None
			and right_consensus is not None
			and left_consensus != right_consensus
			and (left_consensus_ratio + right_consensus_ratio) >= MULTIRES_SALIENT_CONSENSUS_STRENGTH_MIN
		)
		passes_context_gate = bool(
			(
				short_distance >= MULTIRES_SHORT_DISTANCE_MIN
				and medium_distance >= required_medium_distance
				and context_agreement >= MULTIRES_CONTEXT_AGREEMENT_MIN
				and persistence_support >= required_persistence_support
			)
			or (chord_change_support and strong_rapid_context)
			or progressive_context_change
			or medium_dominant_true_change
			or consensus_shift_true_change
			or same_root_quality_true_change
			or salient_novelty_true_change
		)

		local_strength = float(np.clip(value / max(threshold * 1.35, 1e-6), 0.0, 1.0))
		short_support = float(np.clip(short_distance / max(MULTIRES_SHORT_DISTANCE_MIN * 2.20, 1e-6), 0.0, 1.0))
		medium_support = float(np.clip(medium_distance / max(MULTIRES_MEDIUM_DISTANCE_MIN * 2.60, 1e-6), 0.0, 1.0))
		final_confidence = float(np.clip((0.40 * local_strength) + (0.30 * short_support) + (0.30 * medium_support), 0.0, 1.0))
		same_context_consensus = bool(
			left_consensus == right_consensus
			and left_consensus is not None
			and left_consensus != "N"
		)
		same_context_requires_stronger_support = bool(
			same_context_consensus
			and not _is_root_preserving_quality_switch(left_chord, right_chord)
		)
		same_context_override_ok = bool(
			medium_distance >= MULTIRES_SAME_CONTEXT_MEDIUM_MIN
			and final_confidence >= MULTIRES_SAME_CONTEXT_CONFIDENCE_MIN
		)
		same_local_chord = bool(left_chord == right_chord and left_chord != "N")
		same_local_override_ok = bool(
			short_distance >= MULTIRES_SAME_LOCAL_SHORT_MIN
			and medium_distance >= MULTIRES_SAME_LOCAL_MEDIUM_MIN
			and final_confidence >= MULTIRES_SAME_LOCAL_CONFIDENCE_MIN
		) or same_root_quality_true_change
		required_final_confidence = MULTIRES_FINAL_CONFIDENCE_MIN
		if medium_distance < MULTIRES_DISAGREE_MEDIUM_MIN:
			required_final_confidence = max(required_final_confidence, MULTIRES_WEAK_MEDIUM_CONFIDENCE_MIN)
		if persistence_support < MULTIRES_DISAGREE_PERSISTENCE_MIN:
			required_final_confidence = max(required_final_confidence, MULTIRES_LOW_PERSISTENCE_CONFIDENCE_MIN)
		if not short_medium_agree:
			required_final_confidence = max(required_final_confidence, MULTIRES_WEAK_MEDIUM_CONFIDENCE_MIN)

		rapid_local_peak = bool(
			near_peak_rapid_change
			or (
				value >= threshold_floor
				and prominence >= (NOVELTY_MIN_PROMINENCE * 0.70)
				and persistence >= (persistence_floor * 0.65)
			)
		)
		salient_acceptance = bool(
			salient_novelty_true_change
			and final_confidence >= max(required_final_confidence, MULTIRES_SALIENT_CONFIDENCE_MIN)
		)
		if ((passes_local_gate and passes_context_gate and final_confidence >= required_final_confidence) or (
			rapid_local_peak and chord_change_support and strong_rapid_context and final_confidence >= max(required_final_confidence * 0.92, MULTIRES_FINAL_CONFIDENCE_MIN * 0.92)
		) or salient_acceptance) and (not same_context_requires_stronger_support or same_context_override_ok) and (not same_local_chord or same_local_override_ok):
			selected.append(idx)
			accepted_examples.append(
				{
					"timestamp": timestamp,
					"novelty": float(value),
					"shortContextDistance": short_distance,
					"mediumContextDistance": medium_distance,
					"contextAgreementScore": context_agreement,
					"persistenceSupport": persistence_support,
					"shortMediumAgree": short_medium_agree,
					"finalBoundaryConfidence": final_confidence,
					"leftConsensusChord": left_consensus,
					"leftConsensusRatio": left_consensus_ratio,
					"rightConsensusChord": right_consensus,
					"rightConsensusRatio": right_consensus_ratio,
					"leftLocalChord": left_chord,
					"rightLocalChord": right_chord,
				}
			)
		else:
			reasons: list[str] = []
			local_only_event = bool(value >= (threshold * 0.88) and not passes_context_gate)
			if value < threshold:
				reasons.append("below-threshold")
			if prominence < NOVELTY_MIN_PROMINENCE:
				reasons.append("low-prominence")
			if persistence < NOVELTY_PERSISTENCE_DISTANCE:
				reasons.append("no-persistence")
			if not passes_context_gate:
				reasons.append("context-insufficient")
			if same_context_requires_stronger_support and not same_context_override_ok:
				reasons.append("same-context-consensus")
			if same_local_chord and not same_local_override_ok:
				reasons.append("same-local-chord")
			if local_only_event:
				reasons.append("local-only-medium-insufficient")
			if final_confidence < MULTIRES_FINAL_CONFIDENCE_MIN:
				reasons.append("low-boundary-confidence")
			if local_only_event:
				rejected_local_only_count += 1
				rejected_local_only_examples.append(
					{
						"timestamp": timestamp,
						"novelty": float(value),
						"shortContextDistance": short_distance,
						"mediumContextDistance": medium_distance,
						"contextAgreementScore": context_agreement,
						"persistenceSupport": persistence_support,
						"shortMediumAgree": short_medium_agree,
						"finalBoundaryConfidence": final_confidence,
						"leftConsensusChord": left_consensus,
						"leftConsensusRatio": left_consensus_ratio,
						"rightConsensusChord": right_consensus,
						"rightConsensusRatio": right_consensus_ratio,
						"leftLocalChord": left_chord,
						"rightLocalChord": right_chord,
						"reason": "local-only-medium-insufficient",
					}
				)
			rejected.append(
				{
					"timestamp": timestamp,
					"novelty": float(value),
					"prominence": float(prominence),
					"persistence": float(persistence),
					"shortContextDistance": short_distance,
					"mediumContextDistance": medium_distance,
					"contextAgreementScore": context_agreement,
					"persistenceSupport": persistence_support,
					"shortMediumAgree": short_medium_agree,
					"finalBoundaryConfidence": final_confidence,
					"leftConsensusChord": left_consensus,
					"leftConsensusRatio": left_consensus_ratio,
					"rightConsensusChord": right_consensus,
					"rightConsensusRatio": right_consensus_ratio,
					"leftLocalChord": left_chord,
					"rightLocalChord": right_chord,
					"reason": "+".join(reasons) if reasons else "rejected",
				}
			)

	agreement_rate = float(sum(1 for flag in agreement_flags if flag) / len(agreement_flags)) if agreement_flags else 0.0
	diag = {
		"localNoveltyCandidateCount": len(candidates),
		"contextualBoundaryCandidateCount": contextual_candidate_count,
		"multiResolutionAcceptedBoundaryCount": len(selected),
		"rejectedLocalOnlyBoundaryCount": rejected_local_only_count,
		"shortMediumAgreementRate": agreement_rate,
		"meanShortContextDistance": float(mean(short_distances)) if short_distances else 0.0,
		"meanMediumContextDistance": float(mean(medium_distances)) if medium_distances else 0.0,
		"acceptedBoundaryExamples": accepted_examples[:40],
		"rejectedLocalOnlyExamples": rejected_local_only_examples[:40],
	}

	return selected, rejected, candidates, diag

def _multi_resolution_boundary_evidence(
	beat_profiles: list[np.ndarray],
	beat_ranges: list[tuple[int, int]],
	idx: int,
	*,
	hop_length: int,
	sample_rate: int,
) -> dict[str, object]:
	if len(beat_profiles) == 0:
		return {
			"shortDistance": 0.0,
			"mediumDistance": 0.0,
			"contextAgreement": 0.0,
			"persistenceSupport": 0.0,
			"shortMediumAgree": False,
			"shortWindowBeats": MULTIRES_SHORT_CONTEXT_MIN_BEATS,
			"mediumWindowBeats": MULTIRES_MEDIUM_CONTEXT_MIN_BEATS,
		}

	beat_seconds = _local_beat_seconds(beat_ranges, idx, hop_length=hop_length, sample_rate=sample_rate)
	short_window = _adaptive_context_window_beats(
		target_seconds=MULTIRES_SHORT_CONTEXT_TARGET_SECONDS,
		min_beats=MULTIRES_SHORT_CONTEXT_MIN_BEATS,
		max_beats=MULTIRES_SHORT_CONTEXT_MAX_BEATS,
		beat_seconds=beat_seconds,
	)
	medium_window = _adaptive_context_window_beats(
		target_seconds=MULTIRES_MEDIUM_CONTEXT_TARGET_SECONDS,
		min_beats=MULTIRES_MEDIUM_CONTEXT_MIN_BEATS,
		max_beats=MULTIRES_MEDIUM_CONTEXT_MAX_BEATS,
		beat_seconds=beat_seconds,
	)

	short_before = _robust_context_profile(beat_profiles, start=max(0, idx - short_window + 1), end=idx + 1)
	short_after = _robust_context_profile(beat_profiles, start=idx + 1, end=min(len(beat_profiles), idx + 1 + short_window))
	medium_before = _robust_context_profile(beat_profiles, start=max(0, idx - medium_window + 1), end=idx + 1)
	medium_after = _robust_context_profile(beat_profiles, start=idx + 1, end=min(len(beat_profiles), idx + 1 + medium_window))

	short_distance = float(np.clip(1.0 - _cosine_similarity(short_before, short_after), 0.0, 1.0))
	medium_distance = float(np.clip(1.0 - _cosine_similarity(medium_before, medium_after), 0.0, 1.0))
	context_agreement = float(np.clip(1.0 - abs(short_distance - medium_distance), 0.0, 1.0))
	persistence_support = float(np.clip(medium_distance / max(short_distance, 1e-6), 0.0, 1.0))
	short_medium_agree = bool(
		short_distance >= (MULTIRES_SHORT_DISTANCE_MIN * 0.85)
		and medium_distance >= (MULTIRES_MEDIUM_DISTANCE_FLOOR * 0.90)
		and context_agreement >= (MULTIRES_CONTEXT_AGREEMENT_MIN * 0.92)
	)

	return {
		"shortDistance": short_distance,
		"mediumDistance": medium_distance,
		"contextAgreement": context_agreement,
		"persistenceSupport": persistence_support,
		"shortMediumAgree": short_medium_agree,
		"shortWindowBeats": short_window,
		"mediumWindowBeats": medium_window,
	}

def _local_beat_seconds(
	beat_ranges: list[tuple[int, int]],
	idx: int,
	*,
	hop_length: int,
	sample_rate: int,
) -> float:
	if len(beat_ranges) == 0:
		return 0.5

	left = max(0, idx - 2)
	right = min(len(beat_ranges), idx + 3)
	window = beat_ranges[left:right]
	if not window:
		window = beat_ranges

	durations = [float(max(1, end - start) * hop_length / sample_rate) for start, end in window]
	if not durations:
		return 0.5
	return float(max(0.12, median(durations)))

def _adaptive_context_window_beats(*, target_seconds: float, min_beats: int, max_beats: int, beat_seconds: float) -> int:
	if beat_seconds <= 1e-6:
		return int(min_beats)
	estimated = int(round(target_seconds / beat_seconds))
	return int(np.clip(estimated, min_beats, max_beats))

def _quality_balance_for_root(profile: np.ndarray, root_pc: int) -> float:
	arr = np.asarray(profile, dtype=np.float32)
	if arr.shape != (12,):
		return 0.0
	major_evidence = float(
		ROOT_AWARE_QUALITY_THIRD_WEIGHT * arr[(root_pc + 4) % 12]
		+ ROOT_AWARE_QUALITY_SUPPORT_WEIGHT * ((arr[root_pc] + arr[(root_pc + 7) % 12]) / 2.0)
	)
	minor_evidence = float(
		ROOT_AWARE_QUALITY_THIRD_WEIGHT * arr[(root_pc + 3) % 12]
		+ ROOT_AWARE_QUALITY_SUPPORT_WEIGHT * ((arr[root_pc] + arr[(root_pc + 7) % 12]) / 2.0)
	)
	return float(major_evidence - minor_evidence)

def _robust_context_profile(beat_profiles: list[np.ndarray], *, start: int, end: int) -> np.ndarray:
	left = max(0, int(start))
	right = min(len(beat_profiles), int(end))
	if right <= left:
		return np.zeros((12,), dtype=np.float32)

	stack = np.asarray(beat_profiles[left:right], dtype=np.float32)
	if stack.ndim != 2 or stack.shape[0] == 0:
		return np.zeros((12,), dtype=np.float32)

	median_profile = np.median(stack, axis=0)
	mean_profile = np.mean(stack, axis=0)
	profile = _normalize_nonnegative((0.68 * median_profile) + (0.32 * mean_profile))
	return np.asarray(profile, dtype=np.float32)

def _consolidate_boundary_clusters(
	selected_idx: list[int],
	*,
	novelty: list[float],
	candidate_peaks: list[dict[str, object]],
	beat_profiles: list[np.ndarray],
	beat_ranges: list[tuple[int, int]],
	beat_local_chords: list[str],
	hop_length: int,
	sample_rate: int,
	chroma: np.ndarray,
	low_chroma: np.ndarray,
	key_estimate: KeyEstimate | None,
	beat_reliable: bool,
) -> tuple[list[int], dict[str, object]]:
	if len(selected_idx) <= 1:
		return selected_idx, {"boundaryClusterCount": 0, "consolidatedClusterExamples": []}

	peak_by_idx = {int(item.get("index", -1)): item for item in candidate_peaks}
	merged = sorted(selected_idx)
	cluster_count = 0
	examples: list[dict[str, object]] = []
	changed = True
	while changed and len(merged) > 1:
		changed = False
		for pos in range(len(merged) - 1):
			left_idx = merged[pos]
			right_idx = merged[pos + 1]
			left_frame = beat_ranges[left_idx][1]
			right_frame = beat_ranges[right_idx][1]
			seconds_between = float(max(0.0, (right_frame - left_frame) * hop_length / sample_rate))
			beats_between = float(max(0.0, right_idx - left_idx))
			cluster_close = (
				beats_between <= BOUNDARY_CLUSTER_MAX_BEAT_DISTANCE
				if beat_reliable
				else seconds_between <= BOUNDARY_CLUSTER_MAX_SECONDS_DISTANCE
			)
			if not cluster_close:
				continue

			cluster_count += 1
			mid_start = left_frame
			mid_end = right_frame
			if mid_end <= mid_start:
				continue

			left_ctx_start = beat_ranges[max(0, left_idx - 1)][0]
			left_ctx_end = left_frame
			right_ctx_start = right_frame
			right_ctx_end = beat_ranges[min(len(beat_ranges) - 1, right_idx + 1)][1]

			mid_obs = _build_region_observation(
				start=mid_start,
				end=mid_end,
				sample_rate=sample_rate,
				hop_length=hop_length,
				region_chroma=np.asarray(chroma[:, mid_start:mid_end], dtype=np.float32),
				region_low_chroma=np.asarray(low_chroma[:, mid_start:mid_end], dtype=np.float32),
				key_estimate=key_estimate,
			)

			left_profile = _profile_from_frame_range(chroma, low_chroma, left_ctx_start, left_ctx_end)
			mid_profile = _profile_from_frame_range(chroma, low_chroma, mid_start, mid_end)
			right_profile = _profile_from_frame_range(chroma, low_chroma, right_ctx_start, right_ctx_end)
			sim_left_right = _cosine_similarity(left_profile, right_profile)
			sim_left_mid = _cosine_similarity(left_profile, mid_profile)
			sim_mid_right = _cosine_similarity(mid_profile, right_profile)

			mid_beats = beats_between
			left_outer_chord = beat_local_chords[left_idx]
			right_outer_chord = beat_local_chords[min(len(beat_local_chords) - 1, right_idx + 1)]
			mid_duration_short = bool(
				mid_obs.duration_seconds <= BOUNDARY_CONSOLIDATION_SHORT_REGION_SECONDS
				or (beat_reliable and mid_beats <= BOUNDARY_CONSOLIDATION_SHORT_REGION_BEATS)
			)
			mid_weak = bool(
				mid_obs.local_best_score <= BOUNDARY_CONSOLIDATION_WEAK_SCORE_MAX
				and mid_obs.advantage_over_runner_up <= BOUNDARY_CONSOLIDATION_WEAK_MARGIN_MAX
			)
			sides_similar = bool(
				sim_left_right >= BOUNDARY_CONSOLIDATION_SIDE_SIMILARITY_MIN
				and sim_left_mid <= BOUNDARY_CONSOLIDATION_MID_SIMILARITY_MAX
				and sim_mid_right <= BOUNDARY_CONSOLIDATION_MID_SIMILARITY_MAX
			)
			contrast = float(sim_left_right - ((sim_left_mid + sim_mid_right) * 0.5))
			bridge_same_harmony = bool(left_outer_chord == right_outer_chord and left_outer_chord != mid_obs.local_best_chord)

			mid_has_strong_independent_evidence = bool(
				not mid_duration_short
				or mid_obs.local_best_score > 0.20
				or mid_obs.advantage_over_runner_up > 0.055
			)

			if mid_has_strong_independent_evidence and not bridge_same_harmony:
				examples.append(
					{
						"timestamps": [
							float(left_frame * hop_length / sample_rate),
							float(right_frame * hop_length / sample_rate),
						],
						"beatDistance": beats_between,
						"noveltyStrengths": [float(novelty[left_idx]), float(novelty[right_idx])],
						"prominences": [
							float(peak_by_idx.get(left_idx, {}).get("prominence", 0.0)),
							float(peak_by_idx.get(right_idx, {}).get("prominence", 0.0)),
						],
						"harmonicSimilarities": {
							"leftMid": float(sim_left_mid),
							"midRight": float(sim_mid_right),
							"leftRight": float(sim_left_right),
						},
						"selectedSurvivingBoundary": None,
						"rejectionReason": "preserve-strong-intermediate-region",
					}
				)
				continue

			if not (
				mid_duration_short
				and (mid_weak or bridge_same_harmony)
				and sides_similar
				and contrast >= BOUNDARY_CONSOLIDATION_CONTRAST_MIN
				and mid_beats <= BOUNDARY_CONSOLIDATION_MAX_INTERMEDIATE_BEATS
			):
				examples.append(
					{
						"timestamps": [
							float(left_frame * hop_length / sample_rate),
							float(right_frame * hop_length / sample_rate),
						],
						"beatDistance": beats_between,
						"noveltyStrengths": [float(novelty[left_idx]), float(novelty[right_idx])],
						"prominences": [
							float(peak_by_idx.get(left_idx, {}).get("prominence", 0.0)),
							float(peak_by_idx.get(right_idx, {}).get("prominence", 0.0)),
						],
						"harmonicSimilarities": {
							"leftMid": float(sim_left_mid),
							"midRight": float(sim_mid_right),
							"leftRight": float(sim_left_right),
						},
						"selectedSurvivingBoundary": None,
						"rejectionReason": "preserve-cluster-insufficient-collapse-evidence",
					}
				)
				continue

			left_strength = float(novelty[left_idx] + 0.70 * float(peak_by_idx.get(left_idx, {}).get("prominence", 0.0)))
			right_strength = float(novelty[right_idx] + 0.70 * float(peak_by_idx.get(right_idx, {}).get("prominence", 0.0)))
			survivor = left_idx if left_strength >= right_strength else right_idx
			dropped = right_idx if survivor == left_idx else left_idx
			merged = [idx for idx in merged if idx != dropped]
			examples.append(
				{
					"timestamps": [
						float(left_frame * hop_length / sample_rate),
						float(right_frame * hop_length / sample_rate),
					],
					"beatDistance": beats_between,
					"noveltyStrengths": [float(novelty[left_idx]), float(novelty[right_idx])],
					"prominences": [
						float(peak_by_idx.get(left_idx, {}).get("prominence", 0.0)),
						float(peak_by_idx.get(right_idx, {}).get("prominence", 0.0)),
					],
					"harmonicSimilarities": {
						"leftMid": float(sim_left_mid),
						"midRight": float(sim_mid_right),
						"leftRight": float(sim_left_right),
					},
					"selectedSurvivingBoundary": float(beat_ranges[survivor][1] * hop_length / sample_rate),
					"rejectionReason": "collapsed-weak-intermediate-region",
				}
			)
			changed = True
			break

	return sorted(merged), {
		"boundaryClusterCount": cluster_count,
		"consolidatedClusterExamples": examples,
	}

def _profile_from_frame_range(chroma: np.ndarray, low_chroma: np.ndarray, start: int, end: int) -> np.ndarray:
	left = max(0, int(start))
	right = max(left + 1, int(end))
	right = min(chroma.shape[1], right)
	if right <= left:
		return np.zeros((12,), dtype=np.float32)
	high = np.mean(np.clip(np.asarray(chroma[:, left:right], dtype=np.float32), 0.0, None), axis=1)
	low = np.mean(np.clip(np.asarray(low_chroma[:, left:right], dtype=np.float32), 0.0, None), axis=1)
	return _normalize_nonnegative(
		BEAT_PROFILE_HARMONIC_WEIGHT * high + BEAT_PROFILE_LOW_WEIGHT * low
	)
