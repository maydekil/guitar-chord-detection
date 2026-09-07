"""Beat-aware harmonic region construction and boundary selection."""

from __future__ import annotations

from statistics import mean, median

import numpy as np

from chord_engine.analysis_config import *  # noqa: F403 - shared harmonic-region tuning constants
from chord_engine.analysis_models import MusicalRegionObservation
from chord_engine.detector import FrameChordPrediction, KeyEstimate, predict_frame_chord
from chord_engine.harmonic_boundaries import (
    _compute_harmonic_novelty,
    _consolidate_boundary_clusters,
    _profile_from_frame_range,
    _select_harmonic_boundaries,
)
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

def _build_harmonic_regions_from_change_points(
	*,
	boundaries: np.ndarray,
	sample_rate: int,
	hop_length: int,
	chroma: np.ndarray,
	low_chroma: np.ndarray,
	key_estimate: KeyEstimate | None,
	beat_reliable: bool,
) -> tuple[list[MusicalRegionObservation], list[float], list[dict[str, object]], dict[str, object]]:
	if boundaries.size < 2:
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
			"candidateBoundaryCount": 0,
			"acceptedBoundaryCountBeforeConsolidation": 0,
			"acceptedBoundaryCountAfterConsolidation": 0,
			"consolidatedBoundaryCount": 0,
			"boundaryClusterCount": 0,
			"meanBeatsBetweenAcceptedBoundaries": 0.0,
			"medianBeatsBetweenAcceptedBoundaries": 0.0,
			"shortHarmonicRegionCount": 0,
			"harmonicRegionDurationDistribution": {
				"lt1Beat": 0,
				"1to2Beats": 0,
				"2to4Beats": 0,
				"gte4Beats": 0,
			},
			"consolidatedClusterExamples": [],
		}

	beat_ranges: list[tuple[int, int]] = []
	for left, right in zip(boundaries[:-1], boundaries[1:], strict=False):
		start = int(max(0, left))
		end = int(min(chroma.shape[1], right))
		if end > start:
			beat_ranges.append((start, end))
	if len(beat_ranges) == 0:
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
			"candidateBoundaryCount": 0,
			"acceptedBoundaryCountBeforeConsolidation": 0,
			"acceptedBoundaryCountAfterConsolidation": 0,
			"consolidatedBoundaryCount": 0,
			"boundaryClusterCount": 0,
			"meanBeatsBetweenAcceptedBoundaries": 0.0,
			"medianBeatsBetweenAcceptedBoundaries": 0.0,
			"shortHarmonicRegionCount": 0,
			"harmonicRegionDurationDistribution": {
				"lt1Beat": 0,
				"1to2Beats": 0,
				"2to4Beats": 0,
				"gte4Beats": 0,
			},
			"consolidatedClusterExamples": [],
		}

	beat_profiles: list[np.ndarray] = []
	beat_local_chords: list[str] = []
	for start, end in beat_ranges:
		region_chroma = np.asarray(chroma[:, start:end], dtype=np.float32)
		region_low = np.asarray(low_chroma[:, start:end], dtype=np.float32)
		root_aware = _estimate_region_root_aware_identity(region_chroma, region_low, key_estimate=key_estimate)
		profile = _normalize_nonnegative(
			(BEAT_PROFILE_HARMONIC_WEIGHT * np.mean(np.clip(region_chroma, 0.0, None), axis=1))
			+ (BEAT_PROFILE_LOW_WEIGHT * np.mean(np.clip(region_low, 0.0, None), axis=1))
		)
		beat_profiles.append(profile)
		beat_local_chords.append(str(root_aware["winner_label"]))

	novelty = _compute_harmonic_novelty(beat_profiles)
	selected_internal_idx, rejected, candidate_peaks, multires_diag = _select_harmonic_boundaries(
		novelty,
		beat_profiles,
		beat_ranges,
		beat_local_chords,
		hop_length=hop_length,
		sample_rate=sample_rate,
	)
	consolidated_idx, consolidation_diag = _consolidate_boundary_clusters(
		selected_internal_idx,
		novelty=novelty,
		candidate_peaks=candidate_peaks,
		beat_profiles=beat_profiles,
		beat_ranges=beat_ranges,
		beat_local_chords=beat_local_chords,
		hop_length=hop_length,
		sample_rate=sample_rate,
		chroma=chroma,
		low_chroma=low_chroma,
		key_estimate=key_estimate,
		beat_reliable=beat_reliable,
	)

	boundary_frames = [beat_ranges[idx][1] for idx in consolidated_idx]
	region_edges = [beat_ranges[0][0], *boundary_frames, beat_ranges[-1][1]]
	region_beat_lengths = _region_beat_lengths_from_edges(region_edges, beat_ranges)

	observations: list[MusicalRegionObservation] = []
	for region_idx, (left, right) in enumerate(zip(region_edges[:-1], region_edges[1:], strict=False)):
		if right <= left:
			continue
		region_chroma = np.asarray(chroma[:, left:right], dtype=np.float32)
		region_low = np.asarray(low_chroma[:, left:right], dtype=np.float32)
		obs = _build_region_observation(
			start=left,
			end=right,
			sample_rate=sample_rate,
			hop_length=hop_length,
			region_chroma=region_chroma,
			region_low_chroma=region_low,
			key_estimate=key_estimate,
		)
		obs = MusicalRegionObservation(
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
			decoded_chord=obs.decoded_chord,
			beat_length=float(region_beat_lengths[region_idx]) if region_idx < len(region_beat_lengths) else 0.0,
		)
		observations.append(obs)

	accepted_times = [float(frame * hop_length / sample_rate) for frame in boundary_frames]
	beats_between = [float(consolidated_idx[i + 1] - consolidated_idx[i]) for i in range(len(consolidated_idx) - 1)]
	mean_beats_between = float(mean(beats_between)) if beats_between else 0.0
	median_beats_between = float(median(beats_between)) if beats_between else 0.0

	short_region_count = sum(1 for obs in observations if (obs.beat_length > 0.0 and obs.beat_length < 1.0))
	distribution = _harmonic_region_beat_distribution(observations)
	diagnostics = {
		"localNoveltyCandidateCount": int(multires_diag.get("localNoveltyCandidateCount", 0)),
		"contextualBoundaryCandidateCount": int(multires_diag.get("contextualBoundaryCandidateCount", 0)),
		"multiResolutionAcceptedBoundaryCount": int(multires_diag.get("multiResolutionAcceptedBoundaryCount", 0)),
		"rejectedLocalOnlyBoundaryCount": int(multires_diag.get("rejectedLocalOnlyBoundaryCount", 0)),
		"shortMediumAgreementRate": float(multires_diag.get("shortMediumAgreementRate", 0.0)),
		"meanShortContextDistance": float(multires_diag.get("meanShortContextDistance", 0.0)),
		"meanMediumContextDistance": float(multires_diag.get("meanMediumContextDistance", 0.0)),
		"acceptedBoundaryExamples": multires_diag.get("acceptedBoundaryExamples", []),
		"rejectedLocalOnlyExamples": multires_diag.get("rejectedLocalOnlyExamples", []),
		"candidateBoundaryCount": len(candidate_peaks),
		"acceptedBoundaryCountBeforeConsolidation": len(selected_internal_idx),
		"acceptedBoundaryCountAfterConsolidation": len(consolidated_idx),
		"consolidatedBoundaryCount": max(0, len(selected_internal_idx) - len(consolidated_idx)),
		"boundaryClusterCount": int(consolidation_diag.get("boundaryClusterCount", 0)),
		"meanBeatsBetweenAcceptedBoundaries": mean_beats_between,
		"medianBeatsBetweenAcceptedBoundaries": median_beats_between,
		"shortHarmonicRegionCount": short_region_count,
		"harmonicRegionDurationDistribution": distribution,
		"consolidatedClusterExamples": consolidation_diag.get("consolidatedClusterExamples", []),
	}
	return observations, accepted_times, rejected, diagnostics


def _consensus_chord(chords: list[str], *, start: int, end: int) -> str | None:
	label, _ = _consensus_chord_with_ratio(chords, start=start, end=end)
	return label


def _region_beat_lengths_from_edges(region_edges: list[int], beat_ranges: list[tuple[int, int]]) -> list[float]:
	lengths: list[float] = []
	for left, right in zip(region_edges[:-1], region_edges[1:], strict=False):
		beats = 0
		for beat_left, beat_right in beat_ranges:
			if beat_right <= left:
				continue
			if beat_left >= right:
				break
			beats += 1
		lengths.append(float(beats))
	return lengths


def _harmonic_region_beat_distribution(observations: list[MusicalRegionObservation]) -> dict[str, int]:
	bins = {"lt1Beat": 0, "1to2Beats": 0, "2to4Beats": 0, "gte4Beats": 0}
	for obs in observations:
		beats = obs.beat_length
		if beats < 1.0:
			bins["lt1Beat"] += 1
		elif beats < 2.0:
			bins["1to2Beats"] += 1
		elif beats < 4.0:
			bins["2to4Beats"] += 1
		else:
			bins["gte4Beats"] += 1
	return bins
