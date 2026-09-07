"""Analysis orchestration: audio -> features -> detector -> smoothing -> segments."""

from __future__ import annotations

from pathlib import Path
from statistics import mean, median

import numpy as np

from chord_engine.analysis_models import (
	CONTRACT_VERSION,
	AnalysisError,
	AnalysisMetadata,
	AnalysisResult,
	CorrectionEvent,
	MusicalRegionObservation,
	PersistenceDecisionEvent,
	PipelineRun,
	SourceMetadata,
)
from chord_engine.audio import AudioBuffer, AudioDecodeError, load_audio
from chord_engine.confidence import calibrate_output_confidence as _calibrate_output_confidence
from chord_engine.diagnostics import summarize_analysis as _summarize_analysis
from chord_engine.detector import (
	FrameChordPrediction,
	FrameDetectionError,
	KeyEstimate,
	estimate_global_key,
	predict_frame_chord,
)
from chord_engine.features import (
	BeatTiming,
	FeatureExtractionError,
	estimate_beat_timing,
	extract_chroma,
	extract_harmonic_signal,
	extract_low_frequency_chroma,
)
from chord_engine.segmentation import MIN_SEGMENT_DURATION_MS, ChordSegment, segment_frame_predictions
from chord_engine.smoothing import smooth_frame_predictions
from chord_engine.templates import generate_chord_templates

from chord_engine.analysis_config import *  # noqa: F403 - central pipeline tuning constants


def analyze_audio(path: str | Path) -> AnalysisResult:
	"""Run full chord-analysis pipeline and return contract-shaped result."""
	run = _run_pipeline(
		path,
		use_harmonic_preprocessing=True,
		use_beat_sync=True,
		use_key_prior=True,
		use_context_correction=True,
	)
	return run.result


def compare_baseline_vs_improved(path: str | Path) -> dict[str, object]:
	"""Return summary metrics comparing baseline and improved analysis modes."""
	baseline = _run_pipeline(
		path,
		use_harmonic_preprocessing=False,
		use_beat_sync=False,
		use_key_prior=False,
		use_context_correction=False,
	)
	improved = _run_pipeline(
		path,
		use_harmonic_preprocessing=True,
		use_beat_sync=True,
		use_key_prior=True,
		use_context_correction=True,
	)

	baseline_summary = _summarize_analysis(baseline.result, baseline.detected_key, baseline.beat_timing)
	improved_summary = _summarize_analysis(improved.result, improved.detected_key, improved.beat_timing)

	harmonic_regions = improved.region_observations or []
	harmonic_region_durations = [obs.duration_seconds for obs in harmonic_regions]
	region_mean = float(mean(harmonic_region_durations)) if harmonic_region_durations else 0.0
	region_median = float(median(harmonic_region_durations)) if harmonic_region_durations else 0.0
	regions_per_minute = float(
		0.0
		if improved.result.source.duration <= 0.0
		else len(harmonic_regions) / (improved.result.source.duration / 60.0)
	)

	local_switch_no_boundary_examples = [
		{
			"timestamp": peak.get("timestamp"),
			"novelty": peak.get("novelty"),
			"leftLocalChord": peak.get("leftLocalChord"),
			"rightLocalChord": peak.get("rightLocalChord"),
			"reason": peak.get("reason"),
		}
		for peak in (improved.rejected_novelty_peaks or [])
		if peak.get("leftLocalChord") != peak.get("rightLocalChord")
	][:12]

	preserved_real_change_examples = []
	for idx in range(1, len(harmonic_regions)):
		left = harmonic_regions[idx - 1]
		right = harmonic_regions[idx]
		if left.decoded_chord != right.decoded_chord:
			preserved_real_change_examples.append(
				{
					"timestamp": right.start_seconds,
					"fromChord": left.decoded_chord,
					"toChord": right.decoded_chord,
					"leftLocalBest": left.local_best_chord,
					"rightLocalBest": right.local_best_chord,
				}
			)
	if len(preserved_real_change_examples) > 12:
		preserved_real_change_examples = preserved_real_change_examples[:12]

	return {
		"baseline": baseline_summary,
		"improvedBeforeContextCorrection": improved_summary,
		"improved": improved_summary,
		"transitionComparison": {
			"baselineTransitionCount": _count_chord_transitions(baseline.result.analysis.chords),
			"improvedTransitionCount": _count_chord_transitions(improved.result.analysis.chords),
			"baselineTransitionsPerMinute": float(
				0.0
				if baseline.result.source.duration <= 0.0
				else _count_chord_transitions(baseline.result.analysis.chords) / (baseline.result.source.duration / 60.0)
			),
			"improvedTransitionsPerMinute": float(
				0.0
				if improved.result.source.duration <= 0.0
				else _count_chord_transitions(improved.result.analysis.chords) / (improved.result.source.duration / 60.0)
			),
		},
		"musicalPersistence": {
			"decisionCount": len(harmonic_regions),
			"acceptedTransitionCount": _count_chord_transitions(improved.result.analysis.chords),
			"rejectedTransitionCount": len(improved.rejected_novelty_peaks or []),
			"acceptedTransitionReasonCounts": {
				"change-point-decoder": _count_chord_transitions(improved.result.analysis.chords)
			},
			"rootPreservingQualitySwitchCount": improved_summary.get("qualityChangeCount", 0),
			"weakAcceptedOverrideCount": 0,
			"ambiguousQualityDecisionCount": improved_summary.get("ambiguousQualityDecisionCount", 0),
			"rejectedTransitions": [
				{
					"regionIndex": idx,
					"start": peak.get("timestamp"),
					"end": peak.get("timestamp"),
					"currentChord": peak.get("leftLocalChord"),
					"candidateChord": peak.get("rightLocalChord"),
					"keepScore": 0.0,
					"switchScore": float(peak.get("novelty", 0.0)),
					"harmonicChangeEvidence": float(peak.get("novelty", 0.0)),
					"isRootPreservingQualitySwitch": bool(_is_root_preserving_quality_switch(str(peak.get("leftLocalChord")), str(peak.get("rightLocalChord")))),
					"reason": peak.get("reason", "novelty-rejected"),
				}
				for idx, peak in enumerate((improved.rejected_novelty_peaks or [])[:20])
			],
			"acceptedTransitions": [
				{
					"regionIndex": idx,
					"start": change["timestamp"],
					"end": change["timestamp"],
					"fromChord": change["fromChord"],
					"toChord": change["toChord"],
					"keepScore": 0.0,
					"switchScore": 0.0,
					"harmonicChangeEvidence": 0.0,
					"isRootPreservingQualitySwitch": bool(_is_root_preserving_quality_switch(str(change["fromChord"]), str(change["toChord"]))),
					"acceptedWithLowerSwitchScore": False,
					"reason": "change-point-decoder",
				}
				for idx, change in enumerate(preserved_real_change_examples[:20])
			],
			"rootAwareRegionExamples": [
				{
					"regionIndex": idx,
					"start": obs.start_seconds,
					"end": obs.end_seconds,
					"selectedRoot": obs.selected_root,
					"selectedRootConfidence": obs.selected_root_confidence,
					"majorQualityEvidence": obs.major_quality_evidence,
					"minorQualityEvidence": obs.minor_quality_evidence,
					"qualityMargin": obs.quality_margin,
					"templateScore": obs.template_score,
					"templateTopChord": obs.template_top_chord,
					"templateTopScore": obs.template_top_score,
					"rootAwareCombinedScore": obs.combined_score,
					"rootCandidateScores": obs.root_candidate_scores,
					"isQualityAmbiguous": obs.is_quality_ambiguous,
					"winnerChord": obs.winner_chord,
					"decodedChord": obs.decoded_chord,
					"topCandidates": [
						{"chord": chord, "score": score}
						for chord, score in (obs.top_candidates or [])[:3]
					],
				}
				for idx, obs in enumerate(harmonic_regions[:20])
			],
		},
		"contextCorrection": {
			"appliedCount": len(improved.correction_events or []),
			"appliedExamples": [
				{
					"index": event.index,
					"replacedChord": event.replaced_chord,
					"newChord": event.new_chord,
					"start": event.start,
					"end": event.end,
					"duration": event.duration,
					"score": event.score,
					"harmonicPlausibilityBefore": event.harmonic_plausibility_before,
					"harmonicPlausibilityAfter": event.harmonic_plausibility_after,
				}
				for event in (improved.correction_events or [])[:20]
			],
		},
		"harmonicSegmentation": {
			"localNoveltyCandidateCount": improved.local_novelty_candidate_count,
			"contextualBoundaryCandidateCount": improved.contextual_boundary_candidate_count,
			"multiResolutionAcceptedBoundaryCount": improved.multi_resolution_accepted_boundary_count,
			"rejectedLocalOnlyBoundaryCount": improved.rejected_local_only_boundary_count,
			"shortMediumAgreementRate": improved.short_medium_agreement_rate,
			"meanShortContextDistance": improved.mean_short_context_distance,
			"meanMediumContextDistance": improved.mean_medium_context_distance,
			"acceptedBoundaryExamples": (improved.accepted_boundary_examples or [])[:20],
			"rejectedLocalOnlyExamples": (improved.rejected_local_only_examples or [])[:20],
			"candidateBoundaryCount": improved.candidate_boundary_count,
			"acceptedBoundaryCountBeforeConsolidation": improved.accepted_boundary_count_before_consolidation,
			"acceptedBoundaryCountAfterConsolidation": improved.accepted_boundary_count_after_consolidation,
			"consolidatedBoundaryCount": improved.consolidated_boundary_count,
			"boundaryClusterCount": improved.boundary_cluster_count,
			"meanBeatsBetweenAcceptedBoundaries": improved.mean_beats_between_accepted_boundaries,
			"medianBeatsBetweenAcceptedBoundaries": improved.median_beats_between_accepted_boundaries,
			"shortHarmonicRegionCount": improved.short_harmonic_region_count,
			"harmonicRegionDurationDistribution": improved.harmonic_region_duration_distribution or {
				"lt1Beat": 0,
				"1to2Beats": 0,
				"2to4Beats": 0,
				"gte4Beats": 0,
			},
			"harmonicBoundaryCount": max(0, len(harmonic_regions) - 1),
			"harmonicRegionsPerMinute": regions_per_minute,
			"meanHarmonicRegionDuration": region_mean,
			"medianHarmonicRegionDuration": region_median,
			"rejectedNoveltyPeaks": (improved.rejected_novelty_peaks or [])[:30],
			"acceptedBoundaryTimestamps": improved.accepted_boundaries or [],
			"consolidatedClusterExamples": (improved.consolidated_cluster_examples or [])[:20],
			"perRegionSelectedChord": [obs.decoded_chord for obs in harmonic_regions],
			"localSwitchNoBoundaryExamples": local_switch_no_boundary_examples,
			"preservedRealChangeExamples": preserved_real_change_examples,
		},
		"globalDecoding": {
			"globalDecoderPathScore": improved.global_decoder_path_score,
			"regionDecisionsChangedByGlobalDecoding": improved.local_vs_decoded_region_changes,
			"localVsGlobal": [
				{
					"regionIndex": idx,
					"start": obs.start_seconds,
					"end": obs.end_seconds,
					"localBest": obs.local_best_chord,
					"decoded": obs.decoded_chord,
					"localBestScore": obs.local_best_score,
					"top3": [
						{"chord": chord, "score": score}
						for chord, score in (obs.top_candidates or [])[:3]
					],
				}
				for idx, obs in enumerate(harmonic_regions[:30])
			],
		},
	}


def _run_pipeline(
	path: str | Path,
	*,
	use_harmonic_preprocessing: bool,
	use_beat_sync: bool,
	use_key_prior: bool,
	use_context_correction: bool,
) -> PipelineRun:
	"""Internal pipeline executor with explicit feature toggles for benchmarking/tests."""
	try:
		audio = load_audio(path)
		harmonic_samples = extract_harmonic_signal(audio.samples) if use_harmonic_preprocessing else audio.samples
		feature_audio = AudioBuffer(samples=harmonic_samples, sample_rate=audio.sample_rate, duration=audio.duration)
		features = extract_chroma(feature_audio, use_harmonic_preprocessing=False)
		detected_key = estimate_global_key(features.chroma)

		if not use_beat_sync:
			key_prior_context = detected_key if use_key_prior else None
			frame_predictions = _predict_frames(features.chroma, key_estimate=key_prior_context)
			smoothed = smooth_frame_predictions(frame_predictions)
			segments = segment_frame_predictions(
				smoothed,
				hop_length=features.hop_length,
				sample_rate=features.sample_rate,
				min_segment_duration_ms=MIN_SEGMENT_DURATION_MS,
			)
			segments = _clamp_final_segment_end(segments, source_duration=audio.duration)
			segments = _calibrate_output_confidence(segments)
			result = AnalysisResult(
				version=CONTRACT_VERSION,
				source=SourceMetadata(
					path=str(path),
					duration=audio.duration,
					sampleRate=audio.sample_rate,
				),
				analysis=AnalysisMetadata(
					algorithm=ALGORITHM_ID,
					chords=segments,
				),
			)
			return PipelineRun(result=result, detected_key=detected_key)

		low_chroma = extract_low_frequency_chroma(
			harmonic_samples,
			sample_rate=features.sample_rate,
			hop_length=features.hop_length,
			n_frames=features.n_frames,
		)
		beat_timing = estimate_beat_timing(
			harmonic_samples,
			sample_rate=features.sample_rate,
			hop_length=features.hop_length,
			n_frames=features.n_frames,
			max_region_seconds=0.45,
		)
		harmonic_regions, accepted_boundaries, rejected_peaks, segmentation_diag = _build_harmonic_regions_from_change_points(
			boundaries=beat_timing.boundaries,
			sample_rate=features.sample_rate,
			hop_length=features.hop_length,
			chroma=features.chroma,
			low_chroma=low_chroma,
			key_estimate=detected_key if use_key_prior else None,
			beat_reliable=beat_timing.is_reliable,
		)

		decoded_regions, decoder_path_score, changed_count = _decode_global_chord_sequence(
			harmonic_regions,
			detected_key=detected_key if use_key_prior else None,
		)
		decoded_regions = _collapse_regions_by_decoded_chord(decoded_regions)
		decoded_boundaries = [region.start_seconds for region in decoded_regions[1:]] if len(decoded_regions) > 1 else []
		segments = _regions_to_segments(
			decoded_regions,
			hop_length=features.hop_length,
			sample_rate=features.sample_rate,
			source_duration=audio.duration,
		)
		detected_key = _resolve_relative_key_context(segments, detected_key)
		correction_events: list[CorrectionEvent] = []
		if use_context_correction:
			segments, correction_events = _apply_context_aware_short_segment_correction(
				segments,
				detected_key=detected_key,
			)
			segments, diminished_events = _suppress_weak_diminished_passing_segments(
				segments,
				detected_key=detected_key,
			)
			correction_events.extend(diminished_events)
			segments, targeted_events = _refine_low_confidence_long_holds_with_key_candidates(
				segments,
				chroma=features.chroma,
				low_chroma=low_chroma,
				boundaries=beat_timing.boundaries,
				hop_length=features.hop_length,
				sample_rate=features.sample_rate,
				detected_key=detected_key,
			)
			correction_events.extend(targeted_events)
			segments, sequence_events = _refine_long_holds_with_mini_harmonic_sequence_decoder(
				segments,
				chroma=features.chroma,
				low_chroma=low_chroma,
				boundaries=beat_timing.boundaries,
				hop_length=features.hop_length,
				sample_rate=features.sample_rate,
				detected_key=detected_key,
			)
			correction_events.extend(sequence_events)
			segments, cleanup_events = _suppress_weak_timeline_fragments(
				segments,
				detected_key=detected_key,
			)
			correction_events.extend(cleanup_events)
			segments, refinement_events = _refine_long_segments_with_harmonic_evidence(
				segments,
				chroma=features.chroma,
				low_chroma=low_chroma,
				boundaries=beat_timing.boundaries,
				hop_length=features.hop_length,
				sample_rate=features.sample_rate,
				detected_key=detected_key,
			)
			correction_events.extend(refinement_events)
			segments, drift_events = _refine_long_segments_with_sustained_harmonic_drift(
				segments,
				chroma=features.chroma,
				low_chroma=low_chroma,
				boundaries=beat_timing.boundaries,
				hop_length=features.hop_length,
				sample_rate=features.sample_rate,
				detected_key=detected_key,
			)
			correction_events.extend(drift_events)
			segments, diminished_events = _suppress_weak_diminished_passing_segments(
				segments,
				detected_key=detected_key,
			)
			correction_events.extend(diminished_events)
			segments, cleanup_events = _suppress_weak_timeline_fragments(
				segments,
				detected_key=detected_key,
			)
			correction_events.extend(cleanup_events)
		segments = _clamp_final_segment_end(segments, source_duration=audio.duration)
		segments = _calibrate_output_confidence(segments)

		result = AnalysisResult(
			version=CONTRACT_VERSION,
			source=SourceMetadata(
				path=str(path),
				duration=audio.duration,
				sampleRate=audio.sample_rate,
			),
			analysis=AnalysisMetadata(
				algorithm=ALGORITHM_ID,
				chords=segments,
			),
		)
		return PipelineRun(
			result=result,
			detected_key=detected_key,
			beat_timing=beat_timing,
			region_observations=decoded_regions,
			persistence_events=[],
			correction_events=correction_events,
			accepted_boundaries=decoded_boundaries,
			rejected_novelty_peaks=rejected_peaks,
			global_decoder_path_score=decoder_path_score,
			local_vs_decoded_region_changes=changed_count,
			candidate_boundary_count=int(segmentation_diag.get("candidateBoundaryCount", 0)),
			accepted_boundary_count_before_consolidation=int(segmentation_diag.get("acceptedBoundaryCountBeforeConsolidation", 0)),
			accepted_boundary_count_after_consolidation=int(segmentation_diag.get("acceptedBoundaryCountAfterConsolidation", 0)),
			consolidated_boundary_count=int(segmentation_diag.get("consolidatedBoundaryCount", 0)),
			boundary_cluster_count=int(segmentation_diag.get("boundaryClusterCount", 0)),
			mean_beats_between_accepted_boundaries=float(segmentation_diag.get("meanBeatsBetweenAcceptedBoundaries", 0.0)),
			median_beats_between_accepted_boundaries=float(segmentation_diag.get("medianBeatsBetweenAcceptedBoundaries", 0.0)),
			short_harmonic_region_count=int(segmentation_diag.get("shortHarmonicRegionCount", 0)),
			harmonic_region_duration_distribution=segmentation_diag.get("harmonicRegionDurationDistribution", None),
			consolidated_cluster_examples=segmentation_diag.get("consolidatedClusterExamples", None),
			local_novelty_candidate_count=int(segmentation_diag.get("localNoveltyCandidateCount", 0)),
			contextual_boundary_candidate_count=int(segmentation_diag.get("contextualBoundaryCandidateCount", 0)),
			multi_resolution_accepted_boundary_count=int(segmentation_diag.get("multiResolutionAcceptedBoundaryCount", 0)),
			rejected_local_only_boundary_count=int(segmentation_diag.get("rejectedLocalOnlyBoundaryCount", 0)),
			short_medium_agreement_rate=float(segmentation_diag.get("shortMediumAgreementRate", 0.0)),
			mean_short_context_distance=float(segmentation_diag.get("meanShortContextDistance", 0.0)),
			mean_medium_context_distance=float(segmentation_diag.get("meanMediumContextDistance", 0.0)),
			accepted_boundary_examples=segmentation_diag.get("acceptedBoundaryExamples", None),
			rejected_local_only_examples=segmentation_diag.get("rejectedLocalOnlyExamples", None),
		)
	except (AudioDecodeError, FeatureExtractionError, FrameDetectionError) as exc:
		raise AnalysisError(code=exc.code, message=exc.message) from exc
	except Exception as exc:  # pragma: no cover - public boundary guard
		raise AnalysisError("ANALYSIS_FAILED", "Failed to analyze audio") from exc



def _predict_frames(chroma: object, *, key_estimate: KeyEstimate | None) -> list[FrameChordPrediction]:
	arr = np.asarray(chroma, dtype=np.float32)
	if arr.ndim != 2 or arr.shape[0] != 12:
		raise AnalysisError("FEATURE_INVALID_SHAPE", "Expected chroma shape 12 x frames")
	if arr.shape[1] == 0:
		raise AnalysisError("FEATURE_EMPTY", "No chroma frames available")

	return [predict_frame_chord(arr[:, idx], key_estimate=key_estimate) for idx in range(arr.shape[1])]


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


def _consensus_chord(chords: list[str], *, start: int, end: int) -> str | None:
	label, _ = _consensus_chord_with_ratio(chords, start=start, end=end)
	return label


def _consensus_chord_with_ratio(chords: list[str], *, start: int, end: int) -> tuple[str | None, float]:
	left = max(0, int(start))
	right = min(len(chords), int(end))
	if right <= left:
		return None, 0.0

	counts: dict[str, int] = {}
	for chord in chords[left:right]:
		counts[chord] = counts.get(chord, 0) + 1
	if not counts:
		return None, 0.0

	# Deterministic tie-break: frequency, then lexical order.
	label, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
	window_len = max(1, right - left)
	ratio = float(count / window_len)
	return label, ratio


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


def _build_region_observation(
	*,
	start: int,
	end: int,
	sample_rate: int,
	hop_length: int,
	region_chroma: np.ndarray,
	region_low_chroma: np.ndarray,
	key_estimate: KeyEstimate | None,
) -> MusicalRegionObservation:
	duration_seconds = float(max(0.0, (end - start) * hop_length / sample_rate))
	root_aware = _estimate_region_root_aware_identity(region_chroma, region_low_chroma, key_estimate=key_estimate)
	combined_scores = dict(root_aware["combined_scores"])

	mean_vec = np.mean(np.asarray(region_chroma, dtype=np.float32), axis=1)
	energy = float(np.linalg.norm(mean_vec, ord=2))
	n_score = float(np.clip(1.0 - (energy * 1.7), 0.0, 1.0))
	if n_score > 0.0:
		combined_scores["N"] = n_score

	normalizer = float(sum(max(0.0, v) for v in combined_scores.values()))
	if normalizer <= 0.0:
		normalized_scores = {"N": 1.0}
	else:
		normalized_scores = {label: float(max(0.0, score) / normalizer) for label, score in combined_scores.items()}

	sorted_labels = sorted(normalized_scores.items(), key=lambda item: (-item[1], item[0]))
	local_best = sorted_labels[0][0]
	local_best_score = float(sorted_labels[0][1])
	template_top = str(root_aware.get("template_top_chord", ""))
	if template_top in normalized_scores and template_top != local_best:
		template_score = float(normalized_scores[template_top])
		if (local_best_score - template_score) <= 0.02:
			local_best = template_top
			local_best_score = template_score
			sorted_labels = sorted(normalized_scores.items(), key=lambda item: (-item[1], item[0]))
	runner_up = float(sorted_labels[1][1]) if len(sorted_labels) > 1 else 0.0
	advantage = float(max(0.0, local_best_score - runner_up))
	confidence = float(np.clip(local_best_score + (GLOBAL_DECODE_EVIDENCE_MARGIN_WEIGHT * advantage), 0.0, 1.0))

	return MusicalRegionObservation(
		start_frame=start,
		end_frame=end,
		start_seconds=float(start * hop_length / sample_rate),
		end_seconds=float(end * hop_length / sample_rate),
		duration_seconds=duration_seconds,
		winner_chord=local_best,
		winner_confidence=confidence,
		scores=normalized_scores,
		advantage_over_runner_up=advantage,
		root_candidate_scores=root_aware.get("root_scores"),
		selected_root=root_aware.get("selected_root"),
		selected_root_confidence=float(root_aware.get("selected_root_confidence", 0.0)),
		major_quality_evidence=float(root_aware.get("major_quality_evidence", 0.0)),
		minor_quality_evidence=float(root_aware.get("minor_quality_evidence", 0.0)),
		quality_margin=float(root_aware.get("quality_margin", 0.0)),
		template_score=float(root_aware.get("winner_template_score", 0.0)),
		template_top_chord=root_aware.get("template_top_chord"),
		template_top_score=float(root_aware.get("template_top_score", 0.0)),
		combined_score=float(root_aware.get("winner_combined_score", 0.0)),
		is_quality_ambiguous=bool(root_aware.get("is_quality_ambiguous", False)),
		top_candidates=[(label, score) for label, score in sorted_labels[:3]],
		local_best_chord=local_best,
		local_best_score=local_best_score,
		decoded_chord=local_best,
	)


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


def _harmonic_relationship_strength(prev_chord: str, curr_chord: str, detected_key: KeyEstimate | None) -> float:
	if prev_chord == "N" or curr_chord == "N":
		return 0.20

	prev_root, prev_quality = _parse_chord_quality(prev_chord)
	curr_root, curr_quality = _parse_chord_quality(curr_chord)
	if prev_root is None or curr_root is None:
		return 0.0
	if prev_root == curr_root and prev_quality != curr_quality:
		return 0.75

	prev_pc = _chord_root_pc(prev_chord)
	curr_pc = _chord_root_pc(curr_chord)
	if prev_pc is None or curr_pc is None:
		return 0.0

	interval = (curr_pc - prev_pc) % 12
	if interval in (5, 7):
		return 0.70

	if abs((curr_pc - prev_pc) % 12) in (3, 9) and prev_quality != curr_quality:
		return 0.30

	if detected_key is not None and _is_diatonic_chord(prev_chord, detected_key) and _is_diatonic_chord(curr_chord, detected_key):
		return 0.44

	return 0.22


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


def _aggregate_predictions_by_boundaries(
	predictions: list[FrameChordPrediction],
	boundaries: np.ndarray,
) -> list[FrameChordPrediction]:
	if len(predictions) == 0:
		return predictions

	if boundaries.size < 2:
		return predictions

	result = list(predictions)

	for left, right in zip(boundaries[:-1], boundaries[1:], strict=False):
		start = int(max(0, left))
		end = int(min(len(result), right))
		if end <= start:
			continue

		region = result[start:end]
		winner, confidence = _pick_region_winner(region)
		for idx in range(start, end):
			result[idx] = FrameChordPrediction(chord=winner, confidence=confidence)

	return result


def _build_musical_region_observations(
	predictions: list[FrameChordPrediction],
	*,
	boundaries: np.ndarray,
	sample_rate: int,
	hop_length: int,
	chroma: np.ndarray,
	low_chroma: np.ndarray,
	key_estimate: KeyEstimate | None,
) -> list[MusicalRegionObservation]:
	if len(predictions) == 0 or boundaries.size < 2:
		return []

	observations: list[MusicalRegionObservation] = []
	for left, right in zip(boundaries[:-1], boundaries[1:], strict=False):
		start = int(max(0, left))
		end = int(min(len(predictions), right))
		if end <= start:
			continue

		region = predictions[start:end]
		region_chroma = np.asarray(chroma[:, start:end], dtype=np.float32)
		region_low_chroma = np.asarray(low_chroma[:, start:end], dtype=np.float32)
		scores: dict[str, float] = {}
		for pred in region:
			scores[pred.chord] = scores.get(pred.chord, 0.0) + float(np.clip(pred.confidence, 0.0, 1.0))

		if region_chroma.size > 0:
			root_aware = _estimate_region_root_aware_identity(
				region_chroma,
				region_low_chroma,
				key_estimate=key_estimate,
			)
			for label, value in root_aware["combined_scores"].items():
				scores[label] = scores.get(label, 0.0) + value

		total_score = float(sum(scores.values()))
		if total_score <= 0.0:
			normalized_scores = {"N": 1.0}
			winner = "N"
			winner_confidence = 0.0
			advantage = 1.0
			root_candidate_scores = {"N": 1.0}
			selected_root = None
			selected_root_confidence = 0.0
			major_quality_evidence = 0.0
			minor_quality_evidence = 0.0
			quality_margin = 0.0
			template_score = 0.0
			template_top_chord = None
			template_top_score = 0.0
			combined_score = 0.0
			is_quality_ambiguous = False
		else:
			normalized_scores = {label: float(score / total_score) for label, score in scores.items()}
			sorted_labels = sorted(normalized_scores.items(), key=lambda item: (-item[1], item[0]))
			winner = sorted_labels[0][0]
			runner_up = sorted_labels[1][1] if len(sorted_labels) > 1 else 0.0
			advantage = float(max(0.0, sorted_labels[0][1] - runner_up))
			winner_confidence = float(
				np.clip(
					mean([pred.confidence for pred in region if pred.chord == winner]) if any(pred.chord == winner for pred in region) else 0.0,
					0.0,
					1.0,
				)
			)
			if region_chroma.size > 0:
				root_candidate_scores = root_aware["root_scores"]
				selected_root = root_aware["selected_root"]
				selected_root_confidence = float(root_aware["selected_root_confidence"])
				major_quality_evidence = float(root_aware["major_quality_evidence"])
				minor_quality_evidence = float(root_aware["minor_quality_evidence"])
				quality_margin = float(root_aware["quality_margin"])
				template_score = float(root_aware["winner_template_score"])
				template_top_chord = str(root_aware["template_top_chord"])
				template_top_score = float(root_aware["template_top_score"])
				combined_score = float(root_aware["winner_combined_score"])
				is_quality_ambiguous = bool(root_aware["is_quality_ambiguous"])
			else:
				root_candidate_scores = {}
				selected_root = None
				selected_root_confidence = 0.0
				major_quality_evidence = 0.0
				minor_quality_evidence = 0.0
				quality_margin = 0.0
				template_score = 0.0
				template_top_chord = None
				template_top_score = 0.0
				combined_score = 0.0
				is_quality_ambiguous = False

		duration_seconds = float(max(0.0, (end - start) * hop_length / sample_rate))
		observations.append(
			MusicalRegionObservation(
				start_frame=start,
				end_frame=end,
				start_seconds=float(start * hop_length / sample_rate),
				end_seconds=float(end * hop_length / sample_rate),
				duration_seconds=duration_seconds,
				winner_chord=winner,
				winner_confidence=winner_confidence,
				scores=normalized_scores,
				advantage_over_runner_up=advantage,
				root_candidate_scores=root_candidate_scores,
				selected_root=selected_root,
				selected_root_confidence=selected_root_confidence,
				major_quality_evidence=major_quality_evidence,
				minor_quality_evidence=minor_quality_evidence,
				quality_margin=quality_margin,
				template_score=template_score,
				template_top_chord=template_top_chord,
				template_top_score=template_top_score,
				combined_score=combined_score,
				is_quality_ambiguous=is_quality_ambiguous,
			)
		)

	return observations


def _apply_musical_time_harmonic_persistence(
	observations: list[MusicalRegionObservation],
	*,
	n_frames: int,
	detected_key: KeyEstimate | None,
) -> tuple[list[FrameChordPrediction], list[PersistenceDecisionEvent]]:
	if len(observations) == 0:
		return [], []

	decisions: list[tuple[str, float]] = []
	events: list[PersistenceDecisionEvent] = []

	current_chord = observations[0].winner_chord
	current_confidence = observations[0].winner_confidence
	current_streak_beats = 1
	pending_chord: str | None = None
	pending_count = 0
	pending_duration = 0.0

	for idx, obs in enumerate(observations):
		if idx == 0:
			decisions.append((current_chord, current_confidence))
			continue

		candidate = obs.winner_chord
		if candidate == current_chord:
			current_streak_beats += 1
			current_confidence = float(np.clip((current_confidence * 0.5) + (obs.winner_confidence * 0.5), 0.0, 1.0))
			pending_chord = None
			pending_count = 0
			pending_duration = 0.0
			decisions.append((current_chord, current_confidence))
			continue

		if pending_chord == candidate:
			pending_count += 1
			pending_duration += obs.duration_seconds
		else:
			pending_chord = candidate
			pending_count = 1
			pending_duration = obs.duration_seconds

		current_score = float(obs.scores.get(current_chord, 0.0))
		candidate_score = float(obs.scores.get(candidate, 0.0))
		advantage = float(max(0.0, candidate_score - current_score))
		duration_factor = float(np.clip(obs.duration_seconds / MUSICAL_SWITCH_REFERENCE_SECONDS, 0.0, 1.0))
		consecutive_support = float(np.clip(pending_count / 2.0, 0.0, 1.0))
		persistence_factor = float(np.clip(pending_duration / MUSICAL_SWITCH_SUPPORT_SECONDS, 0.0, 1.0))
		neighbor_persistence = (consecutive_support + persistence_factor) / 2.0
		lookahead_same = 1.0 if idx + 1 < len(observations) and observations[idx + 1].winner_chord == candidate else 0.0

		candidate_plausibility = _harmonic_plausibility(candidate, detected_key)
		current_plausibility = _harmonic_plausibility(current_chord, detected_key)
		harmonic_change_evidence = _compute_harmonic_change_evidence(
			observations,
			idx=idx,
			current_chord=current_chord,
			candidate_chord=candidate,
		)
		is_quality_switch = _is_root_preserving_quality_switch(current_chord, candidate)
		quality_ambiguous = bool(obs.is_quality_ambiguous and is_quality_switch)

		switch_score = (
			0.34 * obs.winner_confidence
			+ 0.22 * advantage
			+ 0.16 * duration_factor
			+ 0.18 * neighbor_persistence
			+ 0.06 * candidate_plausibility
			+ 0.04 * lookahead_same
			+ 0.10 * max(0.0, harmonic_change_evidence)
		)
		keep_score = (
			0.36 * current_score
			+ 0.24 * float(np.clip(current_streak_beats / MUSICAL_KEEP_STREAK_REFERENCE_BEATS, 0.0, 1.0))
			+ 0.20 * current_plausibility
			+ 0.20 * float(np.clip(1.0 - advantage, 0.0, 1.0))
			+ 0.10 * max(0.0, -harmonic_change_evidence)
		)

		strong_immediate = bool(
			obs.winner_confidence >= MUSICAL_STRONG_IMMEDIATE_CONFIDENCE
			and advantage >= MUSICAL_STRONG_IMMEDIATE_ADVANTAGE
			and (duration_factor >= 0.45 or lookahead_same >= 1.0)
		)
		dominant_override = bool(
			candidate_score >= MUSICAL_DOMINANT_OVERRIDE_CANDIDATE_SCORE
			and advantage >= MUSICAL_DOMINANT_OVERRIDE_ADVANTAGE
			and duration_factor >= 0.35
		)
		persistent_ready = pending_count >= 2 and pending_duration >= MUSICAL_SWITCH_SUPPORT_SECONDS
		quality_switch_ready = bool(
			not is_quality_switch
			or (
				advantage >= MUSICAL_QUALITY_SWITCH_MIN_ADVANTAGE
				and harmonic_change_evidence >= MUSICAL_QUALITY_SWITCH_MIN_HARMONIC_CHANGE
				and not quality_ambiguous
				and (pending_count >= 2 or lookahead_same >= 1.0)
			)
		)
		ordinary_switch_margin = MUSICAL_SWITCH_MARGIN + (MUSICAL_QUALITY_SWITCH_EXTRA_MARGIN if is_quality_switch else 0.0)
		ordinary_evidence_ready = switch_score > keep_score + ordinary_switch_margin
		ordinary_override = quality_switch_ready and ordinary_evidence_ready and (
			strong_immediate or dominant_override or persistent_ready
		)

		exceptional_override = bool(
			quality_switch_ready
			and switch_score <= keep_score
			and pending_count >= MUSICAL_EXCEPTIONAL_SUPPORT_BEATS
			and pending_duration >= MUSICAL_EXCEPTIONAL_SUPPORT_SECONDS
			and neighbor_persistence >= MUSICAL_EXCEPTIONAL_SUPPORT_PERSISTENCE
			and harmonic_change_evidence >= MUSICAL_EXCEPTIONAL_SUPPORT_HARMONIC_CHANGE
			and obs.winner_confidence >= MUSICAL_EXCEPTIONAL_SUPPORT_CONFIDENCE
			and lookahead_same >= 1.0
		)

		should_switch = ordinary_override or exceptional_override
		accepted_with_lower_switch_score = bool(should_switch and switch_score <= keep_score)

		reason = (
			"exceptional-multibeat-context-override"
			if exceptional_override
			else (
				"keep-quality-ambiguous"
				if (not should_switch and quality_ambiguous)
				else (
				"strong-immediate"
				if should_switch and strong_immediate
				else (
					"dominant-override"
					if should_switch and dominant_override
					else (
						"persistent-support" if should_switch and persistent_ready else (
							"keep-quality-switch-hysteresis" if (not should_switch and is_quality_switch and not quality_switch_ready) else "keep-insufficient-switch-evidence"
						)
					)
				)
				)
			)
		)
		events.append(
			PersistenceDecisionEvent(
				region_index=idx,
				region_start_seconds=obs.start_seconds,
				region_end_seconds=obs.end_seconds,
				action="switch" if should_switch else "keep",
				current_chord=current_chord,
				candidate_chord=candidate,
				keep_score=float(keep_score),
				switch_score=float(switch_score),
				advantage=float(advantage),
				candidate_confidence=float(obs.winner_confidence),
				consecutive_support=pending_count,
				neighbor_persistence=float(neighbor_persistence),
				harmonic_change_evidence=float(harmonic_change_evidence),
				is_root_preserving_quality_switch=is_quality_switch,
				accepted_with_lower_switch_score=accepted_with_lower_switch_score,
				reason=reason,
			)
		)

		if should_switch:
			current_chord = candidate
			current_confidence = float(np.clip(obs.winner_confidence, 0.0, 1.0))
			current_streak_beats = 1
			pending_chord = None
			pending_count = 0
			pending_duration = 0.0
		else:
			current_streak_beats += 1
			current_confidence = float(np.clip((current_confidence * 0.8) + (current_score * 0.2), 0.0, 1.0))

		decisions.append((current_chord, current_confidence))

	frame_predictions: list[FrameChordPrediction] = [FrameChordPrediction(chord="N", confidence=0.0) for _ in range(n_frames)]
	for obs, decision in zip(observations, decisions, strict=True):
		chord, confidence = decision
		for frame_idx in range(obs.start_frame, min(obs.end_frame, n_frames)):
			frame_predictions[frame_idx] = FrameChordPrediction(chord=chord, confidence=float(np.clip(confidence, 0.0, 1.0)))

	return frame_predictions, events


def _estimate_region_root_aware_identity(
	region_chroma: np.ndarray,
	region_low_chroma: np.ndarray,
	*,
	key_estimate: KeyEstimate | None,
) -> dict[str, object]:
	"""Score chord identity by separating root and quality evidence in a region."""
	templates = generate_chord_templates()
	labels = sorted(templates.keys())

	energy = np.mean(np.clip(region_chroma, 0.0, None), axis=1)
	low_energy = np.mean(np.clip(region_low_chroma, 0.0, None), axis=1)
	energy_sum = float(np.sum(energy))
	low_sum = float(np.sum(low_energy))
	if energy_sum > 0.0:
		energy = energy / energy_sum
	if low_sum > 0.0:
		low_energy = low_energy / low_sum

	root_scores_pc: dict[int, float] = {}
	for root_pc in range(12):
		root_val = float(energy[root_pc])
		fifth_val = float(energy[(root_pc + 7) % 12])
		bass_root = float(low_energy[root_pc])
		bass_fifth = float(low_energy[(root_pc + 7) % 12])
		root_scores_pc[root_pc] = (
			ROOT_AWARE_ROOT_ENERGY_WEIGHT * root_val
			+ ROOT_AWARE_FIFTH_ENERGY_WEIGHT * fifth_val
			+ ROOT_AWARE_BASS_ROOT_WEIGHT * bass_root
			+ ROOT_AWARE_BASS_FIFTH_WEIGHT * bass_fifth
		)

	root_score_values = np.asarray([root_scores_pc[idx] for idx in range(12)], dtype=np.float32)
	root_score_values = _normalize_nonnegative(root_score_values)
	root_scores_pc = {idx: float(root_score_values[idx]) for idx in range(12)}

	selected_root_pc = int(np.argmax(root_score_values))
	sorted_root_scores = sorted(root_score_values.tolist(), reverse=True)
	root_margin = float(sorted_root_scores[0] - sorted_root_scores[1]) if len(sorted_root_scores) > 1 else float(sorted_root_scores[0])
	selected_root_confidence = float(np.clip(0.6 * root_score_values[selected_root_pc] + 0.4 * max(0.0, root_margin), 0.0, 1.0))

	major_quality_evidence = float(
		ROOT_AWARE_QUALITY_THIRD_WEIGHT * energy[(selected_root_pc + 4) % 12]
		+ ROOT_AWARE_QUALITY_SUPPORT_WEIGHT * ((energy[selected_root_pc] + energy[(selected_root_pc + 7) % 12]) / 2.0)
	)
	minor_quality_evidence = float(
		ROOT_AWARE_QUALITY_THIRD_WEIGHT * energy[(selected_root_pc + 3) % 12]
		+ ROOT_AWARE_QUALITY_SUPPORT_WEIGHT * ((energy[selected_root_pc] + energy[(selected_root_pc + 7) % 12]) / 2.0)
	)
	quality_margin = float(abs(major_quality_evidence - minor_quality_evidence))
	is_quality_ambiguous = bool(quality_margin < ROOT_AWARE_QUALITY_AMBIGUOUS_MARGIN)

	region_vec = np.mean(region_chroma, axis=1)
	template_scores: dict[str, float] = {}
	for label in labels:
		template_scores[label] = _template_score_with_soft_key(region_vec, templates[label], label, key_estimate)

	template_values = np.asarray([template_scores[label] for label in labels], dtype=np.float32)
	template_values = _normalize_nonnegative(template_values - np.min(template_values))
	template_scores_norm = {label: float(template_values[idx]) for idx, label in enumerate(labels)}
	template_top_label = sorted(template_scores_norm.items(), key=lambda item: (-item[1], item[0]))[0][0]
	template_top_root = _chord_root_pc(template_top_label)

	combined_scores: dict[str, float] = {}
	for label in labels:
		root_pc = _chord_root_pc(label)
		if root_pc is None:
			continue
		_, label_quality = _parse_chord_quality(label)
		is_minor_label = label_quality == "minor"
		is_diminished_label = label_quality == "diminished"
		diminished_quality_evidence = float(
			0.52 * energy[(root_pc + 3) % 12]
			+ 0.38 * energy[(root_pc + 6) % 12]
			+ 0.10 * energy[root_pc]
		)
		if is_diminished_label:
			quality_component = diminished_quality_evidence
		elif is_minor_label:
			quality_component = minor_quality_evidence
		else:
			quality_component = major_quality_evidence
		if is_quality_ambiguous:
			quality_component *= 0.55
		key_context_adjust = 0.0
		if key_estimate is not None:
			key_context_adjust = ROOT_AWARE_KEY_CONTEXT_WEIGHT * (_harmonic_plausibility(label, key_estimate) - 0.5)
			strong_local_quality = bool(
				root_pc == selected_root_pc
				and quality_margin >= ROOT_AWARE_QUALITY_AMBIGUOUS_MARGIN
				and (
					(is_diminished_label and (diminished_quality_evidence > major_quality_evidence))
					or (is_minor_label and (minor_quality_evidence > major_quality_evidence))
					or ((not is_minor_label and not is_diminished_label) and (major_quality_evidence > minor_quality_evidence))
				)
			)
			template_root_support = bool(template_top_root is not None and root_pc == template_top_root)
			if strong_local_quality and not _is_diatonic_chord(label, key_estimate):
				key_context_adjust = max(key_context_adjust, -0.005)
			if template_root_support and label == template_top_label and not _is_diatonic_chord(label, key_estimate):
				key_context_adjust = max(key_context_adjust, -0.003)
		template_root_bonus = 0.0
		if template_top_root is not None and root_pc == template_top_root:
			template_root_bonus = 0.04
		exact_template_bonus = 0.02 if label == template_top_label else 0.0
		combined_scores[label] = (
			ROOT_AWARE_TEMPLATE_WEIGHT * template_scores_norm[label]
			+ ROOT_AWARE_ROOT_WEIGHT * root_scores_pc[root_pc]
			+ ROOT_AWARE_QUALITY_WEIGHT * quality_component
			+ key_context_adjust
			+ template_root_bonus
			+ exact_template_bonus
		)

	combined_values = np.asarray([combined_scores[label] for label in labels], dtype=np.float32)
	combined_values = _normalize_nonnegative(combined_values)
	combined_scores = {label: float(combined_values[idx]) for idx, label in enumerate(labels)}

	winner_label = sorted(combined_scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
	selected_root_label = _root_name_from_pc(selected_root_pc)
	root_candidate_scores = {
		_root_name_from_pc(pc): float(score)
		for pc, score in sorted(root_scores_pc.items(), key=lambda item: item[0])
	}

	return {
		"winner_label": winner_label,
		"combined_scores": combined_scores,
		"root_scores": root_candidate_scores,
		"selected_root": selected_root_label,
		"selected_root_confidence": selected_root_confidence,
		"major_quality_evidence": major_quality_evidence,
		"minor_quality_evidence": minor_quality_evidence,
		"quality_margin": quality_margin,
		"winner_template_score": template_scores_norm.get(winner_label, 0.0),
		"template_top_chord": template_top_label,
		"template_top_score": template_scores_norm.get(template_top_label, 0.0),
		"winner_combined_score": combined_scores.get(winner_label, 0.0),
		"is_quality_ambiguous": is_quality_ambiguous,
	}


def _normalize_nonnegative(values: np.ndarray) -> np.ndarray:
	arr = np.asarray(values, dtype=np.float32)
	arr = np.clip(arr, 0.0, None)
	total = float(np.sum(arr))
	if total <= 0.0:
		if arr.size == 0:
			return arr
		return np.full(arr.shape, 1.0 / arr.size, dtype=np.float32)
	return arr / total


def _template_score_with_soft_key(
	frame_vec: np.ndarray,
	template: np.ndarray,
	chord_name: str,
	key_estimate: KeyEstimate | None,
) -> float:
	raw = _cosine_similarity(frame_vec, template)
	bonus = _key_prior_bonus(chord_name, key_estimate) if key_estimate is not None else 0.0
	return float(raw + bonus)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
	a_norm = float(np.linalg.norm(a, ord=2))
	b_norm = float(np.linalg.norm(b, ord=2))
	if a_norm <= 1e-12 or b_norm <= 1e-12:
		return 0.0
	value = float(np.dot(a, b) / (a_norm * b_norm))
	if not np.isfinite(value):
		return 0.0
	return float(np.clip(value, -1.0, 1.0))


def _key_prior_bonus(chord_name: str, key_estimate: KeyEstimate) -> float:
	root_pc = _chord_root_pc(chord_name)
	if root_pc is None:
		return 0.0

	_, quality = _parse_chord_quality(chord_name)
	match = _diatonic_match_score(root_pc, quality, key_estimate)
	return 0.08 * key_estimate.confidence * match


def _diatonic_match_score(root_pc: int, quality: str | None, key_estimate: KeyEstimate) -> float:
	if key_estimate.mode == "major":
		diatonic_chords = {
			0: "major",
			2: "minor",
			4: "minor",
			5: "major",
			7: "major",
			9: "minor",
			11: "diminished",
		}
		scale_intervals = (0, 2, 4, 5, 7, 9, 11)
	else:
		diatonic_chords = {
			0: "minor",
			2: "diminished",
			3: "major",
			5: "minor",
			7: "minor",
			8: "major",
			10: "major",
		}
		scale_intervals = (0, 2, 3, 5, 7, 8, 10)

	interval = (root_pc - key_estimate.tonic_pc) % 12
	if interval not in scale_intervals:
		return 0.0

	expected_quality = diatonic_chords.get(interval)
	if expected_quality is None:
		return 0.0
	if expected_quality == quality:
		return 1.0
	return 0.30


def _chord_root_pc(chord_name: str) -> int | None:
	if chord_name.endswith("dim"):
		name = chord_name[:-3]
	elif chord_name.endswith("m"):
		name = chord_name[:-1]
	else:
		name = chord_name
	lookup = {
		"C": 0,
		"C#": 1,
		"D": 2,
		"D#": 3,
		"E": 4,
		"F": 5,
		"F#": 6,
		"G": 7,
		"G#": 8,
		"A": 9,
		"A#": 10,
		"B": 11,
	}
	return lookup.get(name)


def _root_name_from_pc(pc: int) -> str:
	labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
	return labels[pc % 12]


def _compute_harmonic_change_evidence(
	observations: list[MusicalRegionObservation],
	*,
	idx: int,
	current_chord: str,
	candidate_chord: str,
) -> float:
	"""Compare current vs candidate support across neighboring musical-time observations."""
	left = max(0, idx - 1)
	right = min(len(observations), idx + 2)
	window = observations[left:right]
	if not window:
		return 0.0

	current_support = float(mean([obs.scores.get(current_chord, 0.0) for obs in window]))
	candidate_support = float(mean([obs.scores.get(candidate_chord, 0.0) for obs in window]))
	return float(np.clip(candidate_support - current_support, -1.0, 1.0))


def _parse_chord_quality(label: str) -> tuple[str | None, str | None]:
	if label == "N":
		return None, None
	if label.endswith("dim"):
		root = label[:-3]
		return root, "diminished"
	if label.endswith("m"):
		root = label[:-1]
		return root, "minor"
	return label, "major"


def _is_root_preserving_quality_switch(current: str, candidate: str) -> bool:
	current_root, current_quality = _parse_chord_quality(current)
	candidate_root, candidate_quality = _parse_chord_quality(candidate)
	if current_root is None or candidate_root is None:
		return False
	if current_root != candidate_root:
		return False
	if current_quality == candidate_quality:
		return False
	return True


def _pick_region_winner(region: list[FrameChordPrediction]) -> tuple[str, float]:
	scores: dict[str, float] = {}
	for pred in region:
		scores[pred.chord] = scores.get(pred.chord, 0.0) + float(np.clip(pred.confidence, 0.0, 1.0))

	if not scores:
		return "N", 0.0

	best_score = max(scores.values())
	best_labels = [label for label, score in scores.items() if np.isclose(score, best_score)]
	winner = sorted(best_labels)[0]
	confidence = float(np.clip(mean([pred.confidence for pred in region if pred.chord == winner]), 0.0, 1.0))
	return winner, confidence


def _clamp_final_segment_end(segments: list[ChordSegment], *, source_duration: float) -> list[ChordSegment]:
	if not segments:
		return segments

	last = segments[-1]
	if last.end <= source_duration + END_TOLERANCE_SECONDS:
		return segments

	new_end = max(last.start + 1e-9, source_duration)
	segments[-1] = ChordSegment(
		start=last.start,
		end=new_end,
		chord=last.chord,
		confidence=last.confidence,
	)
	return segments



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


def _merge_adjacent_same_chord_segments(segments: list[ChordSegment]) -> list[ChordSegment]:
	if not segments:
		return []

	merged = [segments[0]]
	for segment in segments[1:]:
		last = merged[-1]
		if segment.chord != last.chord:
			merged.append(segment)
			continue

		left_duration = _segment_duration(last)
		right_duration = _segment_duration(segment)
		total = left_duration + right_duration
		if total <= 0.0:
			confidence = max(last.confidence, segment.confidence)
		else:
			confidence = ((last.confidence * left_duration) + (segment.confidence * right_duration)) / total

		merged[-1] = ChordSegment(
			start=last.start,
			end=segment.end,
			chord=last.chord,
			confidence=float(np.clip(confidence, 0.0, 1.0)),
		)

	return merged


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


def _weighted_confidence(left: ChordSegment, right: ChordSegment) -> float:
	left_duration = _segment_duration(left)
	right_duration = _segment_duration(right)
	total = left_duration + right_duration
	if total <= 0.0:
		return float(np.clip(max(left.confidence, right.confidence), 0.0, 1.0))
	return float(np.clip(((left.confidence * left_duration) + (right.confidence * right_duration)) / total, 0.0, 1.0))


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


def _diatonic_chord_labels_for_key(key_estimate: KeyEstimate) -> set[str]:
	if key_estimate.mode == "major":
		qualities = {
			0: "",
			2: "m",
			4: "m",
			5: "",
			7: "",
			9: "m",
		}
	else:
		qualities = {
			0: "m",
			3: "",
			5: "m",
			7: "m",
			8: "",
			10: "",
		}
	return {
		f"{_root_name_from_pc((key_estimate.tonic_pc + interval) % 12)}{suffix}"
		for interval, suffix in qualities.items()
	}


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


def _resolve_relative_key_context(
	segments: list[ChordSegment],
	detected_key: KeyEstimate | None,
) -> KeyEstimate | None:
	"""Prefer relative major when the chord-duration center clearly points there."""
	if detected_key is None or detected_key.mode != "minor" or not segments:
		return detected_key

	relative_major_pc = (detected_key.tonic_pc + 3) % 12
	major_tonic = _root_name_from_pc(relative_major_pc)
	major_dominant = _root_name_from_pc((relative_major_pc + 7) % 12)
	major_subdominant = _root_name_from_pc((relative_major_pc + 5) % 12)
	major_mediant_minor = f"{_root_name_from_pc((relative_major_pc + 4) % 12)}m"

	minor_tonic = f"{_root_name_from_pc(detected_key.tonic_pc)}m"
	minor_dominant_minor = f"{_root_name_from_pc((detected_key.tonic_pc + 7) % 12)}m"
	minor_subdominant_minor = f"{_root_name_from_pc((detected_key.tonic_pc + 5) % 12)}m"

	durations = _chord_duration_map(segments)
	major_center_score = (
		durations.get(major_tonic, 0.0)
		+ 0.80 * durations.get(major_dominant, 0.0)
		+ 0.70 * durations.get(major_subdominant, 0.0)
		+ 0.35 * durations.get(major_mediant_minor, 0.0)
	)
	minor_center_score = (
		durations.get(minor_tonic, 0.0)
		+ 0.80 * durations.get(minor_dominant_minor, 0.0)
		+ 0.70 * durations.get(minor_subdominant_minor, 0.0)
	)
	major_tonic_duration = durations.get(major_tonic, 0.0)
	minor_tonic_duration = durations.get(minor_tonic, 0.0)

	if major_center_score >= (minor_center_score * 1.12) and major_tonic_duration >= (minor_tonic_duration * 1.05):
		return KeyEstimate(
			tonic_pc=relative_major_pc,
			mode="major",
			confidence=float(np.clip(detected_key.confidence * 0.92, 0.0, 1.0)),
		)

	return detected_key


def _chord_duration_map(segments: list[ChordSegment]) -> dict[str, float]:
	durations: dict[str, float] = {}
	for segment in segments:
		durations[segment.chord] = durations.get(segment.chord, 0.0) + _segment_duration(segment)
	return durations


def _segment_duration(segment: ChordSegment) -> float:
	return float(max(0.0, segment.end - segment.start))


def _neighbor_support(segment: ChordSegment, detected_key: KeyEstimate | None) -> float:
	duration_norm = min(1.0, _segment_duration(segment) / CONTEXT_NEIGHBOR_MIN_DURATION_SECONDS)
	plausibility = _harmonic_plausibility(segment.chord, detected_key)
	support = 0.52 * segment.confidence + 0.28 * duration_norm + 0.20 * plausibility
	return float(np.clip(support, 0.0, 1.0))


def _is_diatonic_chord(chord_name: str, detected_key: KeyEstimate | None) -> bool:
	if detected_key is None or chord_name == "N":
		return False

	root_to_pc = {
		"C": 0,
		"C#": 1,
		"D": 2,
		"D#": 3,
		"E": 4,
		"F": 5,
		"F#": 6,
		"G": 7,
		"G#": 8,
		"A": 9,
		"A#": 10,
		"B": 11,
	}

	root_name, quality = _parse_chord_quality(chord_name)
	root_pc = root_to_pc.get(root_name)
	if root_pc is None:
		return False

	interval = (root_pc - detected_key.tonic_pc) % 12
	if detected_key.mode == "major":
		diatonic_chords = {
			0: "major",
			2: "minor",
			4: "minor",
			5: "major",
			7: "major",
			9: "minor",
			11: "diminished",
		}
	else:
		diatonic_chords = {
			0: "minor",
			2: "diminished",
			3: "major",
			5: "minor",
			7: "minor",
			8: "major",
			10: "major",
		}

	expected_quality = diatonic_chords.get(interval)
	if expected_quality is None:
		return False
	return expected_quality == quality


def _harmonic_plausibility(chord_name: str, detected_key: KeyEstimate | None) -> float:
	"""Return bounded plausibility used as a soft context signal for correction."""
	if chord_name == "N":
		return HARMONIC_PLAUSIBILITY_MODAL_OR_BORROWED
	if detected_key is None:
		return HARMONIC_PLAUSIBILITY_MODAL_OR_BORROWED

	if _is_diatonic_chord(chord_name, detected_key):
		return HARMONIC_PLAUSIBILITY_DIATONIC

	root_to_pc = {
		"C": 0,
		"C#": 1,
		"D": 2,
		"D#": 3,
		"E": 4,
		"F": 5,
		"F#": 6,
		"G": 7,
		"G#": 8,
		"A": 9,
		"A#": 10,
		"B": 11,
	}

	root_name, quality = _parse_chord_quality(chord_name)
	root_pc = root_to_pc.get(root_name)
	if root_pc is None:
		return HARMONIC_PLAUSIBILITY_NON_DIATONIC

	interval = (root_pc - detected_key.tonic_pc) % 12
	if detected_key.mode == "minor" and interval == 7 and quality == "major":
		# Harmonic-minor dominant major quality (e.g. C# in F#m) is musically plausible.
		return HARMONIC_PLAUSIBILITY_DOMINANT_MAJOR_IN_MINOR

	return HARMONIC_PLAUSIBILITY_NON_DIATONIC
