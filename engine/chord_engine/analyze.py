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
from chord_engine.root_aware import (
	_estimate_region_root_aware_identity,
	_normalize_nonnegative,
	_template_score_with_soft_key,
)
from chord_engine.post_refinement import (
	_apply_context_aware_short_segment_correction,
	_refine_long_holds_with_mini_harmonic_sequence_decoder,
	_refine_long_segments_with_harmonic_evidence,
	_refine_long_segments_with_sustained_harmonic_drift,
	_refine_low_confidence_long_holds_with_key_candidates,
	_suppress_weak_diminished_passing_segments,
	_suppress_weak_timeline_fragments,
)
from chord_engine.playable_progression import _apply_playable_progression_refinement
from chord_engine.music_theory import (
	_chord_root_pc,
	_diatonic_chord_labels_for_key,
	_diatonic_match_score,
	_harmonic_plausibility,
	_harmonic_relationship_strength,
	_is_diatonic_chord,
	_is_root_preserving_quality_switch,
	_key_prior_bonus,
	_parse_chord_quality,
	_root_name_from_pc,
)
from chord_engine.global_decoder import (
	_collapse_regions_by_decoded_chord,
	_decode_global_chord_sequence,
	_regions_to_segments,
)
from chord_engine.harmonic_boundaries import (
	_compute_harmonic_novelty,
	_consolidate_boundary_clusters,
	_select_harmonic_boundaries,
)
from chord_engine.harmonic_regions import _build_harmonic_regions_from_change_points
from chord_engine.features import (
	BeatTiming,
	FeatureExtractionError,
	estimate_beat_timing,
	extract_chroma,
	extract_harmonic_signal,
	extract_low_frequency_chroma,
)
from chord_engine.segment_utils import (
	_chord_duration_map,
	_clamp_final_segment_end,
	_merge_adjacent_same_chord_segments,
	_neighbor_support,
	_segment_duration,
	_weighted_confidence,
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
			segments, playable_events = _apply_playable_progression_refinement(
				segments,
				detected_key=detected_key,
			)
			correction_events.extend(playable_events)
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
