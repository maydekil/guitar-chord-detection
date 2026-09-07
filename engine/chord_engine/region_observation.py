"""Shared construction for harmonic-region observations."""

from __future__ import annotations

import numpy as np

from chord_engine.analysis_config import GLOBAL_DECODE_EVIDENCE_MARGIN_WEIGHT
from chord_engine.analysis_models import MusicalRegionObservation
from chord_engine.detector import KeyEstimate
from chord_engine.root_aware import _estimate_region_root_aware_identity


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
