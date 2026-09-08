"""Root-aware chord identity scoring for harmonic regions."""

from __future__ import annotations

import numpy as np

from chord_engine.analysis_config import *  # noqa: F403 - shared root-aware scoring constants
from chord_engine.detector import KeyEstimate
from chord_engine.music_theory import (
	_chord_root_pc,
	_diatonic_match_score,
	_harmonic_plausibility,
	_is_diatonic_chord,
	_key_prior_bonus,
	_parse_chord_quality,
	_root_name_from_pc,
)
from chord_engine.numeric import _cosine_similarity
from chord_engine.templates import generate_chord_templates

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
			quality_component = _minor_quality_evidence_for_root(energy, root_pc)
		else:
			quality_component = _major_quality_evidence_for_root(energy, root_pc)
		label_quality_margin = _quality_margin_for_root(energy, root_pc)
		if label_quality_margin < ROOT_AWARE_QUALITY_AMBIGUOUS_MARGIN:
			quality_component *= 0.55
		key_context_adjust = 0.0
		if key_estimate is not None:
			key_context_adjust = ROOT_AWARE_KEY_CONTEXT_WEIGHT * (_harmonic_plausibility(label, key_estimate) - 0.5)
			strong_local_quality = bool(
				root_pc == selected_root_pc
				and label_quality_margin >= ROOT_AWARE_QUALITY_AMBIGUOUS_MARGIN
				and (
					(is_diminished_label and (diminished_quality_evidence > major_quality_evidence))
					or (is_minor_label and (_minor_quality_evidence_for_root(energy, root_pc) > _major_quality_evidence_for_root(energy, root_pc)))
					or ((not is_minor_label and not is_diminished_label) and (_major_quality_evidence_for_root(energy, root_pc) > _minor_quality_evidence_for_root(energy, root_pc)))
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


def _major_quality_evidence_for_root(energy: np.ndarray, root_pc: int) -> float:
	return float(
		ROOT_AWARE_QUALITY_THIRD_WEIGHT * energy[(root_pc + 4) % 12]
		+ ROOT_AWARE_QUALITY_SUPPORT_WEIGHT * ((energy[root_pc] + energy[(root_pc + 7) % 12]) / 2.0)
	)


def _minor_quality_evidence_for_root(energy: np.ndarray, root_pc: int) -> float:
	return float(
		ROOT_AWARE_QUALITY_THIRD_WEIGHT * energy[(root_pc + 3) % 12]
		+ ROOT_AWARE_QUALITY_SUPPORT_WEIGHT * ((energy[root_pc] + energy[(root_pc + 7) % 12]) / 2.0)
	)


def _quality_margin_for_root(energy: np.ndarray, root_pc: int) -> float:
	return abs(_major_quality_evidence_for_root(energy, root_pc) - _minor_quality_evidence_for_root(energy, root_pc))

def _template_score_with_soft_key(
	frame_vec: np.ndarray,
	template: np.ndarray,
	chord_name: str,
	key_estimate: KeyEstimate | None,
) -> float:
	raw = _cosine_similarity(frame_vec, template)
	bonus = _key_prior_bonus(chord_name, key_estimate) if key_estimate is not None else 0.0
	return float(raw + bonus)
