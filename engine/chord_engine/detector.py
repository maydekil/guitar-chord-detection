"""Frame-level chord detector based on cosine similarity to chord templates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chord_engine.music_theory import _key_prior_bonus
from chord_engine.numeric import _cosine_similarity
from chord_engine.templates import generate_chord_templates

NO_CHORD_LABEL = "N"
FRAME_BINS = 12

# Centralized detector thresholds for MVP baseline.
MIN_FRAME_ENERGY = 1e-6
MIN_TEMPLATE_SIMILARITY = 0.38
MAJOR_SCALE_INTERVALS = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE_INTERVALS = (0, 2, 3, 5, 7, 8, 10)


@dataclass(frozen=True)
class KeyEstimate:
	"""Estimated global song key used as a soft prior for chord plausibility."""

	tonic_pc: int
	mode: str
	confidence: float

	@property
	def label(self) -> str:
		roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
		suffix = "m" if self.mode == "minor" else ""
		return f"{roots[self.tonic_pc]}{suffix}"


@dataclass(frozen=True)
class FrameChordPrediction:
	"""Frame-level output: best chord label and certainty-style confidence.

	Confidence is a deterministic match certainty score in [0, 1], derived from
	best cosine similarity and separation from the second-best score.
	"""

	chord: str
	confidence: float


class FrameDetectionError(Exception):
	"""Controlled error for invalid frame input."""

	def __init__(self, code: str, message: str) -> None:
		super().__init__(message)
		self.code = code
		self.message = message


def predict_frame_chord(frame: np.ndarray, *, key_estimate: KeyEstimate | None = None) -> FrameChordPrediction:
	"""Predict a chord label for one chroma frame (12 bins)."""

	frame_vec = _validate_frame(frame)
	energy = float(np.linalg.norm(frame_vec, ord=2))

	if energy < MIN_FRAME_ENERGY:
		return FrameChordPrediction(chord=NO_CHORD_LABEL, confidence=1.0)

	templates = generate_chord_templates()
	scored = _score_chord_candidates(frame_vec, templates, key_estimate=key_estimate)

	# Deterministic ordering: highest score first, then lexical chord name for ties.
	scored.sort(key=lambda item: (-item[0], item[1]))
	best_score, best_chord = scored[0]
	second_score = scored[1][0] if len(scored) > 1 else -1.0

	if best_score < MIN_TEMPLATE_SIMILARITY:
		return FrameChordPrediction(chord=NO_CHORD_LABEL, confidence=_no_chord_confidence(best_score))

	confidence = _match_confidence(best_score, second_score)
	return FrameChordPrediction(chord=best_chord, confidence=confidence)


def estimate_global_key(chroma: np.ndarray) -> KeyEstimate | None:
	"""Estimate global key from aggregate chroma as a soft-prior context."""

	arr = np.asarray(chroma, dtype=np.float32)
	if arr.ndim != 2 or arr.shape[0] != FRAME_BINS or arr.shape[1] == 0:
		return None

	agg = np.sum(arr, axis=1)
	energy = float(np.linalg.norm(agg, ord=2))
	if not np.isfinite(energy) or energy < MIN_FRAME_ENERGY:
		return None

	scored: list[tuple[float, int, str]] = []
	for tonic in range(FRAME_BINS):
		major_profile = _scale_membership_profile(tonic, "major")
		minor_profile = _scale_membership_profile(tonic, "minor")
		scored.append((_cosine_similarity(agg, major_profile, min_norm=MIN_FRAME_ENERGY), tonic, "major"))
		scored.append((_cosine_similarity(agg, minor_profile, min_norm=MIN_FRAME_ENERGY), tonic, "minor"))

	scored.sort(key=lambda item: (-item[0], item[1], item[2]))
	best_score, tonic_pc, mode = scored[0]
	second_score = scored[1][0] if len(scored) > 1 else -1.0
	confidence = _match_confidence(best_score, second_score)
	return KeyEstimate(tonic_pc=tonic_pc, mode=mode, confidence=confidence)


def _validate_frame(frame: np.ndarray) -> np.ndarray:
	vec = np.asarray(frame, dtype=np.float32)

	if vec.size == 0:
		raise FrameDetectionError("FRAME_EMPTY", "Frame is empty")

	if vec.ndim != 1:
		raise FrameDetectionError("FRAME_INVALID_SHAPE", "Frame must be a 1D vector")

	if vec.shape[0] != FRAME_BINS:
		raise FrameDetectionError("FRAME_INVALID_SHAPE", "Frame must contain exactly 12 bins")

	if not np.isfinite(vec).all():
		raise FrameDetectionError("FRAME_INVALID_VALUES", "Frame contains NaN or infinite values")

	return vec


def _score_chord_candidates(
	frame_vec: np.ndarray,
	templates: dict[str, np.ndarray],
	*,
	key_estimate: KeyEstimate | None,
) -> list[tuple[float, str]]:
	scored: list[tuple[float, str]] = []
	for name, template in templates.items():
		raw = _cosine_similarity(frame_vec, template, min_norm=MIN_FRAME_ENERGY)
		bonus = _key_prior_bonus(name, key_estimate) if key_estimate else 0.0
		scored.append((raw + bonus, name))
	return scored


def _scale_membership_profile(tonic_pc: int, mode: str) -> np.ndarray:
	intervals = MAJOR_SCALE_INTERVALS if mode == "major" else MINOR_SCALE_INTERVALS
	profile = np.zeros(FRAME_BINS, dtype=np.float32)
	for interval in intervals:
		profile[(tonic_pc + interval) % FRAME_BINS] = 1.0
	return profile


def _match_confidence(best_score: float, second_score: float) -> float:
	best_scaled = (best_score + 1.0) / 2.0
	margin = max(0.0, best_score - second_score)
	confidence = 0.8 * best_scaled + 0.2 * margin
	return float(np.clip(confidence, 0.0, 1.0))


def _no_chord_confidence(best_score: float) -> float:
	# Lower template affinity implies stronger no-chord certainty.
	confidence = 1.0 - max(0.0, best_score)
	return float(np.clip(confidence, 0.0, 1.0))
