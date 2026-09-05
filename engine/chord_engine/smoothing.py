"""Temporal smoothing for frame-level chord predictions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from chord_engine.detector import FrameChordPrediction
from chord_engine.features import DEFAULT_HOP_LENGTH
from chord_engine.audio import TARGET_SAMPLE_RATE

DEFAULT_SMOOTHING_MS = 500


def smoothing_window_frames(
	*,
	hop_length: int = DEFAULT_HOP_LENGTH,
	sample_rate: int = TARGET_SAMPLE_RATE,
	duration_ms: int = DEFAULT_SMOOTHING_MS,
) -> int:
	"""Compute an odd-sized smoothing window from timing configuration."""
	if hop_length <= 0 or sample_rate <= 0 or duration_ms <= 0:
		return 1

	frame_ms = 1000.0 * hop_length / sample_rate
	window = max(1, int(round(duration_ms / frame_ms)))
	if window % 2 == 0:
		window += 1
	return window


def smooth_frame_predictions(
	predictions: Sequence[FrameChordPrediction],
	*,
	window_frames: int | None = None,
) -> list[FrameChordPrediction]:
	"""Apply deterministic centered majority-vote smoothing.

	Tie-breaking priority:
	1. Keep center label if it is among tied winners.
	2. Otherwise pick tied label with highest aggregate confidence in the window.
	3. Otherwise lexicographically smallest label.
	"""
	if len(predictions) == 0:
		return []

	window = window_frames if window_frames is not None else smoothing_window_frames()
	window = max(1, int(window))
	if window % 2 == 0:
		window += 1

	half = window // 2
	out: list[FrameChordPrediction] = []

	for idx, current in enumerate(predictions):
		left = max(0, idx - half)
		right = min(len(predictions), idx + half + 1)
		neighborhood = predictions[left:right]

		winner = _select_winner_label(neighborhood, center_label=current.chord)
		confidence = _aggregate_winner_confidence(neighborhood, winner)
		out.append(FrameChordPrediction(chord=winner, confidence=confidence))

	return out


def _select_winner_label(neighborhood: Sequence[FrameChordPrediction], center_label: str) -> str:
	counts = Counter(item.chord for item in neighborhood)
	max_count = max(counts.values())
	tied = [label for label, count in counts.items() if count == max_count]

	if center_label in tied:
		return center_label

	confidence_sum: dict[str, float] = {}
	for label in tied:
		confidence_sum[label] = float(
			sum(pred.confidence for pred in neighborhood if pred.chord == label)
		)

	best_conf = max(confidence_sum.values())
	tied_by_conf = [label for label, val in confidence_sum.items() if np.isclose(val, best_conf)]
	return sorted(tied_by_conf)[0]


def _aggregate_winner_confidence(neighborhood: Sequence[FrameChordPrediction], winner: str) -> float:
	vals = [pred.confidence for pred in neighborhood if pred.chord == winner]
	if not vals:
		return 0.0
	avg = float(np.mean(vals))
	return float(np.clip(avg, 0.0, 1.0))
