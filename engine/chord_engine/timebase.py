"""Shared timebase helpers for frame-indexed analysis."""

from __future__ import annotations

from chord_engine.audio import TARGET_SAMPLE_RATE

DEFAULT_HOP_LENGTH = 512


def frame_duration_seconds(
	*,
	hop_length: int = DEFAULT_HOP_LENGTH,
	sample_rate: int = TARGET_SAMPLE_RATE,
) -> float:
	if hop_length <= 0 or sample_rate <= 0:
		raise ValueError("hop_length and sample_rate must be positive")
	return float(hop_length / sample_rate)
