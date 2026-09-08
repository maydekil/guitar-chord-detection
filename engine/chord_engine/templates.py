"""Programmatic chord-template generation for triad chord classes."""

from __future__ import annotations

import numpy as np

from chord_engine.features import PITCH_CLASS_ORDER

MAJOR_INTERVALS: tuple[int, int, int] = (0, 4, 7)
MINOR_INTERVALS: tuple[int, int, int] = (0, 3, 7)

MAJOR_SUFFIX = ""
MINOR_SUFFIX = "m"


def _build_template(root_index: int, intervals: tuple[int, ...]) -> np.ndarray:
	template = np.zeros(len(PITCH_CLASS_ORDER), dtype=np.float32)
	for interval in intervals:
		template[(root_index + interval) % len(PITCH_CLASS_ORDER)] = 1.0
	return template


def generate_chord_templates() -> dict[str, np.ndarray]:
	"""Generate the public major/minor triad template vocabulary."""
	templates: dict[str, np.ndarray] = {}

	for root_index, root_name in enumerate(PITCH_CLASS_ORDER):
		major_name = f"{root_name}{MAJOR_SUFFIX}"
		templates[major_name] = _build_template(root_index, MAJOR_INTERVALS)

	for root_index, root_name in enumerate(PITCH_CLASS_ORDER):
		minor_name = f"{root_name}{MINOR_SUFFIX}"
		templates[minor_name] = _build_template(root_index, MINOR_INTERVALS)

	return templates
