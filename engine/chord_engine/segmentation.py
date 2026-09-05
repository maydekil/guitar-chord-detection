"""Convert smoothed frame predictions into non-overlapping time segments."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from collections.abc import Sequence

import numpy as np

from chord_engine.audio import TARGET_SAMPLE_RATE
from chord_engine.detector import FrameChordPrediction
from chord_engine.features import DEFAULT_HOP_LENGTH

MIN_SEGMENT_DURATION_MS = 250


@dataclass(frozen=True)
class ChordSegment:
	"""Chord segment in seconds with aggregated confidence."""

	start: float
	end: float
	chord: str
	confidence: float


@dataclass
class _FrameSegment:
	start_frame: int
	end_frame: int  # exclusive
	chord: str
	confidence_sum: float

	@property
	def frame_count(self) -> int:
		return self.end_frame - self.start_frame


def frame_duration_seconds(
	*,
	hop_length: int = DEFAULT_HOP_LENGTH,
	sample_rate: int = TARGET_SAMPLE_RATE,
) -> float:
	if hop_length <= 0 or sample_rate <= 0:
		raise ValueError("hop_length and sample_rate must be positive")
	return float(hop_length / sample_rate)


def minimum_segment_frames(
	*,
	hop_length: int = DEFAULT_HOP_LENGTH,
	sample_rate: int = TARGET_SAMPLE_RATE,
	min_segment_duration_ms: int = MIN_SEGMENT_DURATION_MS,
) -> int:
	if min_segment_duration_ms <= 0:
		return 1
	frame_sec = frame_duration_seconds(hop_length=hop_length, sample_rate=sample_rate)
	min_frames = ceil((min_segment_duration_ms / 1000.0) / frame_sec)
	return max(1, int(min_frames))


def segment_frame_predictions(
	predictions: Sequence[FrameChordPrediction],
	*,
	hop_length: int = DEFAULT_HOP_LENGTH,
	sample_rate: int = TARGET_SAMPLE_RATE,
	min_segment_duration_ms: int = MIN_SEGMENT_DURATION_MS,
) -> list[ChordSegment]:
	"""Merge frame-level predictions into contiguous chord segments."""
	if len(predictions) == 0:
		return []

	frame_sec = frame_duration_seconds(hop_length=hop_length, sample_rate=sample_rate)
	min_frames = minimum_segment_frames(
		hop_length=hop_length,
		sample_rate=sample_rate,
		min_segment_duration_ms=min_segment_duration_ms,
	)

	segments = _build_initial_segments(predictions)
	segments = _merge_short_segments(segments, min_frames=min_frames)

	out: list[ChordSegment] = []
	for seg in segments:
		start = seg.start_frame * frame_sec
		end = seg.end_frame * frame_sec
		confidence = float(np.clip(seg.confidence_sum / seg.frame_count, 0.0, 1.0))
		out.append(ChordSegment(start=start, end=end, chord=seg.chord, confidence=confidence))

	return out


def _build_initial_segments(predictions: Sequence[FrameChordPrediction]) -> list[_FrameSegment]:
	segments: list[_FrameSegment] = []

	start = 0
	current = predictions[0].chord
	confidence_sum = float(np.clip(predictions[0].confidence, 0.0, 1.0))

	for idx in range(1, len(predictions)):
		pred = predictions[idx]
		conf = float(np.clip(pred.confidence, 0.0, 1.0))
		if pred.chord == current:
			confidence_sum += conf
			continue

		segments.append(
			_FrameSegment(
				start_frame=start,
				end_frame=idx,
				chord=current,
				confidence_sum=confidence_sum,
			)
		)
		start = idx
		current = pred.chord
		confidence_sum = conf

	segments.append(
		_FrameSegment(
			start_frame=start,
			end_frame=len(predictions),
			chord=current,
			confidence_sum=confidence_sum,
		)
	)

	return segments


def _merge_short_segments(segments: list[_FrameSegment], *, min_frames: int) -> list[_FrameSegment]:
	if len(segments) <= 1:
		return segments

	result = [
		_FrameSegment(
			start_frame=s.start_frame,
			end_frame=s.end_frame,
			chord=s.chord,
			confidence_sum=s.confidence_sum,
		)
		for s in segments
	]

	changed = True
	while changed:
		changed = False
		for idx, seg in enumerate(result):
			if seg.frame_count >= min_frames:
				continue
			if len(result) == 1:
				continue

			left_idx = idx - 1 if idx > 0 else None
			right_idx = idx + 1 if idx + 1 < len(result) else None

			if left_idx is None and right_idx is not None:
				_merge_into_right(result, idx, right_idx)
				changed = True
				break

			if right_idx is None and left_idx is not None:
				_merge_into_left(result, left_idx, idx)
				changed = True
				break

			if left_idx is None or right_idx is None:
				continue

			left = result[left_idx]
			right = result[right_idx]

			if left.chord == right.chord:
				_merge_three_same_chord(result, left_idx, idx, right_idx)
				changed = True
				break

			target = _pick_neighbor_target(left, right)
			if target == "left":
				_merge_into_left(result, left_idx, idx)
			else:
				_merge_into_right(result, idx, right_idx)
			changed = True
			break

	return result


def _pick_neighbor_target(left: _FrameSegment, right: _FrameSegment) -> str:
	left_score = left.confidence_sum
	right_score = right.confidence_sum
	if left_score > right_score:
		return "left"
	if right_score > left_score:
		return "right"

	if left.frame_count > right.frame_count:
		return "left"
	if right.frame_count > left.frame_count:
		return "right"

	if left.chord <= right.chord:
		return "left"
	return "right"


def _merge_into_left(items: list[_FrameSegment], left_idx: int, short_idx: int) -> None:
	left = items[left_idx]
	short = items[short_idx]
	left.end_frame = short.end_frame
	left.confidence_sum += short.confidence_sum
	del items[short_idx]


def _merge_into_right(items: list[_FrameSegment], short_idx: int, right_idx: int) -> None:
	short = items[short_idx]
	right = items[right_idx]
	right.start_frame = short.start_frame
	right.confidence_sum += short.confidence_sum
	del items[short_idx]


def _merge_three_same_chord(items: list[_FrameSegment], left_idx: int, mid_idx: int, right_idx: int) -> None:
	left = items[left_idx]
	mid = items[mid_idx]
	right = items[right_idx]
	left.end_frame = right.end_frame
	left.confidence_sum += mid.confidence_sum + right.confidence_sum
	del items[mid_idx:right_idx + 1]
