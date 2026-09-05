"""Ground-truth chord evaluation utilities and deterministic metrics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean

from chord_engine.analyze import AnalysisError, analyze_audio
from chord_engine.segmentation import ChordSegment

TRANSITION_MATCH_TOLERANCE_SECONDS = 0.35


@dataclass(frozen=True)
class AnnotationSegment:
	start: float
	end: float
	chord: str


@dataclass(frozen=True)
class GroundTruthAnnotation:
	segments: list[AnnotationSegment]
	clip_start: float
	clip_end: float


class EvaluationError(Exception):
	"""Controlled error for ground-truth evaluation failures."""

	def __init__(self, code: str, message: str) -> None:
		super().__init__(message)
		self.code = code
		self.message = message

	def to_dict(self, *, version: str = "1") -> dict[str, object]:
		return {
			"version": version,
			"error": {
				"code": self.code,
				"message": self.message,
			},
		}


def evaluate_against_ground_truth(
	audio_path: str | Path,
	annotation_path: str | Path,
	*,
	clip_start_override: str | float | None = None,
	clip_end_override: str | float | None = None,
) -> dict[str, object]:
	"""Run analysis and compare predicted chords against manual annotation."""

	annotation = load_ground_truth_annotation(
		annotation_path,
		clip_start_override=clip_start_override,
		clip_end_override=clip_end_override,
	)
	analysis = analyze_audio(audio_path)
	metrics = evaluate_chord_segments(
		analysis.analysis.chords,
		annotation.segments,
		clip_start=annotation.clip_start,
		clip_end=annotation.clip_end,
	)
	return {
		"version": analysis.version,
		"audioPath": str(audio_path),
		"annotationPath": str(annotation_path),
		"clipStart": annotation.clip_start,
		"clipEnd": annotation.clip_end,
		"metrics": metrics,
	}


def load_ground_truth_annotation(
	annotation_path: str | Path,
	*,
	clip_start_override: str | float | None = None,
	clip_end_override: str | float | None = None,
) -> GroundTruthAnnotation:
	path = Path(annotation_path)
	if not path.exists() or not path.is_file():
		raise EvaluationError("GROUND_TRUTH_FILE_NOT_FOUND", "Ground-truth annotation file not found")

	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except Exception as exc:
		raise EvaluationError("GROUND_TRUTH_PARSE_FAILED", "Failed to parse ground-truth annotation JSON") from exc

	if not isinstance(payload, dict):
		raise EvaluationError("GROUND_TRUTH_INVALID", "Ground-truth annotation must be a JSON object")

	raw_segments = payload.get("segments")
	if not isinstance(raw_segments, list) or len(raw_segments) == 0:
		raise EvaluationError("GROUND_TRUTH_INVALID", "Ground-truth annotation requires non-empty segments")

	segments: list[AnnotationSegment] = []
	for item in raw_segments:
		if not isinstance(item, dict):
			raise EvaluationError("GROUND_TRUTH_INVALID", "Each annotation segment must be an object")
		start = _parse_time_value(item.get("start"))
		end = _parse_time_value(item.get("end"))
		chord = str(item.get("chord", "")).strip()
		if chord == "":
			raise EvaluationError("GROUND_TRUTH_INVALID", "Each annotation segment must include chord")
		if end <= start:
			raise EvaluationError("GROUND_TRUTH_INVALID", "Ground-truth segment end must be greater than start")
		segments.append(AnnotationSegment(start=start, end=end, chord=chord))

	segments.sort(key=lambda seg: (seg.start, seg.end, seg.chord))
	for prev, curr in zip(segments, segments[1:], strict=False):
		if curr.start < prev.end - 1e-9:
			raise EvaluationError("GROUND_TRUTH_INVALID", "Ground-truth segments must be non-overlapping")

	default_clip_start = segments[0].start
	default_clip_end = segments[-1].end
	clip_start = _parse_time_value(clip_start_override) if clip_start_override is not None else _parse_time_value(payload.get("clipStart", default_clip_start))
	clip_end = _parse_time_value(clip_end_override) if clip_end_override is not None else _parse_time_value(payload.get("clipEnd", default_clip_end))

	if clip_end <= clip_start:
		raise EvaluationError("GROUND_TRUTH_INVALID", "clipEnd must be greater than clipStart")

	filtered = [
		AnnotationSegment(start=max(seg.start, clip_start), end=min(seg.end, clip_end), chord=seg.chord)
		for seg in segments
		if seg.end > clip_start and seg.start < clip_end
	]
	filtered = [seg for seg in filtered if seg.end > seg.start]
	if len(filtered) == 0:
		raise EvaluationError("GROUND_TRUTH_INVALID", "No annotation segments overlap requested clip range")

	return GroundTruthAnnotation(
		segments=filtered,
		clip_start=clip_start,
		clip_end=clip_end,
	)


def evaluate_chord_segments(
	predicted: list[ChordSegment],
	ground_truth: list[AnnotationSegment],
	*,
	clip_start: float,
	clip_end: float,
) -> dict[str, object]:
	"""Compute deterministic chord-agreement metrics on a timeline interval."""

	if clip_end <= clip_start:
		raise EvaluationError("EVALUATION_INVALID_RANGE", "clip_end must be greater than clip_start")

	gt = _clip_annotation_segments(ground_truth, clip_start=clip_start, clip_end=clip_end)
	pred = _clip_predicted_segments(predicted, clip_start=clip_start, clip_end=clip_end)
	if len(gt) == 0:
		raise EvaluationError("EVALUATION_EMPTY_GROUND_TRUTH", "No ground-truth coverage in selected clip")

	boundaries = sorted(
		{
			clip_start,
			clip_end,
			*[seg.start for seg in gt],
			*[seg.end for seg in gt],
			*[seg.start for seg in pred],
			*[seg.end for seg in pred],
		}
	)

	total_duration = clip_end - clip_start
	exact_match_duration = 0.0
	root_match_duration = 0.0
	quality_match_duration = 0.0
	quality_duration_total = 0.0
	mismatch_duration_total = 0.0
	confusion_duration: dict[str, float] = {}

	for left, right in zip(boundaries[:-1], boundaries[1:], strict=False):
		if right <= left:
			continue
		mid = (left + right) / 2.0
		gt_label = _label_at_time(gt, mid, default="N")
		pred_label = _label_at_time(pred, mid, default="N")
		duration = right - left

		if pred_label == gt_label:
			exact_match_duration += duration
		else:
			mismatch_duration_total += duration
			key = f"{gt_label}->{pred_label}"
			confusion_duration[key] = confusion_duration.get(key, 0.0) + duration

		gt_root, gt_quality = _parse_root_quality(gt_label)
		pred_root, pred_quality = _parse_root_quality(pred_label)
		if gt_root == pred_root:
			root_match_duration += duration

		if gt_quality is not None:
			quality_duration_total += duration
			if pred_quality == gt_quality:
				quality_match_duration += duration

	exact_segment_matches = 0
	for gt_seg in gt:
		pred_label = _dominant_overlap_label(pred, gt_seg.start, gt_seg.end, default="N")
		if pred_label == gt_seg.chord:
			exact_segment_matches += 1

	gt_transitions = _transition_times_from_annotation(gt)
	pred_transitions = _transition_times_from_predicted(pred)
	false_transition_count, missed_transition_count, boundary_error = _transition_alignment(
		pred_transitions,
		gt_transitions,
		tolerance_seconds=TRANSITION_MATCH_TOLERANCE_SECONDS,
	)

	confusion_pairs = [
		{
			"pair": pair,
			"duration": duration,
			"percentageOfMismatchedTime": 0.0 if mismatch_duration_total <= 0 else (duration / mismatch_duration_total) * 100.0,
		}
		for pair, duration in sorted(confusion_duration.items(), key=lambda item: (-item[1], item[0]))
	]

	return {
		"timeWeightedChordAccuracy": _safe_pct(exact_match_duration, total_duration),
		"exactChordMatchPercentage": _safe_pct(float(exact_segment_matches), float(len(gt))),
		"rootAccuracy": _safe_pct(root_match_duration, total_duration),
		"qualityAccuracy": _safe_pct(quality_match_duration, quality_duration_total),
		"falseTransitionCount": false_transition_count,
		"missedTransitionCount": missed_transition_count,
		"boundaryTimingErrorSeconds": boundary_error,
		"confusionPairs": confusion_pairs[:20],
		"evaluatedDuration": total_duration,
		"groundTruthSegmentCount": len(gt),
		"predictedSegmentCount": len(pred),
	}


def _parse_time_value(value: object) -> float:
	if isinstance(value, (int, float)):
		out = float(value)
		if out < 0:
			raise EvaluationError("GROUND_TRUTH_INVALID", "Time values must be non-negative")
		return out
	if isinstance(value, str):
		t = value.strip()
		if t == "":
			raise EvaluationError("GROUND_TRUTH_INVALID", "Time string cannot be empty")
		if ":" not in t:
			try:
				out = float(t)
			except ValueError as exc:
				raise EvaluationError("GROUND_TRUTH_INVALID", "Invalid time value") from exc
			if out < 0:
				raise EvaluationError("GROUND_TRUTH_INVALID", "Time values must be non-negative")
			return out
		parts = t.split(":")
		if len(parts) not in (2, 3):
			raise EvaluationError("GROUND_TRUTH_INVALID", "Time value must be SS, MM:SS, or HH:MM:SS")
		try:
			nums = [float(part) for part in parts]
		except ValueError as exc:
			raise EvaluationError("GROUND_TRUTH_INVALID", "Invalid time value") from exc
		if any(part < 0 for part in nums):
			raise EvaluationError("GROUND_TRUTH_INVALID", "Time values must be non-negative")
		if len(nums) == 2:
			mm, ss = nums
			return mm * 60.0 + ss
		hh, mm, ss = nums
		return hh * 3600.0 + mm * 60.0 + ss
	raise EvaluationError("GROUND_TRUTH_INVALID", "Invalid time value type")


def _clip_annotation_segments(segments: list[AnnotationSegment], *, clip_start: float, clip_end: float) -> list[AnnotationSegment]:
	clipped: list[AnnotationSegment] = []
	for seg in segments:
		start = max(seg.start, clip_start)
		end = min(seg.end, clip_end)
		if end > start:
			clipped.append(AnnotationSegment(start=start, end=end, chord=seg.chord))
	return clipped


def _clip_predicted_segments(segments: list[ChordSegment], *, clip_start: float, clip_end: float) -> list[AnnotationSegment]:
	clipped: list[AnnotationSegment] = []
	for seg in segments:
		start = max(seg.start, clip_start)
		end = min(seg.end, clip_end)
		if end > start:
			clipped.append(AnnotationSegment(start=start, end=end, chord=seg.chord))
	return clipped


def _label_at_time(segments: list[AnnotationSegment], t: float, *, default: str) -> str:
	for seg in segments:
		if seg.start <= t < seg.end:
			return seg.chord
	return default


def _dominant_overlap_label(segments: list[AnnotationSegment], start: float, end: float, *, default: str) -> str:
	overlaps: dict[str, float] = {}
	for seg in segments:
		left = max(start, seg.start)
		right = min(end, seg.end)
		if right > left:
			overlaps[seg.chord] = overlaps.get(seg.chord, 0.0) + (right - left)
	if not overlaps:
		return default
	return sorted(overlaps.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _transition_times_from_annotation(segments: list[AnnotationSegment]) -> list[float]:
	if len(segments) <= 1:
		return []
	return [segments[idx].end for idx in range(len(segments) - 1) if segments[idx].chord != segments[idx + 1].chord]


def _transition_times_from_predicted(segments: list[AnnotationSegment]) -> list[float]:
	if len(segments) <= 1:
		return []
	return [segments[idx].end for idx in range(len(segments) - 1) if segments[idx].chord != segments[idx + 1].chord]


def _transition_alignment(
	pred_times: list[float],
	gt_times: list[float],
	*,
	tolerance_seconds: float,
) -> tuple[int, int, float | None]:
	if len(pred_times) == 0 and len(gt_times) == 0:
		return 0, 0, None

	matched_pred: set[int] = set()
	matched_gt: set[int] = set()
	errors: list[float] = []

	for gt_idx, gt_time in enumerate(gt_times):
		best_idx = None
		best_error = None
		for pred_idx, pred_time in enumerate(pred_times):
			if pred_idx in matched_pred:
				continue
			err = abs(pred_time - gt_time)
			if err <= tolerance_seconds and (best_error is None or err < best_error):
				best_idx = pred_idx
				best_error = err
		if best_idx is not None and best_error is not None:
			matched_pred.add(best_idx)
			matched_gt.add(gt_idx)
			errors.append(best_error)

	false_count = len(pred_times) - len(matched_pred)
	missed_count = len(gt_times) - len(matched_gt)
	boundary_error = None if len(errors) == 0 else float(mean(errors))
	return false_count, missed_count, boundary_error


def _parse_root_quality(chord: str) -> tuple[str | None, str | None]:
	label = chord.strip()
	if label == "N":
		return None, None
	if label.endswith("m"):
		return label[:-1], "minor"
	return label, "major"


def _safe_pct(numerator: float, denominator: float) -> float:
	if denominator <= 0:
		return 0.0
	return float((numerator / denominator) * 100.0)
