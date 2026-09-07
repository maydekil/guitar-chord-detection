"""Regression-style evaluation set for chord analysis quality tracking."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from chord_engine.analyze import AnalysisError, analyze_audio
from chord_engine.numeric import _safe_pct
from chord_engine.segmentation import ChordSegment


DEFAULT_MIN_AVERAGE_CONFIDENCE = 0.62
DEFAULT_MAX_LOW_CONFIDENCE_PERCENT = 35.0
DEFAULT_MAX_TRANSITIONS_PER_MINUTE = 42.0


@dataclass(frozen=True)
class EvalSample:
	name: str
	audio_path: Path
	expected_key: str | None
	allowed_chords: set[str]
	min_segments: int | None
	max_segments: int | None
	min_average_confidence: float
	max_low_confidence_percent: float
	max_transitions_per_minute: float


class RegressionEvaluationError(Exception):
	"""Controlled error for regression-set evaluation failures."""

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


def evaluate_regression_set(
	manifest_path: str | Path,
	*,
	baseline_path: str | Path | None = None,
) -> dict[str, object]:
	manifest = _load_manifest(manifest_path)
	samples = _parse_samples(manifest)
	baseline = _load_optional_baseline(baseline_path)

	results = []
	for sample in samples:
		results.append(_evaluate_sample(sample, baseline=baseline))

	failed_samples = [result for result in results if result["status"] != "pass"]
	return {
		"version": "1",
		"manifestPath": str(manifest_path),
		"sampleCount": len(results),
		"status": "pass" if not failed_samples else "fail",
		"failedSampleCount": len(failed_samples),
		"samples": results,
	}


def _evaluate_sample(sample: EvalSample, *, baseline: dict[str, Any] | None) -> dict[str, object]:
	if not sample.audio_path.exists() or not sample.audio_path.is_file():
		return {
			"name": sample.name,
			"audioPath": str(sample.audio_path),
			"status": "missing",
			"failures": ["audio file not found"],
		}

	try:
		analysis = analyze_audio(sample.audio_path)
	except AnalysisError as exc:
		return {
			"name": sample.name,
			"audioPath": str(sample.audio_path),
			"status": "error",
			"failures": [exc.message],
		}

	segments = analysis.analysis.chords
	duration = max(0.0, float(analysis.source.duration))
	segment_count = len(segments)
	transitions_per_minute = 0.0 if duration <= 0 else max(0, segment_count - 1) / (duration / 60.0)
	average_confidence = mean([segment.confidence for segment in segments]) if segments else 0.0
	low_confidence_count = sum(1 for segment in segments if segment.confidence < 0.55)
	low_confidence_percent = _safe_pct(float(low_confidence_count), float(segment_count))
	chord_durations = _chord_duration_histogram(segments)
	key_estimate = _estimate_major_key(chord_durations)
	unknown_chords = sorted(chord for chord in chord_durations if sample.allowed_chords and chord not in sample.allowed_chords)
	long_segments = [
		{
			"start": round(segment.start, 2),
			"end": round(segment.end, 2),
			"duration": round(segment.end - segment.start, 2),
			"chord": segment.chord,
			"confidence": round(segment.confidence, 3),
		}
		for segment in segments
		if segment.end - segment.start >= 8.0
	]

	failures: list[str] = []
	if sample.expected_key and key_estimate != sample.expected_key:
		failures.append(f"key estimate {key_estimate} != expected {sample.expected_key}")
	if sample.min_segments is not None and segment_count < sample.min_segments:
		failures.append(f"segment count {segment_count} < min {sample.min_segments}")
	if sample.max_segments is not None and segment_count > sample.max_segments:
		failures.append(f"segment count {segment_count} > max {sample.max_segments}")
	if average_confidence < sample.min_average_confidence:
		failures.append(f"average confidence {average_confidence:.3f} < min {sample.min_average_confidence:.3f}")
	if low_confidence_percent > sample.max_low_confidence_percent:
		failures.append(f"low confidence {low_confidence_percent:.1f}% > max {sample.max_low_confidence_percent:.1f}%")
	if transitions_per_minute > sample.max_transitions_per_minute:
		failures.append(f"transitions/min {transitions_per_minute:.1f} > max {sample.max_transitions_per_minute:.1f}")
	if unknown_chords:
		failures.append(f"outside allowed chord family: {', '.join(unknown_chords)}")

	result: dict[str, object] = {
		"name": sample.name,
		"audioPath": str(sample.audio_path),
		"status": "pass" if not failures else "fail",
		"failures": failures,
		"algorithm": analysis.analysis.algorithm,
		"duration": round(duration, 2),
		"segmentCount": segment_count,
		"transitionsPerMinute": round(transitions_per_minute, 2),
		"averageConfidence": round(average_confidence, 3),
		"lowConfidencePercent": round(low_confidence_percent, 2),
		"keyEstimate": key_estimate,
		"topChords": _top_chords(chord_durations),
		"longSegments": long_segments[:12],
	}
	baseline_sample = _find_baseline_sample(baseline, sample.name)
	if baseline_sample:
		result["baselineDelta"] = _baseline_delta(result, baseline_sample)
	return result


def _load_manifest(path: str | Path) -> dict[str, Any]:
	manifest_path = Path(path)
	if not manifest_path.exists() or not manifest_path.is_file():
		raise RegressionEvaluationError("EVAL_MANIFEST_NOT_FOUND", "Evaluation manifest file not found")
	try:
		payload = json.loads(manifest_path.read_text(encoding="utf-8"))
	except Exception as exc:
		raise RegressionEvaluationError("EVAL_MANIFEST_PARSE_FAILED", "Failed to parse evaluation manifest JSON") from exc
	if not isinstance(payload, dict):
		raise RegressionEvaluationError("EVAL_MANIFEST_INVALID", "Evaluation manifest must be a JSON object")
	return payload


def _parse_samples(manifest: dict[str, Any]) -> list[EvalSample]:
	raw_samples = manifest.get("samples")
	if not isinstance(raw_samples, list) or not raw_samples:
		raise RegressionEvaluationError("EVAL_MANIFEST_INVALID", "Evaluation manifest requires non-empty samples")

	samples: list[EvalSample] = []
	for raw_sample in raw_samples:
		if not isinstance(raw_sample, dict):
			raise RegressionEvaluationError("EVAL_MANIFEST_INVALID", "Each evaluation sample must be an object")
		name = str(raw_sample.get("name", "")).strip()
		audio_path = str(raw_sample.get("audioPath", "")).strip()
		if not name or not audio_path:
			raise RegressionEvaluationError("EVAL_MANIFEST_INVALID", "Each sample requires name and audioPath")
		allowed_chords = {
			str(chord).strip()
			for chord in raw_sample.get("allowedChords", [])
			if str(chord).strip()
		}
		samples.append(
			EvalSample(
				name=name,
				audio_path=_expand_path(audio_path),
				expected_key=_optional_str(raw_sample.get("expectedKey")),
				allowed_chords=allowed_chords,
				min_segments=_optional_int(raw_sample.get("minSegments")),
				max_segments=_optional_int(raw_sample.get("maxSegments")),
				min_average_confidence=_optional_float(raw_sample.get("minAverageConfidence"), DEFAULT_MIN_AVERAGE_CONFIDENCE),
				max_low_confidence_percent=_optional_float(raw_sample.get("maxLowConfidencePercent"), DEFAULT_MAX_LOW_CONFIDENCE_PERCENT),
				max_transitions_per_minute=_optional_float(raw_sample.get("maxTransitionsPerMinute"), DEFAULT_MAX_TRANSITIONS_PER_MINUTE),
			)
		)
	return samples


def _load_optional_baseline(path: str | Path | None) -> dict[str, Any] | None:
	if path is None:
		return None
	baseline_path = Path(path)
	if not baseline_path.exists() or not baseline_path.is_file():
		return None
	try:
		payload = json.loads(baseline_path.read_text(encoding="utf-8"))
	except Exception:
		return None
	return payload if isinstance(payload, dict) else None


def _expand_path(value: str) -> Path:
	return Path(os.path.expandvars(os.path.expanduser(value)))


def _optional_str(value: Any) -> str | None:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _optional_int(value: Any) -> int | None:
	if value is None:
		return None
	try:
		return int(value)
	except (TypeError, ValueError):
		return None


def _optional_float(value: Any, default: float) -> float:
	if value is None:
		return default
	try:
		parsed = float(value)
	except (TypeError, ValueError):
		return default
	return parsed if parsed == parsed else default


def _chord_duration_histogram(segments: list[ChordSegment]) -> dict[str, float]:
	histogram: dict[str, float] = defaultdict(float)
	for segment in segments:
		histogram[segment.chord] += max(0.0, segment.end - segment.start)
	return dict(histogram)


def _top_chords(chord_durations: dict[str, float]) -> list[dict[str, object]]:
	total = sum(chord_durations.values())
	return [
		{
			"chord": chord,
			"duration": round(duration, 2),
			"percentage": round(_safe_pct(duration, total), 2),
		}
		for chord, duration in sorted(chord_durations.items(), key=lambda item: (-item[1], item[0]))[:12]
	]


MAJOR_KEY_CHORDS = {
	"C": {"C", "Dm", "Em", "F", "G", "Am", "Bdim"},
	"C#": {"C#", "D#m", "Fm", "F#", "G#", "A#m", "Cdim"},
	"D": {"D", "Em", "F#m", "G", "A", "Bm", "C#dim"},
	"D#": {"D#", "Fm", "Gm", "G#", "A#", "Cm", "Ddim"},
	"E": {"E", "F#m", "G#m", "A", "B", "C#m", "D#dim"},
	"F": {"F", "Gm", "Am", "A#", "C", "Dm", "Edim"},
	"F#": {"F#", "G#m", "A#m", "B", "C#", "D#m", "Fdim"},
	"G": {"G", "Am", "Bm", "C", "D", "Em", "F#dim"},
	"G#": {"G#", "A#m", "Cm", "C#", "D#", "Fm", "Gdim"},
	"A": {"A", "Bm", "C#m", "D", "E", "F#m", "G#dim"},
	"A#": {"A#", "Cm", "Dm", "D#", "F", "Gm", "Adim"},
	"B": {"B", "C#m", "D#m", "E", "F#", "G#m", "A#dim"},
}


def _estimate_major_key(chord_durations: dict[str, float]) -> str:
	scores = Counter()
	for key, chords in MAJOR_KEY_CHORDS.items():
		scores[key] = sum(duration for chord, duration in chord_durations.items() if chord in chords)
	if not scores:
		return "unknown"
	return scores.most_common(1)[0][0]


def _find_baseline_sample(baseline: dict[str, Any] | None, sample_name: str) -> dict[str, Any] | None:
	if not baseline:
		return None
	raw_samples = baseline.get("samples")
	if not isinstance(raw_samples, list):
		return None
	for sample in raw_samples:
		if isinstance(sample, dict) and sample.get("name") == sample_name:
			return sample
	return None


def _baseline_delta(current: dict[str, object], baseline: dict[str, Any]) -> dict[str, object]:
	return {
		"segmentCount": _numeric_delta(current.get("segmentCount"), baseline.get("segmentCount")),
		"transitionsPerMinute": _numeric_delta(current.get("transitionsPerMinute"), baseline.get("transitionsPerMinute")),
		"averageConfidence": _numeric_delta(current.get("averageConfidence"), baseline.get("averageConfidence")),
		"lowConfidencePercent": _numeric_delta(current.get("lowConfidencePercent"), baseline.get("lowConfidencePercent")),
		"keyChanged": current.get("keyEstimate") != baseline.get("keyEstimate"),
	}


def _numeric_delta(current: object, baseline: object) -> float | None:
	if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
		return None
	return round(float(current) - float(baseline), 3)

