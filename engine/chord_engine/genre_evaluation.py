"""Multi-genre ground-truth evaluation corpus utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from chord_engine.analyze import AnalysisError
from chord_engine.evaluation import EvaluationError, evaluate_against_ground_truth
from chord_engine.numeric import _safe_pct


@dataclass(frozen=True)
class GenreEvalItem:
	id: str
	genre: str
	audio_path: Path
	annotation_path: Path
	clip_start: str | float | None
	clip_end: str | float | None


class GenreEvaluationError(Exception):
	"""Controlled error for multi-genre evaluation failures."""

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


def evaluate_genre_corpus(manifest_path: str | Path) -> dict[str, object]:
	"""Evaluate a local annotated corpus and aggregate metrics by genre."""

	manifest_file = Path(manifest_path)
	manifest = _load_manifest(manifest_file)
	items = _parse_items(manifest, base_dir=manifest_file.parent)

	results = [_evaluate_item(item) for item in items]
	evaluated = [item for item in results if item["status"] == "pass"]
	failed = [item for item in results if item["status"] != "pass"]
	return {
		"version": "1",
		"manifestPath": str(manifest_path),
		"itemCount": len(results),
		"evaluatedItemCount": len(evaluated),
		"failedItemCount": len(failed),
		"status": "pass" if not failed else "fail",
		"metrics": _aggregate_metrics(evaluated),
		"genres": _aggregate_by_genre(evaluated),
		"items": results,
	}


def _evaluate_item(item: GenreEvalItem) -> dict[str, object]:
	try:
		payload = evaluate_against_ground_truth(
			item.audio_path,
			item.annotation_path,
			clip_start_override=item.clip_start,
			clip_end_override=item.clip_end,
		)
	except (AnalysisError, EvaluationError) as exc:
		code = exc.code if isinstance(exc, (AnalysisError, EvaluationError)) else "EVALUATION_FAILED"
		message = exc.message if isinstance(exc, (AnalysisError, EvaluationError)) else "Failed to evaluate item"
		return {
			"id": item.id,
			"genre": item.genre,
			"audioPath": str(item.audio_path),
			"annotationPath": str(item.annotation_path),
			"status": "error",
			"error": {
				"code": code,
				"message": message,
			},
		}

	return {
		"id": item.id,
		"genre": item.genre,
		"audioPath": str(item.audio_path),
		"annotationPath": str(item.annotation_path),
		"status": "pass",
		"clipStart": payload.get("clipStart"),
		"clipEnd": payload.get("clipEnd"),
		"metrics": payload.get("metrics", {}),
	}


def _aggregate_by_genre(items: list[dict[str, object]]) -> dict[str, object]:
	genres = sorted({str(item["genre"]) for item in items})
	return {
		genre: _aggregate_metrics([item for item in items if item["genre"] == genre])
		for genre in genres
	}


def _aggregate_metrics(items: list[dict[str, object]]) -> dict[str, object]:
	if not items:
		return {
			"itemCount": 0,
			"evaluatedDuration": 0.0,
			"timeWeightedChordAccuracy": 0.0,
			"rootAccuracy": 0.0,
			"qualityAccuracy": 0.0,
			"falseTransitionCount": 0,
			"missedTransitionCount": 0,
			"boundaryTimingErrorSeconds": None,
		}

	metric_rows = [_metrics(item) for item in items]
	total_duration = sum(_float(row.get("evaluatedDuration")) for row in metric_rows)
	total_quality_duration = sum(_float(row.get("evaluatedDuration")) for row in metric_rows)
	boundary_errors = [
		_float(row.get("boundaryTimingErrorSeconds"))
		for row in metric_rows
		if row.get("boundaryTimingErrorSeconds") is not None
	]
	gt_segment_total = sum(_float(row.get("groundTruthSegmentCount")) for row in metric_rows)
	return {
		"itemCount": len(items),
		"evaluatedDuration": round(total_duration, 3),
		"timeWeightedChordAccuracy": _round_pct(_weighted_pct(metric_rows, "timeWeightedChordAccuracy", total_duration)),
		"exactChordMatchPercentage": _round_pct(_weighted_pct(metric_rows, "exactChordMatchPercentage", gt_segment_total, "groundTruthSegmentCount")),
		"rootAccuracy": _round_pct(_weighted_pct(metric_rows, "rootAccuracy", total_duration)),
		"qualityAccuracy": _round_pct(_weighted_pct(metric_rows, "qualityAccuracy", total_quality_duration)),
		"falseTransitionCount": int(sum(_float(row.get("falseTransitionCount")) for row in metric_rows)),
		"missedTransitionCount": int(sum(_float(row.get("missedTransitionCount")) for row in metric_rows)),
		"groundTruthSegmentCount": int(gt_segment_total),
		"predictedSegmentCount": int(sum(_float(row.get("predictedSegmentCount")) for row in metric_rows)),
		"boundaryTimingErrorSeconds": None if not boundary_errors else round(mean(boundary_errors), 3),
	}


def _weighted_pct(
	rows: list[dict[str, object]],
	key: str,
	denominator: float,
	weight_key: str = "evaluatedDuration",
) -> float:
	if denominator <= 0:
		return 0.0
	matched = sum((_float(row.get(key)) / 100.0) * _float(row.get(weight_key)) for row in rows)
	return _safe_pct(matched, denominator)


def _round_pct(value: float) -> float:
	return round(value, 3)


def _metrics(item: dict[str, object]) -> dict[str, object]:
	metrics = item.get("metrics")
	return metrics if isinstance(metrics, dict) else {}


def _float(value: object) -> float:
	if isinstance(value, (int, float)):
		return float(value)
	return 0.0


def _load_manifest(path: Path) -> dict[str, Any]:
	if not path.exists() or not path.is_file():
		raise GenreEvaluationError("GENRE_EVAL_MANIFEST_NOT_FOUND", "Genre evaluation manifest file not found")
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except Exception as exc:
		raise GenreEvaluationError("GENRE_EVAL_MANIFEST_PARSE_FAILED", "Failed to parse genre evaluation manifest JSON") from exc
	if not isinstance(payload, dict):
		raise GenreEvaluationError("GENRE_EVAL_MANIFEST_INVALID", "Genre evaluation manifest must be a JSON object")
	return payload


def _parse_items(manifest: dict[str, Any], *, base_dir: Path) -> list[GenreEvalItem]:
	raw_items = manifest.get("items")
	if not isinstance(raw_items, list) or not raw_items:
		raise GenreEvaluationError("GENRE_EVAL_MANIFEST_INVALID", "Genre evaluation manifest requires non-empty items")

	items: list[GenreEvalItem] = []
	for raw_item in raw_items:
		if not isinstance(raw_item, dict):
			raise GenreEvaluationError("GENRE_EVAL_MANIFEST_INVALID", "Each genre evaluation item must be an object")
		item_id = str(raw_item.get("id", "")).strip()
		genre = str(raw_item.get("genre", "")).strip()
		audio_path = str(raw_item.get("audioPath", "")).strip()
		annotation_path = str(raw_item.get("annotationPath", "")).strip()
		if not item_id or not genre or not audio_path or not annotation_path:
			raise GenreEvaluationError("GENRE_EVAL_MANIFEST_INVALID", "Each item requires id, genre, audioPath, and annotationPath")
		items.append(
			GenreEvalItem(
				id=item_id,
				genre=genre,
				audio_path=_resolve_manifest_path(audio_path, base_dir=base_dir),
				annotation_path=_resolve_manifest_path(annotation_path, base_dir=base_dir),
				clip_start=raw_item.get("clipStart"),
				clip_end=raw_item.get("clipEnd"),
			)
		)
	return items


def _resolve_manifest_path(value: str, *, base_dir: Path) -> Path:
	expanded = Path(os.path.expandvars(os.path.expanduser(value)))
	if expanded.is_absolute():
		return expanded
	return base_dir / expanded
