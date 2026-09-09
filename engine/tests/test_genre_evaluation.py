from __future__ import annotations

import json
from pathlib import Path

import pytest

import chord_engine.genre_evaluation as genre_evaluation
from chord_engine.evaluation import EvaluationError


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
	path.write_text(json.dumps(payload), encoding="utf-8")


def _metric_payload(duration: float, exact: float, root: float, quality: float, false_count: int, missed_count: int, gt_count: int, pred_count: int) -> dict[str, object]:
	return {
		"version": "1",
		"clipStart": 0.0,
		"clipEnd": duration,
		"metrics": {
			"evaluatedDuration": duration,
			"timeWeightedChordAccuracy": exact,
			"exactChordMatchPercentage": exact,
			"rootAccuracy": root,
			"qualityAccuracy": quality,
			"falseTransitionCount": false_count,
			"missedTransitionCount": missed_count,
			"boundaryTimingErrorSeconds": 0.2,
			"groundTruthSegmentCount": gt_count,
			"predictedSegmentCount": pred_count,
		},
	}


def test_evaluate_genre_corpus_aggregates_metrics_by_genre(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	manifest = tmp_path / "genre-corpus.json"
	_write_manifest(
		manifest,
		{
			"items": [
				{
					"id": "pop-one",
					"genre": "pop",
					"audioPath": "pop.wav",
					"annotationPath": "pop.gt.json",
				},
				{
					"id": "rock-one",
					"genre": "rock",
					"audioPath": "rock.wav",
					"annotationPath": "rock.gt.json",
				},
			]
		},
	)

	def fake_evaluate(audio_path: str | Path, annotation_path: str | Path, *, clip_start_override: object, clip_end_override: object) -> dict[str, object]:
		assert Path(audio_path).parent == tmp_path
		assert Path(annotation_path).parent == tmp_path
		if Path(audio_path).name == "pop.wav":
			return _metric_payload(10.0, 80.0, 90.0, 70.0, 1, 2, 5, 6)
		return _metric_payload(30.0, 60.0, 70.0, 50.0, 3, 0, 10, 11)

	monkeypatch.setattr(genre_evaluation, "evaluate_against_ground_truth", fake_evaluate)

	result = genre_evaluation.evaluate_genre_corpus(manifest)

	assert result["status"] == "pass"
	assert result["itemCount"] == 2
	assert result["evaluatedItemCount"] == 2
	assert result["failedItemCount"] == 0
	metrics = result["metrics"]
	assert metrics["evaluatedDuration"] == pytest.approx(40.0)
	assert metrics["timeWeightedChordAccuracy"] == pytest.approx(65.0)
	assert metrics["exactChordMatchPercentage"] == pytest.approx(66.667)
	assert metrics["rootAccuracy"] == pytest.approx(75.0)
	assert metrics["qualityAccuracy"] == pytest.approx(55.0)
	assert metrics["falseTransitionCount"] == 4
	assert metrics["missedTransitionCount"] == 2
	assert metrics["groundTruthSegmentCount"] == 15
	assert metrics["predictedSegmentCount"] == 17
	assert sorted(result["genres"].keys()) == ["pop", "rock"]
	assert result["genres"]["pop"]["timeWeightedChordAccuracy"] == pytest.approx(80.0)


def test_evaluate_genre_corpus_reports_item_errors_without_hiding_other_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	manifest = tmp_path / "genre-corpus.json"
	_write_manifest(
		manifest,
		{
			"items": [
				{"id": "ok", "genre": "folk", "audioPath": "ok.wav", "annotationPath": "ok.json"},
				{"id": "bad", "genre": "ska", "audioPath": "bad.wav", "annotationPath": "bad.json"},
			]
		},
	)

	def fake_evaluate(audio_path: str | Path, _annotation_path: str | Path, *, clip_start_override: object, clip_end_override: object) -> dict[str, object]:
		if Path(audio_path).name == "bad.wav":
			raise EvaluationError("GROUND_TRUTH_FILE_NOT_FOUND", "Ground-truth annotation file not found")
		return _metric_payload(4.0, 100.0, 100.0, 100.0, 0, 0, 2, 2)

	monkeypatch.setattr(genre_evaluation, "evaluate_against_ground_truth", fake_evaluate)

	result = genre_evaluation.evaluate_genre_corpus(manifest)

	assert result["status"] == "fail"
	assert result["evaluatedItemCount"] == 1
	assert result["failedItemCount"] == 1
	items = result["items"]
	assert items[0]["status"] == "pass"
	assert items[1]["status"] == "error"
	assert items[1]["error"]["code"] == "GROUND_TRUTH_FILE_NOT_FOUND"
	assert result["metrics"]["timeWeightedChordAccuracy"] == pytest.approx(100.0)


def test_evaluate_genre_corpus_rejects_invalid_manifest(tmp_path: Path) -> None:
	manifest = tmp_path / "invalid.json"
	_write_manifest(manifest, {"items": []})

	with pytest.raises(genre_evaluation.GenreEvaluationError) as exc:
		genre_evaluation.evaluate_genre_corpus(manifest)

	assert exc.value.code == "GENRE_EVAL_MANIFEST_INVALID"
