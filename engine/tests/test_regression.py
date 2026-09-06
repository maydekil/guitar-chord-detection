from __future__ import annotations

import json
from pathlib import Path

import pytest

import chord_engine.regression as regression_module
from chord_engine.analyze import AnalysisMetadata, AnalysisResult, SourceMetadata
from chord_engine.segmentation import ChordSegment


def test_evaluate_regression_set_reports_quality_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	audio = tmp_path / "song.wav"
	audio.write_bytes(b"fake audio")
	manifest = tmp_path / "eval.json"
	manifest.write_text(
		json.dumps(
			{
				"samples": [
					{
						"name": "Song A",
						"audioPath": str(audio),
						"expectedKey": "A",
						"allowedChords": ["A", "Bm", "C#m", "D", "E", "F#m"],
						"minSegments": 2,
						"maxSegments": 8,
					}
				]
			}
		),
		encoding="utf-8",
	)

	def fake_analyze(_path: str | Path) -> AnalysisResult:
		return AnalysisResult(
			version="1",
			source=SourceMetadata(path=str(audio), duration=12.0, sampleRate=22050),
			analysis=AnalysisMetadata(
				algorithm="test-engine",
				chords=[
					ChordSegment(start=0.0, end=4.0, chord="A", confidence=0.8),
					ChordSegment(start=4.0, end=8.0, chord="D", confidence=0.75),
					ChordSegment(start=8.0, end=12.0, chord="E", confidence=0.7),
				],
			),
		)

	monkeypatch.setattr(regression_module, "analyze_audio", fake_analyze)

	result = regression_module.evaluate_regression_set(manifest)

	assert result["status"] == "pass"
	assert result["sampleCount"] == 1
	sample = result["samples"][0]
	assert sample["keyEstimate"] == "A"
	assert sample["segmentCount"] == 3
	assert sample["averageConfidence"] == pytest.approx(0.75)


def test_evaluate_regression_set_flags_missing_audio(tmp_path: Path) -> None:
	manifest = tmp_path / "eval.json"
	manifest.write_text(
		json.dumps({"samples": [{"name": "Missing Song", "audioPath": str(tmp_path / "missing.wav")}]}),
		encoding="utf-8",
	)

	result = regression_module.evaluate_regression_set(manifest)

	assert result["status"] == "fail"
	assert result["failedSampleCount"] == 1
	assert result["samples"][0]["status"] == "missing"
