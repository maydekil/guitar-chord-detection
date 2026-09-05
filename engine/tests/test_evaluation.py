from __future__ import annotations

import json
from pathlib import Path

import pytest

import chord_engine.evaluation as evaluation_module
from chord_engine.analyze import AnalysisMetadata, AnalysisResult, SourceMetadata
from chord_engine.segmentation import ChordSegment


def _write_annotation(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_ground_truth_supports_partial_song_time_strings(tmp_path: Path) -> None:
    annotation = tmp_path / "partial.json"
    _write_annotation(
        annotation,
        {
            "clipStart": "00:30",
            "clipEnd": "01:00",
            "segments": [
                {"start": "00:30", "end": "00:40", "chord": "E"},
                {"start": "00:40", "end": "01:00", "chord": "A"},
            ],
        },
    )

    parsed = evaluation_module.load_ground_truth_annotation(annotation)

    assert parsed.clip_start == pytest.approx(30.0, abs=1e-10)
    assert parsed.clip_end == pytest.approx(60.0, abs=1e-10)
    assert len(parsed.segments) == 2
    assert parsed.segments[0].chord == "E"
    assert parsed.segments[1].chord == "A"


def test_evaluate_chord_segments_computes_deterministic_metrics() -> None:
    gt = [
        evaluation_module.AnnotationSegment(start=0.0, end=2.0, chord="C"),
        evaluation_module.AnnotationSegment(start=2.0, end=4.0, chord="G"),
        evaluation_module.AnnotationSegment(start=4.0, end=6.0, chord="Am"),
    ]
    pred = [
        ChordSegment(start=0.0, end=2.0, chord="C", confidence=0.9),
        ChordSegment(start=2.0, end=3.5, chord="G", confidence=0.8),
        ChordSegment(start=3.5, end=5.0, chord="A", confidence=0.7),
        ChordSegment(start=5.0, end=6.0, chord="Am", confidence=0.8),
    ]

    metrics = evaluation_module.evaluate_chord_segments(pred, gt, clip_start=0.0, clip_end=6.0)

    assert float(metrics["timeWeightedChordAccuracy"]) == pytest.approx(75.0, abs=1e-10)
    assert float(metrics["exactChordMatchPercentage"]) == pytest.approx(66.6666666667, abs=1e-8)
    assert float(metrics["rootAccuracy"]) == pytest.approx(91.6666666667, abs=1e-8)
    assert float(metrics["qualityAccuracy"]) == pytest.approx(83.3333333333, abs=1e-8)
    assert int(metrics["falseTransitionCount"]) == 2
    assert int(metrics["missedTransitionCount"]) == 1
    assert float(metrics["boundaryTimingErrorSeconds"]) == pytest.approx(0.0, abs=1e-10)

    confusion = metrics["confusionPairs"]
    assert confusion[0]["pair"] == "Am->A"
    assert confusion[1]["pair"] == "G->A"


def test_evaluate_against_ground_truth_with_monkeypatched_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    annotation = tmp_path / "gt.json"
    _write_annotation(
        annotation,
        {
            "segments": [
                {"start": 0.0, "end": 1.0, "chord": "E"},
                {"start": 1.0, "end": 2.0, "chord": "Em"},
            ]
        },
    )

    def fake_analyze(_path: str | Path) -> AnalysisResult:
        return AnalysisResult(
            version="1",
            source=SourceMetadata(path=str(_path), duration=2.0, sampleRate=22050),
            analysis=AnalysisMetadata(
                algorithm="chroma-template-v1",
                chords=[
                    ChordSegment(start=0.0, end=1.0, chord="E", confidence=0.8),
                    ChordSegment(start=1.0, end=2.0, chord="Em", confidence=0.8),
                ],
            ),
        )

    monkeypatch.setattr(evaluation_module, "analyze_audio", fake_analyze)

    result = evaluation_module.evaluate_against_ground_truth("/tmp/audio.wav", annotation)

    assert result["version"] == "1"
    metrics = result["metrics"]
    assert float(metrics["timeWeightedChordAccuracy"]) == pytest.approx(100.0, abs=1e-10)
    assert float(metrics["rootAccuracy"]) == pytest.approx(100.0, abs=1e-10)
    assert float(metrics["qualityAccuracy"]) == pytest.approx(100.0, abs=1e-10)
    assert int(metrics["falseTransitionCount"]) == 0
    assert int(metrics["missedTransitionCount"]) == 0
