from __future__ import annotations

import math

import pytest

from chord_engine.detector import FrameChordPrediction
from chord_engine.segmentation import (
    MIN_SEGMENT_DURATION_MS,
    frame_duration_seconds,
    minimum_segment_frames,
    segment_frame_predictions,
)


def _predictions(labels: list[str], confs: list[float] | None = None) -> list[FrameChordPrediction]:
    if confs is None:
        confs = [0.9] * len(labels)
    return [FrameChordPrediction(chord=ch, confidence=cf) for ch, cf in zip(labels, confs, strict=True)]


def _labels(segments) -> list[str]:
    return [seg.chord for seg in segments]


def test_basic_merge() -> None:
    inp = _predictions(["C", "C", "C", "G", "G", "Am", "Am"])
    out = segment_frame_predictions(inp, min_segment_duration_ms=1)
    assert _labels(out) == ["C", "G", "Am"]


def test_start_end_integrity() -> None:
    inp = _predictions(["C", "C", "G", "G", "Am", "Am"])
    out = segment_frame_predictions(inp, min_segment_duration_ms=1)

    for seg in out:
        assert seg.start >= 0.0
        assert seg.end > seg.start


def test_ordering_and_no_overlap_and_contiguous_boundaries() -> None:
    inp = _predictions(["C", "C", "G", "G", "Am", "Am", "F", "F"])
    out = segment_frame_predictions(inp, min_segment_duration_ms=1)

    for i in range(len(out) - 1):
        assert out[i].start < out[i + 1].start
        assert out[i].end <= out[i + 1].start + 1e-12
        assert out[i].end == pytest.approx(out[i + 1].start, abs=1e-12)


def test_confidence_bounds() -> None:
    inp = _predictions(["C", "C", "G", "G"], confs=[1.2, -0.2, 0.8, 0.4])
    out = segment_frame_predictions(inp, min_segment_duration_ms=1)

    assert all(0.0 <= seg.confidence <= 1.0 for seg in out)


def test_stable_confidence_aggregation_mean() -> None:
    inp = _predictions(["C", "C", "C"], confs=[0.9, 0.6, 0.3])
    out = segment_frame_predictions(inp, min_segment_duration_ms=1)

    assert len(out) == 1
    assert out[0].confidence == pytest.approx((0.9 + 0.6 + 0.3) / 3.0, abs=1e-12)


def test_single_chord_sequence_produces_one_segment() -> None:
    inp = _predictions(["C", "C", "C", "C", "C"])
    out = segment_frame_predictions(inp, min_segment_duration_ms=1)

    assert len(out) == 1
    assert out[0].chord == "C"


def test_single_frame_produces_one_valid_segment() -> None:
    inp = _predictions(["C"], confs=[0.7])
    out = segment_frame_predictions(inp)

    assert len(out) == 1
    assert out[0].chord == "C"
    assert out[0].start == pytest.approx(0.0)
    assert out[0].end > 0.0
    assert out[0].confidence == pytest.approx(0.7)


def test_empty_input_returns_empty() -> None:
    assert segment_frame_predictions([]) == []


def test_short_jitter_segment_merges() -> None:
    inp = _predictions(["C", "C", "C", "G", "C", "C", "C"])
    out = segment_frame_predictions(inp)

    assert _labels(out) == ["C"]


def test_real_short_but_valid_transition_protection() -> None:
    min_frames = minimum_segment_frames(min_segment_duration_ms=MIN_SEGMENT_DURATION_MS)
    inp = _predictions(["C"] * min_frames + ["G"] * min_frames)
    out = segment_frame_predictions(inp)

    assert _labels(out) == ["C", "G"]


def test_same_neighbor_merge_strategy() -> None:
    inp = _predictions(["C", "C", "G", "C", "C"])
    out = segment_frame_predictions(inp)

    assert _labels(out) == ["C"]


def test_different_neighbors_merge_strategy_confidence_then_duration_then_lexical() -> None:
    # Short G should merge into Am because right confidence evidence is stronger.
    inp = _predictions(["C", "C", "G", "Am", "Am"], confs=[0.6, 0.6, 0.2, 0.95, 0.95])
    out = segment_frame_predictions(inp, min_segment_duration_ms=30)

    assert _labels(out) == ["C", "Am"]


def test_sustained_n_is_preserved() -> None:
    inp = _predictions(["C", "C", "N", "N", "N", "N", "G", "G"])
    out = segment_frame_predictions(inp, min_segment_duration_ms=1)

    assert _labels(out) == ["C", "N", "G"]


def test_isolated_short_n_merges() -> None:
    inp = _predictions(["C", "C", "C", "N", "C", "C", "C"])
    out = segment_frame_predictions(inp)

    assert _labels(out) == ["C"]


def test_determinism_same_input_same_output() -> None:
    inp = _predictions(["C", "C", "G", "G", "Am", "Am", "F", "F"], confs=[0.7] * 8)
    first = segment_frame_predictions(inp)
    second = segment_frame_predictions(inp)

    assert [(s.start, s.end, s.chord, s.confidence) for s in first] == [
        (s.start, s.end, s.chord, s.confidence) for s in second
    ]


def test_important_integrity_sequence_order_and_geometry() -> None:
    inp = _predictions(["C"] * 4 + ["G"] * 4 + ["Am"] * 4 + ["F"] * 4)
    out = segment_frame_predictions(inp, min_segment_duration_ms=1)

    assert _labels(out) == ["C", "G", "Am", "F"]
    for i in range(len(out) - 1):
        assert out[i].start < out[i].end
        assert out[i].end == pytest.approx(out[i + 1].start, abs=1e-12)


def test_frame_timing_method_uses_hop_and_sample_rate() -> None:
    frame_sec = frame_duration_seconds()
    expected = 512.0 / 22050.0
    assert frame_sec == pytest.approx(expected, abs=1e-12)


def test_default_minimum_duration_is_250ms() -> None:
    min_frames = minimum_segment_frames()
    frame_sec = frame_duration_seconds()
    duration_ms = min_frames * frame_sec * 1000.0

    assert duration_ms >= 250.0
    assert duration_ms < 250.0 + (frame_sec * 1000.0) + 1e-9
