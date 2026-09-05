from __future__ import annotations

from itertools import groupby

from chord_engine.detector import FrameChordPrediction
from chord_engine.smoothing import (
    DEFAULT_SMOOTHING_MS,
    smooth_frame_predictions,
    smoothing_window_frames,
)


def _predictions(labels: list[str], confidence: float = 0.9) -> list[FrameChordPrediction]:
    return [FrameChordPrediction(chord=label, confidence=confidence) for label in labels]


def _labels(predictions: list[FrameChordPrediction]) -> list[str]:
    return [p.chord for p in predictions]


def _compressed(labels: list[str]) -> list[str]:
    return [key for key, _ in groupby(labels)]


def test_stable_sequence_remains_unchanged() -> None:
    inp = _predictions(["C", "C", "C", "C", "C"])
    out = smooth_frame_predictions(inp)
    assert _labels(out) == ["C", "C", "C", "C", "C"]


def test_single_frame_jitter_is_smoothed() -> None:
    inp = _predictions(["C", "C", "C", "G", "C", "C", "C"])
    out = smooth_frame_predictions(inp)
    assert _labels(out) == ["C", "C", "C", "C", "C", "C", "C"]


def test_minor_jitter_is_smoothed() -> None:
    inp = _predictions(["Am", "Am", "Am", "Em", "Am", "Am", "Am"])
    out = smooth_frame_predictions(inp)
    assert _labels(out) == ["Am", "Am", "Am", "Am", "Am", "Am", "Am"]


def test_real_transition_remains_present() -> None:
    inp = _predictions(["C", "C", "C", "C", "G", "G", "G", "G"])
    out = smooth_frame_predictions(inp)
    labels = _labels(out)

    assert "C" in labels
    assert "G" in labels
    assert _compressed(labels) == ["C", "G"]


def test_no_chord_region_remains_unchanged() -> None:
    inp = _predictions(["N", "N", "N", "N", "N"])
    out = smooth_frame_predictions(inp)
    assert _labels(out) == ["N", "N", "N", "N", "N"]


def test_isolated_no_chord_is_smoothed_by_majority() -> None:
    inp = _predictions(["C", "C", "C", "N", "C", "C", "C"])
    out = smooth_frame_predictions(inp)
    assert _labels(out) == ["C", "C", "C", "C", "C", "C", "C"]


def test_sustained_no_chord_region_remains_present() -> None:
    inp = _predictions(["C", "C", "N", "N", "N", "N", "C", "C"])
    out = smooth_frame_predictions(inp)
    labels = _labels(out)

    assert "N" in labels
    assert labels.count("N") >= 2


def test_empty_input_returns_empty() -> None:
    out = smooth_frame_predictions([])
    assert out == []


def test_single_frame_is_unchanged() -> None:
    inp = _predictions(["C"])
    out = smooth_frame_predictions(inp)
    assert _labels(out) == ["C"]


def test_shorter_than_window_sequence_is_safe_and_deterministic() -> None:
    inp = _predictions(["C", "G", "C"])
    first = smooth_frame_predictions(inp)
    second = smooth_frame_predictions(inp)

    assert _labels(first) == _labels(second)
    assert len(first) == len(inp)


def test_output_length_is_preserved() -> None:
    inp = _predictions(["C", "C", "G", "G", "Am", "Am", "N"])
    out = smooth_frame_predictions(inp)
    assert len(out) == len(inp)


def test_confidence_bounds_are_preserved() -> None:
    inp = [
        FrameChordPrediction(chord="C", confidence=0.95),
        FrameChordPrediction(chord="C", confidence=0.75),
        FrameChordPrediction(chord="G", confidence=0.20),
        FrameChordPrediction(chord="C", confidence=0.85),
        FrameChordPrediction(chord="C", confidence=0.90),
    ]
    out = smooth_frame_predictions(inp)

    assert all(0.0 <= p.confidence <= 1.0 for p in out)


def test_determinism_for_same_input() -> None:
    inp = [
        FrameChordPrediction(chord="C", confidence=0.9),
        FrameChordPrediction(chord="G", confidence=0.8),
        FrameChordPrediction(chord="C", confidence=0.85),
        FrameChordPrediction(chord="C", confidence=0.95),
        FrameChordPrediction(chord="Am", confidence=0.7),
    ]

    first = smooth_frame_predictions(inp)
    second = smooth_frame_predictions(inp)

    assert [(x.chord, x.confidence) for x in first] == [(x.chord, x.confidence) for x in second]


def test_over_smoothing_protection_keeps_sustained_regions() -> None:
    inp = _predictions(["C", "C", "C", "C", "G", "G", "G", "G", "Am", "Am", "Am", "Am"])
    out = smooth_frame_predictions(inp)
    compressed = _compressed(_labels(out))

    assert compressed == ["C", "G", "Am"]


def test_window_default_is_odd_and_in_target_ms_band() -> None:
    window = smoothing_window_frames()
    frame_ms = 1000.0 * 512.0 / 22050.0
    approx_ms = window * frame_ms

    assert window % 2 == 1
    assert 300.0 <= approx_ms <= 700.0


def test_tie_break_prefers_center_label_when_tied() -> None:
    inp = [
        FrameChordPrediction(chord="C", confidence=0.9),
        FrameChordPrediction(chord="G", confidence=0.9),
        FrameChordPrediction(chord="Am", confidence=0.9),
    ]
    out = smooth_frame_predictions(inp, window_frames=3)

    # Center frame should keep its label in an equal-count tie.
    assert out[1].chord == "G"


def test_tie_break_uses_confidence_then_lexical() -> None:
    inp = [
        FrameChordPrediction(chord="C", confidence=0.2),
        FrameChordPrediction(chord="C", confidence=0.3),
        FrameChordPrediction(chord="X", confidence=0.1),
        FrameChordPrediction(chord="G", confidence=0.9),
        FrameChordPrediction(chord="G", confidence=0.8),
    ]

    # At center index with window 5, tie between C and G excludes center label X,
    # so tie is broken by higher aggregate confidence.
    out_conf = smooth_frame_predictions(inp, window_frames=5)
    assert out_conf[2].chord == "G"

    inp_lex = [
        FrameChordPrediction(chord="C", confidence=0.5),
        FrameChordPrediction(chord="C", confidence=0.5),
        FrameChordPrediction(chord="X", confidence=0.1),
        FrameChordPrediction(chord="G", confidence=0.5),
        FrameChordPrediction(chord="G", confidence=0.5),
    ]

    # Equal tie confidence should fall back to lexical order.
    out_lex = smooth_frame_predictions(inp_lex, window_frames=5)
    assert out_lex[2].chord == "C"


def test_default_window_ms_constant() -> None:
    assert DEFAULT_SMOOTHING_MS == 500
