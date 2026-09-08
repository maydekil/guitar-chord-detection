"""Protect metrical timing from analysis-window subdivision artifacts."""

from unittest.mock import Mock

import numpy as np
import pytest

import chord_engine.features as features
from chord_engine.lead_sheet import _first_beat_seconds
from chord_engine.musical_timing import _beat_seconds


def _timing(monkeypatch, beats, *, window=0.45):
    monkeypatch.setattr(features.librosa.beat, "beat_track", Mock(return_value=(120.0, beats)))
    return features.estimate_beat_timing(
        np.ones(1000, dtype=np.float32), sample_rate=100, hop_length=1,
        n_frames=1000, max_region_seconds=window,
    )


def test_densification_does_not_invent_beats_or_reliability(monkeypatch):
    beats = np.array([80, 130, 180])
    timing = _timing(monkeypatch, beats, window=0.10)
    assert timing.boundaries.size > 50
    assert timing.beat_count == 3
    assert not timing.is_reliable
    np.testing.assert_array_equal(timing.beat_frames, beats)


def test_grid_origin_is_invariant_to_analysis_window_size(monkeypatch):
    beats = np.arange(80, 981, 50)
    sparse = _timing(monkeypatch, beats, window=0.70)
    dense = _timing(monkeypatch, beats, window=0.10)
    assert sparse.is_reliable and dense.is_reliable
    assert sparse.boundaries.size != dense.boundaries.size
    for timing in (sparse, dense):
        assert timing.beat_count == len(beats)
        assert _first_beat_seconds(beat_timing=timing, hop_length=1, sample_rate=100) == 0.8
        assert _beat_seconds(timing, hop_length=1, sample_rate=100) == pytest.approx(beats / 100)


def test_actual_beat_at_zero_is_preserved(monkeypatch):
    timing = _timing(monkeypatch, np.arange(0, 1000, 50))
    assert features.beat_times_seconds(timing, hop_length=1, sample_rate=100)[0] == 0.0


@pytest.mark.parametrize("failure", [False, True])
def test_failed_or_empty_tracker_has_windows_but_no_metrical_grid(monkeypatch, failure):
    tracker = Mock(side_effect=RuntimeError("tracker failed")) if failure else Mock(return_value=(120.0, []))
    monkeypatch.setattr(features.librosa.beat, "beat_track", tracker)
    timing = features.estimate_beat_timing(
        np.ones(1000), sample_rate=100, hop_length=1, n_frames=1000,
    )
    assert timing.boundaries[0] == 0 and timing.boundaries[-1] == 1000
    assert timing.beat_count == 0
    assert not timing.is_reliable
    assert features.beat_times_seconds(timing, hop_length=1, sample_rate=100) == []


def test_one_frame_window_still_returns_timing_object(monkeypatch):
    timing = _timing(monkeypatch, np.arange(0, 1000, 50), window=0.01)
    assert isinstance(timing, features.BeatTiming)
    assert np.max(np.diff(timing.boundaries)) == 1
    assert timing.beat_count == 20
