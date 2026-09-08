from __future__ import annotations

import numpy as np

from chord_engine.detector import KeyEstimate
from chord_engine.features import BeatTiming
from chord_engine.lead_sheet import arrange_audio_lead_sheet
from chord_engine.segmentation import ChordSegment


def test_audio_lead_sheet_uses_bar_windows_when_beat_timing_is_reliable() -> None:
	segments = [
		ChordSegment(start=0.0, end=1.0, chord="C", confidence=0.88),
		ChordSegment(start=1.0, end=2.0, chord="Em", confidence=0.84),
		ChordSegment(start=2.0, end=3.0, chord="F", confidence=0.86),
		ChordSegment(start=3.0, end=4.0, chord="G", confidence=0.85),
		ChordSegment(start=4.0, end=5.0, chord="C", confidence=0.88),
		ChordSegment(start=5.0, end=6.0, chord="Em", confidence=0.84),
		ChordSegment(start=6.0, end=7.0, chord="F", confidence=0.86),
		ChordSegment(start=7.0, end=8.0, chord="G", confidence=0.85),
	]
	beat_timing = BeatTiming(
		boundaries=np.arange(0, 10, dtype=np.int32),
		tempo_bpm=60.0,
		beat_count=9,
		is_reliable=True,
	)

	arranged = arrange_audio_lead_sheet(
		segments,
		detected_key=KeyEstimate(tonic_pc=0, mode="major", confidence=0.82),
		duration_seconds=8.0,
		beat_timing=beat_timing,
		hop_length=22050,
		sample_rate=22050,
	)

	assert [segment.chord for segment in arranged] == ["C", "G", "C", "G"]
	assert [round(segment.start, 2) for segment in arranged] == [0.0, 3.0, 4.0, 7.0]


def test_audio_lead_sheet_falls_back_to_phrase_windows_without_reliable_beats() -> None:
	segments = [
		ChordSegment(start=0.0, end=2.0, chord="C", confidence=0.88),
		ChordSegment(start=2.0, end=4.0, chord="Em", confidence=0.84),
		ChordSegment(start=4.0, end=6.0, chord="F", confidence=0.86),
	]
	beat_timing = BeatTiming(
		boundaries=np.array([0, 10], dtype=np.int32),
		tempo_bpm=None,
		beat_count=1,
		is_reliable=False,
	)

	arranged = arrange_audio_lead_sheet(
		segments,
		detected_key=KeyEstimate(tonic_pc=0, mode="major", confidence=0.82),
		duration_seconds=6.0,
		beat_timing=beat_timing,
		hop_length=512,
		sample_rate=22050,
	)

	assert [segment.chord for segment in arranged] == ["C", "Em", "F"]
