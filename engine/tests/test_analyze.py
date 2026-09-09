from __future__ import annotations

import os
import subprocess
from pathlib import Path
from shutil import which

import numpy as np
import pytest
import soundfile as sf

import chord_engine.analyze as analyze_module
from chord_engine.analyze import (
    ALGORITHM_ID,
    CONTRACT_VERSION,
    AnalysisError,
    AnalysisMetadata,
    AnalysisResult,
    PipelineRun,
    SourceMetadata,
    analyze_audio,
    compare_baseline_vs_improved,
)
from chord_engine.audio import TARGET_SAMPLE_RATE
from chord_engine.detector import KeyEstimate
from chord_engine.playable_progression import (
    _absorb_tonic_predominant_mediant_approach,
    _apply_major_repeated_tonic_phrase_answer,
    _apply_repeated_phrase_quality_consistency,
    _split_initial_tonic_mediant_before_predominant,
    _stabilize_initial_tonic_pickup,
)
from chord_engine.segmentation import ChordSegment


def _tone(freq: float, duration: float = 1.2, amp: float = 0.45, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _chord(freqs: list[float], duration: float = 1.2, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    stacked = np.stack([_tone(f, duration=duration, sr=sr) for f in freqs], axis=0)
    return np.mean(stacked, axis=0).astype(np.float32)


def _write_wav(path: Path, samples: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> None:
    sf.write(path, np.asarray(samples, dtype=np.float32), sr)


def _segment_labels_with_duration(result_dict: dict[str, object]) -> list[tuple[str, float]]:
    chords = result_dict["analysis"]["chords"]  # type: ignore[index]
    out: list[tuple[str, float]] = []
    for seg in chords:  # type: ignore[assignment]
        duration = float(seg["end"] - seg["start"])
        out.append((str(seg["chord"]), duration))
    return out


def _dominant_segment_label(result_dict: dict[str, object]) -> str:
    labels = _segment_labels_with_duration(result_dict)
    labels.sort(key=lambda x: x[1], reverse=True)
    return labels[0][0]


def test_analyze_synthetic_c_major_file_returns_c(tmp_path: Path) -> None:
    wav = tmp_path / "c_major.wav"
    _write_wav(wav, _chord([261.63, 329.63, 392.00], duration=1.6))

    result = analyze_audio(wav).to_dict()

    assert result["version"] == CONTRACT_VERSION
    assert result["analysis"]["algorithm"] == ALGORITHM_ID  # type: ignore[index]
    assert _dominant_segment_label(result) == "C"


def test_analyze_synthetic_a_minor_file_returns_am(tmp_path: Path) -> None:
    wav = tmp_path / "a_minor.wav"
    _write_wav(wav, _chord([220.00, 261.63, 329.63], duration=1.6))

    result = analyze_audio(wav).to_dict()

    assert _dominant_segment_label(result) == "Am"


def test_analyze_silence_returns_meaningful_n(tmp_path: Path) -> None:
    wav = tmp_path / "silence.wav"
    _write_wav(wav, np.zeros(int(TARGET_SAMPLE_RATE * 1.5), dtype=np.float32))

    result = analyze_audio(wav).to_dict()
    labels = [seg["chord"] for seg in result["analysis"]["chords"]]  # type: ignore[index]

    assert "N" in labels


def test_analyze_progression_preserves_chord_order(tmp_path: Path) -> None:
    c = _chord([261.63, 329.63, 392.00], duration=1.2)
    g = _chord([196.00, 246.94, 293.66], duration=1.2)
    am = _chord([220.00, 261.63, 329.63], duration=1.2)
    f = _chord([174.61, 220.00, 261.63], duration=1.2)
    progression = np.concatenate([c, g, am, f]).astype(np.float32)

    wav = tmp_path / "progression.wav"
    _write_wav(wav, progression)

    result = analyze_audio(wav).to_dict()
    labels = [seg["chord"] for seg in result["analysis"]["chords"]]  # type: ignore[index]

    # Allow boundary shifts, but sustained regions must survive in order.
    compressed = [labels[0]] if labels else []
    for lab in labels[1:]:
        if lab != compressed[-1]:
            compressed.append(lab)

    assert compressed == ["C", "G", "Am", "F"]


def test_analyze_e_major_with_melody_contamination_remains_e(tmp_path: Path) -> None:
    base = _chord([164.81, 207.65, 246.94], duration=1.2)
    melody = np.concatenate(
        [
            _tone(329.63, duration=0.4, amp=0.22),
            _tone(349.23, duration=0.4, amp=0.22),
            _tone(392.00, duration=0.4, amp=0.22),
        ]
    ).astype(np.float32)
    wav = tmp_path / "e_major_melody_contamination.wav"
    _write_wav(wav, (base + melody).astype(np.float32))

    result = analyze_audio(wav).to_dict()
    assert _dominant_segment_label(result) == "E"


def test_analyze_a_major_with_moving_bass_stays_rooted_as_a(tmp_path: Path) -> None:
    upper_a_major = _chord([220.00, 277.18, 329.63], duration=2.4)
    bass = np.concatenate(
        [
            _tone(110.00, duration=0.8, amp=0.30),
            _tone(82.41, duration=0.8, amp=0.30),
            _tone(138.59, duration=0.8, amp=0.30),
        ]
    ).astype(np.float32)
    wav = tmp_path / "a_major_moving_bass.wav"
    _write_wav(wav, (upper_a_major + bass).astype(np.float32))

    result = analyze_audio(wav).to_dict()
    assert _dominant_segment_label(result) == "A"


def test_analyze_inversion_like_bass_does_not_auto_change_root(tmp_path: Path) -> None:
    c_major_upper = _chord([261.63, 329.63, 392.00], duration=1.8)
    low_e_bass = _tone(82.41, duration=1.8, amp=0.34)
    wav = tmp_path / "c_major_with_low_e_bass.wav"
    _write_wav(wav, (c_major_upper + low_e_bass).astype(np.float32))

    result = analyze_audio(wav).to_dict()
    assert _dominant_segment_label(result) == "C"


def test_analyze_invalid_audio_raises_controlled_error(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.txt"
    bad.write_text("not audio", encoding="utf-8")

    with pytest.raises(AnalysisError) as exc:
        analyze_audio(bad)

    assert exc.value.code == "AUDIO_DECODE_FAILED"


def test_analyze_is_deterministic_for_same_file(tmp_path: Path) -> None:
    wav = tmp_path / "deterministic.wav"
    _write_wav(wav, _chord([261.63, 329.63, 392.00], duration=1.2))

    first = analyze_audio(wav).to_dict()
    second = analyze_audio(wav).to_dict()

    assert first["version"] == second["version"]
    assert first["source"] == second["source"]

    first_chords = first["analysis"]["chords"]  # type: ignore[index]
    second_chords = second["analysis"]["chords"]  # type: ignore[index]
    assert len(first_chords) == len(second_chords)

    for a, b in zip(first_chords, second_chords, strict=True):
        assert a["chord"] == b["chord"]
        assert a["start"] == pytest.approx(b["start"], abs=1e-10)
        assert a["end"] == pytest.approx(b["end"], abs=1e-10)
        assert a["confidence"] == pytest.approx(b["confidence"], abs=1e-10)


def test_analyze_segment_integrity_and_duration_bound(tmp_path: Path) -> None:
    wav = tmp_path / "integrity.wav"
    _write_wav(wav, _chord([261.63, 329.63, 392.00], duration=2.0))

    result = analyze_audio(wav).to_dict()
    source_duration = float(result["source"]["duration"])  # type: ignore[index]
    chords = result["analysis"]["chords"]  # type: ignore[index]

    assert len(chords) >= 1
    for seg in chords:
        assert float(seg["start"]) >= 0.0
        assert float(seg["end"]) > float(seg["start"])
        assert 0.0 <= float(seg["confidence"]) <= 1.0

    for prev, nxt in zip(chords, chords[1:], strict=False):
        assert float(prev["start"]) <= float(nxt["start"])
        assert float(prev["end"]) <= float(nxt["start"]) + 1e-9

    assert float(chords[-1]["end"]) <= source_duration + 1e-6


def test_analyze_mp3_path_if_available(tmp_path: Path) -> None:
    if which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    wav = tmp_path / "source.wav"
    mp3 = tmp_path / "source.mp3"
    _write_wav(wav, _chord([220.00, 261.63, 329.63], duration=1.6))

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(wav),
        str(mp3),
    ]
    subprocess.run(cmd, check=True)

    result = analyze_audio(mp3).to_dict()
    assert _dominant_segment_label(result) == "Am"


def test_playable_progression_consistent_quality_across_repeated_root_phrases() -> None:
    key = KeyEstimate(tonic_pc=2, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=2.0, chord="D", confidence=0.86),
        ChordSegment(start=2.0, end=4.0, chord="F#m", confidence=0.84),
        ChordSegment(start=4.0, end=6.0, chord="G", confidence=0.83),
        ChordSegment(start=6.0, end=8.0, chord="A", confidence=0.85),
        ChordSegment(start=8.0, end=10.0, chord="D", confidence=0.84),
        ChordSegment(start=10.0, end=12.0, chord="F#m", confidence=0.83),
        ChordSegment(start=12.0, end=13.6, chord="Gm", confidence=0.58),
        ChordSegment(start=13.6, end=16.0, chord="A", confidence=0.85),
    ]

    refined, events = _apply_repeated_phrase_quality_consistency(segments, key)

    assert [segment.chord for segment in refined] == ["D", "F#m", "G", "A", "D", "F#m", "G", "A"]
    assert any(event.replaced_chord == "Gm" and event.new_chord == "G" for event in events)


def test_playable_progression_preserves_strong_repeated_quality_change() -> None:
    key = KeyEstimate(tonic_pc=2, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=2.0, chord="D", confidence=0.86),
        ChordSegment(start=2.0, end=4.0, chord="F#m", confidence=0.84),
        ChordSegment(start=4.0, end=7.4, chord="G", confidence=0.83),
        ChordSegment(start=7.4, end=9.4, chord="A", confidence=0.85),
        ChordSegment(start=9.4, end=11.4, chord="D", confidence=0.84),
        ChordSegment(start=11.4, end=13.4, chord="F#m", confidence=0.83),
        ChordSegment(start=13.4, end=16.8, chord="Gm", confidence=0.90),
        ChordSegment(start=16.8, end=18.8, chord="A", confidence=0.85),
    ]

    refined, events = _apply_repeated_phrase_quality_consistency(segments, key)

    assert "Gm" in [segment.chord for segment in refined]
    assert not any(event.replaced_chord == "Gm" and event.new_chord == "G" for event in events)


def test_initial_tonic_pickup_can_absorb_weak_relative_minor_substitute() -> None:
    key = KeyEstimate(tonic_pc=0, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=1.0, chord="C", confidence=0.82),
        ChordSegment(start=1.0, end=5.2, chord="Am", confidence=0.76),
        ChordSegment(start=5.2, end=8.0, chord="Em", confidence=0.82),
        ChordSegment(start=8.0, end=11.0, chord="C", confidence=0.86),
    ]

    refined, events = _stabilize_initial_tonic_pickup(segments, key)

    assert [segment.chord for segment in refined[:3]] == ["C", "Em", "C"]
    assert refined[0].start == 0.0
    assert refined[0].end == 5.2
    assert any(event.replaced_chord == "Am" and event.new_chord == "C" for event in events)


def test_initial_tonic_pickup_preserves_strong_relative_minor() -> None:
    key = KeyEstimate(tonic_pc=0, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=1.0, chord="C", confidence=0.84),
        ChordSegment(start=1.0, end=5.2, chord="Am", confidence=0.91),
        ChordSegment(start=5.2, end=8.0, chord="F", confidence=0.82),
        ChordSegment(start=8.0, end=11.0, chord="C", confidence=0.86),
    ]

    refined, events = _stabilize_initial_tonic_pickup(segments, key)

    assert [segment.chord for segment in refined] == ["C", "Am", "F", "C"]
    assert not events


def test_initial_tonic_pickup_can_correct_weak_off_tonic_opening() -> None:
    key = KeyEstimate(tonic_pc=2, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=0.9, chord="F#m", confidence=0.66),
        ChordSegment(start=0.9, end=3.4, chord="D", confidence=0.80),
        ChordSegment(start=3.4, end=6.0, chord="G", confidence=0.84),
        ChordSegment(start=6.0, end=8.5, chord="A", confidence=0.84),
    ]

    refined, events = _stabilize_initial_tonic_pickup(segments, key)

    assert [segment.chord for segment in refined[:3]] == ["D", "G", "A"]
    assert refined[0].start == 0.0
    assert refined[0].end == 3.4
    assert any(event.replaced_chord == "F#m" and event.new_chord == "D" for event in events)


def test_initial_tonic_pickup_can_absorb_weak_opening_dominant() -> None:
    key = KeyEstimate(tonic_pc=2, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=1.0, chord="D", confidence=0.82),
        ChordSegment(start=1.0, end=4.0, chord="A", confidence=0.76),
        ChordSegment(start=4.0, end=6.8, chord="G", confidence=0.84),
        ChordSegment(start=6.8, end=9.4, chord="D", confidence=0.86),
    ]

    refined, events = _stabilize_initial_tonic_pickup(segments, key)

    assert [segment.chord for segment in refined[:3]] == ["D", "G", "D"]
    assert refined[0].end == 4.0
    assert any(event.replaced_chord == "A" and event.new_chord == "D" for event in events)


def test_initial_tonic_pickup_preserves_strong_opening_dominant() -> None:
    key = KeyEstimate(tonic_pc=2, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=1.0, chord="D", confidence=0.82),
        ChordSegment(start=1.0, end=4.0, chord="A", confidence=0.91),
        ChordSegment(start=4.0, end=6.8, chord="G", confidence=0.84),
        ChordSegment(start=6.8, end=9.4, chord="D", confidence=0.86),
    ]

    refined, events = _stabilize_initial_tonic_pickup(segments, key)

    assert [segment.chord for segment in refined] == ["D", "A", "G", "D"]
    assert not events


def test_initial_tonic_can_split_to_mediant_before_predominant() -> None:
    key = KeyEstimate(tonic_pc=2, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=8.8, chord="D", confidence=0.86),
        ChordSegment(start=8.8, end=13.6, chord="G", confidence=0.82),
        ChordSegment(start=13.6, end=17.0, chord="A", confidence=0.84),
    ]

    refined, events = _split_initial_tonic_mediant_before_predominant(segments, key)

    assert [segment.chord for segment in refined] == ["D", "F#m", "G", "A"]
    assert refined[0].start == 0.0
    assert refined[1].end == 8.8
    assert any(event.replaced_chord == "D" and event.new_chord == "F#m" for event in events)


def test_initial_tonic_mediant_split_preserves_short_tonic() -> None:
    key = KeyEstimate(tonic_pc=2, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=4.0, chord="D", confidence=0.86),
        ChordSegment(start=4.0, end=8.8, chord="G", confidence=0.82),
        ChordSegment(start=8.8, end=12.0, chord="A", confidence=0.84),
    ]

    refined, events = _split_initial_tonic_mediant_before_predominant(segments, key)

    assert [segment.chord for segment in refined] == ["D", "G", "A"]
    assert not events


def test_tonic_predominant_mediant_approach_absorbs_short_predominant() -> None:
    key = KeyEstimate(tonic_pc=0, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=5.4, chord="C", confidence=0.86),
        ChordSegment(start=5.4, end=8.0, chord="Dm", confidence=0.72),
        ChordSegment(start=8.0, end=10.6, chord="Em", confidence=0.80),
        ChordSegment(start=10.6, end=13.0, chord="F", confidence=0.84),
    ]

    refined, events = _absorb_tonic_predominant_mediant_approach(segments, key)

    assert [segment.chord for segment in refined] == ["C", "Em", "F"]
    assert refined[1].start == 5.4
    assert refined[1].end == 10.6
    assert any(event.replaced_chord == "Dm" and event.new_chord == "Em" for event in events)


def test_tonic_predominant_mediant_approach_preserves_strong_predominant() -> None:
    key = KeyEstimate(tonic_pc=0, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=5.4, chord="C", confidence=0.86),
        ChordSegment(start=5.4, end=8.0, chord="Dm", confidence=0.91),
        ChordSegment(start=8.0, end=10.6, chord="Em", confidence=0.80),
        ChordSegment(start=10.6, end=13.0, chord="F", confidence=0.84),
    ]

    refined, events = _absorb_tonic_predominant_mediant_approach(segments, key)

    assert [segment.chord for segment in refined] == ["C", "Dm", "Em", "F"]
    assert not events


def test_tonic_predominant_mediant_approach_preserves_predominant_to_dominant() -> None:
    key = KeyEstimate(tonic_pc=0, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=5.4, chord="C", confidence=0.86),
        ChordSegment(start=5.4, end=8.0, chord="Dm", confidence=0.72),
        ChordSegment(start=8.0, end=10.6, chord="G", confidence=0.80),
        ChordSegment(start=10.6, end=13.0, chord="C", confidence=0.84),
    ]

    refined, events = _absorb_tonic_predominant_mediant_approach(segments, key)

    assert [segment.chord for segment in refined] == ["C", "Dm", "G", "C"]
    assert not events


def test_repeated_tonic_phrase_can_form_dominant_answer_phrase() -> None:
    key = KeyEstimate(tonic_pc=0, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=5.4, chord="C", confidence=0.86),
        ChordSegment(start=5.4, end=10.6, chord="Em", confidence=0.80),
        ChordSegment(start=10.6, end=13.0, chord="F", confidence=0.78),
        ChordSegment(start=13.0, end=16.1, chord="C", confidence=0.82),
        ChordSegment(start=16.1, end=19.4, chord="Em", confidence=0.74),
        ChordSegment(start=19.4, end=23.0, chord="F", confidence=0.72),
        ChordSegment(start=23.0, end=27.8, chord="C", confidence=0.73),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == ["C", "Em", "F", "C", "G", "Am", "Dm", "G"]
    assert refined[6].start == 23.0
    assert refined[7].end == 27.8
    assert any(event.replaced_chord == "Em" and event.new_chord == "G" for event in events)


def test_repeated_tonic_phrase_answer_ignores_late_repetition() -> None:
    key = KeyEstimate(tonic_pc=0, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=4.0, chord="C", confidence=0.86),
        ChordSegment(start=4.0, end=8.0, chord="G", confidence=0.86),
        ChordSegment(start=21.0, end=24.0, chord="C", confidence=0.78),
        ChordSegment(start=24.0, end=27.0, chord="Em", confidence=0.76),
        ChordSegment(start=27.0, end=30.0, chord="F", confidence=0.74),
        ChordSegment(start=30.0, end=33.0, chord="C", confidence=0.77),
        ChordSegment(start=33.0, end=36.0, chord="Em", confidence=0.76),
        ChordSegment(start=36.0, end=39.0, chord="F", confidence=0.74),
        ChordSegment(start=39.0, end=42.0, chord="C", confidence=0.77),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == [segment.chord for segment in segments]
    assert not events


def test_repeated_opening_cadence_can_form_mediant_tonic_answer() -> None:
    key = KeyEstimate(tonic_pc=2, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=4.5, chord="D", confidence=0.88),
        ChordSegment(start=4.5, end=9.0, chord="F#m", confidence=0.86),
        ChordSegment(start=9.0, end=13.5, chord="G", confidence=0.82),
        ChordSegment(start=13.5, end=18.0, chord="A", confidence=0.84),
        ChordSegment(start=18.0, end=23.0, chord="D", confidence=0.83),
        ChordSegment(start=23.0, end=28.0, chord="A", confidence=0.80),
        ChordSegment(start=28.0, end=32.0, chord="G", confidence=0.78),
        ChordSegment(start=32.0, end=38.0, chord="A", confidence=0.79),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == ["D", "F#m", "G", "A", "D", "F#m", "G", "D"]
    assert any(event.replaced_chord == "A" and event.new_chord == "F#m" for event in events)


def test_opening_cycle_completion_recovers_shifted_second_major_phrase() -> None:
    key = KeyEstimate(tonic_pc=0, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=5.4, chord="C", confidence=0.90),
        ChordSegment(start=5.4, end=8.8, chord="Em", confidence=0.86),
        ChordSegment(start=8.8, end=11.6, chord="F", confidence=0.84),
        ChordSegment(start=11.6, end=17.1, chord="G", confidence=0.88),
        ChordSegment(start=17.1, end=20.2, chord="Em", confidence=0.78),
        ChordSegment(start=20.2, end=22.7, chord="F", confidence=0.76),
        ChordSegment(start=22.7, end=28.4, chord="C", confidence=0.80),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == ["C", "Em", "F", "G", "C", "Em", "F", "G"]
    assert refined[4].start > 11.6
    assert refined[4].end == 17.1
    assert refined[6].start == 20.2
    assert refined[7].end == 28.4
    assert any(event.replaced_chord == "G" and event.new_chord == "C" for event in events)


def test_completed_opening_cycle_can_recover_post_cycle_tonic_restart() -> None:
    key = KeyEstimate(tonic_pc=0, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=5.4, chord="C", confidence=0.90),
        ChordSegment(start=5.4, end=8.8, chord="Em", confidence=0.86),
        ChordSegment(start=8.8, end=11.6, chord="F", confidence=0.84),
        ChordSegment(start=11.6, end=14.2, chord="G", confidence=0.88),
        ChordSegment(start=14.2, end=17.1, chord="C", confidence=0.84),
        ChordSegment(start=17.1, end=20.2, chord="Em", confidence=0.78),
        ChordSegment(start=20.2, end=25.8, chord="F", confidence=0.76),
        ChordSegment(start=25.8, end=28.4, chord="G", confidence=0.80),
        ChordSegment(start=28.4, end=31.6, chord="Em", confidence=0.76),
        ChordSegment(start=31.6, end=34.9, chord="F", confidence=0.76),
        ChordSegment(start=34.9, end=37.2, chord="C", confidence=0.82),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == ["C", "Em", "F", "G", "C", "Em", "F", "G", "C", "Em", "F", "C"]
    assert refined[8].start == 28.4
    assert refined[8].end == pytest.approx(29.872)
    assert refined[9].start == pytest.approx(29.872)
    assert any(event.replaced_chord == "Em" and event.new_chord == "C" for event in events)


def test_degraded_major_opening_answer_recovers_playable_roles() -> None:
    key = KeyEstimate(tonic_pc=0, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=5.4, chord="C", confidence=0.90),
        ChordSegment(start=5.4, end=10.6, chord="Dm", confidence=0.74),
        ChordSegment(start=10.6, end=13.1, chord="C", confidence=0.74),
        ChordSegment(start=13.1, end=16.1, chord="G", confidence=0.78),
        ChordSegment(start=16.1, end=21.2, chord="Em", confidence=0.72),
        ChordSegment(start=21.2, end=27.8, chord="C", confidence=0.72),
        ChordSegment(start=27.8, end=31.4, chord="Em", confidence=0.78),
        ChordSegment(start=31.4, end=34.8, chord="F", confidence=0.78),
        ChordSegment(start=34.8, end=39.8, chord="C", confidence=0.84),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == ["C", "Em", "F", "C", "G", "Am", "Dm", "G", "C", "Em", "F", "C"]
    assert any(event.replaced_chord == "Dm" and event.new_chord == "Em" for event in events)
    assert any(event.replaced_chord == "Em" and event.new_chord == "Am" for event in events)


def test_relative_minor_key_can_use_clear_relative_major_opening_context() -> None:
    key = KeyEstimate(tonic_pc=6, mode="minor", confidence=0.74)
    segments = [
        ChordSegment(start=0.0, end=5.4, chord="A", confidence=0.90),
        ChordSegment(start=5.4, end=8.8, chord="C#m", confidence=0.86),
        ChordSegment(start=8.8, end=11.6, chord="D", confidence=0.84),
        ChordSegment(start=11.6, end=14.2, chord="E", confidence=0.88),
        ChordSegment(start=14.2, end=17.1, chord="A", confidence=0.84),
        ChordSegment(start=17.1, end=20.2, chord="C#m", confidence=0.78),
        ChordSegment(start=20.2, end=25.8, chord="D", confidence=0.76),
        ChordSegment(start=25.8, end=28.4, chord="E", confidence=0.80),
        ChordSegment(start=28.4, end=31.6, chord="C#m", confidence=0.76),
        ChordSegment(start=31.6, end=34.9, chord="D", confidence=0.76),
        ChordSegment(start=34.9, end=37.2, chord="A", confidence=0.82),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == ["A", "C#m", "D", "E", "A", "C#m", "D", "E", "A", "C#m", "D", "A"]
    assert any(event.replaced_chord == "C#m" and event.new_chord == "A" for event in events)


def test_repeated_opening_cadence_preserves_short_answer_motion() -> None:
    key = KeyEstimate(tonic_pc=2, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=4.5, chord="D", confidence=0.88),
        ChordSegment(start=4.5, end=9.0, chord="F#m", confidence=0.86),
        ChordSegment(start=9.0, end=13.5, chord="G", confidence=0.82),
        ChordSegment(start=13.5, end=18.0, chord="A", confidence=0.84),
        ChordSegment(start=18.0, end=23.0, chord="D", confidence=0.83),
        ChordSegment(start=23.0, end=24.5, chord="A", confidence=0.80),
        ChordSegment(start=24.5, end=27.0, chord="G", confidence=0.78),
        ChordSegment(start=27.0, end=30.0, chord="A", confidence=0.79),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == [segment.chord for segment in segments]
    assert not events


def test_opening_answer_phrase_can_recover_tonic_restart() -> None:
    key = KeyEstimate(tonic_pc=0, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=5.4, chord="C", confidence=0.90),
        ChordSegment(start=5.4, end=10.6, chord="Em", confidence=0.84),
        ChordSegment(start=10.6, end=13.0, chord="F", confidence=0.82),
        ChordSegment(start=13.0, end=16.0, chord="C", confidence=0.86),
        ChordSegment(start=16.0, end=19.4, chord="G", confidence=0.78),
        ChordSegment(start=19.4, end=23.0, chord="Am", confidence=0.76),
        ChordSegment(start=23.0, end=25.7, chord="Dm", confidence=0.76),
        ChordSegment(start=25.7, end=27.9, chord="G", confidence=0.74),
        ChordSegment(start=27.9, end=31.4, chord="Em", confidence=0.78),
        ChordSegment(start=31.4, end=34.8, chord="F", confidence=0.78),
        ChordSegment(start=34.8, end=37.8, chord="C", confidence=0.84),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == ["C", "Em", "F", "C", "G", "Am", "Dm", "G", "C", "F", "C"]
    assert any(event.replaced_chord == "Em" and event.new_chord == "C" for event in events)


def test_opening_answer_phrase_restart_requires_complete_cadence_context() -> None:
    key = KeyEstimate(tonic_pc=0, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=5.4, chord="C", confidence=0.90),
        ChordSegment(start=5.4, end=10.6, chord="Em", confidence=0.84),
        ChordSegment(start=10.6, end=13.0, chord="F", confidence=0.82),
        ChordSegment(start=13.0, end=16.0, chord="C", confidence=0.86),
        ChordSegment(start=16.0, end=19.4, chord="Em", confidence=0.78),
        ChordSegment(start=19.4, end=23.0, chord="C", confidence=0.76),
        ChordSegment(start=23.0, end=25.7, chord="Em", confidence=0.76),
        ChordSegment(start=25.7, end=27.9, chord="F", confidence=0.74),
        ChordSegment(start=27.9, end=31.4, chord="Em", confidence=0.78),
        ChordSegment(start=31.4, end=34.8, chord="F", confidence=0.78),
        ChordSegment(start=34.8, end=39.8, chord="C", confidence=0.84),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == [segment.chord for segment in segments]
    assert not events


def test_opening_phrase_continuation_recovers_mediant_subdominant_motion() -> None:
    key = KeyEstimate(tonic_pc=0, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=5.4, chord="C", confidence=0.90),
        ChordSegment(start=5.4, end=10.6, chord="Em", confidence=0.84),
        ChordSegment(start=10.6, end=13.0, chord="F", confidence=0.82),
        ChordSegment(start=13.0, end=16.0, chord="C", confidence=0.86),
        ChordSegment(start=16.0, end=19.4, chord="G", confidence=0.78),
        ChordSegment(start=19.4, end=23.0, chord="Am", confidence=0.76),
        ChordSegment(start=23.0, end=25.7, chord="Dm", confidence=0.76),
        ChordSegment(start=25.7, end=27.9, chord="G", confidence=0.74),
        ChordSegment(start=27.9, end=31.4, chord="C", confidence=0.80),
        ChordSegment(start=31.4, end=34.8, chord="F", confidence=0.78),
        ChordSegment(start=34.8, end=39.8, chord="C", confidence=0.84),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == ["C", "Em", "F", "C", "G", "Am", "Dm", "G", "C", "Em", "F", "C"]
    assert any(event.replaced_chord == "F" and event.new_chord == "Em" for event in events)
    assert refined[10].start == 34.8
    assert refined[11].end == 39.8


def test_opening_phrase_continuation_preserves_unsplittable_tonic() -> None:
    key = KeyEstimate(tonic_pc=0, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=5.4, chord="C", confidence=0.90),
        ChordSegment(start=5.4, end=10.6, chord="Em", confidence=0.84),
        ChordSegment(start=10.6, end=13.0, chord="F", confidence=0.82),
        ChordSegment(start=13.0, end=16.0, chord="C", confidence=0.86),
        ChordSegment(start=16.0, end=19.4, chord="G", confidence=0.78),
        ChordSegment(start=19.4, end=23.0, chord="Am", confidence=0.76),
        ChordSegment(start=23.0, end=25.7, chord="Dm", confidence=0.76),
        ChordSegment(start=25.7, end=27.9, chord="G", confidence=0.74),
        ChordSegment(start=27.9, end=31.4, chord="C", confidence=0.80),
        ChordSegment(start=31.4, end=34.8, chord="F", confidence=0.78),
        ChordSegment(start=34.8, end=37.8, chord="C", confidence=0.84),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == [segment.chord for segment in segments]
    assert not events


def test_body_cadence_recovery_uses_opening_answer_context() -> None:
    key = KeyEstimate(tonic_pc=2, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=4.5, chord="D", confidence=0.88),
        ChordSegment(start=4.5, end=9.0, chord="F#m", confidence=0.86),
        ChordSegment(start=9.0, end=13.5, chord="G", confidence=0.82),
        ChordSegment(start=13.5, end=18.0, chord="A", confidence=0.84),
        ChordSegment(start=18.0, end=23.0, chord="D", confidence=0.83),
        ChordSegment(start=23.0, end=28.0, chord="F#m", confidence=0.80),
        ChordSegment(start=28.0, end=32.0, chord="G", confidence=0.78),
        ChordSegment(start=32.0, end=38.0, chord="D", confidence=0.79),
        ChordSegment(start=38.0, end=43.0, chord="G", confidence=0.78),
        ChordSegment(start=43.0, end=48.0, chord="A", confidence=0.78),
        ChordSegment(start=48.0, end=52.0, chord="D", confidence=0.76),
        ChordSegment(start=52.0, end=56.0, chord="G", confidence=0.76),
        ChordSegment(start=56.0, end=60.0, chord="A", confidence=0.80),
        ChordSegment(start=60.0, end=64.0, chord="D", confidence=0.82),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == [
        "D",
        "F#m",
        "G",
        "A",
        "D",
        "F#m",
        "G",
        "D",
        "G",
        "A",
        "Bm",
        "Em",
        "A",
        "D",
    ]
    assert any(event.replaced_chord == "D" and event.new_chord == "Bm" for event in events)
    assert any(event.replaced_chord == "G" and event.new_chord == "Em" for event in events)


def test_body_cadence_recovery_handles_subdominant_tail_resolution() -> None:
    key = KeyEstimate(tonic_pc=2, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=4.5, chord="D", confidence=0.88),
        ChordSegment(start=4.5, end=9.0, chord="F#m", confidence=0.86),
        ChordSegment(start=9.0, end=13.5, chord="G", confidence=0.82),
        ChordSegment(start=13.5, end=18.0, chord="A", confidence=0.84),
        ChordSegment(start=18.0, end=23.0, chord="D", confidence=0.83),
        ChordSegment(start=23.0, end=28.0, chord="F#m", confidence=0.80),
        ChordSegment(start=28.0, end=32.0, chord="G", confidence=0.78),
        ChordSegment(start=32.0, end=38.0, chord="D", confidence=0.79),
        ChordSegment(start=38.0, end=43.0, chord="G", confidence=0.78),
        ChordSegment(start=43.0, end=48.0, chord="A", confidence=0.78),
        ChordSegment(start=48.0, end=52.0, chord="D", confidence=0.76),
        ChordSegment(start=52.0, end=56.0, chord="G", confidence=0.76),
        ChordSegment(start=56.0, end=60.0, chord="A", confidence=0.80),
        ChordSegment(start=60.0, end=64.0, chord="G", confidence=0.78),
        ChordSegment(start=64.0, end=68.0, chord="D", confidence=0.82),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == [
        "D",
        "F#m",
        "G",
        "A",
        "D",
        "F#m",
        "G",
        "D",
        "A",
        "Bm",
        "Em",
        "A",
        "D",
        "G",
        "D",
    ]
    assert any(event.replaced_chord == "G" and event.new_chord == "A" for event in events)
    assert any(event.replaced_chord == "D" and event.new_chord == "Em" for event in events)


def test_body_cadence_recovery_ignores_unanchored_progression() -> None:
    key = KeyEstimate(tonic_pc=2, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=4.0, chord="D", confidence=0.88),
        ChordSegment(start=4.0, end=8.0, chord="A", confidence=0.86),
        ChordSegment(start=8.0, end=12.0, chord="G", confidence=0.82),
        ChordSegment(start=12.0, end=16.0, chord="A", confidence=0.84),
        ChordSegment(start=16.0, end=20.0, chord="D", confidence=0.83),
        ChordSegment(start=20.0, end=24.0, chord="A", confidence=0.80),
        ChordSegment(start=24.0, end=28.0, chord="G", confidence=0.78),
        ChordSegment(start=28.0, end=32.0, chord="D", confidence=0.79),
        ChordSegment(start=32.0, end=36.0, chord="G", confidence=0.78),
        ChordSegment(start=36.0, end=40.0, chord="A", confidence=0.78),
        ChordSegment(start=40.0, end=44.0, chord="D", confidence=0.76),
        ChordSegment(start=44.0, end=48.0, chord="G", confidence=0.76),
        ChordSegment(start=48.0, end=52.0, chord="A", confidence=0.80),
        ChordSegment(start=52.0, end=56.0, chord="D", confidence=0.82),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == [segment.chord for segment in segments]
    assert not events


def test_post_cadence_tonic_recovery_preserves_long_subdominant() -> None:
    key = KeyEstimate(tonic_pc=2, mode="major", confidence=0.82)
    segments = [
        ChordSegment(start=0.0, end=4.5, chord="D", confidence=0.88),
        ChordSegment(start=4.5, end=9.0, chord="F#m", confidence=0.86),
        ChordSegment(start=9.0, end=13.5, chord="G", confidence=0.82),
        ChordSegment(start=13.5, end=18.0, chord="A", confidence=0.84),
        ChordSegment(start=18.0, end=22.2, chord="G", confidence=0.83),
        ChordSegment(start=22.2, end=26.0, chord="A", confidence=0.84),
    ]

    refined, events = _apply_major_repeated_tonic_phrase_answer(segments, key)

    assert [segment.chord for segment in refined] == ["D", "F#m", "G", "A", "G", "A"]
    assert not events


def test_analysis_contract_shape_is_preserved(tmp_path: Path) -> None:
    samples = _chord([261.63, 329.63, 392.00], duration=1.2)

    tmp = tmp_path / "contract-shape-guard.wav"
    _write_wav(tmp, samples)
    result = analyze_audio(tmp).to_dict()

    assert set(result.keys()) == {"version", "source", "analysis"}
    assert set(result["source"].keys()) == {"path", "duration", "sampleRate"}  # type: ignore[index]
    assert set(result["analysis"].keys()) == {"algorithm", "chords"}  # type: ignore[index]


def _mock_result(path: str, duration: float, segments: list[ChordSegment]) -> AnalysisResult:
    return AnalysisResult(
        version=CONTRACT_VERSION,
        source=SourceMetadata(path=path, duration=duration, sampleRate=TARGET_SAMPLE_RATE),
        analysis=AnalysisMetadata(algorithm=ALGORITHM_ID, chords=segments),
    )


def _region_observation(
    *,
    start_frame: int,
    end_frame: int,
    winner: str,
    winner_conf: float,
    scores: dict[str, float],
    advantage: float,
) -> analyze_module.MusicalRegionObservation:
    frame_duration_seconds = 512 / 22050
    return analyze_module.MusicalRegionObservation(
        start_frame=start_frame,
        end_frame=end_frame,
        start_seconds=start_frame * frame_duration_seconds,
        end_seconds=end_frame * frame_duration_seconds,
        duration_seconds=(end_frame - start_frame) * frame_duration_seconds,
        winner_chord=winner,
        winner_confidence=winner_conf,
        scores=scores,
        advantage_over_runner_up=advantage,
    )


def _compressed_labels(predictions: list[analyze_module.FrameChordPrediction]) -> list[str]:
    if not predictions:
        return []
    labels = [predictions[0].chord]
    for pred in predictions[1:]:
        if pred.chord != labels[-1]:
            labels.append(pred.chord)
    return labels


def _run_improved(path: Path) -> PipelineRun:
    return analyze_module._run_pipeline(
        path,
        use_harmonic_preprocessing=True,
        use_beat_sync=True,
        use_key_prior=True,
        use_context_correction=False,
    )


def _write_profile_region(target: np.ndarray, start: int, end: int, pcs: list[int], scale: float = 1.0) -> None:
    for pc in pcs:
        target[pc, start:end] = scale


def test_benchmark_diagnostics_calculations_are_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline_segments = [
        ChordSegment(start=0.0, end=2.0, chord="C", confidence=0.9),
        ChordSegment(start=2.0, end=4.0, chord="G", confidence=0.88),
    ]
    improved_segments = [
        ChordSegment(start=0.0, end=0.2, chord="D#", confidence=0.4),
        ChordSegment(start=0.2, end=1.2, chord="C", confidence=0.9),
        ChordSegment(start=1.2, end=1.5, chord="G", confidence=0.8),
        ChordSegment(start=1.5, end=3.0, chord="Am", confidence=0.85),
        ChordSegment(start=3.0, end=4.0, chord="N", confidence=1.0),
    ]

    baseline_run = PipelineRun(
        result=_mock_result("/tmp/mock.wav", duration=4.0, segments=baseline_segments),
        detected_key=KeyEstimate(tonic_pc=0, mode="major", confidence=0.8),
    )
    improved_run = PipelineRun(
        result=_mock_result("/tmp/mock.wav", duration=4.0, segments=improved_segments),
        detected_key=KeyEstimate(tonic_pc=0, mode="major", confidence=0.77),
        persistence_events=[
            analyze_module.PersistenceDecisionEvent(
                region_index=1,
                region_start_seconds=0.2,
                region_end_seconds=1.2,
                action="keep",
                current_chord="C",
                candidate_chord="D#",
                keep_score=0.62,
                switch_score=0.41,
                advantage=0.08,
                candidate_confidence=0.40,
                consecutive_support=1,
                neighbor_persistence=0.30,
                harmonic_change_evidence=-0.12,
                is_root_preserving_quality_switch=False,
                accepted_with_lower_switch_score=False,
                reason="keep-insufficient-switch-evidence",
            ),
            analyze_module.PersistenceDecisionEvent(
                region_index=2,
                region_start_seconds=1.2,
                region_end_seconds=1.5,
                action="switch",
                current_chord="G",
                candidate_chord="Gm",
                keep_score=0.60,
                switch_score=0.58,
                advantage=0.20,
                candidate_confidence=0.74,
                consecutive_support=3,
                neighbor_persistence=0.95,
                harmonic_change_evidence=0.24,
                is_root_preserving_quality_switch=True,
                accepted_with_lower_switch_score=True,
                reason="exceptional-multibeat-context-override",
            ),
            analyze_module.PersistenceDecisionEvent(
                region_index=3,
                region_start_seconds=1.5,
                region_end_seconds=3.0,
                action="switch",
                current_chord="Gm",
                candidate_chord="Am",
                keep_score=0.44,
                switch_score=0.73,
                advantage=0.35,
                candidate_confidence=0.85,
                consecutive_support=2,
                neighbor_persistence=0.86,
                harmonic_change_evidence=0.31,
                is_root_preserving_quality_switch=False,
                accepted_with_lower_switch_score=False,
                reason="dominant-override",
            ),
        ],
    )

    def fake_run_pipeline(
        _path: str | Path,
        *,
        use_harmonic_preprocessing: bool,
        use_beat_sync: bool,
        use_key_prior: bool,
        use_context_correction: bool,
    ) -> PipelineRun:
        if use_harmonic_preprocessing and use_beat_sync and use_key_prior and use_context_correction:
            return improved_run
        if use_harmonic_preprocessing and use_beat_sync and use_key_prior and not use_context_correction:
            return improved_run
        return baseline_run

    monkeypatch.setattr(analyze_module, "_run_pipeline", fake_run_pipeline)

    comparison = compare_baseline_vs_improved("/tmp/mock.wav")
    assert "improvedBeforeContextCorrection" in comparison
    assert "transitionComparison" in comparison
    assert "musicalPersistence" in comparison
    assert "contextCorrection" in comparison
    improved = comparison["improved"]

    assert improved["algorithm"] == ALGORITHM_ID
    assert improved["globalKey"] == "C"
    assert float(improved["globalKeyConfidence"]) == pytest.approx(0.77, abs=1e-10)
    assert int(improved["segmentCount"]) == 5
    assert float(improved["songDuration"]) == pytest.approx(4.0, abs=1e-10)
    assert float(improved["meanSegmentDuration"]) == pytest.approx(0.8, abs=1e-10)
    assert float(improved["medianSegmentDuration"]) == pytest.approx(1.0, abs=1e-10)
    assert float(improved["minSegmentDuration"]) == pytest.approx(0.2, abs=1e-10)
    assert float(improved["maxSegmentDuration"]) == pytest.approx(1.5, abs=1e-10)

    short = improved["shortSegments"]
    assert short["lt250ms"]["count"] == 1
    assert float(short["lt250ms"]["percentage"]) == pytest.approx(20.0, abs=1e-10)
    assert short["lt500ms"]["count"] == 2
    assert float(short["lt500ms"]["percentage"]) == pytest.approx(40.0, abs=1e-10)
    assert short["lt1s"]["count"] == 2
    assert float(short["lt1s"]["percentage"]) == pytest.approx(40.0, abs=1e-10)

    counts = improved["chordOccurrenceCount"]
    durations = improved["chordDurationSeconds"]
    percentages = improved["chordDurationPercentageOfSong"]
    assert counts["C"] == 1
    assert counts["G"] == 1
    assert counts["Am"] == 1
    assert counts["D#"] == 1
    assert counts["N"] == 1
    assert float(durations["Am"]) == pytest.approx(1.5, abs=1e-10)
    assert float(percentages["Am"]) == pytest.approx(37.5, abs=1e-10)

    diatonicity = improved["diatonicity"]
    assert diatonicity["referenceKey"] == "C"
    assert diatonicity["diatonicSegmentCount"] == 3
    assert diatonicity["nonDiatonicSegmentCount"] == 1
    assert float(diatonicity["diatonicDuration"]) == pytest.approx(2.8, abs=1e-10)
    assert float(diatonicity["nonDiatonicDuration"]) == pytest.approx(0.2, abs=1e-10)

    suspicious = improved["suspiciousShortNonDiatonicSegments"]
    assert len(suspicious) == 1
    assert suspicious[0]["chord"] == "D#"
    assert float(suspicious[0]["duration"]) == pytest.approx(0.2, abs=1e-10)
    assert float(suspicious[0]["confidence"]) == pytest.approx(0.4, abs=1e-10)

    transition_comparison = comparison["transitionComparison"]
    assert transition_comparison["baselineTransitionCount"] == 1
    assert transition_comparison["improvedTransitionCount"] == 4
    assert float(transition_comparison["baselineTransitionsPerMinute"]) == pytest.approx(15.0, abs=1e-10)
    assert float(transition_comparison["improvedTransitionsPerMinute"]) == pytest.approx(60.0, abs=1e-10)

    persistence = comparison["musicalPersistence"]
    assert "acceptedTransitionCount" in persistence
    assert "rejectedTransitionCount" in persistence

    harmonic = comparison["harmonicSegmentation"]
    assert {
        "localNoveltyCandidateCount",
        "contextualBoundaryCandidateCount",
        "multiResolutionAcceptedBoundaryCount",
        "rejectedLocalOnlyBoundaryCount",
        "shortMediumAgreementRate",
        "meanShortContextDistance",
        "meanMediumContextDistance",
        "acceptedBoundaryExamples",
        "rejectedLocalOnlyExamples",
        "harmonicBoundaryCount",
        "harmonicRegionsPerMinute",
        "meanHarmonicRegionDuration",
        "medianHarmonicRegionDuration",
        "rejectedNoveltyPeaks",
        "acceptedBoundaryTimestamps",
        "perRegionSelectedChord",
        "localSwitchNoBoundaryExamples",
        "preservedRealChangeExamples",
    }.issubset(set(harmonic.keys()))

    global_decoding = comparison["globalDecoding"]
    assert {
        "globalDecoderPathScore",
        "regionDecisionsChangedByGlobalDecoding",
        "localVsGlobal",
    }.issubset(set(global_decoding.keys()))


def test_real_song_benchmark_before_after_metrics_if_local_file_provided() -> None:
    benchmark_path = os.getenv("GCD_REAL_SONG_PATH")
    if not benchmark_path:
        pytest.skip("GCD_REAL_SONG_PATH not set; skipping local real-song benchmark")

    path = Path(benchmark_path)
    if not path.exists() or not path.is_file():
        pytest.skip("GCD_REAL_SONG_PATH does not point to an existing file")

    comparison = compare_baseline_vs_improved(path)

    baseline = comparison["baseline"]
    improved_before = comparison["improvedBeforeContextCorrection"]
    improved = comparison["improved"]
    transition_comparison = comparison["transitionComparison"]
    musical_persistence = comparison["musicalPersistence"]

    required_keys = {
        "algorithm",
        "contractVersion",
        "segmentCount",
        "transitionCount",
        "rootChangeCount",
        "qualityChangeCount",
        "ambiguousQualityDecisionCount",
        "transitionsPerMinute",
        "estimatedTempoBpm",
        "beatCount",
        "beatReliable",
        "transitionsPerBeat",
        "songDuration",
        "meanSegmentDuration",
        "medianSegmentDuration",
        "minSegmentDuration",
        "maxSegmentDuration",
        "excessiveShortSegmentRate",
        "globalKey",
        "globalKeyConfidence",
        "shortSegments",
        "chordFamilyDistribution",
        "chordOccurrenceCount",
        "chordDurationSeconds",
        "chordDurationPercentageOfSong",
        "diatonicity",
        "suspiciousShortNonDiatonicSegments",
    }
    assert required_keys.issubset(set(baseline.keys()))
    assert required_keys.issubset(set(improved_before.keys()))
    assert required_keys.issubset(set(improved.keys()))

    assert set(transition_comparison.keys()) == {
        "baselineTransitionCount",
        "improvedTransitionCount",
        "baselineTransitionsPerMinute",
        "improvedTransitionsPerMinute",
    }
    assert {
        "decisionCount",
        "acceptedTransitionCount",
        "rejectedTransitionCount",
        "acceptedTransitionReasonCounts",
        "rootPreservingQualitySwitchCount",
        "weakAcceptedOverrideCount",
        "ambiguousQualityDecisionCount",
        "rejectedTransitions",
        "acceptedTransitions",
        "rootAwareRegionExamples",
    }.issubset(set(musical_persistence.keys()))

    assert "harmonicSegmentation" in comparison
    assert "globalDecoding" in comparison
    assert {
        "localNoveltyCandidateCount",
        "contextualBoundaryCandidateCount",
        "multiResolutionAcceptedBoundaryCount",
        "rejectedLocalOnlyBoundaryCount",
        "shortMediumAgreementRate",
        "meanShortContextDistance",
        "meanMediumContextDistance",
        "acceptedBoundaryExamples",
        "rejectedLocalOnlyExamples",
        "candidateBoundaryCount",
        "acceptedBoundaryCountBeforeConsolidation",
        "acceptedBoundaryCountAfterConsolidation",
        "consolidatedBoundaryCount",
        "boundaryClusterCount",
        "meanBeatsBetweenAcceptedBoundaries",
        "medianBeatsBetweenAcceptedBoundaries",
        "shortHarmonicRegionCount",
        "harmonicRegionDurationDistribution",
        "consolidatedClusterExamples",
        "harmonicBoundaryCount",
        "harmonicRegionsPerMinute",
        "meanHarmonicRegionDuration",
        "medianHarmonicRegionDuration",
        "rejectedNoveltyPeaks",
        "acceptedBoundaryTimestamps",
        "perRegionSelectedChord",
        "localSwitchNoBoundaryExamples",
        "preservedRealChangeExamples",
    }.issubset(set(comparison["harmonicSegmentation"].keys()))
    assert {
        "globalDecoderPathScore",
        "regionDecisionsChangedByGlobalDecoding",
        "localVsGlobal",
    }.issubset(set(comparison["globalDecoding"].keys()))

    baseline_short = float(baseline["excessiveShortSegmentRate"])
    improved_short = float(improved["excessiveShortSegmentRate"])
    baseline_segments = int(baseline["segmentCount"])
    improved_segments = int(improved["segmentCount"])

    # Material reduction: short-churn or segment over-fragmentation should improve.
    if baseline_short >= 0.05 or baseline_segments >= 20:
        assert (
            improved_short <= baseline_short * 0.9
            or improved_short <= max(0.0, baseline_short - 0.05)
            or improved_segments <= max(1, int(round(baseline_segments * 0.9)))
        )
    else:
        # If baseline is already stable, improved output must remain non-regressive.
        assert improved_short <= baseline_short + 1e-9
