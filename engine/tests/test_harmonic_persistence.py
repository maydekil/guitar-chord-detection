from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import chord_engine.analyze as analyze_module
from chord_engine.analyze import analyze_audio
from chord_engine.audio import TARGET_SAMPLE_RATE
from chord_engine.detector import KeyEstimate
from chord_engine.segmentation import ChordSegment
from analyze_test_helpers import chord as _chord
from analyze_test_helpers import compressed_labels as _compressed_labels
from analyze_test_helpers import region_observation as _region_observation
from analyze_test_helpers import write_wav as _write_wav

def test_context_correction_fixes_weak_short_isolated_anomaly() -> None:
    segments = [
        ChordSegment(start=0.0, end=1.4, chord="F#m", confidence=0.82),
        ChordSegment(start=1.4, end=1.7, chord="C", confidence=0.41),
        ChordSegment(start=1.7, end=3.6, chord="F#m", confidence=0.86),
    ]
    key = KeyEstimate(tonic_pc=6, mode="minor", confidence=0.75)

    corrected, events = analyze_module._apply_context_aware_short_segment_correction(segments, key)

    assert len(events) == 1
    assert events[0].replaced_chord == "C"
    assert events[0].new_chord == "F#m"
    assert [seg.chord for seg in corrected] == ["F#m"]
    assert corrected[0].start == pytest.approx(0.0, abs=1e-10)
    assert corrected[0].end == pytest.approx(3.6, abs=1e-10)


def test_context_correction_keeps_short_genuine_strong_chord() -> None:
    segments = [
        ChordSegment(start=0.0, end=1.2, chord="Am", confidence=0.80),
        ChordSegment(start=1.2, end=1.5, chord="E", confidence=0.84),
        ChordSegment(start=1.5, end=3.0, chord="Am", confidence=0.83),
    ]
    key = KeyEstimate(tonic_pc=9, mode="minor", confidence=0.77)

    corrected, events = analyze_module._apply_context_aware_short_segment_correction(segments, key)

    assert len(events) == 0
    assert [seg.chord for seg in corrected] == ["Am", "E", "Am"]


def test_context_correction_preserves_secondary_dominant_in_minor() -> None:
    segments = [
        ChordSegment(start=0.0, end=1.3, chord="F#m", confidence=0.81),
        ChordSegment(start=1.3, end=1.7, chord="C#", confidence=0.71),
        ChordSegment(start=1.7, end=3.2, chord="F#m", confidence=0.84),
    ]
    key = KeyEstimate(tonic_pc=6, mode="minor", confidence=0.75)

    corrected, events = analyze_module._apply_context_aware_short_segment_correction(segments, key)

    assert len(events) == 0
    assert [seg.chord for seg in corrected] == ["F#m", "C#", "F#m"]


def test_context_correction_does_not_flatten_sustained_real_transition() -> None:
    segments = [
        ChordSegment(start=0.0, end=1.2, chord="C", confidence=0.84),
        ChordSegment(start=1.2, end=2.1, chord="G", confidence=0.52),
        ChordSegment(start=2.1, end=3.4, chord="Am", confidence=0.80),
    ]
    key = KeyEstimate(tonic_pc=0, mode="major", confidence=0.70)

    corrected, events = analyze_module._apply_context_aware_short_segment_correction(segments, key)

    assert len(events) == 0
    assert [seg.chord for seg in corrected] == ["C", "G", "Am"]


def test_musical_time_persistence_rejects_melody_like_single_region_switch() -> None:
    observations = [
        _region_observation(start_frame=0, end_frame=20, winner="C", winner_conf=0.86, scores={"C": 0.78, "Am": 0.12}, advantage=0.66),
        _region_observation(start_frame=20, end_frame=34, winner="Am", winner_conf=0.61, scores={"Am": 0.54, "C": 0.40}, advantage=0.14),
        _region_observation(start_frame=34, end_frame=54, winner="C", winner_conf=0.85, scores={"C": 0.79, "Am": 0.10}, advantage=0.69),
    ]

    predictions, events = analyze_module._apply_musical_time_harmonic_persistence(
        observations,
        n_frames=54,
        detected_key=KeyEstimate(tonic_pc=0, mode="major", confidence=0.75),
    )

    assert _compressed_labels(predictions) == ["C"]
    assert any(event.action == "keep" and event.candidate_chord == "Am" for event in events)


def test_musical_time_persistence_rejects_moving_bass_instability() -> None:
    observations = [
        _region_observation(start_frame=0, end_frame=18, winner="C", winner_conf=0.84, scores={"C": 0.74, "Em": 0.14}, advantage=0.60),
        _region_observation(start_frame=18, end_frame=30, winner="Em", winner_conf=0.63, scores={"Em": 0.52, "C": 0.43}, advantage=0.09),
        _region_observation(start_frame=30, end_frame=46, winner="C", winner_conf=0.85, scores={"C": 0.76, "Em": 0.13}, advantage=0.63),
        _region_observation(start_frame=46, end_frame=58, winner="F", winner_conf=0.62, scores={"F": 0.51, "C": 0.44}, advantage=0.07),
        _region_observation(start_frame=58, end_frame=76, winner="C", winner_conf=0.84, scores={"C": 0.75, "F": 0.14}, advantage=0.61),
    ]

    predictions, _ = analyze_module._apply_musical_time_harmonic_persistence(
        observations,
        n_frames=76,
        detected_key=KeyEstimate(tonic_pc=0, mode="major", confidence=0.70),
    )

    assert _compressed_labels(predictions) == ["C"]


def test_musical_time_persistence_rejects_short_passing_note_switch() -> None:
    observations = [
        _region_observation(start_frame=0, end_frame=22, winner="C", winner_conf=0.86, scores={"C": 0.80, "Dm": 0.10}, advantage=0.70),
        _region_observation(start_frame=22, end_frame=28, winner="Dm", winner_conf=0.58, scores={"Dm": 0.50, "C": 0.45}, advantage=0.05),
        _region_observation(start_frame=28, end_frame=50, winner="C", winner_conf=0.84, scores={"C": 0.77, "Dm": 0.12}, advantage=0.65),
    ]

    predictions, _ = analyze_module._apply_musical_time_harmonic_persistence(
        observations,
        n_frames=50,
        detected_key=KeyEstimate(tonic_pc=0, mode="major", confidence=0.72),
    )

    assert _compressed_labels(predictions) == ["C"]


def test_transition_acceptance_refinement_stable_e_major_does_not_flicker_to_em() -> None:
    observations = [
        _region_observation(start_frame=0, end_frame=20, winner="E", winner_conf=0.87, scores={"E": 0.80, "Em": 0.12}, advantage=0.68),
        _region_observation(start_frame=20, end_frame=34, winner="Em", winner_conf=0.69, scores={"Em": 0.55, "E": 0.41}, advantage=0.14),
        _region_observation(start_frame=34, end_frame=50, winner="E", winner_conf=0.86, scores={"E": 0.79, "Em": 0.13}, advantage=0.66),
    ]

    predictions, _ = analyze_module._apply_musical_time_harmonic_persistence(
        observations,
        n_frames=50,
        detected_key=KeyEstimate(tonic_pc=4, mode="major", confidence=0.78),
    )

    assert _compressed_labels(predictions) == ["E"]


def test_root_aware_weak_third_evidence_does_not_cause_quality_flicker() -> None:
    frame_duration_seconds = 512 / 22050
    observations = [
        analyze_module.MusicalRegionObservation(
            start_frame=0,
            end_frame=22,
            start_seconds=0.0,
            end_seconds=22 * frame_duration_seconds,
            duration_seconds=22 * frame_duration_seconds,
            winner_chord="E",
            winner_confidence=0.72,
            scores={"E": 0.58, "Em": 0.56},
            advantage_over_runner_up=0.02,
            is_quality_ambiguous=True,
        ),
        analyze_module.MusicalRegionObservation(
            start_frame=22,
            end_frame=44,
            start_seconds=22 * frame_duration_seconds,
            end_seconds=44 * frame_duration_seconds,
            duration_seconds=22 * frame_duration_seconds,
            winner_chord="Em",
            winner_confidence=0.71,
            scores={"Em": 0.57, "E": 0.56},
            advantage_over_runner_up=0.01,
            is_quality_ambiguous=True,
        ),
        analyze_module.MusicalRegionObservation(
            start_frame=44,
            end_frame=66,
            start_seconds=44 * frame_duration_seconds,
            end_seconds=66 * frame_duration_seconds,
            duration_seconds=22 * frame_duration_seconds,
            winner_chord="E",
            winner_confidence=0.73,
            scores={"E": 0.58, "Em": 0.55},
            advantage_over_runner_up=0.03,
            is_quality_ambiguous=True,
        ),
    ]

    predictions, events = analyze_module._apply_musical_time_harmonic_persistence(
        observations,
        n_frames=66,
        detected_key=KeyEstimate(tonic_pc=4, mode="major", confidence=0.74),
    )

    assert _compressed_labels(predictions) == ["E"]
    assert any(event.reason == "keep-quality-ambiguous" for event in events)


def test_analyze_genuine_major_to_minor_change_remains_detectable(tmp_path: Path) -> None:
    e_major = _chord([164.81, 207.65, 246.94], duration=1.4)
    e_minor = _chord([164.81, 196.00, 246.94], duration=1.4)
    wav = tmp_path / "e_major_to_e_minor.wav"
    _write_wav(wav, np.concatenate([e_major, e_minor]).astype(np.float32))

    result = analyze_audio(wav).to_dict()
    labels = [seg["chord"] for seg in result["analysis"]["chords"]]  # type: ignore[index]
    compressed = [labels[0]] if labels else []
    for lab in labels[1:]:
        if lab != compressed[-1]:
            compressed.append(lab)

    assert compressed[:2] == ["E", "Em"]


def test_transition_acceptance_refinement_stable_a_major_does_not_flicker_to_am() -> None:
    observations = [
        _region_observation(start_frame=0, end_frame=20, winner="A", winner_conf=0.86, scores={"A": 0.79, "Am": 0.13}, advantage=0.66),
        _region_observation(start_frame=20, end_frame=36, winner="Am", winner_conf=0.68, scores={"Am": 0.54, "A": 0.42}, advantage=0.12),
        _region_observation(start_frame=36, end_frame=54, winner="A", winner_conf=0.85, scores={"A": 0.78, "Am": 0.14}, advantage=0.64),
    ]

    predictions, _ = analyze_module._apply_musical_time_harmonic_persistence(
        observations,
        n_frames=54,
        detected_key=KeyEstimate(tonic_pc=9, mode="major", confidence=0.77),
    )

    assert _compressed_labels(predictions) == ["A"]


def test_transition_acceptance_refinement_preserves_genuine_e_to_em_quality_change() -> None:
    observations = [
        _region_observation(start_frame=0, end_frame=20, winner="E", winner_conf=0.85, scores={"E": 0.79, "Em": 0.12}, advantage=0.67),
        _region_observation(start_frame=20, end_frame=38, winner="Em", winner_conf=0.83, scores={"Em": 0.76, "E": 0.14}, advantage=0.62),
        _region_observation(start_frame=38, end_frame=56, winner="Em", winner_conf=0.84, scores={"Em": 0.79, "E": 0.10}, advantage=0.69),
    ]

    predictions, events = analyze_module._apply_musical_time_harmonic_persistence(
        observations,
        n_frames=56,
        detected_key=KeyEstimate(tonic_pc=4, mode="major", confidence=0.74),
    )

    assert _compressed_labels(predictions) == ["E", "Em"]
    assert any(event.action == "switch" and event.current_chord == "E" and event.candidate_chord == "Em" for event in events)


def test_transition_acceptance_refinement_preserves_genuine_c_sharp_minor_to_major_change() -> None:
    observations = [
        _region_observation(start_frame=0, end_frame=22, winner="C#m", winner_conf=0.84, scores={"C#m": 0.78, "C#": 0.11}, advantage=0.67),
        _region_observation(start_frame=22, end_frame=40, winner="C#", winner_conf=0.82, scores={"C#": 0.75, "C#m": 0.15}, advantage=0.60),
        _region_observation(start_frame=40, end_frame=58, winner="C#", winner_conf=0.84, scores={"C#": 0.78, "C#m": 0.12}, advantage=0.66),
    ]

    predictions, events = analyze_module._apply_musical_time_harmonic_persistence(
        observations,
        n_frames=58,
        detected_key=KeyEstimate(tonic_pc=6, mode="minor", confidence=0.76),
    )

    assert _compressed_labels(predictions) == ["C#m", "C#"]
    assert any(event.action == "switch" and event.current_chord == "C#m" and event.candidate_chord == "C#" for event in events)


def test_transition_acceptance_refinement_rejects_ordinary_override_when_switch_score_is_lower() -> None:
    observations = [
        _region_observation(start_frame=0, end_frame=24, winner="E", winner_conf=0.86, scores={"E": 0.80, "Em": 0.11}, advantage=0.69),
        _region_observation(start_frame=24, end_frame=40, winner="Em", winner_conf=0.70, scores={"Em": 0.56, "E": 0.39}, advantage=0.17),
        _region_observation(start_frame=40, end_frame=56, winner="Em", winner_conf=0.71, scores={"Em": 0.57, "E": 0.38}, advantage=0.19),
        _region_observation(start_frame=56, end_frame=72, winner="E", winner_conf=0.86, scores={"E": 0.79, "Em": 0.12}, advantage=0.67),
    ]

    predictions, events = analyze_module._apply_musical_time_harmonic_persistence(
        observations,
        n_frames=72,
        detected_key=KeyEstimate(tonic_pc=4, mode="major", confidence=0.78),
    )

    assert _compressed_labels(predictions) == ["E"]
    assert any(event.action == "keep" and event.candidate_chord == "Em" and event.switch_score <= event.keep_score for event in events)


def test_transition_acceptance_refinement_allows_explicit_exceptional_multibeat_override() -> None:
    observations = [
        _region_observation(start_frame=0, end_frame=34, winner="C", winner_conf=0.86, scores={"C": 0.82, "G": 0.10}, advantage=0.72),
        _region_observation(start_frame=34, end_frame=70, winner="G", winner_conf=0.69, scores={"G": 0.58, "C": 0.44}, advantage=0.14),
        _region_observation(start_frame=70, end_frame=106, winner="G", winner_conf=0.69, scores={"G": 0.59, "C": 0.43}, advantage=0.16),
        _region_observation(start_frame=106, end_frame=142, winner="G", winner_conf=0.71, scores={"G": 0.60, "C": 0.42}, advantage=0.18),
        _region_observation(start_frame=142, end_frame=178, winner="G", winner_conf=0.72, scores={"G": 0.62, "C": 0.40}, advantage=0.22),
    ]

    predictions, events = analyze_module._apply_musical_time_harmonic_persistence(
        observations,
        n_frames=178,
        detected_key=KeyEstimate(tonic_pc=0, mode="major", confidence=0.72),
    )

    assert _compressed_labels(predictions) == ["C", "G"]
    assert any(
        event.action == "switch"
        and event.candidate_chord == "G"
        and event.reason == "exceptional-multibeat-context-override"
        and event.switch_score <= event.keep_score
        for event in events
    )


def test_transition_acceptance_refinement_no_nonexceptional_weak_switch_acceptance() -> None:
    observations = [
        _region_observation(start_frame=0, end_frame=20, winner="A", winner_conf=0.85, scores={"A": 0.80, "Am": 0.10}, advantage=0.70),
        _region_observation(start_frame=20, end_frame=40, winner="Am", winner_conf=0.71, scores={"Am": 0.56, "A": 0.36}, advantage=0.20),
        _region_observation(start_frame=40, end_frame=60, winner="Am", winner_conf=0.72, scores={"Am": 0.57, "A": 0.35}, advantage=0.22),
        _region_observation(start_frame=60, end_frame=80, winner="Am", winner_conf=0.73, scores={"Am": 0.58, "A": 0.34}, advantage=0.24),
        _region_observation(start_frame=80, end_frame=100, winner="Am", winner_conf=0.74, scores={"Am": 0.59, "A": 0.33}, advantage=0.26),
    ]

    _, events = analyze_module._apply_musical_time_harmonic_persistence(
        observations,
        n_frames=100,
        detected_key=KeyEstimate(tonic_pc=9, mode="major", confidence=0.73),
    )

    for event in events:
        if event.action == "switch" and event.switch_score <= event.keep_score:
            assert event.reason == "exceptional-multibeat-context-override"


def test_musical_time_persistence_preserves_genuine_c_to_g_transition() -> None:
    observations = [
        _region_observation(start_frame=0, end_frame=18, winner="C", winner_conf=0.84, scores={"C": 0.76, "G": 0.12}, advantage=0.64),
        _region_observation(start_frame=18, end_frame=36, winner="G", winner_conf=0.81, scores={"G": 0.70, "C": 0.22}, advantage=0.48),
        _region_observation(start_frame=36, end_frame=54, winner="G", winner_conf=0.83, scores={"G": 0.75, "C": 0.16}, advantage=0.59),
    ]

    predictions, events = analyze_module._apply_musical_time_harmonic_persistence(
        observations,
        n_frames=54,
        detected_key=KeyEstimate(tonic_pc=0, mode="major", confidence=0.75),
    )

    assert _compressed_labels(predictions) == ["C", "G"]
    assert any(event.action == "switch" and event.candidate_chord == "G" for event in events)


def test_musical_time_persistence_allows_short_strong_genuine_chord() -> None:
    observations = [
        _region_observation(start_frame=0, end_frame=24, winner="C", winner_conf=0.86, scores={"C": 0.78, "G": 0.12}, advantage=0.66),
        _region_observation(start_frame=24, end_frame=40, winner="G", winner_conf=0.86, scores={"G": 0.76, "C": 0.18}, advantage=0.58),
        _region_observation(start_frame=40, end_frame=56, winner="C", winner_conf=0.85, scores={"C": 0.74, "G": 0.20}, advantage=0.54),
        _region_observation(start_frame=56, end_frame=72, winner="C", winner_conf=0.86, scores={"C": 0.77, "G": 0.16}, advantage=0.61),
    ]

    predictions, _ = analyze_module._apply_musical_time_harmonic_persistence(
        observations,
        n_frames=72,
        detected_key=KeyEstimate(tonic_pc=0, mode="major", confidence=0.73),
    )

    assert _compressed_labels(predictions) == ["C", "G", "C"]


def test_musical_time_persistence_is_deterministic() -> None:
    observations = [
        _region_observation(start_frame=0, end_frame=20, winner="C", winner_conf=0.83, scores={"C": 0.72, "G": 0.16}, advantage=0.56),
        _region_observation(start_frame=20, end_frame=34, winner="G", winner_conf=0.64, scores={"G": 0.56, "C": 0.34}, advantage=0.22),
        _region_observation(start_frame=34, end_frame=50, winner="G", winner_conf=0.68, scores={"G": 0.61, "C": 0.29}, advantage=0.32),
    ]

    first_predictions, first_events = analyze_module._apply_musical_time_harmonic_persistence(
        observations,
        n_frames=50,
        detected_key=KeyEstimate(tonic_pc=0, mode="major", confidence=0.74),
    )
    second_predictions, second_events = analyze_module._apply_musical_time_harmonic_persistence(
        observations,
        n_frames=50,
        detected_key=KeyEstimate(tonic_pc=0, mode="major", confidence=0.74),
    )

    assert [pred.chord for pred in first_predictions] == [pred.chord for pred in second_predictions]
    assert [round(pred.confidence, 10) for pred in first_predictions] == [round(pred.confidence, 10) for pred in second_predictions]
    assert first_events == second_events

