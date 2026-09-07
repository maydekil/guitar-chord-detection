from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import chord_engine.analyze as analyze_module
from chord_engine.audio import TARGET_SAMPLE_RATE
from chord_engine.detector import KeyEstimate
from analyze_test_helpers import chord as _chord
from analyze_test_helpers import run_improved as _run_improved
from analyze_test_helpers import tone as _tone
from analyze_test_helpers import write_profile_region as _write_profile_region
from analyze_test_helpers import write_wav as _write_wav

def test_harmonic_change_point_sustained_a_with_moving_melody_stays_single_region(tmp_path: Path) -> None:
    base = _chord([110.0, 138.59, 164.81], duration=2.4)
    melody = np.concatenate(
        [
            _tone(440.0, duration=0.6, amp=0.20),
            _tone(493.88, duration=0.6, amp=0.20),
            _tone(554.37, duration=0.6, amp=0.20),
            _tone(659.25, duration=0.6, amp=0.20),
        ]
    ).astype(np.float32)
    wav = tmp_path / "a_melody_change_point.wav"
    _write_wav(wav, (base + melody).astype(np.float32))

    run = _run_improved(wav)
    assert run.region_observations is not None
    assert len(run.region_observations) == 1
    assert run.region_observations[0].decoded_chord == "A"


def test_harmonic_change_point_sustained_a_with_moving_bass_has_no_false_boundary(tmp_path: Path) -> None:
    upper = _chord([220.0, 277.18, 329.63], duration=2.4)
    moving_bass = np.concatenate(
        [
            _tone(110.0, duration=0.8, amp=0.26),
            _tone(98.0, duration=0.8, amp=0.26),
            _tone(82.41, duration=0.8, amp=0.26),
        ]
    ).astype(np.float32)
    wav = tmp_path / "a_moving_bass_change_point.wav"
    _write_wav(wav, (upper + moving_bass).astype(np.float32))

    run = _run_improved(wav)
    assert run.accepted_boundaries is not None
    assert len(run.accepted_boundaries) == 0
    assert run.region_observations is not None
    assert len(run.region_observations) == 1
    assert run.region_observations[0].decoded_chord == "A"


def test_harmonic_change_point_sustained_a_with_arpeggiated_voicing_has_no_false_boundary(tmp_path: Path) -> None:
    root = _tone(110.0, duration=2.4, amp=0.18)
    arpeggio = np.concatenate(
        [
            _tone(220.0, duration=0.3, amp=0.12),
            _tone(277.18, duration=0.3, amp=0.12),
            _tone(329.63, duration=0.3, amp=0.12),
            _tone(440.0, duration=0.3, amp=0.10),
            _tone(329.63, duration=0.3, amp=0.12),
            _tone(277.18, duration=0.3, amp=0.12),
            _tone(220.0, duration=0.3, amp=0.12),
            _tone(329.63, duration=0.3, amp=0.11),
        ]
    ).astype(np.float32)
    wav = tmp_path / "a_arpeggio_voicing_change_point.wav"
    _write_wav(wav, (root + arpeggio).astype(np.float32))

    run = _run_improved(wav)
    assert run.accepted_boundaries is not None
    assert len(run.accepted_boundaries) == 0
    assert run.region_observations is not None
    assert len(run.region_observations) == 1
    assert run.region_observations[0].decoded_chord == "A"


def test_harmonic_change_point_passing_tones_do_not_create_boundary(tmp_path: Path) -> None:
    sustained_c = _chord([130.81, 164.81, 196.0], duration=2.2)
    passing = np.concatenate(
        [
            _tone(293.66, duration=0.2, amp=0.16),
            _tone(311.13, duration=0.2, amp=0.16),
            _tone(329.63, duration=0.2, amp=0.16),
            _tone(311.13, duration=0.2, amp=0.16),
            _tone(293.66, duration=0.2, amp=0.16),
            _tone(261.63, duration=0.2, amp=0.16),
            _tone(246.94, duration=0.2, amp=0.16),
            _tone(261.63, duration=0.2, amp=0.16),
            _tone(277.18, duration=0.2, amp=0.16),
            _tone(293.66, duration=0.2, amp=0.16),
            _tone(311.13, duration=0.2, amp=0.16),
        ]
    ).astype(np.float32)
    wav = tmp_path / "passing_tones_no_boundary.wav"
    _write_wav(wav, (sustained_c + passing).astype(np.float32))

    run = _run_improved(wav)
    assert run.region_observations is not None
    assert len(run.region_observations) == 1


def test_harmonic_change_point_genuine_a_to_e_creates_boundary(tmp_path: Path) -> None:
    a = _chord([110.0, 138.59, 164.81], duration=1.4)
    e = _chord([164.81, 207.65, 246.94], duration=1.4)
    wav = tmp_path / "a_to_e_boundary.wav"
    _write_wav(wav, np.concatenate([a, e]).astype(np.float32))

    run = _run_improved(wav)
    assert run.region_observations is not None
    labels = [obs.decoded_chord for obs in run.region_observations]
    assert labels[:2] == ["A", "E"]
    assert run.accepted_boundaries is not None
    assert len(run.accepted_boundaries) >= 1


def test_harmonic_change_point_progression_c_g_am_f_produces_four_regions(tmp_path: Path) -> None:
    c = _chord([261.63, 329.63, 392.00], duration=1.3)
    g = _chord([196.00, 246.94, 293.66], duration=1.3)
    am = _chord([220.00, 261.63, 329.63], duration=1.3)
    f = _chord([174.61, 220.00, 261.63], duration=1.3)
    wav = tmp_path / "cgamf_regions.wav"
    _write_wav(wav, np.concatenate([c, g, am, f]).astype(np.float32))

    run = _run_improved(wav)
    assert run.region_observations is not None
    labels = [obs.decoded_chord for obs in run.region_observations]
    assert labels == ["C", "G", "Am", "F"]


def test_harmonic_change_point_rapid_two_chords_survive_with_strong_evidence(tmp_path: Path) -> None:
    c = _chord([261.63, 329.63, 392.00], duration=0.9)
    g = _chord([196.00, 246.94, 293.66], duration=0.9)
    wav = tmp_path / "rapid_c_to_g.wav"
    _write_wav(wav, np.concatenate([c, g]).astype(np.float32))

    run = _run_improved(wav)
    assert run.region_observations is not None
    labels = [obs.decoded_chord for obs in run.region_observations]
    assert labels[:2] == ["C", "G"]


def test_harmonic_change_point_same_root_quality_change_can_create_boundary(tmp_path: Path) -> None:
    e_major = _chord([164.81, 207.65, 246.94], duration=1.4)
    e_minor = _chord([164.81, 196.0, 246.94], duration=1.4)
    wav = tmp_path / "e_quality_boundary.wav"
    _write_wav(wav, np.concatenate([e_major, e_minor]).astype(np.float32))

    run = _run_improved(wav)
    assert run.region_observations is not None
    labels = [obs.decoded_chord for obs in run.region_observations]
    assert labels[:2] == ["E", "Em"]


def test_harmonic_change_point_same_root_quality_change_a_to_am_remains_detectable(tmp_path: Path) -> None:
    a_major = _chord([110.0, 138.59, 164.81], duration=1.5)
    a_minor = _chord([110.0, 130.81, 164.81], duration=1.5)
    wav = tmp_path / "a_quality_boundary.wav"
    _write_wav(wav, np.concatenate([a_major, a_minor]).astype(np.float32))

    run = _run_improved(wav)
    assert run.region_observations is not None
    labels = [obs.decoded_chord for obs in run.region_observations]
    assert labels[:2] == ["A", "Am"]


def test_harmonic_change_point_silence_remains_n(tmp_path: Path) -> None:
    wav = tmp_path / "change_point_silence.wav"
    _write_wav(wav, np.zeros(int(TARGET_SAMPLE_RATE * 1.6), dtype=np.float32))

    run = _run_improved(wav)
    assert run.result.analysis.chords
    assert all(seg.chord == "N" for seg in run.result.analysis.chords)


def test_harmonic_change_point_pipeline_is_deterministic(tmp_path: Path) -> None:
    c = _chord([261.63, 329.63, 392.00], duration=1.2)
    g = _chord([196.00, 246.94, 293.66], duration=1.2)
    wav = tmp_path / "change_point_deterministic.wav"
    _write_wav(wav, np.concatenate([c, g]).astype(np.float32))

    first = _run_improved(wav)
    second = _run_improved(wav)

    assert first.accepted_boundaries == second.accepted_boundaries
    first_labels = [obs.decoded_chord for obs in (first.region_observations or [])]
    second_labels = [obs.decoded_chord for obs in (second.region_observations or [])]
    assert first_labels == second_labels
    assert first.result.to_dict() == second.result.to_dict()


def test_boundary_consolidation_collapses_clustered_peaks_around_single_change() -> None:
    beat_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    novelty = [0.08, 0.36, 0.34, 0.07]
    selected = [1, 2]
    candidate_peaks = [
        {"index": 1, "prominence": 0.11},
        {"index": 2, "prominence": 0.10},
    ]
    beat_profiles = [
        np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
        np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
        np.array([0.2, 0.1, 0.0, 0.2, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.0, 0.1], dtype=np.float32),
        np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
        np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
    ]
    chroma = np.zeros((12, 50), dtype=np.float32)
    low = np.zeros((12, 50), dtype=np.float32)
    _write_profile_region(chroma, 0, 20, [0, 4, 7], 1.0)
    _write_profile_region(chroma, 20, 30, [1, 3, 8], 0.12)
    _write_profile_region(chroma, 30, 50, [0, 4, 7], 1.0)
    _write_profile_region(low, 0, 50, [0, 7], 0.5)

    consolidated, diag = analyze_module._consolidate_boundary_clusters(
        selected,
        novelty=novelty,
        candidate_peaks=candidate_peaks,
        beat_profiles=beat_profiles,
        beat_ranges=beat_ranges,
        beat_local_chords=["C", "C", "D#", "C", "C"],
        hop_length=512,
        sample_rate=22050,
        chroma=chroma,
        low_chroma=low,
        key_estimate=KeyEstimate(tonic_pc=0, mode="major", confidence=0.7),
        beat_reliable=True,
    )

    assert len(consolidated) == 1
    assert int(diag["boundaryClusterCount"]) >= 1


def test_multi_resolution_passing_tone_local_novelty_is_rejected_by_medium_context() -> None:
    beat_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
    # One sharp local excursion around index 2 but stable medium context overall.
    beat_profiles = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.3, 0.0, 0.0, 0.0, 0.2, 0.4, 0.0, 0.2, 0.4, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ]
    novelty = analyze_module._compute_harmonic_novelty(beat_profiles)

    selected, rejected, _, diag = analyze_module._select_harmonic_boundaries(
        novelty,
        beat_profiles,
        beat_ranges,
        ["C", "C", "Dm", "C", "C", "C"],
        hop_length=512,
        sample_rate=22050,
    )

    assert selected == []
    assert int(diag["rejectedLocalOnlyBoundaryCount"]) >= 1
    assert len(rejected) >= 1
    assert any("local-only-medium-insufficient" in str(item.get("reason", "")) for item in rejected)


def test_multi_resolution_rapid_changes_preserved_when_short_and_medium_agree() -> None:
    beat_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    beat_profiles = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.8], dtype=np.float32),
        np.array([0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0], dtype=np.float32),
        np.array([0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0], dtype=np.float32),
        np.array([0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0], dtype=np.float32),
    ]
    novelty = analyze_module._compute_harmonic_novelty(beat_profiles)

    selected, _, _, diag = analyze_module._select_harmonic_boundaries(
        novelty,
        beat_profiles,
        beat_ranges,
        ["C", "G", "Am", "Am", "Am"],
        hop_length=512,
        sample_rate=22050,
    )

    assert selected == [0, 1]
    assert int(diag["multiResolutionAcceptedBoundaryCount"]) == 2
    assert float(diag["shortMediumAgreementRate"]) > 0.0


def test_boundary_consolidation_removes_transient_peak_near_real_boundary(tmp_path: Path) -> None:
    a = _chord([110.0, 138.59, 164.81], duration=1.3)
    transient = _tone(440.0, duration=0.18, amp=0.22)
    e = _chord([164.81, 207.65, 246.94], duration=1.3)
    wav = tmp_path / "transient_near_boundary.wav"
    _write_wav(wav, np.concatenate([a, transient, e]).astype(np.float32))

    run = _run_improved(wav)
    assert run.accepted_boundary_count_before_consolidation >= run.accepted_boundary_count_after_consolidation
    assert run.consolidated_boundary_count >= 0
    assert run.region_observations is not None
    labels = [obs.decoded_chord for obs in run.region_observations]
    assert labels[0] == "A"
    assert "E" in labels


def test_boundary_consolidation_collapses_short_weak_intermediate_region() -> None:
    beat_ranges = [(0, 10), (10, 20), (20, 30), (30, 40)]
    novelty = [0.06, 0.31, 0.30]
    selected = [1, 2]
    candidate_peaks = [
        {"index": 1, "prominence": 0.09},
        {"index": 2, "prominence": 0.08},
    ]
    beat_profiles = [
        np.array([0, 0, 0, 0, 0.2, 0, 1, 0, 0, 0.8, 0, 0], dtype=np.float32),
        np.array([0, 0, 0, 0, 0.2, 0, 1, 0, 0, 0.8, 0, 0], dtype=np.float32),
        np.array([0.1] * 12, dtype=np.float32),
        np.array([0, 0, 0, 0, 0.2, 0, 1, 0, 0, 0.8, 0, 0], dtype=np.float32),
    ]
    chroma = np.zeros((12, 40), dtype=np.float32)
    low = np.zeros((12, 40), dtype=np.float32)
    _write_profile_region(chroma, 0, 20, [6, 9, 1], 1.0)
    _write_profile_region(chroma, 20, 30, [0, 3, 7], 0.1)
    _write_profile_region(chroma, 30, 40, [6, 9, 1], 1.0)
    _write_profile_region(low, 0, 40, [6, 1], 0.4)

    consolidated, _ = analyze_module._consolidate_boundary_clusters(
        selected,
        novelty=novelty,
        candidate_peaks=candidate_peaks,
        beat_profiles=beat_profiles,
        beat_ranges=beat_ranges,
        beat_local_chords=["F#m", "F#m", "C", "F#m"],
        hop_length=512,
        sample_rate=22050,
        chroma=chroma,
        low_chroma=low,
        key_estimate=KeyEstimate(tonic_pc=6, mode="minor", confidence=0.75),
        beat_reliable=True,
    )

    assert len(consolidated) == 1


def test_boundary_consolidation_preserves_two_rapid_strong_changes() -> None:
    beat_ranges = [(0, 10), (10, 20), (20, 30), (30, 40)]
    novelty = [0.08, 0.33, 0.35]
    selected = [1, 2]
    candidate_peaks = [
        {"index": 1, "prominence": 0.10},
        {"index": 2, "prominence": 0.11},
    ]
    beat_profiles = [
        np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
        np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32),
        np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32),
        np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32),
    ]
    chroma = np.zeros((12, 40), dtype=np.float32)
    low = np.zeros((12, 40), dtype=np.float32)
    _write_profile_region(chroma, 0, 20, [0, 4, 7], 1.0)
    _write_profile_region(chroma, 20, 30, [7, 11, 2], 1.0)
    _write_profile_region(chroma, 30, 40, [9, 0, 4], 1.0)
    _write_profile_region(low, 0, 40, [0, 7, 9], 0.45)

    consolidated, diag = analyze_module._consolidate_boundary_clusters(
        selected,
        novelty=novelty,
        candidate_peaks=candidate_peaks,
        beat_profiles=beat_profiles,
        beat_ranges=beat_ranges,
        beat_local_chords=["C", "G", "Am", "Am"],
        hop_length=512,
        sample_rate=22050,
        chroma=chroma,
        low_chroma=low,
        key_estimate=KeyEstimate(tonic_pc=0, mode="major", confidence=0.8),
        beat_reliable=True,
    )

    assert consolidated == [1, 2]
    assert int(diag["boundaryClusterCount"]) >= 1
