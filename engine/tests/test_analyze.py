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
