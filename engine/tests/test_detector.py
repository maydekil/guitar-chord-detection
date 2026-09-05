from __future__ import annotations

import numpy as np
import pytest

from chord_engine.audio import AudioBuffer, TARGET_SAMPLE_RATE
from chord_engine.detector import (
    NO_CHORD_LABEL,
    FrameDetectionError,
    KeyEstimate,
    estimate_global_key,
    predict_frame_chord,
)
from chord_engine.features import extract_chroma
from chord_engine.templates import generate_chord_templates


def _tone(freq: float, duration: float = 1.5, amp: float = 0.45, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _chord(freqs: list[float], duration: float = 1.5, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    stacked = np.stack([_tone(f, duration=duration, sr=sr) for f in freqs], axis=0)
    return np.mean(stacked, axis=0).astype(np.float32)


def _audio_buffer(samples: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> AudioBuffer:
    samples = np.asarray(samples, dtype=np.float32)
    return AudioBuffer(samples=samples, sample_rate=sr, duration=float(samples.shape[0] / sr))


def _representative_frame_from_audio(samples: np.ndarray) -> np.ndarray:
    features = extract_chroma(_audio_buffer(samples))
    frame_energy = np.linalg.norm(features.chroma, axis=0)
    idx = int(np.argmax(frame_energy))
    return features.chroma[:, idx].astype(np.float32)


def _pc_to_freq(pc: int, octave: int) -> float:
    midi = 12 * (octave + 1) + pc
    return float(440.0 * (2.0 ** ((midi - 69) / 12.0)))


def _triad_freqs(root_pc: int, *, minor: bool, root_octave: int = 3) -> list[float]:
    third_interval = 3 if minor else 4
    return [
        _pc_to_freq(root_pc % 12, root_octave),
        _pc_to_freq((root_pc + third_interval) % 12, root_octave + (1 if root_pc + third_interval >= 12 else 0)),
        _pc_to_freq((root_pc + 7) % 12, root_octave + (1 if root_pc + 7 >= 12 else 0)),
    ]


def test_exact_templates_detect_themselves_for_all_24_chords() -> None:
    templates = generate_chord_templates()

    passed = 0
    failed: list[str] = []

    for name, vec in templates.items():
        result = predict_frame_chord(vec)
        if result.chord == name:
            passed += 1
        else:
            failed.append(f"{name}->{result.chord}")

    assert passed == 24, f"failed mappings: {failed}"


def test_c_major_template_detects_c() -> None:
    frame = generate_chord_templates()["C"]
    result = predict_frame_chord(frame)
    assert result.chord == "C"


def test_a_minor_template_detects_am() -> None:
    frame = generate_chord_templates()["Am"]
    result = predict_frame_chord(frame)
    assert result.chord == "Am"


def test_other_roots_detect_correctly() -> None:
    templates = generate_chord_templates()
    checks = {
        "F#": templates["F#"],
        "C#m": templates["C#m"],
        "G": templates["G"],
        "Em": templates["Em"],
    }

    for expected, frame in checks.items():
        result = predict_frame_chord(frame)
        assert result.chord == expected


def test_silence_returns_no_chord() -> None:
    frame = np.zeros(12, dtype=np.float32)
    result = predict_frame_chord(frame)

    assert result.chord == NO_CHORD_LABEL
    assert 0.0 <= result.confidence <= 1.0


def test_near_silence_returns_no_chord() -> None:
    frame = np.full(12, 1e-10, dtype=np.float32)
    result = predict_frame_chord(frame)

    assert result.chord == NO_CHORD_LABEL
    assert result.confidence > 0.9


def test_invalid_input_wrong_bin_count_raises() -> None:
    with pytest.raises(FrameDetectionError) as exc:
        predict_frame_chord(np.ones(11, dtype=np.float32))

    assert exc.value.code == "FRAME_INVALID_SHAPE"


def test_invalid_input_empty_raises() -> None:
    with pytest.raises(FrameDetectionError) as exc:
        predict_frame_chord(np.array([], dtype=np.float32))

    assert exc.value.code == "FRAME_EMPTY"


def test_invalid_input_nan_raises() -> None:
    frame = np.zeros(12, dtype=np.float32)
    frame[3] = np.nan

    with pytest.raises(FrameDetectionError) as exc:
        predict_frame_chord(frame)

    assert exc.value.code == "FRAME_INVALID_VALUES"


def test_invalid_input_inf_raises() -> None:
    frame = np.zeros(12, dtype=np.float32)
    frame[8] = np.inf

    with pytest.raises(FrameDetectionError) as exc:
        predict_frame_chord(frame)

    assert exc.value.code == "FRAME_INVALID_VALUES"


def test_confidence_always_in_unit_interval_for_valid_results() -> None:
    templates = generate_chord_templates()
    frames = list(templates.values()) + [np.zeros(12, dtype=np.float32), np.full(12, 1e-10, dtype=np.float32)]

    for frame in frames:
        result = predict_frame_chord(frame)
        assert 0.0 <= result.confidence <= 1.0


def test_deterministic_output_for_same_input() -> None:
    frame = generate_chord_templates()["G"]
    first = predict_frame_chord(frame)
    second = predict_frame_chord(frame)

    assert first.chord == second.chord
    assert first.confidence == second.confidence


def test_synthetic_audio_integration_c_major_detects_c() -> None:
    # C4, E4, G4
    frame = _representative_frame_from_audio(_chord([261.63, 329.63, 392.00]))
    result = predict_frame_chord(frame)

    assert result.chord == "C"
    assert 0.0 <= result.confidence <= 1.0


def test_synthetic_audio_integration_a_minor_detects_am() -> None:
    # A3, C4, E4
    frame = _representative_frame_from_audio(_chord([220.00, 261.63, 329.63]))
    result = predict_frame_chord(frame)

    assert result.chord == "Am"
    assert 0.0 <= result.confidence <= 1.0


def test_global_key_estimation_detects_c_major_for_c_major_synthetic() -> None:
    samples = _chord([261.63, 329.63, 392.00], duration=1.5)
    features = extract_chroma(_audio_buffer(samples))

    key = estimate_global_key(features.chroma)

    assert key is not None
    assert key.label == "C"
    assert 0.0 <= key.confidence <= 1.0


def test_soft_key_prior_does_not_forbid_non_diatonic_chords() -> None:
    non_diatonic_frame = generate_chord_templates()["D#"]
    key_context = KeyEstimate(tonic_pc=0, mode="major", confidence=1.0)

    result = predict_frame_chord(non_diatonic_frame, key_estimate=key_context)

    assert result.chord == "D#"


def test_synthetic_audio_all_12_major_roots_detect_correct_major_label() -> None:
    labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for root_pc, expected in enumerate(labels):
        frame = _representative_frame_from_audio(_chord(_triad_freqs(root_pc, minor=False), duration=1.4))
        result = predict_frame_chord(frame)
        assert result.chord == expected


def test_synthetic_audio_all_12_minor_roots_detect_correct_minor_label() -> None:
    labels = ["Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "A#m", "Bm"]
    for root_pc, expected in enumerate(labels):
        frame = _representative_frame_from_audio(_chord(_triad_freqs(root_pc, minor=True), duration=1.4))
        result = predict_frame_chord(frame)
        assert result.chord == expected


def test_quality_discrimination_c_major_vs_c_minor() -> None:
    c_major = _representative_frame_from_audio(_chord(_triad_freqs(0, minor=False), duration=1.5))
    c_minor = _representative_frame_from_audio(_chord(_triad_freqs(0, minor=True), duration=1.5))
    assert predict_frame_chord(c_major).chord == "C"
    assert predict_frame_chord(c_minor).chord == "Cm"


def test_quality_discrimination_a_major_vs_a_minor() -> None:
    a_major = _representative_frame_from_audio(_chord(_triad_freqs(9, minor=False), duration=1.5))
    a_minor = _representative_frame_from_audio(_chord(_triad_freqs(9, minor=True), duration=1.5))
    assert predict_frame_chord(a_major).chord == "A"
    assert predict_frame_chord(a_minor).chord == "Am"
