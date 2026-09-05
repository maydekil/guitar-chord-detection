from __future__ import annotations

import numpy as np
import pytest

from chord_engine.audio import AudioBuffer, TARGET_SAMPLE_RATE
from chord_engine.features import (
    DEFAULT_HOP_LENGTH,
    PITCH_CLASS_ORDER,
    FeatureExtractionError,
    estimate_beat_frame_boundaries,
    extract_harmonic_signal,
    extract_chroma,
)


def _make_audio_buffer(samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> AudioBuffer:
    samples32 = np.asarray(samples, dtype=np.float32)
    duration = float(samples32.shape[0] / sample_rate)
    return AudioBuffer(samples=samples32, sample_rate=sample_rate, duration=duration)


def _tone(freq: float, duration: float = 1.0, amp: float = 0.4, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _chord(freqs: list[float], duration: float = 1.0, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    stacked = np.stack([_tone(f, duration=duration, sr=sr) for f in freqs], axis=0)
    return np.mean(stacked, axis=0).astype(np.float32)


def _dominant_pitch_classes(chroma: np.ndarray, top_n: int = 3) -> list[int]:
    mean_energy = np.mean(chroma, axis=1)
    order = np.argsort(mean_energy)[::-1]
    return [int(i) for i in order[:top_n]]


def test_output_has_12_pitch_class_bins() -> None:
    samples = _tone(440.0)
    features = extract_chroma(_make_audio_buffer(samples))

    assert features.chroma.shape[0] == 12
    assert features.pitch_class_order == PITCH_CLASS_ORDER


def test_output_has_at_least_one_frame_for_valid_audio() -> None:
    samples = _tone(220.0, duration=0.2)
    features = extract_chroma(_make_audio_buffer(samples))

    assert features.n_frames >= 1


def test_all_feature_values_are_finite() -> None:
    samples = _tone(330.0)
    features = extract_chroma(_make_audio_buffer(samples))

    assert np.isfinite(features.chroma).all()


def test_silence_produces_finite_values_without_nan_or_inf() -> None:
    samples = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)
    features = extract_chroma(_make_audio_buffer(samples))

    assert features.chroma.shape == (12, 1)
    assert np.isfinite(features.chroma).all()
    assert np.count_nonzero(features.chroma) == 0


def test_deterministic_input_produces_deterministic_output() -> None:
    samples = _tone(523.25, duration=0.8)
    audio = _make_audio_buffer(samples)

    first = extract_chroma(audio)
    second = extract_chroma(audio)

    np.testing.assert_allclose(first.chroma, second.chroma, rtol=1e-6, atol=1e-7)


def test_c_major_synthetic_has_dominant_c_e_g_energy() -> None:
    # C4, E4, G4
    samples = _chord([261.63, 329.63, 392.00], duration=1.2)
    features = extract_chroma(_make_audio_buffer(samples))

    dominant = _dominant_pitch_classes(features.chroma, top_n=4)

    assert 0 in dominant  # C
    assert 4 in dominant  # E
    assert 7 in dominant  # G


def test_a_minor_synthetic_has_dominant_a_c_e_energy() -> None:
    # A3, C4, E4
    samples = _chord([220.00, 261.63, 329.63], duration=1.2)
    features = extract_chroma(_make_audio_buffer(samples))

    dominant = _dominant_pitch_classes(features.chroma, top_n=4)

    assert 9 in dominant  # A
    assert 0 in dominant  # C
    assert 4 in dominant  # E


def test_frame_timing_metadata_is_internally_consistent() -> None:
    samples = _tone(440.0, duration=0.9)
    features = extract_chroma(_make_audio_buffer(samples), hop_length=DEFAULT_HOP_LENGTH)

    times = features.frame_times()

    assert times.shape[0] == features.n_frames
    assert np.all(np.diff(times) >= 0)
    if features.n_frames > 1:
        assert float(times[1] - times[0]) == pytest.approx(features.frame_duration_seconds, rel=1e-5)


def test_feature_extraction_does_not_mutate_input_audio_buffer() -> None:
    samples = _tone(440.0)
    original = samples.copy()
    audio = _make_audio_buffer(samples)

    _ = extract_chroma(audio)

    np.testing.assert_array_equal(audio.samples, original)


def test_invalid_audio_buffer_raises_controlled_error() -> None:
    bad_audio = AudioBuffer(samples=np.array([], dtype=np.float32), sample_rate=TARGET_SAMPLE_RATE, duration=0.0)

    with pytest.raises(FeatureExtractionError) as exc:
        extract_chroma(bad_audio)

    assert exc.value.code == "FEATURE_INVALID_AUDIO"


def test_invalid_sample_rate_raises_controlled_error() -> None:
    samples = _tone(440.0, sr=44100)
    audio = _make_audio_buffer(samples, sample_rate=44100)

    with pytest.raises(FeatureExtractionError) as exc:
        extract_chroma(audio)

    assert exc.value.code == "FEATURE_INVALID_SAMPLE_RATE"


def test_harmonic_signal_extraction_is_deterministic() -> None:
    samples = _chord([261.63, 329.63, 392.00], duration=1.0)

    first = extract_harmonic_signal(samples)
    second = extract_harmonic_signal(samples)

    np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-7)


def test_harmonic_signal_for_silence_returns_input_shape() -> None:
    samples = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)
    harmonic = extract_harmonic_signal(samples)

    assert harmonic.shape == samples.shape
    assert np.isfinite(harmonic).all()


def test_beat_boundaries_always_include_start_and_end() -> None:
    samples = _tone(220.0, duration=1.5)
    features = extract_chroma(_make_audio_buffer(samples))

    boundaries = estimate_beat_frame_boundaries(
        samples,
        sample_rate=TARGET_SAMPLE_RATE,
        hop_length=DEFAULT_HOP_LENGTH,
        n_frames=features.n_frames,
    )

    assert boundaries[0] == 0
    assert boundaries[-1] == features.n_frames
    assert np.all(np.diff(boundaries) >= 0)


def test_beat_boundaries_densify_when_region_is_too_large() -> None:
    samples = _tone(220.0, duration=4.0)
    features = extract_chroma(_make_audio_buffer(samples))

    boundaries = estimate_beat_frame_boundaries(
        samples,
        sample_rate=TARGET_SAMPLE_RATE,
        hop_length=DEFAULT_HOP_LENGTH,
        n_frames=features.n_frames,
        max_region_seconds=0.5,
    )

    max_allowed = int(round(0.5 * TARGET_SAMPLE_RATE / DEFAULT_HOP_LENGTH))
    assert boundaries[0] == 0
    assert boundaries[-1] == features.n_frames
    assert np.max(np.diff(boundaries)) <= max_allowed
