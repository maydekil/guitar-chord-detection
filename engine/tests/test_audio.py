from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which

import numpy as np
import pytest
import soundfile as sf

from chord_engine.audio import AudioDecodeError, TARGET_SAMPLE_RATE, load_audio


@pytest.fixture()
def temp_audio_dir(tmp_path: Path) -> Path:
    return tmp_path


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    sf.write(path, samples, sample_rate)


def test_load_valid_mono_wav(temp_audio_dir: Path) -> None:
    sr = TARGET_SAMPLE_RATE
    duration = 1.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    mono = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)

    wav_path = temp_audio_dir / "mono.wav"
    _write_wav(wav_path, mono, sr)

    audio = load_audio(wav_path)

    assert audio.sample_rate == TARGET_SAMPLE_RATE
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    assert np.max(np.abs(audio.samples)) <= 1.0 + 1e-6
    assert audio.duration == pytest.approx(duration, abs=1e-3)


def test_load_valid_stereo_wav_converts_to_mono(temp_audio_dir: Path) -> None:
    sr = TARGET_SAMPLE_RATE
    frames = sr
    left = np.full(frames, 0.75, dtype=np.float32)
    right = np.full(frames, -0.25, dtype=np.float32)
    stereo = np.column_stack([left, right]).astype(np.float32)

    wav_path = temp_audio_dir / "stereo.wav"
    _write_wav(wav_path, stereo, sr)

    audio = load_audio(wav_path)

    assert audio.samples.ndim == 1
    assert audio.samples.shape[0] == frames
    assert float(np.mean(audio.samples)) == pytest.approx(0.25, abs=2e-3)


def test_resamples_audio_to_target_sample_rate(temp_audio_dir: Path) -> None:
    src_sr = 44100
    duration = 1.0
    t = np.linspace(0.0, duration, int(src_sr * duration), endpoint=False, dtype=np.float32)
    mono = 0.25 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)

    wav_path = temp_audio_dir / "resample.wav"
    _write_wav(wav_path, mono, src_sr)

    audio = load_audio(wav_path)

    assert audio.sample_rate == TARGET_SAMPLE_RATE
    assert audio.samples.shape[0] == int(TARGET_SAMPLE_RATE * duration)
    assert audio.duration == pytest.approx(duration, abs=1e-3)


def test_duration_is_calculated_correctly(temp_audio_dir: Path) -> None:
    sr = TARGET_SAMPLE_RATE
    duration = 0.37
    samples = np.zeros(int(sr * duration), dtype=np.float32)

    wav_path = temp_audio_dir / "duration.wav"
    _write_wav(wav_path, samples, sr)

    audio = load_audio(wav_path)

    assert audio.duration == pytest.approx(duration, abs=1e-3)


def test_invalid_audio_raises_controlled_error(temp_audio_dir: Path) -> None:
    bad_path = temp_audio_dir / "invalid.txt"
    bad_path.write_text("not an audio file", encoding="utf-8")

    with pytest.raises(AudioDecodeError) as exc:
        load_audio(bad_path)

    assert exc.value.code == "AUDIO_DECODE_FAILED"


def test_empty_audio_raises_controlled_error(temp_audio_dir: Path) -> None:
    sr = TARGET_SAMPLE_RATE
    empty = np.array([], dtype=np.float32)

    wav_path = temp_audio_dir / "empty.wav"
    _write_wav(wav_path, empty, sr)

    with pytest.raises(AudioDecodeError) as exc:
        load_audio(wav_path)

    assert exc.value.code == "AUDIO_EMPTY"


def test_load_valid_mp3_if_ffmpeg_available(temp_audio_dir: Path) -> None:
    if which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    sr = TARGET_SAMPLE_RATE
    duration = 0.5
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    mono = 0.4 * np.sin(2.0 * np.pi * 330.0 * t).astype(np.float32)

    wav_path = temp_audio_dir / "source.wav"
    mp3_path = temp_audio_dir / "source.mp3"
    _write_wav(wav_path, mono, sr)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(wav_path),
        str(mp3_path),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        pytest.skip("ffmpeg conversion to mp3 failed")

    audio = load_audio(mp3_path)

    assert audio.sample_rate == TARGET_SAMPLE_RATE
    assert audio.samples.ndim == 1
    assert audio.samples.dtype == np.float32
    assert audio.duration == pytest.approx(duration, abs=2e-2)
