from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from shutil import which

import numpy as np
import pytest
import soundfile as sf

import chord_engine.cli as cli_module
from chord_engine.audio import TARGET_SAMPLE_RATE
from chord_engine.analysis_models import AnalysisMetadata, AnalysisResult, SourceMetadata


def _tone(freq: float, duration: float = 1.4, amp: float = 0.45, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _chord(freqs: list[float], duration: float = 1.4, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    stacked = np.stack([_tone(f, duration=duration, sr=sr) for f in freqs], axis=0)
    return np.mean(stacked, axis=0).astype(np.float32)


def _write_wav(path: Path, samples: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> None:
    sf.write(path, np.asarray(samples, dtype=np.float32), sr)


def _run_cli(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "chord_engine.cli", "analyze", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )


def _run_cli_with_backend(path: Path, backend: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "chord_engine.cli", "analyze", str(path), "--backend", backend],
        text=True,
        capture_output=True,
        check=False,
    )


def _parse_json_stdout(stdout: str) -> dict[str, object]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 1, "stdout must contain exactly one JSON line"
    return json.loads(lines[0])


def _run_eval_cli(audio_path: Path, annotation_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "chord_engine.cli",
            "evaluate-ground-truth",
            str(audio_path),
            str(annotation_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_wav_success_outputs_valid_json(tmp_path: Path) -> None:
    wav = tmp_path / "c_major.wav"
    _write_wav(wav, _chord([261.63, 329.63, 392.00], duration=1.6))

    proc = _run_cli(wav)

    assert proc.returncode == 0
    payload = _parse_json_stdout(proc.stdout)

    assert payload["version"] == "1"
    assert "error" not in payload
    assert "source" in payload
    assert "analysis" in payload


def test_cli_backend_flag_is_passed_to_analyzer(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def fake_analyze(path: str, *, backend: str) -> AnalysisResult:
        seen["path"] = path
        seen["backend"] = backend
        return AnalysisResult("1", SourceMetadata(path, 1.0, 22050), AnalysisMetadata("fake", []))

    monkeypatch.setattr(cli_module, "analyze_audio", fake_analyze)
    proc = cli_module.main(["analyze", "song.wav", "--backend", "essentia"])

    assert proc == 0
    assert seen == {"path": "song.wav", "backend": "essentia"}


def test_cli_backend_env_is_used_when_flag_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def fake_analyze(path: str, *, backend: str) -> AnalysisResult:
        seen["backend"] = backend
        return AnalysisResult("1", SourceMetadata(path, 1.0, 22050), AnalysisMetadata("fake", []))

    monkeypatch.setenv("GCD_CHORD_BACKEND", "essentia")
    monkeypatch.setattr(cli_module, "analyze_audio", fake_analyze)

    assert cli_module.main(["analyze", "song.wav"]) == 0
    assert seen["backend"] == "essentia"


def test_cli_essentia_backend_unavailable_returns_controlled_json(tmp_path: Path) -> None:
    wav = tmp_path / "c_major.wav"
    _write_wav(wav, _chord([261.63, 329.63, 392.00], duration=1.0))

    proc = _run_cli_with_backend(wav, "essentia")

    if proc.returncode == 0:
        pytest.skip("Essentia is installed in this environment; unavailable-backend path is not active")
    payload = _parse_json_stdout(proc.stdout)
    assert payload["version"] == "1"
    assert payload["error"]["code"] == "ESSENTIA_BACKEND_UNAVAILABLE"  # type: ignore[index]


def test_cli_mp3_success_if_available(tmp_path: Path) -> None:
    if which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    wav = tmp_path / "source.wav"
    mp3 = tmp_path / "source.mp3"
    _write_wav(wav, _chord([220.00, 261.63, 329.63], duration=1.6))

    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), str(mp3)],
        check=True,
    )

    proc = _run_cli(mp3)

    assert proc.returncode == 0
    payload = _parse_json_stdout(proc.stdout)
    assert payload["version"] == "1"
    assert "analysis" in payload


def test_cli_missing_file_returns_controlled_json_error_and_nonzero(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"

    proc = _run_cli(missing)

    assert proc.returncode != 0
    payload = _parse_json_stdout(proc.stdout)
    assert payload["version"] == "1"
    assert payload["error"]["code"] == "AUDIO_FILE_NOT_FOUND"  # type: ignore[index]


def test_cli_corrupt_file_returns_controlled_json_error_and_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"not-audio")

    proc = _run_cli(bad)

    assert proc.returncode != 0
    payload = _parse_json_stdout(proc.stdout)
    assert payload["version"] == "1"
    assert payload["error"]["code"] == "AUDIO_DECODE_FAILED"  # type: ignore[index]


def test_cli_required_fields_and_types(tmp_path: Path) -> None:
    wav = tmp_path / "am.wav"
    _write_wav(wav, _chord([220.00, 261.63, 329.63], duration=1.6))

    proc = _run_cli(wav)
    assert proc.returncode == 0

    payload = _parse_json_stdout(proc.stdout)

    assert isinstance(payload["version"], str)
    source = payload["source"]  # type: ignore[index]
    analysis = payload["analysis"]  # type: ignore[index]
    assert isinstance(source["path"], str)
    assert isinstance(source["duration"], (int, float))
    assert isinstance(source["sampleRate"], int)
    assert isinstance(analysis["algorithm"], str)
    assert isinstance(analysis["chords"], list)

    chords = analysis["chords"]
    assert len(chords) >= 1
    for seg in chords:
        assert isinstance(seg["start"], (int, float))
        assert isinstance(seg["end"], (int, float))
        assert isinstance(seg["chord"], str)
        assert isinstance(seg["confidence"], (int, float))


def test_cli_stdout_contains_only_json(tmp_path: Path) -> None:
    wav = tmp_path / "g_major.wav"
    _write_wav(wav, _chord([196.00, 246.94, 293.66], duration=1.6))

    proc = _run_cli(wav)

    assert proc.returncode == 0
    payload = _parse_json_stdout(proc.stdout)
    assert payload["version"] == "1"


def test_cli_serialization_is_deterministic(tmp_path: Path) -> None:
    wav = tmp_path / "deterministic.wav"
    _write_wav(wav, _chord([261.63, 329.63, 392.00], duration=1.6))

    first = _run_cli(wav)
    second = _run_cli(wav)

    assert first.returncode == 0
    assert second.returncode == 0

    first_payload = _parse_json_stdout(first.stdout)
    second_payload = _parse_json_stdout(second.stdout)
    assert first_payload == second_payload
    assert first.stdout.strip() == second.stdout.strip()


def test_cli_evaluate_ground_truth_success_outputs_metrics(tmp_path: Path) -> None:
    wav = tmp_path / "eval.wav"
    _write_wav(wav, _chord([261.63, 329.63, 392.00], duration=2.0))

    annotation = tmp_path / "eval-ground-truth.json"
    annotation.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 2.0, "chord": "C"},
                ]
            }
        ),
        encoding="utf-8",
    )

    proc = _run_eval_cli(wav, annotation)

    assert proc.returncode == 0
    payload = _parse_json_stdout(proc.stdout)
    assert payload["version"] == "1"
    assert "metrics" in payload
    metrics = payload["metrics"]  # type: ignore[index]
    assert "timeWeightedChordAccuracy" in metrics
    assert "exactChordMatchPercentage" in metrics
    assert "rootAccuracy" in metrics
    assert "qualityAccuracy" in metrics
    assert "falseTransitionCount" in metrics
    assert "missedTransitionCount" in metrics
    assert "boundaryTimingErrorSeconds" in metrics
    assert "confusionPairs" in metrics


def test_cli_evaluate_ground_truth_invalid_annotation_returns_controlled_error(tmp_path: Path) -> None:
    wav = tmp_path / "eval-invalid.wav"
    _write_wav(wav, _chord([261.63, 329.63, 392.00], duration=1.2))

    annotation = tmp_path / "invalid-ground-truth.json"
    annotation.write_text("{invalid-json", encoding="utf-8")

    proc = _run_eval_cli(wav, annotation)

    assert proc.returncode != 0
    payload = _parse_json_stdout(proc.stdout)
    assert payload["version"] == "1"
    assert payload["error"]["code"] == "GROUND_TRUTH_PARSE_FAILED"  # type: ignore[index]


def test_cli_evaluate_genre_corpus_outputs_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_evaluate(_manifest_path: str) -> dict[str, object]:
        return {
            "version": "1",
            "status": "pass",
            "itemCount": 1,
            "evaluatedItemCount": 1,
            "failedItemCount": 0,
            "metrics": {"timeWeightedChordAccuracy": 100.0},
            "genres": {"pop": {"timeWeightedChordAccuracy": 100.0}},
            "items": [],
        }

    monkeypatch.setattr(cli_module, "evaluate_genre_corpus", fake_evaluate)

    code = cli_module.main(["evaluate-genre-corpus", "genre-manifest.json"])

    captured = capsys.readouterr()
    assert code == 0
    payload = _parse_json_stdout(captured.out)
    assert payload["version"] == "1"
    assert payload["status"] == "pass"
    assert payload["genres"]["pop"]["timeWeightedChordAccuracy"] == 100.0  # type: ignore[index]


def test_cli_evaluate_genre_corpus_invalid_manifest_returns_controlled_error(tmp_path: Path) -> None:
    manifest = tmp_path / "invalid-genre-manifest.json"
    manifest.write_text("{invalid-json", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "chord_engine.cli",
            "evaluate-genre-corpus",
            str(manifest),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    payload = _parse_json_stdout(proc.stdout)
    assert payload["version"] == "1"
    assert payload["error"]["code"] == "GENRE_EVAL_MANIFEST_PARSE_FAILED"  # type: ignore[index]
