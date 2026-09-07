from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

import chord_engine.analyze as analyze_module
from chord_engine.analyze import ALGORITHM_ID, CONTRACT_VERSION, AnalysisMetadata, AnalysisResult, PipelineRun, SourceMetadata
from chord_engine.audio import TARGET_SAMPLE_RATE
from chord_engine.detector import KeyEstimate
from chord_engine.segmentation import ChordSegment
from chord_engine.timebase import DEFAULT_HOP_LENGTH, frame_duration_seconds


def tone(freq: float, duration: float = 1.2, amp: float = 0.45, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def chord(freqs: list[float], duration: float = 1.2, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    stacked = np.stack([tone(f, duration=duration, sr=sr) for f in freqs], axis=0)
    return np.mean(stacked, axis=0).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> None:
    sf.write(path, np.asarray(samples, dtype=np.float32), sr)


def mock_result(path: str, duration: float, segments: list[ChordSegment]) -> AnalysisResult:
    return AnalysisResult(
        version=CONTRACT_VERSION,
        source=SourceMetadata(path=path, duration=duration, sampleRate=TARGET_SAMPLE_RATE),
        analysis=AnalysisMetadata(algorithm=ALGORITHM_ID, chords=segments),
    )


def region_observation(
    *,
    start_frame: int,
    end_frame: int,
    winner: str,
    winner_conf: float,
    scores: dict[str, float],
    advantage: float,
) -> analyze_module.MusicalRegionObservation:
    frame_seconds = frame_duration_seconds(hop_length=DEFAULT_HOP_LENGTH, sample_rate=TARGET_SAMPLE_RATE)
    return analyze_module.MusicalRegionObservation(
        start_frame=start_frame,
        end_frame=end_frame,
        start_seconds=start_frame * frame_seconds,
        end_seconds=end_frame * frame_seconds,
        duration_seconds=(end_frame - start_frame) * frame_seconds,
        winner_chord=winner,
        winner_confidence=winner_conf,
        scores=scores,
        advantage_over_runner_up=advantage,
    )


def compressed_labels(predictions: list[analyze_module.FrameChordPrediction]) -> list[str]:
    if not predictions:
        return []
    labels = [predictions[0].chord]
    for pred in predictions[1:]:
        if pred.chord != labels[-1]:
            labels.append(pred.chord)
    return labels


def run_improved(path: Path) -> PipelineRun:
    return analyze_module._run_pipeline(
        path,
        use_harmonic_preprocessing=True,
        use_beat_sync=True,
        use_key_prior=True,
        use_context_correction=False,
    )


def write_profile_region(target: np.ndarray, start: int, end: int, pcs: list[int], scale: float = 1.0) -> None:
    for pc in pcs:
        target[pc, start:end] = scale
