"""Optional third-party chord analysis backends.

These backends are explicit experiments. They must preserve the public analysis
contract and never become a hidden dependency of the default local engine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chord_engine.analysis_models import CONTRACT_VERSION, AnalysisMetadata, AnalysisResult, SourceMetadata
from chord_engine.audio import load_audio
from chord_engine.features import extract_chroma
from chord_engine.segment_utils import (
	_clamp_final_segment_end,
	_merge_adjacent_same_chord_segments,
	_segment_duration,
	_weighted_confidence,
)
from chord_engine.segmentation import ChordSegment

ESSENTIA_ALGORITHM_ID = "essentia-chords-v1"
SUPPORTED_EXTERNAL_BACKENDS = ("builtin", "essentia")

_PITCH_NAMES = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
_ESSENTIA_PLAYABLE_COMPACTION_MIN_AUDIO_SECONDS = 6.0
_ESSENTIA_LABEL_SMOOTH_SECONDS = 1.80
_ESSENTIA_FRAGMENT_MAX_SECONDS = 1.35
_ESSENTIA_STRONG_FRAGMENT_MIN_SECONDS = 0.85
_ESSENTIA_STRONG_FRAGMENT_CONFIDENCE = 0.88


class ExternalBackendError(Exception):
	"""Controlled error for optional backend failures."""

	def __init__(self, code: str, message: str) -> None:
		super().__init__(message)
		self.code = code
		self.message = message


def analyze_with_essentia(path: str | Path) -> AnalysisResult:
	"""Analyze chords using Essentia's built-in chord detector on local chroma."""
	try:
		from essentia.standard import ChordsDetection
	except Exception as exc:  # pragma: no cover - depends on optional install
		raise ExternalBackendError(
			"ESSENTIA_BACKEND_UNAVAILABLE",
			"Essentia backend is not installed. Install the optional Essentia dependency to use --backend essentia.",
		) from exc

	audio = load_audio(path)
	features = extract_chroma(audio, use_harmonic_preprocessing=True)
	hpcp = _c_order_chroma_to_essentia_hpcp(features.chroma)
	if hpcp.ndim != 2 or hpcp.shape[1] != 12 or hpcp.shape[0] == 0:
		raise ExternalBackendError("ESSENTIA_FEATURE_INVALID", "Unable to build Essentia-compatible chroma features")

	try:
		labels, strengths = ChordsDetection(hopSize=features.hop_length, windowSize=2)(hpcp)
	except Exception as exc:  # pragma: no cover - boundary around native library
		raise ExternalBackendError("ESSENTIA_ANALYSIS_FAILED", "Essentia chord analysis failed") from exc

	segments = _segments_from_essentia_frames(
		labels,
		strengths,
		hop_length=features.hop_length,
		sample_rate=features.sample_rate,
		source_duration=audio.duration,
	)

	return AnalysisResult(
		version=CONTRACT_VERSION,
		source=SourceMetadata(path=str(path), duration=audio.duration, sampleRate=audio.sample_rate),
		analysis=AnalysisMetadata(algorithm=ESSENTIA_ALGORITHM_ID, chords=segments),
	)


def _segments_from_essentia_frames(
	labels: object,
	strengths: object,
	*,
	hop_length: int,
	sample_rate: int,
	source_duration: float,
) -> list[ChordSegment]:
	label_list = [_normalize_essentia_label(label) for label in list(labels)]
	strength_list = [float(np.clip(value, 0.0, 1.0)) for value in list(strengths)]
	if len(label_list) == 0:
		return [ChordSegment(start=0.0, end=source_duration, chord="N", confidence=1.0)]
	if len(strength_list) != len(label_list):
		strength_list = [0.0 for _ in label_list]

	frame_seconds = float(hop_length / sample_rate)
	if source_duration >= _ESSENTIA_PLAYABLE_COMPACTION_MIN_AUDIO_SECONDS:
		label_list = _smooth_essentia_labels(label_list, strength_list, frame_seconds=frame_seconds)

	segments: list[ChordSegment] = []
	start_idx = 0
	current = label_list[0]
	confidence_sum = strength_list[0]

	for idx in range(1, len(label_list)):
		if label_list[idx] == current:
			confidence_sum += strength_list[idx]
			continue
		segments.append(
			ChordSegment(
				start=start_idx * frame_seconds,
				end=idx * frame_seconds,
				chord=current,
				confidence=float(np.clip(confidence_sum / max(1, idx - start_idx), 0.0, 1.0)),
			)
		)
		start_idx = idx
		current = label_list[idx]
		confidence_sum = strength_list[idx]

	segments.append(
		ChordSegment(
			start=start_idx * frame_seconds,
			end=source_duration,
			chord=current,
			confidence=float(np.clip(confidence_sum / max(1, len(label_list) - start_idx), 0.0, 1.0)),
		)
	)

	segments = [segment for segment in segments if segment.end > segment.start]
	segments = _merge_adjacent_same_chord_segments(segments)
	if source_duration >= _ESSENTIA_PLAYABLE_COMPACTION_MIN_AUDIO_SECONDS:
		segments = _compact_essentia_fragments(segments)
	return _clamp_final_segment_end(segments, source_duration=source_duration)


def _c_order_chroma_to_essentia_hpcp(chroma: np.ndarray) -> np.ndarray:
	"""Convert canonical C..B chroma into Essentia HPCP order A..G#."""
	arr = np.asarray(chroma, dtype=np.float32)
	if arr.ndim != 2 or arr.shape[0] != 12:
		raise ExternalBackendError("ESSENTIA_FEATURE_INVALID", "Expected canonical 12-bin chroma features")
	return arr[[9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8], :].T


def _smooth_essentia_labels(labels: list[str], strengths: list[float], *, frame_seconds: float) -> list[str]:
	if not labels or frame_seconds <= 0.0:
		return labels

	window_frames = max(1, int(round(_ESSENTIA_LABEL_SMOOTH_SECONDS / frame_seconds)))
	if window_frames % 2 == 0:
		window_frames += 1
	radius = window_frames // 2
	smoothed: list[str] = []

	for idx, current in enumerate(labels):
		left = max(0, idx - radius)
		right = min(len(labels), idx + radius + 1)
		votes: dict[str, float] = {}
		for vote_idx in range(left, right):
			label = labels[vote_idx]
			weight = max(0.05, strengths[vote_idx])
			votes[label] = votes.get(label, 0.0) + weight

		current_vote = votes.get(current, 0.0)
		winner, winner_vote = sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0]
		smoothed.append(winner if winner_vote > current_vote else current)

	return smoothed


def _compact_essentia_fragments(segments: list[ChordSegment]) -> list[ChordSegment]:
	working = _merge_adjacent_same_chord_segments(list(segments))
	changed = True
	while changed:
		changed = False
		for idx, current in enumerate(list(working)):
			duration = _segment_duration(current)
			if duration >= _ESSENTIA_FRAGMENT_MAX_SECONDS:
				continue
			if (
				duration >= _ESSENTIA_STRONG_FRAGMENT_MIN_SECONDS
				and current.confidence >= _ESSENTIA_STRONG_FRAGMENT_CONFIDENCE
			):
				continue
			if len(working) <= 1:
				continue

			target = _essentia_fragment_replacement(working, idx)
			if target is None or target == current.chord:
				continue

			working[idx] = ChordSegment(
				start=current.start,
				end=current.end,
				chord=target,
				confidence=_replacement_confidence(current, working, idx, target),
			)
			working = _merge_adjacent_same_chord_segments(working)
			changed = True
			break
	return working


def _essentia_fragment_replacement(segments: list[ChordSegment], idx: int) -> str | None:
	left = segments[idx - 1] if idx > 0 else None
	right = segments[idx + 1] if idx + 1 < len(segments) else None
	if left is None and right is None:
		return None
	if left is None:
		return right.chord if right else None
	if right is None:
		return left.chord
	if left.chord == right.chord:
		return left.chord

	left_score = (_segment_duration(left) * 0.65) + (left.confidence * 0.35)
	right_score = (_segment_duration(right) * 0.65) + (right.confidence * 0.35)
	if abs(left_score - right_score) <= 1e-9:
		return left.chord
	return left.chord if left_score > right_score else right.chord


def _replacement_confidence(current: ChordSegment, segments: list[ChordSegment], idx: int, target: str) -> float:
	neighbors = [
		segment
		for segment in (
			segments[idx - 1] if idx > 0 else None,
			segments[idx + 1] if idx + 1 < len(segments) else None,
		)
		if segment is not None and segment.chord == target
	]
	if not neighbors:
		return current.confidence
	confidence = current.confidence
	pseudo = current
	for neighbor in neighbors:
		confidence = _weighted_confidence(
			ChordSegment(pseudo.start, pseudo.end, target, confidence),
			neighbor,
		)
	return float(np.clip(confidence, 0.0, 1.0))


def _normalize_essentia_label(label: object) -> str:
	raw = str(label).strip()
	if raw in {"", "N", "None", "no_chord"}:
		return "N"

	root = raw
	is_minor = False
	if ":" in root:
		root, quality = root.split(":", 1)
		is_minor = quality.lower().startswith("min")
	elif root.endswith("m") and len(root) > 1:
		root = root[:-1]
		is_minor = True

	root = root.replace("Db", "C#").replace("Eb", "D#").replace("Gb", "F#").replace("Ab", "G#").replace("Bb", "A#")
	return f"{root}m" if root in _PITCH_NAMES and is_minor else root if root in _PITCH_NAMES else "N"
