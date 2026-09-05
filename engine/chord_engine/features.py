"""Harmonic feature extraction (12-bin chroma) for the chord engine."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from chord_engine.audio import AudioBuffer, TARGET_SAMPLE_RATE

DEFAULT_HOP_LENGTH = 512
DEFAULT_N_CHROMA = 12
MIN_SIGNAL_ENERGY = 1e-7
HPSS_MARGIN = 1.0
MAX_BEAT_REGION_SECONDS = 0.7
LOW_FREQUENCY_FMIN_HZ = 65.40639132514966  # C2
LOW_FREQUENCY_OCTAVES = 2
PITCH_CLASS_ORDER = (
	"C",
	"C#",
	"D",
	"D#",
	"E",
	"F",
	"F#",
	"G",
	"G#",
	"A",
	"A#",
	"B",
)

MIN_RELIABLE_BEAT_COUNT = 8
MIN_RELIABLE_TEMPO_BPM = 45.0
MAX_RELIABLE_TEMPO_BPM = 220.0


@dataclass(frozen=True)
class ChromaFeatures:
	"""Feature container used by downstream detector stages."""

	chroma: np.ndarray
	hop_length: int
	sample_rate: int
	pitch_class_order: tuple[str, ...] = PITCH_CLASS_ORDER

	@property
	def n_frames(self) -> int:
		return int(self.chroma.shape[1])

	@property
	def frame_duration_seconds(self) -> float:
		return float(self.hop_length / self.sample_rate)

	def frame_times(self) -> np.ndarray:
		return librosa.frames_to_time(np.arange(self.n_frames), sr=self.sample_rate, hop_length=self.hop_length)


class FeatureExtractionError(Exception):
	"""Controlled error for feature extraction failures."""

	def __init__(self, code: str, message: str) -> None:
		super().__init__(message)
		self.code = code
		self.message = message


@dataclass(frozen=True)
class BeatTiming:
	"""Deterministic beat-timing summary used by musical-time aggregation."""

	boundaries: np.ndarray
	tempo_bpm: float | None
	beat_count: int
	is_reliable: bool


def extract_chroma(
	audio: AudioBuffer,
	*,
	hop_length: int = DEFAULT_HOP_LENGTH,
	n_chroma: int = DEFAULT_N_CHROMA,
	use_harmonic_preprocessing: bool = True,
) -> ChromaFeatures:
	"""Extract a deterministic 12-bin chroma representation from an AudioBuffer."""

	if audio.sample_rate != TARGET_SAMPLE_RATE:
		raise FeatureExtractionError(
			"FEATURE_INVALID_SAMPLE_RATE",
			f"Expected sample rate {TARGET_SAMPLE_RATE}, got {audio.sample_rate}",
		)

	if hop_length <= 0:
		raise FeatureExtractionError("FEATURE_INVALID_CONFIG", "hop_length must be positive")

	if n_chroma != DEFAULT_N_CHROMA:
		raise FeatureExtractionError("FEATURE_INVALID_CONFIG", "Only 12-bin chroma is supported")

	samples = np.asarray(audio.samples, dtype=np.float32)
	if samples.ndim != 1 or samples.size == 0:
		raise FeatureExtractionError("FEATURE_INVALID_AUDIO", "AudioBuffer samples must be non-empty mono")

	# Keep input immutable by working on a copied view for librosa processing.
	working = samples.copy()
	if use_harmonic_preprocessing:
		working = extract_harmonic_signal(working)

	if float(np.max(np.abs(working))) < MIN_SIGNAL_ENERGY:
		chroma = np.zeros((DEFAULT_N_CHROMA, 1), dtype=np.float32)
	else:
		chroma = librosa.feature.chroma_cqt(
			y=working,
			sr=audio.sample_rate,
			hop_length=hop_length,
			n_chroma=DEFAULT_N_CHROMA,
			norm=2,
		)

		chroma = np.asarray(chroma, dtype=np.float32)
		if chroma.ndim != 2 or chroma.shape[0] != DEFAULT_N_CHROMA:
			raise FeatureExtractionError("FEATURE_EXTRACTION_FAILED", "Unexpected chroma shape")

		if chroma.shape[1] == 0:
			chroma = np.zeros((DEFAULT_N_CHROMA, 1), dtype=np.float32)

	chroma = np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
	if not np.isfinite(chroma).all():
		raise FeatureExtractionError("FEATURE_EXTRACTION_FAILED", "Non-finite values in chroma output")

	return ChromaFeatures(
		chroma=chroma,
		hop_length=hop_length,
		sample_rate=audio.sample_rate,
	)


def extract_harmonic_signal(samples: np.ndarray, *, margin: float = HPSS_MARGIN) -> np.ndarray:
	"""Return harmonic-focused signal with deterministic fallback to input signal."""

	arr = np.asarray(samples, dtype=np.float32)
	if arr.ndim != 1:
		raise FeatureExtractionError("FEATURE_INVALID_AUDIO", "AudioBuffer samples must be non-empty mono")

	if arr.size == 0:
		return arr

	if float(np.max(np.abs(arr))) < MIN_SIGNAL_ENERGY:
		return arr

	try:
		harmonic, _ = librosa.effects.hpss(arr, margin=margin)
		harmonic_arr = np.asarray(harmonic, dtype=np.float32)
		if harmonic_arr.shape != arr.shape or not np.isfinite(harmonic_arr).all():
			return arr
		if float(np.max(np.abs(harmonic_arr))) < MIN_SIGNAL_ENERGY:
			return arr
		return harmonic_arr
	except Exception:
		return arr


def extract_low_frequency_chroma(
	samples: np.ndarray,
	*,
	sample_rate: int,
	hop_length: int,
	n_frames: int,
	n_chroma: int = DEFAULT_N_CHROMA,
) -> np.ndarray:
	"""Extract low-frequency chroma support used as a bass/root hint.

	This signal is supporting evidence only and must not be treated as an
	absolute root decision.
	"""

	arr = np.asarray(samples, dtype=np.float32)
	if arr.ndim != 1 or arr.size == 0 or n_frames <= 0:
		return np.zeros((n_chroma, max(1, n_frames)), dtype=np.float32)

	if float(np.max(np.abs(arr))) < MIN_SIGNAL_ENERGY:
		return np.zeros((n_chroma, n_frames), dtype=np.float32)

	try:
		low = librosa.feature.chroma_cqt(
			y=arr,
			sr=sample_rate,
			hop_length=hop_length,
			n_chroma=n_chroma,
			norm=2,
			fmin=LOW_FREQUENCY_FMIN_HZ,
			n_octaves=LOW_FREQUENCY_OCTAVES,
		)
		low = np.nan_to_num(np.asarray(low, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
		if low.ndim != 2 or low.shape[0] != n_chroma:
			return np.zeros((n_chroma, n_frames), dtype=np.float32)
		if low.shape[1] == 0:
			return np.zeros((n_chroma, n_frames), dtype=np.float32)

		if low.shape[1] < n_frames:
			pad = np.repeat(low[:, -1:], n_frames - low.shape[1], axis=1)
			low = np.concatenate((low, pad), axis=1)
		elif low.shape[1] > n_frames:
			low = low[:, :n_frames]

		return low.astype(np.float32, copy=False)
	except Exception:
		return np.zeros((n_chroma, n_frames), dtype=np.float32)


def estimate_beat_frame_boundaries(
	samples: np.ndarray,
	*,
	sample_rate: int,
	hop_length: int,
	n_frames: int,
	max_region_seconds: float = MAX_BEAT_REGION_SECONDS,
) -> np.ndarray:
	"""Estimate beat-synchronous frame boundaries as monotonically increasing indices."""
	timing = estimate_beat_timing(
		samples,
		sample_rate=sample_rate,
		hop_length=hop_length,
		n_frames=n_frames,
		max_region_seconds=max_region_seconds,
	)
	return timing.boundaries


def estimate_beat_timing(
	samples: np.ndarray,
	*,
	sample_rate: int,
	hop_length: int,
	n_frames: int,
	max_region_seconds: float = MAX_BEAT_REGION_SECONDS,
) -> BeatTiming:
	"""Estimate beat boundaries and tempo with deterministic reliability flag."""

	if n_frames <= 0:
		return BeatTiming(
			boundaries=np.array([0], dtype=np.int32),
			tempo_bpm=None,
			beat_count=0,
			is_reliable=False,
		)

	arr = np.asarray(samples, dtype=np.float32)
	if arr.ndim != 1 or arr.size == 0:
		return BeatTiming(
			boundaries=np.array([0, n_frames], dtype=np.int32),
			tempo_bpm=None,
			beat_count=0,
			is_reliable=False,
		)

	tempo_bpm: float | None = None
	try:
		tempo_raw, beat_frames = librosa.beat.beat_track(
			y=arr,
			sr=sample_rate,
			hop_length=hop_length,
			trim=False,
			units="frames",
		)
		tempo_arr = np.asarray(tempo_raw, dtype=np.float32).reshape(-1)
		if tempo_arr.size > 0:
			tempo_value = float(tempo_arr[0])
			if np.isfinite(tempo_value) and tempo_value > 0:
				tempo_bpm = tempo_value
		beats = np.asarray(beat_frames, dtype=np.int32)
	except Exception:
		beats = np.array([], dtype=np.int32)

	if beats.size == 0:
		boundaries = np.array([0, n_frames], dtype=np.int32)
	else:
		beats = beats[(beats > 0) & (beats < n_frames)]
		boundaries = np.concatenate((np.array([0], dtype=np.int32), beats, np.array([n_frames], dtype=np.int32)))
	boundaries = np.unique(boundaries)
	if boundaries[0] != 0:
		boundaries = np.concatenate((np.array([0], dtype=np.int32), boundaries))
	if boundaries[-1] != n_frames:
		boundaries = np.concatenate((boundaries, np.array([n_frames], dtype=np.int32)))

	max_region_frames = max(1, int(round(max_region_seconds * sample_rate / hop_length)))
	if max_region_frames <= 1:
		return boundaries.astype(np.int32, copy=False)

	densified: list[int] = [int(boundaries[0])]
	for left, right in zip(boundaries[:-1], boundaries[1:], strict=False):
		start = int(left)
		end = int(right)
		step = start + max_region_frames
		while step < end:
			densified.append(step)
			step += max_region_frames
		densified.append(end)

	boundaries = np.unique(np.asarray(densified, dtype=np.int32))
	beat_count = int(max(0, boundaries.size - 1))
	tempo_ok = tempo_bpm is not None and MIN_RELIABLE_TEMPO_BPM <= tempo_bpm <= MAX_RELIABLE_TEMPO_BPM
	is_reliable = bool(beat_count >= MIN_RELIABLE_BEAT_COUNT and tempo_ok)
	return BeatTiming(
		boundaries=boundaries,
		tempo_bpm=tempo_bpm,
		beat_count=beat_count,
		is_reliable=is_reliable,
	)
