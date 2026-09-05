"""Audio decoding and preprocessing for the chord engine."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 22050


@dataclass(frozen=True)
class AudioBuffer:
	"""Normalized internal representation used by downstream engine steps."""

	samples: np.ndarray
	sample_rate: int
	duration: float


class AudioDecodeError(Exception):
	"""Controlled error for decode/preprocess failures."""

	def __init__(self, code: str, message: str) -> None:
		super().__init__(message)
		self.code = code
		self.message = message

	def to_dict(self) -> dict[str, dict[str, str]]:
		return {"error": {"code": self.code, "message": self.message}}


def load_audio(path: str | Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> AudioBuffer:
	"""Decode audio, convert to mono, and resample to the target sample rate."""

	try:
		audio_path = Path(path)
		if not audio_path.exists() or not audio_path.is_file():
			raise AudioDecodeError("AUDIO_FILE_NOT_FOUND", "Audio file not found")

		samples, sample_rate = _decode_audio(audio_path)
		mono = _to_mono(samples)
		processed = _resample_if_needed(mono, sample_rate, target_sample_rate)
		processed = _normalize_if_needed(processed)
		_validate_audio(processed, target_sample_rate)

		duration = float(processed.shape[0] / target_sample_rate)
		return AudioBuffer(samples=processed, sample_rate=target_sample_rate, duration=duration)
	except AudioDecodeError:
		raise
	except Exception as exc:  # pragma: no cover - public boundary guard
		raise AudioDecodeError("AUDIO_DECODE_FAILED", "Unable to decode audio file") from exc


def _decode_audio(path: Path) -> tuple[np.ndarray, int]:
	try:
		samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
		return np.asarray(samples, dtype=np.float32), int(sample_rate)
	except Exception:
		try:
			samples, sample_rate = librosa.load(path, sr=None, mono=False, dtype=np.float32)
			return np.asarray(samples, dtype=np.float32), int(sample_rate)
		except Exception as exc:
			raise AudioDecodeError("AUDIO_DECODE_FAILED", "Unable to decode audio file") from exc


def _to_mono(samples: np.ndarray) -> np.ndarray:
	if samples.ndim == 1:
		return samples.astype(np.float32, copy=False)

	if samples.ndim != 2:
		raise AudioDecodeError("AUDIO_INVALID_FORMAT", "Audio channel layout is not supported")

	# Support both (frames, channels) and (channels, frames) conventions.
	if samples.shape[0] <= 8 and samples.shape[1] > samples.shape[0]:
		mono = np.mean(samples, axis=0)
	else:
		mono = np.mean(samples, axis=1)

	return mono.astype(np.float32, copy=False)


def _resample_if_needed(samples: np.ndarray, input_rate: int, target_rate: int) -> np.ndarray:
	if input_rate <= 0:
		raise AudioDecodeError("AUDIO_INVALID_SAMPLE_RATE", "Invalid source sample rate")

	if target_rate <= 0:
		raise AudioDecodeError("AUDIO_INVALID_SAMPLE_RATE", "Invalid target sample rate")

	if input_rate == target_rate:
		return samples.astype(np.float32, copy=False)

	factor = gcd(input_rate, target_rate)
	up = target_rate // factor
	down = input_rate // factor
	resampled = resample_poly(samples, up, down)
	return np.asarray(resampled, dtype=np.float32)


def _normalize_if_needed(samples: np.ndarray) -> np.ndarray:
	peak = float(np.max(np.abs(samples))) if samples.size > 0 else 0.0
	if peak > 1.0 and np.isfinite(peak):
		return (samples / peak).astype(np.float32, copy=False)
	return samples.astype(np.float32, copy=False)


def _validate_audio(samples: np.ndarray, sample_rate: int) -> None:
	if sample_rate <= 0:
		raise AudioDecodeError("AUDIO_INVALID_SAMPLE_RATE", "Invalid sample rate")

	if samples.ndim != 1:
		raise AudioDecodeError("AUDIO_INVALID_FORMAT", "Audio must be mono after preprocessing")

	if samples.size == 0:
		raise AudioDecodeError("AUDIO_EMPTY", "Audio contains no samples")

	if not np.isfinite(samples).all():
		raise AudioDecodeError("AUDIO_INVALID_FORMAT", "Audio contains non-finite samples")

	duration = float(samples.shape[0] / sample_rate)
	if duration <= 0.0:
		raise AudioDecodeError("AUDIO_EMPTY", "Audio duration is zero")
