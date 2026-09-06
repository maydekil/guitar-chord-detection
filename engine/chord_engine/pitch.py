"""Cached audio pitch shifting for transpose playback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib


CONTRACT_VERSION = "1"


@dataclass(frozen=True)
class PitchShiftError(Exception):
	code: str
	message: str

	def to_dict(self) -> dict[str, object]:
		return {
			"version": CONTRACT_VERSION,
			"error": {
				"code": self.code,
				"message": self.message,
			},
		}


def pitch_shift_audio(audio_path: str, *, output_root: str, semitones: int) -> dict[str, object]:
	source_path = Path(audio_path)
	if not source_path.exists():
		raise PitchShiftError("PITCH_SHIFT_INPUT_MISSING", "Audio file tidak ditemukan.")
	if semitones < -11 or semitones > 11:
		raise PitchShiftError("PITCH_SHIFT_INVALID_STEPS", "Transpose audio harus di antara -11 sampai +11 semitone.")

	if semitones == 0:
		return _success_payload(source_path, source_path, semitones)

	output_dir = Path(output_root).expanduser()
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"{_build_cache_name(source_path, semitones)}.wav"
	if output_path.exists() and output_path.stat().st_size > 0:
		return _success_payload(source_path, output_path, semitones)

	try:
		import librosa
		import numpy as np
		import soundfile as sf
	except Exception as exc:
		raise PitchShiftError("PITCH_SHIFT_ENGINE_UNAVAILABLE", "Pitch shift membutuhkan librosa dan soundfile.") from exc

	try:
		audio, sample_rate = librosa.load(str(source_path), sr=None, mono=False)
		if audio.ndim == 1:
			shifted = librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=semitones)
		else:
			shifted_channels = [
				librosa.effects.pitch_shift(y=channel, sr=sample_rate, n_steps=semitones)
				for channel in audio
			]
			min_length = min(len(channel) for channel in shifted_channels)
			shifted = np.vstack([channel[:min_length] for channel in shifted_channels])
			shifted = shifted.T
		sf.write(str(output_path), shifted, sample_rate)
	except Exception as exc:
		raise PitchShiftError("PITCH_SHIFT_FAILED", "Gagal menyesuaikan transpose audio.") from exc

	return _success_payload(source_path, output_path, semitones)


def _success_payload(source_path: Path, output_path: Path, semitones: int) -> dict[str, object]:
	return {
		"version": CONTRACT_VERSION,
		"source": {
			"path": str(source_path),
			"semitones": semitones,
		},
		"audio": {
			"path": str(output_path),
			"format": "wav",
		},
	}


def _build_cache_name(source_path: Path, semitones: int) -> str:
	digest = hashlib.sha256()
	stat = source_path.stat()
	digest.update(str(source_path.resolve()).encode("utf-8"))
	digest.update(str(stat.st_size).encode("utf-8"))
	digest.update(str(int(stat.st_mtime)).encode("utf-8"))
	digest.update(str(semitones).encode("utf-8"))
	return f"{digest.hexdigest()[:24]}-{semitones:+d}"
