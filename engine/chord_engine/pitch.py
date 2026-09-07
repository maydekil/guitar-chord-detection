"""Cached audio pitch shifting for transpose playback."""

from __future__ import annotations

from chord_engine.asset_cache import _source_file_digest
from chord_engine.command_errors import EngineCommandError
from chord_engine.generated_audio_asset import (
	_ensure_output_dir,
	_generated_audio_payload,
	_is_cached_audio_ready,
	_require_existing_audio_source,
)


class PitchShiftError(EngineCommandError):
	pass


def pitch_shift_audio(audio_path: str, *, output_root: str, semitones: int) -> dict[str, object]:
	source_path = _require_existing_audio_source(
		audio_path,
		error_factory=PitchShiftError,
		missing_code="PITCH_SHIFT_INPUT_MISSING",
		missing_message="Audio file tidak ditemukan.",
	)
	if semitones < -11 or semitones > 11:
		raise PitchShiftError("PITCH_SHIFT_INVALID_STEPS", "Transpose audio harus di antara -11 sampai +11 semitone.")

	if semitones == 0:
		return _generated_audio_payload(source_path, source_path, source_extra={"semitones": semitones}, audio_extra={"format": "wav"})

	output_dir = _ensure_output_dir(output_root)
	output_path = output_dir / f"{_source_file_digest(source_path, semitones)}-{semitones:+d}.wav"
	if _is_cached_audio_ready(output_path):
		return _generated_audio_payload(source_path, output_path, source_extra={"semitones": semitones}, audio_extra={"format": "wav"})

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

	return _generated_audio_payload(source_path, output_path, source_extra={"semitones": semitones}, audio_extra={"format": "wav"})
