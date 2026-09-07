"""Optional AI vocal removal support."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

from chord_engine.asset_cache import _source_file_digest
from chord_engine.command_errors import EngineCommandError
from chord_engine.generated_audio_asset import (
	_generated_audio_payload,
	_is_cached_audio_ready,
	_require_existing_audio_source,
)


class VocalRemovalError(EngineCommandError):
	pass


def remove_vocals(audio_path: str, *, output_root: str, model_name: str = "htdemucs") -> dict[str, object]:
	_prepare_model_cache(output_root)
	try:
		import demucs  # noqa: F401
	except Exception as exc:
		raise VocalRemovalError(
			"VOCAL_REMOVAL_ENGINE_UNAVAILABLE",
			"AI Vocal Remove membutuhkan package Python 'demucs'. Install dependency engine lalu coba lagi.",
		) from exc

	source_path = _require_existing_audio_source(
		audio_path,
		error_factory=VocalRemovalError,
		missing_code="VOCAL_REMOVAL_INPUT_MISSING",
		missing_message="Audio file tidak ditemukan.",
	)

	cache_name = _source_file_digest(source_path, model_name)
	cache_root = Path(output_root).expanduser() / cache_name
	output_path = cache_root / model_name / source_path.stem / "no_vocals.wav"
	if _is_cached_audio_ready(output_path):
		return _generated_audio_payload(source_path, output_path, audio_extra={"stem": "instrumental"})

	cache_root.mkdir(parents=True, exist_ok=True)
	command = [
		sys.executable,
		"-m",
		"demucs",
		"--two-stems",
		"vocals",
		"--name",
		model_name,
		"--out",
		str(cache_root),
		str(source_path),
	]

	try:
		completed = subprocess.run(command, check=False, capture_output=True, text=True)
	except Exception as exc:
		raise VocalRemovalError("VOCAL_REMOVAL_FAILED", "Gagal menjalankan AI Vocal Remove.") from exc

	if completed.returncode != 0:
		raise VocalRemovalError("VOCAL_REMOVAL_FAILED", "AI Vocal Remove gagal memproses audio.")
	if not output_path.exists() or output_path.stat().st_size <= 0:
		raise VocalRemovalError("VOCAL_REMOVAL_OUTPUT_MISSING", "Output instrumental tidak ditemukan.")

	return _generated_audio_payload(source_path, output_path, audio_extra={"stem": "instrumental"})


def _prepare_model_cache(output_root: str) -> None:
	cache_root = Path(output_root).expanduser() / "model-cache"
	demucs_cache = cache_root / "demucs"
	huggingface_cache = cache_root / "huggingface"
	torch_cache = cache_root / "torch"
	for path in (demucs_cache, huggingface_cache, torch_cache):
		path.mkdir(parents=True, exist_ok=True)

	os.environ.setdefault("DEMUCS_CACHE", str(demucs_cache))
	os.environ.setdefault("HF_HOME", str(huggingface_cache))
	os.environ.setdefault("TORCH_HOME", str(torch_cache))
