"""Optional AI vocal removal support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import subprocess
import sys


CONTRACT_VERSION = "1"


@dataclass(frozen=True)
class VocalRemovalError(Exception):
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


def remove_vocals(audio_path: str, *, output_root: str, model_name: str = "htdemucs") -> dict[str, object]:
	_prepare_model_cache(output_root)
	try:
		import demucs  # noqa: F401
	except Exception as exc:
		raise VocalRemovalError(
			"VOCAL_REMOVAL_ENGINE_UNAVAILABLE",
			"AI Vocal Remove membutuhkan package Python 'demucs'. Install dependency engine lalu coba lagi.",
		) from exc

	source_path = Path(audio_path)
	if not source_path.exists():
		raise VocalRemovalError("VOCAL_REMOVAL_INPUT_MISSING", "Audio file tidak ditemukan.")

	cache_name = _build_cache_name(source_path, model_name)
	cache_root = Path(output_root).expanduser() / cache_name
	output_path = cache_root / model_name / source_path.stem / "no_vocals.wav"
	if output_path.exists() and output_path.stat().st_size > 0:
		return _success_payload(source_path, output_path)

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

	return _success_payload(source_path, output_path)


def _success_payload(source_path: Path, output_path: Path) -> dict[str, object]:
	return {
		"version": CONTRACT_VERSION,
		"source": {
			"path": str(source_path),
		},
		"audio": {
			"path": str(output_path),
			"stem": "instrumental",
		},
	}


def _build_cache_name(source_path: Path, model_name: str) -> str:
	digest = hashlib.sha256()
	digest.update(str(source_path.resolve()).encode("utf-8"))
	stat = source_path.stat()
	digest.update(str(stat.st_size).encode("utf-8"))
	digest.update(str(int(stat.st_mtime)).encode("utf-8"))
	digest.update(model_name.encode("utf-8"))
	return digest.hexdigest()[:24]


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
