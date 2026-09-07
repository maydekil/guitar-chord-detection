"""Reusable helpers for generated audio assets such as pitch and vocal stems."""

from __future__ import annotations

from pathlib import Path

from chord_engine.command_errors import ENGINE_COMMAND_CONTRACT_VERSION


def _require_existing_audio_source(audio_path: str, *, error_factory, missing_code: str, missing_message: str) -> Path:
	source_path = Path(audio_path)
	if not source_path.exists():
		raise error_factory(missing_code, missing_message)
	return source_path


def _ensure_output_dir(output_root: str) -> Path:
	output_dir = Path(output_root).expanduser()
	output_dir.mkdir(parents=True, exist_ok=True)
	return output_dir


def _is_cached_audio_ready(output_path: Path) -> bool:
	return output_path.exists() and output_path.stat().st_size > 0


def _generated_audio_payload(
	source_path: Path,
	output_path: Path,
	*,
	source_extra: dict[str, object] | None = None,
	audio_extra: dict[str, object] | None = None,
) -> dict[str, object]:
	source: dict[str, object] = {"path": str(source_path)}
	if source_extra:
		source.update(source_extra)

	audio: dict[str, object] = {"path": str(output_path)}
	if audio_extra:
		audio.update(audio_extra)

	return {
		"version": ENGINE_COMMAND_CONTRACT_VERSION,
		"source": source,
		"audio": audio,
	}
