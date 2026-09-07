"""Optional lyric transcription support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from chord_engine.command_errors import ENGINE_COMMAND_CONTRACT_VERSION, EngineCommandError


class LyricsError(EngineCommandError):
	pass


def transcribe_lyrics(audio_path: str, *, model_name: str = "base") -> dict[str, object]:
	try:
		import whisper  # type: ignore[import-not-found]
	except Exception as exc:
		raise LyricsError(
			"LYRICS_ENGINE_UNAVAILABLE",
			"Auto lyric membutuhkan package Python 'openai-whisper'. Install dependency engine lalu coba lagi.",
		) from exc

	try:
		model = whisper.load_model(model_name, download_root=str(_resolve_whisper_cache_dir()))
		result: dict[str, Any] = model.transcribe(audio_path, task="transcribe", fp16=False)
	except Exception as exc:
		raise LyricsError("LYRICS_TRANSCRIPTION_FAILED", "Gagal membuat lyric otomatis dari audio.") from exc

	lines: list[dict[str, object]] = []
	for segment in result.get("segments", []):
		if not isinstance(segment, dict):
			continue
		text = str(segment.get("text", "")).strip()
		if not text:
			continue
		start = _safe_seconds(segment.get("start"))
		end = _safe_seconds(segment.get("end"))
		if start is None:
			continue
		lines.append(
			{
				"start": start,
				"end": end,
				"text": text,
			}
		)

	return {
		"version": ENGINE_COMMAND_CONTRACT_VERSION,
		"source": {
			"path": audio_path,
		},
		"lyrics": {
			"format": "lrc",
			"language": result.get("language") if isinstance(result.get("language"), str) else None,
			"text": "\n".join(f"[{_format_lrc_time(line['start'])}]{line['text']}" for line in lines),
			"lines": lines,
		},
	}


def _safe_seconds(value: object) -> float | None:
	try:
		seconds = float(value)
	except (TypeError, ValueError):
		return None
	if seconds < 0:
		return None
	return seconds


def _resolve_whisper_cache_dir() -> Path:
	configured_cache = os.environ.get("GCD_WHISPER_CACHE_DIR")
	if configured_cache:
		cache_dir = Path(configured_cache).expanduser()
	else:
		cache_dir = Path.home() / ".cache" / "guitar-chord-detection" / "whisper"
	cache_dir.mkdir(parents=True, exist_ok=True)
	return cache_dir


def _format_lrc_time(seconds: object) -> str:
	safe_seconds = max(0.0, float(seconds))
	minutes = int(safe_seconds // 60)
	whole_seconds = int(safe_seconds % 60)
	centiseconds = int((safe_seconds - int(safe_seconds)) * 100)
	return f"{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"
