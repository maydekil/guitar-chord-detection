from __future__ import annotations

import builtins

import pytest

import chord_engine.analyze as analyze_module
import chord_engine.external_backends as external
from chord_engine.analysis_models import AnalysisMetadata, AnalysisResult, SourceMetadata


def test_analyze_audio_default_backend_uses_builtin_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
	result = AnalysisResult("1", SourceMetadata("song.wav", 1.0, 22050), AnalysisMetadata("builtin", []))
	run = analyze_module.PipelineRun(result=result, detected_key=None)
	monkeypatch.setattr(analyze_module, "analyze_audio_run", lambda path, include_diagnostics: run)

	assert analyze_module.analyze_audio("song.wav") is result


def test_analyze_audio_essentia_backend_dispatches_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
	result = AnalysisResult("1", SourceMetadata("song.wav", 1.0, 22050), AnalysisMetadata("essentia", []))
	monkeypatch.setattr(analyze_module, "analyze_with_essentia", lambda path: result)

	assert analyze_module.analyze_audio("song.wav", backend="essentia") is result


def test_analyze_audio_unsupported_backend_returns_controlled_error() -> None:
	with pytest.raises(analyze_module.AnalysisError) as exc:
		analyze_module.analyze_audio("song.wav", backend="unknown")

	assert exc.value.code == "ANALYSIS_BACKEND_UNSUPPORTED"


def test_essentia_backend_missing_dependency_returns_controlled_error(monkeypatch: pytest.MonkeyPatch) -> None:
	real_import = builtins.__import__

	def fake_import(name: str, *args: object, **kwargs: object) -> object:
		if name == "essentia.standard":
			raise ImportError("no essentia")
		return real_import(name, *args, **kwargs)

	monkeypatch.setattr(builtins, "__import__", fake_import)

	with pytest.raises(external.ExternalBackendError) as exc:
		external.analyze_with_essentia("song.wav")

	assert exc.value.code == "ESSENTIA_BACKEND_UNAVAILABLE"


def test_essentia_label_normalization_stays_in_mvp_vocabulary() -> None:
	raw_labels = ["C", "A:min", "Bb:maj", "F#m", "", "G#:min", "H:maj"]

	normalized = [external._normalize_essentia_label(label) for label in raw_labels]

	assert normalized == ["C", "Am", "A#", "F#m", "N", "G#m", "N"]


def test_chroma_is_reordered_from_c_canonical_to_essentia_a_first_hpcp() -> None:
	chroma = external.np.arange(24, dtype=external.np.float32).reshape(12, 2)

	hpcp = external._c_order_chroma_to_essentia_hpcp(chroma)

	assert hpcp.shape == (2, 12)
	assert hpcp[0].tolist() == [18, 20, 22, 0, 2, 4, 6, 8, 10, 12, 14, 16]


def test_essentia_frame_segments_are_contiguous_and_clamped() -> None:
	segments = external._segments_from_essentia_frames(
		["C", "C", "G:min", "G:min", "N"],
		[0.7, 0.9, 0.4, 0.6, 0.2],
		hop_length=512,
		sample_rate=22050,
		source_duration=0.20,
	)

	assert [(seg.chord, round(seg.start, 3), round(seg.end, 3)) for seg in segments] == [
		("C", 0.0, 0.046),
		("Gm", 0.046, 0.093),
		("N", 0.093, 0.2),
	]
	assert all(0.0 <= seg.confidence <= 1.0 for seg in segments)


def test_essentia_playable_compaction_reduces_short_chord_flicker() -> None:
	labels = (
		["C"] * 60
		+ ["D:min"] * 8
		+ ["C"] * 65
		+ ["G"] * 80
		+ ["A:min"] * 7
		+ ["G"] * 90
	)
	strengths = [0.70 for _ in labels]

	segments = external._segments_from_essentia_frames(
		labels,
		strengths,
		hop_length=512,
		sample_rate=22050,
		source_duration=8.0,
	)

	assert [segment.chord for segment in segments] == ["C", "G"]
	assert all((segment.end - segment.start) >= 1.35 for segment in segments)


def test_essentia_playable_compaction_preserves_strong_sustained_short_chord() -> None:
	labels = ["C"] * 90 + ["D:min"] * 40 + ["G"] * 120
	strengths = [0.65 for _ in range(90)] + [0.95 for _ in range(40)] + [0.72 for _ in range(120)]

	segments = external._segments_from_essentia_frames(
		labels,
		strengths,
		hop_length=512,
		sample_rate=22050,
		source_duration=8.0,
	)

	assert [segment.chord for segment in segments] == ["C", "Dm", "G"]
