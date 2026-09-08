"""Command-line interface for chord engine analysis contract."""

from __future__ import annotations

import argparse
import json
import os
import sys

from chord_engine.analyze import CONTRACT_VERSION, AnalysisError, analyze_audio
from chord_engine.evaluation import EvaluationError, evaluate_against_ground_truth
from chord_engine.external_backends import SUPPORTED_EXTERNAL_BACKENDS
from chord_engine.lyrics import LyricsError, transcribe_lyrics
from chord_engine.pitch import PitchShiftError, pitch_shift_audio
from chord_engine.regression import RegressionEvaluationError, evaluate_regression_set
from chord_engine.vocals import VocalRemovalError, remove_vocals


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="chord_engine")
	subparsers = parser.add_subparsers(dest="command", required=True)

	analyze_parser = subparsers.add_parser("analyze")
	analyze_parser.add_argument("path")
	analyze_parser.add_argument("--backend", choices=SUPPORTED_EXTERNAL_BACKENDS, default=None)

	lyrics_parser = subparsers.add_parser("transcribe-lyrics")
	lyrics_parser.add_argument("path")
	lyrics_parser.add_argument("--model", default="base")

	vocals_parser = subparsers.add_parser("remove-vocals")
	vocals_parser.add_argument("path")
	vocals_parser.add_argument("--output-root", required=True)
	vocals_parser.add_argument("--model", default="htdemucs")

	pitch_parser = subparsers.add_parser("pitch-shift-audio")
	pitch_parser.add_argument("path")
	pitch_parser.add_argument("--output-root", required=True)
	pitch_parser.add_argument("--semitones", type=int, required=True)

	eval_parser = subparsers.add_parser("evaluate-ground-truth")
	eval_parser.add_argument("audio_path")
	eval_parser.add_argument("annotation_path")
	eval_parser.add_argument("--clip-start", default=None)
	eval_parser.add_argument("--clip-end", default=None)

	eval_set_parser = subparsers.add_parser("evaluate-set")
	eval_set_parser.add_argument("manifest_path")
	eval_set_parser.add_argument("--baseline", default=None)
	eval_set_parser.add_argument("--output", default=None)

	return parser


def _print_json(payload: dict[str, object]) -> None:
	print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
	parser = _build_parser()
	args = parser.parse_args(argv)

	if args.command == "analyze":
		return _run_analyze(args.path, backend=args.backend)
	if args.command == "transcribe-lyrics":
		return _run_transcribe_lyrics(args.path, model_name=args.model)
	if args.command == "remove-vocals":
		return _run_remove_vocals(args.path, output_root=args.output_root, model_name=args.model)
	if args.command == "pitch-shift-audio":
		return _run_pitch_shift_audio(args.path, output_root=args.output_root, semitones=args.semitones)
	if args.command == "evaluate-ground-truth":
		return _run_evaluate_ground_truth(
			args.audio_path,
			args.annotation_path,
			clip_start=args.clip_start,
			clip_end=args.clip_end,
		)
	if args.command == "evaluate-set":
		return _run_evaluate_set(args.manifest_path, baseline_path=args.baseline, output_path=args.output)

	parser.print_usage(sys.stderr)
	return 2


def _run_analyze(path: str, *, backend: str | None = None) -> int:
	try:
		selected_backend = backend or os.getenv("GCD_CHORD_BACKEND", "builtin")
		result = analyze_audio(path, backend=selected_backend)
		_print_json(result.to_dict())
		return 0
	except AnalysisError as exc:
		_print_json(exc.to_dict())
		return 1


def _run_transcribe_lyrics(path: str, *, model_name: str) -> int:
	try:
		_print_json(transcribe_lyrics(path, model_name=model_name))
		return 0
	except LyricsError as exc:
		_print_json(exc.to_dict())
		return 1


def _run_remove_vocals(path: str, *, output_root: str, model_name: str) -> int:
	try:
		_print_json(remove_vocals(path, output_root=output_root, model_name=model_name))
		return 0
	except VocalRemovalError as exc:
		_print_json(exc.to_dict())
		return 1


def _run_pitch_shift_audio(path: str, *, output_root: str, semitones: int) -> int:
	try:
		_print_json(pitch_shift_audio(path, output_root=output_root, semitones=semitones))
		return 0
	except PitchShiftError as exc:
		_print_json(exc.to_dict())
		return 1
	except Exception as exc:  # pragma: no cover - boundary safeguard
		print(f"Unexpected pitch shift CLI failure: {exc}", file=sys.stderr)
		_print_json(
			{
				"version": CONTRACT_VERSION,
				"error": {
					"code": "PITCH_SHIFT_FAILED",
					"message": "Failed to pitch shift audio",
				},
			}
		)
		return 1
	except Exception as exc:  # pragma: no cover - boundary safeguard
		print(f"Unexpected vocal removal CLI failure: {exc}", file=sys.stderr)
		_print_json(
			{
				"version": CONTRACT_VERSION,
				"error": {
					"code": "VOCAL_REMOVAL_FAILED",
					"message": "Failed to remove vocals",
				},
			}
		)
		return 1
	except Exception as exc:  # pragma: no cover - boundary safeguard
		print(f"Unexpected lyrics CLI failure: {exc}", file=sys.stderr)
		_print_json(
			{
				"version": CONTRACT_VERSION,
				"error": {
					"code": "LYRICS_TRANSCRIPTION_FAILED",
					"message": "Failed to transcribe lyrics",
				},
			}
		)
		return 1
	except Exception as exc:  # pragma: no cover - boundary safeguard
		print(f"Unexpected CLI failure: {exc}", file=sys.stderr)
		_print_json(
			{
				"version": CONTRACT_VERSION,
				"error": {
					"code": "ANALYSIS_FAILED",
					"message": "Failed to analyze audio",
				},
			}
		)
		return 1


def _run_evaluate_ground_truth(
	audio_path: str,
	annotation_path: str,
	*,
	clip_start: str | None,
	clip_end: str | None,
) -> int:
	try:
		payload = evaluate_against_ground_truth(
			audio_path,
			annotation_path,
			clip_start_override=clip_start,
			clip_end_override=clip_end,
		)
		_print_json(payload)
		return 0
	except (AnalysisError, EvaluationError) as exc:
		if isinstance(exc, AnalysisError):
			_print_json(exc.to_dict())
		else:
			_print_json(exc.to_dict(version=CONTRACT_VERSION))
		return 1


def _run_evaluate_set(
	manifest_path: str,
	*,
	baseline_path: str | None,
	output_path: str | None,
) -> int:
	try:
		payload = evaluate_regression_set(manifest_path, baseline_path=baseline_path)
		if output_path:
			from pathlib import Path

			Path(output_path).parent.mkdir(parents=True, exist_ok=True)
			Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
		_print_json(payload)
		return 0 if payload.get("status") == "pass" else 1
	except RegressionEvaluationError as exc:
		_print_json(exc.to_dict(version=CONTRACT_VERSION))
		return 1
	except Exception as exc:  # pragma: no cover - boundary safeguard
		print(f"Unexpected CLI failure: {exc}", file=sys.stderr)
		_print_json(
			{
				"version": CONTRACT_VERSION,
				"error": {
					"code": "EVALUATION_FAILED",
					"message": "Failed to evaluate against ground truth",
				},
			}
		)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
