"""Command-line interface for chord engine analysis contract."""

from __future__ import annotations

import argparse
import json
import sys

from chord_engine.analyze import CONTRACT_VERSION, AnalysisError, analyze_audio
from chord_engine.evaluation import EvaluationError, evaluate_against_ground_truth
from chord_engine.lyrics import LyricsError, transcribe_lyrics


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="chord_engine")
	subparsers = parser.add_subparsers(dest="command", required=True)

	analyze_parser = subparsers.add_parser("analyze")
	analyze_parser.add_argument("path")

	lyrics_parser = subparsers.add_parser("transcribe-lyrics")
	lyrics_parser.add_argument("path")
	lyrics_parser.add_argument("--model", default="base")

	eval_parser = subparsers.add_parser("evaluate-ground-truth")
	eval_parser.add_argument("audio_path")
	eval_parser.add_argument("annotation_path")
	eval_parser.add_argument("--clip-start", default=None)
	eval_parser.add_argument("--clip-end", default=None)

	return parser


def _print_json(payload: dict[str, object]) -> None:
	print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
	parser = _build_parser()
	args = parser.parse_args(argv)

	if args.command == "analyze":
		return _run_analyze(args.path)
	if args.command == "transcribe-lyrics":
		return _run_transcribe_lyrics(args.path, model_name=args.model)
	if args.command == "evaluate-ground-truth":
		return _run_evaluate_ground_truth(
			args.audio_path,
			args.annotation_path,
			clip_start=args.clip_start,
			clip_end=args.clip_end,
		)

	parser.print_usage(sys.stderr)
	return 2


def _run_analyze(path: str) -> int:
	try:
		result = analyze_audio(path)
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
