"""The playback and evaluation entry points must share one pipeline run."""

from unittest.mock import Mock

import chord_engine.analyze as analysis
from chord_engine.analysis_models import AnalysisMetadata, AnalysisResult, PipelineRun, SourceMetadata


def test_production_analysis_omits_experimental_diagnostics(monkeypatch):
    result = AnalysisResult("1", SourceMetadata("song.wav", 1.0, 22050), AnalysisMetadata("test", []))
    run = PipelineRun(result=result, detected_key=None)
    pipeline = Mock(return_value=run)
    monkeypatch.setattr(analysis, "_run_pipeline", pipeline)

    assert analysis.analyze_audio("song.wav") is result
    pipeline.assert_called_once()
    assert pipeline.call_args.kwargs["include_diagnostics"] is False
    assert pipeline.call_args.kwargs["use_context_correction"] is True


def test_evaluation_gets_output_and_diagnostics_from_same_run(monkeypatch):
    result = AnalysisResult("1", SourceMetadata("song.wav", 1.0, 22050), AnalysisMetadata("test", []))
    run = PipelineRun(result=result, detected_key=None, musical_timing={"beatCount": 2})
    pipeline = Mock(return_value=run)
    monkeypatch.setattr(analysis, "_run_pipeline", pipeline)

    assert analysis.analyze_audio_run("song.wav") is run
    pipeline.assert_called_once()
    assert pipeline.call_args.kwargs["include_diagnostics"] is True
