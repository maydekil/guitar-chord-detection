"""Public and diagnostic analysis models shared by the engine pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from chord_engine.detector import KeyEstimate
from chord_engine.features import BeatTiming
from chord_engine.segmentation import ChordSegment

CONTRACT_VERSION = "1"


@dataclass(frozen=True)
class SourceMetadata:
    path: str
    duration: float
    sampleRate: int


@dataclass(frozen=True)
class AnalysisMetadata:
    algorithm: str
    chords: list[ChordSegment]
    detectedChords: list[ChordSegment] | None = None
    leadSheetChords: list[ChordSegment] | None = None
    leadSheetSource: str | None = None


@dataclass(frozen=True)
class AnalysisResult:
    version: str
    source: SourceMetadata
    analysis: AnalysisMetadata

    def to_dict(self) -> dict[str, object]:
        analysis: dict[str, object] = {
            "algorithm": self.analysis.algorithm,
            "chords": _segments_to_dicts(self.analysis.chords),
        }
        if self.analysis.detectedChords is not None:
            analysis["detectedChords"] = _segments_to_dicts(self.analysis.detectedChords)
        if self.analysis.leadSheetChords is not None:
            analysis["leadSheetChords"] = _segments_to_dicts(self.analysis.leadSheetChords)
        if self.analysis.leadSheetSource is not None:
            analysis["leadSheetSource"] = self.analysis.leadSheetSource
        return {
            "version": self.version,
            "source": {
                "path": self.source.path,
                "duration": self.source.duration,
                "sampleRate": self.source.sampleRate,
            },
            "analysis": analysis,
        }


def _segments_to_dicts(segments: list[ChordSegment]) -> list[dict[str, object]]:
    return [
        {
            "start": seg.start,
            "end": seg.end,
            "chord": seg.chord,
            "confidence": seg.confidence,
        }
        for seg in segments
    ]


@dataclass(frozen=True)
class PipelineRun:
    """Internal run result used for benchmark diagnostics."""

    result: AnalysisResult
    detected_key: KeyEstimate | None
    beat_timing: BeatTiming | None = None
    region_observations: list["MusicalRegionObservation"] | None = None
    persistence_events: list["PersistenceDecisionEvent"] | None = None
    correction_events: list["CorrectionEvent"] | None = None
    accepted_boundaries: list[float] | None = None
    rejected_novelty_peaks: list[dict[str, object]] | None = None
    global_decoder_path_score: float | None = None
    local_vs_decoded_region_changes: int = 0
    candidate_boundary_count: int = 0
    accepted_boundary_count_before_consolidation: int = 0
    accepted_boundary_count_after_consolidation: int = 0
    consolidated_boundary_count: int = 0
    boundary_cluster_count: int = 0
    mean_beats_between_accepted_boundaries: float = 0.0
    median_beats_between_accepted_boundaries: float = 0.0
    short_harmonic_region_count: int = 0
    harmonic_region_duration_distribution: dict[str, int] | None = None
    consolidated_cluster_examples: list[dict[str, object]] | None = None
    local_novelty_candidate_count: int = 0
    contextual_boundary_candidate_count: int = 0
    multi_resolution_accepted_boundary_count: int = 0
    rejected_local_only_boundary_count: int = 0
    short_medium_agreement_rate: float = 0.0
    mean_short_context_distance: float = 0.0
    mean_medium_context_distance: float = 0.0
    accepted_boundary_examples: list[dict[str, object]] | None = None
    rejected_local_only_examples: list[dict[str, object]] | None = None
    musical_timing: dict[str, object] | None = None


@dataclass(frozen=True)
class MusicalRegionObservation:
    """Beat-aware regional chord evidence used for persistence decisions."""

    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    winner_chord: str
    winner_confidence: float
    scores: dict[str, float]
    advantage_over_runner_up: float
    root_candidate_scores: dict[str, float] | None = None
    selected_root: str | None = None
    selected_root_confidence: float = 0.0
    major_quality_evidence: float = 0.0
    minor_quality_evidence: float = 0.0
    quality_margin: float = 0.0
    template_score: float = 0.0
    template_top_chord: str | None = None
    template_top_score: float = 0.0
    combined_score: float = 0.0
    is_quality_ambiguous: bool = False
    top_candidates: list[tuple[str, float]] | None = None
    local_best_chord: str | None = None
    local_best_score: float = 0.0
    decoded_chord: str | None = None
    beat_length: float = 0.0


@dataclass(frozen=True)
class PersistenceDecisionEvent:
    """Decision trace for keep vs switch behavior in musical time."""

    region_index: int
    region_start_seconds: float
    region_end_seconds: float
    action: str
    current_chord: str
    candidate_chord: str
    keep_score: float
    switch_score: float
    advantage: float
    candidate_confidence: float
    consecutive_support: int
    neighbor_persistence: float
    harmonic_change_evidence: float
    is_root_preserving_quality_switch: bool
    accepted_with_lower_switch_score: bool
    reason: str


@dataclass(frozen=True)
class CorrectionEvent:
    """Records a deterministic context-based correction decision for diagnostics."""

    index: int
    replaced_chord: str
    new_chord: str
    start: float
    end: float
    duration: float
    original_confidence: float
    new_confidence: float
    score: float
    harmonic_plausibility_before: float
    harmonic_plausibility_after: float


class AnalysisError(Exception):
    """Controlled orchestration error suitable for API/CLI conversion."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, object]:
        return {
            "version": CONTRACT_VERSION,
            "error": {
                "code": self.code,
                "message": self.message,
            },
        }
