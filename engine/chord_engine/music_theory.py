"""Music-theory helpers used by chord scoring, decoding, and diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chord_engine.analysis_config import (
    HARMONIC_PLAUSIBILITY_DIATONIC,
    HARMONIC_PLAUSIBILITY_DOMINANT_MAJOR_IN_MINOR,
    HARMONIC_PLAUSIBILITY_MODAL_OR_BORROWED,
    HARMONIC_PLAUSIBILITY_NON_DIATONIC,
    KEY_PRIOR_WEIGHT,
)

if TYPE_CHECKING:
    from chord_engine.detector import KeyEstimate

ROOT_TO_PC = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}
ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _key_prior_bonus(chord_name: str, key_estimate: KeyEstimate) -> float:
    root_pc = _chord_root_pc(chord_name)
    if root_pc is None:
        return 0.0

    _, quality = _parse_chord_quality(chord_name)
    match = _diatonic_match_score(root_pc, quality, key_estimate)
    return KEY_PRIOR_WEIGHT * key_estimate.confidence * match


def _diatonic_match_score(root_pc: int, quality: str | None, key_estimate: KeyEstimate) -> float:
    if key_estimate.mode == "major":
        diatonic_chords = {
            0: "major",
            2: "minor",
            4: "minor",
            5: "major",
            7: "major",
            9: "minor",
            11: "diminished",
        }
        scale_intervals = (0, 2, 4, 5, 7, 9, 11)
    else:
        diatonic_chords = {
            0: "minor",
            2: "diminished",
            3: "major",
            5: "minor",
            7: "minor",
            8: "major",
            10: "major",
        }
        scale_intervals = (0, 2, 3, 5, 7, 8, 10)

    interval = (root_pc - key_estimate.tonic_pc) % 12
    if interval not in scale_intervals:
        return 0.0

    expected_quality = diatonic_chords.get(interval)
    if expected_quality is None:
        return 0.0
    if expected_quality == quality:
        return 1.0
    return 0.30


def _chord_root_pc(chord_name: str) -> int | None:
    if chord_name.endswith("dim"):
        name = chord_name[:-3]
    elif chord_name.endswith("m"):
        name = chord_name[:-1]
    else:
        name = chord_name
    return ROOT_TO_PC.get(name)


def _root_name_from_pc(pc: int) -> str:
    return ROOT_NAMES[pc % 12]


def _parse_chord_quality(label: str) -> tuple[str | None, str | None]:
    if label == "N":
        return None, None
    if label.endswith("dim"):
        root = label[:-3]
        return root, "diminished"
    if label.endswith("m"):
        root = label[:-1]
        return root, "minor"
    return label, "major"


def _is_root_preserving_quality_switch(current: str, candidate: str) -> bool:
    current_root, current_quality = _parse_chord_quality(current)
    candidate_root, candidate_quality = _parse_chord_quality(candidate)
    if current_root is None or candidate_root is None:
        return False
    if current_root != candidate_root:
        return False
    if current_quality == candidate_quality:
        return False
    return True


def _diatonic_chord_labels_for_key(key_estimate: KeyEstimate) -> set[str]:
    if key_estimate.mode == "major":
        qualities = {
            0: "",
            2: "m",
            4: "m",
            5: "",
            7: "",
            9: "m",
        }
    else:
        qualities = {
            0: "m",
            3: "",
            5: "m",
            7: "m",
            8: "",
            10: "",
        }
    return {
        f"{_root_name_from_pc((key_estimate.tonic_pc + interval) % 12)}{suffix}"
        for interval, suffix in qualities.items()
    }


def _is_diatonic_chord(chord_name: str, detected_key: KeyEstimate | None) -> bool:
    if detected_key is None or chord_name == "N":
        return False

    root_name, quality = _parse_chord_quality(chord_name)
    root_pc = ROOT_TO_PC.get(root_name)
    if root_pc is None:
        return False

    interval = (root_pc - detected_key.tonic_pc) % 12
    if detected_key.mode == "major":
        diatonic_chords = {
            0: "major",
            2: "minor",
            4: "minor",
            5: "major",
            7: "major",
            9: "minor",
            11: "diminished",
        }
    else:
        diatonic_chords = {
            0: "minor",
            2: "diminished",
            3: "major",
            5: "minor",
            7: "minor",
            8: "major",
            10: "major",
        }

    expected_quality = diatonic_chords.get(interval)
    if expected_quality is None:
        return False
    return expected_quality == quality


def _harmonic_plausibility(chord_name: str, detected_key: KeyEstimate | None) -> float:
    """Return bounded plausibility used as a soft context signal for correction."""
    if chord_name == "N":
        return HARMONIC_PLAUSIBILITY_MODAL_OR_BORROWED
    if detected_key is None:
        return HARMONIC_PLAUSIBILITY_MODAL_OR_BORROWED

    if _is_diatonic_chord(chord_name, detected_key):
        return HARMONIC_PLAUSIBILITY_DIATONIC

    root_name, quality = _parse_chord_quality(chord_name)
    root_pc = ROOT_TO_PC.get(root_name)
    if root_pc is None:
        return HARMONIC_PLAUSIBILITY_NON_DIATONIC

    interval = (root_pc - detected_key.tonic_pc) % 12
    if detected_key.mode == "minor" and interval == 7 and quality == "major":
        return HARMONIC_PLAUSIBILITY_DOMINANT_MAJOR_IN_MINOR

    return HARMONIC_PLAUSIBILITY_NON_DIATONIC


def _harmonic_relationship_strength(prev_chord: str, curr_chord: str, detected_key: KeyEstimate | None) -> float:
	if prev_chord == "N" or curr_chord == "N":
		return 0.20

	prev_root, prev_quality = _parse_chord_quality(prev_chord)
	curr_root, curr_quality = _parse_chord_quality(curr_chord)
	if prev_root is None or curr_root is None:
		return 0.0
	if prev_root == curr_root and prev_quality != curr_quality:
		return 0.75

	prev_pc = _chord_root_pc(prev_chord)
	curr_pc = _chord_root_pc(curr_chord)
	if prev_pc is None or curr_pc is None:
		return 0.0

	interval = (curr_pc - prev_pc) % 12
	if interval in (5, 7):
		return 0.70

	if abs((curr_pc - prev_pc) % 12) in (3, 9) and prev_quality != curr_quality:
		return 0.30

	if detected_key is not None and _is_diatonic_chord(prev_chord, detected_key) and _is_diatonic_chord(curr_chord, detected_key):
		return 0.44

	return 0.22
