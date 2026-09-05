from __future__ import annotations

import numpy as np

from chord_engine.features import PITCH_CLASS_ORDER
from chord_engine.templates import (
    MAJOR_INTERVALS,
    MINOR_INTERVALS,
    generate_chord_templates,
)


def _active_bins(template: np.ndarray) -> list[int]:
    return [int(i) for i, v in enumerate(template.tolist()) if v > 0.5]


def _rotated_bins(root_index: int, intervals: tuple[int, ...], size: int = 12) -> list[int]:
    return sorted(((root_index + step) % size) for step in intervals)


def test_exactly_24_templates_are_generated() -> None:
    templates = generate_chord_templates()
    assert len(templates) == 24


def test_exactly_12_major_templates_exist() -> None:
    templates = generate_chord_templates()
    major_names = [name for name in templates if not name.endswith("m")]
    assert len(major_names) == 12


def test_exactly_12_minor_templates_exist() -> None:
    templates = generate_chord_templates()
    minor_names = [name for name in templates if name.endswith("m")]
    assert len(minor_names) == 12


def test_every_template_has_12_bins() -> None:
    templates = generate_chord_templates()
    assert all(vec.shape == (12,) for vec in templates.values())


def test_all_template_values_are_finite() -> None:
    templates = generate_chord_templates()
    merged = np.stack(list(templates.values()), axis=0)
    assert np.isfinite(merged).all()


def test_c_major_active_bins_are_c_e_g() -> None:
    templates = generate_chord_templates()
    assert _active_bins(templates["C"]) == [0, 4, 7]


def test_a_minor_active_bins_are_a_c_e() -> None:
    templates = generate_chord_templates()
    assert _active_bins(templates["Am"]) == [0, 4, 9]


def test_f_sharp_major_matches_rotated_major_intervals() -> None:
    templates = generate_chord_templates()
    root = PITCH_CLASS_ORDER.index("F#")
    expected = _rotated_bins(root, MAJOR_INTERVALS)
    assert _active_bins(templates["F#"]) == expected


def test_c_sharp_minor_matches_rotated_minor_intervals() -> None:
    templates = generate_chord_templates()
    root = PITCH_CLASS_ORDER.index("C#")
    expected = _rotated_bins(root, MINOR_INTERVALS)
    assert _active_bins(templates["C#m"]) == expected


def test_every_chord_name_is_unique() -> None:
    templates = generate_chord_templates()
    names = list(templates.keys())
    assert len(names) == len(set(names))


def test_no_unsupported_chord_class_is_generated() -> None:
    templates = generate_chord_templates()
    unsupported = {
        "N",
        "C7",
        "Cmaj7",
        "Cm7",
        "Csus2",
        "Csus4",
        "Cdim",
        "Caug",
        "Db",
        "Eb",
        "Gb",
        "Ab",
        "Bb",
    }
    assert unsupported.isdisjoint(templates.keys())


def test_template_generation_is_deterministic() -> None:
    first = generate_chord_templates()
    second = generate_chord_templates()

    assert list(first.keys()) == list(second.keys())
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_template_pitch_class_order_matches_feature_order() -> None:
    templates = generate_chord_templates()

    assert PITCH_CLASS_ORDER == (
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    )

    # Validate that root templates align with pitch-class index positions.
    for idx, pitch_name in enumerate(PITCH_CLASS_ORDER):
        major_active = _active_bins(templates[pitch_name])
        minor_active = _active_bins(templates[f"{pitch_name}m"])
        assert idx in major_active
        assert idx in minor_active
