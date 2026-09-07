"""Shared numeric helpers for deterministic analysis math."""

from __future__ import annotations

import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray, *, min_norm: float = 1e-12) -> float:
    a_norm = float(np.linalg.norm(a, ord=2))
    b_norm = float(np.linalg.norm(b, ord=2))
    if a_norm <= min_norm or b_norm <= min_norm:
        return 0.0

    value = float(np.dot(a, b) / (a_norm * b_norm))
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -1.0, 1.0))


def _safe_pct(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float((numerator / denominator) * 100.0)
