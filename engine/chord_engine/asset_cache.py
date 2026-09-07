"""Shared helpers for deterministic generated-audio cache keys."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _source_file_digest(source_path: Path, *parts: object) -> str:
    digest = hashlib.sha256()
    stat = source_path.stat()
    digest.update(str(source_path.resolve()).encode("utf-8"))
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(str(int(stat.st_mtime)).encode("utf-8"))
    for part in parts:
        digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()[:24]
