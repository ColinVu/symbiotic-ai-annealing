"""Picklist stem / numeric-id helpers shared by every CV stage."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

_STEM_RE = re.compile(r"^(?:picklist[_-]?)?(\d+)$", re.I)
_FILE_SUFFIXES = {".mp4", ".mov", ".m4v", ".csv", ".json", ".pt"}


def _as_stem(value: Union[str, Path]) -> str:
    text = str(value).strip()
    path = Path(text)
    if path.suffix.lower() in _FILE_SUFFIXES or "/" in text or "\\" in text:
        return path.stem
    return path.name


def id_from_stem(value: Union[str, Path]) -> str:
    """Return the numeric picklist id, preserving any zero-padding from the name."""
    stem = _as_stem(value)
    match = _STEM_RE.match(stem)
    if not match:
        raise ValueError(f"Cannot parse picklist id from {value!r}")
    return match.group(1)


def normalize_stem(value: Union[str, Path]) -> str:
    """Canonical video / label stem, e.g. ``picklist_001``."""
    stem = _as_stem(value)
    if stem.lower().startswith("picklist_"):
        pid = id_from_stem(stem)
        return f"picklist_{pid}"
    match = _STEM_RE.match(stem)
    if not match:
        raise ValueError(f"Cannot normalize picklist stem from {value!r}")
    return f"picklist_{match.group(1)}"


def label_csv_name(value: Union[str, Path]) -> str:
    return f"{normalize_stem(value)}.csv"
