"""Read ordered object labels from the project's wide ground-truth CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List


def load_ground_truth_labels(ground_truth_csv: str, video_stem: str) -> List[str]:
    """Return non-empty pick labels for one video column, in row order."""
    path = Path(ground_truth_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"No header row in ground truth CSV: {path}")
        header_to_key = {
            raw.strip(): raw
            for raw in reader.fieldnames
            if raw is not None and raw.strip()
        }
        if video_stem not in header_to_key:
            raise ValueError(
                f"Video stem {video_stem!r} not found in ground truth columns. "
                f"Available: {sorted(header_to_key)!r}"
            )
        key = header_to_key[video_stem]
        labels = [str(row.get(key) or "").strip() for row in reader]

    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"No non-empty ground truth labels for column {video_stem!r} in {path}")
    return labels
