"""Load wide ground_truth.csv (one column per video stem)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Set


def load_ground_truth_column(path: str | Path, video_stem: str) -> List[str]:
    p = Path(path)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header: {p}")
        header_to_key: Dict[str, str] = {}
        for raw_h in reader.fieldnames:
            if raw_h is None:
                continue
            s = raw_h.strip()
            if s:
                header_to_key[s] = raw_h
        if video_stem not in header_to_key:
            raise KeyError(
                f"Column {video_stem!r} not in {p}. "
                f"Available: {sorted(header_to_key.keys())!r}"
            )
        col = header_to_key[video_stem]
        out: List[str] = []
        for row in reader:
            cell = (row or {}).get(col, "")
            if cell is None:
                continue
            t = str(cell).strip()
            if t:
                out.append(t)
    if not out:
        raise ValueError(f"No labels for {video_stem!r} in {p}")
    return out


def list_stems_in_ground_truth(path: str | Path) -> Set[str]:
    p = Path(path)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return set()
        return {h.strip() for h in reader.fieldnames if h and h.strip()}
