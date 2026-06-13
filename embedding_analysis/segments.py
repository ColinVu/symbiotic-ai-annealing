"""Map CARRY_WITH segments to per-frame embedding lists (aligned to ground truth order)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SegmentEmbeddings:
    segment_idx: int
    true_label: str
    frame_indices_1based: List[int]
    embeddings: np.ndarray  # (T, d) or empty (0, d)
    is_placeholder: bool


def build_segments(
    carry_intervals_1based: List[Tuple[int, int]],
    expected_labels: List[str],
    emb_by_frame: Dict[int, np.ndarray],
    frame_skip: int,
) -> List[SegmentEmbeddings]:
    """
    One segment per CARRY_WITH interval, in order, paired with expected_labels[i].

    If len(intervals) != len(expected_labels), we trim to the minimum and note in report.
    """
    n = min(len(carry_intervals_1based), len(expected_labels))
    if n < len(carry_intervals_1based) or n < len(expected_labels):
        # caller logs
        pass

    out: List[SegmentEmbeddings] = []
    for i in range(n):
        lo, hi = carry_intervals_1based[i]
        label = expected_labels[i]
        frames: List[int] = []
        rows: List[np.ndarray] = []
        for f in range(lo, hi + 1):
            if f % int(frame_skip) != 0:
                continue
            e = emb_by_frame.get(f)
            if e is not None:
                frames.append(f)
                rows.append(np.asarray(e, dtype=np.float64).reshape(-1))
        if not rows:
            dim = next(iter(emb_by_frame.values())).shape[0] if emb_by_frame else 512
            out.append(
                SegmentEmbeddings(
                    segment_idx=i,
                    true_label=label,
                    frame_indices_1based=[],
                    embeddings=np.zeros((0, int(dim)), dtype=np.float64),
                    is_placeholder=True,
                )
            )
        else:
            out.append(
                SegmentEmbeddings(
                    segment_idx=i,
                    true_label=label,
                    frame_indices_1based=frames,
                    embeddings=np.stack(rows, axis=0),
                    is_placeholder=False,
                )
            )
    return out


def middle_index_sorted(frames: List[int]) -> int:
    """Index into frames list (not 1-based frame) for middle sample."""
    if not frames:
        return -1
    return len(frames) // 2
