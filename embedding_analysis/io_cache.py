"""Load per-frame cached embeddings (label_frame_N_hash.npy)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# symbiote cache: {label}_frame_{frame_number}_{md5}.npy
_FRAME_RE = re.compile(r"^(.+)_frame_(\d+)_([a-f0-9]{32})\.npy$")


def load_cache_dir(cache_dir: str | Path) -> Dict[int, np.ndarray]:
    """
    Map 1-based frame index -> raw CLIP embedding.
    If duplicate frame indices exist, first file in sorted order wins (warn at higher level).
    """
    d: Path = Path(cache_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"Not a directory: {d}")
    by_frame: Dict[int, np.ndarray] = {}
    for name in sorted(os.listdir(d)):
        if not name.endswith(".npy") or name.endswith("_seg.npy"):
            continue
        m = _FRAME_RE.match(name)
        if not m:
            continue
        frame_1based = int(m.group(2))
        path = d / name
        by_frame[frame_1based] = np.load(path).astype(np.float64).reshape(-1)
    return by_frame


def candidate_frames_in_intervals(
    carry_intervals_1based: List[Tuple[int, int]],
    frame_skip: int,
) -> List[int]:
    """Frames the training pipeline would consider (1-based, inclusive intervals)."""
    out: List[int] = []
    for lo, hi in carry_intervals_1based:
        for f in range(lo, hi + 1):
            if f % int(frame_skip) == 0:
                out.append(f)
    return out
