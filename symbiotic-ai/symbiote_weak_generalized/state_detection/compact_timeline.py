"""
Parse compact picklist state CSVs: one row per state *boundary* (start frame + code).

Codes (single ASCII letter, case-insensitive):
  a -> PICK
  e -> CARRY_WITH
  i -> PLACE
  m -> CARRY_EMPTY

Each row marks the **first frame** where that state begins (see ``frame_indexing``).
Carry runs from that frame **inclusive** until the frame **before** the next state's
first frame (converted to timestamps for ``video_processor``).

Frame indexing
--------------
``frame_indexing="opencv0"`` (default): frame column is 0-based index as in OpenCV
``CAP_PROP_POS_FRAMES`` (first frame = 0). The weak pipeline uses 1-based
``frame_count`` after each read; we convert to timestamps so that
``frame_time = frame_num / fps`` matches embedded frames when ``frame_num`` is
1-based and equals ``opencv_index + 1``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .detector import HandState

_CODE_TO_STATE = {
    "a": HandState.PICK.value,
    "e": HandState.CARRY_WITH.value,
    "i": HandState.PLACE.value,
    "m": HandState.CARRY_EMPTY.value,
}

_FRAME_COL_CANDIDATES = ("frame", "frame_index", "f", "start_frame", "frame_start")
_STATE_COL_CANDIDATES = ("state", "code", "s", "label", "state_code")
_HEADER_TOKENS = {x.lower() for x in _FRAME_COL_CANDIDATES} | {x.lower() for x in _STATE_COL_CANDIDATES}


def _normalize_code(ch: str) -> str:
    ch = str(ch).strip().lower()
    if len(ch) != 1 or ch not in _CODE_TO_STATE:
        raise ValueError(f"Invalid state code {ch!r}; expected one of a,e,i,m")
    return ch


def parse_glue_column(cell: str) -> List[Tuple[int, str]]:
    """
    Recover (frame, code) pairs from a single corrupted column, e.g. '0m80a149e'.
    Also accepts line-based '0 m' glued without separator.
    """
    s = str(cell).strip().replace(",", " ")
    pairs: List[Tuple[int, str]] = []
    for m in re.finditer(r"(\d+)\s*([AaEeIiMm])", s):
        pairs.append((int(m.group(1)), _normalize_code(m.group(2))))
    return pairs


def read_compact_state_table(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a compact timeline file. Handles tab/comma sep, two columns, or one glued column.

    Returns columns ``frame_index`` (int, meaning per ``frame_indexing``) and ``code`` (str).
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: List[Tuple[int, str]] = []

    for line in raw:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\t,;]+", line)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            try:
                fi = int(float(parts[0]))
            except ValueError:
                if parts[0].lower() in _HEADER_TOKENS:
                    continue
                glued = parse_glue_column(line)
                rows.extend(glued)
                continue
            rows.append((fi, _normalize_code(parts[1])))
        elif len(parts) == 1:
            rows.extend(parse_glue_column(parts[0]))
        else:
            continue

    if not rows:
        raise ValueError(f"No rows parsed from {path}")

    df = pd.DataFrame(rows, columns=["frame_index", "code"])
    df = df.sort_values("frame_index").drop_duplicates(subset=["frame_index"], keep="first")
    df = df.reset_index(drop=True)
    return df


def repair_compact_state_csv(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    frame_indexing: str = "opencv0",
) -> Path:
    """
    Read possibly corrupted input, write a clean two-column TSV ``frame_index\\tcode``.

    If *output_path* is None, writes ``<stem>_repaired.tsv`` beside the input.
    """
    input_path = Path(input_path)
    df = read_compact_state_table(input_path)
    out = Path(output_path) if output_path else input_path.with_name(f"{input_path.stem}_repaired.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["frame_index\tcode"] + [f"{int(r.frame_index)}\t{r.code}" for r in df.itertuples(index=False)]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def compact_table_to_state_segments(
    df: pd.DataFrame,
    fps: float = 29.97,
    frame_indexing: str = "opencv0",
    video_duration_sec: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build ``timestamp_start``, ``timestamp_end``, ``state`` rows (one per *interval*).

    Consecutive rows define boundaries: state *k* runs from frame F_k inclusive
    until the frame before F_{k+1} (last frame of segment is F_{k+1}-1 in the same
    indexing as F_k).

    ``frame_indexing``:
      - ``opencv0``: frame column is 0-based; pipeline ``frame_num`` is 1-based with
        ``frame_num = opencv_index + 1``, so ``t = frame_num / fps = (opencv0+1)/fps``.
      - ``pipeline1``: frame column already matches 1-based ``frame_num`` in the embedder.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")

    records = []
    n = len(df)
    for i in range(n):
        fi = int(df.iloc[i]["frame_index"])
        code = str(df.iloc[i]["code"]).lower()
        state = _CODE_TO_STATE[code]

        if frame_indexing == "opencv0":
            t_start = (fi + 1) / fps
        elif frame_indexing == "pipeline1":
            t_start = fi / fps
        else:
            raise ValueError("frame_indexing must be 'opencv0' or 'pipeline1'")

        if i + 1 < n:
            fi_next = int(df.iloc[i + 1]["frame_index"])
            if frame_indexing == "opencv0":
                t_end = fi_next / fps
            else:
                t_end = (fi_next - 1) / fps + 1e-6
        else:
            if video_duration_sec is not None and video_duration_sec > t_start:
                t_end = video_duration_sec
            else:
                t_end = t_start + 1.0

        if t_end < t_start:
            t_end = t_start + 1e-6

        records.append(
            {
                "timestamp_start": float(t_start),
                "timestamp_end": float(t_end),
                "state": state,
            }
        )

    return pd.DataFrame.from_records(records)


def load_compact_state_csv_as_pipeline_df(
    csv_path: Union[str, Path],
    fps: float = 29.97,
    frame_indexing: str = "opencv0",
    video_duration_sec: Optional[float] = None,
) -> pd.DataFrame:
    """Parse file on disk and return the DataFrame expected by ``video_processor``."""
    df = read_compact_state_table(csv_path)
    return compact_table_to_state_segments(df, fps=fps, frame_indexing=frame_indexing, video_duration_sec=video_duration_sec)


def carry_with_pipeline_frame_intervals_1based(
    csv_path: Union[str, Path],
    total_frames: int,
    frame_indexing: str = "opencv0",
) -> List[Tuple[int, int]]:
    """
    For each CARRY_WITH segment (code ``e``) in a compact timeline CSV, return an
    inclusive ``[start, end]`` range in the same **1-based** frame index convention
    as ``symbiote_weak.preprocessing.video_processor`` (first decoded frame has
    ``frame_count == 1``; with ``opencv0``, row ``F`` marks the first opencv index
    ``F`` for that state, i.e. 1-based start ``F + 1``).

    The last ``e`` segment ends at ``min(total_frames, ...)`` so ranges stay in-bounds.
    """
    df = read_compact_state_table(csv_path)
    intervals: List[Tuple[int, int]] = []
    n = len(df)
    tf = max(int(total_frames), 0)
    for i in range(n):
        code = str(df.iloc[i]["code"]).strip().lower()
        if code != "e":
            continue
        fi = int(df.iloc[i]["frame_index"])
        if frame_indexing == "opencv0":
            start = fi + 1
            if i + 1 < n:
                end = int(df.iloc[i + 1]["frame_index"])
            else:
                end = tf
        elif frame_indexing == "pipeline1":
            start = fi
            if i + 1 < n:
                end = int(df.iloc[i + 1]["frame_index"]) - 1
            else:
                end = tf
        else:
            raise ValueError("frame_indexing must be 'opencv0' or 'pipeline1'")
        end = min(end, tf)
        if start <= end and start >= 1:
            intervals.append((start, end))
    return intervals


def carry_empty_pipeline_frame_intervals_1based(
    csv_path: Union[str, Path],
    total_frames: int,
    frame_indexing: str = "opencv0",
) -> List[Tuple[int, int]]:
    """
    For each CARRY_EMPTY segment (code ``m``) in a compact timeline CSV, return an
    inclusive ``[start, end]`` range in the same **1-based** frame convention as
    ``carry_with_pipeline_frame_intervals_1based`` and ``video_processor``.
    """
    df = read_compact_state_table(csv_path)
    intervals: List[Tuple[int, int]] = []
    n = len(df)
    tf = max(int(total_frames), 0)
    for i in range(n):
        code = str(df.iloc[i]["code"]).strip().lower()
        if code != "m":
            continue
        fi = int(df.iloc[i]["frame_index"])
        if frame_indexing == "opencv0":
            start = fi + 1
            if i + 1 < n:
                end = int(df.iloc[i + 1]["frame_index"])
            else:
                end = tf
        elif frame_indexing == "pipeline1":
            start = fi
            if i + 1 < n:
                end = int(df.iloc[i + 1]["frame_index"]) - 1
            else:
                end = tf
        else:
            raise ValueError("frame_indexing must be 'opencv0' or 'pipeline1'")
        end = min(end, tf)
        if start <= end and start >= 1:
            intervals.append((start, end))
    return intervals


def load_state_labels_auto(
    csv_path: Union[str, Path],
    fps: float = 29.97,
    frame_indexing: str = "opencv0",
    video_duration_sec: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load manual state labels: legacy wide CSV or compact frame+code format.

    Legacy columns: ``timestamp_start``, ``timestamp_end``, ``state``.
    """
    csv_path = Path(csv_path)
    head = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()[:3]
    header = head[0].lower() if head else ""
    if "timestamp_start" in header and "timestamp_end" in header:
        df = pd.read_csv(csv_path)
        expected = {"timestamp_start", "timestamp_end", "state"}
        if not expected.issubset(df.columns):
            raise ValueError(f"Legacy label CSV missing columns {expected}: {csv_path}")
        return df

    return load_compact_state_csv_as_pipeline_df(
        csv_path,
        fps=fps,
        frame_indexing=frame_indexing,
        video_duration_sec=video_duration_sec,
    )


__all__ = [
    "read_compact_state_table",
    "repair_compact_state_csv",
    "compact_table_to_state_segments",
    "load_compact_state_csv_as_pipeline_df",
    "carry_with_pipeline_frame_intervals_1based",
    "carry_empty_pipeline_frame_intervals_1based",
    "load_state_labels_auto",
]
