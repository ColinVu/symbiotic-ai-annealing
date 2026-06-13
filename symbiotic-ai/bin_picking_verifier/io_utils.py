"""Load state CSVs, picklist JSON, and ArUco grid map."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CODE_TO_STATE = {
    "a": "PICK",
    "e": "CARRY_WITH",
    "i": "PLACE",
    "m": "CARRY_EMPTY",
}

_FRAME_COL_CANDIDATES = ("frame", "frame_index", "f", "start_frame", "frame_start")
_STATE_COL_CANDIDATES = ("state", "code", "s", "label", "state_code")
_HEADER_TOKENS = {x.lower() for x in _FRAME_COL_CANDIDATES} | {x.lower() for x in _STATE_COL_CANDIDATES}


@dataclass(frozen=True)
class StateSegment:
    """Inclusive frame range [start_frame, end_frame] in 0-based OpenCV indexing."""

    state: str
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class BinMapEntry:
    marker_id: int
    sku: str
    row: int
    col: int


@dataclass
class ArucoMap:
    dictionary: str
    rows: int
    cols: int
    bins: List[BinMapEntry]
    marker_id_to_entry: dict[int, BinMapEntry]
    sku_to_marker_id: dict[str, int]

    @classmethod
    def from_json_path(cls, path: Union[str, Path]) -> ArucoMap:
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.exception("Failed to read ArUco map %s: %s", path, e)
            raise

        dictionary = str(raw.get("dictionary", "DICT_5X5_1000"))
        grid = raw.get("grid") or {}
        rows = int(grid.get("rows", 4))
        cols = int(grid.get("cols", 6))
        bins_raw = raw.get("bins") or []
        bins: List[BinMapEntry] = []
        mid_map: dict[int, BinMapEntry] = {}
        sku_map: dict[str, int] = {}
        for b in bins_raw:
            entry = BinMapEntry(
                marker_id=int(b["marker_id"]),
                sku=str(b["sku"]).strip(),
                row=int(b["row"]),
                col=int(b["col"]),
            )
            bins.append(entry)
            mid_map[entry.marker_id] = entry
            sku_map[entry.sku] = entry.marker_id
        return cls(
            dictionary=dictionary,
            rows=rows,
            cols=cols,
            bins=bins,
            marker_id_to_entry=mid_map,
            sku_to_marker_id=sku_map,
        )


def load_picklist_json(path: Union[str, Path]) -> dict[str, Any]:
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.exception("Failed to read picklist JSON %s: %s", path, e)
        raise


def _normalize_code(ch: str) -> str:
    ch = str(ch).strip().lower()
    if len(ch) != 1 or ch not in _CODE_TO_STATE:
        raise ValueError(f"Invalid state code {ch!r}; expected one of a,e,i,m")
    return ch


def parse_glue_column(cell: str) -> List[Tuple[int, str]]:
    """Recover (frame, code) pairs from a single corrupted column."""
    s = str(cell).strip().replace(",", " ")
    pairs: List[Tuple[int, str]] = []
    for m in re.finditer(r"(\d+)\s*([AaEeIiMm])", s):
        pairs.append((int(m.group(1)), _normalize_code(m.group(2))))
    return pairs


def read_compact_state_table(path: Union[str, Path]) -> pd.DataFrame:
    """Load compact timeline: columns frame_index (int), code (str)."""
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


def _compact_to_segments(df: pd.DataFrame, total_frames: int) -> List[StateSegment]:
    """Convert compact frame_index + code table to inclusive segments (opencv 0-based)."""
    if total_frames <= 0:
        return []

    n = len(df)
    segments: List[StateSegment] = []
    max_idx = max(0, total_frames - 1)

    for i in range(n):
        fi = int(df.iloc[i]["frame_index"])
        code = str(df.iloc[i]["code"]).strip().lower()
        state = _CODE_TO_STATE[code]
        start = max(0, min(max_idx, fi))
        if i + 1 < n:
            fi_next = int(df.iloc[i + 1]["frame_index"])
            end = max(start, min(max_idx, fi_next - 1))
        else:
            end = max_idx
        segments.append(StateSegment(state=state, start_frame=start, end_frame=end))

    return segments


def _legacy_to_segments(df: pd.DataFrame, fps: float, total_frames: int) -> List[StateSegment]:
    """Legacy timestamp_start, timestamp_end, state -> inclusive frame indices."""
    if total_frames <= 0 or fps <= 0:
        return []

    max_idx = max(0, total_frames - 1)
    segments: List[StateSegment] = []

    lower = {c.lower(): c for c in df.columns}
    col_ts = lower.get("timestamp_start", "timestamp_start")
    col_te = lower.get("timestamp_end", "timestamp_end")
    col_st = lower.get("state", "state")

    for _, row in df.iterrows():
        t0 = float(row[col_ts])
        t1 = float(row[col_te])
        state_raw = str(row[col_st]).strip().upper().replace(" ", "_")

        start = int(round(t0 * fps))
        end = int(round(t1 * fps)) - 1
        start = max(0, min(max_idx, start))
        end = max(start, min(max_idx, end))
        segments.append(StateSegment(state=state_raw, start_frame=start, end_frame=end))

    return segments


def load_state_segments(
    csv_path: Union[str, Path],
    fps: float,
    total_frames: int,
) -> List[StateSegment]:
    """
    Load state labels: auto-detect legacy wide CSV vs compact frame+code format.

    All segments use 0-based frame indices consistent with OpenCV ``CAP_PROP_POS_FRAMES``.
    """
    csv_path = Path(csv_path)
    head = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()[:3]
    header = head[0].lower() if head else ""

    if "timestamp_start" in header and "timestamp_end" in header:
        df = pd.read_csv(csv_path)
        lower = {c.lower() for c in df.columns}
        expected = {"timestamp_start", "timestamp_end", "state"}
        if not expected.issubset(lower):
            raise ValueError(f"Legacy label CSV missing columns {expected}: {csv_path}")
        logger.info("Loaded legacy timestamp state CSV: %s", csv_path)
        return _legacy_to_segments(df, fps=fps, total_frames=total_frames)

    df = read_compact_state_table(csv_path)
    logger.info("Loaded compact frame_index state CSV: %s", csv_path)
    return _compact_to_segments(df, total_frames=total_frames)


def pick_segments_only(segments: List[StateSegment]) -> List[StateSegment]:
    """Return only PICK segments in temporal order."""
    return [s for s in segments if s.state == "PICK"]


def video_stem_to_picklist_paths(stem: str, json_dir: Path) -> List[Path]:
    """
    Resolve picklist JSON: try ``{stem}.json`` then 3-digit id from ``picklist_XXX``.
    """
    candidates = [json_dir / f"{stem}.json"]
    m = re.match(r"picklist_(\d+)$", stem, re.I)
    if m:
        num = m.group(1)
        if len(num) >= 3:
            candidates.append(json_dir / f"picklist_{num[-3:]}.json")
        candidates.append(json_dir / f"picklist_{num}.json")
    out: List[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out
