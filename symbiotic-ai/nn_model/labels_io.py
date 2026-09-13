"""Read/write picklist boundary label files.

Two formats are in play:

  htk  csv_labels/picklist_NNN.csv - one quoted, tab-separated row per
       boundary:   "0\tsil"
  csv  bulk_predict output - a header row then comma-separated rows:
       frame_index,code
       0,sil

`read_label_file` auto-detects both (plus plain whitespace separation), so
predicted labels can be fed straight back through the training pipeline.
This module deliberately has no dependency on states.py or data.py — it is
pure text parsing, which keeps data.py free to import it.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence, Tuple

LabelFormat = str  # "htk" | "csv"

Boundary = Tuple[int, str]


def _split_row(line: str) -> List[str]:
    if "\t" in line:
        return line.split("\t")
    if "," in line:
        return line.split(",")
    return line.split()


def parse_label_text(text: str) -> List[Boundary]:
    """Parse label file contents -> sorted [(frame_index, token)].

    Rows whose first field is not an integer are skipped, which quietly
    absorbs the `frame_index,code` header without special-casing it.
    """
    boundaries: List[Boundary] = []
    for line in text.splitlines():
        line = line.strip().strip('"').strip()
        if not line:
            continue
        parts = _split_row(line)
        if len(parts) < 2:
            continue
        head = parts[0].strip().strip('"')
        try:
            frame_idx = int(head)
        except ValueError:
            continue  # header row, or a comment
        token = parts[1].strip().strip('"').lower()
        if not token:
            continue
        boundaries.append((frame_idx, token))
    boundaries.sort(key=lambda x: x[0])
    return boundaries


def read_label_file(path: Path) -> Tuple[List[Boundary], List[str]]:
    """Read a label file in either format -> (boundaries, tokens)."""
    text = Path(path).read_text(encoding="utf-8")
    boundaries = parse_label_text(text)
    return boundaries, [tok for _, tok in boundaries]


def detect_format(path: Path) -> LabelFormat:
    """Best-effort format sniff, for reporting rather than parsing."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            return "htk"
        if "," in line:
            return "csv"
    return "csv"


def write_label_file(
    path: Path, boundaries: Sequence[Boundary], fmt: LabelFormat = "htk"
) -> None:
    """Write boundary rows in the requested format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "htk":
        lines = [f'"{int(fi)}\t{code}"' for fi, code in boundaries]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return
    if fmt == "csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_index", "code"])
            for fi, code in boundaries:
                w.writerow([int(fi), code])
        return
    raise ValueError(f"unknown label format: {fmt!r}")
