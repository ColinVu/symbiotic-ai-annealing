#!/usr/bin/env python3
"""
Repair compact picklist state CSVs (frame + a/e/i/m) when pasted into one column.

In-place mode (default): renames the original to ``<stem>_broken.csv``, then writes
a clean comma-separated CSV to the original path (same basename as before).

Examples
--------
  # Repair one file in place
  python scripts/repair_picklist_state_csv.py hmm-testing/picklist_labels/picklist_091.csv

  # Repair every .csv in a directory (skips ``*_broken.csv``)
  python scripts/repair_picklist_state_csv.py hmm-testing/picklist_labels/ --batch

  # Write repaired data to another path only; does not rename the input
  python scripts/repair_picklist_state_csv.py picklist_091.csv -o /tmp/fixed.csv

Uses ``symbiote_weak.state_detection.compact_timeline`` (run from repo ``symbiotic-ai/``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as ``python scripts/repair_picklist_state_csv.py`` from symbiotic-ai/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _write_compact_csv(path: Path, df) -> None:
    """Two-column CSV: frame_index,code (comma-separated; safe for integer + single-letter codes)."""
    lines = ["frame_index,code"] + [f"{int(r.frame_index)},{r.code}" for r in df.itertuples(index=False)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _repair_inplace(input_path: Path) -> Path:
    from symbiote_weak.state_detection.compact_timeline import read_compact_state_table

    original = input_path.resolve()
    if original.stem.endswith("_broken"):
        raise SystemExit(f"Refusing to repair backup file (stem ends with _broken): {original}")

    broken_path = original.with_name(f"{original.stem}_broken{original.suffix}")
    if broken_path.exists():
        raise SystemExit(f"Backup already exists, remove or rename it first: {broken_path}")

    df = read_compact_state_table(original)
    original.rename(broken_path)
    _write_compact_csv(original, df)
    return original


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair compact frame+state CSV files")
    parser.add_argument("path", help="Input .csv/.tsv file or directory (with --batch)")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Treat path as a directory and repair every *.csv inside",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write repaired CSV here only; does not rename the input file",
    )
    args = parser.parse_args()

    from symbiote_weak.state_detection.compact_timeline import read_compact_state_table

    if args.output:
        df = read_compact_state_table(args.path)
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        _write_compact_csv(outp, df)
        print(outp)
        return

    if args.batch:
        d = args.path
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(".csv"):
                continue
            if name.lower().endswith("_broken.csv"):
                continue
            p = Path(d) / name
            if p.stem.endswith("_broken"):
                continue
            out = _repair_inplace(p)
            print(out)
    else:
        out = _repair_inplace(Path(args.path))
        print(out)


if __name__ == "__main__":
    main()
