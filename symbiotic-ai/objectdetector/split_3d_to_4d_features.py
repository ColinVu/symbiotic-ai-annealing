#!/usr/bin/env python3
"""
Transform 3-D HTK feature files into a new 3-D representation optimised for
pick/carry/place/empty HMM classification.

Reads hmm-testing/processed_features_3d files (p_hold, direction, net_aruco) and writes
hmm-testing/processed_features_3d_v2 files with:
  direction    -- smoothed palm direction, unchanged  [-1, 1]
  empty_conf   -- inverted, clipped p_hold signal     [0, 1]
  place_conf   -- normalised negative-only ArUco      [0, 1]

Transforms:
  direction  = direction  (no change)
  empty_conf = clip(0.2 - p_hold, 0, 0.2) / 0.2
               → 1.0 when p_hold≈0 (empty canyon), 0.0 when p_hold≥0.2
  place_conf = clip(max(-net_aruco, 0), 0, 3) / 3
               → 1.0 at max observed place-ArUco magnitude (3), 0.0 for pick/carry

Usage:
  python3 split_3d_to_4d_features.py
  python3 split_3d_to_4d_features.py --input ../hmm-testing/processed_features_3d --output ../hmm-testing/processed_features_3d_v2
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

_EMPTY_CONF_THRESHOLD = 0.2   # p_hold values at or above this → empty_conf = 0
_PLACE_CONF_MAX = 3.0         # maximum observed |negative net_aruco| in dataset


def compute_features(p_hold: float, direction: float, net_aruco: float) -> Tuple[float, float, float]:
    empty_conf = max(0.0, min(_EMPTY_CONF_THRESHOLD - p_hold, _EMPTY_CONF_THRESHOLD)) / _EMPTY_CONF_THRESHOLD
    place_conf = max(0.0, min(-net_aruco, _PLACE_CONF_MAX)) / _PLACE_CONF_MAX
    return direction, empty_conf, place_conf


def transform_line(line: str) -> str:
    parts = line.split()
    if len(parts) != 3:
        raise ValueError(f"expected 3 values, got {len(parts)}: {line!r}")
    direction, empty_conf, place_conf = compute_features(
        float(parts[0]), float(parts[1]), float(parts[2])
    )
    return f"{direction:.6e} {empty_conf:.6e} {place_conf:.6e}\n"


def transform_file(input_path: Path, output_path: Path) -> int:
    """Transform one picklist file; return number of frames written."""
    lines_out: List[str] = []
    with input_path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                lines_out.append(transform_line(stripped))
            except ValueError as exc:
                raise ValueError(f"{input_path}:{line_no}: {exc}") from exc

    if not lines_out:
        raise ValueError(f"{input_path}: no feature rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(lines_out)
    return len(lines_out)


def discover_input_files(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.iterdir() if p.is_file() and not p.name.startswith("."))


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Transform 3-D HTK features into direction / empty_conf / place_conf.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=script_dir.parent / "hmm-testing" / "processed_features_3d",
        help="Directory with 3-D HTK feature files (default: ../hmm-testing/processed_features_3d)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=script_dir.parent / "hmm-testing" / "processed_features_3d_v2",
        help="Output directory for transformed 3-D HTK feature files (default: ../hmm-testing/processed_features_3d_v2)",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_dir.is_dir():
        logging.error("Input directory does not exist: %s", input_dir)
        return 1

    input_files = discover_input_files(input_dir)
    if not input_files:
        logging.error("No feature files found in %s", input_dir)
        return 1

    logging.info("Found %d file(s) in %s", len(input_files), input_dir)
    logging.info("Transforms: direction (unchanged) | empty_conf (p_hold) | place_conf (neg ArUco)")

    errors: List[Tuple[str, str]] = []
    written = 0
    for input_path in input_files:
        output_path = output_dir / input_path.name
        try:
            n_frames = transform_file(input_path, output_path)
            written += 1
            logging.info("Wrote %s (%d frames)", output_path, n_frames)
        except Exception as exc:
            logging.exception("Failed to transform %s", input_path.name)
            errors.append((input_path.name, str(exc)))

    logging.info("=" * 60)
    logging.info("Done: %d file(s) -> %s", written, output_dir)
    if errors:
        logging.warning("%d file(s) failed:", len(errors))
        for name, msg in errors:
            logging.warning("  - %s: %s", name, msg)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
