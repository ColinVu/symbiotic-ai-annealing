#!/usr/bin/env python3
"""
Build normalized 9-D HTK feature files from objectdetector/csv_outputs timeseries.

Reads per-picklist CSVs (stride-sampled), step/hold-expands to full frame rate,
computes deltas, applies global per-dimension z-score normalization (except
pick/place counts which use clamp+scale to [0,1]), and writes
processed_features_2d-style files (space-separated floats, no extension).

Feature order (9 dims):
  p_hold_raw, p_hold_smoothened, direction, pick_count, place_count,
  delta_p_hold, delta_direction, delta_pick_count, delta_place_count

Normalization:
  dims 0,1,2,5,6 -> global z-score
  dims 3,4,7,8   -> clamp [0, ARUCO_COUNT_MAX] / ARUCO_COUNT_MAX  (no z-score)

Usage:
  python3 build_aruco_htk_features.py
  python3 build_aruco_htk_features.py --csv-dir csv_outputs --output ../hmm-testing/processed_features_9d
  python3 build_aruco_htk_features.py --match-length-dir ../processed_features_2d -v
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FEATURE_NAMES: Tuple[str, ...] = (
    "p_hold_raw",
    "p_hold_smoothened",
    "direction",
    "pick_count",
    "place_count",
    "delta_p_hold",
    "delta_direction",
    "delta_pick_count",
    "delta_place_count",
)

CSV_SUFFIXES: Dict[str, str] = {
    "p_hold_raw": "_holding_timeseries.csv",
    "p_hold_smoothened": "_holding_timeseries_smoothed.csv",
    "direction": "_hand_direction_smoothed.csv",
    "net_aruco": "_aruco_net_timeseries_smoothed.csv",
}

VALUE_COLUMNS: Dict[str, str] = {
    "p_hold_raw": "p_hold",
    "p_hold_smoothened": "p_hold_smoothed",
    "direction": "direction_smoothed",
    "net_aruco": "net_aruco_smoothed",
}


def discover_stems(csv_dir: Path) -> List[str]:
    """Picklist stems that have all four required CSV files."""
    stems: set[str] = set()
    for path in csv_dir.glob("*_holding_timeseries.csv"):
        stem = path.name[: -len("_holding_timeseries.csv")]
        if all((csv_dir / f"{stem}{suffix}").is_file() for suffix in CSV_SUFFIXES.values()):
            stems.add(stem)
    return sorted(stems)


def load_timeseries_csv(csv_path: Path, value_col: str) -> Dict[int, float]:
    """Load frame_idx -> value; empty cells become NaN."""
    by_frame: Dict[int, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or value_col not in reader.fieldnames:
            raise ValueError(f"{csv_path}: missing column {value_col!r}")
        for row in reader:
            raw = row[value_col].strip()
            by_frame[int(row["frame_idx"])] = float(raw) if raw else float("nan")
    if not by_frame:
        raise ValueError(f"{csv_path}: no data rows")
    return by_frame


def forward_fill_nans(values: np.ndarray) -> np.ndarray:
    """Forward-fill NaN along a 1-D knot array; leading NaNs -> 0."""
    out = values.copy()
    if len(out) == 0:
        return out
    if np.isnan(out[0]):
        out[0] = 0.0
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    return out


def align_to_frames(ref_frames: np.ndarray, by_frame: Dict[int, float]) -> np.ndarray:
    """Map a frame-indexed series onto ref_frames; missing keys -> NaN."""
    return np.array([by_frame.get(int(fi), float("nan")) for fi in ref_frames], dtype=np.float64)


def load_picklist_sources(csv_dir: Path, stem: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load and align the four source CSVs for one picklist."""
    by_key: Dict[str, Dict[int, float]] = {}
    for key, suffix in CSV_SUFFIXES.items():
        path = csv_dir / f"{stem}{suffix}"
        by_key[key] = load_timeseries_csv(path, VALUE_COLUMNS[key])

    ref_frames = np.array(sorted(by_key["p_hold_raw"].keys()), dtype=np.int64)
    if len(ref_frames) == 0:
        raise ValueError(f"{stem}: no holding raw frames")

    p_hold_raw = align_to_frames(ref_frames, by_key["p_hold_raw"])
    p_hold_sm = forward_fill_nans(align_to_frames(ref_frames, by_key["p_hold_smoothened"]))
    direction = forward_fill_nans(align_to_frames(ref_frames, by_key["direction"]))
    net = forward_fill_nans(align_to_frames(ref_frames, by_key["net_aruco"]))

    knots = {
        "p_hold_raw": p_hold_raw,
        "p_hold_smoothened": p_hold_sm,
        "direction": direction,
        "pick_count": np.maximum(net, 0.0),
        "place_count": np.maximum(-net, 0.0),
    }
    return ref_frames, knots


def resolve_total_frames(
    stem: str,
    knot_frames: np.ndarray,
    match_length_dir: Optional[Path],
) -> int:
    """Full-rate frame count: match existing 2D file if present, else last knot + 1."""
    if match_length_dir is not None:
        ref = match_length_dir / stem
        if ref.is_file():
            with ref.open(encoding="utf-8", errors="replace") as f:
                n_lines = sum(1 for line in f if line.strip())
            if n_lines > 0:
                return n_lines
    return int(knot_frames[-1]) + 1


def expand_full_rate(
    knot_frames: np.ndarray,
    knot_values: Dict[str, np.ndarray],
    total_frames: int,
) -> Dict[str, np.ndarray]:
    """Step/hold expansion: each knot value is held constant until the next knot."""
    out: Dict[str, np.ndarray] = {}
    knot_idx = knot_frames.astype(np.int64)
    for key in ("p_hold_raw", "p_hold_smoothened", "direction", "pick_count", "place_count"):
        arr = np.empty(total_frames, dtype=np.float64)
        vals = knot_values[key]
        for i, start in enumerate(knot_idx):
            end = int(knot_idx[i + 1]) if i + 1 < len(knot_idx) else total_frames
            end = min(end, total_frames)
            arr[start:end] = vals[i]
        # Fill any frames before the first knot with the first knot value
        if knot_idx[0] > 0:
            arr[: knot_idx[0]] = vals[0]
        out[key] = arr
    return out


def frame_deltas(signal: np.ndarray) -> np.ndarray:
    """First-order delta with delta[0] = 0."""
    d = np.zeros_like(signal)
    if len(signal) > 1:
        d[1:] = np.diff(signal)
    return d


ARUCO_COUNT_MAX = 5.0  # counts clamped to [0, ARUCO_COUNT_MAX] then divided


def scale_count(arr: np.ndarray) -> np.ndarray:
    """Clamp to [0, ARUCO_COUNT_MAX] and divide to give [0, 1]."""
    return np.clip(arr, 0.0, ARUCO_COUNT_MAX) / ARUCO_COUNT_MAX


def build_feature_matrix(knot_frames: np.ndarray, knots: Dict[str, np.ndarray], total_frames: int) -> np.ndarray:
    """Assemble T x 9 feature matrix (unnormalized except pick/place which are pre-scaled to [0,1])."""
    full = expand_full_rate(knot_frames, knots, total_frames)
    # pick_count and place_count use clamp+scale instead of z-score; keep them raw
    # here so normalize_matrix can still be called uniformly — we override cols 3 & 4
    # with pre-scaled values and set their mean/std to 0/1 in compute_global_stats.
    pick_scaled = scale_count(full["pick_count"])
    place_scaled = scale_count(full["place_count"])
    cols = [
        full["p_hold_raw"],
        full["p_hold_smoothened"],
        full["direction"],
        pick_scaled,
        place_scaled,
        frame_deltas(full["p_hold_raw"]),
        frame_deltas(full["direction"]),
        frame_deltas(pick_scaled),
        frame_deltas(place_scaled),
    ]
    return np.column_stack(cols)


# Columns that are already on a fixed scale ([0,1] or derived deltas of [0,1])
# and should NOT be z-scored. Indices into the 9-D feature vector:
#   3 = pick_count (scaled), 4 = place_count (scaled)
#   7 = delta_pick_count,    8 = delta_place_count
PRESCALED_DIMS: Tuple[int, ...] = (3, 4, 7, 8)


def compute_global_stats(matrices: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean/std; pre-scaled dims get mean=0, std=1 (no-op normalization)."""
    if not matrices:
        raise RuntimeError("No feature matrices to normalize")
    X = np.vstack(matrices)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std = np.where(std < 1e-10, 1.0, std)
    # Leave pre-scaled columns unchanged
    for dim in PRESCALED_DIMS:
        mean[dim] = 0.0
        std[dim] = 1.0
    return mean, std


def normalize_matrix(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (matrix - mean) / std


def write_feature_file(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in matrix:
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")


def save_stats(output_dir: Path, mean: np.ndarray, std: np.ndarray) -> None:
    np.savez_compressed(output_dir / "normalization_stats.npz", mean=mean, std=std)
    with (output_dir / "normalization_stats.txt").open("w", encoding="utf-8") as f:
        f.write("# Per-dimension normalization statistics (z-score: (x - mean) / std)\n")
        f.write(f"# Dimensions: {len(mean)}\n")
        f.write("# " + ", ".join(FEATURE_NAMES) + "\n\n")
        f.write("Dimension\tName\tMean\tStd\n")
        for i, name in enumerate(FEATURE_NAMES):
            f.write(f"{i}\t{name}\t{mean[i]:.6f}\t{std[i]:.6f}\n")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Build normalized 9-D HTK features from csv_outputs timeseries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--csv-dir",
        type=Path,
        default=script_dir / "csv_outputs",
        help="Directory with per-picklist CSV timeseries (default: csv_outputs/)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=script_dir.parent / "hmm-testing" / "processed_features_9d",
        help="Output directory for normalized HTK feature files",
    )
    ap.add_argument(
        "--match-length-dir",
        type=Path,
        default=script_dir.parent / "processed_features_2d",
        help="Use line counts from this dir for full-rate length (default: ../processed_features_2d). "
        "Pass empty string to disable.",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    csv_dir = args.csv_dir.resolve()
    output_dir = args.output.resolve()
    match_dir: Optional[Path]
    if args.match_length_dir is None or str(args.match_length_dir).strip() == "":
        match_dir = None
    else:
        match_dir = args.match_length_dir.resolve()

    if not csv_dir.is_dir():
        logging.error("CSV directory does not exist: %s", csv_dir)
        return 1

    stems = discover_stems(csv_dir)
    if not stems:
        logging.error("No complete picklist CSV sets found in %s", csv_dir)
        return 1

    logging.info("Found %d picklist(s) in %s", len(stems), csv_dir)

    raw_by_stem: Dict[str, np.ndarray] = {}
    errors: List[Tuple[str, str]] = []

    for stem in stems:
        try:
            knot_frames, knots = load_picklist_sources(csv_dir, stem)
            total_frames = resolve_total_frames(stem, knot_frames, match_dir)
            matrix = build_feature_matrix(knot_frames, knots, total_frames)
            raw_by_stem[stem] = matrix
            logging.debug("%s: %d knots -> %d frames x %d dims", stem, len(knot_frames), *matrix.shape)
        except Exception as exc:
            logging.exception("Failed to build features for %s", stem)
            errors.append((stem, str(exc)))

    if not raw_by_stem:
        logging.error("No picklists built successfully")
        return 1

    mean, std = compute_global_stats(list(raw_by_stem.values()))
    logging.info("Global stats over %d picklists, %d dims", len(raw_by_stem), len(FEATURE_NAMES))
    logging.info("Mean range: [%.6f, %.6f]", mean.min(), mean.max())
    logging.info("Std range: [%.6f, %.6f]", std.min(), std.max())

    output_dir.mkdir(parents=True, exist_ok=True)
    save_stats(output_dir, mean, std)

    for stem, matrix in sorted(raw_by_stem.items()):
        normed = normalize_matrix(matrix, mean, std)
        out_path = output_dir / stem
        write_feature_file(out_path, normed)
        logging.info("Wrote %s (%d frames, %d dims)", out_path, len(normed), normed.shape[1])

    logging.info("=" * 60)
    logging.info("Done: %d file(s) -> %s", len(raw_by_stem), output_dir)
    if errors:
        logging.warning("%d picklist(s) failed:", len(errors))
        for stem, msg in errors:
            logging.warning("  - %s: %s", stem, msg)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
