"""Boundary RMSE evaluation for predicted HMM segment CSVs."""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from symbiote_weak.state_detection.two_stage import (
    boundary_rmse_seconds,
    boundary_timestamps_from_segments,
)

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def _discover_pairs(pred_dir: str, label_dir: str) -> List[Tuple[str, str, str]]:
    pdir = Path(pred_dir)
    ldir = Path(label_dir)
    pairs: List[Tuple[str, str, str]] = []
    for pred in sorted(pdir.glob("*.csv")):
        stem = pred.stem.replace("_states", "").replace("_predicted_states", "")
        gt = ldir / f"{stem}.csv"
        if gt.is_file():
            pairs.append((stem, str(pred), str(gt)))
    return pairs


def evaluate(pred_dir: str, label_dir: str, output_csv: str) -> float:
    pairs = _discover_pairs(pred_dir, label_dir)
    if not pairs:
        raise RuntimeError("No matched prediction/label CSV pairs found.")
    rows = []
    for stem, pred_csv, gt_csv in pairs:
        pred_df = pd.read_csv(pred_csv)
        gt_df = pd.read_csv(gt_csv)
        rmse = boundary_rmse_seconds(
            boundary_timestamps_from_segments(pred_df),
            boundary_timestamps_from_segments(gt_df),
        )
        rows.append({"id": stem, "boundary_rmse_seconds": rmse, "pred_csv": pred_csv, "gt_csv": gt_csv})
    out = pd.DataFrame(rows).sort_values("boundary_rmse_seconds")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    return float(out["boundary_rmse_seconds"].mean())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m symbiote.hmm_boundary_eval",
        description="Compute boundary RMSE between predicted state CSVs and GT CSV labels.",
    )
    p.add_argument("--pred-dir", required=True, help="Directory containing predicted CSV files.")
    p.add_argument("--label-dir", required=True, help="Directory containing ground-truth CSV labels.")
    p.add_argument("--output-csv", default="boundary_rmse_summary.csv", help="Output summary CSV path.")
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    avg = evaluate(args.pred_dir, args.label_dir, args.output_csv)
    print(f"[hmm_boundary_eval] average boundary_rmse_seconds={avg:.4f}")
    print(f"[hmm_boundary_eval] wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
