"""Evaluate model on all picklists in csv_labels; print aggregated 4x4 confusion matrix."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .data import (
    DEFAULT_FEATURES_3D,
    DEFAULT_GROUND_TRUTH,
    DEFAULT_LABELS,
    expand_boundary_labels,
    load_ground_truth_counts,
    load_processed_3d_features,
    parse_boundary_labels,
)
from .decode import DecodeMode, run_decode
from .metrics import evaluate_predictions, format_report
from .predict import load_checkpoint, predict_logits


def discover_labeled_picklists(labels_dir: Path) -> List[str]:
    """Picklist ids with non-empty boundary labels in csv_labels/."""
    ids: List[str] = []
    for path in labels_dir.glob("picklist_*.csv"):
        m = re.match(r"picklist_(\d+)$", path.stem, re.I)
        if not m:
            continue
        boundaries, _ = parse_boundary_labels(path)
        if boundaries:
            ids.append(m.group(1))
    return sorted(ids, key=int)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Predict all csv_labels picklists; print 4x4 state confusion matrix (no CSV output).",
    )
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_3D)
    ap.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    ap.add_argument(
        "--decode-mode",
        choices=["argmax", "constrained", "count_constrained"],
        default="constrained",
    )
    ap.add_argument("--use-ground-truth-count", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    labels_dir = args.labels_dir.resolve()
    features_dir = args.features_dir.resolve()
    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")
    if not features_dir.is_dir():
        raise SystemExit(f"Features directory not found: {features_dir}")

    picklist_ids = discover_labeled_picklists(labels_dir)
    if not picklist_ids:
        raise SystemExit(f"No non-empty picklist_*.csv files in {labels_dir}")

    device = torch.device(args.device)
    model, config, mean, std = load_checkpoint(args.checkpoint, device)
    gt_counts = load_ground_truth_counts(args.ground_truth) if args.use_ground_truth_count else {}
    decode_mode: DecodeMode = args.decode_mode

    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []
    evaluated = 0
    skipped: List[str] = []

    for pid in picklist_ids:
        feat_path = features_dir / f"picklist_{pid}"
        if not feat_path.is_file():
            skipped.append(f"picklist_{pid} (missing features)")
            continue

        label_path = labels_dir / f"picklist_{pid}.csv"
        boundaries, _ = parse_boundary_labels(label_path)
        if not boundaries:
            skipped.append(f"picklist_{pid} (empty labels)")
            continue

        try:
            features = load_processed_3d_features(feat_path)
        except (ValueError, OSError) as exc:
            skipped.append(f"picklist_{pid} ({exc})")
            continue

        n_frames = features.shape[0]
        y_true = expand_boundary_labels(boundaries, n_frames)
        logits = predict_logits(model, features, mean, std, device)
        carry_count: Optional[int] = gt_counts.get(pid) if args.use_ground_truth_count else None
        if decode_mode == "count_constrained" and carry_count is None:
            print(f"Warning: no object count for picklist_{pid}, using constrained decode")
        y_pred, _ = run_decode(logits, decode_mode, carry_count)

        min_len = min(len(y_true), len(y_pred))
        y_true_list.append(y_true[:min_len])
        y_pred_list.append(y_pred[:min_len])
        evaluated += 1

    if not y_true_list:
        raise SystemExit("No picklists evaluated (check features_dir vs csv_labels overlap).")

    metrics = evaluate_predictions(y_true_list, y_pred_list)
    report = format_report(metrics)
    print(f"Evaluated {evaluated} picklist(s) from {labels_dir}")
    print(f"Decode mode: {decode_mode}")
    if args.use_ground_truth_count:
        print("Object-count constraint: enabled")
    if skipped:
        print(f"Skipped {len(skipped)} picklist(s):")
        for line in skipped:
            print(f"  - {line}")
    print()
    print(report)


if __name__ == "__main__":
    main()
