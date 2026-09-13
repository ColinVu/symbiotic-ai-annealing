"""Evaluate trained model against ground-truth labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import (
    DEFAULT_FEATURES_3D,
    DEFAULT_CSV_OUTPUTS,
    DEFAULT_GROUND_TRUTH,
    DEFAULT_LABELS,
    discover_picklist_ids,
    load_dataset,
    train_val_split,
)
from .decode import DecodeMode, decode
from .metrics import evaluate_predictions, format_report, labels_to_segment_starts, boundary_rmse
from .predict import load_checkpoint, predict_logits


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate NN state classifier.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--features-dir", type=Path, default=None)
    ap.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    ap.add_argument("--output-dir", type=Path, default=Path("nn_model/eval"))
    ap.add_argument(
        "--decode-mode",
        choices=["argmax", "constrained", "count_constrained"],
        default="constrained",
    )
    ap.add_argument("--split", choices=["val", "train", "all"], default="val")
    ap.add_argument("--use-ground-truth-count", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, config, mean, std = load_checkpoint(args.checkpoint, device)
    feature_source = config["feature_source"]

    if args.features_dir is None:
        features_dir = DEFAULT_FEATURES_3D if feature_source == "processed_3d" else DEFAULT_CSV_OUTPUTS
    else:
        features_dir = args.features_dir

    all_ids = discover_picklist_ids(feature_source, features_dir, args.labels_dir)
    train_ids, val_ids = train_val_split(all_ids, val_ratio=0.2, seed=42)
    if args.split == "val":
        eval_ids = val_ids
    elif args.split == "train":
        eval_ids = train_ids
    else:
        eval_ids = all_ids

    samples = load_dataset(eval_ids, feature_source, features_dir, args.labels_dir, args.ground_truth)
    y_true_list = []
    y_pred_list = []
    expected_counts = []
    boundary_rmses = []

    decode_mode: DecodeMode = args.decode_mode
    for sample in samples:
        logits = predict_logits(model, sample.features, mean, std, device)
        carry_count = sample.object_count if args.use_ground_truth_count else None
        mode = decode_mode
        if mode == "count_constrained" and carry_count is None:
            mode = "constrained"
        pred = decode(logits, mode=mode, carry_count=carry_count)
        if sample.labels is not None:
            y_true_list.append(sample.labels)
        y_pred_list.append(pred)
        expected_counts.append(sample.object_count)

        gt_starts = labels_to_segment_starts(sample.labels, sample.fps)
        pred_starts = labels_to_segment_starts(pred, sample.fps)
        rms, n = boundary_rmse(gt_starts, pred_starts)
        if n > 0:
            boundary_rmses.append(rms)

    metrics = evaluate_predictions(
        y_true_list,
        y_pred_list,
        expected_carry_counts=expected_counts if args.use_ground_truth_count else None,
    )
    if boundary_rmses:
        metrics["mean_boundary_rmse_sec"] = float(np.mean(boundary_rmses))

    report = format_report(metrics)
    if "mean_boundary_rmse_sec" in metrics:
        report += f"\n\nMean boundary RMSE: {metrics['mean_boundary_rmse_sec']:.4f} s"
    report += f"\nDecode mode: {decode_mode}"
    report += f"\nEval split: {args.split} ({len(samples)} picklists)"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"report_{args.decode_mode}_{args.split}.txt"
    out_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
