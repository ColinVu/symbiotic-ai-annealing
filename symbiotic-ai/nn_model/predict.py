"""Run inference and write state predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .data import (
    DEFAULT_FEATURES_3D,
    DEFAULT_CSV_OUTPUTS,
    DEFAULT_GROUND_TRUTH,
    DEFAULT_LABELS,
    FeatureSource,
    load_ground_truth_counts,
    load_picklist_sample,
    normalize_features,
)
from .decode import DecodeMode, decode
from .models import build_model
from .states import IDX_TO_STATE


def load_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_model(
        config["model_type"],
        config["input_dim"],
        hidden=config.get("hidden", 0),
        tcn_channels=config.get("tcn_channels", 32),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    mean = np.array(config["mean"], dtype=np.float64)
    std = np.array(config["std"], dtype=np.float64)
    return model, config, mean, std


@torch.no_grad()
def predict_logits(model, features: np.ndarray, mean: np.ndarray, std: np.ndarray, device) -> np.ndarray:
    x = normalize_features(features, mean, std).astype(np.float32)
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    logits = model(x_t)[0].cpu().numpy()
    return logits


def write_frame_predictions(
    out_path: Path,
    frame_labels: np.ndarray,
    fps: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "time_s", "state"])
        for i, label_idx in enumerate(frame_labels):
            w.writerow([i, f"{i / fps:.3f}", IDX_TO_STATE[int(label_idx)]])


def write_segment_csv(
    out_path: Path,
    frame_labels: np.ndarray,
    fps: float,
) -> None:
    from .metrics import labels_to_segment_starts

    segments = labels_to_segment_starts(frame_labels, fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["state", "timestamp_start", "timestamp_end"])
        for i, (start, state) in enumerate(segments):
            end = segments[i + 1][0] if i + 1 < len(segments) else len(frame_labels) / fps
            w.writerow([state, f"{start:.6f}", f"{end:.6f}"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict states for picklist videos.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--picklist-ids", type=str, default=None, help="Comma-separated ids")
    ap.add_argument("--features-dir", type=Path, default=None)
    ap.add_argument(
        "--labels-dir",
        type=Path,
        default=DEFAULT_LABELS,
        help="Directory of csv_labels files (optional; omit or pass '' to run without ground truth)",
    )
    ap.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    ap.add_argument("--output-dir", type=Path, default=Path("nn_model/predictions"))
    ap.add_argument(
        "--decode-mode",
        choices=["argmax", "constrained", "count_constrained"],
        default="constrained",
    )
    ap.add_argument("--use-ground-truth-count", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, config, mean, std = load_checkpoint(args.checkpoint, device)
    feature_source: FeatureSource = config["feature_source"]

    if args.features_dir is None:
        features_dir = DEFAULT_FEATURES_3D if feature_source == "processed_3d" else DEFAULT_CSV_OUTPUTS
    else:
        features_dir = args.features_dir

    if args.picklist_ids:
        picklist_ids = [p.strip() for p in args.picklist_ids.split(",") if p.strip()]
    else:
        picklist_ids = config.get("val_ids", [])

    gt = load_ground_truth_counts(args.ground_truth) if args.use_ground_truth_count else {}
    decode_mode: DecodeMode = args.decode_mode
    labels_dir = args.labels_dir if args.labels_dir and str(args.labels_dir) else None

    for pid in picklist_ids:
        sample = load_picklist_sample(
            pid, feature_source, features_dir, labels_dir, gt, fps=30.0
        )
        logits = predict_logits(model, sample.features, mean, std, device)
        carry_count: Optional[int] = None
        if decode_mode == "count_constrained":
            carry_count = sample.object_count
            if carry_count is None:
                print(f"Warning: no object count for picklist_{pid}, using constrained decode")
                decode_mode_use: DecodeMode = "constrained"
            else:
                decode_mode_use = "count_constrained"
        else:
            decode_mode_use = decode_mode

        path = decode(logits, mode=decode_mode_use, carry_count=carry_count)
        stem = f"picklist_{pid}"
        write_frame_predictions(args.output_dir / f"{stem}_frames.csv", path, sample.fps)
        write_segment_csv(args.output_dir / f"{stem}_segments.csv", path, sample.fps)
        print(f"Wrote predictions for {stem} -> {args.output_dir}")


if __name__ == "__main__":
    main()
