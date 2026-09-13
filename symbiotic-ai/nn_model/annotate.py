"""Run inference and annotate videos with predicted state labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from .data import (
    DEFAULT_FEATURES_3D,
    DEFAULT_CSV_OUTPUTS,
    DEFAULT_GROUND_TRUTH,
    DEFAULT_LABELS,
    DEFAULT_VIDEO_DIR,
    FeatureSource,
    load_ground_truth_counts,
    load_picklist_sample,
)
from .decode import DecodeMode, run_decode
from .predict import (
    load_checkpoint,
    predict_logits,
    write_frame_predictions,
    write_segment_csv,
)
from .states import IDX_TO_STATE

STATE_COLORS_BGR = {
    "PICK": (0, 255, 0),
    "CARRY_WITH": (255, 255, 0),
    "PLACE": (0, 165, 255),
    "CARRY_EMPTY": (0, 0, 255),
}


def resolve_video_path(video_dir: Path, picklist_id: str) -> Path:
    """Find picklist video under video_dir (tries common extensions/casing)."""
    stem = f"picklist_{picklist_id}"
    candidates = [
        video_dir / f"{stem}.MP4",
        video_dir / f"{stem}.mp4",
        video_dir / f"{stem}.MOV",
        video_dir / f"{stem}.mov",
        video_dir / f"picklist_{int(picklist_id)}.MP4",
        video_dir / f"picklist_{int(picklist_id)}.mp4",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No video found for picklist_{picklist_id} under {video_dir}")


def draw_state_overlay(
    frame_bgr: np.ndarray,
    state: str,
    frame_idx: int,
    time_sec: float,
    gt_state: Optional[str] = None,
) -> None:
    """Draw predicted state (and optional ground truth) on frame."""
    h, w = frame_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.9, min(w, h) / 900.0)
    thickness = max(2, int(round(scale * 2)))

    info = f"Frame {frame_idx}  |  {time_sec:.2f}s"
    cv2.putText(frame_bgr, info, (16, int(36 * scale)), font, scale * 0.6, (255, 255, 255), thickness, cv2.LINE_AA)

    pred_text = f"State: {state}"
    pred_color = STATE_COLORS_BGR.get(state, (255, 255, 255))
    cv2.putText(
        frame_bgr,
        pred_text,
        (16, h - int(50 * scale)),
        font,
        scale,
        pred_color,
        thickness,
        cv2.LINE_AA,
    )

    if gt_state is not None:
        gt_text = f"GT:    {gt_state}"
        gt_color = STATE_COLORS_BGR.get(gt_state, (200, 200, 200))
        cv2.putText(
            frame_bgr,
            gt_text,
            (16, h - int(16 * scale)),
            font,
            scale * 0.85,
            gt_color,
            thickness,
            cv2.LINE_AA,
        )
        if gt_state != state:
            cv2.rectangle(frame_bgr, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)


def annotate_video(
    video_path: Path,
    frame_labels: np.ndarray,
    output_path: Path,
    fps: Optional[float] = None,
    gt_labels: Optional[np.ndarray] = None,
) -> int:
    """Write annotated video; returns number of frames written."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps is None or fps <= 0:
        fps = video_fps
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, video_fps, (fw, fh))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer for {output_path}")

    frame_idx = 0
    n_labels = len(frame_labels)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        label_idx = int(frame_labels[min(frame_idx, n_labels - 1)]) if n_labels > 0 else 0
        state = IDX_TO_STATE[label_idx]
        gt_state = None
        if gt_labels is not None and frame_idx < len(gt_labels):
            gt_state = IDX_TO_STATE[int(gt_labels[frame_idx])]
        draw_state_overlay(frame, state, frame_idx, frame_idx / video_fps, gt_state=gt_state)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return frame_idx


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict states and annotate picklist videos.")
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
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--output-dir", type=Path, default=Path("nn_model/annotations"))
    ap.add_argument(
        "--decode-mode",
        choices=["argmax", "constrained", "count_constrained"],
        default="constrained",
    )
    ap.add_argument("--use-ground-truth-count", action="store_true")
    ap.add_argument("--show-gt", action="store_true", help="Overlay ground-truth labels from csv_labels")
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

    gt_counts = load_ground_truth_counts(args.ground_truth) if args.use_ground_truth_count else {}
    decode_mode: DecodeMode = args.decode_mode
    labels_dir = args.labels_dir if args.labels_dir and str(args.labels_dir) else None

    for pid in picklist_ids:
        sample = load_picklist_sample(
            pid, feature_source, features_dir, labels_dir, gt_counts, fps=30.0
        )
        logits = predict_logits(model, sample.features, mean, std, device)
        carry_count = sample.object_count if decode_mode == "count_constrained" else None
        if decode_mode == "count_constrained" and carry_count is None:
            print(f"Warning: no object count for picklist_{pid}, using constrained decode")
        path, mode_used = run_decode(logits, decode_mode, carry_count)

        stem = f"picklist_{pid}"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_frame_predictions(args.output_dir / f"{stem}_frames.csv", path, sample.fps)
        write_segment_csv(args.output_dir / f"{stem}_segments.csv", path, sample.fps)

        video_path = resolve_video_path(args.video_dir, pid)
        out_video = args.output_dir / f"{stem}_annotated.mp4"
        gt_labels = sample.labels if (args.show_gt and sample.labels is not None) else None
        n_frames = annotate_video(video_path, path, out_video, fps=sample.fps, gt_labels=gt_labels)
        print(
            f"{stem}: {video_path.name} + checkpoint -> {out_video} "
            f"({n_frames} frames, decode={mode_used})"
        )


if __name__ == "__main__":
    main()
