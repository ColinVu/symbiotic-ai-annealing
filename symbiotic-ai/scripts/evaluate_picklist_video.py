#!/usr/bin/env python3
"""Evaluate or export carry-only predictions for one picklist video.

This script:
1) Runs frame-level inference on a video
2) Uses compact/manual state labels to extract CARRY_WITH frame intervals
3) Optionally flattens picklist JSON into expected label sequence (one label per carry segment)
4) Always exports per-frame predictions for CARRY_WITH intervals
5) If picklist JSON provided, computes segment-level Top-1 / Top-3 metrics
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Ensure imports work when running as a script from repo root.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symbiote_weak.inference.recognizer import ObjectRecognizer
from symbiote_weak.preprocessing.video_processor import process_video_frames
from symbiote_weak.state_detection.compact_timeline import (
    carry_with_pipeline_frame_intervals_1based,
)


@dataclass
class SegmentEval:
    segment_idx: int
    start_frame_1based: int
    end_frame_1based: int
    expected_label: str
    num_predicted_frames: int
    predicted_top1_label: str
    top1_hit: bool
    top3_hit: bool
    mean_top1_confidence: float


def _flatten_picklists(json_path: Path) -> List[str]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    picklists = data.get("picklists", [])
    if not isinstance(picklists, list):
        raise ValueError(f"Invalid picklists format in {json_path}")
    flat: List[str] = []
    for block in picklists:
        if not isinstance(block, list):
            raise ValueError(f"Invalid picklist block in {json_path}: {block!r}")
        flat.extend(str(x) for x in block)
    if not flat:
        raise ValueError(f"No picklist labels found in {json_path}")
    return flat


def _read_inference_csv(csv_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            top3_labels = [x.strip() for x in str(row.get("top_3_labels", "")).split(";") if x.strip()]
            rows.append(
                {
                    "frame_number": int(float(row["frame_number"])),
                    "predicted_label": str(row["predicted_label"]),
                    "confidence": float(row["confidence"]),
                    "top3_labels": top3_labels,
                }
            )
    return rows


def _predict_rows_from_embeddings(
    recognizer: ObjectRecognizer,
    embeddings: List[np.ndarray],
    frame_numbers: List[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for emb, fn in zip(embeddings, frame_numbers):
        top3 = recognizer.model.predict_top_k(emb, k=3)
        if not top3:
            continue
        rows.append(
            {
                "frame_number": int(fn),
                "predicted_label": str(top3[0][0]),
                "confidence": float(top3[0][1]),
                "top3_labels": [str(lbl) for lbl, _ in top3],
            }
        )
    return rows


def _evaluate_segments(
    intervals_1based: List[Tuple[int, int]],
    expected_labels: List[str],
    inference_rows: List[Dict[str, object]],
) -> List[SegmentEval]:
    segment_count = min(len(intervals_1based), len(expected_labels))
    by_frame: Dict[int, Dict[str, object]] = {int(r["frame_number"]): r for r in inference_rows}
    out: List[SegmentEval] = []

    for i in range(segment_count):
        start_f, end_f = intervals_1based[i]
        expected = expected_labels[i]
        seg_rows = [
            by_frame[f]
            for f in range(start_f, end_f + 1)
            if f in by_frame
        ]
        if not seg_rows:
            out.append(
                SegmentEval(
                    segment_idx=i,
                    start_frame_1based=start_f,
                    end_frame_1based=end_f,
                    expected_label=expected,
                    num_predicted_frames=0,
                    predicted_top1_label="",
                    top1_hit=False,
                    top3_hit=False,
                    mean_top1_confidence=0.0,
                )
            )
            continue

        top1_votes = Counter(str(r["predicted_label"]) for r in seg_rows)
        top1_label = top1_votes.most_common(1)[0][0]
        top1_hit = top1_label == expected

        top3_hit = False
        for r in seg_rows:
            if expected in r["top3_labels"]:
                top3_hit = True
                break

        mean_conf = sum(float(r["confidence"]) for r in seg_rows) / len(seg_rows)

        out.append(
            SegmentEval(
                segment_idx=i,
                start_frame_1based=start_f,
                end_frame_1based=end_f,
                expected_label=expected,
                num_predicted_frames=len(seg_rows),
                predicted_top1_label=top1_label,
                top1_hit=top1_hit,
                top3_hit=top3_hit,
                mean_top1_confidence=mean_conf,
            )
        )
    return out


def _carry_frame_predictions(
    intervals_1based: List[Tuple[int, int]],
    inference_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    by_frame: Dict[int, Dict[str, object]] = {int(r["frame_number"]): r for r in inference_rows}
    out: List[Dict[str, object]] = []
    for seg_idx, (start_f, end_f) in enumerate(intervals_1based):
        for f in range(start_f, end_f + 1):
            r = by_frame.get(f)
            if r is None:
                continue
            out.append(
                {
                    "segment_idx": seg_idx,
                    "frame_number": f,
                    "segment_start_frame": start_f,
                    "segment_end_frame": end_f,
                    "predicted_label": str(r["predicted_label"]),
                    "confidence": float(r["confidence"]),
                    "top_3_labels": ";".join(r["top3_labels"]),
                }
            )
    return out


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on picklist video.")
    parser.add_argument("--video", required=True, help="Video path")
    parser.add_argument("--model-dir", required=True, help="Model directory")
    parser.add_argument(
        "--picklist-json",
        required=False,
        default=None,
        help="Optional picklist JSON path. If omitted, script only exports carry-only predictions.",
    )
    parser.add_argument("--state-label-csv", required=True, help="State label CSV path")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Frame skip for carry-only extraction (default 1 = evaluate every frame)",
    )
    parser.add_argument("--threshold", type=float, default=50.0, help="Blur threshold")
    parser.add_argument(
        "--compact-frame-indexing",
        type=str,
        default="opencv0",
        choices=["opencv0", "pipeline1"],
        help="Frame indexing mode used in compact state CSV",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path for frame predictions CSV (default: .cursor/eval_<video_stem>_inference.csv)",
    )
    parser.add_argument(
        "--carry-preds-csv",
        default=None,
        help="Optional path for carry-only per-frame predictions CSV "
        "(default: .cursor/eval_<video_stem>_carry_predictions.csv)",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write JSON summary/report",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    model_dir = Path(args.model_dir).resolve()
    picklist_json = Path(args.picklist_json).resolve() if args.picklist_json else None
    state_csv = Path(args.state_label_csv).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    if picklist_json is not None and not picklist_json.exists():
        raise FileNotFoundError(f"Picklist JSON not found: {picklist_json}")
    if not state_csv.exists():
        raise FileNotFoundError(f"State labels CSV not found: {state_csv}")

    if args.output_csv:
        output_csv = Path(args.output_csv).resolve()
    else:
        output_csv = REPO_ROOT / ".cursor" / f"eval_{video_path.stem}_inference.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.carry_preds_csv:
        carry_preds_csv = Path(args.carry_preds_csv).resolve()
    else:
        carry_preds_csv = REPO_ROOT / ".cursor" / f"eval_{video_path.stem}_carry_predictions.csv"

    # 1) Build carry intervals
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    intervals = carry_with_pipeline_frame_intervals_1based(
        state_csv,
        total_frames=total_frames,
        frame_indexing=args.compact_frame_indexing,
    )

    # 2) Extract carry-only frame embeddings using the same preprocessing path as training
    recognizer = ObjectRecognizer(str(model_dir))

    def _no_cache(*_args, **_kwargs) -> None:
        return None

    cache_dir = str(REPO_ROOT / ".cursor" / "eval_cache")
    embeddings, _labels, _syn, _states, frame_indices = process_video_frames(
        video_path=str(video_path),
        label="eval",
        model=recognizer.clip_model,
        processor=recognizer.processor,
        cache_dir=cache_dir,
        save_frame_to_cache_func=_no_cache,
        threshold=args.threshold,
        frame_skip=args.frame_skip,
        state_detection_func=None,
        verbose=args.verbose,
        allowed_frame_intervals_1based=intervals,
    )

    # 3) Predict labels for extracted carry embeddings
    infer_rows = _predict_rows_from_embeddings(recognizer, embeddings, frame_indices)
    _write_csv(
        output_csv,
        [
            {
                "frame_number": r["frame_number"],
                "predicted_label": r["predicted_label"],
                "confidence": r["confidence"],
                "top_3_labels": ";".join(r["top3_labels"]),
            }
            for r in infer_rows
        ],
        fieldnames=["frame_number", "predicted_label", "confidence", "top_3_labels"],
    )

    # 4) Write carry-only per-frame output
    carry_rows = _carry_frame_predictions(intervals, infer_rows)
    _write_csv(
        carry_preds_csv,
        carry_rows,
        fieldnames=[
            "segment_idx",
            "frame_number",
            "segment_start_frame",
            "segment_end_frame",
            "predicted_label",
            "confidence",
            "top_3_labels",
        ],
    )

    print("\n=== Carry-Only Predictions ===")
    print(f"Video: {video_path}")
    print(f"Model: {model_dir}")
    print(f"Carry intervals: {len(intervals)}")
    print(f"Predicted carry frames: {len(carry_rows)}")
    print(f"Carry predictions CSV: {carry_preds_csv}")

    # Stop here if no ground-truth JSON was provided.
    if picklist_json is None:
        return

    # 5) Optional evaluation with expected labels
    expected_labels = _flatten_picklists(picklist_json)
    seg_evals = _evaluate_segments(intervals, expected_labels, infer_rows)

    n = len(seg_evals)
    if n == 0:
        raise RuntimeError("No segments to evaluate (check labels and picklist files).")

    with_preds = sum(1 for s in seg_evals if s.num_predicted_frames > 0)
    top1_hits = sum(1 for s in seg_evals if s.top1_hit)
    top3_hits = sum(1 for s in seg_evals if s.top3_hit)
    mean_conf = sum(s.mean_top1_confidence for s in seg_evals) / n

    print("\n=== Picklist Video Eval ===")
    print(f"Video: {video_path}")
    print(f"Model: {model_dir}")
    print(f"Inference CSV: {output_csv}")
    print(f"Carry predictions CSV: {carry_preds_csv}")
    print(f"Carry segments used: {n}")
    print(f"Segments with >=1 predicted frame: {with_preds}/{n}")
    print(f"Segment Top-1 accuracy: {top1_hits}/{n} = {top1_hits / n:.3f}")
    print(f"Segment Top-3 hit-rate: {top3_hits}/{n} = {top3_hits / n:.3f}")
    print(f"Mean segment top-1 confidence: {mean_conf:.3f}")

    print("\nPer-segment details:")
    print("idx,start,end,expected,pred_top1,frames,top1_hit,top3_hit,mean_conf")
    for s in seg_evals:
        print(
            f"{s.segment_idx},{s.start_frame_1based},{s.end_frame_1based},"
            f"{s.expected_label},{s.predicted_top1_label or 'NONE'},"
            f"{s.num_predicted_frames},{int(s.top1_hit)},{int(s.top3_hit)},"
            f"{s.mean_top1_confidence:.3f}"
        )

    if args.report_json:
        report_path = Path(args.report_json).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "video": str(video_path),
            "model_dir": str(model_dir),
            "picklist_json": str(picklist_json),
            "state_label_csv": str(state_csv),
            "inference_csv": str(output_csv),
            "frame_skip": args.frame_skip,
            "threshold": args.threshold,
            "compact_frame_indexing": args.compact_frame_indexing,
            "metrics": {
                "carry_segments_used": n,
                "segments_with_predictions": with_preds,
                "segment_top1_hits": top1_hits,
                "segment_top1_accuracy": top1_hits / n,
                "segment_top3_hits": top3_hits,
                "segment_top3_hit_rate": top3_hits / n,
                "mean_segment_top1_confidence": mean_conf,
            },
            "segments": [
                {
                    "segment_idx": s.segment_idx,
                    "start_frame_1based": s.start_frame_1based,
                    "end_frame_1based": s.end_frame_1based,
                    "expected_label": s.expected_label,
                    "predicted_top1_label": s.predicted_top1_label,
                    "num_predicted_frames": s.num_predicted_frames,
                    "top1_hit": s.top1_hit,
                    "top3_hit": s.top3_hit,
                    "mean_top1_confidence": s.mean_top1_confidence,
                }
                for s in seg_evals
            ],
        }
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote report JSON: {report_path}")


if __name__ == "__main__":
    main()

