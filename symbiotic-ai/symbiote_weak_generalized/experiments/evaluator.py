"""Evaluate a saved centroid model on one picklist video (carry segments).

Uses ``ground_truth.csv`` (wide ordered labels per video column) for expected
labels. Blur threshold defaults to 50 (sweep constant) for alignment with training.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..inference.recognizer import ObjectRecognizer
from ..preprocessing.video_processor import process_video_frames
from ..state_detection.compact_timeline import carry_with_pipeline_frame_intervals_1based

from .sweep_config import SWEEP_FIXED_THRESHOLD


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


def _load_ground_truth_labels(ground_truth_csv: str, video_stem: str) -> List[str]:
    """
    Load verified ordered labels from a wide CSV: header row = stems per column,
    each subsequent row = one pick index; empty cells end that column's sequence.
    """
    path = Path(ground_truth_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header row in ground truth CSV: {path}")

        # Map stripped stem -> actual DictReader key (raw header cell)
        header_to_key: Dict[str, str] = {}
        for raw_h in reader.fieldnames:
            if raw_h is None:
                continue
            s = raw_h.strip()
            if s:
                header_to_key[s] = raw_h

        if video_stem not in header_to_key:
            raise ValueError(
                f"Video stem {video_stem!r} not found in ground truth columns. "
                f"Available: {sorted(header_to_key.keys())!r}"
            )
        col_key = header_to_key[video_stem]

        labels: List[str] = []
        for row in reader:
            raw = (row or {}).get(col_key, "")
            if raw is None:
                continue
            label = str(raw).strip()
            if label:
                labels.append(label)

    if not labels:
        raise ValueError(f"No non-empty ground truth labels for column {video_stem!r} in {path}")
    return labels


def _predict_rows_from_embeddings(
    recognizer: ObjectRecognizer,
    embeddings: List[np.ndarray],
    frame_numbers: List[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for emb, fn in zip(embeddings, frame_numbers):
        # Match inference: hand neutralizer + CLIP adapter (if saved on model_dir)
        processed = recognizer._postprocess_embedding(np.asarray(emb, dtype=np.float64))
        top3 = recognizer.model.predict_top_k(processed, k=3)
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
    inference_rows: List[Dict[str, Any]],
) -> List[SegmentEval]:
    segment_count = min(len(intervals_1based), len(expected_labels))
    by_frame: Dict[int, Dict[str, Any]] = {int(r["frame_number"]): r for r in inference_rows}
    out: List[SegmentEval] = []

    for i in range(segment_count):
        start_f, end_f = intervals_1based[i]
        expected = expected_labels[i]
        seg_rows = [by_frame[f] for f in range(start_f, end_f + 1) if f in by_frame]
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


def evaluate_model(
    model_dir: str,
    video_path: str,
    ground_truth_csv: str,
    state_label_csv_path: str,
    *,
    frame_skip: int = 4,
    compact_frame_indexing: str = "opencv0",
    threshold: float = SWEEP_FIXED_THRESHOLD,
    verbose: bool = False,
    eval_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Segment-level Top-1 accuracy and Top-3 hit rate on CARRY_WITH frames.

    Args:
        model_dir: Saved weak-sup model directory (centroids + metadata).
        video_path: Evaluation video (stem must match a column in ``ground_truth_csv``).
        ground_truth_csv: Wide CSV with one column per video stem and ordered true labels.
        state_label_csv_path: Compact manual state CSV for the video.
        frame_skip: Must match training if comparing runs (default 4).
        compact_frame_indexing: ``opencv0`` or ``pipeline1``.
        threshold: Blur threshold; sweeps should pass :data:`SWEEP_FIXED_THRESHOLD` (50).
        verbose: Forwarded to ``process_video_frames``.
        eval_cache_dir: Optional directory for eval-time frame cache; default
            ``<model_dir>/../.sweep_eval_cache_<stem>`` under the parent of model_dir.
    """
    video_p = Path(video_path).resolve()
    model_p = Path(model_dir).resolve()
    gt_csv_p = Path(ground_truth_csv).resolve()
    state_csv = Path(state_label_csv_path).resolve()

    for p, name in (
        (video_p, "video"),
        (model_p, "model_dir"),
        (gt_csv_p, "ground_truth_csv"),
        (state_csv, "state_label_csv"),
    ):
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    cap = cv2.VideoCapture(str(video_p))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()

    intervals = carry_with_pipeline_frame_intervals_1based(
        state_csv,
        total_frames=total_frames,
        frame_indexing=compact_frame_indexing,
    )

    recognizer = ObjectRecognizer(str(model_p))

    if eval_cache_dir:
        cache_dir = str(Path(eval_cache_dir).resolve())
    else:
        cache_dir = str(model_p.parent / f".sweep_eval_cache_{video_p.stem}")

    embeddings, _labels, _syn, _states, frame_indices = process_video_frames(
        video_path=str(video_p),
        label="sweep_eval",
        model=recognizer.clip_model,
        processor=recognizer.processor,
        cache_dir=cache_dir,
        threshold=float(threshold),
        frame_skip=int(frame_skip),
        state_detection_func=None,
        verbose=verbose,
        allowed_frame_intervals_1based=intervals,
    )

    infer_rows = _predict_rows_from_embeddings(recognizer, embeddings, frame_indices)
    expected_labels = _load_ground_truth_labels(str(gt_csv_p), video_p.stem)
    seg_evals = _evaluate_segments(intervals, expected_labels, infer_rows)

    n = len(seg_evals)
    if n == 0:
        raise RuntimeError("No segments to evaluate (check picklist vs state CSV).")

    with_preds = sum(1 for s in seg_evals if s.num_predicted_frames > 0)
    top1_hits = sum(1 for s in seg_evals if s.top1_hit)
    top3_hits = sum(1 for s in seg_evals if s.top3_hit)
    mean_conf = sum(s.mean_top1_confidence for s in seg_evals) / n

    return {
        "video": str(video_p),
        "model_dir": str(model_p),
        "ground_truth_csv": str(gt_csv_p),
        "video_stem": video_p.stem,
        "state_label_csv": str(state_csv),
        "frame_skip": int(frame_skip),
        "threshold": float(threshold),
        "compact_frame_indexing": compact_frame_indexing,
        "metrics": {
            "carry_segments_used": n,
            "segments_with_predictions": with_preds,
            "segment_top1_hits": top1_hits,
            "segment_top1_accuracy": top1_hits / n,
            "segment_top3_hits": top3_hits,
            "segment_top3_hit_rate": top3_hits / n,
            "mean_segment_top1_confidence": mean_conf,
        },
        "segments": [asdict(s) for s in seg_evals],
    }
