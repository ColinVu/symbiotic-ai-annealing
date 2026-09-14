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

SHELF_DIGIT_MAP = {"1": "c", "2": "d", "3": "e", "4": "f", "5": "g"}


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
    shelf_predicted_top1_label: str
    shelf_top1_hit: bool
    mean_top1_confidence: float


def shelf_for_video_stem(video_stem: str) -> Optional[str]:
    """Map picklist suffixes 1..5 to shelves c..g."""
    return SHELF_DIGIT_MAP.get(str(video_stem)[-1:])


def _predict_shelf_label(
    recognizer: ObjectRecognizer,
    processed_embedding: np.ndarray,
    shelf_prefix: Optional[str],
) -> Optional[str]:
    """Predict using only centroids belonging to the video's shelf."""
    if shelf_prefix is None:
        return None
    candidates = {
        label: centroid
        for label, centroid in recognizer.model.centroids.items()
        if str(label).lower().startswith(shelf_prefix)
    }
    if not candidates:
        return None
    x = recognizer.model._l2_normalize(processed_embedding.reshape(-1))
    return min(
        candidates,
        key=lambda label: recognizer.model.cosine_distance(x, candidates[label]),
    )


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
    shelf_prefix: Optional[str] = None,
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
                "shelf_predicted_label": _predict_shelf_label(
                    recognizer, processed, shelf_prefix
                ),
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
                    shelf_predicted_top1_label="",
                    shelf_top1_hit=False,
                    mean_top1_confidence=0.0,
                )
            )
            continue

        top1_votes = Counter(str(r["predicted_label"]) for r in seg_rows)
        top1_label = top1_votes.most_common(1)[0][0]
        top1_hit = top1_label == expected

        shelf_votes = Counter(
            str(r["shelf_predicted_label"])
            for r in seg_rows
            if r.get("shelf_predicted_label")
        )
        shelf_label = shelf_votes.most_common(1)[0][0] if shelf_votes else ""

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
                shelf_predicted_top1_label=shelf_label,
                shelf_top1_hit=shelf_label == expected,
                mean_top1_confidence=mean_conf,
            )
        )
    return out


def _metric_block(hits: int, count: int) -> Dict[str, Any]:
    return {
        "hits": int(hits),
        "count": int(count),
        "accuracy": (float(hits) / float(count)) if count else None,
    }


def _extended_metrics(
    *,
    video_stem: str,
    intervals: List[Tuple[int, int]],
    expected_labels: List[str],
    inference_rows: List[Dict[str, Any]],
    segment_evals: List[SegmentEval],
) -> Dict[str, Any]:
    """Compute frame, segment, and shelf-constrained metrics for one video."""
    by_frame: Dict[int, Dict[str, Any]] = {
        int(r["frame_number"]): r for r in inference_rows
    }
    frame_hits = 0
    frame_count = 0
    for (start_f, end_f), expected in zip(intervals, expected_labels):
        rows = [by_frame[f] for f in range(start_f, end_f + 1) if f in by_frame]
        frame_count += len(rows)
        frame_hits += sum(
            1 for row in rows if row.get("predicted_label") == expected
        )

    segment_top1_hits = sum(1 for item in segment_evals if item.top1_hit)
    shelf_segment_hits = sum(1 for item in segment_evals if item.shelf_top1_hit)
    segment_count = len(segment_evals)
    frame_block = _metric_block(frame_hits, frame_count)
    segment_block = _metric_block(segment_top1_hits, segment_count)
    shelf_segment_block = _metric_block(shelf_segment_hits, segment_count)

    shelf = shelf_for_video_stem(video_stem)
    by_shelf: Dict[str, Any] = {}
    if shelf is not None:
        by_shelf[shelf] = {
            "frame": frame_block,
            "segment_top1": segment_block,
            "shelf_constrained_segment_top1": shelf_segment_block,
        }

    return {
        "frame": frame_block,
        "segment_top1": segment_block,
        "shelf_constrained_segment_top1": shelf_segment_block,
        "by_shelf": by_shelf,
    }


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
    cache_label: str = "sweep_eval",
    embed_missing: bool = True,
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
        cache_label: Cache key prefix. Training uses the first picklist SKU;
            pass that here to reuse the shared cache. Default ``sweep_eval``
            preserves the previous evaluator key.
        embed_missing: If False, never call CLIP; skip frames absent from cache.
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

    recognizer = ObjectRecognizer(str(model_p), load_clip=bool(embed_missing))

    if eval_cache_dir:
        cache_dir = str(Path(eval_cache_dir).resolve())
    else:
        cache_dir = str(model_p.parent / f".sweep_eval_cache_{video_p.stem}")

    proc_stats: Dict[str, Any] = {}
    embeddings, _labels, _syn, _states, frame_indices = process_video_frames(
        video_path=str(video_p),
        label=str(cache_label),
        model=recognizer.clip_model,
        processor=recognizer.processor,
        cache_dir=cache_dir,
        threshold=float(threshold),
        frame_skip=int(frame_skip),
        state_detection_func=None,
        verbose=verbose,
        allowed_frame_intervals_1based=intervals,
        embed_missing=bool(embed_missing),
        stats_out=proc_stats,
    )

    video_stem = video_p.stem
    shelf_prefix = shelf_for_video_stem(video_stem)
    infer_rows = _predict_rows_from_embeddings(
        recognizer,
        embeddings,
        frame_indices,
        shelf_prefix=shelf_prefix,
    )
    expected_labels = _load_ground_truth_labels(str(gt_csv_p), video_p.stem)
    seg_evals = _evaluate_segments(intervals, expected_labels, infer_rows)
    extended = _extended_metrics(
        video_stem=video_stem,
        intervals=intervals,
        expected_labels=expected_labels,
        inference_rows=infer_rows,
        segment_evals=seg_evals,
    )

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
        "cache_label": str(cache_label),
        "cache_hits": int(proc_stats.get("cache_hits") or 0),
        "embed_missing": bool(embed_missing),
        "metrics": {
            "frame": extended["frame"],
            "segment_top1": extended["segment_top1"],
            "shelf_constrained_segment_top1": extended[
                "shelf_constrained_segment_top1"
            ],
            "frame_hits": extended["frame"]["hits"],
            "frames_with_predictions": extended["frame"]["count"],
            "frame_accuracy": extended["frame"]["accuracy"],
            "carry_segments_used": n,
            "segments_with_predictions": with_preds,
            "segment_top1_hits": top1_hits,
            "segment_top1_accuracy": top1_hits / n,
            "segment_top3_hits": top3_hits,
            "segment_top3_hit_rate": top3_hits / n,
            "shelf_constrained_segment_top1_hits": extended[
                "shelf_constrained_segment_top1"
            ]["hits"],
            "shelf_constrained_segment_top1_accuracy": extended[
                "shelf_constrained_segment_top1"
            ]["accuracy"],
            "by_shelf": extended["by_shelf"],
            "mean_segment_top1_confidence": mean_conf,
        },
        "segments": [asdict(s) for s in seg_evals],
    }
