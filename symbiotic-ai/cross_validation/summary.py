"""Cross-fold metric aggregation for the combined pipeline CV run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .layout import CvLayout


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _ratio(num: float, den: float) -> Optional[float]:
    if not den:
        return None
    return float(num) / float(den)


def _seg_frame_acc(rec: Dict[str, Any]) -> Optional[float]:
    seg = rec.get("segmentation") or {}
    acc = seg.get("frame_accuracy")
    return float(acc) if acc is not None else None


def _test_metric_acc(test: Dict[str, Any], hits_key: str, count_key: str) -> Optional[float]:
    hits = int(test.get(hits_key) or 0)
    count = int(test.get(count_key) or 0)
    return _ratio(hits, count)


def _coverage_totals(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    keys = (
        "candidate_frames",
        "cache_hits",
        "skipped_missing",
        "skipped_out_of_interval",
        "usable_frames",
        "skipped_invalid",
        "failures",
    )
    totals = {k: 0 for k in keys}
    for rec in records:
        cov = rec.get("cache_coverage") or {}
        videos = cov.get("videos") if isinstance(cov, dict) else None
        rows = videos if isinstance(videos, list) else [cov] if cov else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            for k in keys:
                totals[k] += int(row.get(k, 0) or 0)
        if isinstance(cov, dict):
            for k in keys:
                if k in cov and not videos:
                    totals[k] = int(cov.get(k, 0) or 0)
    return totals


def write_summary(
    cv_root: Union[str, Path],
    fold_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Write ``summary/fold_metrics.csv``, ``fold_metrics.json``, and
    ``aggregate_metrics.json``. Training-fit scores are never mixed into
    held-out annealing metrics.
    """
    layout = CvLayout(cv_root)
    layout.summary_dir.mkdir(parents=True, exist_ok=True)

    top1_hits = 0
    top3_hits = 0
    segs = 0
    per_fold_top1: List[Optional[float]] = []
    per_fold_top3: List[Optional[float]] = []
    per_fold_seg: List[Optional[float]] = []
    per_fold_frame: List[Optional[float]] = []
    per_fold_shelf_segment: List[Optional[float]] = []
    seg_correct = 0.0
    seg_total = 0.0
    frame_hits = 0
    frame_count = 0
    shelf_segment_hits = 0
    shelf_segment_count = 0
    shelf_totals: Dict[str, Dict[str, Dict[str, int]]] = {}

    for rec in fold_records:
        test = rec.get("annealing_test") or {}
        h1 = int(test.get("segment_top1_hits") or 0)
        h3 = int(test.get("segment_top3_hits") or 0)
        n = int(test.get("carry_segments_used") or 0)
        top1_hits += h1
        top3_hits += h3
        segs += n
        per_fold_top1.append(test.get("segment_top1_accuracy"))
        per_fold_top3.append(test.get("segment_top3_hit_rate"))
        frame_hits += int(test.get("frame_hits") or 0)
        frame_count += int(test.get("frame_count") or 0)
        shelf_segment_hits += int(
            test.get("shelf_constrained_segment_top1_hits") or 0
        )
        shelf_segment_count += n
        per_fold_frame.append(
            _test_metric_acc(test, "frame_hits", "frame_count")
        )
        per_fold_shelf_segment.append(
            _test_metric_acc(
                test,
                "shelf_constrained_segment_top1_hits",
                "carry_segments_used",
            )
        )
        for shelf, metrics in (test.get("by_shelf") or {}).items():
            shelf_total = shelf_totals.setdefault(shelf, {})
            for metric_name, block in metrics.items():
                target = shelf_total.setdefault(
                    metric_name, {"hits": 0, "count": 0}
                )
                target["hits"] += int((block or {}).get("hits") or 0)
                target["count"] += int((block or {}).get("count") or 0)
        acc = _seg_frame_acc(rec)
        per_fold_seg.append(acc)
        seg = rec.get("segmentation") or {}
        if acc is not None and seg.get("n_frames"):
            n_frames = float(seg["n_frames"])
            seg_correct += acc * n_frames
            seg_total += n_frames
        elif acc is not None:
            # Fall back to unweighted mean later via macro only.
            pass

    aggregate_by_shelf: Dict[str, Dict[str, Any]] = {}
    for shelf, metrics in shelf_totals.items():
        shelf_output: Dict[str, Any] = {}
        for metric_name, block in metrics.items():
            hits = int(block["hits"])
            count = int(block["count"])
            shelf_output[f"{metric_name}_hits"] = hits
            shelf_output[f"{metric_name}_count"] = count
            shelf_output[f"{metric_name}_accuracy"] = _ratio(hits, count)
        aggregate_by_shelf[shelf] = shelf_output

    aggregate = {
        "n_folds": len(fold_records),
        "segmentation": {
            "micro_frame_accuracy": _ratio(seg_correct, seg_total),
            "macro_frame_accuracy": _mean(per_fold_seg),
        },
        "annealing_test": {
            "frame_hits": frame_hits,
            "frame_count": frame_count,
            "frame_accuracy": _ratio(frame_hits, frame_count),
            "micro_frame_accuracy": _ratio(frame_hits, frame_count),
            "macro_frame_accuracy": _mean(per_fold_frame),
            "micro_segment_top1_accuracy": _ratio(top1_hits, segs),
            "micro_segment_top3_hit_rate": _ratio(top3_hits, segs),
            "macro_segment_top1_accuracy": _mean(per_fold_top1),
            "macro_segment_top3_hit_rate": _mean(per_fold_top3),
            "shelf_constrained_segment_top1_hits": shelf_segment_hits,
            "shelf_constrained_segment_top1_accuracy": _ratio(
                shelf_segment_hits, shelf_segment_count
            ),
            "micro_shelf_constrained_segment_top1_accuracy": _ratio(
                shelf_segment_hits, shelf_segment_count
            ),
            "macro_shelf_constrained_segment_top1_accuracy": _mean(
                per_fold_shelf_segment
            ),
            "by_shelf": aggregate_by_shelf,
            "segment_top1_hits": top1_hits,
            "segment_top3_hits": top3_hits,
            "carry_segments_used": segs,
        },
        "cache_coverage": _coverage_totals(fold_records),
        "skips_failures": {
            "skipped_missing": _coverage_totals(fold_records).get("skipped_missing", 0),
            "failures": _coverage_totals(fold_records).get("failures", 0),
        },
    }

    layout.fold_metrics_json.write_text(
        json.dumps(list(fold_records), indent=2) + "\n", encoding="utf-8"
    )
    layout.aggregate_metrics.write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )

    fieldnames = [
        "fold",
        "n_train",
        "n_test",
        "n_segmentation_train",
        "n_segmentation_test",
        "frame_accuracy",
        "segmentation_frame_accuracy",
        "segment_top1_accuracy",
        "segment_top3_hit_rate",
        "shelf_constrained_segment_top1_accuracy",
        "segment_top1_hits",
        "segment_top3_hits",
        "carry_segments_used",
    ]
    for shelf in ("c", "d", "e", "f", "g"):
        fieldnames.extend(
            [
                f"{shelf}_frame_accuracy",
                f"{shelf}_segment_top1_accuracy",
                f"{shelf}_shelf_constrained_segment_top1_accuracy",
            ]
        )
    with layout.fold_metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in fold_records:
            test = rec.get("annealing_test") or {}
            row = {
                "fold": rec.get("fold"),
                "n_train": rec.get("n_train"),
                "n_test": rec.get("n_test"),
                "n_segmentation_train": rec.get("n_segmentation_train"),
                "n_segmentation_test": rec.get("n_segmentation_test"),
                "frame_accuracy": test.get("frame_accuracy"),
                "segmentation_frame_accuracy": _seg_frame_acc(rec),
                "segment_top1_accuracy": test.get("segment_top1_accuracy"),
                "segment_top3_hit_rate": test.get("segment_top3_hit_rate"),
                "shelf_constrained_segment_top1_accuracy": test.get(
                    "shelf_constrained_segment_top1_accuracy"
                ),
                "segment_top1_hits": test.get("segment_top1_hits"),
                "segment_top3_hits": test.get("segment_top3_hits"),
                "carry_segments_used": test.get("carry_segments_used"),
            }
            for shelf in ("c", "d", "e", "f", "g"):
                shelf_metrics = (test.get("by_shelf") or {}).get(shelf) or {}
                row[f"{shelf}_frame_accuracy"] = (
                    (shelf_metrics.get("frame") or {}).get("accuracy")
                )
                row[f"{shelf}_segment_top1_accuracy"] = (
                    (shelf_metrics.get("segment_top1") or {}).get("accuracy")
                )
                row[f"{shelf}_shelf_constrained_segment_top1_accuracy"] = (
                    (shelf_metrics.get("shelf_constrained_segment_top1") or {}).get(
                        "accuracy"
                    )
                )
            writer.writerow(row)
    return aggregate
