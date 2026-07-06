"""Score ILR final segment labels against ordered ground truth (no video inference)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..training.weak_supervision import WeakSupervisedTrainer

from .evaluator import _load_ground_truth_labels


def write_final_assignments_csv(
    output_path: str,
    trainer: WeakSupervisedTrainer,
    *,
    ground_truth_csv: Optional[str] = None,
    video_stems_order: Optional[List[str]] = None,
) -> int:
    """
    Write one row per segment: ILR final label after ``fit()`` (``last_refined_labels``).

    Columns: ``video_stem``, ``segment_id``, ``assigned_label``, and when
    *ground_truth_csv* is provided and has a column for the video:
    ``ground_truth``, ``hit`` (assigned == ground truth for that segment index).

    Returns:
        Number of rows written (excluding header).
    """
    refined = trainer.last_refined_labels
    if not refined:
        raise RuntimeError(
            "Trainer has no last_refined_labels; call after fit() completes."
        )

    gt_path = str(Path(ground_truth_csv).resolve()) if ground_truth_csv else None
    gt_by_stem: Dict[str, List[str]] = {}
    if gt_path and Path(gt_path).is_file():
        stems = video_stems_order or sorted({vid for vid, _ in refined.keys()})
        for stem in stems:
            try:
                gt_by_stem[stem] = _load_ground_truth_labels(gt_path, stem)
            except ValueError:
                pass

    rows: List[Tuple[str, int, str, str, str, str]] = []
    for (vid, seg_id), lab in sorted(refined.items(), key=lambda x: (x[0][0], x[0][1])):
        assigned = str(lab)
        gt_lab = ""
        hit = ""
        expected = gt_by_stem.get(vid)
        if expected is not None and 0 <= int(seg_id) < len(expected):
            gt_lab = expected[int(seg_id)]
            hit = "1" if assigned == gt_lab else "0"
        rows.append((str(vid), int(seg_id), assigned, gt_lab, hit))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_stem",
        "segment_id",
        "assigned_label",
        "ground_truth",
        "hit",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for vid, seg_id, assigned, gt_lab, hit in rows:
            w.writerow(
                {
                    "video_stem": vid,
                    "segment_id": seg_id,
                    "assigned_label": assigned,
                    "ground_truth": gt_lab,
                    "hit": hit,
                }
            )
    return len(rows)


def score_final_assignments_hit_rate(run_dir: str) -> Dict[str, Any]:
    """
    Read ``{run_dir}/final_assignments.csv`` and compute micro hit rate:

    ``assignment_hit_rate = hits / compared`` where *compared* counts only rows
    with ``hit`` in ``{"0","1"}`` (i.e. ground truth was available for that segment).
    """
    path = Path(run_dir) / "final_assignments.csv"
    if not path.is_file():
        return {
            "run_dir": str(path.resolve()),
            "scoring_mode": "final_assignments_csv_hit_rate",
            "metrics": None,
            "error": "final_assignments.csv not found (train must save with ground_truth_csv in config)",
        }

    hits = 0
    compared = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "hit" not in reader.fieldnames:
            return {
                "run_dir": str(path.resolve()),
                "scoring_mode": "final_assignments_csv_hit_rate",
                "metrics": None,
                "error": "final_assignments.csv missing 'hit' column",
            }
        for row in reader:
            h = (row.get("hit") or "").strip()
            if h not in ("0", "1"):
                continue
            compared += 1
            if h == "1":
                hits += 1

    rate = (hits / compared) if compared else 0.0
    return {
        "run_dir": str(path.resolve()),
        "scoring_mode": "final_assignments_csv_hit_rate",
        "metrics": {
            "assignment_hit_rate": float(rate),
            "assignment_hits": int(hits),
            "assignment_compared": int(compared),
            # mirror for results tables that expect segment_top1 naming
            "segment_top1_hits": int(hits),
            "segment_top1_accuracy": float(rate),
            "carry_segments_used": int(compared),
            "segments_with_predictions": int(compared),
        },
    }


def score_training_assignments_vs_ground_truth(
    trainer: WeakSupervisedTrainer,
    ground_truth_csv: str,
    video_stems: List[str],
) -> Dict[str, Any]:
    """
    Compare ``trainer.last_refined_labels`` (after ``fit()``) to ``ground_truth.csv``.

    For each *video_stem*, expected pick order is the non-empty cells in that column.
    Predicted label for pick *i* is ``last_refined_labels[(stem, i)]`` when present.

    Micro-averaged top-1 accuracy: total correct / total compared (one row per
    expected pick index). Missing predicted label for an index counts as wrong.
    """
    refined = trainer.last_refined_labels
    if not refined:
        raise RuntimeError("Trainer has no last_refined_labels; fit() did not complete.")

    gt_path = str(Path(ground_truth_csv).resolve())

    per_video: List[Dict[str, Any]] = []
    total_correct = 0
    total_compared = 0

    for stem in video_stems:
        try:
            expected = _load_ground_truth_labels(gt_path, stem)
        except ValueError:
            per_video.append(
                {
                    "video_stem": stem,
                    "skipped": True,
                    "reason": "no_ground_truth_column_or_empty",
                }
            )
            continue

        pred_by_seg: Dict[int, str] = {}
        for (vid, seg_id), lab in refined.items():
            if vid == stem:
                pred_by_seg[int(seg_id)] = str(lab)

        hits = 0
        n = len(expected)
        mismatches: List[Dict[str, Any]] = []
        segment_predictions: List[Dict[str, Any]] = []
        for i, exp in enumerate(expected):
            total_compared += 1
            pred = pred_by_seg.get(i, "")
            pred_label = pred or ""
            ok = pred_label == exp
            segment_predictions.append(
                {
                    "segment_idx": i,
                    "expected": exp,
                    "predicted": pred_label,
                    "hit": bool(ok),
                }
            )
            if ok:
                hits += 1
                total_correct += 1
            else:
                mismatches.append({"segment_idx": i, "expected": exp, "predicted": pred_label})

        acc = hits / n if n else 0.0
        per_video.append(
            {
                "video_stem": stem,
                "skipped": False,
                "segments_compared": n,
                "segment_top1_hits": hits,
                "segment_top1_accuracy": acc,
                "predicted_segment_count": len(pred_by_seg),
                "mismatch_count": len(mismatches),
                "segment_predictions": segment_predictions,
                "sample_mismatches": mismatches[:20],
            }
        )

    if total_compared == 0:
        raise RuntimeError(
            "No ground-truth rows compared (check ground_truth.csv columns vs --videos stems)."
        )

    micro_top1 = total_correct / total_compared

    return {
        "ground_truth_csv": gt_path,
        "scoring_mode": "training_assignment_vs_ground_truth",
        "video_stems_requested": list(video_stems),
        "metrics": {
            "carry_segments_used": total_compared,
            "segments_with_predictions": total_compared,
            "segment_top1_hits": total_correct,
            "segment_top1_accuracy": micro_top1,
            "segment_top3_hits": None,
            "segment_top3_hit_rate": None,
            "mean_segment_top1_confidence": None,
        },
        "per_video": per_video,
    }


__all__ = [
    "score_training_assignments_vs_ground_truth",
    "score_final_assignments_hit_rate",
    "write_final_assignments_csv",
]
