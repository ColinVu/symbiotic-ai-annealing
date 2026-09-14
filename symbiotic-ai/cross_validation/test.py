"""Held-out K-fold annealing evaluation and cross-fold summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .layout import CvLayout
from .manifests import iter_fold_manifests, require_cv_pair, require_existing_dataset
from .status import mark_stage, stage_completed
from .stems import normalize_stem
from .summary import write_summary
from .videos import resolve_video_path


def _gt_columns(ground_truth_csv: str) -> set:
    import csv

    with open(ground_truth_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {
            h.strip()
            for h in (reader.fieldnames or [])
            if h and str(h).strip()
        }


def _first_sku(fold: Dict[str, Any], stem: str, json_dir: Optional[str]) -> Optional[str]:
    skus = fold.get("first_skus") or {}
    if stem in skus:
        return str(skus[stem])
    if json_dir:
        path = Path(json_dir) / f"{stem}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            picklists = payload.get("picklists") or []
            if picklists:
                first = picklists[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, list) and first:
                    return str(first[0])
    return None


def _metric_totals(
    per_video: List[Dict[str, Any]],
    metric_key: str,
) -> Dict[str, Any]:
    """Sum a nested evaluator metric across videos."""
    hits = 0
    count = 0
    for evaluation in per_video:
        block = (evaluation.get("metrics") or {}).get(metric_key) or {}
        hits += int(block.get("hits") or 0)
        count += int(block.get("count") or 0)
    return {
        "hits": hits,
        "count": count,
        "accuracy": (hits / count) if count else None,
    }


def _shelf_metric_totals(per_video: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum the three requested metric families separately for each shelf."""
    totals: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for evaluation in per_video:
        by_shelf = (evaluation.get("metrics") or {}).get("by_shelf") or {}
        for shelf, shelf_metrics in by_shelf.items():
            shelf_total = totals.setdefault(shelf, {})
            for metric_key, block in shelf_metrics.items():
                current = shelf_total.setdefault(metric_key, {"hits": 0, "count": 0})
                current["hits"] += int((block or {}).get("hits") or 0)
                current["count"] += int((block or {}).get("count") or 0)
    for shelf_metrics in totals.values():
        for block in shelf_metrics.values():
            block["accuracy"] = (
                block["hits"] / block["count"] if block["count"] else None
            )
    return totals


def run_kfold_test(
    *,
    cv_root: Union[str, Path],
    k_fold: int,
    videos_dir: str,
    ground_truth_csv: str,
    cache_dir: str,
    json_dir: Optional[str] = None,
    frame_skip: int = 4,
    compact_frame_indexing: str = "opencv0",
    threshold: float = 50.0,
    verbose: bool = False,
    embed_missing: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Evaluate each fold's annealing model on held-out ``annealing_test_ids``."""
    require_cv_pair(k_fold, cv_root)
    dataset = require_existing_dataset(cv_root, int(k_fold))
    layout = CvLayout(cv_root)
    if json_dir is None:
        json_dir = (dataset.get("paths") or {}).get("json_dir")
    if videos_dir is None:
        videos_dir = (dataset.get("paths") or {}).get("videos_dir")
    if not videos_dir:
        raise SystemExit("K-fold test requires --videos-dir")
    require_existing_dataset(
        cv_root,
        int(k_fold),
        expected_paths={"videos_dir": videos_dir, "json_dir": json_dir},
    )
    if not Path(ground_truth_csv).is_file():
        raise SystemExit(f"ground_truth.csv not found: {ground_truth_csv}")

    gt_cols = _gt_columns(ground_truth_csv)
    from symbiote_weak_generalized.experiments.evaluator import evaluate_model

    fold_records: List[Dict[str, Any]] = []
    for fold in iter_fold_manifests(cv_root):
        fold_n = int(fold["fold"])
        fl = layout.fold(fold_n)
        if stage_completed(cv_root, fold_n, "annealing_test") and not overwrite:
            metrics_path = fl.annealing_test / "metrics.json"
            test_metrics = {}
            if metrics_path.is_file():
                test_metrics = (json.loads(metrics_path.read_text()) or {}).get("metrics") or {}
            has_extended_metrics = (
                "frame_accuracy" in test_metrics
                and "by_shelf" in test_metrics
            )
            if not has_extended_metrics:
                print(
                    f"Fold {fold_n}: existing metrics lack extended accuracy fields, "
                    "recomputing"
                )
            else:
                print(f"Fold {fold_n}: annealing_test already complete, loading metrics")
                nn = json.loads(fl.nn_metrics.read_text()) if fl.nn_metrics.is_file() else {}
                cov = {}
                cov_path = fl.annealing_model / "cache_coverage.json"
                if cov_path.is_file():
                    cov = json.loads(cov_path.read_text(encoding="utf-8"))
                fold_records.append(
                    {
                        "fold": fold_n,
                        "n_train": len(fold["annealing_train_ids"]),
                        "n_test": len(fold["annealing_test_ids"]),
                        "n_segmentation_train": len(fold["segmentation_train_ids"]),
                        "n_segmentation_test": len(fold["segmentation_test_ids"]),
                        "segmentation": nn.get("segmentation"),
                        "annealing_test": test_metrics,
                        "cache_coverage": cov,
                    }
                )
                continue

        if not fl.annealing_model.is_dir():
            raise SystemExit(f"Fold {fold_n} is missing annealing model at {fl.annealing_model}")
        fl.annealing_test.mkdir(parents=True, exist_ok=True)
        per_video = []
        skips: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for stem in fold["annealing_test_ids"]:
            stem = normalize_stem(stem)
            if stem not in gt_cols:
                skips.append({"video_stem": stem, "reason": "no_ground_truth_column"})
                continue
            state_csv = fl.nn_test_labels / f"{stem}.csv"
            if not state_csv.is_file():
                skips.append({"video_stem": stem, "reason": "missing_test_labels", "path": str(state_csv)})
                continue
            sku = _first_sku(fold, stem, json_dir)
            if not sku:
                skips.append({"video_stem": stem, "reason": "missing_first_sku"})
                continue
            try:
                video = resolve_video_path(videos_dir, stem)
                ev = evaluate_model(
                    model_dir=str(fl.annealing_model),
                    video_path=str(video),
                    ground_truth_csv=ground_truth_csv,
                    state_label_csv_path=str(state_csv),
                    frame_skip=frame_skip,
                    compact_frame_indexing=compact_frame_indexing,
                    threshold=threshold,
                    verbose=verbose,
                    eval_cache_dir=str(Path(cache_dir) / stem),
                    cache_label=sku,
                    embed_missing=embed_missing,
                )
            except Exception as exc:
                failures.append({"video_stem": stem, "reason": "evaluate_failed", "detail": str(exc)})
                continue
            per_video.append(ev)
            (fl.annealing_test / f"{stem}_eval.json").write_text(
                json.dumps(ev, indent=2) + "\n", encoding="utf-8"
            )

        top1 = sum(int(e["metrics"]["segment_top1_hits"]) for e in per_video)
        top3 = sum(int(e["metrics"]["segment_top3_hits"]) for e in per_video)
        segs = sum(int(e["metrics"]["carry_segments_used"]) for e in per_video)
        frame = _metric_totals(per_video, "frame")
        shelf_segment_top1 = _metric_totals(
            per_video, "shelf_constrained_segment_top1"
        )
        test_metrics = {
            "frame_hits": frame["hits"],
            "frame_count": frame["count"],
            "frame_accuracy": frame["accuracy"],
            "segment_top1_hits": top1,
            "segment_top3_hits": top3,
            "carry_segments_used": segs,
            "segment_top1_accuracy": (top1 / segs) if segs else None,
            "segment_top3_hit_rate": (top3 / segs) if segs else None,
            "shelf_constrained_segment_top1_hits": shelf_segment_top1["hits"],
            "shelf_constrained_segment_top1_accuracy": shelf_segment_top1[
                "accuracy"
            ],
            "by_shelf": _shelf_metric_totals(per_video),
            "n_videos_scored": len(per_video),
            "n_skipped": len(skips),
            "n_failures": len(failures),
        }
        (fl.annealing_test / "metrics.json").write_text(
            json.dumps(
                {"metrics": test_metrics, "skipped": skips, "failures": failures},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        nn = json.loads(fl.nn_metrics.read_text()) if fl.nn_metrics.is_file() else {}
        cov = {}
        cov_path = fl.annealing_model / "cache_coverage.json"
        if cov_path.is_file():
            cov = json.loads(cov_path.read_text(encoding="utf-8"))
        fold_records.append(
            {
                "fold": fold_n,
                "n_train": len(fold["annealing_train_ids"]),
                "n_test": len(fold["annealing_test_ids"]),
                "n_segmentation_train": len(fold["segmentation_train_ids"]),
                "n_segmentation_test": len(fold["segmentation_test_ids"]),
                "segmentation": nn.get("segmentation"),
                "annealing_test": test_metrics,
                "cache_coverage": cov,
            }
        )
        mark_stage(cv_root, fold_n, "annealing_test")

    return write_summary(cv_root, fold_records)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Held-out K-fold annealing evaluation and summary.",
    )
    ap.add_argument("--k-fold", type=int, required=True)
    ap.add_argument("--cv-results-dir", type=Path, required=True)
    ap.add_argument("--videos-dir", type=Path, default=None)
    ap.add_argument("--ground-truth-csv", type=Path, required=True)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Shared input classifier .cache root (per-video subdirs).",
    )
    ap.add_argument("--picklist-jsons", type=Path, default=None)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument(
        "--compact-frame-indexing",
        choices=["opencv0", "pipeline1"],
        default="opencv0",
    )
    ap.add_argument("--threshold", type=float, default=50.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--cache-only",
        action="store_true",
        help="Do not CLIP-embed missing test frames; skip cache misses.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dataset = require_existing_dataset(args.cv_results_dir, int(args.k_fold))
    videos_dir = args.videos_dir
    if videos_dir is None and (dataset.get("paths") or {}).get("videos_dir"):
        videos_dir = Path(dataset["paths"]["videos_dir"])
    if videos_dir is None:
        raise SystemExit("--videos-dir is required (not stored in the dataset manifest)")

    aggregate = run_kfold_test(
        cv_root=args.cv_results_dir,
        k_fold=int(args.k_fold),
        videos_dir=str(videos_dir),
        ground_truth_csv=str(args.ground_truth_csv),
        cache_dir=str(args.cache_dir),
        json_dir=str(args.picklist_jsons) if args.picklist_jsons else None,
        frame_skip=args.frame_skip,
        compact_frame_indexing=args.compact_frame_indexing,
        threshold=args.threshold,
        verbose=args.verbose,
        embed_missing=not args.cache_only,
        overwrite=args.overwrite,
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
