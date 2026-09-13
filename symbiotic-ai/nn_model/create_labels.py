"""CV-facing label generation: fold checkpoints -> compact annealing CSVs.

Non-CV usage writes one output directory (same files as ``nn_model.bulk_predict``).
In ``--k-fold`` mode each fold's checkpoint labels the full annealing dataset,
split into ``fold_NN/nn_model/labels/train/`` and ``.../test/``. Segmentation
metrics are computed only on that fold's ``segmentation_test_ids``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from .bulk_predict import generate_label_csvs
from .data import (
    DEFAULT_FEATURES_3D,
    DEFAULT_GROUND_TRUTH,
    DEFAULT_LABELS,
    DEFAULT_PICKLIST_JSONS,
    DEFAULT_PICKLIST_LABELS,
    FeatureSource,
    load_ground_truth_counts,
    load_picklist_sample,
)
from .metrics import evaluate_predictions
from .sequences import normalize_picklist_id


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _lookup_dense(dense_preds: Dict[str, np.ndarray], pid: str) -> Optional[np.ndarray]:
    key = normalize_picklist_id(str(pid))
    if key in dense_preds:
        return dense_preds[key]
    for candidate in (pid, str(pid).lstrip("0") or "0", f"picklist_{key}"):
        if candidate in dense_preds:
            return dense_preds[candidate]
    return None


def _score_segmentation(
    picklist_ids: Sequence[str],
    dense_preds: Dict[str, np.ndarray],
    feature_source: FeatureSource,
    features_dir: Union[str, Path],
    labels_dir: Union[str, Path],
    ground_truth: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Frame/state metrics for generated labels vs ``csv_labels`` (test IDs only)."""
    features_dir = Path(features_dir)
    labels_dir = Path(labels_dir)
    gt_counts = (
        load_ground_truth_counts(Path(ground_truth))
        if ground_truth is not None and Path(ground_truth).is_file()
        else {}
    )
    y_true_list = []
    y_pred_list = []
    expected_counts = []
    scored_ids = []
    n_frames = 0
    for raw in picklist_ids:
        pid = normalize_picklist_id(str(raw))
        pred = _lookup_dense(dense_preds, pid)
        if pred is None:
            continue
        sample = load_picklist_sample(
            pid,
            feature_source,
            features_dir,
            labels_dir,
            gt_counts or None,
        )
        if sample.labels is None:
            continue
        n = min(len(sample.labels), len(pred))
        if n <= 0:
            continue
        y_true_list.append(np.asarray(sample.labels[:n], dtype=np.int64))
        y_pred_list.append(np.asarray(pred[:n], dtype=np.int64))
        expected_counts.append(sample.object_count)
        scored_ids.append(pid)
        n_frames += int(n)

    if not y_true_list:
        return {
            "frame_accuracy": None,
            "macro_f1": None,
            "n_frames": 0,
            "n_picklists": 0,
            "picklist_ids": [],
        }
    metrics = evaluate_predictions(
        y_true_list,
        y_pred_list,
        expected_carry_counts=expected_counts,
    )
    metrics["n_frames"] = n_frames
    metrics["n_picklists"] = len(scored_ids)
    metrics["picklist_ids"] = scored_ids
    return _jsonable(metrics)


def _run_kfold_labels(args) -> None:
    from cross_validation.layout import CvLayout
    from cross_validation.manifests import (
        iter_fold_manifests,
        require_cv_pair,
        require_existing_dataset,
    )
    from cross_validation.status import mark_stage, stage_completed
    from cross_validation.stems import id_from_stem, normalize_stem

    require_cv_pair(args.k_fold, args.cv_results_dir)
    dataset = require_existing_dataset(args.cv_results_dir, int(args.k_fold))
    layout = CvLayout(args.cv_results_dir)
    paths = dataset.get("paths") or {}
    json_dir = args.picklist_jsons or paths.get("json_dir")
    features_dir = paths.get("features_dir") or args.features_dir or DEFAULT_FEATURES_3D
    labels_dir = paths.get("segmentation_labels_dir") or args.labels_dir or DEFAULT_LABELS
    features_dir = Path(features_dir)

    for fold in iter_fold_manifests(args.cv_results_dir):
        fold_n = int(fold["fold"])
        fl = layout.fold(fold_n)
        if stage_completed(args.cv_results_dir, fold_n, "nn_labels") and not args.overwrite:
            print(f"Fold {fold_n}: nn_labels already complete, skipping")
            continue
        if not fl.nn_checkpoint.is_file():
            raise SystemExit(f"Fold {fold_n} is missing {fl.nn_checkpoint}")
        train_stems = [normalize_stem(s) for s in fold["annealing_train_ids"]]
        test_stems = [normalize_stem(s) for s in fold["annealing_test_ids"]]
        all_ids = [id_from_stem(s) for s in train_stems + test_stems]
        test_ids = {id_from_stem(s) for s in test_stems}
        staging = fl.nn_checkpoint_dir / "labels" / "_all"
        if staging.exists():
            shutil.rmtree(staging)
        dense = generate_label_csvs(
            checkpoint=fl.nn_checkpoint,
            features_dir=features_dir,
            output_dir=staging,
            picklist_ids=all_ids,
            picklist_jsons=json_dir,
            labels_dir=labels_dir,
            ground_truth=args.ground_truth,
            sequence_source=args.sequence_source,
            trailing_m=args.trailing_m,
            whole_cycles=args.whole_cycles,
            skip_malformed=args.skip_malformed,
            fallback_decode=args.fallback_decode,
            output_format="csv",
            device=args.device,
            prior_scale=args.prior_scale,
            min_duration=args.min_duration,
            transition_scale=args.transition_scale,
            verbose=True,
        )
        fl.nn_train_labels.mkdir(parents=True, exist_ok=True)
        fl.nn_test_labels.mkdir(parents=True, exist_ok=True)
        for pid in list(dense):
            src = staging / f"picklist_{pid}.csv"
            if not src.is_file():
                continue
            dest_dir = fl.nn_test_labels if pid in test_ids else fl.nn_train_labels
            dest = dest_dir / f"picklist_{pid}.csv"
            dest.write_bytes(src.read_bytes())
        shutil.rmtree(staging, ignore_errors=True)

        seg_test_ids = [id_from_stem(s) for s in fold["segmentation_test_ids"]]
        metrics = _score_segmentation(
            picklist_ids=seg_test_ids,
            dense_preds=dense,
            feature_source="processed_3d",
            features_dir=features_dir,
            labels_dir=labels_dir,
            ground_truth=args.ground_truth,
        )
        fl.nn_metrics.write_text(
            json.dumps(
                {
                    "fold": fold_n,
                    "checkpoint": str(fl.nn_checkpoint),
                    "segmentation": metrics,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        mark_stage(args.cv_results_dir, fold_n, "nn_labels")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate compact state-boundary CSVs from an NN checkpoint.",
    )
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_3D)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_PICKLIST_LABELS)
    ap.add_argument("--labels-dir", type=Path, default=None)
    ap.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    ap.add_argument("--picklist-jsons", type=Path, default=None)
    ap.add_argument(
        "--sequence-source",
        choices=["auto", "labels", "json"],
        default="json",
    )
    ap.add_argument("--trailing-m", action="store_true")
    ap.add_argument("--whole-cycles", action="store_true")
    ap.add_argument("--skip-malformed", action="store_true")
    ap.add_argument(
        "--fallback-decode",
        choices=["constrained", "skip"],
        default="constrained",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prior-scale", type=float, default=1.0)
    ap.add_argument("--min-duration", type=int, default=1)
    ap.add_argument("--transition-scale", type=float, default=0.0)
    ap.add_argument("--k-fold", type=int, default=None)
    ap.add_argument("--cv-results-dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.k_fold is not None or args.cv_results_dir is not None:
        _run_kfold_labels(args)
        return

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --k-fold is set")
    from .bulk_predict import discover_processed_3d_picklists

    features_dir = args.features_dir.resolve()
    picklist_ids = discover_processed_3d_picklists(features_dir)
    json_dir = args.picklist_jsons if args.picklist_jsons is not None else DEFAULT_PICKLIST_JSONS
    generate_label_csvs(
        checkpoint=args.checkpoint,
        features_dir=features_dir,
        output_dir=args.output_dir,
        picklist_ids=picklist_ids,
        picklist_jsons=json_dir,
        labels_dir=args.labels_dir or DEFAULT_LABELS,
        ground_truth=args.ground_truth,
        sequence_source=args.sequence_source,
        trailing_m=args.trailing_m,
        whole_cycles=args.whole_cycles,
        skip_malformed=args.skip_malformed,
        fallback_decode=args.fallback_decode,
        device=args.device,
        prior_scale=args.prior_scale,
        min_duration=args.min_duration,
        transition_scale=args.transition_scale,
    )


if __name__ == "__main__":
    main()
