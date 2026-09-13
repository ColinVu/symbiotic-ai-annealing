"""K-fold annealing train / train-from-cache orchestration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .layout import CvLayout
from .manifests import iter_fold_manifests, require_cv_pair, require_existing_dataset
from .status import mark_stage, stage_completed
from .stems import normalize_stem


def _copy_train_assignment_metrics(model_dir: Path, train_dir: Path) -> None:
    train_dir.mkdir(parents=True, exist_ok=True)
    src = model_dir / "final_assignments.csv"
    if src.is_file():
        shutil.copy2(src, train_dir / "final_assignments.csv")
        from symbiote_weak_generalized.experiments.assignment_score import (
            score_final_assignments_hit_rate,
        )

        scored = score_final_assignments_hit_rate(str(model_dir))
        (train_dir / "metrics.json").write_text(
            json.dumps(scored, indent=2) + "\n", encoding="utf-8"
        )


def run_kfold_annealing(
    *,
    cv_root: Union[str, Path],
    k_fold: int,
    videos_dir: str,
    picklist_json_dir: str,
    cache_dir: str,
    config: Dict[str, Any],
    frame_skip: int = 4,
    verbose: bool = True,
    compact_frame_indexing: str = "opencv0",
    from_cache: bool = True,
    threshold: float = 50.0,
    htk_model_dir: Optional[str] = None,
    aruco_config_path: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """Train one annealing model per fold on ``annealing_train_ids`` only."""
    require_cv_pair(k_fold, cv_root)
    require_existing_dataset(
        cv_root,
        int(k_fold),
        expected_paths={
            "videos_dir": videos_dir,
            "json_dir": picklist_json_dir,
        },
    )
    layout = CvLayout(cv_root)

    from symbiote_weak_generalized.pipelines.video_training import (
        run_multi_video_training,
        run_multi_video_training_from_cache,
    )

    if not cache_dir:
        raise SystemExit("K-fold annealing requires --cache-dir (shared classifier .cache)")

    for fold in iter_fold_manifests(cv_root):
        fold_n = int(fold["fold"])
        fl = layout.fold(fold_n)
        if stage_completed(cv_root, fold_n, "annealing_train") and not overwrite:
            print(f"Fold {fold_n}: annealing_train already complete, skipping")
            continue
        train_stems = [normalize_stem(s) for s in fold["annealing_train_ids"]]
        labels_dir = str(fl.nn_train_labels)
        if not Path(labels_dir).is_dir():
            raise SystemExit(
                f"Fold {fold_n} is missing generated train labels at {labels_dir}"
            )
        fl.annealing_model.mkdir(parents=True, exist_ok=True)
        if from_cache:
            run_multi_video_training_from_cache(
                videos_dir=videos_dir,
                picklist_json_dir=picklist_json_dir,
                manual_labels_dir=labels_dir,
                base_output_dir=str(fl.annealing_model),
                config=config,
                cache_dir=cache_dir,
                frame_skip=frame_skip,
                verbose=verbose,
                compact_frame_indexing=compact_frame_indexing,
                include_stems=train_stems,
            )
        else:
            run_multi_video_training(
                videos_dir=videos_dir,
                picklist_json_dir=picklist_json_dir,
                manual_labels_dir=labels_dir,
                base_output_dir=str(fl.annealing_model),
                config=config,
                threshold=threshold,
                frame_skip=frame_skip,
                verbose=verbose,
                htk_model_dir=htk_model_dir,
                aruco_config_path=aruco_config_path,
                compact_frame_indexing=compact_frame_indexing,
                include_stems=train_stems,
                cache_dir=cache_dir,
            )
        _copy_train_assignment_metrics(fl.annealing_model, fl.annealing_train)
        mark_stage(cv_root, fold_n, "annealing_train")
