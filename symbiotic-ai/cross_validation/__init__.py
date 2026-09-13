"""K-fold cross-validation orchestration for the combined pipeline."""

from .layout import CvLayout, FoldLayout
from .manifests import (
    create_or_load_dataset,
    iter_fold_manifests,
    require_cv_pair,
    require_existing_dataset,
)
from .status import mark_stage, stage_completed
from .stems import id_from_stem, normalize_stem
from .summary import write_summary
from .videos import resolve_video_path

__all__ = [
    "CvLayout",
    "FoldLayout",
    "create_or_load_dataset",
    "iter_fold_manifests",
    "require_cv_pair",
    "require_existing_dataset",
    "mark_stage",
    "stage_completed",
    "id_from_stem",
    "normalize_stem",
    "write_summary",
    "resolve_video_path",
]
