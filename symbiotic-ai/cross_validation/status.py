"""Per-fold stage status so interrupted CV commands can resume completed folds."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union

from .layout import CvLayout

STAGES = ("nn_train", "nn_labels", "annealing_train", "annealing_test")


def _status_path(cv_root: Union[str, Path], fold: int) -> Path:
    return CvLayout(cv_root).fold(int(fold)).status_file


def load_status(cv_root: Union[str, Path], fold: int) -> Dict[str, Any]:
    path = _status_path(cv_root, fold)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def stage_completed(cv_root: Union[str, Path], fold: int, stage: str) -> bool:
    rec = load_status(cv_root, fold).get(stage) or {}
    return bool(rec.get("completed"))


def mark_stage(
    cv_root: Union[str, Path],
    fold: int,
    stage: str,
    *,
    extra: Dict[str, Any] | None = None,
) -> None:
    path = _status_path(cv_root, fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    status = load_status(cv_root, fold)
    rec: Dict[str, Any] = {
        "completed": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        rec.update(extra)
    status[stage] = rec
    path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
