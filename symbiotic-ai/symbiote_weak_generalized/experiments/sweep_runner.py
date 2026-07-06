"""Orchestrate multiple train-from-cache runs for hyperparameter sweeps."""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import AbstractSet, Any, Dict, List, Optional

from ..pipelines.video_training import run_multi_video_training_from_cache

from .sweep_config import SWEEP_FIXED_THRESHOLD


def _serialize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-serializable snapshot (paths not included)."""
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, dict)):
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = str(v)
        else:
            out[k] = str(v)
    out["sweep_note_threshold_fixed"] = SWEEP_FIXED_THRESHOLD
    return out


def run_sweep(
    videos_dir: str,
    picklist_json_dir: str,
    manual_labels_dir: str,
    base_output_dir: str,
    param_combinations: List[Dict[str, Any]],
    cache_dir: Optional[str] = None,
    frame_skip: int = 4,
    verbose: bool = False,
    compact_frame_indexing: str = "opencv0",
    exclude_stems: Optional[AbstractSet[str]] = None,
) -> List[Dict[str, Any]]:
    """
    For each merged config in *param_combinations*, train into
    ``{base_output_dir}/run_{i}_{timestamp}/`` and record success/failure.

    Blur threshold is not used here (train-from-cache). Sweep scoring uses each
    run's ``final_assignments.csv`` hit rate (see ``score_final_assignments_hit_rate``).

    Returns:
        List of result dicts with keys including ``run_dir``, ``config``,
        ``success``, ``error`` (optional), ``trainer`` (only on success; caller
        may drop reference).
    """
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: List[Dict[str, Any]] = []

    for idx, config in enumerate(param_combinations):
        run_dir = str(Path(base_output_dir) / f"run_{idx:04d}_{ts}")
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        params_path = Path(run_dir) / "params.json"
        with params_path.open("w", encoding="utf-8") as f:
            json.dump(_serialize_config(config), f, indent=2)

        rec: Dict[str, Any] = {
            "run_index": idx,
            "run_dir": run_dir,
            "config": config,
            "success": False,
        }
        try:
            trainer = run_multi_video_training_from_cache(
                videos_dir=videos_dir,
                picklist_json_dir=picklist_json_dir,
                manual_labels_dir=manual_labels_dir,
                base_output_dir=run_dir,
                config=config,
                cache_dir=cache_dir,
                frame_skip=frame_skip,
                verbose=verbose,
                compact_frame_indexing=compact_frame_indexing,
                exclude_stems=exclude_stems,
            )
            rec["success"] = True
            rec["trainer"] = trainer
        except SystemExit as e:
            msg = str(e)
            rec["error"] = msg
            _write_error_log(run_dir, msg + "\n" + traceback.format_exc())
        except Exception:
            rec["error"] = traceback.format_exc()
            _write_error_log(run_dir, rec["error"])

        results.append(rec)

    return results


def _write_error_log(run_dir: str, text: str) -> None:
    err_path = Path(run_dir) / "error.log"
    err_path.write_text(text, encoding="utf-8")
