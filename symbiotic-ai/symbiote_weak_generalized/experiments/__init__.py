"""Hyperparameter sweep utilities for weakly supervised video training."""

from .assignment_score import (
    score_final_assignments_hit_rate,
    score_training_assignments_vs_ground_truth,
    write_final_assignments_csv,
)
from .evaluator import evaluate_model
from .results_manager import aggregate_results, find_best_config, save_results_csv, save_summary_report
from .sweep_config import (
    PARAM_GRID,
    SWEEP_FIXED_THRESHOLD,
    SWEEP_PARAM_KEYS,
    build_param_grid_from_cli,
    describe_param_grid,
    generate_configs,
    grid_size,
)
from .sweep_runner import run_sweep

__all__ = [
    "PARAM_GRID",
    "SWEEP_PARAM_KEYS",
    "SWEEP_FIXED_THRESHOLD",
    "build_param_grid_from_cli",
    "score_final_assignments_hit_rate",
    "score_training_assignments_vs_ground_truth",
    "write_final_assignments_csv",
    "aggregate_results",
    "describe_param_grid",
    "evaluate_model",
    "find_best_config",
    "generate_configs",
    "grid_size",
    "run_sweep",
    "save_results_csv",
    "save_summary_report",
]
