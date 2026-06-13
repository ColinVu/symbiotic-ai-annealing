"""Configuration constants for the weakly supervised video-to-classification pipeline."""

import os

os.environ["GLOG_minloglevel"] = "2"

from ..lib.embedding import MODEL

DEFAULT_CONFIG = {
    "ilr_epochs": 500,
    "skip_ilr": False,
    "use_cluster_voting": False,
    "initial_temp": 1.0,
    "temp_decay": "exponential",
    "decay_rate": 0.99,
    "random_seed": 42,
    "variance_eps": 1e-6,
    "bad_swap_cool_divisor": 50.0,
    "detect_empty": False,
    "min_frames_per_cluster": 3,
    "ilr_allow_cross_round_swaps": False,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
}

__all__ = ['DEFAULT_CONFIG', 'MODEL']
