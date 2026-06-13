"""Configuration constants for the weakly supervised video-to-classification pipeline."""

import os

os.environ["GLOG_minloglevel"] = "2"

from ..lib.embedding import MODEL

DEFAULT_CONFIG = {
    "ilr_epochs": 1000,
    "skip_ilr": False,
    "use_cluster_voting": False,
    "initial_temp": 1.0,
    "temp_decay": "exponential",
    "decay_rate": 0.99,
    "min_temp": 0.05,
    "random_seed": 42,
    "variance_eps": 1e-6,
    "bad_swap_cool_divisor": 50.0,
    "detect_empty": False,
    "min_frames_per_cluster": 3,
    "ilr_allow_cross_round_swaps": False,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    # Iterated model pipeline (optional; replaces standard ILR when enabled)
    "use_iterated_model": False,
    "hand_embeddings_dir": "./hmm-testing/hand_embeddings",
    "apply_hand_pca": True,  # Apply PCA preprocessing before annealing
    "n_components": 50,  # Number of PCA components to remove
    "sa_iters": 100,
    "adapter_epochs": 10,
    "adapter_lr": 1e-3,
    "adapter_batch_size": 32,
    "refinement_loops": 3,
    "triplet_margin": 0.1,
    "proxy_energy_margin": 0.1,
    "ground_truth_csv": "../models/classifier/ground_truth.csv",
}

__all__ = ['DEFAULT_CONFIG', 'MODEL']
