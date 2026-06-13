"""Configuration constants for the video-to-classification pipeline."""

import os
import sys

# Suppress MediaPipe warnings
os.environ["GLOG_minloglevel"] = "2"

# Add lib folder to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))

from embedding import MODEL

# Default training configuration
DEFAULT_CONFIG = {
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "batch_size": 16,
    "learning_rate": 0.001,
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "hidden_dim": 128,  # Hidden layer size for MLP classifier
    "dropout": 0.3,
    "random_seed": 42,
}

__all__ = ['DEFAULT_CONFIG', 'MODEL']
