"""Training utilities for weakly supervised learning."""

from .cluster_voting import cluster_based_initialization
from .weak_supervision import WeakSupervisedTrainer, Segment, LabelKey

__all__ = [
    "WeakSupervisedTrainer",
    "Segment",
    "LabelKey",
    "cluster_based_initialization",
]
