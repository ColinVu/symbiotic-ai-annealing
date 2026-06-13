"""Type definitions and data structures for the pipeline."""

from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np

# Type aliases
EmbeddingArray = np.ndarray
LabelStr = str
PathStr = str

@dataclass
class DatasetInfo:
    """Container for dataset information."""
    embeddings: List[EmbeddingArray]
    labels: List[LabelStr]
    image_paths: List[PathStr]
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    embedding_dim: int

@dataclass
class DatasetSplits:
    """Container for train/val/test splits."""
    train: Dict[str, List]
    val: Dict[str, List]
    test: Dict[str, List]

@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    top1_accuracy: float
    top3_accuracy: float
    confusion_matrix: np.ndarray
    confusion_matrix_raw: np.ndarray
    predictions: np.ndarray
    true_labels: np.ndarray
    probabilities: np.ndarray
    label_names: List[str]

@dataclass
class TrainingHistory:
    """Container for training history."""
    train_loss: List[float]
    val_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]

__all__ = [
    'EmbeddingArray',
    'LabelStr',
    'PathStr',
    'DatasetInfo',
    'DatasetSplits',
    'EvaluationResults',
    'TrainingHistory'
]
