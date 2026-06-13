"""Dataset splitting utilities."""

from typing import List, Dict
from collections import defaultdict
import random
import numpy as np


def stratified_split(
    embeddings: List[np.ndarray],
    labels: List[str],
    image_paths: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, Dict[str, List]]:
    """
    Split dataset into train/val/test with stratification by class.
    
    Ensures each class appears in all splits proportionally.
    
    Args:
        embeddings: List of embedding arrays
        labels: List of string labels
        image_paths: List of image paths
        train_ratio: Fraction for training (default 0.70)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing
        'embeddings', 'labels', 'image_paths' lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Group by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    # Initialize splits
    splits = {
        "train": {"embeddings": [], "labels": [], "image_paths": []},
        "val": {"embeddings": [], "labels": [], "image_paths": []},
        "test": {"embeddings": [], "labels": [], "image_paths": []},
    }
    
    # Stratified split for each class
    for label, indices in class_indices.items():
        # Shuffle indices for this class
        random.shuffle(indices)
        
        n = len(indices)
        # Ensure at least 1 sample in val and test when we have enough (avoids empty val/test)
        if n == 1:
            n_train, n_val, n_test = 1, 0, 0
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            n_test = max(1, int(n * test_ratio))
            # If we over-allocated, give remainder to train
            if n_train + n_val + n_test > n:
                n_train = n - n_val - n_test
                n_train = max(1, n_train)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Add to splits
        for idx in train_indices:
            splits["train"]["embeddings"].append(embeddings[idx])
            splits["train"]["labels"].append(labels[idx])
            splits["train"]["image_paths"].append(image_paths[idx])
        
        for idx in val_indices:
            splits["val"]["embeddings"].append(embeddings[idx])
            splits["val"]["labels"].append(labels[idx])
            splits["val"]["image_paths"].append(image_paths[idx])
        
        for idx in test_indices:
            splits["test"]["embeddings"].append(embeddings[idx])
            splits["test"]["labels"].append(labels[idx])
            splits["test"]["image_paths"].append(image_paths[idx])
    
    return splits


__all__ = ['stratified_split']
