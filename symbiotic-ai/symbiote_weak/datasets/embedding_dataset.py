"""PyTorch Dataset for embeddings."""

from typing import List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """PyTorch Dataset for embeddings."""
    
    def __init__(self, embeddings: List[np.ndarray], labels: List[str], label_to_idx: Dict[str, int]):
        self.embeddings = [torch.tensor(e, dtype=torch.float32) for e in embeddings]
        self.labels = [label_to_idx[l] for l in labels]
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


__all__ = ['EmbeddingDataset']
