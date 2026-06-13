from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


class ClipEmbedder:
    """Thin wrapper around a CLIP-like vision model."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.inference_mode()
    def embed(self, images: Sequence[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """Returns embeddings with shape (len(images), embedding_dim)."""
        if not images:
            return np.empty((0, 0))

        embeddings: List[np.ndarray] = []

        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self.processor(images=list(batch), return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            embeddings.append(image_features.cpu().numpy())

        return np.vstack(embeddings)

    def to(self, device: str) -> "ClipEmbedder":
        self.device = device
        self.model.to(device)
        return self


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


