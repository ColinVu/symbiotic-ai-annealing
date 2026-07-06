"""Centroid-based classifier for weakly supervised CLIP embeddings (cosine, no PCA)."""

from typing import Any, Dict, List, Tuple
import numpy as np


class CentroidModel:
    """
    Centroid-based classifier: nearest centroid by cosine distance on raw CLIP space.
    """

    def __init__(
        self,
        centroids: Dict[str, np.ndarray],
        label_to_idx: Dict[str, int],
        idx_to_label: Dict[int, str],
    ):
        self.centroids = centroids
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        first = next(iter(centroids.values()), None)
        self.embedding_dim = int(first.shape[0]) if first is not None else 0

    def _l2_normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors for cosine distance."""
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            return vectors / norm if norm > 0 else vectors
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between two vectors."""
        a_norm = self._l2_normalize(a)
        b_norm = self._l2_normalize(b)
        return 1 - float(np.dot(a_norm, b_norm))

    def predict(self, embedding: np.ndarray) -> str:
        """Predict label for a raw CLIP embedding."""
        x = self._l2_normalize(embedding.reshape(-1))

        best_label = None
        best_distance = float("inf")

        for label, centroid in self.centroids.items():
            dist = self.cosine_distance(x, centroid)
            if dist < best_distance:
                best_distance = dist
                best_label = label

        return best_label

    def predict_with_confidence(self, embedding: np.ndarray) -> Tuple[str, float]:
        x = self._l2_normalize(embedding.reshape(-1))
        probs = self._compute_probabilities(x)

        best_label = max(probs, key=probs.get)
        return best_label, probs[best_label]

    def _compute_probabilities(self, x_norm: np.ndarray) -> Dict[str, float]:
        """Softmax over negative cosine distances (x must be L2-normalized)."""
        distances = {}
        for label, centroid in self.centroids.items():
            distances[label] = self.cosine_distance(x_norm, centroid)

        neg_distances = {label: -d for label, d in distances.items()}
        max_neg = max(neg_distances.values())
        exp_scores = {label: np.exp(nd - max_neg) for label, nd in neg_distances.items()}
        total = sum(exp_scores.values())

        return {label: score / total for label, score in exp_scores.items()}

    def predict_proba(self, embedding: np.ndarray) -> Dict[str, float]:
        x = self._l2_normalize(embedding.reshape(-1))
        return self._compute_probabilities(x)

    def predict_ambiguous_set(
        self,
        embedding: np.ndarray,
        relative_margin: float = 0.08,
        min_absolute: float = 0.02,
    ) -> Dict[str, Any]:
        """Best centroid plus classes within a cosine-distance band."""
        x = self._l2_normalize(embedding.reshape(-1))
        dists: Dict[str, float] = {}
        for label, centroid in self.centroids.items():
            dists[label] = self.cosine_distance(x, centroid)
        best_label = min(dists, key=dists.get)
        best_d = dists[best_label]
        thresh = best_d + max(min_absolute, relative_margin * max(best_d, 1e-8))
        ambiguous = sorted(
            [lab for lab, d in dists.items() if d <= thresh],
            key=lambda lab: dists[lab],
        )
        return {
            "label": best_label,
            "best_distance": float(best_d),
            "threshold": float(thresh),
            "ambiguous_labels": ambiguous,
            "distances": {k: float(v) for k, v in dists.items()},
        }

    def predict_top_k(self, embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        probs = self.predict_proba(embedding)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:k]

    @property
    def num_classes(self) -> int:
        return len(self.centroids)

    @property
    def labels(self) -> List[str]:
        return list(self.centroids.keys())


__all__ = ["CentroidModel"]
