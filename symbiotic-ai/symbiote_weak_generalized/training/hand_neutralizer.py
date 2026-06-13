"""
PCA-based hand embedding neutralizer: remove top principal directions learned from
empty-hand CLIP embeddings (skin tone / lighting bias).
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA


class HandNeutralizer:
    """
    Fit PCA on a small set of empty-hand embeddings, then subtract projections onto
    the top ``n_components`` principal axes from any input embedding.
    """

    def __init__(
        self,
        hand_embeddings_dir: str,
        n_components: int = 3,
        verbose: bool = True,
    ):
        self.hand_embeddings_dir = os.path.abspath(hand_embeddings_dir)
        self.n_components = int(n_components)
        self.verbose = bool(verbose)
        self._pca: Optional[PCA] = None
        self._components: Optional[np.ndarray] = None  # (k, D) orthonormal rows
        self._mean: Optional[np.ndarray] = None
        self._enabled = False
        self._fit()

    def _load_empty_hand_matrix(self) -> Optional[np.ndarray]:
        if not os.path.isdir(self.hand_embeddings_dir):
            if self.verbose:
                print(
                    f"  [HandNeutralizer] Directory not found: {self.hand_embeddings_dir!r} "
                    "(neutralization disabled)."
                )
            return None
        paths = sorted(
            glob.glob(os.path.join(self.hand_embeddings_dir, "*.npy"))
            + glob.glob(os.path.join(self.hand_embeddings_dir, "*.NPY"))
        )
        if not paths:
            if self.verbose:
                print(
                    f"  [HandNeutralizer] No .npy files in {self.hand_embeddings_dir!r} "
                    "(neutralization disabled)."
                )
            return None
        blocks: List[np.ndarray] = []
        for p in paths:
            arr = np.load(p, allow_pickle=False)
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                if self.verbose:
                    print(f"  [HandNeutralizer] Skip {p!r}: expected 1D or 2D array")
                continue
            if arr.shape[0] == 0:
                continue
            blocks.append(arr)
        if not blocks:
            return None
        return np.vstack(blocks)

    def _fit(self) -> None:
        X = self._load_empty_hand_matrix()
        if X is None or X.shape[0] < 2:
            if self.verbose and X is not None and X.shape[0] < 2:
                print(
                    "  [HandNeutralizer] Need at least 2 empty-hand vectors for PCA "
                    "(neutralization disabled)."
                )
            self._enabled = False
            return
        n_samples, dim = X.shape
        k = min(self.n_components, n_samples - 1, dim)
        if k <= 0:
            self._enabled = False
            return
        self._pca = PCA(n_components=k, svd_solver="full", random_state=0)
        self._pca.fit(X)
        # sklearn: components_ shape (k, D), rows are eigenvectors in original space
        self._components = np.asarray(self._pca.components_, dtype=np.float64)
        self._mean = np.asarray(self._pca.mean_, dtype=np.float64)
        self._enabled = True
        if self.verbose:
            ev = self._pca.explained_variance_ratio_
            print(
                f"  [HandNeutralizer] Fitted PCA k={k} on {n_samples} empty-hand rows, dim={dim}; "
                f"explained_variance_ratio (sum)={float(ev.sum()):.4f}"
            )

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self._components is not None)

    def neutralize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Remove projections onto fitted principal components (per row).

        Args:
            embeddings: (N, D) or (D,) float array.

        Returns:
            Same shape, float64; identity if disabled.
        """
        em = np.asarray(embeddings, dtype=np.float64)
        if em.size == 0:
            return em
        single = em.ndim == 1
        if single:
            em = em.reshape(1, -1)
        if not self.enabled or self._components is None:
            out = em.copy()
            return out[0] if single else out
        # Center with PCA training mean for consistent subspace removal
        X = em - self._mean.reshape(1, -1)
        # Projection onto row-space of components: (X @ C.T) @ C
        coeff = X @ self._components.T
        proj = coeff @ self._components
        out = X - proj
        if single:
            return out[0]
        return out

    def state_dict(self) -> dict:
        """Serializable state for model persistence."""
        return {
            "hand_embeddings_dir": self.hand_embeddings_dir,
            "n_components": self.n_components,
            "enabled": self.enabled,
            "mean": None if self._mean is None else self._mean.tolist(),
            "components": None if self._components is None else self._components.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict, verbose: bool = False) -> "HandNeutralizer":
        obj = cls.__new__(cls)  # type: ignore
        obj.hand_embeddings_dir = str(state.get("hand_embeddings_dir", ""))
        obj.n_components = int(state.get("n_components", 3))
        obj.verbose = bool(verbose)
        obj._pca = None
        enabled = bool(state.get("enabled", False))
        mean = state.get("mean")
        comp = state.get("components")
        if enabled and mean is not None and comp is not None:
            obj._mean = np.asarray(mean, dtype=np.float64)
            obj._components = np.asarray(comp, dtype=np.float64)
            obj._enabled = True
        else:
            obj._mean = None
            obj._components = None
            obj._enabled = False
        return obj


__all__ = ["HandNeutralizer"]
