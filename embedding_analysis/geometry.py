"""Unit-sphere helpers and cosine similarity (no training dependencies)."""

from __future__ import annotations

import numpy as np


def l2_normalize_rows(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return (x / n).astype(np.float64)


def spherical_mean(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Mean direction: L2 rows, mean, renormalize."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("spherical_mean: empty")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1)
    mask = (norms >= eps) & np.isfinite(norms)
    if not np.any(mask):
        return np.zeros(arr.shape[1], dtype=np.float64)
    u = arr[mask] / norms[mask, np.newaxis]
    s = u.mean(axis=0)
    ns = float(np.linalg.norm(s))
    if ns < eps or not np.isfinite(ns):
        return np.zeros(arr.shape[1], dtype=np.float64)
    return (s / ns).astype(np.float64)


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cos sim for row vectors (n, d) -> (n, n)."""
    u = l2_normalize_rows(vectors)
    return (u @ u.T).astype(np.float64)


def pairwise_cosine_off_diagonal(cos_matrix: np.ndarray) -> np.ndarray:
    n = cos_matrix.shape[0]
    if n < 2:
        return np.array([], dtype=np.float64)
    iu = np.triu_indices(n, k=1)
    return cos_matrix[iu].astype(np.float64)
