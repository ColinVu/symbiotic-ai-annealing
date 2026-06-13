"""Aggregate segment embeddings across videos for global item centroids and item x item sim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .geometry import cosine_similarity_matrix, spherical_mean
from .segments import SegmentEmbeddings


def aggregate_all_frames_by_item(
    per_video: List[Tuple[str, List[SegmentEmbeddings]]],
) -> Dict[str, List[np.ndarray]]:
    """
    Collect raw frame embedding rows per item (string label), across all videos/segments.
    `per_video` is list of (video_stem, segments) from successful runs.
    """
    by_item: Dict[str, List[np.ndarray]] = {}
    for _stem, segs in per_video:
        for s in segs:
            if s.is_placeholder or s.embeddings.shape[0] == 0:
                continue
            lab = s.true_label
            if lab not in by_item:
                by_item[lab] = []
            by_item[lab].append(s.embeddings)
    return by_item


def item_centroids_from_aggregated_frames(
    by_item: Dict[str, List[np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Spherical mean over *all* frames of that item, concatenated across segments."""
    out: Dict[str, np.ndarray] = {}
    for lab, blocks in sorted(by_item.items()):
        if not blocks:
            continue
        E = np.concatenate(blocks, axis=0)
        if E.shape[0] == 0:
            continue
        c = spherical_mean(E)
        if np.linalg.norm(c) < 1e-12:
            continue
        out[lab] = c
    return out


@dataclass
class ItemItemMatrix:
    item_labels: List[str]
    matrix: np.ndarray

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "item_labels": self.item_labels,
            "matrix": self.matrix.tolist(),
        }


def build_item_to_item_matrix(
    item_to_centroid: Dict[str, np.ndarray],
) -> ItemItemMatrix:
    """One row per item: unit-normed centroid; MxM cosine matrix."""
    labels = sorted(item_to_centroid.keys())
    if not labels:
        return ItemItemMatrix(item_labels=[], matrix=np.zeros((0, 0), dtype=np.float64))
    M = np.stack([item_to_centroid[k] for k in labels], axis=0)
    return ItemItemMatrix(
        item_labels=labels,
        matrix=cosine_similarity_matrix(M),
    )


def _stack_labeled_frame_rows(
    by_item: Dict[str, List[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """All cached frame rows with string labels, or None if nothing to stack."""
    if not by_item:
        return None
    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    for lab in sorted(by_item.keys()):
        blocks = by_item[lab]
        if not blocks:
            continue
        E = np.concatenate(blocks, axis=0)
        if E.shape[0] == 0:
            continue
        for r in range(E.shape[0]):
            X_list.append(E[r].astype(np.float64, copy=False))
            y_list.append(lab)
    if not X_list:
        return None
    return np.stack(X_list, axis=0), np.asarray(y_list, dtype=object)


def build_item_to_item_matrix_logreg(
    by_item: Dict[str, List[np.ndarray]],
) -> ItemItemMatrix:
    """
    One L2-penalized binary logistic model per class (one-vs-rest on other classes),
    using balanced class weighting within each subproblem. Cosine similarity of the
    rows of coef_ (n_classes x d), L2-normalized like other cosine geometry here.

    If there are **exactly two** item labels, rows are **spherical class means** (same
    as centroid mode): binary `coef_` in scikit-learn is a single hyperplane, and
    K=2 one-vs-rest weight directions are not a good pairwise item basis.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Logistic regression requires scikit-learn. "
            "Install with: pip install scikit-learn"
        ) from e

    stacked = _stack_labeled_frame_rows(by_item)
    if stacked is None:
        return ItemItemMatrix(item_labels=[], matrix=np.zeros((0, 0), dtype=np.float64))
    X, y_str = stacked
    u = np.unique(y_str)
    if X.shape[0] < 2 or u.size < 2:
        return ItemItemMatrix(item_labels=[], matrix=np.zeros((0, 0), dtype=np.float64))

    # Sort for stable order / matrix alignment
    classes = sorted([str(s) for s in u.tolist()])
    if len(classes) == 2:
        # Binary sklearn LR yields one coef row; OVR for K=2 often gives opposing directions.
        # Use spherical class means (same as non–log-reg path) for a proper 2x2 cosine matrix.
        rows: list[np.ndarray] = []
        for cls in classes:
            blocks = by_item.get(cls, [])
            if not blocks:
                return ItemItemMatrix(item_labels=[], matrix=np.zeros((0, 0), dtype=np.float64))
            E = np.concatenate(blocks, axis=0)
            rows.append(spherical_mean(E))
        W = np.stack(rows, axis=0)
        return ItemItemMatrix(
            item_labels=classes,
            matrix=cosine_similarity_matrix(W),
        )

    W_rows: list[np.ndarray] = []
    for cls in classes:
        y_bin = (y_str == cls).astype(int)
        if y_bin.max() == y_bin.min():
            return ItemItemMatrix(item_labels=[], matrix=np.zeros((0, 0), dtype=np.float64))
        lr = LogisticRegression(
            max_iter=4000,
            solver="lbfgs",
            C=1.0,
            class_weight="balanced",
        )
        lr.fit(X, y_bin)
        W_rows.append(np.asarray(lr.coef_, dtype=np.float64).reshape(-1))
    W = np.stack(W_rows, axis=0)
    return ItemItemMatrix(
        item_labels=classes,
        matrix=cosine_similarity_matrix(W),
    )
