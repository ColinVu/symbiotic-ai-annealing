"""
Fast cache-only confusion-matrix evaluation for saved centroid models.

Rebuilds the same carry segments as ``train-from-cache``, predicts with saved
centroids (no CLIP load, no annealing), and writes frame-sampled and
per-segment kNN confusion matrices as CSV + PNG.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from ..experiments.evaluator import _load_ground_truth_labels
from ..persistence.model_io import load_model
from ..pipelines.video_training import (
    _list_videos_in_folder,
    _load_picklists_nested_from_json,
    _process_single_video_from_cache,
)
from ..training.weak_supervision import Segment, spherical_mean
from ..visualization.plots import plot_confusion_matrix

# Last digit of picklist stem → shelf prefix (c/d/e/f/g)
SHELF_DIGIT_MAP: Dict[str, str] = {"1": "c", "2": "d", "3": "e", "4": "f", "5": "g"}
SHELF_ORDER: List[str] = ["c", "d", "e", "f", "g"]


def _shelf_for_stem(stem: str) -> Optional[str]:
    """Return shelf prefix for a picklist stem based on its last character."""
    return SHELF_DIGIT_MAP.get(stem[-1]) if stem else None


def _shelf_sort_key(label: str) -> Tuple[int, str]:
    """Sort key: group by shelf prefix order (c→g), then alphabetically within shelf."""
    prefix = label[0].lower() if label else ""
    shelf_idx = SHELF_ORDER.index(prefix) if prefix in SHELF_ORDER else len(SHELF_ORDER)
    return (shelf_idx, label)


@dataclass
class PredictionRow:
    video_stem: str
    segment_id: int
    true_label: str
    predicted_label: str
    confidence: float
    sample_index: int = -1
    num_cached_frames: int = 0


@dataclass
class EvalAccumulator:
    frame_rows: List[PredictionRow] = field(default_factory=list)
    segment_rows: List[PredictionRow] = field(default_factory=list)
    # Per-segment kNN predictions constrained to the correct shelf only
    shelf_rows: List[PredictionRow] = field(default_factory=list)
    skipped_segments: List[Dict[str, Any]] = field(default_factory=list)
    skipped_videos: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class _VideoSegmentBundle:
    video_stem: str
    segments: List[Segment]
    ground_truth: List[str]
    shelf_prefix: Optional[str]


@dataclass
class CachedKNNIndex:
    """L2-normalised training embeddings with per-frame ground-truth labels."""

    embeddings: np.ndarray  # (N, D), unit-norm rows
    labels: List[str]
    video_stems: List[str]

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])

    @classmethod
    def build_from_videos(
        cls,
        predictor: "CachedCentroidPredictor",
        bundles: Sequence[_VideoSegmentBundle],
        *,
        fa_labels: Optional[Dict[Tuple[str, int], str]] = None,
        verbose: bool = True,
    ) -> "CachedKNNIndex":
        """Build the index.

        Label source priority (per segment):
          1. ``fa_labels[(video_stem, segment_id)]`` when provided (final_assignments)
          2. ``bundle.ground_truth[segment_id]`` fallback
        This lets the index cover all cached videos regardless of gt.csv coverage.
        """
        emb_chunks: List[np.ndarray] = []
        labels: List[str] = []
        video_stems: List[str] = []

        for bundle in bundles:
            for seg in bundle.segments:
                if seg.is_placeholder or seg.embeddings.size == 0:
                    continue
                key = (bundle.video_stem, int(seg.segment_id))
                if fa_labels is not None:
                    label = fa_labels.get(key)
                elif seg.segment_id < len(bundle.ground_truth):
                    label = bundle.ground_truth[int(seg.segment_id)]
                else:
                    label = None
                if not label:
                    continue
                em = np.asarray(seg.embeddings, dtype=np.float64)
                for i in range(em.shape[0]):
                    processed = predictor.postprocess_embedding(em[i])
                    emb_chunks.append(_l2_normalize_row(processed))
                    labels.append(label)
                    video_stems.append(bundle.video_stem)

        if not emb_chunks:
            return cls(
                embeddings=np.zeros((0, 0), dtype=np.float64),
                labels=[],
                video_stems=[],
            )

        matrix = np.vstack(emb_chunks).astype(np.float64)
        if verbose:
            print(f"kNN index: {matrix.shape[0]} training frames from {len(bundles)} videos")
        return cls(embeddings=matrix, labels=labels, video_stems=video_stems)

    def predict(
        self,
        query_emb: np.ndarray,
        k: int,
        *,
        exclude_video: Optional[str] = None,
        shelf_prefix: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Return majority label among *k* nearest neighbours and vote fraction."""
        if self.size == 0:
            return "", 0.0

        q = _l2_normalize_row(np.asarray(query_emb, dtype=np.float64).reshape(-1))
        mask = np.ones(self.size, dtype=bool)
        if exclude_video is not None:
            mask &= np.array([vs != exclude_video for vs in self.video_stems], dtype=bool)
        if shelf_prefix is not None:
            mask &= np.array(
                [lab.startswith(shelf_prefix) for lab in self.labels], dtype=bool
            )

        if not mask.any():
            if shelf_prefix is not None:
                return self.predict(
                    query_emb, k, exclude_video=exclude_video, shelf_prefix=None
                )
            return "", 0.0

        sub_emb = self.embeddings[mask]
        sub_labels = [lab for lab, keep in zip(self.labels, mask) if keep]
        # Cosine distance on unit vectors: 1 - dot product
        sims = sub_emb @ q
        dists = 1.0 - sims
        k_eff = min(max(1, int(k)), len(sub_labels))
        nn_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
        neighbor_labels = [sub_labels[int(i)] for i in nn_idx]
        vote_counts = Counter(neighbor_labels)
        pred_label, votes = vote_counts.most_common(1)[0]
        conf = float(votes) / float(k_eff)
        return str(pred_label), conf


def _l2_normalize_row(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / max(n, eps)).astype(np.float64)


class CachedCentroidPredictor:
    """Load saved centroids + optional transforms without CLIP."""

    def __init__(self, model_dir: str, device: Optional[str] = None):
        self.model_dir = str(Path(model_dir).resolve())
        self.model, self.metadata = load_model(self.model_dir)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._hand_neutralizer = None
        self._clip_adapter = None
        self._load_transforms()

    def _load_transforms(self) -> None:
        hn_path = os.path.join(self.model_dir, "hand_neutralizer.json")
        ap_path = os.path.join(self.model_dir, "clip_adapter.pt")
        if os.path.isfile(hn_path):
            from ..training.hand_neutralizer import HandNeutralizer

            with open(hn_path, "r", encoding="utf-8") as f:
                self._hand_neutralizer = HandNeutralizer.from_state_dict(
                    json.load(f), verbose=False
                )
        if os.path.isfile(ap_path):
            from ..training.clip_adapter import CLIPAdapter

            dim = int(self.metadata.get("embedding_dim", 512))
            m = CLIPAdapter(dim)
            m.load_state_dict(torch.load(ap_path, map_location=self.device))
            m.eval()
            if self.device == "cuda":
                m = m.to(self.device)
            self._clip_adapter = m

    def postprocess_embedding(self, emb: np.ndarray) -> np.ndarray:
        """Match ``ObjectRecognizer._postprocess_embedding``."""
        x = np.asarray(emb, dtype=np.float64).reshape(-1)
        if self._hand_neutralizer is not None and self._hand_neutralizer.enabled:
            x = np.asarray(self._hand_neutralizer.neutralize(x), dtype=np.float64).reshape(-1)
        if self._clip_adapter is not None:
            with torch.no_grad():
                t = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
                x = self._clip_adapter(t).cpu().numpy().reshape(-1)
        return x.astype(np.float64)

    def predict_with_confidence(self, emb: np.ndarray) -> Tuple[str, float]:
        processed = self.postprocess_embedding(emb)
        return self.model.predict_with_confidence(processed)

    def predict_constrained(
        self, processed_emb: np.ndarray, shelf_prefix: str
    ) -> Tuple[str, float]:
        """Nearest-centroid prediction restricted to labels starting with ``shelf_prefix``.

        Softmax confidence is renormalised over the candidate subset.
        Falls back to unconstrained if no centroids match the prefix.
        """
        candidates = {
            lab: c
            for lab, c in self.model.centroids.items()
            if lab.startswith(shelf_prefix)
        }
        if not candidates:
            return self.model.predict_with_confidence(processed_emb)
        x = self.model._l2_normalize(processed_emb.reshape(-1))
        best_label, best_dist = None, float("inf")
        for lab, centroid in candidates.items():
            dist = self.model.cosine_distance(x, centroid)
            if dist < best_dist:
                best_dist, best_label = dist, lab
        # Renormalised softmax confidence over candidate subset
        import math
        neg_dists = {
            lab: -self.model.cosine_distance(x, centroid)
            for lab, centroid in candidates.items()
        }
        max_neg = max(neg_dists.values())
        exp_scores = {lab: math.exp(v - max_neg) for lab, v in neg_dists.items()}
        total = sum(exp_scores.values())
        conf = exp_scores.get(best_label, 0.0) / max(total, 1e-12)
        return str(best_label), float(conf)


def _ordered_labels(
    centroid_labels: Sequence[str],
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
) -> List[str]:
    """Collect all labels and sort them grouped by shelf (c→d→e→f→g), then alphabetically."""
    all_labels: set = set()
    for lab in list(centroid_labels) + list(true_labels) + list(pred_labels):
        if lab:
            all_labels.add(lab)
    return sorted(all_labels, key=_shelf_sort_key)


def _build_confusion(
    rows: Sequence[PredictionRow],
    label_order: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, float]:
    if not rows:
        empty = np.zeros((len(label_order), len(label_order)), dtype=np.int64)
        return empty, empty.astype(np.float64), 0.0
    y_true = [r.true_label for r in rows]
    y_pred = [r.predicted_label for r in rows]
    cm_raw = confusion_matrix(y_true, y_pred, labels=list(label_order))
    row_sums = np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
    cm_norm = cm_raw.astype(np.float64) / row_sums
    hits = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = hits / len(rows)
    return cm_raw, cm_norm, acc


def _write_prediction_csv(path: Path, rows: Sequence[PredictionRow], *, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "frame":
        fieldnames = [
            "video_stem",
            "segment_id",
            "sample_index",
            "true_label",
            "predicted_label",
            "confidence",
            "hit",
        ]
    else:
        fieldnames = [
            "video_stem",
            "segment_id",
            "true_label",
            "predicted_label",
            "confidence",
            "num_cached_frames",
            "hit",
        ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            hit = int(r.true_label == r.predicted_label)
            if mode == "frame":
                w.writerow(
                    {
                        "video_stem": r.video_stem,
                        "segment_id": r.segment_id,
                        "sample_index": r.sample_index,
                        "true_label": r.true_label,
                        "predicted_label": r.predicted_label,
                        "confidence": f"{r.confidence:.6f}",
                        "hit": hit,
                    }
                )
            else:
                w.writerow(
                    {
                        "video_stem": r.video_stem,
                        "segment_id": r.segment_id,
                        "true_label": r.true_label,
                        "predicted_label": r.predicted_label,
                        "confidence": f"{r.confidence:.6f}",
                        "num_cached_frames": r.num_cached_frames,
                        "hit": hit,
                    }
                )


def _write_matrix_csv(path: Path, cm: np.ndarray, labels: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *labels])
        for i, true_lab in enumerate(labels):
            w.writerow([true_lab, *[int(cm[i, j]) for j in range(len(labels))]])


def _write_matrix_csv_float(path: Path, cm: np.ndarray, labels: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *labels])
        for i, true_lab in enumerate(labels):
            w.writerow([true_lab, *[f"{cm[i, j]:.6f}" for j in range(len(labels))]])


def _segment_knn_prediction(
    knn_index: CachedKNNIndex,
    processed_frames: np.ndarray,
    *,
    video_stem: str,
    knn_k: int,
    shelf_prefix: Optional[str],
) -> Tuple[str, float]:
    """Spherical mean of segment frames → single kNN query.

    Aggregating first removes per-frame noise before searching neighbours,
    giving the same denoising benefit as spherical-mean centroid classification
    while using the richer kNN decision boundary.
    """
    mean_emb = spherical_mean(processed_frames)
    return knn_index.predict(
        mean_emb,
        knn_k,
        exclude_video=video_stem,
        shelf_prefix=shelf_prefix,
    )


def _evaluate_segments_for_video(
    predictor: CachedCentroidPredictor,
    knn_index: CachedKNNIndex,
    segments: List[Segment],
    video_stem: str,
    ground_truth: List[str],
    *,
    shelf_prefix: Optional[str],
    knn_k: int,
    samples_per_segment: int,
    rng: np.random.Generator,
    acc: EvalAccumulator,
) -> None:
    for seg in segments:
        if seg.is_placeholder or seg.embeddings.size == 0:
            acc.skipped_segments.append(
                {
                    "video_stem": video_stem,
                    "segment_id": int(seg.segment_id),
                    "reason": "placeholder_or_empty_embeddings",
                }
            )
            continue
        if seg.segment_id >= len(ground_truth):
            acc.skipped_segments.append(
                {
                    "video_stem": video_stem,
                    "segment_id": int(seg.segment_id),
                    "reason": "segment_id_outside_ground_truth",
                }
            )
            continue

        true_label = ground_truth[int(seg.segment_id)]
        em = np.asarray(seg.embeddings, dtype=np.float64)
        n_frames = int(em.shape[0])

        # Postprocess each frame in model space, then kNN majority-vote per segment.
        processed_frames = np.vstack(
            [predictor.postprocess_embedding(em[i]) for i in range(n_frames)]
        )

        pred_label, conf = _segment_knn_prediction(
            knn_index,
            processed_frames,
            video_stem=video_stem,
            knn_k=knn_k,
            shelf_prefix=None,
        )
        if pred_label:
            acc.segment_rows.append(
                PredictionRow(
                    video_stem=video_stem,
                    segment_id=int(seg.segment_id),
                    true_label=true_label,
                    predicted_label=str(pred_label),
                    confidence=float(conf),
                    num_cached_frames=n_frames,
                )
            )
        else:
            acc.skipped_segments.append(
                {
                    "video_stem": video_stem,
                    "segment_id": int(seg.segment_id),
                    "reason": "knn_no_neighbors",
                }
            )

        if shelf_prefix is not None:
            shelf_pred, shelf_conf = _segment_knn_prediction(
                knn_index,
                processed_frames,
                video_stem=video_stem,
                knn_k=knn_k,
                shelf_prefix=shelf_prefix,
            )
            if shelf_pred:
                acc.shelf_rows.append(
                    PredictionRow(
                        video_stem=video_stem,
                        segment_id=int(seg.segment_id),
                        true_label=true_label,
                        predicted_label=str(shelf_pred),
                        confidence=float(shelf_conf),
                        num_cached_frames=n_frames,
                    )
                )

        # Frame-sampled predictions (unconstrained)
        k = min(max(1, samples_per_segment), n_frames)
        if k == n_frames:
            sample_indices = np.arange(n_frames, dtype=np.int64)
        else:
            sample_indices = rng.choice(n_frames, size=k, replace=False)
        for sample_idx in sample_indices:
            frame_emb = processed_frames[int(sample_idx)]
            pred_l, sample_conf = predictor.model.predict_with_confidence(frame_emb)
            acc.frame_rows.append(
                PredictionRow(
                    video_stem=video_stem,
                    segment_id=int(seg.segment_id),
                    true_label=true_label,
                    predicted_label=str(pred_l),
                    confidence=float(sample_conf),
                    sample_index=int(sample_idx),
                )
            )


def _load_final_assignments(fa_csv_path: str) -> Dict[Tuple[str, int], str]:
    """Load final_assignments.csv → {(video_stem, segment_id): assigned_label}."""
    out: Dict[Tuple[str, int], str] = {}
    fa_path = Path(fa_csv_path)
    if not fa_path.is_file():
        return out
    with fa_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            stem = (row.get("video_stem") or "").strip()
            seg_id_raw = (row.get("segment_id") or "").strip()
            label = (row.get("assigned_label") or "").strip()
            if stem and seg_id_raw.isdigit() and label:
                out[(stem, int(seg_id_raw))] = label
    return out


def run_confusion_from_cache(
    *,
    model_dir: str,
    videos_dir: str,
    picklist_json_dir: str,
    manual_labels_dir: str,
    ground_truth_csv: str,
    output_dir: str,
    cache_dir: Optional[str] = None,
    final_assignments_csv: Optional[str] = None,
    frame_skip: int = 4,
    compact_frame_indexing: str = "opencv0",
    samples_per_segment: int = 1,
    knn_k: int = 15,
    random_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    model_path = Path(model_dir).resolve()
    out_path = Path(output_dir).resolve()
    gt_path = Path(ground_truth_csv).resolve()
    cache_root = Path(cache_dir).resolve() if cache_dir else model_path / ".cache"

    # Resolve final_assignments.csv — default to <model_dir>/final_assignments.csv
    if final_assignments_csv is None:
        fa_path_candidate = model_path / "final_assignments.csv"
        final_assignments_csv = str(fa_path_candidate) if fa_path_candidate.is_file() else None

    if not model_path.is_dir():
        raise SystemExit(f"Error: model-dir not found: {model_path}")
    if not gt_path.is_file():
        raise SystemExit(f"Error: ground-truth-csv not found: {gt_path}")
    if not cache_root.is_dir():
        raise SystemExit(f"Error: cache-dir not found: {cache_root}")

    # Load final_assignments labels for kNN index
    fa_labels: Optional[Dict[Tuple[str, int], str]] = None
    if final_assignments_csv and Path(final_assignments_csv).is_file():
        fa_labels = _load_final_assignments(final_assignments_csv)
        if verbose:
            print(f"Loaded {len(fa_labels)} segment labels from final_assignments.csv")
    else:
        if verbose:
            print("No final_assignments.csv found — kNN index will use ground_truth.csv labels")

    predictor = CachedCentroidPredictor(str(model_path))
    video_paths = _list_videos_in_folder(videos_dir)
    rng = np.random.default_rng(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    acc = EvalAccumulator()

    if verbose:
        print("=" * 60)
        print("CONFUSION MATRIX FROM CACHE")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Cache root: {cache_root}")
        print(f"Videos: {len(video_paths)}")
        print(f"Samples per segment (frame mode): {samples_per_segment}")
        print(f"kNN k (segment + shelf modes): {knn_k}")
        print(f"Output: {out_path}")

    # ── Pass 1: load ALL videos that have cache + picklist into bundles ──────
    # These are used for kNN index building (labeled via final_assignments).
    # Videos without ground_truth.csv entries are index-only (no evaluation).
    eval_bundles: List[_VideoSegmentBundle] = []   # has gt → will be evaluated
    index_bundles: List[_VideoSegmentBundle] = []  # no gt → kNN training only

    for video_path in video_paths:
        stem = Path(video_path).stem
        json_path = os.path.join(picklist_json_dir, f"{stem}.json")
        if not os.path.isfile(json_path):
            acc.skipped_videos.append({"video_stem": stem, "reason": "missing_picklist_json"})
            continue
        per_video_cache = cache_root / stem
        if not per_video_cache.is_dir():
            acc.skipped_videos.append({"video_stem": stem, "reason": "missing_cache_dir"})
            continue

        # Determine whether this video has ground truth for evaluation
        try:
            ground_truth = _load_ground_truth_labels(str(gt_path), stem)
            has_gt = True
        except ValueError:
            ground_truth = []
            has_gt = False

        # Skip if no fa_labels entry and no gt — nothing to label this video with
        if not has_gt and fa_labels is None:
            acc.skipped_videos.append({"video_stem": stem, "reason": "no_labels_available"})
            continue

        picklists_nested = _load_picklists_nested_from_json(json_path)
        segments, _flat_picklist, video_name = _process_single_video_from_cache(
            video_path,
            picklists_nested,
            str(per_video_cache),
            manual_labels_dir,
            require_manual_label_csv=True,
            compact_frame_indexing=compact_frame_indexing,
            frame_skip=frame_skip,
            verbose=verbose,
        )
        shelf_prefix = _shelf_for_stem(stem)
        bundle = _VideoSegmentBundle(
            video_stem=video_name,
            segments=segments,
            ground_truth=ground_truth,
            shelf_prefix=shelf_prefix,
        )
        if has_gt:
            eval_bundles.append(bundle)
        else:
            index_bundles.append(bundle)

    all_bundles = eval_bundles + index_bundles
    if verbose:
        print(
            f"Bundles: {len(eval_bundles)} eval videos, "
            f"{len(index_bundles)} index-only videos ({len(all_bundles)} total)"
        )

    # ── Build kNN index from ALL bundles, labeled by final_assignments ───────
    knn_index = CachedKNNIndex.build_from_videos(
        predictor, all_bundles, fa_labels=fa_labels, verbose=verbose
    )

    for bundle in eval_bundles:
        _evaluate_segments_for_video(
            predictor,
            knn_index,
            bundle.segments,
            bundle.video_stem,
            bundle.ground_truth,
            shelf_prefix=bundle.shelf_prefix,
            knn_k=knn_k,
            samples_per_segment=samples_per_segment,
            rng=rng,
            acc=acc,
        )
        if verbose:
            n_seg = sum(1 for r in acc.segment_rows if r.video_stem == bundle.video_stem)
            n_frame = sum(1 for r in acc.frame_rows if r.video_stem == bundle.video_stem)
            n_shelf = sum(1 for r in acc.shelf_rows if r.video_stem == bundle.video_stem)
            shelf_tag = f", shelf={bundle.shelf_prefix}" if bundle.shelf_prefix else ""
            print(
                f"  [{bundle.video_stem}] segments={n_seg}, "
                f"frame_samples={n_frame}, shelf_constrained={n_shelf}{shelf_tag}"
            )

    if not acc.segment_rows and not acc.frame_rows:
        raise SystemExit(
            "Error: No predictions collected. Check cache, ground_truth.csv columns, and manual labels."
        )

    centroid_labels = list(predictor.metadata.get("centroid_labels", predictor.model.labels))
    all_true = [r.true_label for r in acc.frame_rows + acc.segment_rows + acc.shelf_rows]
    all_pred = [r.predicted_label for r in acc.frame_rows + acc.segment_rows + acc.shelf_rows]
    # All three matrices use the same shelf-grouped label order
    label_order = _ordered_labels(centroid_labels, all_true, all_pred)

    frame_cm_raw, frame_cm_norm, frame_acc = _build_confusion(acc.frame_rows, label_order)
    seg_cm_raw, seg_cm_norm, seg_acc = _build_confusion(acc.segment_rows, label_order)
    shelf_cm_raw, shelf_cm_norm, shelf_acc = _build_confusion(acc.shelf_rows, label_order)

    # Per-shelf accuracy for the shelf-constrained matrix
    shelf_acc_by_shelf: Dict[str, Any] = {}
    for shelf_prefix, shelf_letter in sorted(
        {v: v for v in SHELF_DIGIT_MAP.values()}.items()
    ):
        shelf_subset = [r for r in acc.shelf_rows if r.true_label.startswith(shelf_letter)]
        if shelf_subset:
            hits = sum(1 for r in shelf_subset if r.true_label == r.predicted_label)
            shelf_acc_by_shelf[shelf_letter] = {
                "count": len(shelf_subset),
                "hits": hits,
                "accuracy": hits / len(shelf_subset),
            }
        else:
            shelf_acc_by_shelf[shelf_letter] = {"count": 0, "hits": 0, "accuracy": None}

    out_path.mkdir(parents=True, exist_ok=True)

    # Prediction row CSVs
    _write_prediction_csv(out_path / "frame_sample_predictions.csv", acc.frame_rows, mode="frame")
    _write_prediction_csv(out_path / "segment_mean_predictions.csv", acc.segment_rows, mode="segment")
    _write_prediction_csv(out_path / "shelf_constrained_predictions.csv", acc.shelf_rows, mode="segment")

    # Count matrices
    _write_matrix_csv(out_path / "frame_sample_confusion_counts.csv", frame_cm_raw, label_order)
    _write_matrix_csv(out_path / "segment_mean_confusion_counts.csv", seg_cm_raw, label_order)
    _write_matrix_csv(out_path / "shelf_constrained_confusion_counts.csv", shelf_cm_raw, label_order)

    # Normalised matrices
    _write_matrix_csv_float(
        out_path / "frame_sample_confusion_normalized.csv", frame_cm_norm, label_order
    )
    _write_matrix_csv_float(
        out_path / "segment_mean_confusion_normalized.csv", seg_cm_norm, label_order
    )
    _write_matrix_csv_float(
        out_path / "shelf_constrained_confusion_normalized.csv", shelf_cm_norm, label_order
    )

    # PNGs — no tick labels (too many classes), shelf-ordered
    plot_confusion_matrix(
        frame_cm_norm,
        frame_cm_raw,
        list(label_order),
        str(out_path / "frame_sample_confusion.png"),
        show_labels=False,
    )
    plot_confusion_matrix(
        seg_cm_norm,
        seg_cm_raw,
        list(label_order),
        str(out_path / "segment_mean_confusion.png"),
        show_labels=False,
    )
    plot_confusion_matrix(
        shelf_cm_norm,
        shelf_cm_raw,
        list(label_order),
        str(out_path / "shelf_constrained_confusion.png"),
        show_labels=False,
    )

    summary: Dict[str, Any] = {
        "model_dir": str(model_path),
        "cache_dir": str(cache_root),
        "ground_truth_csv": str(gt_path),
        "output_dir": str(out_path),
        "frame_skip": int(frame_skip),
        "compact_frame_indexing": compact_frame_indexing,
        "samples_per_segment": int(samples_per_segment),
        "knn_k": int(knn_k),
        "knn_train_frames": knn_index.size,
        "knn_train_videos": len(all_bundles),
        "final_assignments_csv": str(final_assignments_csv) if final_assignments_csv else None,
        "random_seed": int(random_seed),
        "shelf_digit_map": SHELF_DIGIT_MAP,
        "label_order": label_order,
        "metrics": {
            "frame_sample_count": len(acc.frame_rows),
            "frame_sample_accuracy": float(frame_acc),
            "segment_mean_count": len(acc.segment_rows),
            "segment_mean_accuracy": float(seg_acc),
            "shelf_constrained_count": len(acc.shelf_rows),
            "shelf_constrained_accuracy": float(shelf_acc),
            "shelf_constrained_accuracy_by_shelf": shelf_acc_by_shelf,
        },
        "skipped_videos": acc.skipped_videos,
        "skipped_segments": acc.skipped_segments,
    }
    with (out_path / "confusion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print("\nResults:")
        print(f"  Frame-sampled:      n={len(acc.frame_rows)}, accuracy={frame_acc * 100:.2f}%")
        print(f"  Segment-kNN:       n={len(acc.segment_rows)}, accuracy={seg_acc * 100:.2f}%")
        print(f"  Shelf-constrained:  n={len(acc.shelf_rows)}, accuracy={shelf_acc * 100:.2f}%")
        for shelf_letter, stats in shelf_acc_by_shelf.items():
            if stats["accuracy"] is not None:
                print(
                    f"    shelf {shelf_letter}: n={stats['count']}, "
                    f"hits={stats['hits']}, accuracy={stats['accuracy'] * 100:.2f}%"
                )
        print(f"  Wrote outputs to {out_path}")

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build frame-sampled and per-segment kNN confusion matrices from cached embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", type=str, required=True, help="Saved centroid model directory")
    p.add_argument(
        "--videos",
        type=str,
        required=True,
        help="Directory of videos (metadata only; same as train-from-cache)",
    )
    p.add_argument("--picklist-json-dir", type=str, required=True)
    p.add_argument("--manual-labels-dir", type=str, required=True)
    p.add_argument("--ground-truth-csv", type=str, required=True)
    p.add_argument(
        "--final-assignments-csv",
        type=str,
        default=None,
        help="final_assignments.csv used to label kNN training frames (default: <model-dir>/final_assignments.csv)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write CSV/PNG outputs (default: <model-dir>/confusion_eval)",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Per-video cache root (default: <model-dir>/.cache)",
    )
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument(
        "--compact-frame-indexing",
        type=str,
        default="opencv0",
        choices=["opencv0", "pipeline1"],
    )
    p.add_argument(
        "--samples-per-segment",
        type=int,
        default=1,
        help="Random cached frames to score per pick segment (frame-sampled matrix)",
    )
    p.add_argument(
        "--knn-k",
        type=int,
        default=15,
        help="Number of nearest training frames for segment-kNN and shelf-constrained modes",
    )
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    script_dir = Path(__file__).resolve().parents[2]
    model_dir = os.path.normpath(os.path.join(script_dir, args.model_dir))
    videos_dir = os.path.normpath(os.path.join(script_dir, args.videos))
    picklist_json_dir = os.path.normpath(os.path.join(script_dir, args.picklist_json_dir))
    manual_labels_dir = os.path.normpath(os.path.join(script_dir, args.manual_labels_dir))
    ground_truth_csv = os.path.normpath(os.path.join(script_dir, args.ground_truth_csv))
    output_dir = (
        os.path.normpath(os.path.join(script_dir, args.output_dir))
        if args.output_dir
        else os.path.join(model_dir, "confusion_eval")
    )
    cache_dir = (
        os.path.normpath(os.path.join(script_dir, args.cache_dir))
        if args.cache_dir
        else None
    )
    final_assignments_csv = (
        os.path.normpath(os.path.join(script_dir, args.final_assignments_csv))
        if args.final_assignments_csv
        else None
    )

    run_confusion_from_cache(
        model_dir=model_dir,
        videos_dir=videos_dir,
        picklist_json_dir=picklist_json_dir,
        manual_labels_dir=manual_labels_dir,
        ground_truth_csv=ground_truth_csv,
        output_dir=output_dir,
        cache_dir=cache_dir,
        final_assignments_csv=final_assignments_csv,
        frame_skip=args.frame_skip,
        compact_frame_indexing=args.compact_frame_indexing,
        samples_per_segment=args.samples_per_segment,
        knn_k=args.knn_k,
        random_seed=args.random_seed,
        verbose=bool(args.verbose and not args.quiet),
    )


if __name__ == "__main__":
    main()
