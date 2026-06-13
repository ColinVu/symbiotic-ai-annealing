"""
Weakly supervised training (ILR) for CLIP embeddings.

ILR refines picklist assignments by minimizing **frame-level** cosine distance to
per-label centroids (spherical mean of all frames with that label), with
**segment-level** label swaps (all frames in a segment share a label).

Picklist label format (training)
--------------------------------
``--label`` must be a JSON array of strings, one entry per **carry segment**
(temporal pick order in the video), e.g.::

    '["apple", "banana", "apple"]'

- Order in the list is **ignored** for supervision: the multiset of strings
  must match the multiset of objects picked; ILR discovers which segment
  corresponds to which list entry.
- Duplicate SKUs are allowed (same string repeated); segments are still
  distinguishable by appearance in embedding space.
- Length must equal the number of ``CARRY_WITH`` segments detected for the
  video (the pipeline may trim/pad segments to match).

**Multi-picklist videos** (``example.json``): use ``picklists: [[...], [...], ...]``.
Each inner list is one picklist round (ordered relative to other rounds, not
within the list). Carry segments in time order map to rounds in sequence. By
default ILR swaps are only between segments that share the same candidate
multiset; set ``ilr_allow_cross_round_swaps`` to also exchange labels across
rounds in the same video (each segment's label must stay in that segment's
multiset).

See also: ``weak_supervision_clip_exhaustive_swap.py`` for the legacy
exhaustive pairwise swap + Metropolis acceptance (not used by default).

**Empty-hand split:** disabled. Training does not split carry segments into
separate ``empty_hand`` segments; carry segments stay as produced by the pipeline.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.cluster import KMeans

LabelKey = Tuple[str, int]

# Discovered / fixed label for empty-hand frames (cross-segment clustering).
EMPTY_HAND_LABEL = "empty_hand"


def spherical_mean(vectors: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Mean direction on the unit sphere: L2-normalize each row, average, re-normalize.
    All-zero rows (e.g. placeholder segments) are skipped; if all rows are zero,
    returns a zero vector of the same dimension.
    """
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        raise ValueError("spherical_mean: empty input")
    norms = np.linalg.norm(arr, axis=1)
    mask = (norms >= eps) & np.isfinite(norms)
    if not np.any(mask):
        return np.zeros(arr.shape[1], dtype=np.float64)
    unit = arr[mask] / norms[mask, np.newaxis]
    s = unit.mean(axis=0)
    ns = float(np.linalg.norm(s))
    if ns < eps or not np.isfinite(ns):
        return np.zeros(arr.shape[1], dtype=np.float64)
    return (s / ns).astype(np.float64)


@dataclass
class Segment:
    """A video segment containing multiple frame embeddings."""

    segment_id: int
    embeddings: np.ndarray  # Shape: (num_frames, embedding_dim)
    video_id: str
    #: Weak-supervision multiset for this segment (same tuple for all segments in one round).
    candidate_labels: Optional[Tuple[str, ...]] = None
    #: True if this segment is a placeholder (no valid frames in its time window).
    is_placeholder: bool = False

    @property
    def label_key(self) -> LabelKey:
        """Unique key for label assignment (avoids collisions across videos)."""
        return (self.video_id, self.segment_id)

    @property
    def mean_embedding(self) -> np.ndarray:
        """Spherical mean of frame embeddings (unit-norm CLIP directions)."""
        return spherical_mean(self.embeddings)

    def compute_frame_costs(
        self,
        centroid: np.ndarray,
        cosine_distance_fn: Callable[[np.ndarray, np.ndarray], float],
    ) -> float:
        """
        Sum of per-frame cosine distances to ``centroid`` (e.g. ``1 - cos(frame, c)``).
        Used for ILR energy: segment-level swaps, frame-level costs.
        """
        em = np.asarray(self.embeddings, dtype=np.float64)
        if em.size == 0:
            return 0.0
        c = np.asarray(centroid, dtype=np.float64)
        return float(
            sum(cosine_distance_fn(em[i], c) for i in range(em.shape[0]))
        )


class WeakSupervisedTrainer:
    """
    Weakly supervised trainer using Iterative Label Refinement (ILR).

    ILR minimizes total cosine distance over **all frames** (sum of
    ``1 - cos(frame, centroid(label))``) with centroids = spherical mean of
    all frames assigned to that label. Swaps are still **segment-level** (all
    frames in a segment share the label); only the energy is frame-wise for
    discriminative signal.

    By default, swaps are only between segments sharing the same picklist round
    (identical ``candidate_labels``). Set ``ilr_allow_cross_round_swaps=True``
    to also consider pairs from **different** rounds in the same video; each
    segment's label after a swap must still belong to **that segment's**
    ``candidate_labels`` multiset (picklists remain hard constraints).

    Incremental updates: ``fit_iterative`` chooses a label permutation minimizing
    cosine distance to existing centroids, then EWMA or equal weight per video.
    """

    def __init__(
        self,
        ilr_epochs: int = 500,
        initial_temp: float = 1.0,
        temp_decay: str = "exponential",
        decay_rate: float = 0.98,
        random_seed: int = 42,
        variance_eps: float = 1e-6,
        bad_swap_cool_divisor: float = 200.0,
        detect_empty: bool = False,
        min_frames_per_cluster: int = 3,
        ilr_allow_cross_round_swaps: bool = False,
    ):
        self.ilr_epochs = ilr_epochs
        self.initial_temp = initial_temp
        self.temp_decay = temp_decay
        self.decay_rate = decay_rate
        self.random_seed = random_seed
        self.variance_eps = variance_eps
        self.bad_swap_cool_divisor = bad_swap_cool_divisor
        self.detect_empty = detect_empty
        self.min_frames_per_cluster = min_frames_per_cluster
        #: If False (default), ILR only swaps segments with identical ``candidate_labels``
        #: (same picklist round). If True, any two segments in the same video may swap
        #: provided each receives a label from its own multiset (cross-round exchange).
        self.ilr_allow_cross_round_swaps = ilr_allow_cross_round_swaps

        self.centroids: Dict[str, np.ndarray] = {}
        self.centroid_stds: Dict[str, np.ndarray] = {}
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        #: Per SKU label, per training-video stem: spherical mean of that video's segment means.
        #: Centroids are the spherical mean across videos (equal weight per video).
        self.label_video_means: Dict[str, Dict[str, np.ndarray]] = {}
        self.last_refined_labels: Optional[Dict[LabelKey, str]] = None
        #: Stems of videos whose embeddings were used (mirrors model_metadata embedded_video_stems when loaded).
        self.embedded_video_stems: List[str] = []

        #: Optional CLIP residual adapter (trained by iterated-model pipeline); CPU ``nn.Module``.
        self.clip_adapter: Optional[torch.nn.Module] = None
        #: Serializable hand-PCA state (``HandNeutralizer.state_dict()``).
        self.hand_neutralizer_state: Optional[Dict[str, Any]] = None

        random.seed(random_seed)
        np.random.seed(random_seed)

    def split_segments_by_empty_detection(
        self,
        segments: List[Segment],
        min_frames_per_cluster: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[List[Segment], List[Segment]]:
        """
        Legacy helper: split carry segments into item vs empty-hand via k=2 PCA
        clustering. **Not used** by ``fit()``; training keeps pipeline segments unchanged.

        Returns:
            (item_segments, empty_segments)
        """
        min_f = (
            min_frames_per_cluster
            if min_frames_per_cluster is not None
            else self.min_frames_per_cluster
        )
        if not segments:
            return [], []

        item_out: List[Segment] = []
        splittable: List[Tuple[Segment, np.ndarray, np.ndarray]] = []

        for seg in segments:
            embs = np.asarray(seg.embeddings, dtype=np.float64)
            n = embs.shape[0]
            if n < 2 * min_f:
                item_out.append(
                    Segment(
                        segment_id=seg.segment_id,
                        embeddings=embs,
                        video_id=seg.video_id,
                        candidate_labels=seg.candidate_labels,
                    )
                )
                continue

            klab = KMeans(
                n_clusters=2,
                random_state=self.random_seed,
                n_init=10,
            ).fit_predict(embs)
            c0 = embs[klab == 0].mean(axis=0)
            c1 = embs[klab == 1].mean(axis=0)
            centers = np.stack([c0, c1], axis=0)
            splittable.append((seg, klab.astype(np.int32, copy=False), centers))

        m = len(splittable)
        if m == 0:
            return item_out, []

        if m > 20:
            if verbose:
                print(
                    f"  [empty detection] {m} splittable segments (>20); "
                    f"skipping empty split for video {splittable[0][0].video_id!r}"
                )
            for seg, _, _ in splittable:
                embs = np.asarray(seg.embeddings, dtype=np.float64)
                item_out.append(
                    Segment(
                        segment_id=seg.segment_id,
                        embeddings=embs,
                        video_id=seg.video_id,
                        candidate_labels=seg.candidate_labels,
                    )
                )
            return item_out, []

        if m == 1:
            seg, _, _ = splittable[0]
            embs = np.asarray(seg.embeddings, dtype=np.float64)
            item_out.append(
                Segment(
                    segment_id=seg.segment_id,
                    embeddings=embs,
                    video_id=seg.video_id,
                    candidate_labels=seg.candidate_labels,
                )
            )
            if verbose:
                print(
                    f"  [empty detection] single splittable segment — skipping split "
                    f"for video {seg.video_id!r}"
                )
            return item_out, []

        centers_stack = np.stack([t[2] for t in splittable], axis=0)

        def _cos_sim(u: np.ndarray, v: np.ndarray) -> float:
            nu = float(np.linalg.norm(u))
            nv = float(np.linalg.norm(v))
            if nu == 0.0 or nv == 0.0:
                return 0.0
            return float(np.dot(u, v) / (nu * nv))

        best_mask = 0
        best_score = float("-inf")
        for mask in range(1 << m):
            score = 0.0
            for i in range(m):
                for j in range(i + 1, m):
                    ei = (mask >> i) & 1
                    ej = (mask >> j) & 1
                    score += _cos_sim(centers_stack[i, ei], centers_stack[j, ej])
            if score > best_score:
                best_score = score
                best_mask = mask

        segmented_empty_chunks: List[Tuple[Segment, np.ndarray]] = []

        for idx_s, (seg, klab, _centers) in enumerate(splittable):
            ei = (best_mask >> idx_s) & 1
            embs = np.asarray(seg.embeddings, dtype=np.float64)
            empty_mask = klab == ei
            item_embs = embs[~empty_mask]
            empty_embs = embs[empty_mask]

            if item_embs.shape[0] < min_f:
                item_out.append(
                    Segment(
                        segment_id=seg.segment_id,
                        embeddings=embs,
                        video_id=seg.video_id,
                        candidate_labels=seg.candidate_labels,
                    )
                )
                continue

            if empty_embs.shape[0] < min_f:
                item_out.append(
                    Segment(
                        segment_id=seg.segment_id,
                        embeddings=item_embs,
                        video_id=seg.video_id,
                        candidate_labels=seg.candidate_labels,
                    )
                )
                continue

            item_out.append(
                Segment(
                    segment_id=seg.segment_id,
                    embeddings=item_embs,
                    video_id=seg.video_id,
                    candidate_labels=seg.candidate_labels,
                )
            )
            segmented_empty_chunks.append((seg, empty_embs))

        k_empty_total = len(segmented_empty_chunks)
        empty_tuple = tuple([EMPTY_HAND_LABEL] * k_empty_total) if k_empty_total else tuple()
        empty_out: List[Segment] = []
        for seg, empty_embs in segmented_empty_chunks:
            empty_out.append(
                Segment(
                    segment_id=100_000 + int(seg.segment_id),
                    embeddings=empty_embs,
                    video_id=seg.video_id,
                    candidate_labels=empty_tuple if empty_tuple else None,
                )
            )

        if verbose and k_empty_total:
            vid = segments[0].video_id
            print(
                f"  [empty detection] video {vid!r}: {m} splittable segment(s) -> "
                f"{len(item_out)} item segment(s), {k_empty_total} empty segment(s) "
                f"(cross-seg score={best_score:.4f})"
            )

        return item_out, empty_out

    def _l2_normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = self._l2_normalize(a.reshape(1, -1))[0]
        b_norm = self._l2_normalize(b.reshape(1, -1))[0]
        return 1 - np.dot(a_norm, b_norm)

    def _candidate_multiset_tuple(self, seg: Segment, flat_picklist: Optional[List[str]]) -> Tuple[str, ...]:
        if seg.candidate_labels is not None:
            return seg.candidate_labels
        if flat_picklist is None:
            raise ValueError(
                "Segment missing candidate_labels and no flat picklist was provided for this video"
            )
        return tuple(flat_picklist)

    def initialize_labels(
        self,
        segments: List[Segment],
        picklist: List[str],
        video_id: str,
    ) -> Dict[LabelKey, str]:
        """Backward-compatible single flat multiset (all segments share *picklist*)."""
        if len(segments) != len(picklist):
            raise ValueError(
                f"Number of segments ({len(segments)}) must match picklist length ({len(picklist)})"
            )
        for seg in segments:
            seg.candidate_labels = tuple(picklist)
        return self._initialize_labels_for_segments(segments, picklist)

    def _initialize_labels_for_segments(
        self,
        segments: List[Segment],
        flat_picklist: Optional[List[str]],
    ) -> Dict[LabelKey, str]:
        """
        One random bijection per multiset group (same ``candidate_labels`` tuple).
        """
        labels: Dict[LabelKey, str] = {}
        groups: Dict[Tuple[str, Tuple[str, ...]], List[Segment]] = defaultdict(list)
        for seg in segments:
            tup = self._candidate_multiset_tuple(seg, flat_picklist)
            groups[(seg.video_id, tup)].append(seg)

        for (_vid_key, multiset), segs in groups.items():
            segs_sorted = sorted(segs, key=lambda s: s.segment_id)
            draw = list(multiset)
            if len(segs_sorted) != len(draw):
                raise ValueError(
                    f"Group size mismatch: {len(segs_sorted)} segments vs multiset size {len(draw)} "
                    f"for multiset {multiset}"
                )
            random.shuffle(draw)
            for seg, lab in zip(segs_sorted, draw):
                labels[seg.label_key] = lab
        return labels

    def compute_centroids(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
    ) -> Dict[str, np.ndarray]:
        """Per-label centroids: spherical mean of all frames with that label (not segment means)."""
        label_frames: Dict[str, List[np.ndarray]] = defaultdict(list)

        for seg in segments:
            label = labels[seg.label_key]
            em = np.asarray(seg.embeddings, dtype=np.float64)
            if em.size == 0:
                continue
            label_frames[label].append(em)

        centroids: Dict[str, np.ndarray] = {}
        for label, blocks in label_frames.items():
            all_frames = np.vstack(blocks)
            centroids[label] = spherical_mean(all_frames)

        return centroids

    def compute_centroid_stds(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
    ) -> Dict[str, np.ndarray]:
        """Per-label std over embedding dims on all frames (ddof=1); used for fit_iterative persistence."""
        label_frames: Dict[str, List[np.ndarray]] = defaultdict(list)
        for seg in segments:
            em = np.asarray(seg.embeddings, dtype=np.float64)
            if em.size == 0:
                continue
            label_frames[labels[seg.label_key]].append(em)

        stds: Dict[str, np.ndarray] = {}
        for label, blocks in label_frames.items():
            all_frames = np.vstack(blocks)
            d = all_frames.shape[1]
            if all_frames.shape[0] < 2:
                stds[label] = np.ones(d, dtype=np.float64)
            else:
                s = np.nan_to_num(all_frames.std(axis=0, ddof=1), nan=1e-5)
                stds[label] = np.maximum(s, self.variance_eps)
        return stds

    def _sync_centroids_from_label_video_means(self) -> None:
        """Set each centroid to the spherical mean of per-video means (equal weight per video)."""
        for lab, by_vid in self.label_video_means.items():
            mats = [np.asarray(v, dtype=np.float64) for v in by_vid.values()]
            if not mats:
                continue
            self.centroids[lab] = spherical_mean(np.stack(mats, axis=0))

    def _rebuild_label_video_means_from_segments(
        self,
        segments: List[Segment],
        refined_labels: Dict[LabelKey, str],
    ) -> None:
        """
        For each (label, video), spherical mean of all frames in segments with that
        label in that video; then centroids = spherical mean across per-video means.
        """
        acc: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
        for seg in segments:
            lab = refined_labels[seg.label_key]
            em = np.asarray(seg.embeddings, dtype=np.float64)
            if em.size == 0:
                continue
            acc[lab][seg.video_id].append(em)
        self.label_video_means = {}
        for lab, by_vid in acc.items():
            self.label_video_means[lab] = {}
            for vid, blocks in by_vid.items():
                all_frames = np.vstack(blocks)
                self.label_video_means[lab][vid] = spherical_mean(all_frames)
        self._sync_centroids_from_label_video_means()

    def compute_total_cost(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
        centroids: Dict[str, np.ndarray],
    ) -> float:
        """Sum of per-frame cosine distance to the assigned label centroid (same as ``compute_total_cosine_cost``)."""
        return self.compute_total_cosine_cost(segments, labels, centroids)

    def compute_total_variance_cost(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
        centroids: Dict[str, np.ndarray],
        stds: Dict[str, np.ndarray],
    ) -> float:
        """Deprecated: variance-normalized squared distance (legacy) per frame. Use ``compute_total_cosine_cost``."""
        total = 0.0
        for seg in segments:
            label = labels[seg.label_key]
            mean = centroids[label]
            std = stds[label]
            em = np.asarray(seg.embeddings, dtype=np.float64)
            if em.size == 0:
                continue
            for i in range(em.shape[0]):
                total += float(
                    np.sum(
                        ((mean - em[i]) ** 2) / (std**2 + self.variance_eps)
                    )
                )
        return total

    def compute_total_cosine_cost(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
        centroids: Dict[str, np.ndarray],
    ) -> float:
        """Sum over all frames of ``1 - cos(frame, centroid(label))``; centroids are spherical means of all frames per label."""
        total = 0.0
        for seg in segments:
            lab = labels[seg.label_key]
            total += seg.compute_frame_costs(centroids[lab], self.cosine_distance)
        return float(total)

    def compute_proxy_triplet_energy(
        self,
        embedding: np.ndarray,
        label: str,
        centroids: Dict[str, np.ndarray],
        margin: float = 0.1,
    ) -> float:
        """
        Per-frame proxy-triplet energy::

            d_p = cosine_distance(emb, centroid[label])
            d_n = min_{lab != label} cosine_distance(emb, centroid[lab])
            energy = max(d_p - d_n + margin, 0)
        """
        emb = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if label not in centroids:
            return 0.0
        d_p = self.cosine_distance(emb, centroids[label])
        d_n = float("inf")
        for lab, c in centroids.items():
            if lab == label:
                continue
            d_n = min(d_n, self.cosine_distance(emb, c))
        if not math.isfinite(d_n):
            d_n = 0.0
        return float(max(d_p - d_n + float(margin), 0.0))

    def compute_total_proxy_triplet_energy(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
        centroids: Dict[str, np.ndarray],
        margin: float = 0.1,
    ) -> float:
        """Sum of per-frame proxy-triplet energies."""
        total = 0.0
        for seg in segments:
            lab = labels[seg.label_key]
            em = np.asarray(seg.embeddings, dtype=np.float64)
            if em.size == 0:
                continue
            for i in range(em.shape[0]):
                total += self.compute_proxy_triplet_energy(em[i], lab, centroids, margin)
        return float(total)

    def _compute_segment_distances_to_centroids(
        self,
        segments: List[Segment],
        centroids: Dict[str, np.ndarray],
    ) -> Dict[LabelKey, Dict[str, float]]:
        """
        Precompute cosine distances from each segment's frames to all centroids.
        Returns: {segment.label_key: {label: avg_cosine_distance}}
        """
        seg_dists: Dict[LabelKey, Dict[str, float]] = {}
        for seg in segments:
            em = np.asarray(seg.embeddings, dtype=np.float64)
            if em.size == 0:
                seg_dists[seg.label_key] = {lab: 0.0 for lab in centroids}
                continue
            dists: Dict[str, float] = {}
            for lab, c in centroids.items():
                # Average cosine distance across all frames in this segment
                total_dist = sum(self.cosine_distance(em[i], c) for i in range(em.shape[0]))
                dists[lab] = total_dist / max(em.shape[0], 1)
            seg_dists[seg.label_key] = dists
        return seg_dists

    def _compute_segment_proxy_energy(
        self,
        seg: Segment,
        label: str,
        seg_dists: Dict[str, float],
        margin: float,
    ) -> float:
        """
        Compute proxy-triplet energy for one segment using precomputed distances.
        seg_dists: dict mapping label -> avg_cosine_distance for this segment.
        """
        em = np.asarray(seg.embeddings, dtype=np.float64)
        if em.size == 0:
            return 0.0
        d_p = seg_dists.get(label, 0.0)
        # Find nearest incorrect label
        d_n = float("inf")
        for lab, dist in seg_dists.items():
            if lab != label:
                d_n = min(d_n, dist)
        if not math.isfinite(d_n):
            d_n = 0.0
        # Energy per frame: multiply by frame count
        return float(max(d_p - d_n + float(margin), 0.0) * em.shape[0])

    def _update_centroids_for_labels(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
        centroids: Dict[str, np.ndarray],
        affected_labels: List[str],
    ) -> None:
        """
        Incrementally update centroids for only the affected labels.
        Modifies centroids dict in-place.
        """
        for label in affected_labels:
            # Collect all embeddings for this label
            label_embeddings = []
            for seg in segments:
                if labels.get(seg.label_key) == label:
                    em = np.asarray(seg.embeddings, dtype=np.float64)
                    if em.size > 0:
                        for i in range(em.shape[0]):
                            label_embeddings.append(em[i])
            
            if label_embeddings:
                # Compute spherical mean (same as compute_centroids)
                stacked = np.stack(label_embeddings, axis=0)
                raw_mean = np.mean(stacked, axis=0)
                norm = np.linalg.norm(raw_mean)
                if norm > 1e-12:
                    centroids[label] = raw_mean / norm
                else:
                    centroids[label] = raw_mean

    def _update_distances_for_labels(
        self,
        segments: List[Segment],
        centroids: Dict[str, np.ndarray],
        seg_dist_cache: Dict[LabelKey, Dict[str, float]],
        affected_labels: List[str],
    ) -> None:
        """
        Incrementally update distances from all segments to the affected label centroids.
        Modifies seg_dist_cache in-place.
        """
        for seg in segments:
            em = np.asarray(seg.embeddings, dtype=np.float64)
            if em.size == 0:
                continue
            for label in affected_labels:
                if label in centroids:
                    # Recompute average distance from this segment to this centroid
                    total_dist = sum(
                        self.cosine_distance(em[i], centroids[label])
                        for i in range(em.shape[0])
                    )
                    seg_dist_cache[seg.label_key][label] = total_dist / max(em.shape[0], 1)

    def _pair_proxy_triplet_reduction_fast(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
        seg1: Segment,
        seg2: Segment,
        label1: str,
        label2: str,
        centroids: Dict[str, np.ndarray],
        seg_dist_cache: Dict[LabelKey, Dict[str, float]],
        proxy_margin: float,
    ) -> float:
        """
        Compute energy change from swapping labels between seg1 and seg2.
        Accounts for how the centroid changes affect ALL segments with these labels.
        Positive if swap reduces energy.
        """
        # Find all segments with label1 or label2
        affected_segments = []
        for seg in segments:
            seg_label = labels.get(seg.label_key)
            if seg_label == label1 or seg_label == label2:
                affected_segments.append(seg)
        
        # Compute energy before swap for affected segments
        e_before = 0.0
        for seg in affected_segments:
            seg_label = labels[seg.label_key]
            e_before += self._compute_segment_proxy_energy(
                seg, seg_label, seg_dist_cache[seg.label_key], proxy_margin
            )
        
        # Simulate the swap and compute new centroids for affected labels
        trial_labels = dict(labels)
        trial_labels[seg1.label_key] = label2
        trial_labels[seg2.label_key] = label1
        
        # Recompute centroids for the two affected labels
        trial_centroids = dict(centroids)
        self._update_centroids_for_labels(
            segments, trial_labels, trial_centroids, [label1, label2]
        )
        
        # Recompute distances for affected segments to the new centroids
        trial_dist_cache = {}
        for seg in affected_segments:
            em = np.asarray(seg.embeddings, dtype=np.float64)
            if em.size == 0:
                trial_dist_cache[seg.label_key] = {lab: 0.0 for lab in trial_centroids}
                continue
            dists = {}
            for lab, c in trial_centroids.items():
                total_dist = sum(
                    self.cosine_distance(em[i], c) for i in range(em.shape[0])
                )
                dists[lab] = total_dist / max(em.shape[0], 1)
            trial_dist_cache[seg.label_key] = dists
        
        # Compute energy after swap for affected segments
        e_after = 0.0
        for seg in affected_segments:
            seg_label = trial_labels[seg.label_key]
            e_after += self._compute_segment_proxy_energy(
                seg, seg_label, trial_dist_cache[seg.label_key], proxy_margin
            )
        
        return float(e_before - e_after)

    # def refine_labels_proxy_triplet_sa(
    #     self,
    #     segments: List[Segment],
    #     labels: Dict[LabelKey, str],
    #     *,
    #     sa_iters: int,
    #     proxy_margin: float,
    #     verbose: bool = True,
    # ) -> Dict[LabelKey, str]:
    #     """
    #     Same swap structure as ``refine_labels``, but uses proxy-triplet energy
    #     (``_pair_proxy_triplet_reduction``) instead of cosine cost reduction.
    #     """
    #     labels = labels.copy()
    #     real_segments = [seg for seg in segments if not seg.is_placeholder]

    #     if verbose:
    #         n_ph = len(segments) - len(real_segments)
    #         if n_ph:
    #             print(
    #                 f"  [Proxy-Triplet SA] {n_ph} placeholder segment(s) excluded from swaps"
    #             )

    #     videos: Dict[str, List[Segment]] = {}
    #     for seg in real_segments:
    #         videos.setdefault(seg.video_id, []).append(seg)

    #     cool_divisor = max(self.bad_swap_cool_divisor, max(float(sa_iters), 1.0) / 4.0)
    #     min_bad_swap_prob = 0.05

    #     if verbose:
    #         print("\n" + "=" * 60)
    #         print("PROXY-TRIPLET SIMULATED ANNEALING (segment swaps)")
    #         print("=" * 60)
    #         print(f"Segments (real): {len(real_segments)}")
    #         print(f"Videos: {len(videos)}")
    #         print(f"SA iterations (epochs): {sa_iters}")
    #         print(f"Proxy margin: {proxy_margin}")
    #         if self.ilr_allow_cross_round_swaps:
    #             print(
    #                 "Swap pairs: same video, **cross-round allowed** "
    #                 "(labels must remain in each segment's candidate multiset)"
    #             )
    #         else:
    #             print("Swap pairs: same video, **same picklist round only**")

    #     centroids0 = self.compute_centroids(real_segments, labels)
    #     initial_energy = self.compute_total_proxy_triplet_energy(
    #         real_segments, labels, centroids0, proxy_margin
    #     )
    #     if verbose:
    #         print(f"Initial proxy-triplet energy: {initial_energy:.4f}")

    #     best_labels = labels.copy()
    #     best_energy = initial_energy

    #     # Initialize centroids and distance cache once
    #     centroids = self.compute_centroids(real_segments, labels)
    #     seg_dist_cache = self._compute_segment_distances_to_centroids(
    #         real_segments, centroids
    #     )

    #     for epoch in range(int(sa_iters)):
    #         # Process each video's segments for potential swaps
    #         for _video_id, video_segments in videos.items():
    #             if len(video_segments) < 2:
    #                 continue

    #             seg1 = video_segments[random.randrange(len(video_segments))]
    #             dr_by_key: Dict[LabelKey, float] = {}
    #             seg_by_key: Dict[LabelKey, Segment] = {}

    #             for seg2 in video_segments:
    #                 if seg2 is seg1:
    #                     continue
    #                 if labels[seg1.label_key] == labels[seg2.label_key]:
    #                     continue
    #                 label1 = labels[seg1.label_key]
    #                 label2 = labels[seg2.label_key]
    #                 if seg1.candidate_labels is None or seg2.candidate_labels is None:
    #                     continue
    #                 if label1 not in seg2.candidate_labels or label2 not in seg1.candidate_labels:
    #                     continue
    #                 if (
    #                     not self.ilr_allow_cross_round_swaps
    #                     and seg1.candidate_labels != seg2.candidate_labels
    #                 ):
    #                     continue
    #                 # Compute energy reduction for this swap candidate
    #                 # This accounts for all segments with the affected labels
    #                 dr = self._pair_proxy_triplet_reduction_fast(
    #                     real_segments, labels, seg1, seg2, label1, label2,
    #                     centroids, seg_dist_cache, proxy_margin
    #                 )
    #                 lk2 = seg2.label_key
    #                 dr_by_key[lk2] = dr
    #                 seg_by_key[lk2] = seg2

    #             if not dr_by_key:
    #                 continue

    #             keys = list(dr_by_key.keys())
    #             best_lk = keys[0]
    #             for lk2, dr in dr_by_key.items():
    #                 if dr > dr_by_key[best_lk]:
    #                     best_lk = lk2

    #             if dr_by_key[best_lk] <= 0:
    #                 reduction_sum = sum(dr_by_key.values())
    #                 if abs(reduction_sum) < 1e-12:
    #                     continue
    #                 # When all swaps are bad (negative), we want to pick the LEAST bad one
    #                 # Use absolute values so less-negative (better) swaps get higher probability
    #                 abs_drs = [abs(dr) for dr in dr_by_key.values()]
    #                 abs_sum = sum(abs_drs)
    #                 if abs_sum < 1e-12:
    #                     continue
    #                 # Higher absolute value = worse swap, so invert for probabilities
    #                 # The least bad swap (smallest absolute value) should have highest probability
    #                 inv_abs = [abs_sum - abs_dr for abs_dr in abs_drs]
    #                 inv_sum = sum(inv_abs)
    #                 if inv_sum < 1e-12:
    #                     # All swaps equally bad, pick uniformly
    #                     probs = np.ones(len(keys), dtype=np.float64) / len(keys)
    #                 else:
    #                     probs = np.array(
    #                         [math.exp(inv_val / inv_sum) for inv_val in inv_abs],
    #                         dtype=np.float64,
    #                     )
    #                     probs = probs / probs.sum()
    #                 pick_i = int(np.random.choice(len(keys), p=probs))
    #                 swap_lk = keys[pick_i]
    #                 swap_seg2 = seg_by_key[swap_lk]

    #                 p_do_swap = max(
    #                     min_bad_swap_prob,
    #                     math.exp(-epoch / cool_divisor),
    #                 )
    #                 to_swap = int(np.random.choice([0, 1], p=[1 - p_do_swap, p_do_swap]))
    #                 if to_swap == 0:
    #                     continue
    #             else:
    #                 positive = {lk: dr for lk, dr in dr_by_key.items() if dr > 0}
    #                 if not positive:
    #                     continue
    #                 pos_sum = sum(positive.values())
    #                 if pos_sum < 1e-12:
    #                     continue
    #                 pos_keys = list(positive.keys())
    #                 probs = np.array(
    #                     [math.exp(dr / pos_sum) for dr in positive.values()],
    #                     dtype=np.float64,
    #                 )
    #                 probs = probs / probs.sum()
    #                 pick_i = int(np.random.choice(len(pos_keys), p=probs))
    #                 swap_lk = pos_keys[pick_i]
    #                 swap_seg2 = seg_by_key[swap_lk]

    #             # Get labels before swap for incremental update
    #             label1 = labels[seg1.label_key]
    #             label2 = labels[swap_seg2.label_key]

    #             # Perform the swap
    #             lk1, lk2 = seg1.label_key, swap_seg2.label_key
    #             labels[lk1], labels[lk2] = labels[lk2], labels[lk1]

    #             # Immediately update affected centroids and distances after this swap
    #             affected_labels = [label1, label2]
    #             self._update_centroids_for_labels(
    #                 real_segments, labels, centroids, affected_labels
    #             )
    #             self._update_distances_for_labels(
    #                 real_segments, centroids, seg_dist_cache, affected_labels
    #             )

    #         # Compute final energy at end of epoch for tracking
    #         current_energy = self.compute_total_proxy_triplet_energy(
    #             real_segments, labels, centroids, proxy_margin
    #         )
    #         if current_energy < best_energy:
    #             best_energy = current_energy
    #             best_labels = labels.copy()

    #         if verbose and int(sa_iters) >= 50 and (epoch + 1) % max(1, int(sa_iters) // 10) == 0:
    #             print(f"SA epoch {epoch+1:4d}: proxy_energy={current_energy:.4f}")

    #     if verbose:
    #         print(f"\nFinal proxy-triplet energy: {best_energy:.4f} (from {initial_energy:.4f})")
    #         imp = (
    #             (initial_energy - best_energy) / initial_energy * 100
    #             if initial_energy > 0
    #             else 0.0
    #         )
    #         print(f"Improvement: {imp:.4f}%")

    #     return best_labels

    def refine_labels_proxy_triplet_sa(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
        *,
        sa_iters: int,
        proxy_margin: float,
        verbose: bool = True,
    ) -> Dict[LabelKey, str]:
        labels = labels.copy()
        real_segments = [seg for seg in segments if not seg.is_placeholder]
        
        # Pre-calculate centroids ONCE per epoch
        centroids = self.compute_centroids(real_segments, labels)
        
        # 1. Precompute Energy Matrix: Segment x Potential_Label
        # This makes the swap check O(1)
        def get_energy_map(segs, current_centroids):
            # map: {label_key: {label_string: energy_value}}
            emap = {}
            for s in segs:
                emap[s.label_key] = {}
                for l_str in current_centroids.keys():
                    emap[s.label_key][l_str] = self.compute_proxy_triplet_energy(
                        s.mean_embedding, l_str, current_centroids, proxy_margin
                    ) * s.embeddings.shape[0]
            return emap

        energy_map = get_energy_map(real_segments, centroids)
        
        videos: Dict[str, List[Segment]] = {}
        for seg in real_segments:
            videos.setdefault(seg.video_id, []).append(seg)

        cool_divisor = max(self.bad_swap_cool_divisor, float(sa_iters) / 4.0)

        for epoch in range(int(sa_iters)):
            # Refresh energy map and centroids every few epochs to prevent drift
            if epoch % 5 == 0 and epoch > 0:
                centroids = self.compute_centroids(real_segments, labels)
                energy_map = get_energy_map(real_segments, centroids)

            temp = self._get_temperature(epoch)

            for _video_id, video_segments in videos.items():
                if len(video_segments) < 2: continue
                
                # 1. THROUGHPUT FIX: Try every segment as an 'anchor'
                shuffled_indices = list(range(len(video_segments)))
                random.shuffle(shuffled_indices)
                
                for idx in shuffled_indices:
                    s1 = video_segments[idx]
                    # Select a random target segment to attempt a swap
                    s2 = random.choice(video_segments)
                    
                    if s1 is s2: continue
                    
                    lk1, lk2 = s1.label_key, s2.label_key
                    lab1, lab2 = labels[lk1], labels[lk2]
                    
                    if lab1 == lab2: continue
                    
                    # Constraint check
                    if s1.candidate_labels and lab2 not in s1.candidate_labels: continue
                    if s2.candidate_labels and lab1 not in s2.candidate_labels: continue
                    if not self.ilr_allow_cross_round_swaps and s1.candidate_labels != s2.candidate_labels:
                        continue

                    # 2. LOCAL DELTA (using precomputed energy map)
                    e_before = energy_map[lk1][lab1] + energy_map[lk2][lab2]
                    e_after  = energy_map[lk1][lab2] + energy_map[lk2][lab1]
                    delta = e_after - e_before

                    # 3. METROPOLIS ACCEPTANCE
                    # If delta < 0, it's a good move. If not, roll the dice based on Temp.
                    if delta < 0 or self._accept_swap(delta, temp):
                        labels[lk1], labels[lk2] = lab2, lab1
                    
            if verbose and (epoch + 1) % max(1, int(sa_iters) // 10) == 0:
                current_energy = sum(energy_map[seg.label_key][labels[seg.label_key]] for seg in real_segments)
                print(f"  SA Epoch {epoch+1}/{sa_iters} | Energy: {current_energy:.4f} | Temp: {temp:.3f}")

        return labels

    def _fit_iterated_model(
        self,
        all_segments: List[Segment],
        labels: Dict[LabelKey, str],
        clip_dim: int,
        *,
        hand_embeddings_dir: str,
        refinement_loops: int,
        sa_iters: int,
        adapter_epochs: int,
        adapter_lr: float,
        adapter_batch_size: int,
        triplet_margin: float,
        proxy_energy_margin: float,
        verbose: bool = True,
    ) -> Dict[LabelKey, str]:
        """
        Outer loop: neutralize → proxy-triplet SA → train CLIPAdapter (triplet) → apply adapter.
        """
        from .hand_neutralizer import HandNeutralizer
        from .clip_adapter import CLIPAdapter, apply_adapter_to_numpy, train_clip_adapter

        real_segments = [s for s in all_segments if not s.is_placeholder]
        labels = dict(labels)

        neutralizer = HandNeutralizer(hand_embeddings_dir, n_components=20, verbose=verbose)
        self.hand_neutralizer_state = neutralizer.state_dict()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        adapter: Optional[torch.nn.Module] = None

        if verbose:
            print("\n" + "=" * 60)
            print("ITERATED MODEL (neutralize → SA → adapter → update)")
            print("=" * 60)
            print(f"Refinement loops: {refinement_loops}")
            print(f"Hand embeddings dir: {hand_embeddings_dir}")

        refined = dict(labels)

        for loop_i in range(int(refinement_loops)):
            if verbose:
                print(f"\n--- Refinement loop {loop_i + 1}/{int(refinement_loops)} ---")

            # Step A: neutralize
            for seg in real_segments:
                em = np.asarray(seg.embeddings, dtype=np.float64)
                if em.size == 0:
                    continue
                seg.embeddings = neutralizer.neutralize(em)

            # Step B: SA with proxy-triplet energy
            refined = self.refine_labels_proxy_triplet_sa(
                all_segments,
                refined,
                sa_iters=int(sa_iters),
                proxy_margin=float(proxy_energy_margin),
                verbose=verbose,
            )

            # Build frame matrix for adapter training (real segments only)
            # Use subset sampling if too many frames to speed up training
            MAX_FRAMES_PER_CLASS = 100
            frames_by_class: Dict[int, List[np.ndarray]] = {}
            for seg in real_segments:
                em = np.asarray(seg.embeddings, dtype=np.float64)
                if em.size == 0:
                    continue
                lab = refined[seg.label_key]
                li = int(self.label_to_idx[lab])
                if li not in frames_by_class:
                    frames_by_class[li] = []
                for i in range(em.shape[0]):
                    frames_by_class[li].append(em[i])

            frames_list: List[np.ndarray] = []
            labs_list: List[int] = []
            total_before_subsample = sum(len(v) for v in frames_by_class.values())
            for li, class_frames in frames_by_class.items():
                if len(class_frames) > MAX_FRAMES_PER_CLASS:
                    # Random subsample
                    indices = np.random.choice(
                        len(class_frames), MAX_FRAMES_PER_CLASS, replace=False
                    )
                    class_frames = [class_frames[i] for i in indices]
                frames_list.extend(class_frames)
                labs_list.extend([li] * len(class_frames))

            if len(frames_list) < 2:
                if verbose:
                    print("  [Iterated model] Not enough frames for adapter training; stopping loop.")
                break
            X_frames = np.stack(frames_list, axis=0)
            y_frames = np.asarray(labs_list, dtype=np.int64)

            if verbose and total_before_subsample != len(frames_list):
                print(
                    f"  [Iterated model] Subsampled {len(frames_list)} frames from "
                    f"{total_before_subsample} for adapter training "
                    f"(max {MAX_FRAMES_PER_CLASS} per class)"
                )

            if adapter is None:
                adapter = CLIPAdapter(int(clip_dim)).to(device)

            # Step C: train adapter
            adapter, train_info = train_clip_adapter(
                X_frames,
                y_frames,
                embed_dim=int(clip_dim),
                epochs=int(adapter_epochs),
                batch_size=int(adapter_batch_size),
                lr=float(adapter_lr),
                triplet_margin=float(triplet_margin),
                device=device,
                verbose=verbose,
                model_in=adapter,
            )
            if verbose:
                print(
                    f"  [Iterated model] Adapter train loss (last epoch avg): "
                    f"{train_info.get('loss', 0.0):.6f}"
                )

            # Step D: update embeddings with torch.no_grad() for efficiency
            adapter.eval()
            with torch.no_grad():
                for seg in real_segments:
                    em = np.asarray(seg.embeddings, dtype=np.float64)
                    if em.size == 0:
                        continue
                    seg.embeddings = apply_adapter_to_numpy(adapter, em, device=device)
            adapter = adapter.cpu()

        if adapter is not None:
            adapter_cpu = adapter.cpu()
            self.clip_adapter = adapter_cpu
        return refined

    def evaluate_swap(
        self,
        segments: List[Segment],
        seg_a: Segment,
        seg_b: Segment,
        labels: Dict[LabelKey, str],
        _label_stats: Optional[Dict[str, Tuple[np.ndarray, int]]] = None,
    ) -> float:
        """
        Cosine-cost delta (new - old) if labels were swapped (used by
        ``weak_supervision_clip_exhaustive_swap``). Negative = improvement.
        Frame-level cost with LOO spherical centroids (``_pair_cosine_distance_reduction``).
        ``_label_stats`` is ignored (legacy signature).
        """
        reduction = self._pair_cosine_distance_reduction(
            segments, labels, seg_a, seg_b
        )
        return float(-reduction)

    def _get_temperature(self, epoch: int) -> float:
        if self.temp_decay == "exponential":
            return self.initial_temp * (self.decay_rate**epoch)
        if self.temp_decay == "linear":
            return self.initial_temp * (1 - epoch / self.ilr_epochs)
        return self.initial_temp * (self.decay_rate**epoch)

    def _accept_swap(self, delta_cost: float, temperature: float) -> bool:
        """Metropolis-style acceptance (CLIP exhaustive pairwise reference)."""
        if delta_cost < 0:
            return True
        if temperature <= 0:
            return False
        p_accept = np.exp(-delta_cost / max(temperature, 1e-10))
        return random.random() < p_accept

    def _loo_spherical_centroid(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
        label: str,
        exclude: Segment,
    ) -> Optional[np.ndarray]:
        """
        Spherical mean of all frames in segments with ``label``,
        leave-one-out: frames from ``exclude`` are not included.
        """
        mates = [
            s
            for s in segments
            if labels[s.label_key] == label and s.label_key != exclude.label_key
        ]
        if not mates:
            mates = [s for s in segments if labels[s.label_key] == label]
        if not mates:
            return None
        blocks: List[np.ndarray] = []
        for m in mates:
            em = np.asarray(m.embeddings, dtype=np.float64)
            if em.size:
                blocks.append(em)
        if not blocks:
            return None
        return spherical_mean(np.vstack(blocks))

    def _pair_cosine_distance_reduction(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
        seg1: Segment,
        seg2: Segment,
    ) -> float:
        """
        Positive if swapping labels between seg1 and seg2 would reduce the total
        frame-level cosine cost under LOO spherical centroids for the two current labels.
        """
        p1 = labels[seg1.label_key]
        p2 = labels[seg2.label_key]
        if p1 == p2:
            return 0.0

        c1_ex1 = self._loo_spherical_centroid(segments, labels, p1, seg1)
        c2_ex2 = self._loo_spherical_centroid(segments, labels, p2, seg2)
        if c1_ex1 is None or c2_ex2 is None:
            return 0.0

        curr = seg1.compute_frame_costs(c1_ex1, self.cosine_distance)
        curr += seg2.compute_frame_costs(c2_ex2, self.cosine_distance)
        new_cost = seg1.compute_frame_costs(c2_ex2, self.cosine_distance)
        new_cost += seg2.compute_frame_costs(c1_ex1, self.cosine_distance)
        return float(curr - new_cost)

    def refine_labels(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
        verbose: bool = True,
    ) -> Dict[LabelKey, str]:
        """
        ILR: random anchor segment per video per epoch, cosine-distance energy
        (summed over all frames: 1 - cos(f, c_label)), Metropolis-style
        bad-swap schedule.

        By default, swaps are only between segments in the **same picklist round**
        (identical ``candidate_labels``). With ``self.ilr_allow_cross_round_swaps``,
        segments from **different** rounds in the same video may swap if each
        post-swap label is still in that segment's multiset. Placeholders excluded.
        """
        labels = labels.copy()

        # Filter out placeholder segments from ILR
        real_segments = [seg for seg in segments if not seg.is_placeholder]
        
        if verbose:
            n_placeholders = len(segments) - len(real_segments)
            if n_placeholders:
                print(
                    f"  [ILR] {n_placeholders} placeholder segment(s) excluded from swap logic "
                    "(preserving their initial labels)"
                )

        videos: Dict[str, List[Segment]] = {}
        for seg in real_segments:
            videos.setdefault(seg.video_id, []).append(seg)

        # Keep exploration alive longer on high-epoch runs. The legacy fixed divisor
        # (50) cools too quickly and effectively disables bad-swap escapes early.
        cool_divisor = max(self.bad_swap_cool_divisor, self.ilr_epochs / 4.0)
        min_bad_swap_prob = 0.05

        if verbose:
            print("\n" + "=" * 60)
            print("ITERATIVE LABEL REFINEMENT (cosine distance, frame-level energy)")
            print("=" * 60)
            print(f"Segments (real): {len(real_segments)}")
            print(f"Videos: {len(videos)}")
            print(f"Epochs: {self.ilr_epochs}")
            print(
                "Bad-swap cooling: max("
                f"{min_bad_swap_prob:.2f}, exp(-epoch / {cool_divisor:.2f}))"
            )
            if self.ilr_allow_cross_round_swaps:
                print(
                    "ILR swap pairs: same video, **cross-round allowed** "
                    "(labels must remain in each segment's candidate multiset)"
                )
            else:
                print("ILR swap pairs: same video, **same picklist round only**")

        centroids = self.compute_centroids(real_segments, labels)
        initial_cost = self.compute_total_cosine_cost(real_segments, labels, centroids)

        if verbose:
            print(f"Initial cosine cost: {initial_cost:.4f}")

        best_labels = labels.copy()
        best_cost = initial_cost

        for epoch in range(self.ilr_epochs):
            temp = self._get_temperature(epoch)
            
            # 1. Update centroids and Energy Map ONCE per epoch
            centroids = self.compute_centroids(real_segments, labels)
            
            # Precompute cost for every segment if assigned to every possible label
            # emap[label_key][label_string] = sum(1 - cos(frame, centroid))
            energy_map = {}
            for s in real_segments:
                energy_map[s.label_key] = {}
                for l_str, c_vec in centroids.items():
                    energy_map[s.label_key][l_str] = s.compute_frame_costs(c_vec, self.cosine_distance)

            # 2. HIGH THROUGHPUT: Process all segments in random order
            all_real_keys = [s.label_key for s in real_segments]
            random.shuffle(all_real_keys)
            seg_obj_map = {s.label_key: s for s in real_segments}

            for lk1 in all_real_keys:
                seg1 = seg_obj_map[lk1]
                video_segments = videos[seg1.video_id]
                if len(video_segments) < 2: continue

                # Pick a random partner in the same video
                seg2 = random.choice(video_segments)
                if seg1 is seg2: continue

                lk2 = seg2.label_key
                lab1, lab2 = labels[lk1], labels[lk2]
                if lab1 == lab2: continue

                # Hard constraints check
                if seg1.candidate_labels and lab2 not in seg1.candidate_labels: continue
                if seg2.candidate_labels and lab1 not in seg2.candidate_labels: continue
                if not self.ilr_allow_cross_round_swaps and seg1.candidate_labels != seg2.candidate_labels:
                    continue

                # 3. O(1) ENERGY DELTA (using precomputed map)
                # Energy change if we swap: (new pairs) - (current pairs)
                e_before = energy_map[lk1][lab1] + energy_map[lk2][lab2]
                e_after  = energy_map[lk1][lab2] + energy_map[lk2][lab1]
                delta = e_after - e_before

                # 4. METROPOLIS ACCEPTANCE
                if delta < 0 or self._accept_swap(delta, temp):
                    labels[lk1], labels[lk2] = lab2, lab1

            # End-of-epoch tracking
            current_cost = sum(energy_map[lk][labels[lk]] for lk in all_real_keys)
            if current_cost < best_cost:
                best_cost = current_cost
                best_labels = labels.copy()

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:4d}: cosine_cost={current_cost:.4f} | temp={temp:.3f}")

        if verbose:
            print(f"\nFinal cosine cost: {best_cost:.4f} (from {initial_cost:.4f})")
            improvement = (
                (initial_cost - best_cost) / initial_cost * 100
                if initial_cost > 0
                else 0
            )
            print(f"Improvement: {improvement:.4f}%")

        return best_labels

    def fit(
        self,
        video_segments: Dict[str, Tuple[List[Segment], List[str]]],
        verbose: bool = True,
        skip_ilr: bool = False,
        initial_cluster_voting_csv: Optional[str] = None,
        use_cluster_voting: bool = False,
        *,
        use_iterated_model: bool = False,
        hand_embeddings_dir: Optional[str] = None,
        refinement_loops: int = 3,
        sa_iters: int = 100,
        adapter_epochs: int = 10,
        adapter_lr: float = 1e-3,
        adapter_batch_size: int = 32,
        triplet_margin: float = 0.1,
        proxy_energy_margin: float = 0.1,
    ) -> "WeakSupervisedTrainer":
        all_embeddings = []
        all_segments: List[Segment] = []

        for video_id, (segments, _) in video_segments.items():
            for seg in segments:
                # Exclude placeholder segments from frame stacking (they have zero embeddings)
                if not seg.is_placeholder:
                    all_embeddings.append(seg.embeddings)
                all_segments.append(seg)

        if len(all_embeddings) == 0:
            raise ValueError("No real segments provided for training (all are placeholders)")

        flat_embeddings = np.vstack(all_embeddings)
        clip_dim = int(flat_embeddings.shape[1])

        if verbose:
            n_placeholders = len(all_segments) - len(all_embeddings)
            print("\n" + "=" * 60)
            print("CLIP EMBEDDINGS (no PCA)")
            print("=" * 60)
            print(f"Total segments: {len(all_segments)}")
            if n_placeholders:
                print(f"  Real segments: {len(all_segments) - n_placeholders}")
                print(f"  Placeholder segments: {n_placeholders}")
            print(f"Total frames: {flat_embeddings.shape[0]}")
            print(f"CLIP embedding dimension: {clip_dim}")
            real_fc = np.array(
                [seg.embeddings.shape[0] for seg in all_segments if not seg.is_placeholder],
                dtype=np.int64,
            )
            if real_fc.size:
                print(
                    "Frames per real segment (each row → ILR / centroids): "
                    f"min={int(real_fc.min())}, median={float(np.median(real_fc)):.1f}, "
                    f"mean={float(real_fc.mean()):.2f}, max={int(real_fc.max())}"
                )
                by_vid: Dict[str, List[int]] = defaultdict(list)
                for seg in all_segments:
                    if seg.is_placeholder:
                        continue
                    by_vid[seg.video_id].append(int(seg.embeddings.shape[0]))
                print("Per-video (seg count, frame total, per-seg min/median/max frames):")
                for vid in sorted(by_vid.keys()):
                    vc = np.array(by_vid[vid], dtype=np.int64)
                    print(
                        f"  {vid}: {len(vc)} seg, {int(vc.sum())} frames, "
                        f"{int(vc.min())}/{float(np.median(vc)):.1f}/{int(vc.max())}"
                    )

        for seg in all_segments:
            if seg.is_placeholder:
                seg.embeddings = np.zeros((1, clip_dim), dtype=np.float64)

        labels: Dict[LabelKey, str] = {}
        unique_labels: set = set()

        for video_id, (segments, picklist) in video_segments.items():
            has_any = any(seg.candidate_labels is not None for seg in segments)
            has_all = all(seg.candidate_labels is not None for seg in segments)
            if has_any and not has_all:
                raise ValueError(
                    f"Video {video_id!r}: set candidate_labels on every carry segment, or on none"
                )
            if not has_all:
                if not picklist:
                    raise ValueError(f"Video {video_id!r}: provide picklist or segment.candidate_labels")
                for seg in segments:
                    seg.candidate_labels = tuple(picklist)
            for seg in segments:
                if seg.candidate_labels is not None:
                    unique_labels.update(seg.candidate_labels)
                else:
                    unique_labels.update(picklist)

        if use_cluster_voting:
            from .cluster_voting import cluster_based_initialization_with_details

            labels, label_confidence = cluster_based_initialization_with_details(
                all_segments, None, verbose=verbose
            )
            if initial_cluster_voting_csv:
                from .cluster_voting import write_initial_cluster_voting_matrix_csv

                write_initial_cluster_voting_matrix_csv(
                    initial_cluster_voting_csv, labels, label_confidence, video_segments
                )
                if verbose:
                    print(f"\nWrote initial cluster-voting matrix: {initial_cluster_voting_csv}")
        else:
            labels = self._initialize_labels_for_segments(all_segments, None)
            label_confidence = {lk: 0.0 for lk in labels}
            if verbose:
                print(
                    "\nInitial labels: random bijection per picklist multiset "
                    "(enable global cluster voting with --cluster-voting)"
                )

        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        if verbose:
            print(f"\nUnique labels: {list(self.label_to_idx.keys())}")
            print(
                "Initial label assignment complete "
                f"({'global cluster voting' if use_cluster_voting else 'random per multiset'})"
            )

        if skip_ilr:
            if verbose:
                print("\n--no-annealing: skipping ILR (using initial labels only)")
            refined_labels = dict(labels)
        elif use_iterated_model:
            if not hand_embeddings_dir:
                raise ValueError("use_iterated_model=True requires hand_embeddings_dir")
            refined_labels = self._fit_iterated_model(
                all_segments,
                labels,
                clip_dim,
                hand_embeddings_dir=str(hand_embeddings_dir),
                refinement_loops=int(refinement_loops),
                sa_iters=int(sa_iters),
                adapter_epochs=int(adapter_epochs),
                adapter_lr=float(adapter_lr),
                adapter_batch_size=int(adapter_batch_size),
                triplet_margin=float(triplet_margin),
                proxy_energy_margin=float(proxy_energy_margin),
                verbose=verbose,
            )
        else:
            refined_labels = self.refine_labels(all_segments, labels, verbose=verbose)

        self.last_refined_labels = dict(refined_labels)
        
        # Filter out placeholders for final centroid/std computation
        real_segments_final = [seg for seg in all_segments if not seg.is_placeholder]
        
        self._rebuild_label_video_means_from_segments(real_segments_final, refined_labels)
        self.centroid_stds = self.compute_centroid_stds(real_segments_final, refined_labels)

        if verbose:
            print("\n" + "=" * 60)
            print("FINAL CENTROIDS")
            print("=" * 60)
            for label, centroid in self.centroids.items():
                print(f"  {label}: dim={centroid.shape[0]}, norm={np.linalg.norm(centroid):.4f}")

        return self

    def fit_iterative(
        self,
        segment_embeddings: List[Union[np.ndarray, List[np.ndarray]]],
        pick_labels: List[str],
        beta: float,
        verbose: bool = False,
        video_id: Optional[str] = None,
        equal_video_weight: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Tuple[str, ...]]:
        """
        Single new picklist: choose label permutation minimizing weighted distance
        to existing centroids/stds, then either EWMA-update means (legacy) or
        replace this video's per-label spherical means and set each centroid to the
        spherical mean across videos (``equal_video_weight``). Std dictionaries are **not**
        updated (same as legacy HSV).

        **New labels**: If pick_labels contains SKUs not in the existing model,
        those centroids are initialized from the new video's segment means (no EWMA,
        spherical mean of L2-normalized segment vectors for that label).

        Args:
            segment_embeddings: One array per pick (raw CLIP dim ``(D,)`` or
                frame stack ``(T, D)``); internally reduced to per-segment mean, L2-normalized.
            pick_labels: Multiset of class names (order ignored). Can include new SKUs
                not previously trained.
            beta: Weight on **previous** centroid; ``(1 - beta)`` on the new mean
                vector for each assigned label. Ignored when ``equal_video_weight``
                is True. Only applies to existing labels; new labels are initialized
                without EWMA.
            video_id: Training video stem (basename without extension). Required when
                ``equal_video_weight`` is True.
            equal_video_weight: If True, store one spherical mean per (label, video)
                and set each centroid to the spherical mean across videos
                (so N videos ⇒ weight 1/N each). Re-embedding the same video replaces
                that video's row.

        Returns:
            ``(centroids, centroid_stds, chosen_permutation_tuple)``
        """
        if not self.centroids:
            raise ValueError("Call fit() before fit_iterative().")

        if len(segment_embeddings) != len(pick_labels):
            raise ValueError(
                f"segment_embeddings length ({len(segment_embeddings)}) "
                f"must match pick_labels ({len(pick_labels)})"
            )

        vecs_raw = []
        for emb in segment_embeddings:
            arr = np.asarray(emb, dtype=np.float64)
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            elif arr.ndim != 1:
                raise ValueError("Each segment embedding must be (D,) or (T, D)")
            vecs_raw.append(arr)

        flat = np.stack(vecs_raw, axis=0)
        transformed = self._l2_normalize(flat)

        if equal_video_weight:
            if not video_id:
                raise ValueError("equal_video_weight=True requires video_id (video stem).")

        classes = sorted(set(pick_labels))

        if equal_video_weight and not self.label_video_means:
            stems = list(getattr(self, "embedded_video_stems", []) or [])
            if len(stems) == 1:
                v = stems[0]
                self.label_video_means = {
                    lab: {v: np.asarray(vec, dtype=np.float64).copy()}
                    for lab, vec in self.centroids.items()
                }
            elif len(stems) == 0:
                raise ValueError(
                    "Equal video weight needs label_video_means.json or a model with exactly "
                    "one embedded_video_stems entry to migrate. Re-run training with current code "
                    "or replace the model directory."
                )
            else:
                raise ValueError(
                    "Equal video weight needs label_video_means.json from training with this "
                    "version, or a single-video model to bootstrap from. Re-run initial "
                    "training, or remove the model directory and start over."
                )

        # Identify new labels not in existing centroids
        new_labels = [c for c in classes if c not in self.centroids]

        if new_labels:
            # Initialize new centroids and stds from the new video's segments
            # Map pick_labels indices to new label centroids
            new_label_vecs: Dict[str, List[np.ndarray]] = {lab: [] for lab in new_labels}
            for i, lab in enumerate(pick_labels):
                if lab in new_labels:
                    arr = np.asarray(segment_embeddings[i], dtype=np.float64)
                    if arr.ndim == 2:
                        arr = arr.mean(axis=0)
                    new_label_vecs[lab].append(arr)
            
            # Spherical mean in CLIP space; stds for legacy persistence only
            for lab in new_labels:
                if new_label_vecs[lab]:
                    stacked = np.stack(new_label_vecs[lab], axis=0)
                    transformed_new = self._l2_normalize(stacked)
                    self.centroids[lab] = spherical_mean(transformed_new)
                    if transformed_new.shape[0] > 1:
                        self.centroid_stds[lab] = np.maximum(
                            transformed_new.std(axis=0, ddof=1),
                            self.variance_eps
                        )
                    else:
                        self.centroid_stds[lab] = np.ones(transformed_new.shape[1], dtype=np.float64)
                    # Update label mappings
                    if lab not in self.label_to_idx:
                        new_idx = max(self.label_to_idx.values()) + 1
                        self.label_to_idx[lab] = new_idx
                        self.idx_to_label[new_idx] = lab

        # Rebuild vector_distances for ALL classes (including newly initialized ones)
        vector_distances_all_picks: List[Dict[str, float]] = []
        for vec in transformed:
            vd: Dict[str, float] = {}
            for key in classes:
                mean_k = self.centroids[key]
                vd[key] = float(self.cosine_distance(vec, mean_k))
            vector_distances_all_picks.append(vd)

        distinct_perms = set(itertools.permutations(pick_labels))
        smallest: Tuple[Tuple[str, ...], float] = (tuple(), float("inf"))

        for perm in distinct_perms:
            total_distance = 0.0
            for i, lab in enumerate(perm):
                total_distance += vector_distances_all_picks[i][lab]
            if total_distance < smallest[1]:
                smallest = (perm, total_distance)

        chosen_perm, total_d = smallest
        if verbose:
            print(f"fit_iterative: best permutation total distance={total_d:.4f}")

        if equal_video_weight:
            idx_by_label: Dict[str, List[int]] = defaultdict(list)
            for i, lab in enumerate(chosen_perm):
                idx_by_label[lab].append(i)
            for lab, idxs in idx_by_label.items():
                vecs = transformed[idxs]
                self.label_video_means.setdefault(lab, {})[video_id] = spherical_mean(vecs)
            self._sync_centroids_from_label_video_means()
        else:
            for i, lab in enumerate(chosen_perm):
                vec_u = transformed[i]
                old = self.centroids[lab]
                blended = beta * old + (1.0 - beta) * vec_u
                self.centroids[lab] = self._l2_normalize(blended.reshape(1, -1))[0]

        return self.centroids, self.centroid_stds, chosen_perm

    def predict(self, embedding: np.ndarray) -> str:
        if not self.centroids:
            raise ValueError("Model not fitted. Call fit() first.")

        transformed = self._l2_normalize(embedding.reshape(1, -1))[0]

        best_label = None
        best_distance = float("inf")

        for label, centroid in self.centroids.items():
            dist = self.cosine_distance(transformed, centroid)
            if dist < best_distance:
                best_distance = dist
                best_label = label

        return best_label

    def predict_proba(self, embedding: np.ndarray) -> Dict[str, float]:
        if not self.centroids:
            raise ValueError("Model not fitted. Call fit() first.")

        transformed = self._l2_normalize(embedding.reshape(1, -1))[0]

        distances = {}
        for label, centroid in self.centroids.items():
            distances[label] = self.cosine_distance(transformed, centroid)

        neg_distances = {label: -d for label, d in distances.items()}
        max_neg = max(neg_distances.values())
        exp_scores = {label: np.exp(nd - max_neg) for label, nd in neg_distances.items()}
        total = sum(exp_scores.values())

        return {label: score / total for label, score in exp_scores.items()}


__all__ = ["Segment", "WeakSupervisedTrainer", "LabelKey", "EMPTY_HAND_LABEL", "spherical_mean"]
