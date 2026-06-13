"""
Cluster-based initial labeling (constrained picklist assignment).

1. **Spherical K-means on all cached frames** (real segments only) with
   ``k = |distinct SKUs|``; each segment is assigned the cluster that wins a
   **majority vote** among its frame cluster ids. If total frames
   ``< n_clusters``, falls back to K-means on **segment means**. Centroids
   are still only reference geometry for the next steps.

2. **Greedy cluster → global SKU** on real segments only: confidence-based
   matching locks which K-means centroid represents which SKU. This defines
   reference vectors for step 3 only.

3. **Strict final assignment (Hungarian)**: for each ``(video_id, round
   multiset)`` group, **every** segment in that round (real and placeholder) is
   matched to **exactly one multiset slot** and each slot to **exactly one**
   segment (``len(segments_in_group) == len(multiset)`` required). Cost is the
   **sum over frames** ``sum_f (1 - cos(f, ref))`` to each reference (greedy-matched
   K-means centroid, or pooled spherical mean over **all frames** from qualifying
   segments).

Output labels come **only** from step 3 (or from random bijection fallbacks when
K-means cannot be run). Placeholders participate in step 3 like any other segment.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from .weak_supervision import LabelKey, Segment, spherical_mean

ClusterAssignmentSteps = List[Tuple[int, int, str, float]]


def _cosine_distance_matrix(X: np.ndarray, R: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Pairwise cosine distance between rows of X and R (each m x d and n x d)."""
    xn = np.linalg.norm(X, axis=1, keepdims=True)
    xn = np.maximum(xn, eps)
    rn = np.linalg.norm(R, axis=1, keepdims=True)
    rn = np.maximum(rn, eps)
    Xu = X / xn
    Ru = R / rn
    sim = Xu @ Ru.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return (1.0 - sim).astype(np.float64)


def _segment_total_cosine_distance_to_ref(
    seg: Segment,
    ref: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Sum of per-frame cosine distances ``1 - cos(f, ref)`` to a (possibly
    unnormalized) reference direction. Matches ILR frame-level energy geometry.
    """
    em = np.asarray(seg.embeddings, dtype=np.float64)
    if em.size == 0:
        return 1e6
    r = np.asarray(ref, dtype=np.float64).reshape(-1)
    rn = float(np.linalg.norm(r))
    if rn < eps:
        return 1e6
    r_u = r / rn
    total = 0.0
    for i in range(em.shape[0]):
        fn = float(np.linalg.norm(em[i]))
        if fn < eps:
            total += 1.0
        else:
            f_u = em[i] / fn
            cos_sim = float(np.clip(np.dot(f_u, r_u), -1.0, 1.0))
            total += 1.0 - cos_sim
    return float(total)


def _candidate_multiset_tuple(seg: Segment, flat_picklist: Optional[List[str]]) -> Tuple[str, ...]:
    if seg.candidate_labels is not None:
        return seg.candidate_labels
    if flat_picklist is None:
        raise ValueError(
            "Segment missing candidate_labels and no flat picklist was provided for this video"
        )
    return tuple(flat_picklist)


def _random_bijection_per_groups(
    segments: List[Segment],
    flat_picklist: Optional[List[str]],
) -> Dict[LabelKey, str]:
    """Legacy: one random shuffle per (video_id, multiset) group."""
    labels: Dict[LabelKey, str] = {}
    groups: Dict[Tuple[str, Tuple[str, ...]], List[Segment]] = defaultdict(list)
    for seg in segments:
        tup = _candidate_multiset_tuple(seg, flat_picklist)
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


def _voting_is_degenerate(segments: List[Segment]) -> bool:
    """If every segment shares the same multiset, per-cluster vote is uniform."""
    tuples = [seg.candidate_labels for seg in segments if seg.candidate_labels is not None]
    if not tuples:
        return True
    first = tuples[0]
    return all(t == first for t in tuples)


def _unique_items_union(segments: List[Segment], flat_picklist: Optional[List[str]]) -> List[str]:
    items: Set[str] = set()
    for seg in segments:
        tup = _candidate_multiset_tuple(seg, flat_picklist)
        items.update(tup)
    return sorted(items)


def _cluster_segments_kmeans_segment_means(
    segments: List[Segment],
    n_clusters: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy: spherical K-means on one row per segment (segment spherical mean)."""
    X_raw = np.stack([seg.mean_embedding.astype(np.float64) for seg in segments], axis=0)
    if X_raw.shape[0] < n_clusters:
        raise ValueError(f"KMeans: n_samples={X_raw.shape[0]} < n_clusters={n_clusters}")
    rn = np.linalg.norm(X_raw, axis=1, keepdims=True)
    rn = np.maximum(rn, 1e-12)
    X = X_raw / rn
    km = KMeans(
        n_clusters=n_clusters,
        n_init="auto",
        random_state=random_state,
    )
    km.fit(X)
    assignments = km.labels_.astype(np.int32)
    centers = km.cluster_centers_.astype(np.float64)
    cn = np.linalg.norm(centers, axis=1, keepdims=True)
    cn = np.maximum(cn, 1e-12)
    centers = (centers / cn).astype(np.float64)
    k = n_clusters
    dim_std = np.zeros(k, dtype=np.float64)
    for c in range(k):
        mask = assignments == c
        if np.sum(mask) <= 1:
            dim_std[c] = 0.0
        else:
            dim_std[c] = float(np.mean(np.std(X[mask], axis=0)))
    return assignments, centers, dim_std


def _cluster_segments_kmeans(
    segments: List[Segment],
    n_clusters: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Frame-level spherical K-means: pool all frames, fit K-means on unit-normalized
    rows; assign each segment to the cluster id that occurs most among its frames.

    If total cached frames ``< n_clusters``, falls back to ``_cluster_segments_kmeans_segment_means``.

    Returns:
        assignments: cluster id per segment (majority of frame labels)
        cluster_centers: (k, dim) unit-norm rows
        cluster_mean_std: (k,) mean within-cluster std over dims on normalized frame rows
    """
    all_frames: List[np.ndarray] = []
    frame_to_seg: List[int] = []
    for seg_idx, seg in enumerate(segments):
        em = np.asarray(seg.embeddings, dtype=np.float64)
        if em.size == 0:
            continue
        for row in range(em.shape[0]):
            all_frames.append(em[row].astype(np.float64, copy=False))
            frame_to_seg.append(seg_idx)

    n_frames = len(all_frames)
    if n_frames < n_clusters:
        return _cluster_segments_kmeans_segment_means(segments, n_clusters, random_state)

    X_raw = np.stack(all_frames, axis=0)
    rn = np.linalg.norm(X_raw, axis=1, keepdims=True)
    rn = np.maximum(rn, 1e-12)
    X = X_raw / rn
    km = KMeans(
        n_clusters=n_clusters,
        n_init="auto",
        random_state=random_state,
    )
    km.fit(X)
    frame_cluster_labels = km.labels_.astype(np.int32)

    centers = km.cluster_centers_.astype(np.float64)
    cn = np.linalg.norm(centers, axis=1, keepdims=True)
    cn = np.maximum(cn, 1e-12)
    centers = (centers / cn).astype(np.float64)

    n_seg = len(segments)
    segment_assignments = np.zeros(n_seg, dtype=np.int32)
    for seg_idx in range(n_seg):
        idxs = [i for i, s in enumerate(frame_to_seg) if s == seg_idx]
        if not idxs:
            segment_assignments[seg_idx] = 0
            continue
        labs = frame_cluster_labels[np.asarray(idxs, dtype=np.int64)]
        segment_assignments[seg_idx] = int(np.bincount(labs, minlength=n_clusters).argmax())

    k = n_clusters
    dim_std = np.zeros(k, dtype=np.float64)
    for c in range(k):
        mask = frame_cluster_labels == c
        if np.sum(mask) <= 1:
            dim_std[c] = 0.0
        else:
            dim_std[c] = float(np.mean(np.std(X[mask], axis=0)))

    return segment_assignments, centers, dim_std


def _segments_by_cluster(
    segments: List[Segment],
    assignments: np.ndarray,
) -> Dict[int, List[Segment]]:
    out: Dict[int, List[Segment]] = defaultdict(list)
    for seg, c in zip(segments, assignments):
        out[int(c)].append(seg)
    return dict(out)


def _confidence_matrix(
    by_cluster: Dict[int, List[Segment]],
    cluster_ids: Set[int],
    item_ids: Set[str],
    flat_picklist: Optional[List[str]],
) -> Dict[int, Dict[str, float]]:
    """
    For each cluster c and item i: count segments in c with i in candidate multiset.
    Row-normalize over ``item_ids``. Uses ``_candidate_multiset_tuple`` so vote
    counts match segments that only have ``flat_picklist`` fallback.
    """
    weights: Dict[int, Dict[str, float]] = {}
    for c in cluster_ids:
        segs = by_cluster.get(c, [])
        raw: Dict[str, float] = {i: 0.0 for i in item_ids}
        for seg in segs:
            tup = _candidate_multiset_tuple(seg, flat_picklist)
            for lab in set(tup):
                if lab in item_ids:
                    raw[lab] += 1.0
        s = sum(raw[i] for i in item_ids)
        if s <= 0:
            n = max(len(item_ids), 1)
            weights[c] = {i: 1.0 / n for i in item_ids}
        else:
            weights[c] = {i: raw[i] / s for i in item_ids}
    return weights


def _greedy_cluster_item_matching(
    by_cluster: Dict[int, List[Segment]],
    all_cluster_ids: Set[int],
    all_items: Set[str],
    flat_picklist: Optional[List[str]],
) -> Tuple[Dict[int, str], ClusterAssignmentSteps]:
    """Sequential max-confidence matching of K-means clusters to unique global SKUs."""
    unassigned_c = set(all_cluster_ids)
    unassigned_i = set(all_items)
    cluster_to_item: Dict[int, str] = {}
    steps: ClusterAssignmentSteps = []
    step_idx = 0

    while unassigned_c and unassigned_i:
        w = _confidence_matrix(by_cluster, unassigned_c, unassigned_i, flat_picklist)
        best_c, best_i, best_v = None, None, -1.0
        for c in sorted(unassigned_c):
            row = w[c]
            for i in sorted(unassigned_i):
                v = row.get(i, 0.0)
                if v > best_v or (v == best_v and (best_c is None or c < best_c)):
                    best_v, best_c, best_i = v, c, i
        if best_c is None or best_i is None:
            break
        cluster_to_item[best_c] = best_i
        step_idx += 1
        steps.append((step_idx, best_c, best_i, float(best_v)))
        unassigned_c.remove(best_c)
        unassigned_i.remove(best_i)

    for c in sorted(unassigned_c):
        if unassigned_i:
            pick = sorted(unassigned_i)[0]
            cluster_to_item[c] = pick
            unassigned_i.remove(pick)
            step_idx += 1
            steps.append((step_idx, c, pick, 0.0))
        elif all_items:
            lab = sorted(all_items)[0]
            cluster_to_item[c] = lab
            step_idx += 1
            steps.append((step_idx, c, lab, 0.0))

    return cluster_to_item, steps


def _item_to_cluster_map(cluster_to_item: Dict[int, str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c, lab in cluster_to_item.items():
        out[lab] = int(c)
    return out


def _segment_labels_from_cluster_assignment(
    real_segments: List[Segment],
    assignments: np.ndarray,
    cluster_to_item: Dict[int, str],
    flat_picklist: Optional[List[str]],
    cluster_centers: np.ndarray,
    item_to_cluster: Dict[str, int],
) -> Tuple[Dict[LabelKey, str], Dict[LabelKey, float]]:
    """
    After greedy cluster→SKU: each segment inherits the SKU matched to its
    K-means cluster. If that SKU is not in the segment's candidate multiset,
    pick the multiset label whose greedy-matched centroid has lowest **total**
    frame cosine distance (same geometry as the Hungarian refinement step).
    """
    labels: Dict[LabelKey, str] = {}
    conf: Dict[LabelKey, float] = {}
    for idx, seg in enumerate(real_segments):
        c = int(assignments[idx])
        lab = cluster_to_item.get(c)
        cands_tup = _candidate_multiset_tuple(seg, flat_picklist)
        cands = list(dict.fromkeys(cands_tup))

        if lab is not None and lab in cands_tup:
            chosen = lab
        else:
            best_lab: Optional[str] = None
            best_d = float("inf")
            for j in cands:
                cid = item_to_cluster.get(j)
                if cid is None or not (0 <= cid < cluster_centers.shape[0]):
                    continue
                ref = cluster_centers[cid].astype(np.float64)
                d = _segment_total_cosine_distance_to_ref(seg, ref)
                if d < best_d:
                    best_d, best_lab = d, j
            chosen = best_lab if best_lab is not None else (cands[0] if cands else "unknown")

        if chosen == "unknown":
            conf[seg.label_key] = 0.0
        else:
            cid = item_to_cluster.get(chosen)
            if cid is not None and 0 <= cid < cluster_centers.shape[0]:
                ref = cluster_centers[cid].astype(np.float64)
                total_d = _segment_total_cosine_distance_to_ref(seg, ref)
                em = np.asarray(seg.embeddings, dtype=np.float64)
                n_fr = em.shape[0]
                if n_fr <= 0:
                    conf[seg.label_key] = 0.0
                else:
                    avg_d = total_d / float(n_fr)
                    conf[seg.label_key] = float(max(0.0, min(1.0, 1.0 - avg_d)))
            else:
                conf[seg.label_key] = 0.5
        labels[seg.label_key] = chosen

    return labels, conf


def _pooled_mean_for_item(
    item: str,
    segments: List[Segment],
    flat_picklist: Optional[List[str]],
) -> np.ndarray:
    """Spherical mean over **all frames** from segments whose multiset contains ``item``."""
    rows: List[np.ndarray] = []
    for seg in segments:
        tup = _candidate_multiset_tuple(seg, flat_picklist)
        if item not in tup:
            continue
        em = np.asarray(seg.embeddings, dtype=np.float64)
        if em.size == 0:
            continue
        for i in range(em.shape[0]):
            rows.append(em[i].astype(np.float64, copy=False))
    if not rows:
        raise ValueError(f"No segments contain item {item!r} in candidate_labels")
    return spherical_mean(np.stack(rows, axis=0))


def _assign_picklist_rounds_hungarian(
    segments: List[Segment],
    flat_picklist: Optional[List[str]],
    cluster_centers: np.ndarray,
    item_to_cluster: Dict[str, int],
    *,
    pooled_reference_segments: List[Segment],
    use_pooled_fallback: bool = True,
) -> Tuple[Dict[LabelKey, str], Dict[LabelKey, float]]:
    """
    Final strict labeling: each ``(video_id, multiset)`` group must have exactly
    as many segments as multiset entries. Hungarian assigns a **bijection**
    (minimum total distance) from segments to slots; output is the only
    supervision from cluster voting for these segments.

    Cost for segment ↔ slot is ``sum_f (1 - cos(f, ref))`` over frames (matches ILR).
    """
    groups: Dict[Tuple[str, Tuple[str, ...]], List[Segment]] = defaultdict(list)
    for seg in segments:
        tup = _candidate_multiset_tuple(seg, flat_picklist)
        groups[(seg.video_id, tup)].append(seg)

    labels: Dict[LabelKey, str] = {}
    conf: Dict[LabelKey, float] = {}

    for (_, multiset), segs in groups.items():
        segs_sorted = sorted(segs, key=lambda s: s.segment_id)
        slots = list(multiset)
        m = len(segs_sorted)
        n_slots = len(slots)
        if m != n_slots:
            vid = segs_sorted[0].video_id if segs_sorted else "?"
            raise ValueError(
                f"Strict picklist round requires len(segments)==len(multiset): "
                f"{m} segments vs {n_slots} multiset slots for video {vid!r}, multiset={multiset!r}"
            )

        ref_cols: List[np.ndarray] = []
        for lab in slots:
            cid = item_to_cluster.get(lab)
            if cid is not None and 0 <= cid < cluster_centers.shape[0]:
                ref_cols.append(cluster_centers[cid].astype(np.float64))
            elif use_pooled_fallback:
                ref_cols.append(
                    _pooled_mean_for_item(lab, pooled_reference_segments, flat_picklist)
                )
            else:
                raise KeyError(f"No cluster for item {lab!r}")

        R = np.stack(ref_cols, axis=0).astype(np.float64)
        for i in range(R.shape[0]):
            ni = float(np.linalg.norm(R[i]))
            if ni < 1e-12:
                v = np.zeros(R.shape[1], dtype=np.float64)
                v[0] = 1.0
                R[i] = v
            else:
                R[i] = R[i] / ni

        cost = np.zeros((m, n_slots), dtype=np.float64)
        for seg_idx in range(m):
            seg = segs_sorted[seg_idx]
            for slot_idx in range(n_slots):
                cost[seg_idx, slot_idx] = _segment_total_cosine_distance_to_ref(
                    seg, R[slot_idx]
                )

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            seg = segs_sorted[int(r)]
            lab = slots[int(c)]
            total_d = float(cost[int(r), int(c)])
            labels[seg.label_key] = lab
            em = np.asarray(seg.embeddings, dtype=np.float64)
            n_fr = em.shape[0]
            if n_fr <= 0:
                conf[seg.label_key] = 0.0
            else:
                avg_d = total_d / float(n_fr)
                conf[seg.label_key] = float(max(0.0, min(1.0, 1.0 - avg_d)))

    return labels, conf


def cluster_based_initialization(
    segments: List[Segment],
    flat_picklist: Optional[List[str]],
    *,
    verbose: bool = False,
    random_state: int = 42,
) -> Dict[LabelKey, str]:
    labels, _ = cluster_based_initialization_with_details(
        segments,
        flat_picklist,
        verbose=verbose,
        random_state=random_state,
    )
    return labels


def cluster_based_initialization_with_details(
    segments: List[Segment],
    flat_picklist: Optional[List[str]],
    *,
    verbose: bool = False,
    random_state: int = 42,
) -> Tuple[Dict[LabelKey, str], Dict[LabelKey, float]]:
    """
    K-means + greedy cluster→SKU on **real** segments define reference centroids only.
    **Final** labels: strict Hungarian bijection per ``(video_id, multiset)`` over
    **all** segments in each round (including placeholders).
    """
    if not segments:
        return {}, {}

    real_segments = [seg for seg in segments if not seg.is_placeholder]
    placeholder_segments = [seg for seg in segments if seg.is_placeholder]

    if verbose and placeholder_segments:
        print(
            f"  [cluster voting] {len(placeholder_segments)} placeholder segment(s): "
            "excluded from K-means/greedy; included in strict per-round Hungarian"
        )

    if not real_segments:
        if placeholder_segments:
            labels = _random_bijection_per_groups(segments, flat_picklist)
            return labels, {k: 0.0 for k in labels}
        return {}, {}

    if _voting_is_degenerate(real_segments):
        if verbose:
            print("  [cluster voting] degenerate multisets — using random bijection per group")
        labels = _random_bijection_per_groups(segments, flat_picklist)
        return labels, {k: 0.0 for k in labels}

    U = _unique_items_union(real_segments, flat_picklist)
    n_seg = len(real_segments)
    n_items = len(U)
    if n_items == 0:
        labels = _random_bijection_per_groups(segments, flat_picklist)
        return labels, {k: 0.0 for k in labels}

    if n_items > n_seg:
        if verbose:
            print(
                f"  [cluster voting] |unique items|={n_items} > n_real_segments={n_seg} — "
                "using random bijection per group (all segments)"
            )
        labels = _random_bijection_per_groups(segments, flat_picklist)
        return labels, {k: 0.0 for k in labels}

    k = n_items
    assignments, centers, cluster_std = _cluster_segments_kmeans(
        real_segments, k, random_state=random_state
    )
    by_cluster = _segments_by_cluster(real_segments, assignments)
    all_c = set(range(k))
    all_i = set(U)
    cluster_to_item, steps = _greedy_cluster_item_matching(by_cluster, all_c, all_i, flat_picklist)
    item_to_cluster = _item_to_cluster_map(cluster_to_item)

    vote_labels, _vote_conf = _segment_labels_from_cluster_assignment(
        real_segments,
        assignments,
        cluster_to_item,
        flat_picklist,
        centers,
        item_to_cluster,
    )

    labels, label_confidence = _assign_picklist_rounds_hungarian(
        segments,
        flat_picklist,
        centers,
        item_to_cluster,
        pooled_reference_segments=real_segments,
        use_pooled_fallback=True,
    )

    if verbose:
        n_vid = len({seg.video_id for seg in real_segments})
        print(
            f"  [cluster voting] global KMeans k={k} (frames pooled; segment means if frames < k), "
            f"n_real_segments={n_seg}, n_videos={n_vid}"
        )
        print("  [cluster voting] per-cluster mean std (within-cluster, mean over dims):")
        for c in range(k):
            lab = cluster_to_item.get(c, "?")
            print(f"    cluster {c} -> item {lab!r}, mean_dim_std={cluster_std[c]:.6f}")
        print("  [cluster voting] greedy cluster→item order (cluster -> label @ confidence):")
        for step_idx, c, lab, conf in steps:
            print(f"    step {step_idx:02d}: cluster {c} -> {lab} @ {conf:.4f}")
        n_refin = sum(
            1
            for seg in real_segments
            if vote_labels.get(seg.label_key) != labels.get(seg.label_key)
        )
        print(
            f"  [cluster voting] K-means vote vs final Hungarian (real only): "
            f"{n_refin}/{n_seg} segment label(s) differ"
        )
        print(
            "  [cluster voting] final output: strict Hungarian per round "
            "(each multiset slot ↔ one segment, all segments in round)"
        )
        print("  [cluster voting] final cluster-to-label mapping (reference centroids):")
        for c in sorted(cluster_to_item.keys()):
            lab = cluster_to_item[c]
            print(f"    cluster {c} -> {lab}")

    if verbose:
        n_all = len(segments)
        n_bad = sum(
            1
            for seg in segments
            if seg.candidate_labels is not None
            and labels[seg.label_key] not in seg.candidate_labels
        )
        if n_bad:
            print(
                f"  [cluster voting] WARNING: {n_bad}/{n_all} segment(s) have label outside multiset "
                "(unexpected after Hungarian)"
            )

    return labels, label_confidence


def write_initial_cluster_voting_matrix_csv(
    output_path: str,
    labels: Dict[LabelKey, str],
    label_confidence: Dict[LabelKey, float],
    video_segments: Dict[str, Tuple[List[Segment], Any]],
) -> None:
    """
    Write a CSV matrix: rows are segment order (1..N within each video, sorted by
    ``segment_id``); columns are two per video (label + confidence). Shorter videos
    leave trailing blanks.
    """
    video_ids = sorted(video_segments.keys())
    per_video_ordered: Dict[str, List[Segment]] = {}
    max_rows = 0
    for vid in video_ids:
        segs, _ = video_segments[vid]
        ordered = sorted(segs, key=lambda s: s.segment_id)
        per_video_ordered[vid] = ordered
        max_rows = max(max_rows, len(ordered))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header: List[str] = ["segment_#"]
        for vid in video_ids:
            header.append(f"{vid}_label")
            header.append(f"{vid}_confidence")
        w.writerow(header)
        for row in range(max_rows):
            out_row: List[Any] = [row + 1]
            for vid in video_ids:
                segs = per_video_ordered[vid]
                if row < len(segs):
                    key = segs[row].label_key
                    out_row.append(labels[key])
                    out_row.append(f"{label_confidence.get(key, 0.0):.4f}")
                else:
                    out_row.append("")
                    out_row.append("")
            w.writerow(out_row)


__all__ = [
    "cluster_based_initialization",
    "cluster_based_initialization_with_details",
    "write_initial_cluster_voting_matrix_csv",
]
