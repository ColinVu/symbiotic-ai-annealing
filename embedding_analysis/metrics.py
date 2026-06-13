"""Within-segment and cross-segment (same item) cosine analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .geometry import (
    cosine_similarity_matrix,
    l2_normalize_rows,
    pairwise_cosine_off_diagonal,
    spherical_mean,
)
from .segments import SegmentEmbeddings, middle_index_sorted


@dataclass
class WithinSegmentStats:
    segment_idx: int
    true_label: str
    num_frames: int
    mean_pairwise_cos: Optional[float]
    min_pairwise_cos: Optional[float]
    max_pairwise_cos: Optional[float]
    std_pairwise_cos: Optional[float]
    cos_middle_to_centroid: Optional[float]
    mean_cos_to_centroid: Optional[float]
    # thirds: mean cos to centroid for first/mid/last third of frames (by time order)
    mean_cos_begin_third: Optional[float]
    mean_cos_mid_third: Optional[float]
    mean_cos_end_third: Optional[float]


@dataclass
class CrossSegmentSameItem:
    true_label: str
    num_segments: int
    middle_frame_pairwise_cos_mean: float
    middle_frame_pairwise_cos_min: float
    middle_frame_pairwise_cos_max: float
    # Pearson correlation of raw embedding components (per pair), summarized
    middle_embedding_pearson_mean: float


@dataclass
class SameItemSegmentMatrix:
    """
    Spherical-mean per segment, then pairwise cos sim between segment means
    (rows/cols = one segment per row; global runs use one row per (video, segment) pair).
    """

    true_label: str
    matrix: "np.ndarray"  # (n, n) float
    row_labels: List[str]
    video_stem_per_row: List[str]
    segment_idx_per_row: List[int]


def _centroid(emb: np.ndarray) -> np.ndarray:
    if emb.size == 0:
        return np.array([])
    return spherical_mean(emb)


def _thirds_indices(n: int) -> Tuple[slice, slice, slice]:
    if n < 3:
        return (slice(0, n), slice(0, 0), slice(0, 0))
    t = n // 3
    r = n % 3
    a = t + (1 if r > 0 else 0)
    b = t + (1 if r > 1 else 0)
    c = n - a - b
    return (slice(0, a), slice(a, a + b), slice(a + b, a + b + c))


def analyze_within_segment(seg: SegmentEmbeddings) -> WithinSegmentStats:
    E = seg.embeddings
    t = E.shape[0] if E.size else 0
    if t == 0:
        return WithinSegmentStats(
            segment_idx=seg.segment_idx,
            true_label=seg.true_label,
            num_frames=0,
            mean_pairwise_cos=None,
            min_pairwise_cos=None,
            max_pairwise_cos=None,
            std_pairwise_cos=None,
            cos_middle_to_centroid=None,
            mean_cos_to_centroid=None,
            mean_cos_begin_third=None,
            mean_cos_mid_third=None,
            mean_cos_end_third=None,
        )

    u = l2_normalize_rows(E)
    c = _centroid(E)
    if np.linalg.norm(c) < 1e-12:
        cos_to_c = np.zeros(t, dtype=np.float64)
    else:
        cn = c / (np.linalg.norm(c) + 1e-12)
        cos_to_c = (u @ cn).astype(np.float64)

    if t == 1:
        pmean = pmin = pmax = 1.0
        pstd = 0.0
    else:
        cmat = cosine_similarity_matrix(E)
        off = pairwise_cosine_off_diagonal(cmat)
        pmean = float(np.mean(off)) if off.size else 1.0
        pmin = float(np.min(off)) if off.size else 1.0
        pmax = float(np.max(off)) if off.size else 1.0
        pstd = float(np.std(off)) if off.size else 0.0

    mid_i = middle_index_sorted(seg.frame_indices_1based)
    cos_mid_cc = float(cos_to_c[mid_i]) if t else None

    s0, s1, s2 = _thirds_indices(t)
    b_mean = float(np.mean(cos_to_c[s0])) if s0.stop - s0.start > 0 else None
    m_mean = float(np.mean(cos_to_c[s1])) if s1.stop - s1.start > 0 else None
    e_mean = float(np.mean(cos_to_c[s2])) if s2.stop - s2.start > 0 else None

    return WithinSegmentStats(
        segment_idx=seg.segment_idx,
        true_label=seg.true_label,
        num_frames=t,
        mean_pairwise_cos=pmean,
        min_pairwise_cos=pmin,
        max_pairwise_cos=pmax,
        std_pairwise_cos=pstd,
        cos_middle_to_centroid=cos_mid_cc,
        mean_cos_to_centroid=float(np.mean(cos_to_c)),
        mean_cos_begin_third=b_mean,
        mean_cos_mid_third=m_mean,
        mean_cos_end_third=e_mean,
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    da = float(np.sqrt(np.sum(a * a)))
    db = float(np.sqrt(np.sum(b * b)))
    if da < 1e-12 or db < 1e-12:
        return float("nan")
    return float(np.sum(a * b) / (da * db))


def analyze_cross_segment_same_item(
    segments: List[SegmentEmbeddings],
) -> List[CrossSegmentSameItem]:
    """For each true label, compare middle-of-segment embeddings across segments."""
    by_item: Dict[str, List[SegmentEmbeddings]] = defaultdict(list)
    for s in segments:
        if not s.is_placeholder and s.embeddings.shape[0] > 0:
            by_item[s.true_label].append(s)

    rows: List[CrossSegmentSameItem] = []
    for lab, segs in sorted(by_item.items()):
        if len(segs) < 2:
            continue
        mids: List[np.ndarray] = []
        raw_mids: List[np.ndarray] = []
        for s in sorted(segs, key=lambda x: x.segment_idx):
            u = l2_normalize_rows(s.embeddings)
            k = s.embeddings.shape[0] // 2
            raw_mids.append(np.asarray(s.embeddings[k], dtype=np.float64).ravel())
            mids.append(u[k])

        M = np.stack(mids, axis=0)  # (k, d)
        ccm = cosine_similarity_matrix(M)
        off = pairwise_cosine_off_diagonal(ccm)
        p_mean = float(np.mean(off)) if off.size else 1.0
        p_min = float(np.min(off)) if off.size else 1.0
        p_max = float(np.max(off)) if off.size else 1.0

        pears: List[float] = []
        m = M.shape[0]
        for i in range(m):
            for j in range(i + 1, m):
                pears.append(_pearson(raw_mids[i], raw_mids[j]))
        pears_valid = [x for x in pears if np.isfinite(x)]
        pear_m = float(np.mean(pears_valid)) if pears_valid else float("nan")

        rows.append(
            CrossSegmentSameItem(
                true_label=lab,
                num_segments=len(segs),
                middle_frame_pairwise_cos_mean=p_mean,
                middle_frame_pairwise_cos_min=p_min,
                middle_frame_pairwise_cos_max=p_max,
                middle_embedding_pearson_mean=pear_m,
            )
        )
    return rows


def _segment_row_key(video_stem: str, seg_idx: int) -> str:
    return f"{video_stem}#s{seg_idx}"


def build_global_segment_similarity_by_item(
    per_video: List[Tuple[str, List[SegmentEmbeddings]]],
) -> List[SameItemSegmentMatrix]:
    """
    For each item label, pool **all** CARRY segments with embeddings across **all** successful
    videos in this run, compute one spherical mean per segment, and build an NxN cosine matrix.
    """
    by_item: Dict[str, List[Tuple[str, SegmentEmbeddings]]] = defaultdict(list)
    for video_stem, segs in per_video:
        for s in segs:
            if not s.is_placeholder and s.embeddings.shape[0] > 0:
                by_item[s.true_label].append((video_stem, s))

    out: List[SameItemSegmentMatrix] = []
    for lab, items in sorted(by_item.items()):
        if len(items) < 2:
            continue
        # Stable order: video, then segment index
        items.sort(key=lambda t: (t[0], t[1].segment_idx))
        means: List[np.ndarray] = []
        v_list: List[str] = []
        s_list: List[int] = []
        labels: List[str] = []
        for video_stem, s in items:
            m = _centroid(s.embeddings)
            if np.linalg.norm(m) < 1e-12:
                continue
            means.append(m)
            v_list.append(video_stem)
            s_list.append(s.segment_idx)
            labels.append(_segment_row_key(video_stem, s.segment_idx))
        if len(means) < 2:
            continue
        Mm = np.stack(means, axis=0)
        ccm = cosine_similarity_matrix(Mm)
        out.append(
            SameItemSegmentMatrix(
                true_label=lab,
                matrix=ccm.astype(np.float64),
                row_labels=labels,
                video_stem_per_row=v_list,
                segment_idx_per_row=s_list,
            )
        )
    return out


def run_full_report(
    segments: List[SegmentEmbeddings],
) -> Dict[str, Any]:
    within = [analyze_within_segment(s) for s in segments]
    cross = analyze_cross_segment_same_item(segments)
    return {
        "within_segments": [asdict(x) for x in within],
        "cross_segment_same_ground_item": [asdict(x) for x in cross],
    }
