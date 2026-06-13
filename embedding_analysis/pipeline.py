"""End-to-end analysis for one video."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .io_cache import load_cache_dir
from .io_ground_truth import load_ground_truth_column
from .report import write_reports
from .segments import build_segments


def _video_frame_count(video_path: Path) -> int:
    try:
        import cv2  # type: ignore
    except Exception:
        return 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return n


def _ensure_symbiote_path(symbiotic_ai_root: Path) -> None:
    s = str(symbiotic_ai_root.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def get_carry_intervals(
    manual_csv: Path,
    total_frames: int,
    frame_indexing: str,
    symbiotic_ai_root: Path,
) -> List[Tuple[int, int]]:
    _ensure_symbiote_path(symbiotic_ai_root)
    from symbiote_weak.state_detection.compact_timeline import (  # type: ignore
        carry_with_pipeline_frame_intervals_1based,
    )

    return carry_with_pipeline_frame_intervals_1based(
        str(manual_csv), total_frames, frame_indexing=frame_indexing
    )


def count_expected_frames_in_intervals(
    intervals: List[Tuple[int, int]], frame_skip: int
) -> int:
    n = 0
    for lo, hi in intervals:
        for f in range(lo, hi + 1):
            if f % int(frame_skip) != 0:
                continue
            n += 1
    return n


def analyze_video(
    *,
    video_stem: str,
    video_path: Path,
    ground_truth_csv: Path,
    manual_labels_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    symbiotic_ai_root: Path,
    frame_skip: int = 4,
    frame_indexing: str = "opencv0",
    total_frames_override: Optional[int] = None,
    hand_neutralize_components: int = 0,
    hand_embeddings_dir: Optional[str] = None,
) -> Dict[str, Any]:
    expected = load_ground_truth_column(ground_truth_csv, video_stem)

    if total_frames_override is not None and int(total_frames_override) > 0:
        total_frames = int(total_frames_override)
    else:
        total_frames = _video_frame_count(video_path)
    if total_frames <= 0:
        raise RuntimeError(
            f"Could not read frame count for {video_path}. "
            "Install opencv-python, or pass total_frames_override, or use --total-frames."
        )

    man = manual_labels_dir / f"{video_stem}.csv"
    if not man.is_file():
        raise FileNotFoundError(f"Manual state CSV: {man}")

    intervals = get_carry_intervals(
        man, total_frames, frame_indexing, symbiotic_ai_root
    )
    emb_by_frame = load_cache_dir(cache_dir)
    if not emb_by_frame:
        raise FileNotFoundError(f"No frame embeddings in {cache_dir}")

    # Apply hand neutralization if requested
    neutralizer = None
    if hand_neutralize_components > 0 and hand_embeddings_dir:
        _ensure_symbiote_path(symbiotic_ai_root)
        try:
            from symbiote_weak_generalized.training.hand_neutralizer import HandNeutralizer  # type: ignore

            neutralizer = HandNeutralizer(
                hand_embeddings_dir,
                n_components=hand_neutralize_components,
                verbose=True,
            )
            if neutralizer.enabled:
                print(
                    f"  [embedding_analysis] Applying hand neutralization (n_components={hand_neutralize_components})"
                )
                emb_by_frame = {
                    frame_num: neutralizer.neutralize(emb)
                    for frame_num, emb in emb_by_frame.items()
                }
            else:
                print(
                    f"  [embedding_analysis] Hand neutralization disabled (not enough data or dir not found)"
                )
        except Exception as e:
            print(
                f"  [embedding_analysis] Warning: could not apply hand neutralization: {e}",
                file=sys.stderr,
            )

    dim = int(next(iter(emb_by_frame.values())).shape[0])
    want = count_expected_frames_in_intervals(intervals, frame_skip)
    have = 0
    for lo, hi in intervals:
        for f in range(lo, hi + 1):
            if f % int(frame_skip) != 0:
                continue
            if f in emb_by_frame:
                have += 1

    notes: List[str] = []
    n_int = len(intervals)
    n_gt = len(expected)
    n_use = min(n_int, n_gt)
    if n_int != n_gt:
        notes.append(
            f"Interval count ({n_int}) != ground-truth row count ({n_gt}); using first {n_use}."
        )
    if n_use < n_int:
        intervals = intervals[:n_use]
    if n_use < n_gt:
        expected = expected[:n_use]
        notes.append("Trimmed ground truth to match intervals.")

    segments = build_segments(
        intervals, expected, emb_by_frame, frame_skip=frame_skip
    )

    with_pair = [s for s in segments if s.embeddings.shape[0] >= 2]
    mean_w = None
    if with_pair:
        from .metrics import analyze_within_segment

        stats = [analyze_within_segment(s) for s in with_pair]
        mean_w = float(
            np.mean(
                [x.mean_pairwise_cos for x in stats if x.mean_pairwise_cos is not None]
            )
        )

    coverage: Dict[str, Any] = {
        "clip_dim": dim,
        "total_frames_in_video": total_frames,
        "carry_intervals": n_use,
        "expected_embeddable_frames_in_carry": want,
        "embeddings_found_in_carry": have,
        "frame_skip": frame_skip,
        "hand_neutralize_components": hand_neutralize_components if neutralizer and neutralizer.enabled else 0,
        "mean_pairwise_cos_within_segment_with_2plus_frames": mean_w,
    }
    coverage["summary"] = (
        f"Found {have}/{want} cached frames inside CARRY intervals "
        f"(frame_skip={frame_skip}); "
        f"{len([s for s in segments if s.is_placeholder])} empty/partial segment(s). "
    )
    if mean_w is not None:
        coverage["summary"] += f"Mean within-seg pairwise cos (2+ fr) ≈ {mean_w:.4f}."

    jpath, mpath = write_reports(
        output_dir, video_stem, segments, coverage, notes
    )
    return {
        "json": str(jpath),
        "markdown": str(mpath),
        "segments_built": len(segments),
        "coverage": coverage,
        "notes": notes,
        "segments": segments,
    }
