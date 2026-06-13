"""Helpers for two-stage 4-label state decoding.

Stage A predicts coarse states:
    INTERACT, CARRY

Stage B predicts subtype states with dedicated models:
    INTERACT -> PICK, PLACE
    CARRY    -> CARRY_WITH, CARRY_EMPTY
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .feature_extraction import FeatureExtractor

FINE_STATES = ["PICK", "CARRY_WITH", "PLACE", "CARRY_EMPTY"]
COARSE_STATES = ["INTERACT", "CARRY"]
INTERACT_STATES = ["PICK", "PLACE"]
CARRY_STATES = ["CARRY_WITH", "CARRY_EMPTY"]
_INTERACT_FINE = {"PICK", "PLACE"}
_CARRY_FINE = {"CARRY_WITH", "CARRY_EMPTY"}


def map_fine_to_coarse(state: str) -> str:
    if state in _INTERACT_FINE:
        return "INTERACT"
    if state in _CARRY_FINE:
        return "CARRY"
    return state


def annotations_to_coarse(annotations: pd.DataFrame) -> pd.DataFrame:
    out = annotations.copy()
    out["state"] = out["state"].map(map_fine_to_coarse)
    return out


def annotations_for_states(annotations: pd.DataFrame, allowed_states: Sequence[str]) -> pd.DataFrame:
    allowed = set(allowed_states)
    return annotations[annotations["state"].isin(allowed)].reset_index(drop=True)


def derive_expected_sequence_from_segments(segments: pd.DataFrame) -> List[str]:
    """Derive a de-duplicated expected state sequence from labeled segments."""
    if segments.empty or "state" not in segments.columns:
        return []
    expected: List[str] = []
    prev: Optional[str] = None
    for st in segments["state"].astype(str).tolist():
        if st != prev:
            expected.append(st)
            prev = st
    return expected


def project_labels_to_expected_sequence(
    labels: Sequence[str],
    expected_sequence: Optional[Sequence[str]],
) -> List[str]:
    """Project decoded labels to expected order while preserving run durations."""
    out = [str(x) for x in labels]
    if not out or not expected_sequence:
        return out
    expected = [str(x) for x in expected_sequence if str(x)]
    if not expected:
        return out
    projected: List[str] = []
    seq_idx = 0
    for label in out:
        while seq_idx + 1 < len(expected) and label == expected[seq_idx + 1]:
            seq_idx += 1
        projected.append(expected[seq_idx])
    return projected


def merge_short_segments(
    labels: Sequence[str],
    frame_numbers: Sequence[int],
    fps: float,
    min_segment_seconds: float = 0.15,
) -> List[str]:
    """Merge very short runs into neighboring labels to reduce jitter."""
    if not labels:
        return []
    if len(labels) != len(frame_numbers):
        raise ValueError("labels length must match frame_numbers length")
    min_seg = max(0.0, float(min_segment_seconds))
    if min_seg <= 0.0:
        return [str(x) for x in labels]

    out = [str(x) for x in labels]
    runs: List[Tuple[str, int, int]] = []
    start = 0
    curr = out[0]
    for i in range(1, len(out)):
        if out[i] != curr:
            runs.append((curr, start, i))
            curr = out[i]
            start = i
    runs.append((curr, start, len(out)))

    for run_idx, (_, s, e) in enumerate(runs):
        st = frame_numbers[s] / fps if fps > 0 else float(s)
        et = frame_numbers[e - 1] / fps if fps > 0 else float(e - 1)
        dur = max(0.0, et - st)
        if dur >= min_seg:
            continue
        left = runs[run_idx - 1][0] if run_idx > 0 else None
        right = runs[run_idx + 1][0] if run_idx + 1 < len(runs) else None
        replacement = right if right is not None else left
        if replacement is None:
            continue
        for j in range(s, e):
            out[j] = replacement
    return out


def _state_at_time(t: float, segments: Sequence[Tuple[float, float, str]]) -> Optional[str]:
    for s, e, st in segments:
        if s <= t <= e:
            return st
    return None


def segments_to_frame_labels(
    segments: pd.DataFrame,
    frame_numbers: Sequence[int],
    fps: float,
) -> List[Optional[str]]:
    if segments.empty:
        return [None] * len(frame_numbers)
    segs = [
        (float(r.timestamp_start), float(r.timestamp_end), str(r.state))
        for r in segments.itertuples()
    ]
    out: List[Optional[str]] = []
    for fn in frame_numbers:
        t = fn / fps if fps > 0 else 0.0
        out.append(_state_at_time(t, segs))
    return out


def apply_feature_mask(features: np.ndarray, feature_mask: Optional[List[int]]) -> np.ndarray:
    if feature_mask is None:
        return features
    if features.ndim != 2:
        raise ValueError("features must be 2D")
    masked = np.zeros_like(features)
    valid = [i for i in feature_mask if 0 <= i < features.shape[1]]
    if valid:
        masked[:, valid] = features[:, valid]
    return masked


def build_coarse_runs(coarse_labels: Sequence[Optional[str]]) -> List[Tuple[str, int, int]]:
    runs: List[Tuple[str, int, int]] = []
    if not coarse_labels:
        return runs
    curr = coarse_labels[0] if coarse_labels[0] in {"INTERACT", "CARRY"} else "CARRY"
    start = 0
    for i in range(1, len(coarse_labels)):
        nxt = coarse_labels[i] if coarse_labels[i] in {"INTERACT", "CARRY"} else curr
        if nxt != curr:
            runs.append((str(curr), start, i))
            curr = nxt
            start = i
    runs.append((str(curr), start, len(coarse_labels)))
    return runs


def _run_fallback_label(
    coarse_state: str,
    last_interact_label: str,
    frame_feat: np.ndarray,
) -> str:
    if coarse_state == "INTERACT":
        n = int(frame_feat.shape[0])
        idx = FeatureExtractor.IDX_ARUCO_SIGNED
        # Back-compat: old 15-D vectors used scaled signed context at index 14.
        if n <= 14:
            aruco = 0.0
        elif n <= 15:
            aruco = float(frame_feat[14])
        else:
            aruco = float(frame_feat[idx]) if n > idx else 0.0
        pick_p = float(frame_feat[FeatureExtractor.IDX_ARUCO_PICK]) if n > FeatureExtractor.IDX_ARUCO_PICK else 0.0
        place_p = float(frame_feat[FeatureExtractor.IDX_ARUCO_PLACE]) if n > FeatureExtractor.IDX_ARUCO_PLACE else 0.0
        if abs(aruco) >= 0.2:
            # Positive signed context = pick bins dominate.
            return "PICK" if aruco > 0.0 else "PLACE"
        if pick_p >= 0.2 and pick_p > place_p:
            return "PICK"
        if place_p >= 0.2 and place_p > pick_p:
            return "PLACE"
        return "PLACE" if last_interact_label == "PICK" else "PICK"

    orientation_z = float(frame_feat[12]) if frame_feat.shape[0] > 12 else 0.0
    velocity_y = float(frame_feat[3]) if frame_feat.shape[0] > 3 else 0.0
    carry_score = 0.0
    carry_score += 0.75 if last_interact_label == "PLACE" else -0.75
    carry_score += 0.15 * np.sign(orientation_z)
    carry_score += 0.10 * np.sign(-velocity_y)
    return "CARRY_EMPTY" if carry_score >= 0 else "CARRY_WITH"


def decode_subtype_with_runs(
    coarse_frame_labels: Sequence[Optional[str]],
    features: np.ndarray,
    fps: float,
    frame_numbers: Sequence[int],
    interact_detector=None,
    carry_detector=None,
    strict_cycle: bool = False,
    word_penalty: float = 0.0,
    grammar_scale: float = 1.0,
    interact_mask: Optional[List[int]] = None,
    carry_mask: Optional[List[int]] = None,
    expected_interact_sequence: Optional[Sequence[str]] = None,
    expected_carry_sequence: Optional[Sequence[str]] = None,
) -> List[str]:
    """Decode subtype labels per coarse run using dedicated subtype models."""
    if len(coarse_frame_labels) != int(features.shape[0]):
        raise ValueError("coarse labels length must match features rows")
    n = int(features.shape[0])
    out: List[Optional[str]] = [None] * n
    last_interact_label = "PICK"

    runs = build_coarse_runs(coarse_frame_labels)
    for coarse_state, s, e in runs:
        run_feats = features[s:e]
        run_frames = list(frame_numbers[s:e])
        if run_feats.shape[0] == 0:
            continue
        detector = interact_detector if coarse_state == "INTERACT" else carry_detector
        run_mask = interact_mask if coarse_state == "INTERACT" else carry_mask
        masked_feats = apply_feature_mask(run_feats, run_mask)

        decoded: List[Optional[str]] = [None] * run_feats.shape[0]
        if detector is not None and run_feats.shape[0] >= 3:
            try:
                expected_seq = expected_interact_sequence if coarse_state == "INTERACT" else expected_carry_sequence
                seg_df = detector.decode(
                    masked_feats,
                    fps,
                    frame_numbers=run_frames,
                    verbose=False,
                    word_penalty=word_penalty,
                    grammar_scale=grammar_scale,
                    strict_cycle=strict_cycle,
                    expected_sequence=list(expected_seq) if expected_seq else None,
                )
                decoded = segments_to_frame_labels(seg_df, run_frames, fps)
            except Exception:
                decoded = [None] * run_feats.shape[0]

        for local_i in range(run_feats.shape[0]):
            label = decoded[local_i]
            if label is None:
                label = _run_fallback_label(coarse_state, last_interact_label, run_feats[local_i])
            out[s + local_i] = label
            if label in _INTERACT_FINE:
                last_interact_label = label

    # Fill any remaining gaps with robust fallback.
    for i in range(n):
        if out[i] is None:
            coarse_state = coarse_frame_labels[i] if coarse_frame_labels[i] in {"INTERACT", "CARRY"} else "CARRY"
            out[i] = _run_fallback_label(str(coarse_state), last_interact_label, features[i])
            if out[i] in _INTERACT_FINE:
                last_interact_label = str(out[i])

    return [str(x) for x in out]


def frame_labels_to_segments(
    labels: Sequence[str],
    frame_numbers: Sequence[int],
    fps: float,
) -> pd.DataFrame:
    if not labels:
        return pd.DataFrame(columns=["timestamp_start", "timestamp_end", "state"])
    if len(labels) != len(frame_numbers):
        raise ValueError("labels length must match frame_numbers length")

    segments: List[Dict[str, float | str]] = []
    curr = labels[0]
    start_idx = 0
    for i in range(1, len(labels)):
        if labels[i] != curr:
            s_t = frame_numbers[start_idx] / fps if fps > 0 else float(start_idx)
            e_t = frame_numbers[i - 1] / fps if fps > 0 else float(i - 1)
            segments.append({"timestamp_start": s_t, "timestamp_end": e_t, "state": curr})
            curr = labels[i]
            start_idx = i

    s_t = frame_numbers[start_idx] / fps if fps > 0 else float(start_idx)
    e_t = frame_numbers[-1] / fps if fps > 0 else float(len(labels) - 1)
    segments.append({"timestamp_start": s_t, "timestamp_end": e_t, "state": curr})
    return pd.DataFrame(segments)


def boundary_timestamps_from_segments(segments: pd.DataFrame) -> List[float]:
    """Return segment end timestamps excluding the final segment end."""
    if segments.empty:
        return []
    vals = [float(v) for v in segments["timestamp_end"].tolist()]
    if len(vals) <= 1:
        return []
    return vals[:-1]


def boundary_rmse_seconds(pred_boundaries: Sequence[float], gt_boundaries: Sequence[float]) -> float:
    """RMSE between nearest boundary pairs in seconds."""
    pred = np.array(list(pred_boundaries), dtype=float)
    gt = np.array(list(gt_boundaries), dtype=float)
    if pred.size == 0 or gt.size == 0:
        return float("inf")
    errs = []
    for g in gt:
        errs.append(float(np.min(np.abs(pred - g))))
    for p in pred:
        errs.append(float(np.min(np.abs(gt - p))))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("inf")

