"""High-level training pipeline for the HTK HMM state detector."""

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .aruco_detection import ArucoDetector
from .config import DEFAULT_HTK_CONFIG, HTKConfig, STATE_CYCLE
from .feature_extraction import FeatureExtractor
from .htk_interface import HTKStateDetector
from .two_stage import (
    CARRY_STATES,
    COARSE_STATES,
    INTERACT_STATES,
    annotations_to_coarse,
    apply_feature_mask,
    frame_labels_to_segments,
)


def _validate_state_sequence(annotations: pd.DataFrame) -> None:
    """Validate that *annotations* follow the required state cycle.

    Rules:
    * States must follow: PICK -> CARRY_WITH -> PLACE -> CARRY_EMPTY (repeat).
    * No state skipping.
    * Timestamps must be monotonically increasing with no gaps.
    """
    states = annotations["state"].tolist()

    # Check all states are valid
    valid = set(STATE_CYCLE)
    for s in states:
        if s not in valid:
            raise ValueError(
                f"Invalid state '{s}' in annotations. "
                f"Valid states: {STATE_CYCLE}"
            )

    # Check cycle order (allow starting at any point in the cycle)
    if len(states) >= 2:
        for i in range(1, len(states)):
            prev_idx = STATE_CYCLE.index(states[i - 1])
            curr_idx = STATE_CYCLE.index(states[i])
            expected_next = (prev_idx + 1) % len(STATE_CYCLE)
            if curr_idx != expected_next and curr_idx != prev_idx:
                raise ValueError(
                    f"State cycle violation at row {i}: "
                    f"'{states[i-1]}' -> '{states[i]}' is not allowed. "
                    f"Expected '{STATE_CYCLE[expected_next]}' or same state."
                )

    # Check monotonically increasing timestamps
    for i in range(1, len(annotations)):
        if annotations.iloc[i]["timestamp_start"] < annotations.iloc[i - 1]["timestamp_end"]:
            raise ValueError(
                f"Timestamps not monotonically increasing at row {i}: "
                f"start {annotations.iloc[i]['timestamp_start']} < "
                f"prev end {annotations.iloc[i-1]['timestamp_end']}"
            )


# ---------------------------------------------------------------------------
# Feature cache helpers
# ---------------------------------------------------------------------------

def _cache_key(
    video_path: str,
    frame_skip: int,
    blur_threshold: float,
    feature_mask: Optional[List[int]] = None,
    aruco_persistence_frames: int = 0,
    aruco_smoothing_window: int = 1,
) -> dict:
    """Return a dict of cache-busting parameters for *video_path*."""
    return {
        "mtime": os.path.getmtime(video_path),
        "frame_skip": frame_skip,
        "blur_threshold": blur_threshold,
        "feature_mask": list(feature_mask) if feature_mask is not None else None,
        "aruco_persistence_frames": int(aruco_persistence_frames),
        "aruco_smoothing_window": int(aruco_smoothing_window),
        "feature_dim": int(FeatureExtractor.FEATURE_DIM),
    }


def _load_manifest(manifest_path: str) -> dict:
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest_path: str, manifest: dict) -> None:
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def _load_cached_features(
    cache_dir: str,
    manifest: dict,
    video_path: str,
    frame_skip: int,
    blur_threshold: float,
    feature_mask: Optional[List[int]],
    aruco_persistence_frames: int,
    aruco_smoothing_window: int,
    verbose: bool,
) -> Optional[np.ndarray]:
    """Return cached feature array if valid, else None."""
    stem = Path(video_path).stem
    if stem not in manifest:
        return None

    expected = _cache_key(
        video_path,
        frame_skip,
        blur_threshold,
        feature_mask,
        aruco_persistence_frames=aruco_persistence_frames,
        aruco_smoothing_window=aruco_smoothing_window,
    )
    if manifest[stem] != expected:
        if verbose:
            print(f"  [cache] '{stem}' cache stale (video or params changed), re-extracting.")
        return None

    npy_path = os.path.join(cache_dir, f"{stem}.npy")
    if not os.path.isfile(npy_path):
        return None

    features = np.load(npy_path)
    if verbose:
        print(f"  [cache] '{stem}' loaded from cache ({features.shape[0]} frames).")
    return features


def _save_cached_features(
    cache_dir: str,
    manifest: dict,
    video_path: str,
    features: np.ndarray,
    frame_skip: int,
    blur_threshold: float,
    feature_mask: Optional[List[int]],
    aruco_persistence_frames: int,
    aruco_smoothing_window: int,
) -> None:
    """Write feature array to cache and update manifest."""
    stem = Path(video_path).stem
    npy_path = os.path.join(cache_dir, f"{stem}.npy")
    np.save(npy_path, features)
    manifest[stem] = _cache_key(
        video_path,
        frame_skip,
        blur_threshold,
        feature_mask,
        aruco_persistence_frames=aruco_persistence_frames,
        aruco_smoothing_window=aruco_smoothing_window,
    )


def _get_frame_numbers_and_fps(video_path: str, frame_skip: int, n_rows: int) -> tuple[list[int], float]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        fps = 30.0
    frames = [i for i in range(1, total + 1) if i % frame_skip == 0]
    if len(frames) > n_rows:
        frames = frames[:n_rows]
    elif len(frames) < n_rows:
        if len(frames) == 0:
            frames = list(range(1, n_rows + 1))
        else:
            last = frames[-1]
            while len(frames) < n_rows:
                last += frame_skip
                frames.append(last)
    return frames, float(fps)


def _state_at_time(t: float, annotations: pd.DataFrame) -> Optional[str]:
    for _, row in annotations.iterrows():
        s = float(row["timestamp_start"])
        e = float(row["timestamp_end"])
        if s <= t <= e:
            return str(row["state"])
    return None


def _build_subtype_training_pair(
    features: np.ndarray,
    frame_numbers: List[int],
    fps: float,
    annotations: pd.DataFrame,
    allowed_states: set[str],
) -> Optional[tuple[np.ndarray, pd.DataFrame]]:
    labels: List[str] = []
    selected_features: List[np.ndarray] = []
    selected_frames: List[int] = []
    for i, fn in enumerate(frame_numbers):
        t = fn / fps if fps > 0 else 0.0
        st = _state_at_time(t, annotations)
        if st is None or st not in allowed_states:
            continue
        labels.append(st)
        selected_features.append(features[i])
        selected_frames.append(fn)
    if not selected_features:
        return None
    sub_feats = np.vstack(selected_features)
    segs = frame_labels_to_segments(labels, selected_frames, fps)
    if segs.empty:
        return None
    return sub_feats, segs


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------

def train_state_detector(
    video_paths: List[str],
    annotation_paths: List[str],
    output_dir: str,
    aruco_config_path: Optional[str] = None,
    clip_model=None,
    clip_processor=None,
    frame_skip: int = 4,
    blur_threshold: float = 100.0,
    feature_mask: Optional[List[int]] = None,
    coarse_feature_mask: Optional[List[int]] = None,
    interact_feature_mask: Optional[List[int]] = None,
    carry_feature_mask: Optional[List[int]] = None,
    pipeline_mode: str = "two-stage",
    aruco_persistence_frames: int = 0,
    aruco_smoothing_window: int = 1,
    min_segment_seconds: float = 0.15,
    config: Optional[HTKConfig] = None,
    verbose: bool = True,
) -> str:
    """Train an HTK HMM state detector from annotated videos.

    Each video must have a corresponding CSV annotation file with columns
    ``[timestamp_start, timestamp_end, state]``.

    Extracted feature vectors are cached in ``{output_dir}/feature_cache/``
    so that subsequent runs skip the (slow) per-frame MediaPipe extraction
    for videos whose file and parameters have not changed.

    Args:
        video_paths: Paths to training video files.
        annotation_paths: Matching CSV annotation files (same order).
        output_dir: Where to save the trained HMM.
        aruco_config_path: Path to ``aruco_bins.json``.
        clip_model: Loaded CLIP model (optional, for object confidence).
        clip_processor: Loaded CLIP processor.
        frame_skip: Process every N-th frame.
        blur_threshold: Laplacian variance threshold.
        config: ``HTKConfig`` instance (defaults to ``DEFAULT_HTK_CONFIG``).
        verbose: Print progress.

    Returns:
        Path to the final trained model directory.
    """
    if len(video_paths) != len(annotation_paths):
        raise ValueError(
            "Number of videos must equal number of annotation files."
        )

    cfg = config or DEFAULT_HTK_CONFIG
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Feature cache setup
    cache_dir = os.path.join(output_dir, "feature_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "manifest.json")
    manifest = _load_manifest(manifest_path)

    # Initialise ARUCO detector
    aruco_detector = ArucoDetector(
        aruco_dict_type=cfg.aruco_dict_type,
        distance_decay=cfg.aruco_distance_decay,
    )
    if aruco_config_path and os.path.isfile(aruco_config_path):
        aruco_detector.load_bin_config(aruco_config_path)

    # Initialise feature extractor (only used when cache misses occur)
    feature_extractor = FeatureExtractor(
        aruco_detector=aruco_detector,
        clip_model=clip_model,
        clip_processor=clip_processor,
        feature_mask=None,
        aruco_persistence_frames=aruco_persistence_frames,
        aruco_smoothing_window=aruco_smoothing_window,
    )

    # Extract (or load from cache) features for all training videos
    records: List[dict] = []
    cache_hits = 0
    cache_misses = 0

    for video_path, annotation_path in zip(video_paths, annotation_paths):
        if verbose:
            print(f"\n[HMM Training] Processing: {video_path}")

        annotations = pd.read_csv(annotation_path)
        _validate_state_sequence(annotations)

        # Try cache first
        features = _load_cached_features(
            cache_dir,
            manifest,
            video_path,
            frame_skip,
            blur_threshold,
            feature_mask,
            aruco_persistence_frames,
            aruco_smoothing_window,
            verbose,
        )

        if features is not None:
            cache_hits += 1
        else:
            # Cache miss — run full feature extraction
            features, frame_numbers, fps = feature_extractor.extract_video_features(
                video_path,
                frame_skip=frame_skip,
                blur_threshold=blur_threshold,
                verbose=verbose,
            )
            feature_extractor.reset()

            if features.shape[0] == 0:
                if verbose:
                    print(f"  WARNING: No features extracted from {video_path}, skipping.")
                continue

            _save_cached_features(
                cache_dir,
                manifest,
                video_path,
                features,
                frame_skip,
                blur_threshold,
                feature_mask,
                aruco_persistence_frames,
                aruco_smoothing_window,
            )
            cache_misses += 1

        if features is not None and ("frame_numbers" not in locals() or "fps" not in locals()):
            frame_numbers, fps = _get_frame_numbers_and_fps(video_path, frame_skip, int(features.shape[0]))
        records.append(
            {
                "features": features,
                "annotations": annotations,
                "frame_numbers": frame_numbers,
                "fps": fps,
            }
        )
        if "frame_numbers" in locals():
            del frame_numbers
        if "fps" in locals():
            del fps

    # Persist updated manifest
    _save_manifest(manifest_path, manifest)

    if verbose and (cache_hits + cache_misses) > 0:
        print(
            f"\n[HMM Training] Feature cache: "
            f"{cache_hits} hit(s), {cache_misses} extracted and saved."
        )

    if len(records) == 0:
        raise ValueError("No usable training data extracted from any video.")

    # Train HTK HMM
    if verbose:
        print(f"\n[HMM Training] Training on {len(records)} video(s) ...")

    coarse_mask = coarse_feature_mask if coarse_feature_mask is not None else feature_mask
    interact_mask = interact_feature_mask if interact_feature_mask is not None else feature_mask
    carry_mask = carry_feature_mask if carry_feature_mask is not None else feature_mask

    fine_training_data: List[tuple[np.ndarray, pd.DataFrame]] = []
    coarse_training_data: List[tuple[np.ndarray, pd.DataFrame]] = []
    interact_training_data: List[tuple[np.ndarray, pd.DataFrame]] = []
    carry_training_data: List[tuple[np.ndarray, pd.DataFrame]] = []

    for rec in records:
        feats = rec["features"]
        ann = rec["annotations"]
        frame_numbers = rec["frame_numbers"]
        fps = rec["fps"]

        fine_training_data.append((apply_feature_mask(feats, feature_mask), ann))
        coarse_training_data.append((apply_feature_mask(feats, coarse_mask), annotations_to_coarse(ann)))

        inter_pair = _build_subtype_training_pair(
            apply_feature_mask(feats, interact_mask),
            frame_numbers,
            fps,
            ann,
            set(INTERACT_STATES),
        )
        if inter_pair is not None:
            interact_training_data.append(inter_pair)

        carry_pair = _build_subtype_training_pair(
            apply_feature_mask(feats, carry_mask),
            frame_numbers,
            fps,
            ann,
            set(CARRY_STATES),
        )
        if carry_pair is not None:
            carry_training_data.append(carry_pair)

    # Keep legacy-compatible fine model available in every mode.
    fine_detector = HTKStateDetector(output_dir, config=cfg)
    fine_detector.train(fine_training_data, output_dir, verbose=verbose)

    if pipeline_mode == "two-stage":
        coarse_dir = os.path.join(output_dir, "coarse_model")
        if verbose:
            print(f"[HMM Training] Training coarse model in: {coarse_dir}")
        coarse_detector = HTKStateDetector(coarse_dir, config=cfg, state_labels=COARSE_STATES)
        coarse_detector.train(coarse_training_data, coarse_dir, verbose=verbose)

        if interact_training_data:
            interact_dir = os.path.join(output_dir, "interact_model")
            if verbose:
                print(f"[HMM Training] Training interact model in: {interact_dir}")
            interact_detector = HTKStateDetector(interact_dir, config=cfg, state_labels=INTERACT_STATES)
            interact_detector.train(interact_training_data, interact_dir, verbose=verbose)

        if carry_training_data:
            carry_dir = os.path.join(output_dir, "carry_model")
            if verbose:
                print(f"[HMM Training] Training carry model in: {carry_dir}")
            carry_detector = HTKStateDetector(carry_dir, config=cfg, state_labels=CARRY_STATES)
            carry_detector.train(carry_training_data, carry_dir, verbose=verbose)

    final_dir = os.path.join(output_dir, "models", "hmm_final")
    pipeline_cfg = {
        "pipeline_mode": pipeline_mode,
        "coarse_model_dir": "coarse_model" if pipeline_mode == "two-stage" else None,
        "interact_model_dir": "interact_model" if pipeline_mode == "two-stage" else None,
        "carry_model_dir": "carry_model" if pipeline_mode == "two-stage" else None,
        "coarse_states": COARSE_STATES if pipeline_mode == "two-stage" else None,
        "interact_states": INTERACT_STATES if pipeline_mode == "two-stage" else None,
        "carry_states": CARRY_STATES if pipeline_mode == "two-stage" else None,
        "fine_states": STATE_CYCLE,
        "aruco_persistence_frames": int(aruco_persistence_frames),
        "aruco_smoothing_window": int(aruco_smoothing_window),
        "min_segment_seconds": float(min_segment_seconds),
    }
    with open(os.path.join(final_dir, "pipeline_config.json"), "w") as f:
        json.dump(pipeline_cfg, f, indent=2)

    if verbose:
        print(f"[HMM Training] Done. Model saved to: {final_dir}")

    return final_dir


__all__ = ["train_state_detector"]
