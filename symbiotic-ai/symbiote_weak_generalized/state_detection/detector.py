"""State detection for hand tracking in videos.

This module provides the public ``detect_states_from_video`` function used
by the rest of the pipeline.  When a trained HTK HMM model directory and
ARUCO configuration are supplied it runs real state detection; otherwise it
falls back to the original placeholder that marks every frame as
CARRY_WITH (so the training pipeline continues to work without an HMM).
"""

import enum
import os
from typing import List, Optional

import numpy as np
import pandas as pd


class HandState(enum.Enum):
    """Hand states for object manipulation."""

    PICK = "PICK"
    CARRY_WITH = "CARRY_WITH"
    PLACE = "PLACE"
    CARRY_EMPTY = "CARRY_EMPTY"


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def detect_states_from_video(
    video_path: str,
    embeddings: List[np.ndarray],
    frame_numbers: List[int],
    fps: float,
    htk_model_dir: Optional[str] = None,
    aruco_config_path: Optional[str] = None,
    frame_skip: int = 4,
    blur_threshold: float = 100.0,
    clip_model=None,
    clip_processor=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Detect hand states for each frame in a video.

    When *htk_model_dir* points to a directory that contains a trained HTK
    HMM (the output of ``training.train_state_detector``), the full
    feature-extraction + Viterbi-decoding pipeline is executed.

    When *htk_model_dir* is ``None`` **or** the directory does not exist,
    the function falls back to the placeholder behaviour (all frames
    labelled CARRY_WITH) so existing callers are not broken.

    Args:
        video_path: Path to the video file.
        embeddings: CLIP embeddings (kept for backward compatibility with
            the call signature expected by ``video_processor.py``).
        frame_numbers: Frame numbers corresponding to *embeddings*.
        fps: Video frames per second.
        htk_model_dir: Path to trained HTK model directory (optional).
        aruco_config_path: Path to ``aruco_bins.json`` (optional).
        frame_skip: Process every N-th frame during feature extraction.
        blur_threshold: Laplacian blur threshold.
        clip_model: Loaded CLIP model (optional).
        clip_processor: Loaded CLIP processor (optional).
        verbose: Print progress.

    Returns:
        DataFrame with columns ``[timestamp_start, timestamp_end, state]``.
    """
    # Decide whether we can run the HTK pipeline
    use_htk = (
        htk_model_dir is not None
        and os.path.isdir(htk_model_dir)
    )

    if not use_htk:
        return _placeholder_detection(frame_numbers, fps)

    # ---- Full HTK pipeline ----
    from .aruco_detection import ArucoDetector
    from .config import DEFAULT_HTK_CONFIG
    from .feature_extraction import FeatureExtractor
    from .htk_interface import HTKStateDetector

    aruco_detector = ArucoDetector(
        aruco_dict_type=DEFAULT_HTK_CONFIG.aruco_dict_type,
        distance_decay=DEFAULT_HTK_CONFIG.aruco_distance_decay,
    )
    if aruco_config_path and os.path.isfile(aruco_config_path):
        aruco_detector.load_bin_config(aruco_config_path)

    feature_extractor = FeatureExtractor(
        aruco_detector=aruco_detector,
        clip_model=clip_model,
        clip_processor=clip_processor,
    )

    features, frame_nums, video_fps = feature_extractor.extract_video_features(
        video_path,
        frame_skip=frame_skip,
        blur_threshold=blur_threshold,
        verbose=verbose,
    )

    if features.shape[0] == 0:
        if verbose:
            print("[StateDetection] No features extracted; falling back to placeholder.")
        return _placeholder_detection(frame_numbers, fps)

    htk_detector = HTKStateDetector(htk_model_dir)
    state_segments = htk_detector.decode(
        features, video_fps, frame_numbers=frame_nums, verbose=verbose
    )

    return state_segments


# ------------------------------------------------------------------
# Placeholder fallback (original behaviour)
# ------------------------------------------------------------------

def _placeholder_detection(
    frame_numbers: List[int], fps: float
) -> pd.DataFrame:
    """Return all frames as CARRY_WITH (pass-through placeholder)."""
    if len(frame_numbers) == 0:
        return pd.DataFrame(columns=["timestamp_start", "timestamp_end", "state"])

    start_time = frame_numbers[0] / fps if fps > 0 else 0.0
    end_time = frame_numbers[-1] / fps if fps > 0 else 0.0

    return pd.DataFrame(
        [
            {
                "timestamp_start": start_time,
                "timestamp_end": end_time,
                "state": HandState.CARRY_WITH.value,
            }
        ]
    )


__all__ = ["HandState", "detect_states_from_video"]
