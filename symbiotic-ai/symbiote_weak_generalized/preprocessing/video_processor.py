"""Video frame extraction and processing utilities."""

import os
import sys
from typing import List, Tuple, Optional, Callable
import cv2
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor
import mediapipe as mp

from ..lib.hand_detection import segment_hand_at_full_resolution
from ..embeddings.cache_manager import (
    load_frame_from_cache,
    load_frame_seg_from_cache,
    save_frame_to_cache,
)
from .blur_detection import is_blurry


def _prepare_hand_crops_for_embedding(
    image_rgb: np.ndarray,
    hands_detector,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run hand detection/blur checks on 1080p (or smaller) and return crops.

    Returns:
        (full_resolution_crop_for_clip, processing_resolution_crop_for_blur_check)
    """
    crop_full, crop_proc = segment_hand_at_full_resolution(image_rgb, hands_detector)
    if crop_proc is None or crop_proc.size == 0:
        return None, None
    if crop_full is None or crop_full.size == 0:
        return None, crop_proc
    return crop_full, crop_proc


def process_video_frames(
    video_path: str,
    label: str,
    model: AutoModel,
    processor: AutoProcessor,
    cache_dir: str,
    threshold: float = 50.0,
    frame_skip: int = 4,
    state_detection_func: Optional[Callable] = None,
    verbose: bool = True,
    allowed_frame_intervals_1based: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[List[np.ndarray], List[str], List[str], pd.DataFrame, List[int]]:
    """
    Extract non-blurry frames from video and embed them directly.
    
    Hand detection and blur filtering run on a downscaled copy (max 1080p) so
    MediaPipe stays reliable. The detected crop box is projected back onto the
    original-resolution frame before CLIP embedding and cache storage.
    
    Restrict which frames are embedded using ``allowed_frame_intervals_1based``
    (e.g. CARRY_WITH spans from a compact label CSV). There is no second-pass
    filtering by detected state; state_detection_func only supplies
    ``state_results`` for downstream segment alignment.
    
    Args:
        video_path: Path to the video file
        label: Class label for these frames
        model: CLIP model for embedding
        processor: CLIP processor
        cache_dir: Directory to cache embeddings
        threshold: Blur threshold (Laplacian variance, default 50.0)
        frame_skip: Process every Nth frame (default 4)
        state_detection_func: Optional function to build state timeline (not used to drop frames)
        verbose: Whether to print progress
        allowed_frame_intervals_1based: If set, only decode/embed frames whose
            1-based frame index lies in one of these inclusive [start, end] ranges
            (e.g. CARRY_WITH spans from a compact state CSV).
    
    Returns:
        Tuple of (embeddings, labels, synthetic_paths, state_results, frame_indices) where:
        - embeddings: List of numpy arrays
        - labels: List of label strings (all same)
        - synthetic_paths: List of synthetic paths for tracking (video_name_frame_N)
        - state_results: DataFrame with state detection results
        - frame_indices: 1-based video frame index for each returned embedding
    """
    if verbose:
        print("\n" + "="*60)
        print("PROCESSING VIDEO FRAMES")
        print("="*60)
        print(f"Video: {video_path}")
        print(f"Label: {label}")
        print(f"Blur threshold: {threshold}")
        print(f"Frame skip: {frame_skip}")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create MediaPipe Hands detector ONCE for the entire video
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3,
        max_num_hands=2
    )
    
    try:
        # Open video
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if verbose:
            print(f"Total frames in video: {total_frames}")
            print(f"FPS: {fps:.2f}")
            print(f"Processing every {frame_skip} frames...")
            if allowed_frame_intervals_1based:
                print(
                    f"CARRY-only windowing: {len(allowed_frame_intervals_1based)} interval(s) from labels"
                )
            if state_detection_func is not None:
                print(f"State detection: ENABLED (for timeline only; no post-filter)")
        
        # First pass: extract all valid embeddings
        all_embeddings = []
        all_frame_numbers = []
        embedded_crop_hw: List[Tuple[int, int]] = []
        source_frame_hw = (
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
        
        frame_count = 0
        processed_count = 0
        embedded_count = 0
        cache_hits = 0
        
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames according to frame_skip
            if frame_count % frame_skip != 0:
                continue
            
            processed_count += 1

            if allowed_frame_intervals_1based is not None:
                if not any(lo <= frame_count <= hi for (lo, hi) in allowed_frame_intervals_1based):
                    continue
            
            # CHECK CACHE FIRST
            cached_embedding = load_frame_from_cache(label, frame_count, cache_dir)
            if cached_embedding is not None:
                all_embeddings.append(cached_embedding)
                all_frame_numbers.append(frame_count)
                cached_seg = load_frame_seg_from_cache(label, frame_count, cache_dir)
                if cached_seg is not None and cached_seg.size > 0:
                    embedded_crop_hw.append(
                        (int(cached_seg.shape[0]), int(cached_seg.shape[1]))
                    )
                cache_hits += 1
                embedded_count += 1
                if verbose and embedded_count % 20 == 0:
                    print(f"  Frame {frame_count}: ✓ CACHED ({cache_hits} cached, {embedded_count - cache_hits} new)")
                continue
            
            # Convert to RGB; detect/blur on downscaled copy, embed full-res crop
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            crop_full, crop_proc = _prepare_hand_crops_for_embedding(
                image_rgb, hands_detector
            )

            if crop_proc is None:
                if verbose:
                    print(f"  Frame {frame_count}: ✗ FAILED (hand not detected)")
                continue

            if crop_full is None:
                if verbose:
                    print(f"  Frame {frame_count}: ✗ FAILED (empty full-resolution crop)")
                continue

            # Blur check uses the 1080p (processing) crop
            if is_blurry(crop_proc, threshold):
                if verbose:
                    print(f"  Frame {frame_count}: BLURRY HAND (skipped)")
                continue

            # Embed the projected full-resolution hand crop
            try:
                inputs = processor(images=[crop_full], return_tensors="pt").to(model.device)
                
                # Generate embedding (frozen CLIP)
                with torch.no_grad():
                    embeddings_tensor = model.get_image_features(**inputs)
                embedding = embeddings_tensor.cpu().numpy()[0]
                
                # Collect all embeddings for state detection
                all_embeddings.append(embedding)
                all_frame_numbers.append(frame_count)
                embedded_crop_hw.append(
                    (int(crop_full.shape[0]), int(crop_full.shape[1]))
                )
                
                embedded_count += 1
                
                if verbose:
                    print(f"  Frame {frame_count}: ✓ EMBEDDED ({embedded_count} total)")
                    
            except Exception as e:
                if verbose:
                    print(f"  Frame {frame_count}: ✗ FAILED (embedding error: {e})")
        
        capture.release()
        
        if verbose:
            print(f"\n✓ First pass complete!")
            print(f"  Total frames: {frame_count}")
            print(f"  Frames checked: {processed_count}")
            print(f"  Frames embedded: {embedded_count}")
            print(f"  Cache hits: {cache_hits} ({100 * cache_hits / embedded_count if embedded_count > 0 else 0:.1f}%)")
            print(f"  New embeddings: {embedded_count - cache_hits}")
            if embedded_crop_hw:
                crop_h = [h for h, _ in embedded_crop_hw]
                crop_w = [w for _, w in embedded_crop_hw]
                print(
                    f"Embedded hand crops ({video_name}): count={len(embedded_crop_hw)}, "
                    f"video_frame_HW={source_frame_hw}, "
                    f"crop_H={min(crop_h)}-{max(crop_h)}, crop_W={min(crop_w)}-{max(crop_w)}, "
                    f"crop_HW_unique={sorted(set(embedded_crop_hw))}"
                )
        
        if len(all_embeddings) == 0:
            raise ValueError(f"No frames could be embedded from video {video_path}!")
        
        # Run state detection if provided
        state_results = pd.DataFrame()
        if state_detection_func is not None:
            if verbose:
                print(f"\nRunning state detection...")
            state_results = state_detection_func(
                video_path=video_path,
                embeddings=all_embeddings,
                frame_numbers=all_frame_numbers,
                fps=fps
            )
            if verbose:
                print(f"  States detected: {len(state_results)} segments")
        
        # Cache every first-pass embedding (carry windows already enforced above when configured)
        embeddings = []
        labels = []
        synthetic_paths = []
        embedding_frame_indices: List[int] = []
        cached_count = 0
        newly_cached = 0

        if verbose:
            print(f"\nVerifying cached frames...")

        cap = cv2.VideoCapture(video_path)
        for embedding, frame_num in zip(all_embeddings, all_frame_numbers):
            # Check if already cached (we just loaded it)
            existing_cache = load_frame_from_cache(label, frame_num, cache_dir)
            if existing_cache is None:
                # Need to cache this new embedding
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                ret, frame = cap.read()
                if ret:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    crop_full, _crop_proc = _prepare_hand_crops_for_embedding(
                        image_rgb, hands_detector
                    )
                    if crop_full is not None and crop_full.size > 0:
                        save_frame_to_cache(
                            label, frame_num, cache_dir, embedding, crop_full
                        )
                        newly_cached += 1
            
            embeddings.append(embedding)
            labels.append(label)
            synthetic_paths.append(f"{label}_frame_{frame_num}")
            embedding_frame_indices.append(frame_num)
            cached_count += 1
        cap.release()
        
        if verbose:
            print(f"\n✓ Video processing complete!")
            print(f"  Total frames: {frame_count}")
            print(f"  Frames embedded: {embedded_count}")
            print(f"  Frames returned: {cached_count}")
            print(f"  Newly cached: {newly_cached}")
            print(f"  Label: {label}")
        
        if len(embeddings) == 0:
            raise ValueError(f"No frames could be cached from video {video_path}!")
        
        return embeddings, labels, synthetic_paths, state_results, embedding_frame_indices
    
    finally:
        # Always close the MediaPipe detector to release resources
        hands_detector.close()


__all__ = ['process_video_frames']
