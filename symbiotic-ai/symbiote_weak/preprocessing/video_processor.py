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

from ..lib.hand_detection import segment_hand
from ..embeddings.cache_manager import load_frame_from_cache, save_frame_to_cache
from .blur_detection import is_blurry


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
    
    This function combines the blur detection logic from blurry.py with the
    embedding pipeline. Frames are NOT saved to disk - they are embedded directly
    and cached for future runs. Cache is checked before re-embedding.
    
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
                cache_hits += 1
                embedded_count += 1
                if verbose and embedded_count % 20 == 0:
                    print(f"  Frame {frame_count}: ✓ CACHED ({cache_hits} cached, {embedded_count - cache_hits} new)")
                continue
            
            # Convert to RGB and downscale if needed
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Downscale 4K (or larger) frames to 1080p so MediaPipe detects
            # the hand reliably regardless of source camera resolution.
            h_orig, w_orig = image_rgb.shape[:2]
            if w_orig > 1920 or h_orig > 1080:
                scale = min(1920 / w_orig, 1080 / h_orig)
                image_rgb = cv2.resize(
                    image_rgb,
                    (int(w_orig * scale), int(h_orig * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            
            # Segment the hand (pass the reusable detector)
            segmented = segment_hand(image_rgb, hands_detector)
            
            if segmented is None:
                if verbose:
                    print(f"  Frame {frame_count}: ✗ FAILED (hand not detected)")
                continue
            
            if segmented.size == 0:
                if verbose:
                    print(f"  Frame {frame_count}: ✗ FAILED (empty segmentation)")
                continue
            
            # Now check if the SEGMENTED HAND is blurry
            if is_blurry(segmented, threshold):
                if verbose:
                    print(f"  Frame {frame_count}: BLURRY HAND (skipped)")
                continue
            
            # Segmented hand is not blurry - embed it
            try:
                # Process with CLIP
                inputs = processor(images=[segmented], return_tensors="pt").to(model.device)
                
                # Generate embedding (frozen CLIP)
                with torch.no_grad():
                    embeddings_tensor = model.get_image_features(**inputs)
                embedding = embeddings_tensor.cpu().numpy()[0]
                
                # Collect all embeddings for state detection
                all_embeddings.append(embedding)
                all_frame_numbers.append(frame_count)
                
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
                    h_orig, w_orig = image_rgb.shape[:2]
                    if w_orig > 1920 or h_orig > 1080:
                        scale = min(1920 / w_orig, 1080 / h_orig)
                        image_rgb = cv2.resize(
                            image_rgb,
                            (int(w_orig * scale), int(h_orig * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                    segmented = segment_hand(image_rgb, hands_detector)
                    if segmented is not None:
                        save_frame_to_cache(label, frame_num, cache_dir, embedding, segmented)
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
