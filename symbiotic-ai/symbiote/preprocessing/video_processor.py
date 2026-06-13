"""Video frame extraction and processing utilities."""

import os
import sys
from typing import List, Tuple, Optional, Set, Callable
import cv2
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor

# Add lib folder to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from hand_detection import segment_hand

from .blur_detection import is_blurry


def process_video_frames(
    video_path: str,
    label: str,
    model: AutoModel,
    processor: AutoProcessor,
    cache_dir: str,
    save_frame_to_cache_func,  # Function to save frame cache
    threshold: float = 100.0,
    frame_skip: int = 4,
    state_filter: Optional[Set[str]] = None,
    state_detection_func: Optional[Callable] = None,
    verbose: bool = True
) -> Tuple[List[np.ndarray], List[str], List[str], pd.DataFrame]:
    """
    Extract non-blurry frames from video and embed them directly.
    
    This function combines the blur detection logic from blurry.py with the
    embedding pipeline. Frames are NOT saved to disk - they are embedded directly
    and cached for future runs.
    
    State detection can optionally filter which frames are cached based on detected
    hand states (e.g., only cache frames where hand is carrying an object).
    
    Args:
        video_path: Path to the video file
        label: Class label for these frames
        model: CLIP model for embedding
        processor: CLIP processor
        cache_dir: Directory to cache embeddings
        save_frame_to_cache_func: Function to save frame to cache
        threshold: Blur threshold (Laplacian variance, default 100.0)
        frame_skip: Process every Nth frame (default 4)
        state_filter: Optional set of state strings to filter by (e.g., {"CARRY_WITH"})
        state_detection_func: Optional function to detect states from video
        verbose: Whether to print progress
    
    Returns:
        Tuple of (embeddings, labels, synthetic_paths, state_results) where:
        - embeddings: List of numpy arrays
        - labels: List of label strings (all same)
        - synthetic_paths: List of synthetic paths for tracking (video_name_frame_N)
        - state_results: DataFrame with state detection results
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
        if state_detection_func is not None:
            print(f"State detection: ENABLED")
            if state_filter:
                print(f"State filter: {state_filter}")
    
    # First pass: extract all valid embeddings
    all_embeddings = []
    all_frame_numbers = []
    
    frame_count = 0
    processed_count = 0
    embedded_count = 0
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames according to frame_skip
        if frame_count % frame_skip != 0:
            continue
        
        processed_count += 1
        
        # First, segment the hand
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        segmented = segment_hand(image_rgb)
        
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
    
    # Filter embeddings by state if filter is provided
    embeddings = []
    labels = []
    synthetic_paths = []
    cached_count = 0
    
    if state_filter is not None and state_detection_func is not None:
        # Filter based on state detection
        if verbose:
            print(f"\nFiltering frames by state: {state_filter}")
        
        for i, (embedding, frame_num) in enumerate(zip(all_embeddings, all_frame_numbers)):
            # Check if this frame is in an allowed state
            frame_time = frame_num / fps if fps > 0 else 0.0
            
            # Find which state segment this frame belongs to
            in_allowed_state = False
            for _, row in state_results.iterrows():
                if row['timestamp_start'] <= frame_time <= row['timestamp_end']:
                    if row['state'] in state_filter:
                        in_allowed_state = True
                        break
            
            if in_allowed_state:
                # Re-open video to get the frame for caching
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                ret, frame = cap.read()
                if ret:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    segmented = segment_hand(image_rgb)
                    if segmented is not None:
                        # Cache this frame
                        save_frame_to_cache_func(label, frame_num, cache_dir, embedding, segmented)
                        embeddings.append(embedding)
                        labels.append(label)
                        synthetic_paths.append(f"{label}_frame_{frame_num}")
                        cached_count += 1
                cap.release()
    else:
        # No filtering - cache all embeddings
        if verbose:
            print(f"\nNo state filtering - caching all frames...")
        
        # Re-open video to cache all frames
        cap = cv2.VideoCapture(video_path)
        for embedding, frame_num in zip(all_embeddings, all_frame_numbers):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            ret, frame = cap.read()
            if ret:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                segmented = segment_hand(image_rgb)
                if segmented is not None:
                    save_frame_to_cache_func(label, frame_num, cache_dir, embedding, segmented)
                    embeddings.append(embedding)
                    labels.append(label)
                    synthetic_paths.append(f"{label}_frame_{frame_num}")
                    cached_count += 1
        cap.release()
    
    if verbose:
        print(f"\n✓ Video processing complete!")
        print(f"  Total frames: {frame_count}")
        print(f"  Frames embedded: {embedded_count}")
        print(f"  Frames cached: {cached_count}")
        print(f"  Label: {label}")
    
    if len(embeddings) == 0:
        raise ValueError(f"No frames passed state filtering from video {video_path}!")
    
    return embeddings, labels, synthetic_paths, state_results


__all__ = ['process_video_frames']
