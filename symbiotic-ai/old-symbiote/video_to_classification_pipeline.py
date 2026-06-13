#!/usr/bin/env python3
"""
Video-to-Classification Pipeline

This module trains a lightweight classifier on CLIP embeddings extracted from video frames.
The CLIP encoder is frozen - only a small classifier head is trained.

Features:
- Extracts non-blurry frames from video using Laplacian variance
- Directly embeds frames without saving them to disk
- Uses caching system to accumulate embeddings across multiple video runs
- Train/validation/test split with stratification
- Early stopping based on validation loss
- Confusion matrix and detailed evaluation metrics
- Model persistence and inference API

Usage:
    # Train from video
    python video_to_classification_pipeline.py train --video path/to/video.mp4 --label "object_name"
    
    # Train with custom parameters
    python video_to_classification_pipeline.py train --video path/to/video.mp4 --label "object_name" --threshold 150.0 --frame-skip 4
"""

import argparse
import sys
import os
import numpy as np
import cv2
from transformers import AutoModel, AutoProcessor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
import random
import hashlib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Suppress MediaPipe warnings
os.environ["GLOG_minloglevel"] = "2"

# Add lib folder to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

from embedding import MODEL
from hand_detection import segment_hand

# Optional: HEIC support (Apple image format). Install with: pip install pillow-heif
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAS_HEIC = True
except ImportError:
    HAS_HEIC = False


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "batch_size": 16,
    "learning_rate": 0.001,
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "hidden_dim": 128,  # Hidden layer size for MLP classifier
    "dropout": 0.3,
    "random_seed": 42,
}


# ============================================================================
# BLUR DETECTION (from blurry.py)
# ============================================================================

def is_blurry(image: np.ndarray, threshold: float) -> bool:
    """
    Check if an image is blurry using Laplacian variance.
    
    Uses the blur detection method from blurry.py:
    https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    
    Args:
        image: RGB or BGR image as numpy array
        threshold: Laplacian variance threshold (lower = more blurry)
    
    Returns:
        True if image is blurry (variance < threshold), False otherwise
    """
    # Convert to grayscale for Laplacian computation
    # Handle both RGB and BGR inputs
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image[:,:,0]
    else:
        gray = image
    
    # Compute Laplacian variance
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


# ============================================================================
# IMAGE LOADING (supports JPG, PNG, and HEIC when pillow-heif is installed)
# ============================================================================

def load_image_as_rgb(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image as RGB numpy array.
    
    Supports: JPG, JPEG, PNG (via OpenCV).
    Supports: HEIC (via Pillow + pillow-heif if installed).
    
    Returns:
        RGB image as numpy array (H, W, 3), or None if load failed
    """
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext == '.heic':
        if not HAS_HEIC:
            return None
        try:
            pil_img = Image.open(image_path)
            pil_img = pil_img.convert('RGB')
            return np.array(pil_img)
        except Exception:
            return None
    
    # JPG, PNG, etc. via OpenCV
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ============================================================================
# EMBEDDING CACHE (reuse embeddings across runs for faster training)
# ============================================================================

def get_cache_path(image_path: str, cache_dir: str, file_type: str = "embedding") -> str:
    """
    Generate a unique cache path for an image.
    
    Uses filename + modification time so cache is invalidated when the image changes.
    """
    filename = os.path.basename(image_path)
    mtime = os.path.getmtime(image_path)
    cache_key = hashlib.md5(f"{filename}_{mtime}".encode()).hexdigest()
    if file_type == "embedding":
        cache_filename = f"{os.path.splitext(filename)[0]}_{cache_key}.npy"
    else:
        cache_filename = f"{os.path.splitext(filename)[0]}_{cache_key}_seg.npy"
    return os.path.join(cache_dir, cache_filename)


def load_from_cache(image_path: str, cache_dir: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load embedding and segmented image from cache if available."""
    try:
        emb_cache_path = get_cache_path(image_path, cache_dir, "embedding")
        seg_cache_path = get_cache_path(image_path, cache_dir, "segmented")
        if os.path.exists(emb_cache_path) and os.path.exists(seg_cache_path):
            embedding = np.load(emb_cache_path)
            segmented = np.load(seg_cache_path)
            return (embedding, segmented)
    except Exception:
        pass
    return None


def save_to_cache(image_path: str, cache_dir: str, embedding: np.ndarray, segmented: np.ndarray):
    """Save embedding and segmented image to cache."""
    try:
        np.save(get_cache_path(image_path, cache_dir, "embedding"), embedding)
        np.save(get_cache_path(image_path, cache_dir, "segmented"), segmented)
    except Exception:
        pass


def save_frame_to_cache(label: str, frame_number: int, cache_dir: str, embedding: np.ndarray, segmented: np.ndarray):
    """
    Save embedding and segmented frame to cache for a video frame.
    
    Creates a synthetic filename based on label and frame number.
    The label is embedded in the filename so it can be extracted when loading.
    
    Args:
        label: Class label for this frame
        frame_number: Frame number in the video
        cache_dir: Directory to save cache files
        embedding: Embedding array to save
        segmented: Segmented image array to save
    """
    try:
        # Create synthetic filename with label embedded
        # Format: label_frame_N_hash.npy
        cache_key = hashlib.md5(f"{label}_frame_{frame_number}".encode()).hexdigest()
        emb_cache_filename = f"{label}_frame_{frame_number}_{cache_key}.npy"
        seg_cache_filename = f"{label}_frame_{frame_number}_{cache_key}_seg.npy"
        
        emb_cache_path = os.path.join(cache_dir, emb_cache_filename)
        seg_cache_path = os.path.join(cache_dir, seg_cache_filename)
        
        np.save(emb_cache_path, embedding)
        np.save(seg_cache_path, segmented)
    except Exception:
        pass


# ============================================================================
# EMBEDDING FUNCTIONS (Reused from batch_compare.py)
# ============================================================================

def embed_image(
    image_path: str,
    model: AutoModel,
    processor: AutoProcessor,
    cache_dir: Optional[str] = None,
    verbose: bool = False
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Embed an image using CLIP, after hand segmentation.
    
    This function reuses the exact same pipeline as batch_compare.py.
    When cache_dir is set, loads from cache if available and saves after embedding.
    
    Args:
        image_path: Path to the image file
        model: CLIP model
        processor: CLIP processor
        cache_dir: If set, use this directory to cache embeddings (faster subsequent runs)
        verbose: Whether to print detailed errors
    
    Returns:
        Tuple of (embedding, segmented_image) or None if failed
    """
    # Try to load from cache first
    if cache_dir:
        cached = load_from_cache(image_path, cache_dir)
        if cached is not None:
            return cached

    try:
        # Load image (supports JPG, PNG, HEIC with pillow-heif)
        image_rgb = load_image_as_rgb(image_path)
        if image_rgb is None:
            if verbose:
                ext = os.path.splitext(image_path)[1].lower()
                if ext == '.heic' and not HAS_HEIC:
                    print(f"    Skipped (HEIC requires pillow-heif): {os.path.basename(image_path)}")
                else:
                    print(f"    Could not load: {os.path.basename(image_path)}")
            return None
        
        # Segment hand (same as batch_compare.py)
        segmented = segment_hand(image_rgb)
        if segmented is None:
            if verbose:
                print(f"    Hand not detected: {os.path.basename(image_path)}")
            return None
        
        if segmented.size == 0:
            if verbose:
                print(f"    Empty segmentation: {os.path.basename(image_path)}")
            return None
        
        # Process with CLIP
        inputs = processor(images=[segmented], return_tensors="pt").to(model.device)
        
        # Generate embedding (frozen CLIP)
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
        embedding = embeddings.cpu().numpy()[0]
        
        # Save to cache for future runs
        if cache_dir:
            save_to_cache(image_path, cache_dir, embedding, segmented)
        
        return (embedding, segmented)
        
    except Exception as e:
        if verbose:
            print(f"    Error embedding {os.path.basename(image_path)}: {e}")
        return None


def embed_frame(
    frame: np.ndarray,
    model: AutoModel,
    processor: AutoProcessor,
    verbose: bool = False
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Embed a video frame using CLIP, after hand segmentation.
    
    Args:
        frame: BGR frame as numpy array
        model: CLIP model
        processor: CLIP processor
        verbose: Whether to print detailed errors
    
    Returns:
        Tuple of (embedding, segmented_image) or None if failed
    """
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Segment hand
        segmented = segment_hand(image_rgb)
        if segmented is None:
            if verbose:
                print(f"    Hand not detected in frame")
            return None
        
        if segmented.size == 0:
            if verbose:
                print(f"    Empty segmentation in frame")
            return None
        
        # Process with CLIP
        inputs = processor(images=[segmented], return_tensors="pt").to(model.device)
        
        # Generate embedding (frozen CLIP)
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
        embedding = embeddings.cpu().numpy()[0]
        
        return (embedding, segmented)
        
    except Exception as e:
        if verbose:
            print(f"    Error embedding frame: {e}")
        return None


def embed_image_for_inference(
    image_path: str,
    model: AutoModel,
    processor: AutoProcessor
) -> Optional[torch.Tensor]:
    """
    Embed an image for inference (returns tensor, no segmented image).
    
    Args:
        image_path: Path to the image file
        model: CLIP model
        processor: CLIP processor
    
    Returns:
        Embedding tensor or None if failed
    """
    result = embed_image(image_path, model, processor, verbose=False)
    if result is None:
        return None
    embedding, _ = result
    return torch.tensor(embedding, dtype=torch.float32)


# ============================================================================
# VIDEO FRAME EXTRACTION AND EMBEDDING
# ============================================================================

def process_video_frames(
    video_path: str,
    label: str,
    model: AutoModel,
    processor: AutoProcessor,
    cache_dir: str,
    threshold: float = 100.0,
    frame_skip: int = 4,
    verbose: bool = True
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Extract non-blurry frames from video and embed them directly.
    
    This function combines the blur detection logic from blurry.py with the
    embedding pipeline. Frames are NOT saved to disk - they are embedded directly
    and cached for future runs.
    
    Args:
        video_path: Path to the video file
        label: Class label for these frames
        model: CLIP model for embedding
        processor: CLIP processor
        cache_dir: Directory to cache embeddings
        threshold: Blur threshold (Laplacian variance, default 100.0)
        frame_skip: Process every Nth frame (default 4)
        verbose: Whether to print progress
    
    Returns:
        Tuple of (embeddings, labels, synthetic_paths) where:
        - embeddings: List of numpy arrays
        - labels: List of label strings (all same)
        - synthetic_paths: List of synthetic paths for tracking (video_name_frame_N)
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
    
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if verbose:
        print(f"Total frames in video: {total_frames}")
        print(f"Processing every {frame_skip} frames...")
    
    embeddings = []
    labels = []
    synthetic_paths = []
    
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
            
            # Save to cache with label embedded in filename
            save_frame_to_cache(label, frame_count, cache_dir, embedding, segmented)
            
            # Add to dataset
            embeddings.append(embedding)
            labels.append(label)
            synthetic_paths.append(f"{label}_frame_{frame_count}")
            
            embedded_count += 1
            
            if verbose:
                print(f"  Frame {frame_count}: ✓ EMBEDDED ({embedded_count} total)")
                
        except Exception as e:
            if verbose:
                print(f"  Frame {frame_count}: ✗ FAILED (embedding error: {e})")
    
    capture.release()
    
    if verbose:
        print(f"\n✓ Video processing complete!")
        print(f"  Total frames: {frame_count}")
        print(f"  Frames checked: {processed_count}")
        print(f"  Frames embedded: {embedded_count}")
        print(f"  Label: {label}")
    
    if len(embeddings) == 0:
        raise ValueError(f"No frames could be embedded from video {video_path}!")
    
    return embeddings, labels, synthetic_paths


# ============================================================================
# DATASET CONSTRUCTION
# ============================================================================

def scan_dataset(
    data_dir: str,
    model: AutoModel,
    processor: AutoProcessor,
    cache_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Recursively scan data directory and build dataset.
    
    Directory structure expected:
        data_dir/
            a/
                image1.jpg
                image2.jpg
            b/
                image3.jpg
            ...
    
    Each subfolder name is treated as a class label (single character).
    
    Args:
        data_dir: Path to data directory
        model: CLIP model for embedding
        processor: CLIP processor
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing:
            - embeddings: List of numpy arrays
            - labels: List of string labels
            - image_paths: List of image paths
            - label_to_idx: Dict mapping label -> numeric index
            - idx_to_label: Dict mapping numeric index -> label
            - embedding_dim: Dimension of embeddings
    """
    if verbose:
        print("\n" + "="*60)
        print("SCANNING DATASET")
        print("="*60)
        print(f"Data directory: {data_dir}")
    
    # Find all subfolders (class labels)
    subfolders = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    
    if len(subfolders) == 0:
        raise ValueError(f"No subfolders found in {data_dir}. Expected folder structure: data_dir/label/images")
    
    # Sort for deterministic ordering
    subfolders = sorted(subfolders)
    
    # Create label mappings
    label_to_idx = {label: idx for idx, label in enumerate(subfolders)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    if verbose:
        print(f"Found {len(subfolders)} classes: {subfolders}")
        print(f"Label mapping: {label_to_idx}")
    
    # Collect all images and embeddings
    embeddings = []
    labels = []
    image_paths = []
    
    # Include HEIC so we can load them when pillow-heif is installed
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.heic', '.HEIC'}
    
    if verbose and not HAS_HEIC:
        print("\nNote: HEIC files will be skipped. Install pillow-heif for HEIC support: pip install pillow-heif")
    if verbose and cache_dir:
        print("\nNote: Using embedding cache - previously embedded images will be loaded from cache")
    
    cached_count = 0
    for label in subfolders:
        folder_path = os.path.join(data_dir, label)
        
        # Find all images in this folder
        image_files = [
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1] in image_extensions
        ]
        
        if verbose:
            print(f"\nProcessing class '{label}': {len(image_files)} images")
        
        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(folder_path, image_file)
            
            is_cached = False
            if cache_dir:
                is_cached = load_from_cache(image_path, cache_dir) is not None
            
            if verbose:
                status = " [CACHED]" if is_cached else ""
                print(f"  [{idx}/{len(image_files)}] {image_file}...{status}", end=" ", flush=True)
            
            result = embed_image(image_path, model, processor, cache_dir=cache_dir, verbose=False)
            
            if result is not None:
                if is_cached:
                    cached_count += 1
                embedding, _ = result
                embeddings.append(embedding)
                labels.append(label)
                image_paths.append(image_path)
                if verbose:
                    print("✓")
            else:
                if verbose:
                    print("✗ SKIPPED")
    
    if len(embeddings) == 0:
        raise ValueError("No images could be embedded successfully!")
    
    embedding_dim = embeddings[0].shape[0]
    
    if verbose:
        print(f"\n✓ Dataset scan complete!")
        print(f"  Total samples: {len(embeddings)}")
        if cached_count > 0:
            print(f"  Loaded from cache: {cached_count} (newly embedded: {len(embeddings) - cached_count})")
        print(f"  Embedding dimension: {embedding_dim}")
        
        # Print class distribution
        class_counts = defaultdict(int)
        for label in labels:
            class_counts[label] += 1
        print(f"  Class distribution:")
        for label, count in sorted(class_counts.items()):
            print(f"    {label}: {count} samples")
    
    return {
        "embeddings": embeddings,
        "labels": labels,
        "image_paths": image_paths,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "embedding_dim": embedding_dim,
    }


def build_image_to_label_mapping(data_dir: str, verbose: bool = False) -> Dict[str, str]:
    """
    Build a mapping of cache keys (basename_hash) to their labels by scanning folder structure.
    
    This is used to migrate old cache files that don't have labels in their filenames.
    The hash is computed the same way as classifier_pipeline.py: md5(filename_mtime)
    
    Args:
        data_dir: Directory containing training data (subfolders = classes)
        verbose: Whether to print progress
    
    Returns:
        Dictionary mapping cache key (basename_hash) -> label
    """
    mapping = {}
    duplicate_basenames = set()
    
    if not os.path.exists(data_dir):
        return mapping
    
    # Find all subfolders (class labels)
    subfolders = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.heic', '.HEIC'}
    
    for label in subfolders:
        folder_path = os.path.join(data_dir, label)
        
        # Find all images in this folder
        try:
            image_files = [
                f for f in os.listdir(folder_path)
                if os.path.splitext(f)[1] in image_extensions
            ]
            
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                basename = os.path.splitext(image_file)[0]
                
                # Compute hash the same way as get_cache_path() in classifier_pipeline.py
                try:
                    mtime = os.path.getmtime(image_path)
                    cache_key_input = f"{image_file}_{mtime}"
                    cache_hash = hashlib.md5(cache_key_input.encode()).hexdigest()
                    
                    # Store mapping with full cache key (basename_hash)
                    cache_key = f"{basename}_{cache_hash}"
                    mapping[cache_key] = label
                    
                    # Also store just basename for simple lookup (will be overwritten if duplicate)
                    if basename in mapping and mapping[basename] != label:
                        duplicate_basenames.add(basename)
                    mapping[basename] = label
                except Exception:
                    continue
        except Exception:
            continue
    
    if verbose:
        print(f"Built image→label mapping with {len(mapping)} entries")
        if duplicate_basenames:
            print(f"  Warning: {len(duplicate_basenames)} duplicate basenames found (using hash for disambiguation)")
    
    return mapping


def load_all_cached_embeddings(
    cache_dir: str,
    image_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load all cached embeddings from the cache directory.
    
    This allows accumulating embeddings across multiple video processing runs.
    Supports both NEW and OLD cache formats:
    - NEW: label_frame_N_hash.npy (video_to_classification_pipeline)
    - OLD: imagename_hash.npy (classifier_pipeline)
    
    For OLD format files, uses image_dir to build filename→label mapping.
    
    Args:
        cache_dir: Directory containing cached embeddings
        image_dir: Directory with image folders (for old cache label lookup), optional
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing:
            - embeddings: List of numpy arrays
            - labels: List of string labels (extracted from filename prefix)
            - synthetic_paths: List of synthetic paths
            - label_to_idx: Dict mapping label -> numeric index
            - idx_to_label: Dict mapping numeric index -> label
            - embedding_dim: Dimension of embeddings
    """
    if verbose:
        print("\n" + "="*60)
        print("LOADING CACHED EMBEDDINGS")
        print("="*60)
        print(f"Cache directory: {cache_dir}")
    
    if not os.path.exists(cache_dir):
        raise ValueError(f"Cache directory does not exist: {cache_dir}")
    
    # Find all embedding cache files (not segmented files)
    all_files = os.listdir(cache_dir)
    if verbose:
        print(f"Total files in cache directory: {len(all_files)}")
    
    cache_files = [
        f for f in all_files
        if f.endswith('.npy') and not f.endswith('_seg.npy')
    ]
    
    if verbose:
        print(f"Embedding files found (*.npy, not *_seg.npy): {len(cache_files)}")
    
    if len(cache_files) == 0:
        raise ValueError(f"No cached embeddings found in {cache_dir}")
    
    # Build image→label mapping for OLD format cache files (if image_dir provided)
    image_to_label = {}
    if image_dir and os.path.exists(image_dir):
        if verbose:
            print(f"Building image→label mapping from: {image_dir}")
        image_to_label = build_image_to_label_mapping(image_dir, verbose=verbose)
    
    # Parse labels from cache files
    # Format: videoname_frame_N_hash.npy -> extract label from videoname
    label_counts = defaultdict(int)
    embeddings = []
    labels = []
    synthetic_paths = []
    
    failed_count = 0
    success_count = 0
    skipped_format_count = 0
    old_format_count = 0
    
    for cache_file in sorted(cache_files):
        try:
            # Load embedding
            emb_path = os.path.join(cache_dir, cache_file)
            embedding = np.load(emb_path)
            
            # Extract label from filename
            # Support TWO formats:
            # 1. NEW format (video_to_classification_pipeline): label_frame_N_hash.npy
            # 2. OLD format (classifier_pipeline): imagename_hash.npy
            
            filename_without_ext = cache_file.replace('.npy', '')
            
            # Try NEW format first (contains '_frame_')
            if '_frame_' in filename_without_ext:
                parts = filename_without_ext.split('_frame_')
                if len(parts) >= 2:
                    video_label = parts[0]  # This is the label
                    frame_info = parts[1].split('_')[0]  # Frame number
                    
                    embeddings.append(embedding)
                    labels.append(video_label)
                    synthetic_paths.append(f"{video_label}_frame_{frame_info}")
                    label_counts[video_label] += 1
                    success_count += 1
                else:
                    skipped_format_count += 1
            else:
                # OLD format: imagename_hash.npy
                # Extract image basename (remove hash)
                parts = filename_without_ext.rsplit('_', 1)  # Split on last underscore to remove hash
                if len(parts) == 2:
                    image_basename = parts[0]
                    image_hash = parts[1]
                    
                    # Try to look up label using full cache key (basename_hash) first for disambiguation
                    cache_key = f"{image_basename}_{image_hash}"
                    label_for_image = None
                    
                    if cache_key in image_to_label:
                        # Exact match with hash - most reliable
                        label_for_image = image_to_label[cache_key]
                    elif image_basename in image_to_label:
                        # Fallback to basename-only lookup
                        label_for_image = image_to_label[image_basename]
                    
                    if label_for_image:
                        embeddings.append(embedding)
                        labels.append(label_for_image)
                        synthetic_paths.append(image_basename)
                        label_counts[label_for_image] += 1
                        success_count += 1
                        old_format_count += 1
                    else:
                        # No label mapping found - skip this file
                        if verbose and skipped_format_count < 5:
                            print(f"  Warning: No label found for {image_basename} (need image_dir parameter)")
                        skipped_format_count += 1
                else:
                    skipped_format_count += 1
                    
        except Exception as e:
            if verbose and failed_count < 10:  # Only show first 10 errors
                print(f"  Warning: Could not load {cache_file}: {e}")
            failed_count += 1
            continue
    
    if verbose:
        print(f"Successfully loaded: {success_count} embeddings")
        if old_format_count > 0:
            print(f"  - Old format (from classifier_pipeline): {old_format_count}")
        if (success_count - old_format_count) > 0:
            print(f"  - New format (from video_to_classification_pipeline): {success_count - old_format_count}")
        if failed_count > 0:
            print(f"Failed to load: {failed_count} files")
        if skipped_format_count > 0:
            print(f"Skipped (no label mapping): {skipped_format_count} files")
    
    if len(embeddings) == 0:
        raise ValueError("No embeddings could be loaded from cache!")
    
    # Create label mappings
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    embedding_dim = embeddings[0].shape[0]
    
    if verbose:
        print(f"✓ Loaded {len(embeddings)} embeddings from cache")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Found {len(unique_labels)} classes:")
        for label in unique_labels:
            print(f"    {label}: {label_counts[label]} samples")
    
    return {
        "embeddings": embeddings,
        "labels": labels,
        "image_paths": synthetic_paths,  # Use synthetic paths as "image_paths"
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "embedding_dim": embedding_dim,
    }


# ============================================================================
# DATASET SPLITTING
# ============================================================================

def stratified_split(
    embeddings: List[np.ndarray],
    labels: List[str],
    image_paths: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, Dict[str, List]]:
    """
    Split dataset into train/val/test with stratification by class.
    
    Ensures each class appears in all splits proportionally.
    
    Args:
        embeddings: List of embedding arrays
        labels: List of string labels
        image_paths: List of image paths
        train_ratio: Fraction for training (default 0.70)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing
        'embeddings', 'labels', 'image_paths' lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Group by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    # Initialize splits
    splits = {
        "train": {"embeddings": [], "labels": [], "image_paths": []},
        "val": {"embeddings": [], "labels": [], "image_paths": []},
        "test": {"embeddings": [], "labels": [], "image_paths": []},
    }
    
    # Stratified split for each class
    for label, indices in class_indices.items():
        # Shuffle indices for this class
        random.shuffle(indices)
        
        n = len(indices)
        # Ensure at least 1 sample in val and test when we have enough (avoids empty val/test)
        if n == 1:
            n_train, n_val, n_test = 1, 0, 0
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            n_test = max(1, int(n * test_ratio))
            # If we over-allocated, give remainder to train
            if n_train + n_val + n_test > n:
                n_train = n - n_val - n_test
                n_train = max(1, n_train)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Add to splits
        for idx in train_indices:
            splits["train"]["embeddings"].append(embeddings[idx])
            splits["train"]["labels"].append(labels[idx])
            splits["train"]["image_paths"].append(image_paths[idx])
        
        for idx in val_indices:
            splits["val"]["embeddings"].append(embeddings[idx])
            splits["val"]["labels"].append(labels[idx])
            splits["val"]["image_paths"].append(image_paths[idx])
        
        for idx in test_indices:
            splits["test"]["embeddings"].append(embeddings[idx])
            splits["test"]["labels"].append(labels[idx])
            splits["test"]["image_paths"].append(image_paths[idx])
    
    return splits


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class EmbeddingDataset(Dataset):
    """PyTorch Dataset for embeddings."""
    
    def __init__(self, embeddings: List[np.ndarray], labels: List[str], label_to_idx: Dict[str, int]):
        self.embeddings = [torch.tensor(e, dtype=torch.float32) for e in embeddings]
        self.labels = [label_to_idx[l] for l in labels]
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ============================================================================
# CLASSIFIER MODEL
# ============================================================================

class ClassifierHead(nn.Module):
    """
    Lightweight classifier head for CLIP embeddings.
    
    Architecture: Linear -> ReLU -> Dropout -> Linear
    
    This is intentionally simple - the heavy lifting is done by CLIP.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


# ============================================================================
# TRAINING
# ============================================================================

def train_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: ClassifierHead,
    config: Dict[str, Any],
    device: str = "cpu",
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train the classifier with early stopping.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model: Classifier model
        config: Configuration dictionary
        device: Device to train on
        verbose: Whether to print progress
    
    Returns:
        Dictionary with training history (train_loss, val_loss, train_acc, val_acc)
    """
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    if verbose:
        print("\n" + "="*60)
        print("TRAINING CLASSIFIER")
        print("="*60)
        print(f"Max epochs: {config['max_epochs']}")
        print(f"Early stopping patience: {config['early_stopping_patience']}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Device: {device}")
        print()
    
    for epoch in range(config["max_epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * embeddings.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device)
                
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * embeddings.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        if verbose:
            print(f"Epoch {epoch+1:3d}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"Restored best model (val_loss={best_val_loss:.4f})")
    
    return history


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_classifier(
    test_loader: DataLoader,
    model: ClassifierHead,
    idx_to_label: Dict[int, str],
    device: str = "cpu",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate classifier on test set.
    
    Args:
        test_loader: Test data loader
        model: Trained classifier model
        idx_to_label: Mapping from index to label
        device: Device to evaluate on
        verbose: Whether to print results
    
    Returns:
        Dictionary with evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings = embeddings.to(device)
            
            outputs = model(embeddings)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels)
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Top-1 accuracy
    top1_acc = (all_preds == all_labels).mean()
    
    # Top-3 accuracy
    num_classes = all_probs.shape[1]
    k = min(3, num_classes)
    top_k_preds = np.argsort(all_probs, axis=1)[:, -k:]
    top3_acc = np.mean([label in preds for label, preds in zip(all_labels, top_k_preds)])
    
    # Confusion matrix (raw counts)
    cm_raw = confusion_matrix(all_labels, all_preds)
    # Normalize by row so each row sums to 1 (proportion of that true class predicted as each class)
    row_sums = np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
    cm = (cm_raw.astype(float) / row_sums)
    
    # Per-class metrics
    labels_list = sorted(idx_to_label.keys())
    label_names = [idx_to_label[i] for i in labels_list]
    
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Test Accuracy (Top-1): {top1_acc:.4f} ({top1_acc*100:.2f}%)")
        print(f"Test Accuracy (Top-{k}): {top3_acc:.4f} ({top3_acc*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=label_names))
        print("\nConfusion Matrix (count and row-normalized %):")
        for i in range(cm.shape[0]):
            row = [f"{int(cm_raw[i, j])} ({cm[i, j]:.2f})" for j in range(cm.shape[1])]
            print("  ", "  ".join(row))
    
    return {
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "confusion_matrix": cm,
        "confusion_matrix_raw": cm_raw,
        "predictions": all_preds,
        "true_labels": all_labels,
        "probabilities": all_probs,
        "label_names": label_names,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    cm_raw: np.ndarray,
    label_names: List[str],
    output_path: str
):
    """Plot and save confusion matrix with both counts and row-normalized % (0-1)."""
    # Build annotations: count on first line, percent on second
    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{int(cm_raw[i, j])}\n({cm[i, j]:.2f})"
    annot_flat = annot.ravel().tolist()
    annot_2d = np.array(annot_flat).reshape(cm.shape)
    plt.figure(figsize=(10, 8))
    if HAS_SEABORN:
        sns.heatmap(
            cm, annot=annot_2d, fmt='', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names,
            vmin=0, vmax=1
        )
    else:
        plt.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        plt.colorbar()
        plt.xticks(np.arange(len(label_names)), label_names)
        plt.yticks(np.arange(len(label_names)), label_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (count and row-normalized %)')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f'{int(cm_raw[i, j])}\n({cm[i, j]:.2f})',
                         ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: str
):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history["train_loss"], label='Train')
    axes[0].plot(history["val_loss"], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history["train_acc"], label='Train')
    axes[1].plot(history["val_acc"], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Training history saved to: {output_path}")


# ============================================================================
# PERSISTENCE
# ============================================================================

def save_model(
    model: ClassifierHead,
    label_to_idx: Dict[str, int],
    idx_to_label: Dict[int, str],
    embedding_dim: int,
    config: Dict[str, Any],
    output_dir: str
):
    """
    Save trained model and metadata.
    
    Saves:
        - model_weights.pth: Classifier weights
        - model_metadata.json: Label mapping, dimensions, config
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    weights_path = os.path.join(output_dir, "model_weights.pth")
    torch.save(model.state_dict(), weights_path)
    
    # Save metadata
    metadata = {
        "label_to_idx": label_to_idx,
        "idx_to_label": {str(k): v for k, v in idx_to_label.items()},  # JSON needs string keys
        "embedding_dim": embedding_dim,
        "num_classes": len(label_to_idx),
        "config": config,
        "clip_model": MODEL,
    }
    
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to: {output_dir}")
    print(f"  - Weights: {weights_path}")
    print(f"  - Metadata: {metadata_path}")


def load_model(model_dir: str, device: str = "cpu") -> Tuple[ClassifierHead, Dict[str, Any]]:
    """
    Load trained model and metadata.
    
    Args:
        model_dir: Directory containing saved model
        device: Device to load model to
    
    Returns:
        Tuple of (model, metadata)
    """
    # Load metadata
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Convert idx_to_label keys back to int
    metadata["idx_to_label"] = {int(k): v for k, v in metadata["idx_to_label"].items()}
    
    # Create model
    config = metadata["config"]
    model = ClassifierHead(
        input_dim=metadata["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=metadata["num_classes"],
        dropout=config["dropout"]
    )
    
    # Load weights
    weights_path = os.path.join(model_dir, "model_weights.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, metadata


# ============================================================================
# INFERENCE API
# ============================================================================

class ObjectRecognizer:
    """
    High-level API for object recognition inference.
    
    Usage:
        recognizer = ObjectRecognizer("path/to/model")
        result = recognizer.predict("path/to/image.jpg")
        print(f"Predicted: {result['label']} (confidence: {result['confidence']:.2f})")
    """
    
    def __init__(self, model_dir: str, device: str = None):
        """
        Initialize the recognizer.
        
        Args:
            model_dir: Directory containing saved model
            device: Device to use (auto-detected if None)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load classifier
        self.model, self.metadata = load_model(model_dir, device)
        
        # Load CLIP model
        print(f"Loading CLIP model ({self.metadata['clip_model']})...")
        self.clip_model = AutoModel.from_pretrained(self.metadata['clip_model'])
        self.clip_model.eval()
        if device == "cuda":
            self.clip_model = self.clip_model.to(device)
        
        self.processor = AutoProcessor.from_pretrained(self.metadata['clip_model'])
        print("Ready for inference!")
    
    def predict(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Predict the object label for an image.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Dictionary with:
                - label: Predicted class label
                - confidence: Confidence score (0-1)
                - all_scores: Dict of all class scores
            Or None if embedding failed
        """
        # Get embedding
        embedding = embed_image_for_inference(image_path, self.clip_model, self.processor)
        
        if embedding is None:
            return None
        
        # Run classifier
        embedding = embedding.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(embedding)
            probs = torch.softmax(output, dim=1)
        
        probs = probs.cpu().numpy()[0]
        
        # Get prediction
        pred_idx = np.argmax(probs)
        pred_label = self.metadata["idx_to_label"][pred_idx]
        confidence = probs[pred_idx]
        
        # Get all scores
        all_scores = {
            self.metadata["idx_to_label"][i]: float(probs[i])
            for i in range(len(probs))
        }
        
        return {
            "label": pred_label,
            "confidence": float(confidence),
            "all_scores": all_scores,
        }
    
    def predict_top_k(self, image_path: str, k: int = 3) -> Optional[List[Tuple[str, float]]]:
        """
        Get top-k predictions for an image.
        
        Args:
            image_path: Path to the image file
            k: Number of top predictions to return
        
        Returns:
            List of (label, confidence) tuples, sorted by confidence
            Or None if embedding failed
        """
        result = self.predict(image_path)
        if result is None:
            return None
        
        sorted_scores = sorted(result["all_scores"].items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:k]


# ============================================================================
# MAIN TRAINING FUNCTION (Video-based)
# ============================================================================

def run_video_training(
    video_path: str,
    label: str,
    base_output_dir: str,
    config: Dict[str, Any],
    threshold: float = 100.0,
    frame_skip: int = 4,
    image_dir: Optional[str] = None,
    verbose: bool = True
):
    """
    Run the complete training pipeline from video frames.
    
    This function:
    1. Extracts non-blurry frames from the video
    2. Embeds frames directly (no disk storage) and caches them
    3. Loads all cached embeddings (from this and previous runs)
    4. Trains a classifier on all accumulated data
    5. Saves model and results to a folder named after the video
    
    Args:
        video_path: Path to the video file
        label: Class label for frames from this video
        base_output_dir: Base directory for output (e.g., ../models/classifier)
        config: Training configuration
        threshold: Blur detection threshold (default 100.0)
        frame_skip: Process every Nth frame (default 4)
        image_dir: Directory with image folders for old cache label lookup (optional)
        verbose: Whether to print progress
    """
    # Set random seeds
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create base output directory
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create cache directory (shared across all videos)
    cache_dir = os.path.join(base_output_dir, ".cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Embedding cache: {cache_dir}")
    
    # Load CLIP model
    print("="*60)
    print("LOADING CLIP MODEL")
    print("="*60)
    print(f"Model: {MODEL}")
    
    clip_model = AutoModel.from_pretrained(MODEL)
    clip_model.eval()  # Freeze CLIP
    if device == "cuda":
        clip_model = clip_model.to(device)
    
    processor = AutoProcessor.from_pretrained(MODEL)
    print(f"✓ CLIP model loaded (device: {device})")
    
    # Process video frames and add to cache
    video_embeddings, video_labels, video_paths = process_video_frames(
        video_path, label, clip_model, processor, cache_dir,
        threshold=threshold, frame_skip=frame_skip, verbose=verbose
    )
    
    if verbose:
        print("\n" + "="*60)
        print("SUMMARY: VIDEO PROCESSING")
        print("="*60)
        print(f"✓ Successfully processed {len(video_embeddings)} frames from video")
        print(f"✓ All frames cached and labeled as: '{label}'")
        print(f"✓ Frames added to cache (will accumulate with existing data)")
    
    # Load ALL cached embeddings (including from previous videos and old classifier_pipeline cache)
    dataset = load_all_cached_embeddings(cache_dir, image_dir=image_dir, verbose=verbose)
    
    if verbose:
        print("\n" + "="*60)
        print("DATA ACCUMULATION INFO")
        print("="*60)
        print(f"Total training samples across all labels: {len(dataset['embeddings'])}")
        print(f"Breakdown by label:")
        label_counts = defaultdict(int)
        for lbl in dataset['labels']:
            label_counts[lbl] += 1
        for lbl in sorted(label_counts.keys()):
            is_current = " ← CURRENT VIDEO" if lbl == label else ""
            print(f"  - {lbl}: {label_counts[lbl]} samples{is_current}")
        print(f"\nNote: If you run this again with label '{label}', more frames will be ADDED (not replaced)")
    
    # Create output directory named after the video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(base_output_dir, video_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Split dataset
    print("\n" + "="*60)
    print("SPLITTING DATASET (FROM ENTIRE CACHE)")
    print("="*60)
    print(f"Using ALL {len(dataset['embeddings'])} cached samples for train/val/test split")
    print(f"Split ratios: Train={config['train_ratio']}, Val={config['val_ratio']}, Test={config['test_ratio']}")
    
    splits = stratified_split(
        dataset["embeddings"],
        dataset["labels"],
        dataset["image_paths"],
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        random_seed=config["random_seed"]
    )
    
    print(f"\nSplit results:")
    print(f"  Train: {len(splits['train']['labels'])} samples")
    print(f"  Val:   {len(splits['val']['labels'])} samples")
    print(f"  Test:  {len(splits['test']['labels'])} samples")
    
    # Show breakdown by label in each split
    if verbose:
        print(f"\nPer-label breakdown in TEST set:")
        test_label_counts = defaultdict(int)
        for lbl in splits['test']['labels']:
            test_label_counts[lbl] += 1
        for lbl in sorted(test_label_counts.keys()):
            print(f"  - {lbl}: {test_label_counts[lbl]} test samples")
        
        print(f"\nPer-label breakdown in TRAIN set:")
        train_label_counts = defaultdict(int)
        for lbl in splits['train']['labels']:
            train_label_counts[lbl] += 1
        for lbl in sorted(train_label_counts.keys()):
            print(f"  - {lbl}: {train_label_counts[lbl]} train samples")
    
    # Create data loaders
    train_dataset = EmbeddingDataset(
        splits["train"]["embeddings"],
        splits["train"]["labels"],
        dataset["label_to_idx"]
    )
    val_dataset = EmbeddingDataset(
        splits["val"]["embeddings"],
        splits["val"]["labels"],
        dataset["label_to_idx"]
    )
    test_dataset = EmbeddingDataset(
        splits["test"]["embeddings"],
        splits["test"]["labels"],
        dataset["label_to_idx"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Create classifier
    num_classes = len(dataset["label_to_idx"])
    classifier = ClassifierHead(
        input_dim=dataset["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=num_classes,
        dropout=config["dropout"]
    )
    
    print(f"\nClassifier architecture:")
    print(f"  Input:  {dataset['embedding_dim']} (CLIP embedding)")
    print(f"  Hidden: {config['hidden_dim']}")
    print(f"  Output: {num_classes} classes")
    
    # Train
    history = train_classifier(
        train_loader, val_loader, classifier,
        config, device=device, verbose=verbose
    )
    
    # Evaluate on test set (from ALL cached data)
    if verbose:
        print("\n" + "="*60)
        print("EVALUATING ON TEST SET (ALL CACHED DATA)")
        print("="*60)
        print(f"Test set contains data from ALL labels in cache, not just current video")
    
    eval_results = evaluate_classifier(
        test_loader, classifier,
        dataset["idx_to_label"],
        device=device, verbose=verbose
    )
    
    # Save model
    save_model(
        classifier,
        dataset["label_to_idx"],
        dataset["idx_to_label"],
        dataset["embedding_dim"],
        config,
        output_dir
    )
    
    # Save plots
    plot_training_history(history, os.path.join(output_dir, "training_history.png"))
    plot_confusion_matrix(
        eval_results["confusion_matrix"],
        eval_results["confusion_matrix_raw"],
        eval_results["label_names"],
        os.path.join(output_dir, "confusion_matrix.png")
    )
    
    # Save evaluation results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "top1_accuracy": eval_results["top1_accuracy"],
            "top3_accuracy": eval_results["top3_accuracy"],
            "num_test_samples": len(splits["test"]["labels"]),
            "num_classes": num_classes,
            "class_labels": eval_results["label_names"],
            "video_processed": video_name,
            "label": label,
            "frames_embedded": len(video_embeddings),
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"✓ Processed {len(video_embeddings)} frames from video '{video_name}'")
    print(f"✓ Trained model on {len(dataset['embeddings'])} total samples across {num_classes} classes")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - model_weights.pth")
    print(f"  - model_metadata.json")
    print(f"  - training_history.png")
    print(f"  - confusion_matrix.png")
    print(f"  - evaluation_results.json")
    print(f"\nFinal Test Accuracy: {eval_results['top1_accuracy']*100:.2f}%")
    print(f"\n💡 TIP: Run again with a different video/label to ADD more training data!")
    
    return classifier, eval_results


# ============================================================================
# LEGACY TRAINING FUNCTION (kept for compatibility)
# ============================================================================

def run_training(
    data_dir: str,
    output_dir: str,
    config: Dict[str, Any],
    verbose: bool = True,
    use_cache: bool = True
):
    """
    Run the complete training pipeline.
    
    Args:
        data_dir: Directory containing training data (subfolders = classes)
        output_dir: Directory to save model and results
        config: Training configuration
        verbose: Whether to print progress
        use_cache: If True, cache embeddings in output_dir/.cache for faster subsequent runs
    """
    # Set random seeds
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create cache directory (embedding cache for faster re-runs)
    cache_dir = None
    if use_cache:
        cache_dir = os.path.join(output_dir, ".cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Embedding cache: {cache_dir}")
    elif verbose:
        print("Embedding cache: disabled (--no-cache)")
    
    # Load CLIP model
    print("="*60)
    print("LOADING CLIP MODEL")
    print("="*60)
    print(f"Model: {MODEL}")
    
    clip_model = AutoModel.from_pretrained(MODEL)
    clip_model.eval()  # Freeze CLIP
    if device == "cuda":
        clip_model = clip_model.to(device)
    
    processor = AutoProcessor.from_pretrained(MODEL)
    print(f"✓ CLIP model loaded (device: {device})")
    
    # Scan dataset and build embeddings (uses cache when enabled)
    dataset = scan_dataset(data_dir, clip_model, processor, cache_dir=cache_dir, verbose=verbose)
    
    # Split dataset
    print("\n" + "="*60)
    print("SPLITTING DATASET")
    print("="*60)
    
    splits = stratified_split(
        dataset["embeddings"],
        dataset["labels"],
        dataset["image_paths"],
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        random_seed=config["random_seed"]
    )
    
    print(f"Train: {len(splits['train']['labels'])} samples")
    print(f"Val:   {len(splits['val']['labels'])} samples")
    print(f"Test:  {len(splits['test']['labels'])} samples")
    
    # Create data loaders
    train_dataset = EmbeddingDataset(
        splits["train"]["embeddings"],
        splits["train"]["labels"],
        dataset["label_to_idx"]
    )
    val_dataset = EmbeddingDataset(
        splits["val"]["embeddings"],
        splits["val"]["labels"],
        dataset["label_to_idx"]
    )
    test_dataset = EmbeddingDataset(
        splits["test"]["embeddings"],
        splits["test"]["labels"],
        dataset["label_to_idx"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Create classifier
    num_classes = len(dataset["label_to_idx"])
    classifier = ClassifierHead(
        input_dim=dataset["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=num_classes,
        dropout=config["dropout"]
    )
    
    print(f"\nClassifier architecture:")
    print(f"  Input:  {dataset['embedding_dim']} (CLIP embedding)")
    print(f"  Hidden: {config['hidden_dim']}")
    print(f"  Output: {num_classes} classes")
    
    # Train
    history = train_classifier(
        train_loader, val_loader, classifier,
        config, device=device, verbose=verbose
    )
    
    # Evaluate
    eval_results = evaluate_classifier(
        test_loader, classifier,
        dataset["idx_to_label"],
        device=device, verbose=verbose
    )
    
    # Save model
    save_model(
        classifier,
        dataset["label_to_idx"],
        dataset["idx_to_label"],
        dataset["embedding_dim"],
        config,
        output_dir
    )
    
    # Save plots
    plot_training_history(history, os.path.join(output_dir, "training_history.png"))
    plot_confusion_matrix(
        eval_results["confusion_matrix"],
        eval_results["confusion_matrix_raw"],
        eval_results["label_names"],
        os.path.join(output_dir, "confusion_matrix.png")
    )
    
    # Save evaluation results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "top1_accuracy": eval_results["top1_accuracy"],
            "top3_accuracy": eval_results["top3_accuracy"],
            "num_test_samples": len(splits["test"]["labels"]),
            "num_classes": num_classes,
            "class_labels": eval_results["label_names"],
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"  - model_weights.pth")
    print(f"  - model_metadata.json")
    print(f"  - training_history.png")
    print(f"  - confusion_matrix.png")
    print(f"  - evaluation_results.json")
    print(f"\nFinal Test Accuracy: {eval_results['top1_accuracy']*100:.2f}%")
    
    return classifier, eval_results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Video-to-Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from video (basic)
  python video_to_classification_pipeline.py train --video ../videos/object1.mp4 --label "object1"
  
  # Train with custom parameters
  python video_to_classification_pipeline.py train --video ../videos/object1.mp4 --label "object1" --threshold 150.0 --frame-skip 6
  
  # Predict on a single image
  python video_to_classification_pipeline.py predict --model-dir ../models/classifier/video_name --image ../images/test.jpg
  
  # Get top-3 predictions
  python video_to_classification_pipeline.py predict --model-dir ../models/classifier/video_name --image ../images/test.jpg --top-k 3
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a classifier from video")
    train_parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file to process"
    )
    train_parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Class label for frames from this video"
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="../models/classifier",
        help="Base directory to save model and results (subfolder will be created for video)"
    )
    train_parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Blur detection threshold (Laplacian variance, default 100.0)"
    )
    train_parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Process every Nth frame (default 4)"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["max_epochs"],
        help="Maximum number of training epochs"
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_CONFIG["early_stopping_patience"],
        help="Early stopping patience"
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
        help="Learning rate"
    )
    train_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_CONFIG["hidden_dim"],
        help="Hidden layer dimension"
    )
    train_parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show detailed progress"
    )
    train_parser.add_argument(
        "--image-dir",
        type=str,
        default="../images/image-testing",
        help="Directory with image folders (for loading old cache files from classifier_pipeline)"
    )
    train_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache (re-embed all images every run)"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict on a single image")
    predict_parser.add_argument(
        "--model-dir",
        type=str,
        default="../models/classifier",
        help="Directory containing trained model"
    )
    predict_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image to classify"
    )
    predict_parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top predictions to show"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.command == "train":
        video_path = args.video
        if not os.path.isabs(video_path):
            video_path = os.path.normpath(os.path.join(script_dir, video_path))
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        
        base_output_dir = os.path.normpath(os.path.join(script_dir, args.output_dir))
        
        # Resolve image_dir path
        image_dir = None
        if args.image_dir:
            image_dir = args.image_dir
            if not os.path.isabs(image_dir):
                image_dir = os.path.normpath(os.path.join(script_dir, image_dir))
        
        # Build config
        config = DEFAULT_CONFIG.copy()
        config["max_epochs"] = args.epochs
        config["early_stopping_patience"] = args.patience
        config["learning_rate"] = args.lr
        config["hidden_dim"] = args.hidden_dim
        
        run_video_training(
            video_path=video_path,
            label=args.label,
            base_output_dir=base_output_dir,
            config=config,
            threshold=args.threshold,
            frame_skip=args.frame_skip,
            image_dir=image_dir,
            verbose=args.verbose
        )
        
    elif args.command == "predict":
        model_dir = os.path.normpath(os.path.join(script_dir, args.model_dir))
        image_path = args.image
        
        if not os.path.exists(image_path):
            # Try relative to script dir
            image_path = os.path.normpath(os.path.join(script_dir, args.image))
        
        if not os.path.exists(image_path):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        
        # Load recognizer
        recognizer = ObjectRecognizer(model_dir)
        
        # Run prediction
        if args.top_k == 1:
            result = recognizer.predict(image_path)
            if result is None:
                print("Error: Could not process image (hand detection may have failed)")
                sys.exit(1)
            
            print(f"\nPrediction for: {os.path.basename(image_path)}")
            print(f"  Label: {result['label']}")
            print(f"  Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        else:
            results = recognizer.predict_top_k(image_path, k=args.top_k)
            if results is None:
                print("Error: Could not process image (hand detection may have failed)")
                sys.exit(1)
            
            print(f"\nTop-{args.top_k} predictions for: {os.path.basename(image_path)}")
            for rank, (label, conf) in enumerate(results, 1):
                print(f"  {rank}. {label}: {conf:.4f} ({conf*100:.2f}%)")


if __name__ == "__main__":
    main()
