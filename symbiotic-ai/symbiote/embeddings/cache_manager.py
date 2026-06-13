"""Embedding cache management utilities."""

import os
import hashlib
from typing import Optional, Tuple
import numpy as np


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


__all__ = [
    'get_cache_path',
    'load_from_cache',
    'save_to_cache',
    'save_frame_to_cache'
]
