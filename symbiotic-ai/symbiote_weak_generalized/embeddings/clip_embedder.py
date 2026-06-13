"""CLIP embedding generation utilities."""

import os
import sys
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from ..lib.hand_detection import segment_hand

from ..preprocessing.image_loader import load_image_as_rgb, HAS_HEIC
from .cache_manager import load_from_cache, save_to_cache


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


__all__ = ['embed_image', 'embed_frame', 'embed_image_for_inference']
