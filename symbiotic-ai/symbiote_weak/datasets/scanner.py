"""Dataset scanning and loading utilities."""

import os
import hashlib
from typing import Dict, Any, Optional
from collections import defaultdict
import numpy as np
from transformers import AutoModel, AutoProcessor

from ..embeddings.clip_embedder import embed_image
from ..embeddings.cache_manager import load_from_cache
from ..preprocessing.image_loader import HAS_HEIC


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
        cache_dir: Optional directory to cache embeddings
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


__all__ = ['scan_dataset', 'build_image_to_label_mapping', 'load_all_cached_embeddings']
