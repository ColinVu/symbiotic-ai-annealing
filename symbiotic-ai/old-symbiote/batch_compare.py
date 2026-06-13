#!/usr/bin/env python3
"""
Batch image comparison tool for testing all images in a directory.

This script:
1. Embeds all images in the test directory
2. Compares each image with all others using cosine similarity
3. Generates visual comparison files
4. Generates CSV with all similarity scores

Usage:
    python batch_compare.py [--input-dir <path>] [--verbose]
"""

import argparse
import sys
import os
import numpy as np
import cv2
from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image, ImageDraw, ImageFont
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pickle
import hashlib

# Suppress MediaPipe warnings
os.environ["GLOG_minloglevel"] = "2"

# Add lib folder to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

from embedding import MODEL
from hand_detection import segment_hand


def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    similarity = np.dot(embedding1_norm, embedding2_norm)
    return float(similarity)


def calculate_cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine distance between two embeddings."""
    return 1 - calculate_cosine_similarity(embedding1, embedding2)


def get_cache_path(image_path: str, cache_dir: str, file_type: str = "embedding") -> str:
    """
    Generate a unique cache path for an image.
    
    Args:
        image_path: Original image path
        cache_dir: Cache directory
        file_type: 'embedding' or 'segmented'
    
    Returns:
        Path to cache file
    """
    # Use image filename and modification time for cache key
    filename = os.path.basename(image_path)
    mtime = os.path.getmtime(image_path)
    
    # Create hash of filename + mtime for uniqueness
    cache_key = hashlib.md5(f"{filename}_{mtime}".encode()).hexdigest()
    
    if file_type == "embedding":
        cache_filename = f"{os.path.splitext(filename)[0]}_{cache_key}.npy"
    else:  # segmented
        cache_filename = f"{os.path.splitext(filename)[0]}_{cache_key}_seg.npy"
    
    return os.path.join(cache_dir, cache_filename)


def load_from_cache(image_path: str, cache_dir: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load embedding and segmented image from cache if available.
    
    Returns:
        Tuple of (embedding, segmented_image) or None if not cached
    """
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
        emb_cache_path = get_cache_path(image_path, cache_dir, "embedding")
        seg_cache_path = get_cache_path(image_path, cache_dir, "segmented")
        
        np.save(emb_cache_path, embedding)
        np.save(seg_cache_path, segmented)
    except Exception as e:
        # Cache save failure is non-fatal
        pass


def embed_image_safe(
    image_path: str,
    model: AutoModel,
    processor: AutoProcessor,
    cache_dir: Optional[str] = None,
    verbose: bool = False
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Safely embed an image with error handling and caching.
    
    Returns:
        Tuple of (embedding, segmented_image) or None if failed
    """
    # Try to load from cache first
    if cache_dir:
        cached = load_from_cache(image_path, cache_dir)
        if cached is not None:
            return cached
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            if verbose:
                print(f"    ✗ Could not load: {os.path.basename(image_path)}")
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Segment hand
        segmented = segment_hand(image_rgb)
        if segmented is None:
            if verbose:
                print(f"    ✗ Hand not detected: {os.path.basename(image_path)}")
            return None
        
        if segmented.size == 0:
            if verbose:
                print(f"    ✗ Empty segmentation: {os.path.basename(image_path)}")
            return None
        
        # Process with CLIP
        inputs = processor(images=[segmented], return_tensors="pt").to(model.device)
        
        # Generate embedding
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
        embedding = embeddings.cpu().numpy()[0]
        
        # Save to cache
        if cache_dir:
            save_to_cache(image_path, cache_dir, embedding, segmented)
        
        return (embedding, segmented)
        
    except Exception as e:
        if verbose:
            print(f"    ✗ Error embedding {os.path.basename(image_path)}: {e}")
        return None


def create_thumbnail(image_data, size: Tuple[int, int]) -> Image.Image:
    """
    Create a thumbnail from image data.
    
    Args:
        image_data: Either a file path (str) or numpy array (RGB image)
        size: Target size as (width, height)
    """
    if isinstance(image_data, str):
        # Load from path
        img = Image.open(image_data)
    else:
        # Convert numpy array to PIL Image
        img = Image.fromarray(image_data)
    
    img.thumbnail(size, Image.Resampling.LANCZOS)
    return img


def create_comparison_image(
    base_image_data,
    base_image_name: str,
    comparisons: List[Tuple[str, float]],
    comparison_images: Dict[str, np.ndarray],
    output_path: str,
    thumbnail_size: int = 150,
    grid_cols: int = 5
):
    """
    Create a visual comparison image showing base image and all comparisons.
    
    Args:
        base_image_data: Numpy array of the base (segmented) image
        base_image_name: Name of the base image for display
        comparisons: List of (image_path, similarity_score) tuples
        comparison_images: Dict mapping image paths to segmented image arrays
        output_path: Where to save the result
        thumbnail_size: Size of thumbnail images
        grid_cols: Number of columns in the comparison grid
    """
    # Calculate dimensions
    margin = 20
    text_height = 60
    base_thumb_size = thumbnail_size * 2
    
    # Calculate grid dimensions
    num_comparisons = len(comparisons)
    grid_rows = (num_comparisons + grid_cols - 1) // grid_cols
    
    # Calculate canvas size
    canvas_width = margin + base_thumb_size + margin + (grid_cols * (thumbnail_size + margin))
    canvas_height = max(
        margin + base_thumb_size + text_height + margin,
        margin + text_height + (grid_rows * (thumbnail_size + text_height + margin))
    )
    
    # Create canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Load font
    try:
        title_font = ImageFont.truetype("arial.ttf", 20)
        label_font = ImageFont.truetype("arial.ttf", 12)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Add base image thumbnail (using segmented image)
    base_thumb = create_thumbnail(base_image_data, (base_thumb_size, base_thumb_size))
    canvas.paste(base_thumb, (margin, margin + text_height))
    
    # Add base image title
    draw.text((margin, margin), f"Base: {base_image_name}", fill='black', font=title_font)
    
    # Add comparison grid
    grid_start_x = margin + base_thumb_size + margin
    grid_start_y = margin + text_height
    
    for idx, (comp_path, similarity) in enumerate(comparisons):
        row = idx // grid_cols
        col = idx % grid_cols
        
        x = grid_start_x + col * (thumbnail_size + margin)
        y = grid_start_y + row * (thumbnail_size + text_height + margin)
        
        # Add thumbnail (using segmented image)
        try:
            if comp_path in comparison_images:
                comp_thumb = create_thumbnail(comparison_images[comp_path], (thumbnail_size, thumbnail_size))
                canvas.paste(comp_thumb, (x, y))
            else:
                # Fallback to original image if segmented not available
                comp_thumb = create_thumbnail(comp_path, (thumbnail_size, thumbnail_size))
                canvas.paste(comp_thumb, (x, y))
        except Exception as e:
            # Draw error box if thumbnail fails
            draw.rectangle([x, y, x + thumbnail_size, y + thumbnail_size], outline='red', width=2)
        
        # Add label
        comp_name = os.path.basename(comp_path)
        distance = 1 - similarity
        
        # Color code by similarity
        if distance <= 0.05:
            color = 'green'
        elif distance <= 0.15:
            color = 'blue'
        elif distance <= 0.30:
            color = 'orange'
        else:
            color = 'red'
        
        label_y = y + thumbnail_size + 5
        draw.text((x, label_y), comp_name[:20], fill='black', font=label_font)
        draw.text((x, label_y + 15), f"Sim: {similarity:.4f}", fill=color, font=label_font)
        draw.text((x, label_y + 30), f"Dist: {distance:.4f}", fill=color, font=label_font)
    
    # Save
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Batch compare all images in a directory"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="../images/image-testing",
        help="Directory containing images to compare"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../images/testing-results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress information"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (re-embed all images)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.normpath(os.path.join(script_dir, args.input_dir))
    output_dir = os.path.normpath(os.path.join(script_dir, args.output_dir))
    
    print("="*60)
    print("BATCH IMAGE COMPARISON")
    print("="*60)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready")
    
    # Create cache directory
    cache_dir = None
    if not args.no_cache:
        cache_dir = os.path.join(output_dir, ".cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"✓ Cache directory ready: {cache_dir}")
    else:
        print(f"⚠ Caching disabled - will re-embed all images")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and
        os.path.splitext(f)[1] in image_extensions
    ]
    
    print(f"✓ Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("Error: No images found in input directory")
        sys.exit(1)
    
    # Load CLIP model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    
    try:
        print("  - Loading CLIP model...", end=" ", flush=True)
        model = AutoModel.from_pretrained(MODEL)
        model.eval()
        print("✓")
        
        print("  - Loading processor...", end=" ", flush=True)
        processor = AutoProcessor.from_pretrained(MODEL)
        print("✓")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  - Using device: {device}")
        
        if device == "cuda":
            model = model.to(device)
        
    except Exception as e:
        print("✗ FAILED")
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Embed all images
    print("\n" + "="*60)
    print("EMBEDDING IMAGES")
    print("="*60)
    if cache_dir:
        print("Note: Using cache - previously embedded images will be loaded from cache")
    
    embeddings = {}
    segmented_images = {}
    valid_images = []
    cached_count = 0
    
    for idx, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        
        # Check if cached
        is_cached = False
        if cache_dir:
            cached = load_from_cache(image_path, cache_dir)
            if cached is not None:
                is_cached = True
        
        status_str = "[CACHED]" if is_cached else ""
        print(f"[{idx}/{len(image_files)}] Embedding {image_name}... {status_str}", end=" ", flush=True)
        
        result = embed_image_safe(image_path, model, processor, cache_dir, args.verbose)
        
        if result is not None:
            embedding, segmented = result
            embeddings[image_path] = embedding
            segmented_images[image_path] = segmented
            valid_images.append(image_path)
            if is_cached:
                cached_count += 1
            print("✓")
        else:
            print("✗ SKIPPED (see error above)" if args.verbose else "✗ SKIPPED")
    
    print(f"\n✓ Successfully embedded {len(valid_images)}/{len(image_files)} images")
    if cached_count > 0:
        print(f"  ({cached_count} loaded from cache, {len(valid_images) - cached_count} newly embedded)")
    
    if len(valid_images) == 0:
        print("Error: No images could be embedded successfully")
        sys.exit(1)
    
    # Calculate similarity matrix
    print("\n" + "="*60)
    print("CALCULATING SIMILARITIES")
    print("="*60)
    
    num_comparisons = len(valid_images) * (len(valid_images) - 1) // 2
    print(f"Total comparisons: {num_comparisons}")
    
    similarity_matrix = {}
    comparison_count = 0
    
    for i, img1 in enumerate(valid_images):
        for j, img2 in enumerate(valid_images):
            if i != j:
                similarity = calculate_cosine_similarity(embeddings[img1], embeddings[img2])
                similarity_matrix[(img1, img2)] = similarity
                comparison_count += 1
    
    print(f"✓ Calculated {comparison_count} similarities")
    
    # Group images by item name (e.g., "item04" from "item04-1.JPG")
    print("\n" + "="*60)
    print("GROUPING IMAGES BY ITEM")
    print("="*60)
    
    item_groups = {}
    for image_path in valid_images:
        image_name = os.path.basename(image_path)
        # Extract item name (e.g., "item04" from "item04-1.JPG")
        # Split by '-' and take the first part
        if '-' in image_name:
            item_name = image_name.split('-')[0]
        else:
            # If no dash, use the whole name without extension as item name
            item_name = os.path.splitext(image_name)[0]
        
        if item_name not in item_groups:
            item_groups[item_name] = []
        item_groups[item_name].append(image_path)
    
    print(f"✓ Found {len(item_groups)} unique items:")
    for item_name, images in item_groups.items():
        print(f"  - {item_name}: {len(images)} image(s)")
    
    # Generate comparison images (one per item, using first valid image)
    print("\n" + "="*60)
    print("GENERATING COMPARISON IMAGES")
    print("="*60)
    print(f"Creating {len(item_groups)} comparison files (one per item)...")
    
    for idx, (item_name, item_images) in enumerate(sorted(item_groups.items()), 1):
        # Use the first image in the group as the base
        base_image = item_images[0]
        base_name = os.path.basename(base_image)
        
        print(f"[{idx}/{len(item_groups)}] Creating comparison for {item_name} (using {base_name})...", end=" ", flush=True)
        
        # Get all comparisons for this image (compare to ALL images, including same item)
        comparisons = []
        for other_image in valid_images:
            if other_image != base_image:
                similarity = similarity_matrix[(base_image, other_image)]
                comparisons.append((other_image, similarity))
        
        # Sort by similarity (most similar first)
        comparisons.sort(key=lambda x: x[1], reverse=True)
        
        # Create comparison image with item name (using segmented images)
        output_path = os.path.join(output_dir, f"{item_name}_comparison.png")
        
        try:
            create_comparison_image(
                segmented_images[base_image],
                base_name,
                comparisons,
                segmented_images,
                output_path
            )
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Generate CSV file
    print("\n" + "="*60)
    print("GENERATING CSV FILE")
    print("="*60)
    
    csv_path = os.path.join(output_dir, "similarity_matrix.csv")
    
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['Image'] + [os.path.basename(img) for img in valid_images]
            writer.writerow(header)
            
            # Write similarity matrix
            for img1 in valid_images:
                row = [os.path.basename(img1)]
                for img2 in valid_images:
                    if img1 == img2:
                        row.append('1.0000')  # Self-similarity
                    else:
                        similarity = similarity_matrix[(img1, img2)]
                        row.append(f'{similarity:.4f}')
                writer.writerow(row)
        
        print(f"✓ CSV saved to: {csv_path}")
        
    except Exception as e:
        print(f"✗ Error generating CSV: {e}")
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Calculate average similarities (excluding self-comparisons)
    all_similarities = [sim for (img1, img2), sim in similarity_matrix.items() if img1 != img2]
    
    if all_similarities:
        avg_similarity = np.mean(all_similarities)
        min_similarity = np.min(all_similarities)
        max_similarity = np.max(all_similarities)
        
        print(f"Average similarity: {avg_similarity:.4f}")
        print(f"Min similarity:     {min_similarity:.4f}")
        print(f"Max similarity:     {max_similarity:.4f}")
        
        # Count by threshold
        very_similar = sum(1 for s in all_similarities if (1-s) <= 0.05)
        similar = sum(1 for s in all_similarities if 0.05 < (1-s) <= 0.15)
        somewhat = sum(1 for s in all_similarities if 0.15 < (1-s) <= 0.30)
        not_similar = sum(1 for s in all_similarities if (1-s) > 0.30)
        
        print(f"\nSimilarity distribution:")
        print(f"  VERY SIMILAR (dist ≤ 0.05):     {very_similar} ({100*very_similar/len(all_similarities):.1f}%)")
        print(f"  SIMILAR (0.05 < dist ≤ 0.15):   {similar} ({100*similar/len(all_similarities):.1f}%)")
        print(f"  SOMEWHAT (0.15 < dist ≤ 0.30):  {somewhat} ({100*somewhat/len(all_similarities):.1f}%)")
        print(f"  NOT SIMILAR (dist > 0.30):      {not_similar} ({100*not_similar/len(all_similarities):.1f}%)")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"  - {len(item_groups)} comparison images (one per item)")
    print(f"  - 1 CSV file with full similarity matrix")
    print(f"\nNote: Processed {len(valid_images)} total images grouped into {len(item_groups)} items")


if __name__ == "__main__":
    main()
