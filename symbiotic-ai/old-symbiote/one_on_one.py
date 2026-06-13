#!/usr/bin/env python3
"""
One-on-one image comparison tool for testing individual image similarity.

This script compares two images by:
1. Segmenting the hand from each image
2. Computing CLIP embeddings for each hand crop
3. Calculating the cosine similarity between the embeddings

Usage:
    python one_on_one.py <image1_path> <image2_path>
"""

import argparse
import sys
import os
import numpy as np
import cv2
from transformers import AutoModel, AutoProcessor
import torch

# Suppress MediaPipe warnings
os.environ["GLOG_minloglevel"] = "2"

# Add lib folder to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

from embedding import MODEL
from hand_detection import segment_hand


def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity between 0 and 1 (1 = identical)
    """
    # Normalize embeddings
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1_norm, embedding2_norm)
    
    return float(similarity)


def calculate_cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine distance between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine distance between 0 and 2 (0 = identical)
    """
    similarity = calculate_cosine_similarity(embedding1, embedding2)
    distance = 1 - similarity
    return float(distance)


def embed_image_safe(image: cv2.typing.MatLike, model: AutoModel, processor: AutoProcessor, image_name: str, verbose: bool = False):
    """
    Safely embed an image with error handling and progress tracking.
    
    Args:
        image: OpenCV image
        model: CLIP model
        processor: CLIP processor
        image_name: Name for logging
        verbose: Whether to show detailed progress
    
    Returns:
        numpy array of embedding or None if failed
    """
    try:
        # Convert BGR to RGB
        print(f"  - Converting {image_name} to RGB...", end=" ", flush=True)
        if verbose:
            print(f"\n    [Shape before: {image.shape}]", end=" ")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if verbose:
            print(f"[Shape after: {image_rgb.shape}]", end=" ")
        print("✓")
        
        # Segment hand
        print(f"  - Segmenting hand in {image_name} (using MediaPipe)...", end=" ", flush=True)
        if verbose:
            print("\n    [This may take 5-10 seconds]...", end=" ")
        
        try:
            segmented = segment_hand(image_rgb)
        except Exception as seg_error:
            print("✗ FAILED")
            print(f"    Error during hand segmentation: {seg_error}")
            return None
            
        if segmented is None:
            print("✗ FAILED")
            print(f"    Error: Could not detect hand in {image_name}")
            print(f"    Tip: Make sure the image shows a clear hand")
            return None
        
        if verbose:
            print(f"[Segmented shape: {segmented.shape}]", end=" ")
        print("✓")
        
        # Check segmented image is valid
        if segmented.size == 0:
            print("✗ FAILED")
            print(f"    Error: Segmented image is empty")
            return None
        
        # Process with CLIP
        print(f"  - Processing with CLIP processor...", end=" ", flush=True)
        try:
            inputs = processor(images=[segmented], return_tensors="pt").to(model.device)
            if verbose:
                print(f"\n    [Input tensor shape: {inputs.pixel_values.shape}]", end=" ")
        except Exception as proc_error:
            print("✗ FAILED")
            print(f"    Error during processing: {proc_error}")
            return None
        print("✓")
        
        # Generate embedding
        print(f"  - Generating embedding with CLIP model...", end=" ", flush=True)
        try:
            with torch.no_grad():
                embeddings = model.get_image_features(**inputs)
            embedding = embeddings.cpu().numpy()[0]
            
            if verbose:
                print(f"\n    [Embedding shape: {embedding.shape}]", end=" ")
        except Exception as emb_error:
            print("✗ FAILED")
            print(f"    Error during embedding generation: {emb_error}")
            return None
        print("✓")
        
        return embedding
        
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        raise
    except Exception as e:
        print(f"✗ FAILED")
        print(f"    Unexpected error: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return None


def compare_images(image1_path: str, image2_path: str, model: AutoModel, processor: AutoProcessor, verbose: bool = False):
    """
    Compare two images using the symbiotic-ai framework.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        model: CLIP model for embeddings
        processor: CLIP processor
        verbose: Whether to show detailed progress
    
    Returns:
        Tuple of (similarity, distance, embedding1, embedding2)
    """
    # Load images
    print(f"\n[1/4] Loading images...")
    print(f"  - Reading image 1: {os.path.basename(image1_path)}...", end=" ", flush=True)
    image1 = cv2.imread(image1_path)
    if image1 is None:
        print("✗ FAILED")
        raise ValueError(f"Could not load image: {image1_path}")
    print(f"✓ ({image1.shape[1]}x{image1.shape[0]})")
    
    print(f"  - Reading image 2: {os.path.basename(image2_path)}...", end=" ", flush=True)
    image2 = cv2.imread(image2_path)
    if image2 is None:
        print("✗ FAILED")
        raise ValueError(f"Could not load image: {image2_path}")
    print(f"✓ ({image2.shape[1]}x{image2.shape[0]})")
    
    # Embed image 1
    print(f"\n[2/4] Processing image 1...")
    embedding1 = embed_image_safe(image1, model, processor, "image 1", verbose)
    if embedding1 is None:
        raise ValueError(f"Could not embed image 1 (see error above)")
    
    # Embed image 2
    print(f"\n[3/4] Processing image 2...")
    embedding2 = embed_image_safe(image2, model, processor, "image 2", verbose)
    if embedding2 is None:
        raise ValueError(f"Could not embed image 2 (see error above)")
    
    # Calculate similarity and distance
    print(f"\n[4/4] Calculating similarity...")
    print(f"  - Computing cosine similarity...", end=" ", flush=True)
    similarity = calculate_cosine_similarity(embedding1, embedding2)
    distance = calculate_cosine_distance(embedding1, embedding2)
    print("✓")
    
    return similarity, distance, embedding1, embedding2


def main():
    parser = argparse.ArgumentParser(
        description="Compare two images one-on-one using cosine similarity of their embeddings"
    )
    parser.add_argument(
        "image1",
        type=str,
        help="Path to first image"
    )
    parser.add_argument(
        "image2",
        type=str,
        help="Path to second image"
    )
    parser.add_argument(
        "--show-embeddings",
        action="store_true",
        help="Show the first 10 values of each embedding"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress information"
    )
    
    args = parser.parse_args()
    
    # Set verbose mode
    if args.verbose:
        print("\n[VERBOSE MODE ENABLED]")
    
    # Validate image files exist
    if not os.path.exists(args.image1):
        print(f"Error: Image 1 not found: {args.image1}")
        sys.exit(1)
    
    if not os.path.exists(args.image2):
        print(f"Error: Image 2 not found: {args.image2}")
        sys.exit(1)
    
    # Load CLIP model
    print("="*60)
    print("ONE-ON-ONE IMAGE COMPARISON")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"\nImage 1: {os.path.basename(args.image1)}")
    print(f"Image 2: {os.path.basename(args.image2)}")
    print("\nNote: First run may take 1-2 minutes to download the model.")
    print("      Subsequent runs will be faster.")
    print("      Press Ctrl+C at any time to cancel.")
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    print("\nLoading CLIP model...")
    
    try:
        print("  - Loading model...", end=" ", flush=True)
        model = AutoModel.from_pretrained(MODEL)
        model.eval()  # Set to evaluation mode
        print("✓")
        
        print("  - Loading processor...", end=" ", flush=True)
        processor = AutoProcessor.from_pretrained(MODEL)
        print("✓")
        
        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  - Using device: {device}")
        
        if device == "cuda":
            model = model.to(device)
            print(f"  - Model moved to GPU")
        
        print("\n✓ Model loaded successfully!")
        
    except Exception as e:
        print("✗ FAILED")
        print(f"\nError loading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection (model may need to download)")
        print("  2. Ensure you have enough disk space")
        print("  3. Try running: pip install --upgrade transformers torch")
        sys.exit(1)
    
    # Compare images
    try:
        similarity, distance, embedding1, embedding2 = compare_images(
            args.image1,
            args.image2,
            model,
            processor,
            args.verbose
        )
        
        # Print results
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(f"Image 1: {args.image1}")
        print(f"Image 2: {args.image2}")
        print()
        print(f"Cosine Similarity: {similarity:.6f}")
        print(f"Cosine Distance:   {distance:.6f}")
        print()
        
        # Interpretation
        print("Interpretation:")
        if distance <= 0.05:
            print("  ✓ VERY SIMILAR (would match with default threshold)")
        elif distance <= 0.15:
            print("  ~ SIMILAR (but below default threshold)")
        elif distance <= 0.30:
            print("  ○ SOMEWHAT SIMILAR")
        else:
            print("  ✗ NOT SIMILAR")
        
        print()
        print(f"Default DETECT_THRESHOLD: 0.05 (similarity >= 0.95)")
        print(f"These images would {'MATCH' if distance <= 0.05 else 'NOT MATCH'} in inference")
        print("="*60)
        
        # Show embeddings if requested
        if args.show_embeddings:
            print("\nEmbedding Preview (first 10 values):")
            print(f"Image 1: {embedding1[:10]}")
            print(f"Image 2: {embedding2[:10]}")
            print(f"Embedding dimensions: {embedding1.shape}")
        
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("INTERRUPTED BY USER")
        print("="*60)
        print("Process was interrupted. This is safe.")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n\n" + "="*60)
        print("ERROR")
        print("="*60)
        print(f"Error during comparison: {e}")
        
        if args.verbose:
            print("\nFull traceback:")
            import traceback
            traceback.print_exc()
        
        print("\nTroubleshooting:")
        print("  1. Ensure images contain visible hands")
        print("  2. Try with --verbose flag for more details")
        print("  3. Check that images are valid JPG/PNG files")
        print("  4. If it hangs, press Ctrl+C to stop")
        sys.exit(1)


if __name__ == "__main__":
    main()
