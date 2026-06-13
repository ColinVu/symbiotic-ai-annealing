"""Video inference pipeline for frame-by-frame object classification with CSV output.

This pipeline processes videos and outputs inference results without adding data to training cache.
"""

import os
import sys
import csv
from typing import List, Dict
import cv2
import numpy as np

# Add lib folder to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from hand_detection import segment_hand

from ..preprocessing.blur_detection import is_blurry
from ..inference.recognizer import ObjectRecognizer


def run_video_inference(
    video_path: str,
    model_dir: str,
    output_csv: str,
    threshold: float = 100.0,
    frame_skip: int = 5,
    verbose: bool = True
) -> str:
    """
    Run inference on video frames and output results to CSV.
    
    This pipeline:
    1. Loads a trained classifier model
    2. Extracts frames from video (with frame skipping)
    3. Filters frames without detected hands
    4. Filters blurry frames
    5. Runs inference on each valid frame
    6. Outputs results to CSV file
    
    IMPORTANT: Does NOT add data to training cache/dataset.
    
    Args:
        video_path: Path to video file
        model_dir: Path to trained model directory
        output_csv: Path to output CSV file
        threshold: Blur detection threshold (Laplacian variance, default 100.0)
        frame_skip: Process every Nth frame (default 5)
        verbose: Whether to print progress
    
    Returns:
        Path to output CSV file
    """
    if verbose:
        print("\n" + "="*60)
        print("VIDEO INFERENCE PIPELINE")
        print("="*60)
        print(f"Video: {video_path}")
        print(f"Model: {model_dir}")
        print(f"Output: {output_csv}")
        print(f"Blur threshold: {threshold}")
        print(f"Frame skip: {frame_skip}")
    
    # Load recognizer
    if verbose:
        print("\nLoading model...")
    recognizer = ObjectRecognizer(model_dir)
    
    # Open video
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if verbose:
        print(f"\nVideo info:")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Processing every {frame_skip} frames...")
        print("\nProcessing frames...")
    
    # Process frames
    results = []
    frame_count = 0
    processed_count = 0
    inference_count = 0
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames according to frame_skip
        if frame_count % frame_skip != 0:
            continue
        
        processed_count += 1
        
        # Calculate timestamp
        timestamp = frame_count / fps if fps > 0 else 0.0
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Segment hand
        segmented = segment_hand(image_rgb)
        
        if segmented is None:
            if verbose:
                print(f"  Frame {frame_count} (t={timestamp:.2f}s): NO HAND (skipped)")
            continue
        
        if segmented.size == 0:
            if verbose:
                print(f"  Frame {frame_count} (t={timestamp:.2f}s): EMPTY (skipped)")
            continue
        
        # Check if blurry
        if is_blurry(segmented, threshold):
            if verbose:
                print(f"  Frame {frame_count} (t={timestamp:.2f}s): BLURRY (skipped)")
            continue
        
        # Run inference
        try:
            # Get top-3 predictions
            top_k_results = recognizer.predict_top_k(video_path, k=3)
            
            if top_k_results is None:
                if verbose:
                    print(f"  Frame {frame_count} (t={timestamp:.2f}s): INFERENCE FAILED")
                continue
            
            # Extract top prediction
            predicted_label = top_k_results[0][0]
            confidence = top_k_results[0][1]
            
            # Extract top-3 labels and confidences
            top_3_labels = ";".join([label for label, _ in top_k_results])
            top_3_confidences = ";".join([f"{conf:.4f}" for _, conf in top_k_results])
            
            # Record result
            results.append({
                'frame_number': frame_count,
                'timestamp': timestamp,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'top_3_labels': top_3_labels,
                'top_3_confidences': top_3_confidences
            })
            
            inference_count += 1
            
            if verbose:
                print(f"  Frame {frame_count} (t={timestamp:.2f}s): {predicted_label} ({confidence:.2f})")
        
        except Exception as e:
            if verbose:
                print(f"  Frame {frame_count} (t={timestamp:.2f}s): ERROR ({e})")
    
    capture.release()
    
    if verbose:
        print(f"\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Total frames: {frame_count}")
        print(f"Frames checked: {processed_count}")
        print(f"Inferences made: {inference_count}")
    
    # Write results to CSV
    if len(results) == 0:
        if verbose:
            print("\nWarning: No frames could be processed. Creating empty CSV.")
        results = []  # Empty CSV with headers only
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['frame_number', 'timestamp', 'predicted_label', 'confidence', 
                      'top_3_labels', 'top_3_confidences']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    if verbose:
        print(f"\nResults saved to: {output_csv}")
        print(f"Total inferences: {len(results)}")
    
    return output_csv


__all__ = ['run_video_inference']
