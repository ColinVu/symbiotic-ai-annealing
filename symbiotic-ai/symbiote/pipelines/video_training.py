"""Video-based training pipeline."""

import os
import json
import random
from typing import Dict, Any, Optional
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor

from ..core.config import MODEL
from ..preprocessing.video_processor import process_video_frames
from ..embeddings.cache_manager import save_frame_to_cache
from ..datasets.scanner import load_all_cached_embeddings
from ..datasets.splitter import stratified_split
from ..datasets.embedding_dataset import EmbeddingDataset
from ..models.classifier import ClassifierHead
from ..training.trainer import train_classifier
from ..training.evaluator import evaluate_classifier
from ..persistence.model_io import save_model
from ..visualization.plots import plot_training_history, plot_confusion_matrix
from ..state_detection.detector import detect_states_from_video, HandState


def run_video_training(
    video_path: str,
    label: str,
    base_output_dir: str,
    config: Dict[str, Any],
    threshold: float = 100.0,
    frame_skip: int = 4,
    image_dir: Optional[str] = None,
    verbose: bool = True,
    htk_model_dir: Optional[str] = None,
    aruco_config_path: Optional[str] = None,
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
        htk_model_dir: Path to trained HTK HMM model for real state detection (optional)
        aruco_config_path: Path to ARUCO marker configuration JSON (optional)
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
    
    # Build state detection function with HTK parameters
    def _state_detect(vp, emb, fn, fps):
        return detect_states_from_video(
            vp, emb, fn, fps,
            htk_model_dir=htk_model_dir,
            aruco_config_path=aruco_config_path,
            frame_skip=frame_skip,
            blur_threshold=threshold,
            clip_model=clip_model,
            clip_processor=processor,
            verbose=verbose,
        )

    # Process video frames and add to cache
    video_embeddings, video_labels, video_paths, state_results = process_video_frames(
        video_path, label, clip_model, processor, cache_dir,
        save_frame_to_cache,  # Pass the cache function
        threshold=threshold, 
        frame_skip=frame_skip,
        state_filter={HandState.CARRY_WITH.value},  # Only cache CARRY_WITH frames
        state_detection_func=_state_detect,
        verbose=verbose
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
    
    # Save state detection results
    if not state_results.empty:
        state_results_path = os.path.join(output_dir, "state_detection.csv")
        state_results.to_csv(state_results_path, index=False)
        if verbose:
            print(f"✓ State detection results saved to: state_detection.csv")
    
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
    if not state_results.empty:
        print(f"  - state_detection.csv")
    print(f"\nFinal Test Accuracy: {eval_results['top1_accuracy']*100:.2f}%")
    print(f"\n💡 TIP: Run again with a different video/label to ADD more training data!")
    
    return classifier, eval_results


__all__ = ['run_video_training']
