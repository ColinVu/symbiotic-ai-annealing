"""Image directory-based training pipeline (legacy)."""

import os
import json
import random
from typing import Dict, Any
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor

from ..core.config import MODEL
from ..datasets.scanner import scan_dataset
from ..datasets.splitter import stratified_split
from ..datasets.embedding_dataset import EmbeddingDataset
from ..models.classifier import ClassifierHead
from ..training.trainer import train_classifier
from ..training.evaluator import evaluate_classifier
from ..persistence.model_io import save_model
from ..visualization.plots import plot_training_history, plot_confusion_matrix


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


__all__ = ['run_training']
