"""Model persistence utilities."""

import os
import json
from typing import Dict, Any, Tuple
from pathlib import Path
import torch

from ..core.config import MODEL
from ..models.classifier import ClassifierHead


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


__all__ = ['save_model', 'load_model']
