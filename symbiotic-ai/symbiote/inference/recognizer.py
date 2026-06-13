"""Object recognition inference API."""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from ..persistence.model_io import load_model
from ..embeddings.clip_embedder import embed_image_for_inference


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


__all__ = ['ObjectRecognizer']
