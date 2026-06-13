"""Object recognition inference API using centroid-based classification."""

from typing import Optional, Dict, Any, List, Tuple
import json
import os
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from ..persistence.model_io import load_model
from ..embeddings.clip_embedder import embed_image_for_inference
from ..models.classifier import CentroidModel


class ObjectRecognizer:
    """
    High-level API for object recognition inference using centroid model.
    
    Usage:
        recognizer = ObjectRecognizer("path/to/model")
        result = recognizer.predict("path/to/image.jpg")
        print(f"Predicted: {result['label']} (confidence: {result['confidence']:.2f})")
    """
    
    def __init__(self, model_dir: str, device: str = None):
        """
        Initialize the recognizer.
        
        Args:
            model_dir: Directory containing saved centroid model
            device: Device to use for CLIP (auto-detected if None)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.model, self.metadata = load_model(model_dir)
        
        print(f"Loading CLIP model ({self.metadata['clip_model']})...")
        self.clip_model = AutoModel.from_pretrained(self.metadata['clip_model'])
        self.clip_model.eval()
        if device == "cuda":
            self.clip_model = self.clip_model.to(device)
        
        self.processor = AutoProcessor.from_pretrained(self.metadata['clip_model'])
        print(f"Ready for inference! ({self.model.num_classes} classes)")

        self._hand_neutralizer = None
        self._clip_adapter_nn = None
        hn_path = os.path.join(model_dir, "hand_neutralizer.json")
        ap_path = os.path.join(model_dir, "clip_adapter.pt")
        if os.path.isfile(hn_path):
            from ..training.hand_neutralizer import HandNeutralizer

            with open(hn_path, "r", encoding="utf-8") as f:
                self._hand_neutralizer = HandNeutralizer.from_state_dict(json.load(f), verbose=False)
        if os.path.isfile(ap_path):
            from ..training.clip_adapter import CLIPAdapter

            dim = int(self.metadata.get("embedding_dim", 512))
            m = CLIPAdapter(dim)
            m.load_state_dict(torch.load(ap_path, map_location=device))
            m.eval()
            if device == "cuda":
                m = m.to(device)
            self._clip_adapter_nn = m
    
    def _postprocess_embedding(self, emb: np.ndarray) -> np.ndarray:
        """Apply saved hand neutralizer + CLIP adapter (iterated-model), if present."""
        x = np.asarray(emb, dtype=np.float64).reshape(-1)
        if self._hand_neutralizer is not None and self._hand_neutralizer.enabled:
            x = np.asarray(self._hand_neutralizer.neutralize(x), dtype=np.float64).reshape(-1)
        if self._clip_adapter_nn is not None:
            with torch.no_grad():
                t = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
                x = self._clip_adapter_nn(t).cpu().numpy().reshape(-1)
        return x.astype(np.float64)

    def _get_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Get CLIP embedding for an image."""
        embedding = embed_image_for_inference(image_path, self.clip_model, self.processor)
        if embedding is None:
            return None
        return self._postprocess_embedding(embedding.numpy())
    
    def predict(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Predict the object label for an image.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Dictionary with:
                - label: Predicted class label
                - confidence: Confidence score (0-1)
                - all_scores: Dict of all class probabilities
            Or None if embedding failed
        """
        embedding = self._get_embedding(image_path)
        if embedding is None:
            return None
        
        label, confidence = self.model.predict_with_confidence(embedding)
        all_scores = self.model.predict_proba(embedding)
        
        return {
            "label": label,
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
        embedding = self._get_embedding(image_path)
        if embedding is None:
            return None
        
        return self.model.predict_top_k(embedding, k=k)
    
    def predict_ambiguous_set(
        self,
        image_path: str,
        relative_margin: float = 0.08,
        min_absolute: float = 0.02,
    ) -> Optional[Dict[str, Any]]:
        """
        Like ``predict`` but returns ``ambiguous_labels``: all classes within a
        distance band of the best centroid (see ``CentroidModel.predict_ambiguous_set``).
        """
        embedding = self._get_embedding(image_path)
        if embedding is None:
            return None
        out = self.model.predict_ambiguous_set(
            embedding,
            relative_margin=relative_margin,
            min_absolute=min_absolute,
        )
        probs = self.model.predict_proba(embedding)
        out["all_scores"] = probs
        return out

    def predict_embedding(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Predict from a pre-computed CLIP embedding.
        
        Args:
            embedding: Raw CLIP embedding (same dim as training); postprocessed like ``predict``.
            
        Returns:
            Dictionary with label, confidence, and all_scores
        """
        emb = self._postprocess_embedding(np.asarray(embedding, dtype=np.float64))
        label, confidence = self.model.predict_with_confidence(emb)
        all_scores = self.model.predict_proba(emb)
        
        return {
            "label": label,
            "confidence": float(confidence),
            "all_scores": all_scores,
        }


__all__ = ['ObjectRecognizer']
