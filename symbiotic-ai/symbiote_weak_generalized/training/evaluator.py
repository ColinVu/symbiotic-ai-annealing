"""Model evaluation utilities."""

from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from ..models.classifier import ClassifierHead


def evaluate_classifier(
    test_loader: DataLoader,
    model: ClassifierHead,
    idx_to_label: Dict[int, str],
    device: str = "cpu",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate classifier on test set.
    
    Args:
        test_loader: Test data loader
        model: Trained classifier model
        idx_to_label: Mapping from index to label
        device: Device to evaluate on
        verbose: Whether to print results
    
    Returns:
        Dictionary with evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings = embeddings.to(device)
            
            outputs = model(embeddings)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels)
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Top-1 accuracy
    top1_acc = (all_preds == all_labels).mean()
    
    # Top-3 accuracy
    num_classes = all_probs.shape[1]
    k = min(3, num_classes)
    top_k_preds = np.argsort(all_probs, axis=1)[:, -k:]
    top3_acc = np.mean([label in preds for label, preds in zip(all_labels, top_k_preds)])
    
    # Confusion matrix (raw counts)
    cm_raw = confusion_matrix(all_labels, all_preds)
    # Normalize by row so each row sums to 1 (proportion of that true class predicted as each class)
    row_sums = np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
    cm = (cm_raw.astype(float) / row_sums)
    
    # Per-class metrics
    labels_list = sorted(idx_to_label.keys())
    label_names = [idx_to_label[i] for i in labels_list]
    
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Test Accuracy (Top-1): {top1_acc:.4f} ({top1_acc*100:.2f}%)")
        print(f"Test Accuracy (Top-{k}): {top3_acc:.4f} ({top3_acc*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=label_names))
        print("\nConfusion Matrix (count and row-normalized %):")
        for i in range(cm.shape[0]):
            row = [f"{int(cm_raw[i, j])} ({cm[i, j]:.2f})" for j in range(cm.shape[1])]
            print("  ", "  ".join(row))
    
    return {
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "confusion_matrix": cm,
        "confusion_matrix_raw": cm_raw,
        "predictions": all_preds,
        "true_labels": all_labels,
        "probabilities": all_probs,
        "label_names": label_names,
    }


__all__ = ['evaluate_classifier']
