"""Visualization utilities for training and evaluation results."""

from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def plot_confusion_matrix(
    cm: np.ndarray,
    cm_raw: np.ndarray,
    label_names: List[str],
    output_path: str
):
    """Plot and save confusion matrix with both counts and row-normalized % (0-1)."""
    # Build annotations: count on first line, percent on second
    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{int(cm_raw[i, j])}\n({cm[i, j]:.2f})"
    annot_flat = annot.ravel().tolist()
    annot_2d = np.array(annot_flat).reshape(cm.shape)
    plt.figure(figsize=(10, 8))
    if HAS_SEABORN:
        sns.heatmap(
            cm, annot=annot_2d, fmt='', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names,
            vmin=0, vmax=1
        )
    else:
        plt.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        plt.colorbar()
        plt.xticks(np.arange(len(label_names)), label_names)
        plt.yticks(np.arange(len(label_names)), label_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (count and row-normalized %)')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f'{int(cm_raw[i, j])}\n({cm[i, j]:.2f})',
                         ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: str
):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history["train_loss"], label='Train')
    axes[0].plot(history["val_loss"], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history["train_acc"], label='Train')
    axes[1].plot(history["val_acc"], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Training history saved to: {output_path}")


__all__ = ['plot_confusion_matrix', 'plot_training_history', 'HAS_SEABORN']
