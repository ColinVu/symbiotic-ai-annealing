"""Visualization utilities for training and evaluation results."""

from typing import Dict, List, Optional
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
    output_path: str,
    *,
    show_labels: bool = True,
    show_annotations: Optional[bool] = None,
):
    """Plot and save confusion matrix with both counts and row-normalized % (0-1)."""
    n = int(cm.shape[0])
    if show_annotations is None:
        show_annotations = show_labels and n <= 30

    if show_labels:
        x_labels = label_names
        y_labels = label_names
        figsize = (10, 8)
    else:
        x_labels = False
        y_labels = False
        figsize = (max(10, min(n * 0.12, 24)), max(8, min(n * 0.12, 24)))

    annot_2d = None
    if show_annotations:
        annot = np.empty(cm.shape, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{int(cm_raw[i, j])}\n({cm[i, j]:.2f})"
        annot_flat = annot.ravel().tolist()
        annot_2d = np.array(annot_flat).reshape(cm.shape)

    plt.figure(figsize=figsize)
    if HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=annot_2d,
            fmt='' if show_annotations else '',
            cmap='Blues',
            xticklabels=x_labels,
            yticklabels=y_labels,
            vmin=0,
            vmax=1,
            cbar=True,
        )
        if not show_labels:
            plt.xlabel('Predicted class index')
            plt.ylabel('True class index')
    else:
        plt.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        plt.colorbar()
        if show_labels:
            plt.xticks(np.arange(len(label_names)), label_names)
            plt.yticks(np.arange(len(label_names)), label_names)
        else:
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('Predicted class index')
            plt.ylabel('True class index')
        plt.title('Confusion Matrix (count and row-normalized %)')
        if show_annotations:
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j,
                        i,
                        f'{int(cm_raw[i, j])}\n({cm[i, j]:.2f})',
                        ha='center',
                        va='center',
                        color='black',
                    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
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
