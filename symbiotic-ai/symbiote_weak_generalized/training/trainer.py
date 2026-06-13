"""Model training utilities."""

from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models.classifier import ClassifierHead


def train_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: ClassifierHead,
    config: Dict[str, Any],
    device: str = "cpu",
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train the classifier with early stopping.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model: Classifier model
        config: Configuration dictionary
        device: Device to train on
        verbose: Whether to print progress
    
    Returns:
        Dictionary with training history (train_loss, val_loss, train_acc, val_acc)
    """
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    if verbose:
        print("\n" + "="*60)
        print("TRAINING CLASSIFIER")
        print("="*60)
        print(f"Max epochs: {config['max_epochs']}")
        print(f"Early stopping patience: {config['early_stopping_patience']}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Device: {device}")
        print()
    
    for epoch in range(config["max_epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * embeddings.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device)
                
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * embeddings.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        if verbose:
            print(f"Epoch {epoch+1:3d}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"Restored best model (val_loss={best_val_loss:.4f})")
    
    return history


__all__ = ['train_classifier']
