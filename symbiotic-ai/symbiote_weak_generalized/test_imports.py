#!/usr/bin/env python3
"""Test script to verify refactored imports work correctly."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing imports...")

try:
    from symbiote.core import DEFAULT_CONFIG, MODEL
    print("[OK] Core config imported")
    
    from symbiote.preprocessing import is_blurry, load_image_as_rgb
    print("[OK] Preprocessing imported")
    
    from symbiote.embeddings import embed_image, save_frame_to_cache
    print("[OK] Embeddings imported")
    
    from symbiote.datasets import scan_dataset, stratified_split, EmbeddingDataset
    print("[OK] Datasets imported")
    
    from symbiote.models import ClassifierHead
    print("[OK] Models imported")
    
    from symbiote.training import train_classifier, evaluate_classifier
    print("[OK] Training imported")
    
    from symbiote.persistence import save_model, load_model
    print("[OK] Persistence imported")
    
    from symbiote.visualization import plot_confusion_matrix, plot_training_history
    print("[OK] Visualization imported")
    
    from symbiote.inference import ObjectRecognizer
    print("[OK] Inference imported")
    
    from symbiote.pipelines import run_video_training, run_training
    print("[OK] Pipelines imported")
    
    from symbiote.cli import main
    print("[OK] CLI imported")
    
    print("\n[SUCCESS] All imports successful!")
    print(f"\nDEFAULT_CONFIG keys: {list(DEFAULT_CONFIG.keys())}")
    print(f"MODEL: {MODEL}")
    
except ImportError as e:
    print(f"\n[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
