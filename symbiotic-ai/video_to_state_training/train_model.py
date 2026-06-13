from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from .eaf_parser import eaf_to_state_sequence
from .hmm_model import HandStateHMM


def load_training_data(
    embeddings_dir: Path,
    labels_dir: Path,
) -> tuple[np.ndarray, List[List[str]], List[float]]:
    """Load pre-computed embeddings and their corresponding labels.
    
    Args:
        embeddings_dir: Directory containing embedding files (.npz files)
        labels_dir: Directory containing label files (.eaf files)
        
    Returns:
        Tuple of (all_embeddings, state_sequences, all_timestamps)
    """
    # Find all embedding files
    embedding_files = sorted(embeddings_dir.glob("*.npz"))
    if not embedding_files:
        raise ValueError(f"No .npz embedding files found in {embeddings_dir}")
    
    print(f"Found {len(embedding_files)} embedding files")
    
    all_embeddings: List[np.ndarray] = []
    all_timestamps: List[float] = []
    state_sequences: List[List[str]] = []
    
    for emb_file in tqdm(embedding_files, desc="Loading embeddings"):
        # Load embeddings
        try:
            data = np.load(emb_file)
            embeddings = data['embeddings']
            timestamps = data['timestamps'].tolist()
        except Exception as e:
            print(f"Warning: Error loading {emb_file.name}: {e}, skipping")
            continue
        
        # Find corresponding label file (case-insensitive)
        label_path = None
        for file in labels_dir.glob("*"):
            if file.stem.lower() == emb_file.stem.lower() and file.suffix.lower() == ".eaf":
                label_path = file
                break
        
        if label_path is None:
            print(f"Warning: No label file found for {emb_file.name}, skipping")
            continue
        
        # Parse labels from EAF file
        try:
            state_sequence = eaf_to_state_sequence(label_path, timestamps)
        except Exception as e:
            print(f"Error parsing labels for {emb_file.name}: {e}, skipping")
            continue
        
        # Verify lengths match
        if len(state_sequence) != len(embeddings):
            print(
                f"Warning: State sequence length ({len(state_sequence)}) "
                f"does not match embedding length ({len(embeddings)}) "
                f"for {emb_file.name}. Truncating to match."
            )
            min_len = min(len(state_sequence), len(embeddings))
            state_sequence = state_sequence[:min_len]
            embeddings = embeddings[:min_len]
            timestamps = timestamps[:min_len]
        
        all_embeddings.append(embeddings)
        all_timestamps.extend(timestamps)
        state_sequences.append(state_sequence)
    
    if not all_embeddings:
        raise ValueError("No valid training data found")
    
    # Concatenate all embeddings
    combined_embeddings = np.vstack(all_embeddings)
    
    print(f"Total training samples: {len(combined_embeddings)}")
    print(f"Number of videos: {len(state_sequences)}")
    
    return combined_embeddings, state_sequences, all_timestamps


def train_hmm(
    embeddings_dir: Path,
    labels_dir: Path,
    output_path: Path,
    max_dim: int = 356,
    cycle_strength: float = 0.10,
    n_iter: int = 100,
) -> None:
    """Train HMM model on pre-computed embeddings and labels.
    
    Args:
        embeddings_dir: Directory containing pre-computed embeddings (.npz files)
        labels_dir: Directory containing label files (.eaf)
        output_path: Path to save trained model (pickle file)
        max_dim: Maximum PCA dimensions
        cycle_strength: HMM cycle strength parameter
        n_iter: Number of EM training iterations
    """
    print("Loading training data...")
    embeddings, state_sequences, timestamps = load_training_data(
        embeddings_dir=embeddings_dir,
        labels_dir=labels_dir,
    )
    
    print("Training HMM model...")
    hmm = HandStateHMM(
        max_dim=max_dim,
        cycle_strength=cycle_strength,
        n_iter=n_iter,
    )
    
    hmm.train(embeddings, state_sequences)
    
    print(f"Saving model to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(hmm, f)
    
    print("Training complete!")
    print(f"Model saved to {output_path}")
    
    # Print some statistics
    print("\nTraining statistics:")
    print(f"  Total frames: {len(embeddings)}")
    print(f"  Number of videos: {len(state_sequences)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    if hmm.pca is not None:
        print(f"  Reduced dimension: {hmm.pca.n_components_}")
    
    # Print state distribution
    from collections import Counter
    all_states = [s for seq in state_sequences for s in seq]
    state_counts = Counter(all_states)
    print("\nState distribution:")
    for state, count in sorted(state_counts.items()):
        print(f"  {state}: {count} frames ({100 * count / len(all_states):.1f}%)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HMM model on pre-computed embeddings and labels."
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("embeddings"),
        help="Directory containing pre-computed embeddings (.npz files)",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Directory containing label files (.eaf files)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save trained model (pickle file)",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=356,
        help="Maximum dimension after PCA reduction",
    )
    parser.add_argument(
        "--cycle-strength",
        type=float,
        default=0.10,
        help="Base probability of transitioning to next state in cycle (0.0-1.0)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=100,
        help="Number of EM training iterations",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_hmm(
        embeddings_dir=args.embeddings_dir,
        labels_dir=args.labels_dir,
        output_path=args.output,
        max_dim=args.max_dim,
        cycle_strength=args.cycle_strength,
        n_iter=args.n_iter,
    )


if __name__ == "__main__":
    main()

