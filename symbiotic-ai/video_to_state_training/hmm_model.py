from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA

STATE_SYMBOLS = ["a", "e", "i", "m"]
STATE_LABELS = {
    "a": "pick",
    "e": "carry",
    "i": "place",
    "m": "carry_empty",
}


@dataclass
class HMMResult:
    state_sequence: List[str]
    reduced_embeddings: np.ndarray
    timestamps: Sequence[float]


class HandStateHMM:
    def __init__(
        self,
        max_dim: int = 356,
        cycle_strength: float = 0.10,
        n_iter: int = 100,
    ) -> None:
        """
        Args:
            max_dim: Maximum dimension after PCA reduction
            cycle_strength: Base probability of transitioning to next state in cycle (used for initialization)
            n_iter: Number of EM iterations for training (0 means no training, use initialization only)
        """
        self.max_dim = max_dim
        self.cycle_strength = cycle_strength
        self.n_iter = n_iter
        self.pca: PCA | None = None
        self.model: GaussianHMM | None = None

    def _reduce(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return embeddings

        n_components = min(self.max_dim, embeddings.shape[1], embeddings.shape[0])
        if n_components <= 0:
            raise ValueError("Not enough embeddings to perform PCA.")

        self.pca = PCA(n_components=n_components)
        return self.pca.fit_transform(embeddings)

    def _initial_stats(
        self, 
        reduced_embeddings: np.ndarray,
        state_assignments: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize means and variances for each state.
        
        Args:
            reduced_embeddings: Reduced embeddings
            state_assignments: Optional array of state indices for each embedding.
                             If None, uses circular assignment.
        """
        n_samples = reduced_embeddings.shape[0]
        
        if state_assignments is None:
            # Default: circular assignment
            assignments = np.arange(n_samples) % len(STATE_SYMBOLS)
        else:
            assignments = state_assignments

        means = []
        variances = []
        for state_index in range(len(STATE_SYMBOLS)):
            state_points = reduced_embeddings[assignments == state_index]
            if len(state_points) == 0:
                # fallback to global stats
                state_points = reduced_embeddings
            means.append(np.mean(state_points, axis=0))
            variances.append(np.var(state_points, axis=0) + 1e-3)

        return np.vstack(means), np.vstack(variances)

    def _build_transition_matrix(
        self,
        state_sequences: Optional[List[List[str]]] = None,
    ) -> np.ndarray:
        """Build transition matrix from training data or use cycle structure.
        
        Args:
            state_sequences: Optional list of state sequences for training.
                            If provided, estimates transition probabilities from data.
        """
        n_components = len(STATE_SYMBOLS)
        transmat = np.zeros((n_components, n_components))
        
        if state_sequences is not None and len(state_sequences) > 0:
            # Estimate transition matrix from training data
            transition_counts = np.zeros((n_components, n_components))
            
            for seq in state_sequences:
                for i in range(len(seq) - 1):
                    from_state = seq[i]
                    to_state = seq[i + 1]
                    
                    from_idx = STATE_SYMBOLS.index(from_state) if from_state in STATE_SYMBOLS else 0
                    to_idx = STATE_SYMBOLS.index(to_state) if to_state in STATE_SYMBOLS else 0
                    
                    transition_counts[from_idx, to_idx] += 1
            
            # Normalize to probabilities
            row_sums = transition_counts.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0  # Avoid division by zero
            transmat = transition_counts / row_sums
            
            # Add small probability to cycle transitions to maintain structure
            forward = self.cycle_strength * 0.1  # Small weight
            for idx in range(n_components):
                transmat[idx, idx] = transmat[idx, idx] * 0.9 + (1.0 - forward) * 0.1
                transmat[idx, (idx + 1) % n_components] = (
                    transmat[idx, (idx + 1) % n_components] * 0.9 + forward * 0.1
                )
        else:
            # Use cycle structure as default
            forward = self.cycle_strength
            stay = 1.0 - forward
            for idx in range(n_components):
                transmat[idx, idx] = stay
                transmat[idx, (idx + 1) % n_components] = forward
        
        return transmat

    def _build_model(
        self,
        reduced_embeddings: np.ndarray,
        state_assignments: Optional[np.ndarray] = None,
        state_sequences: Optional[List[List[str]]] = None,
    ) -> GaussianHMM:
        means, variances = self._initial_stats(reduced_embeddings, state_assignments)
        n_components, n_features = means.shape

        # Initialize start probabilities (uniform or from data)
        startprob = np.ones(n_components) / n_components
        
        # Build transition matrix
        transmat = self._build_transition_matrix(state_sequences)

        model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            init_params="",
            params="",
            n_iter=self.n_iter,
        )

        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means
        model.covars_ = variances

        return model

    def train(
        self,
        embeddings: np.ndarray,
        state_sequences: List[List[str]],
        lengths: Optional[List[int]] = None,
    ) -> None:
        """Train the HMM model on labeled data.
        
        Args:
            embeddings: All embeddings from training videos (concatenated)
            state_sequences: List of state sequences, one per training video
            lengths: Optional list of sequence lengths. If None, inferred from state_sequences.
        """
        if lengths is None:
            lengths = [len(seq) for seq in state_sequences]
        
        if sum(lengths) != len(embeddings):
            raise ValueError(
                f"Total sequence length {sum(lengths)} does not match "
                f"number of embeddings {len(embeddings)}"
            )
        
        # Reduce dimensionality
        reduced = self._reduce(embeddings)
        
        # Convert state sequences to indices
        state_indices_list = []
        for seq in state_sequences:
            indices = [STATE_SYMBOLS.index(s) if s in STATE_SYMBOLS else 0 for s in seq]
            state_indices_list.append(np.array(indices))
        
        # Flatten for initial stats
        all_state_indices = np.concatenate(state_indices_list)
        
        # Build and train model
        self.model = self._build_model(
            reduced,
            state_assignments=all_state_indices,
            state_sequences=state_sequences,
        )
        
        # Train using Baum-Welch algorithm
        if self.n_iter > 0:
            self.model.fit(reduced, lengths=lengths)

    def infer(
        self,
        embeddings: np.ndarray,
        timestamps: Sequence[float],
    ) -> HMMResult:
        """Infer state sequence from embeddings using trained model.
        
        Args:
            embeddings: Embeddings to predict states for
            timestamps: Timestamps for each embedding
            
        Returns:
            HMMResult with predicted state sequence
        """
        if len(embeddings) != len(timestamps):
            raise ValueError("Embeddings and timestamps must have the same length.")

        # If model not initialized, initialize it now (for backward compatibility)
        if self.model is None:
            reduced = self._reduce(embeddings)
            self.model = self._build_model(reduced)
        else:
            # Use existing PCA transform
            if self.pca is None:
                reduced = self._reduce(embeddings)
            else:
                reduced = self.pca.transform(embeddings)
        
        if reduced.size == 0:
            return HMMResult([], reduced, timestamps)

        state_indices = self.model.predict(reduced)
        state_sequence = [STATE_SYMBOLS[index] for index in state_indices]

        return HMMResult(state_sequence, reduced, timestamps)


def summarise_states(sequence: Sequence[str], timestamps: Sequence[float], frame_duration: float) -> List[Dict[str, float | str]]:
    if not sequence:
        return []

    rows: List[Dict[str, float | str]] = []

    current_symbol = sequence[0]
    segment_start = timestamps[0]

    for idx in range(1, len(sequence)):
        if sequence[idx] != current_symbol:
            rows.append(
                {
                    "start_time": segment_start,
                    "end_time": timestamps[idx],
                    "state_symbol": current_symbol,
                    "state_name": STATE_LABELS[current_symbol],
                }
            )
            segment_start = timestamps[idx]
            current_symbol = sequence[idx]

    rows.append(
        {
            "start_time": segment_start,
            "end_time": timestamps[-1] + frame_duration,
            "state_symbol": current_symbol,
            "state_name": STATE_LABELS[current_symbol],
        }
    )

    return rows


