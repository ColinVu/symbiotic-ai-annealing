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
    orientations: Optional[np.ndarray] = None


class HandStateHMM:
    def __init__(
        self,
        max_dim: int = 356,
        cycle_strength: float = 0.10,
        orientation_threshold: float = 0.3,
        orientation_smoothing: int = 5,
    ) -> None:
        """
        Args:
            max_dim: Maximum dimension after PCA reduction
            cycle_strength: Base probability of transitioning to next state in cycle
            orientation_threshold: Minimum orientation change (dot product) to allow transition
            orientation_smoothing: Number of frames to average for orientation smoothing
        """
        self.max_dim = max_dim
        self.cycle_strength = cycle_strength
        self.orientation_threshold = orientation_threshold
        self.orientation_smoothing = orientation_smoothing
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

    def _initial_stats(self, reduced_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = reduced_embeddings.shape[0]
        assignments = np.arange(n_samples) % len(STATE_SYMBOLS)

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

    def _smooth_orientations(self, orientations: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to reduce MediaPipe noise."""
        if len(orientations) == 0:
            return orientations
        
        if self.orientation_smoothing <= 1:
            return orientations
        
        smoothed = np.zeros_like(orientations)
        window = self.orientation_smoothing
        
        for i in range(len(orientations)):
            start = max(0, i - window // 2)
            end = min(len(orientations), i + window // 2 + 1)
            window_orientations = orientations[start:end]
            
            # Average the orientation vectors
            smoothed[i] = np.mean(window_orientations, axis=0)
            # Renormalize
            norm = np.linalg.norm(smoothed[i])
            if norm > 1e-6:
                smoothed[i] = smoothed[i] / norm
        
        return smoothed

    def _compute_orientation_changes(self, orientations: np.ndarray) -> np.ndarray:
        """Compute orientation change signals between consecutive frames."""
        if len(orientations) < 2:
            return np.zeros(len(orientations))
        
        smoothed = self._smooth_orientations(orientations)
        changes = np.zeros(len(orientations))

        prev_vec: Optional[np.ndarray] = None
        for i, vec in enumerate(smoothed):
            if np.isnan(vec).any():
                changes[i] = 0.0
                continue

            if prev_vec is None:
                prev_vec = vec
                continue

            dot_product = float(np.dot(vec, prev_vec))
            dot_product = max(min(dot_product, 1.0), -1.0)
            changes[i] = 1.0 - dot_product
            prev_vec = vec

        return changes

    def _build_model(
        self,
        reduced_embeddings: np.ndarray,
        orientations: Optional[np.ndarray] = None,
    ) -> GaussianHMM:
        means, variances = self._initial_stats(reduced_embeddings)
        n_components, n_features = means.shape

        startprob = np.zeros(n_components)
        startprob[0] = 1.0

        # Base transition matrix with cycle structure
        transmat = np.zeros((n_components, n_components))
        forward = self.cycle_strength
        stay = 1.0 - forward
        for idx in range(n_components):
            transmat[idx, idx] = stay
            transmat[idx, (idx + 1) % n_components] = forward

        # Store orientations for per-frame transition adjustment during prediction
        # The transition matrix will be adjusted dynamically based on orientation changes
        # For now, use average change to set base transition probabilities
        if orientations is not None and len(orientations) == len(reduced_embeddings):
            orientation_changes = self._compute_orientation_changes(orientations)
            avg_change = np.mean(orientation_changes[1:]) if len(orientation_changes) > 1 else 0.0
            
            # Adjust base transition probabilities based on average orientation stability
            # If orientation changes frequently, allow more transitions
            # If orientation is stable, prefer staying in current state
            if avg_change > self.orientation_threshold:
                # High change: allow transitions (orientation is changing, so states can change)
                transition_factor = 1.0
            else:
                # Low change: reduce transition probability (orientation stable, stay in state)
                transition_factor = 0.2
            
            # Adjust transition matrix
            for idx in range(n_components):
                transmat[idx, idx] = stay + (1.0 - transition_factor) * forward
                transmat[idx, (idx + 1) % n_components] = forward * transition_factor

        model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            init_params="",
            params="",
            n_iter=0,
        )

        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means
        model.covars_ = variances

        return model

    def _smooth_state_sequence(
        self,
        state_sequence: List[str],
        orientations: Optional[np.ndarray],
    ) -> List[str]:
        """
        Post-process state sequence using orientation changes to prevent rapid cycling.
        Only allows state transitions when orientation changes significantly.
        """
        if orientations is None or len(orientations) != len(state_sequence):
            return state_sequence
        
        if len(state_sequence) < 2:
            return state_sequence
        
        orientation_changes = self._compute_orientation_changes(orientations)
        smoothed_sequence = [state_sequence[0]]
        
        for i in range(1, len(state_sequence)):
            current_state = state_sequence[i]
            prev_state = smoothed_sequence[-1]
            
            # If state changed, check if orientation change is significant enough
            if current_state != prev_state:
                orientation_change = orientation_changes[i]
                
                # Only allow transition if orientation changed significantly
                if orientation_change > self.orientation_threshold:
                    smoothed_sequence.append(current_state)
                else:
                    # Orientation didn't change enough - keep previous state
                    smoothed_sequence.append(prev_state)
            else:
                # State didn't change - keep it
                smoothed_sequence.append(current_state)
        
        return smoothed_sequence

    def infer(
        self,
        embeddings: np.ndarray,
        timestamps: Sequence[float],
        orientations: Optional[np.ndarray] = None,
    ) -> HMMResult:
        if len(embeddings) != len(timestamps):
            raise ValueError("Embeddings and timestamps must have the same length.")
        
        if orientations is not None and len(orientations) != len(embeddings):
            raise ValueError("Orientations must have the same length as embeddings.")

        reduced = self._reduce(embeddings)
        if reduced.size == 0:
            return HMMResult([], reduced, timestamps, orientations=orientations)

        self.model = self._build_model(reduced, orientations=orientations)
        state_indices = self.model.predict(reduced)
        state_sequence = [STATE_SYMBOLS[index] for index in state_indices]
        
        # Apply orientation-based smoothing to prevent rapid cycling
        state_sequence = self._smooth_state_sequence(state_sequence, orientations)

        return HMMResult(state_sequence, reduced, timestamps, orientations=orientations)


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


