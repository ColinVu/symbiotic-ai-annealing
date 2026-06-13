"""
Weakly supervised training (ILR) - Fast Epoch-Refreshed Logic.
Uses once-per-epoch energy mapping for high performance.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.cluster import KMeans

LabelKey = Tuple[str, int]
EMPTY_HAND_LABEL = "empty_hand"

def spherical_mean(vectors: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Mean direction on the unit sphere."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        raise ValueError("spherical_mean: empty input")
    norms = np.linalg.norm(arr, axis=1)
    mask = (norms >= eps) & np.isfinite(norms)
    if not np.any(mask):
        return np.zeros(arr.shape[1], dtype=np.float64)
    unit = arr[mask] / norms[mask, np.newaxis]
    s = unit.mean(axis=0)
    ns = float(np.linalg.norm(s))
    if ns < eps or not np.isfinite(ns):
        return np.zeros(arr.shape[1], dtype=np.float64)
    return (s / ns).astype(np.float64)

@dataclass
class Segment:
    """A video segment containing multiple frame embeddings."""
    segment_id: int
    embeddings: np.ndarray
    video_id: str
    candidate_labels: Optional[Tuple[str, ...]] = None
    is_placeholder: bool = False

    @property
    def label_key(self) -> LabelKey:
        return (self.video_id, self.segment_id)

    def compute_frame_costs(self, centroid: np.ndarray, cosine_distance_fn: Callable) -> float:
        em = np.asarray(self.embeddings, dtype=np.float64)
        if em.size == 0: return 0.0
        c = np.asarray(centroid, dtype=np.float64)
        return float(sum(cosine_distance_fn(em[i], c) for i in range(em.shape[0])))

class WeakSupervisedTrainer:
    def __init__(
        self,
        ilr_epochs: int = 500,
        initial_temp: float = 1.0,
        temp_decay: str = "exponential",
        decay_rate: float = 0.99,
        random_seed: int = 42,
        variance_eps: float = 1e-6,
        bad_swap_cool_divisor: float = 750.0,
        detect_empty: bool = False,
        min_frames_per_cluster: int = 3,
        ilr_allow_cross_round_swaps: bool = False,
        min_temp: float = 0.05,
    ):
        self.ilr_epochs = ilr_epochs
        self.initial_temp = initial_temp
        self.temp_decay = temp_decay
        self.decay_rate = decay_rate
        self.random_seed = random_seed
        self.variance_eps = variance_eps
        self.bad_swap_cool_divisor = bad_swap_cool_divisor
        self.detect_empty = detect_empty
        self.min_frames_per_cluster = min_frames_per_cluster
        self.ilr_allow_cross_round_swaps = ilr_allow_cross_round_swaps
        self.min_temp = min_temp

        self.centroids: Dict[str, np.ndarray] = {}
        self.centroid_stds: Dict[str, np.ndarray] = {}
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self.label_video_means: Dict[str, Dict[str, np.ndarray]] = {}
        self.hand_neutralizer_state: Optional[Dict[str, Any]] = None
        # Metric-learning residual MLP (optional; used when use_iterated_model=True)
        self.adapter_model: Optional[Any] = None
        # Alias for persistence (model_io saves ``clip_adapter``)
        self.clip_adapter: Optional[Any] = None
        # Final segment labels after fit(); used by sweep / assignment_score
        self.last_refined_labels: Optional[Dict[LabelKey, str]] = None

        random.seed(random_seed)
        np.random.seed(random_seed)

    def _l2_normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = self._l2_normalize(a.reshape(1, -1))[0]
        b_norm = self._l2_normalize(b.reshape(1, -1))[0]
        return 1 - np.dot(a_norm, b_norm)

    def compute_centroids(self, segments: List[Segment], labels: Dict[LabelKey, str]) -> Dict[str, np.ndarray]:
        label_frames = defaultdict(list)
        for seg in segments:
            if not seg.is_placeholder:
                label_frames[labels[seg.label_key]].append(seg.embeddings)
        return {l: spherical_mean(np.vstack(b)) for l, b in label_frames.items() if b}

    def compute_total_cosine_cost(self, segments: List[Segment], labels: Dict[LabelKey, str], centroids: Dict[str, np.ndarray]) -> float:
        total = 0.0
        for seg in segments:
            if not seg.is_placeholder:
                total += seg.compute_frame_costs(centroids[labels[seg.label_key]], self.cosine_distance)
        return float(total)

    def refine_labels(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
        verbose: bool = True,
    ) -> Dict[LabelKey, str]:
        labels = labels.copy()
        real_segments = [seg for seg in segments if not seg.is_placeholder]
        videos = defaultdict(list)
        for seg in real_segments:
            videos[seg.video_id].append(seg)

        # 1. INITIAL STATS
        centroids = self.compute_centroids(real_segments, labels)
        best_cost = self.compute_total_cosine_cost(real_segments, labels, centroids)
        best_labels = labels.copy()
        
        if verbose:
            print(f"\nInitial cost: {best_cost:.4f}")

        video_ids = list(videos.keys())
        class_names = sorted(centroids.keys())

        for epoch in range(self.ilr_epochs):
            temp = self._get_temperature(epoch)
            random.shuffle(video_ids)

            # 2. VECTORIZED ENERGY MAPPING (Massive Speedup)
            centroids = self.compute_centroids(real_segments, labels)
            # Stack all centroids into (K, D) matrix
            C_matrix = np.stack([centroids[c] for c in class_names]) 
            
            energy_map = {}
            for s in real_segments:
                # Vectorized Cost: 1 - (Frames @ Centroids^T)
                # Results in (T, K) matrix; sum over T frames to get (K,) costs
                costs_per_class = np.sum(1.0 - (s.embeddings @ C_matrix.T), axis=0)
                energy_map[s.label_key] = dict(zip(class_names, costs_per_class))

            for vid_id in video_ids:
                video_segments = videos[vid_id]
                if len(video_segments) < 2: continue
                
                for seg1 in video_segments:
                    lk1 = seg1.label_key
                    lab1 = labels[lk1]
                    
                    dr_by_key = {}
                    for seg2 in video_segments:
                        lk2 = seg2.label_key
                        lab2 = labels[lk2]
                        if seg1 is seg2 or lab1 == lab2: continue
                        
                        if seg1.candidate_labels and lab2 not in seg1.candidate_labels: continue
                        if seg2.candidate_labels and lab1 not in seg2.candidate_labels: continue

                        # O(1) Dictionary Lookup
                        reduction = (energy_map[lk1][lab1] + energy_map[lk2][lab2]) - \
                                    (energy_map[lk1][lab2] + energy_map[lk2][lab1])
                        dr_by_key[lk2] = reduction

                    if not dr_by_key: continue
                    best_lk = max(dr_by_key, key=dr_by_key.get)
                    delta = -dr_by_key[best_lk]

                    if delta < 0 or self._accept_swap(delta, temp):
                        labels[lk1], labels[best_lk] = labels[best_lk], labels[lk1]

            # Track global best
            current_cost = sum(energy_map[lk][labels[lk]] for lk in labels if lk in energy_map)
            if current_cost < best_cost:
                best_cost, best_labels = current_cost, labels.copy()

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:4d}: cost={current_cost:.4f} | temp={temp:.3f}")

        return best_labels
    
    def _get_temperature(self, epoch: int) -> float:
        if self.temp_decay == "exponential":
            t = self.initial_temp * (self.decay_rate**epoch)
        elif self.temp_decay == "linear":
            t = self.initial_temp * (1 - epoch / self.ilr_epochs)
        elif self.temp_decay == "cosine":
            # Cosine Annealing: stays warmer longer, then drops toward min_temp at the end
            t = self.min_temp + 0.5 * (self.initial_temp - self.min_temp) * \
                (1 + math.cos(math.pi * epoch / self.ilr_epochs))
        else:
            t = self.initial_temp * (self.decay_rate**epoch)
            
        # Ensure temperature never drops below the 'jiggle' threshold
        return max(t, self.min_temp)

    def _accept_swap(self, delta_cost: float, temperature: float) -> bool:
        if delta_cost < 0: return True
        if temperature <= 0: return False
        p_accept = np.exp(-delta_cost / max(temperature, 1e-10))
        return random.random() < p_accept

    def _fit_with_adapter_loop(
        self,
        all_segments: List[Segment],
        initial_labels: Dict[LabelKey, str],
        config: Dict[str, Any],
        *,
        verbose: bool = True,
    ) -> Dict[LabelKey, str]:
        """
        Multi-loop ILR: SA (refine_labels) -> flatten frames -> train CLIPAdapter ->
        apply adapter + L2 normalize -> repeat; then one final SA pass.
        """
        from .clip_adapter import CLIPAdapter, apply_adapter_to_numpy, train_clip_adapter
        from .geometry import l2_normalize_rows

        refinement_loops = int(config.get("refinement_loops", 3))
        adapter_epochs = int(config.get("adapter_epochs", 10))
        adapter_lr = float(config.get("adapter_lr", 1e-3))
        adapter_batch_size = int(config.get("adapter_batch_size", 32))
        triplet_margin = float(config.get("triplet_margin", 0.1))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        real_segments = [seg for seg in all_segments if not seg.is_placeholder]
        refined: Dict[LabelKey, str] = dict(initial_labels)

        embed_dim = 512
        for seg in real_segments:
            em = np.asarray(seg.embeddings, dtype=np.float64)
            if em.ndim >= 2 and em.shape[-1] > 0:
                embed_dim = int(em.shape[-1])
                break
            if em.ndim == 1 and em.size > 0:
                embed_dim = int(em.shape[0])
                break

        if verbose:
            print("\n" + "=" * 60)
            print("ITERATED MODEL (SA -> CLIPAdapter -> space warp)")
            print("=" * 60)
            print(f"Refinement loops: {refinement_loops}")
            print(f"Embedding dim: {embed_dim}")
            print(f"Adapter epochs: {adapter_epochs}, batch_size: {adapter_batch_size}")

        for loop_idx in range(refinement_loops):
            if verbose:
                print(f"\n--- Refinement loop {loop_idx + 1}/{refinement_loops} ---")

            # Step 1: SA — sync centroids before SA (refine_labels also refreshes per epoch)
            self.centroids = self.compute_centroids(real_segments, refined)
            refined = self.refine_labels(all_segments, refined, verbose=verbose)

            # Step 2: flatten (N, D) frames and integer labels
            frames_list: List[np.ndarray] = []
            labels_list: List[int] = []
            for seg in real_segments:
                em = np.asarray(seg.embeddings, dtype=np.float64)
                if em.size == 0:
                    continue
                label_str = refined[seg.label_key]
                label_idx = int(self.label_to_idx[label_str])
                if em.ndim == 1:
                    frames_list.append(em)
                    labels_list.append(label_idx)
                else:
                    for frame_idx in range(em.shape[0]):
                        frames_list.append(em[frame_idx])
                        labels_list.append(label_idx)

            if len(frames_list) < 2:
                if verbose:
                    print("  [Adapter] Not enough frames for training; stopping loop.")
                break

            X_frames = np.stack(frames_list, axis=0)
            y_frames = np.asarray(labels_list, dtype=np.int64)

            if np.unique(y_frames).size < 2:
                if verbose:
                    print("  [Adapter] Need >= 2 classes for triplet training; stopping loop.")
                break

            if verbose:
                print(f"  [Adapter] Training on {X_frames.shape[0]} frames, dim={X_frames.shape[1]}")

            # Step 3: metric learning (continue from self.adapter_model if set)
            if self.adapter_model is None:
                self.adapter_model = CLIPAdapter(embed_dim)
            elif int(embed_dim) != int(self.adapter_model.fc1.in_features):
                raise ValueError(
                    f"adapter embed dim {self.adapter_model.fc1.in_features} != data dim {embed_dim}"
                )

            self.adapter_model, train_info = train_clip_adapter(
                X_frames,
                y_frames,
                embed_dim=embed_dim,
                epochs=adapter_epochs,
                batch_size=adapter_batch_size,
                lr=adapter_lr,
                triplet_margin=triplet_margin,
                device=device,
                verbose=verbose,
                model_in=self.adapter_model,
            )
            self.clip_adapter = self.adapter_model

            if verbose:
                print(f"  [Adapter] Train loss (last epoch avg): {train_info.get('loss', 0.0):.6f}")

            # Step 4: warp embedding space + L2 normalize rows
            self.adapter_model.eval()
            with torch.no_grad():
                for seg in real_segments:
                    em = np.asarray(seg.embeddings, dtype=np.float64)
                    if em.size == 0:
                        continue
                    adapted = apply_adapter_to_numpy(self.adapter_model, em, device=device)
                    seg.embeddings = l2_normalize_rows(adapted)

            self.adapter_model = self.adapter_model.cpu()
            self.clip_adapter = self.adapter_model

            self.centroids = self.compute_centroids(real_segments, refined)
            if verbose:
                current_cost = self.compute_total_cosine_cost(real_segments, refined, self.centroids)
                print(f"  [Loop {loop_idx + 1}] Energy after adaptation: {current_cost:.4f}")

        if verbose:
            print("\n--- Final refinement pass ---")

        self.centroids = self.compute_centroids(real_segments, refined)
        refined = self.refine_labels(all_segments, refined, verbose=verbose)

        if self.adapter_model is not None:
            self.clip_adapter = self.adapter_model

        return refined

    def fit(self, video_segments, verbose: bool = True, skip_ilr: bool = False, use_cluster_voting: bool = False, initial_cluster_voting_csv: Optional[str] = None, **kwargs) -> "WeakSupervisedTrainer":
        all_segments = []
        unique_labels = set()
        for vid, (segs, picklist) in video_segments.items():
            for s in segs:
                if not s.candidate_labels: s.candidate_labels = tuple(picklist)
                all_segments.append(s)
                unique_labels.update(s.candidate_labels)
        
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        if use_cluster_voting:
            from .cluster_voting import cluster_based_initialization_with_details
            labels, label_confidence = cluster_based_initialization_with_details(all_segments, None, verbose=verbose)
            if initial_cluster_voting_csv:
                from .cluster_voting import write_initial_cluster_voting_matrix_csv
                write_initial_cluster_voting_matrix_csv(initial_cluster_voting_csv, labels, label_confidence, video_segments)
        else:
            labels = {}
            groups = defaultdict(list)
            for s in all_segments: groups[(s.video_id, s.candidate_labels)].append(s)
            for (_, multiset), segs in groups.items():
                draw = list(multiset)
                random.shuffle(draw)
                for s, l in zip(sorted(segs, key=lambda x: x.segment_id), draw): labels[s.label_key] = l

        if not skip_ilr:
            if bool(kwargs.get("use_iterated_model", False)):
                refined = self._fit_with_adapter_loop(
                    all_segments,
                    labels,
                    kwargs,
                    verbose=verbose,
                )
            else:
                refined = self.refine_labels(all_segments, labels, verbose=verbose)
        else:
            refined = labels

        self.last_refined_labels = dict(refined)
        self.centroids = self.compute_centroids(all_segments, refined)
        return self

    def predict(self, embedding: np.ndarray) -> str:
        if not self.centroids: raise ValueError("Model not fitted. Call fit() first.")
        transformed = self._l2_normalize(embedding.reshape(1, -1))[0]
        best_label, best_distance = None, float("inf")
        for label, centroid in self.centroids.items():
            dist = self.cosine_distance(transformed, centroid)
            if dist < best_distance:
                best_distance, best_label = dist, label
        return best_label

    def predict_proba(self, embedding: np.ndarray) -> Dict[str, float]:
        if not self.centroids: raise ValueError("Model not fitted. Call fit() first.")
        transformed = self._l2_normalize(embedding.reshape(1, -1))[0]
        distances = {l: self.cosine_distance(transformed, c) for l, c in self.centroids.items()}
        neg_distances = {label: -d for label, d in distances.items()}
        max_neg = max(neg_distances.values())
        exp_scores = {label: np.exp(nd - max_neg) for label, nd in neg_distances.items()}
        total = sum(exp_scores.values())
        return {label: score / total for label, score in exp_scores.items()}

__all__ = ["Segment", "WeakSupervisedTrainer", "LabelKey", "EMPTY_HAND_LABEL", "spherical_mean"]