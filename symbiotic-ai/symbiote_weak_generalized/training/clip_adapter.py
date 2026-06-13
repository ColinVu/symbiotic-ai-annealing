"""
Small residual MLP adapter on top of frozen CLIP embeddings.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPAdapter(nn.Module):
    """
    MLP: Linear(D,D) -> ReLU -> Linear(D,D) with residual ``out = x + mlp(x)``.
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        d = int(embed_dim)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return x + self.fc2(h)


def _l2_normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)
    return x / n


def _spherical_mean_np(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("spherical_mean: empty")
    norms = np.linalg.norm(arr, axis=1)
    mask = norms >= eps
    if not np.any(mask):
        return np.zeros(arr.shape[1], dtype=np.float64)
    unit = arr[mask] / norms[mask, np.newaxis]
    s = unit.mean(axis=0)
    ns = float(np.linalg.norm(s))
    if ns < eps:
        return np.zeros(arr.shape[1], dtype=np.float64)
    return (s / ns).astype(np.float64)


# def train_clip_adapter(
#     frame_embeddings: np.ndarray,
#     frame_labels: np.ndarray,
#     *,
#     embed_dim: int,
#     epochs: int,
#     batch_size: int,
#     lr: float,
#     triplet_margin: float,
#     device: Optional[str] = None,
#     verbose: bool = False,
#     model_in: Optional[CLIPAdapter] = None,
# ) -> Tuple[CLIPAdapter, Dict[str, float]]:
#     """
#     Train CLIPAdapter with TripletMarginLoss.

#     Triplets (hard negative by centroid):
#       - anchor: random frame embedding
#       - positive: spherical mean of all frames with same label (recomputed periodically)
#       - negative: centroid of the *wrong* label with smallest cosine distance to anchor

#     Args:
#         frame_embeddings: (N, D)
#         frame_labels: (N,) int class indices aligned with rows of frame_embeddings
#     """
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#     dev = torch.device(device)

#     X = np.asarray(frame_embeddings, dtype=np.float64)
#     y = np.asarray(frame_labels, dtype=np.int64)
#     if X.ndim != 2:
#         raise ValueError("frame_embeddings must be (N, D)")
#     if y.shape[0] != X.shape[0]:
#         raise ValueError("frame_labels length must match number of rows")

#     n, d = X.shape
#     if int(embed_dim) != d:
#         raise ValueError(f"embed_dim {embed_dim} does not match data dim {d}")

#     classes = sorted(int(c) for c in np.unique(y))
#     if len(classes) < 2:
#         if verbose:
#             print("  [CLIPAdapter] Need >=2 classes for triplet training; skipping update.")
#         model = (model_in if model_in is not None else CLIPAdapter(d)).to(dev)
#         return model.cpu(), {"loss": 0.0, "skipped": 1.0}

#     model = (model_in if model_in is not None else CLIPAdapter(d)).to(dev)
#     opt = torch.optim.Adam(model.parameters(), lr=float(lr))
#     loss_fn = nn.TripletMarginLoss(margin=float(triplet_margin), p=2)

#     X_t = torch.from_numpy(X).float().to(dev)
#     y_t = torch.from_numpy(y).long().to(dev)

#     def class_centroids() -> Tuple[torch.Tensor, torch.Tensor]:
#         """Returns C (K, D) and class_ids (K,) for labels present."""
#         vecs = []
#         ids = []
#         for c in classes:
#             mask = y_t == c
#             if not torch.any(mask):
#                 continue
#             sm = _spherical_mean_np(X_t[mask].detach().cpu().numpy())
#             vecs.append(torch.from_numpy(sm).float().to(dev))
#             ids.append(c)
#         return torch.stack(vecs, dim=0), torch.tensor(ids, device=dev, dtype=torch.long)

#     last_loss = 0.0
#     for ep in range(int(epochs)):
#         C, c_ids = class_centroids()
#         # shuffle indices
#         perm = torch.randperm(n, device=dev)
#         ep_loss = 0.0
#         steps = 0
#         bs = max(1, int(batch_size))
#         for start in range(0, n, bs):
#             idx = perm[start : start + bs]
#             anchor_raw = X_t[idx]
#             y_anchor = y_t[idx]

#             with torch.no_grad():
#                 pos_list = []
#                 neg_list = []
#                 for j in range(idx.shape[0]):
#                     c = int(y_anchor[j].item())
#                     pos_mask = c_ids == c
#                     if not torch.any(pos_mask):
#                         pos_list.append(anchor_raw[j])
#                     else:
#                         pos_list.append(C[pos_mask][0])
#                     # negatives: centroids of classes != c (hard = closest wrong centroid)
#                     other_mask = c_ids != c
#                     if not torch.any(other_mask):
#                         neg_list.append(torch.zeros(d, device=dev))
#                         continue
#                     C_other = C[other_mask]
#                     a = _l2_normalize_rows(anchor_raw[j : j + 1])
#                     oc = _l2_normalize_rows(C_other)
#                     sim = (oc @ a.T).squeeze(-1)
#                     dist = 1.0 - sim
#                     kmin = int(torch.argmin(dist))
#                     neg_list.append(C_other[kmin])
#                 positive = torch.stack(pos_list, dim=0)
#                 negative = torch.stack(neg_list, dim=0)

#             opt.zero_grad()
#             a_out = model(anchor_raw)
#             p_out = model(positive)
#             n_out = model(negative)
#             a_n = _l2_normalize_rows(a_out)
#             p_n = _l2_normalize_rows(p_out)
#             n_n = _l2_normalize_rows(n_out)
#             loss = loss_fn(a_n, p_n, n_n)
#             loss.backward()
#             opt.step()
#             ep_loss += float(loss.item())
#             steps += 1
#         last_loss = ep_loss / max(steps, 1)
#         if verbose and (ep + 1) % max(1, epochs // 5) == 0:
#             print(f"  [CLIPAdapter] epoch {ep+1}/{epochs} mean_loss={last_loss:.6f}")

#     return model.cpu(), {"loss": last_loss}

def train_clip_adapter(
    frame_embeddings: np.ndarray,
    frame_labels: np.ndarray,
    *,
    embed_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    triplet_margin: float,
    device: Optional[str] = None,
    verbose: bool = False,
    model_in: Optional[CLIPAdapter] = None,
) -> Tuple[CLIPAdapter, Dict[str, float]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    X_t = torch.from_numpy(frame_embeddings).float().to(dev)
    y_t = torch.from_numpy(frame_labels).long().to(dev)
    n, d = X_t.shape
    classes = torch.unique(y_t)

    model = (model_in if model_in is not None else CLIPAdapter(d)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.TripletMarginLoss(margin=float(triplet_margin), p=2)

    # Optimized GPU Centroid Calculation
    def get_gpu_centroids(embeddings, labels):
        vecs = []
        for c in classes:
            mask = (labels == c)
            if not torch.any(mask): continue
            # Spherical mean on GPU: Average -> Normalize
            m = embeddings[mask].mean(dim=0)
            vecs.append(F.normalize(m, p=2, dim=0))
        return torch.stack(vecs), classes

    last_loss = 0.0
    for ep in range(int(epochs)):
        model.eval()
        with torch.no_grad():
            # Get current centroids in the adapted space
            adapted_X = model(X_t)
            C, c_ids = get_gpu_centroids(adapted_X, y_t)
            
            # Map labels to centroid indices for fast lookup
            label_to_idx = {val.item(): i for i, val in enumerate(c_ids)}
            target_centroid_idx = torch.tensor([label_to_idx[ly.item()] for ly in y_t], device=dev)

        model.train()
        perm = torch.randperm(n, device=dev)
        ep_loss, steps = 0.0, 0
        bs = max(1, int(batch_size))

        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            anchors = X_t[idx]
            
            # Vectorized Triplet Mining (No Python Loops)
            with torch.no_grad():
                current_adapted = model(anchors)
                # Compute distance matrix between batch and ALL centroids
                # dist = 1 - cos_sim
                logits = torch.mm(F.normalize(current_adapted, p=2, dim=1), C.t())
                dists = 1.0 - logits
                
                # Positives: The centroid for the anchor's actual label
                pos_idx = target_centroid_idx[idx]
                positives = C[pos_idx]
                
                # Negatives: Closest centroid that ISN'T the correct label
                mask = torch.ones_like(dists, dtype=torch.bool)
                mask[torch.arange(idx.size(0)), pos_idx] = False
                # Fill correct slots with infinity to find min of others
                dists_masked = dists.clone()
                dists_masked[~mask] = float('inf')
                neg_idx = torch.argmin(dists_masked, dim=1)
                negatives = C[neg_idx]

            opt.zero_grad()
            a_out = F.normalize(model(anchors), p=2, dim=1)
            p_out = F.normalize(model(positives), p=2, dim=1)
            n_out = F.normalize(model(negatives), p=2, dim=1)
            
            loss = loss_fn(a_out, p_out, n_out)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            steps += 1
            
        last_loss = ep_loss / max(steps, 1)
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  [CLIPAdapter] epoch {ep+1}/{epochs} loss={last_loss:.6f}")

    return model.cpu(), {"loss": last_loss}


@torch.no_grad()
def apply_adapter_to_numpy(
    model: CLIPAdapter,
    embeddings: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    """Apply adapter to (N,D) or (D,) numpy; returns float64 numpy same shape."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    em = np.asarray(embeddings, dtype=np.float64)
    single = em.ndim == 1
    if single:
        em = em.reshape(1, -1)
    m = model.to(dev)
    x = torch.from_numpy(em).float().to(dev)
    y = m(x).detach().cpu().numpy().astype(np.float64)
    if single:
        return y[0]
    return y


__all__ = ["CLIPAdapter", "train_clip_adapter", "apply_adapter_to_numpy"]
