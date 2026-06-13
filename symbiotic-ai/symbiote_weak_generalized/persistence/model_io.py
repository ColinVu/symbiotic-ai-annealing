"""Model persistence utilities for centroid-based weakly supervised models."""

import os
import json
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import numpy as np
import torch

from ..core.config import MODEL
from ..models.classifier import CentroidModel
from ..training.weak_supervision import WeakSupervisedTrainer


def save_model(
    trainer: WeakSupervisedTrainer,
    config: Dict[str, Any],
    output_dir: str,
    append_embedded_video_stem: Optional[str] = None,
    embedded_video_stems_override: Optional[List[str]] = None,
):
    """
    Save trained centroid model and metadata.

    Saves:
        - centroids.npy: Centroid vectors for each label (unit-norm CLIP directions)
        - centroid_stds.npy: Per-label std (legacy / fit_iterative), if present
        - label_video_means.json: Optional; per-label, per-video spherical means
        - model_metadata.json: Label mappings, config, dimensions
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    prev_embedded: List[str] = []
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            prev_meta = json.load(f)
            prev_embedded = list(prev_meta.get("embedded_video_stems", []))
    if embedded_video_stems_override is not None:
        seen = set()
        stems = []
        for s in embedded_video_stems_override:
            if s not in seen:
                seen.add(s)
                stems.append(s)
    else:
        stems = list(prev_embedded)
        if append_embedded_video_stem and append_embedded_video_stem not in stems:
            stems.append(append_embedded_video_stem)

    centroid_labels = sorted(trainer.centroids.keys())
    centroids_array = np.array([trainer.centroids[label] for label in centroid_labels])
    centroids_path = os.path.join(output_dir, "centroids.npy")
    np.save(centroids_path, centroids_array)

    has_std = bool(getattr(trainer, "centroid_stds", None)) and all(
        label in trainer.centroid_stds for label in centroid_labels
    )
    if has_std:
        std_array = np.array([trainer.centroid_stds[label] for label in centroid_labels])
        std_path = os.path.join(output_dir, "centroid_stds.npy")
        np.save(std_path, std_array)

    lvm = getattr(trainer, "label_video_means", None) or {}
    has_lvm = bool(lvm)

    metadata = {
        "label_to_idx": trainer.label_to_idx,
        "idx_to_label": {str(k): v for k, v in trainer.idx_to_label.items()},
        "centroid_labels": centroid_labels,
        "embedding_dim": int(centroids_array.shape[1]) if centroids_array.size else 0,
        "num_classes": len(trainer.centroids),
        "config": config,
        "clip_model": MODEL,
        "model_type": "centroid_cosine",
        "has_centroid_stds": has_std,
        "has_label_video_means": has_lvm,
        "embedded_video_stems": stems,
        "has_clip_adapter": bool(
            getattr(trainer, "clip_adapter", None) is not None
            or getattr(trainer, "adapter_model", None) is not None
        ),
        "has_hand_neutralizer": bool(getattr(trainer, "hand_neutralizer_state", None)),
        "use_iterated_model": bool(config.get("use_iterated_model", False)),
    }

    if has_lvm:
        serializable = {
            lab: {vid: np.asarray(vec, dtype=np.float64).tolist() for vid, vec in by_vid.items()}
            for lab, by_vid in lvm.items()
        }
        lvm_path = os.path.join(output_dir, "label_video_means.json")
        with open(lvm_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    assignments_path = os.path.join(output_dir, "final_assignments.csv")
    try:
        from ..experiments.assignment_score import write_final_assignments_csv

        gt_csv = config.get("ground_truth_csv")
        n_rows = write_final_assignments_csv(
            assignments_path,
            trainer,
            ground_truth_csv=str(gt_csv) if gt_csv else None,
            video_stems_order=stems if stems else None,
        )
        print(f"  - final_assignments.csv ({n_rows} segments)")
    except RuntimeError as e:
        print(f"  (skipped final_assignments.csv: {e})")

    _adapter = getattr(trainer, "adapter_model", None) or getattr(trainer, "clip_adapter", None)
    if _adapter is not None:
        ap = os.path.join(output_dir, "clip_adapter.pt")
        torch.save(_adapter.state_dict(), ap)
        print(f"  - clip_adapter.pt")

    hns = getattr(trainer, "hand_neutralizer_state", None)
    if hns:
        hpath = os.path.join(output_dir, "hand_neutralizer.json")
        with open(hpath, "w", encoding="utf-8") as f:
            json.dump(hns, f, indent=2)
        print(f"  - hand_neutralizer.json")

    print(f"\nModel saved to: {output_dir}")
    print(f"  - centroids.npy")
    if metadata["has_centroid_stds"]:
        print(f"  - centroid_stds.npy")
    if has_lvm:
        print(f"  - label_video_means.json")
    print(f"  - model_metadata.json")


def load_weak_trainer(
    model_dir: str,
    ilr_epochs_override: Optional[int] = None,
    random_seed_override: Optional[int] = None,
) -> WeakSupervisedTrainer:
    """
    Reload a ``WeakSupervisedTrainer`` from disk (for ``fit_iterative`` / incremental).
    """
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata["idx_to_label"] = {int(k): v for k, v in metadata["idx_to_label"].items()}
    cfg = metadata.get("config", {})

    trainer = WeakSupervisedTrainer(
        ilr_epochs=ilr_epochs_override if ilr_epochs_override is not None else int(cfg.get("ilr_epochs", 500)),
        initial_temp=float(cfg.get("initial_temp", 1.0)),
        temp_decay=str(cfg.get("temp_decay", "exponential")),
        decay_rate=float(cfg.get("decay_rate", 0.99)),
        random_seed=random_seed_override if random_seed_override is not None else int(cfg.get("random_seed", 42)),
        variance_eps=float(cfg.get("variance_eps", 1e-6)),
        bad_swap_cool_divisor=float(cfg.get("bad_swap_cool_divisor", 50.0)),
        detect_empty=bool(cfg.get("detect_empty", False)),
        min_frames_per_cluster=int(cfg.get("min_frames_per_cluster", 3)),
        ilr_allow_cross_round_swaps=bool(cfg.get("ilr_allow_cross_round_swaps", False)),
        min_temp=float(cfg.get("min_temp", 0.05)),
    )

    centroid_labels = metadata["centroid_labels"]
    centroids_arr = np.load(os.path.join(model_dir, "centroids.npy"))
    trainer.centroids = {
        label: centroids_arr[i] for i, label in enumerate(centroid_labels)
    }
    trainer.label_to_idx = metadata["label_to_idx"]
    trainer.idx_to_label = metadata["idx_to_label"]

    std_path = os.path.join(model_dir, "centroid_stds.npy")
    if os.path.isfile(std_path):
        std_arr = np.load(std_path)
        trainer.centroid_stds = {
            label: std_arr[i] for i, label in enumerate(centroid_labels)
        }
    else:
        dim = centroids_arr.shape[1]
        trainer.centroid_stds = {label: np.ones(dim) for label in centroid_labels}

    trainer.embedded_video_stems = list(metadata.get("embedded_video_stems", []))

    lvm_path = os.path.join(model_dir, "label_video_means.json")
    if os.path.isfile(lvm_path):
        with open(lvm_path, "r", encoding="utf-8") as f:
            raw_lvm = json.load(f)
        trainer.label_video_means = {
            lab: {vid: np.asarray(vec, dtype=np.float64) for vid, vec in by_vid.items()}
            for lab, by_vid in raw_lvm.items()
        }
    else:
        trainer.label_video_means = {}

    from ..training.clip_adapter import CLIPAdapter

    ap = os.path.join(model_dir, "clip_adapter.pt")
    if os.path.isfile(ap):
        dim = int(centroids_arr.shape[1])
        m = CLIPAdapter(dim)
        m.load_state_dict(torch.load(ap, map_location="cpu"))
        m.eval()
        trainer.clip_adapter = m
        trainer.adapter_model = m
    else:
        trainer.clip_adapter = None
        trainer.adapter_model = None

    hn_path = os.path.join(model_dir, "hand_neutralizer.json")
    if os.path.isfile(hn_path):
        with open(hn_path, "r", encoding="utf-8") as f:
            trainer.hand_neutralizer_state = json.load(f)
    else:
        trainer.hand_neutralizer_state = None

    return trainer


def load_model(model_dir: str) -> Tuple[CentroidModel, Dict[str, Any]]:
    """
    Load trained centroid model and metadata for inference.
    """
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata["idx_to_label"] = {int(k): v for k, v in metadata["idx_to_label"].items()}

    centroids_path = os.path.join(model_dir, "centroids.npy")
    centroids_array = np.load(centroids_path)

    centroid_labels = metadata["centroid_labels"]
    centroids = {label: centroids_array[i] for i, label in enumerate(centroid_labels)}

    model = CentroidModel(
        centroids=centroids,
        label_to_idx=metadata["label_to_idx"],
        idx_to_label=metadata["idx_to_label"],
    )

    return model, metadata


__all__ = ["save_model", "load_model", "load_weak_trainer"]
