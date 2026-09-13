"""Train a per-frame state classifier with cross-entropy.

Frame-level CE is chosen over CTC deliberately: alignment happens afterwards
in hmm_align, and Viterbi needs dense, calibrated posteriors on every frame.
CTC supervises only the label ordering and leaves the frames between spikes
unconstrained, which is what made the boundaries unusable.

Class priors from the training labels are saved into the checkpoint, because
the HMM emission model needs them to convert posteriors to likelihoods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data import (
    DEFAULT_CSV_OUTPUTS,
    DEFAULT_FEATURES_3D,
    DEFAULT_GROUND_TRUTH,
    DEFAULT_LABELS,
    DEFAULT_PICKLIST_JSONS,
    DEFAULT_PICKLIST_LABELS,
    DEFAULT_VIDEO_DIR,
    NUM_STATES,
    FeatureSource,
    compute_label_priors,
    compute_normalization_stats,
    discover_picklist_ids,
    feature_dim_for_source,
    load_dataset,
    normalize_features,
    train_val_split,
)
from .hmm_align import duration_stats, estimate_self_loop_logprobs
from .models import build_model
from .states import STATES

IGNORE_INDEX = -1


class FrameDataset(Dataset):
    """Yields (features, dense frame labels, picklist_id)."""

    def __init__(self, samples, mean: np.ndarray, std: np.ndarray):
        self.samples = samples
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = normalize_features(s.features, self.mean, self.std).astype(np.float32)
        y = s.labels.astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y), s.picklist_id


def collate_batch(batch):
    """Pad features with zeros and labels with IGNORE_INDEX to the batch max."""
    xs, ys, ids = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    d = xs[0].shape[1]
    x_pad = torch.zeros(len(xs), max_len, d)
    y_pad = torch.full((len(ys), max_len), IGNORE_INDEX, dtype=torch.long)
    lengths = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        t = x.shape[0]
        x_pad[i, :t] = x
        y_pad[i, :t] = y[:t]
        lengths.append(t)
    return x_pad, y_pad, torch.tensor(lengths), list(ids)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, n = 0.0, 0
    for x, y, _, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)                                  # (N, T, C)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """Returns (mean loss, frame accuracy) over non-padded frames."""
    model.eval()
    total, n = 0.0, 0
    correct, seen = 0, 0
    for x, y, _, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_y = y.reshape(-1)
        total += float(criterion(flat_logits, flat_y).item())
        n += 1
        mask = flat_y != IGNORE_INDEX
        if mask.any():
            pred = flat_logits[mask].argmax(dim=-1)
            correct += int((pred == flat_y[mask]).sum().item())
            seen += int(mask.sum().item())
    return total / max(n, 1), (correct / seen if seen else 0.0)


def build_frame_model(model_type, input_dim, hidden, tcn_channels, device):
    """Build the model and verify it emits exactly NUM_STATES logits (no blank)."""
    try:
        model = build_model(
            model_type, input_dim, hidden=hidden,
            tcn_channels=tcn_channels, num_classes=NUM_STATES,
        )
    except TypeError:
        model = build_model(model_type, input_dim, hidden=hidden, tcn_channels=tcn_channels)
        print("Note: build_model() does not accept num_classes; using its default width.")

    model.to(device)
    with torch.no_grad():
        out_dim = int(model(torch.zeros(1, 8, input_dim, device=device)).shape[-1])
    if out_dim != NUM_STATES:
        raise SystemExit(
            f"Model emits {out_dim} classes but cross-entropy training needs "
            f"{NUM_STATES} (one per state, no blank). If this model was sized "
            f"for CTC it has an extra blank column — set num_classes={NUM_STATES}."
        )
    return model


def _inner_split(picklist_ids: Sequence[str], val_ratio: float, seed: int):
    """Inner train/val split drawn only from *picklist_ids* (never outer test)."""
    ids = list(picklist_ids)
    if len(ids) <= 1:
        return ids, ids
    train_ids, val_ids = train_val_split(ids, val_ratio, seed)
    if not train_ids:
        return ids, ids
    return train_ids, val_ids


def train_classifier(
    *,
    picklist_ids: Sequence[str],
    feature_source: FeatureSource = "processed_3d",
    features_dir: Union[str, Path],
    labels_dir: Union[str, Path],
    ground_truth: Union[str, Path],
    output_dir: Union[str, Path],
    model_type: str = "tcn",
    hidden: int = 0,
    tcn_channels: int = 32,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
    class_weights: bool = False,
    select_on: str = "accuracy",
    extra_config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Path:
    """
    Train a frame classifier on an explicit picklist-id list.

    Inner train/validation checkpoint selection is drawn only from
    ``picklist_ids``. Outer test IDs (if recorded in ``extra_config``) are
    never loaded for normalization, optimization, or model selection.
    """
    features_dir = Path(features_dir)
    labels_dir = Path(labels_dir)
    ground_truth = Path(ground_truth)
    output_dir = Path(output_dir)

    outer_test = set(str(x) for x in (extra_config or {}).get("outer_test_ids") or [])
    leaked = [pid for pid in picklist_ids if str(pid) in outer_test]
    if leaked:
        raise ValueError(f"Outer test IDs passed as training IDs: {leaked}")

    if not picklist_ids:
        raise SystemExit("No picklists with features and labels found.")

    train_ids, val_ids = _inner_split(picklist_ids, val_ratio, seed)
    if verbose:
        print(f"Train: {len(train_ids)} picklists, Val: {len(val_ids)} picklists")

    train_samples = load_dataset(
        train_ids, feature_source, features_dir, labels_dir,
        ground_truth, require_frame_labels=True,
    )
    val_samples = load_dataset(
        val_ids, feature_source, features_dir, labels_dir,
        ground_truth, require_frame_labels=True,
    )
    if not train_samples:
        raise SystemExit("No usable training samples with dense frame labels.")
    if not val_samples:
        val_samples = train_samples

    mean, std = compute_normalization_stats(train_samples)
    priors = compute_label_priors(train_samples)
    input_dim = feature_dim_for_source(feature_source)
    if verbose:
        print(
            "Class priors: "
            + ", ".join(f"{STATES[i]}={priors[i]:.3f}" for i in range(NUM_STATES))
        )

    label_arrays = [s.labels for s in train_samples if s.labels is not None]
    dur = duration_stats(label_arrays, NUM_STATES)
    self_loop_logprobs = estimate_self_loop_logprobs(label_arrays, NUM_STATES)
    if verbose:
        print(
            "Mean durations (frames): "
            + ", ".join(f"{STATES[i]}={dur['mean_duration'][i]:.1f}" for i in range(NUM_STATES))
        )

    train_loader = DataLoader(
        FrameDataset(train_samples, mean, std),
        batch_size=batch_size, shuffle=True, collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        FrameDataset(val_samples, mean, std),
        batch_size=batch_size, shuffle=False, collate_fn=collate_batch,
    )

    torch_device = torch.device(device)
    model = build_frame_model(model_type, input_dim, hidden, tcn_channels, torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    weight = None
    if class_weights:
        weight = torch.tensor(1.0 / np.clip(priors, 1e-6, None), dtype=torch.float32, device=torch_device)
        weight = weight / weight.mean()
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=weight)

    best = float("inf") if select_on == "loss" else -1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, torch_device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, torch_device)
        if verbose:
            print(f"Epoch {epoch:3d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_frame_acc={va_acc:.4f}")

        metric = va_loss if select_on == "loss" else va_acc
        better = metric < best if select_on == "loss" else metric > best
        if better:
            best = metric
            config: Dict[str, Any] = {
                "feature_source": feature_source,
                "model_type": model_type,
                "input_dim": input_dim,
                "hidden": hidden,
                "tcn_channels": tcn_channels,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "train_ids": list(train_ids),
                "val_ids": list(val_ids),
                "outer_train_ids": list(picklist_ids),
                "loss": "ce",
                "num_classes": NUM_STATES,
                "num_states": NUM_STATES,
                "priors": priors.tolist(),
                "self_loop_logprobs": self_loop_logprobs.tolist(),
                "mean_durations": dur["mean_duration"].tolist(),
                "segment_counts": dur["n_segments"].tolist(),
                "class_weighted": bool(class_weights),
                "states": list(STATES),
                "val_frame_acc": va_acc,
            }
            if extra_config:
                config.update(extra_config)
            torch.save({"model_state": model.state_dict(), "config": config}, ckpt_path)
            (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    if not ckpt_path.is_file():
        raise SystemExit("Training finished without saving a checkpoint.")
    if verbose:
        label = "val loss" if select_on == "loss" else "val frame accuracy"
        print(f"Best {label}: {best:.4f}")
        print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path


def _run_kfold_train(args, feature_source: FeatureSource, features_dir: Path) -> None:
    from cross_validation.layout import CvLayout
    from cross_validation.manifests import create_or_load_dataset, iter_fold_manifests, require_cv_pair
    from cross_validation.status import mark_stage, stage_completed
    from cross_validation.stems import id_from_stem

    require_cv_pair(args.k_fold, args.cv_results_dir)
    create_or_load_dataset(
        cv_root=args.cv_results_dir,
        k_fold=int(args.k_fold),
        seed=int(args.seed),
        annealing_labels_dir=args.annealing_labels_dir,
        segmentation_labels_dir=args.labels_dir,
        videos_dir=args.videos_dir,
        json_dir=args.picklist_jsons,
        features_dir=features_dir,
        overwrite=bool(args.overwrite),
    )
    layout = CvLayout(args.cv_results_dir)
    for fold in iter_fold_manifests(args.cv_results_dir):
        fold_n = int(fold["fold"])
        fl = layout.fold(fold_n)
        if stage_completed(args.cv_results_dir, fold_n, "nn_train") and not args.overwrite:
            print(f"Fold {fold_n}: nn_train already complete, skipping")
            continue
        train_ids = [id_from_stem(s) for s in fold["segmentation_train_ids"]]
        test_ids = [id_from_stem(s) for s in fold["segmentation_test_ids"]]
        if not train_ids:
            raise SystemExit(f"Fold {fold_n} has no segmentation_train_ids")
        train_classifier(
            picklist_ids=train_ids,
            feature_source=feature_source,
            features_dir=features_dir,
            labels_dir=args.labels_dir,
            ground_truth=args.ground_truth,
            output_dir=fl.nn_checkpoint_dir,
            model_type=args.model,
            hidden=args.hidden,
            tcn_channels=args.tcn_channels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_ratio=args.val_ratio,
            seed=args.seed,
            device=args.device,
            class_weights=args.class_weights,
            select_on=args.select_on,
            extra_config={"outer_test_ids": test_ids, "fold": fold_n},
        )
        mark_stage(args.cv_results_dir, fold_n, "nn_train")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train per-frame state classifier (cross-entropy).")
    ap.add_argument(
        "--feature-source", choices=["processed_3d", "csv_outputs"], default="processed_3d"
    )
    ap.add_argument("--features-dir", type=Path, default=None)
    ap.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    ap.add_argument("--output-dir", type=Path, default=Path("nn_model/checkpoints"))
    ap.add_argument("--model", choices=["mlp", "tcn"], default="tcn")
    ap.add_argument("--hidden", type=int, default=0, help="MLP hidden dim (0=linear)")
    ap.add_argument("--tcn-channels", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--class-weights",
        action="store_true",
        help="Weight CE by inverse class frequency. Usually leave off: the HMM "
        "divides by the prior at alignment time, and doing both double-counts.",
    )
    ap.add_argument(
        "--select-on",
        choices=["loss", "accuracy"],
        default="accuracy",
        help="Metric used to pick the best checkpoint.",
    )
    ap.add_argument("--k-fold", type=int, default=None, help="Run K-fold CV over all folds")
    ap.add_argument("--cv-results-dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true", help="Redo completed CV folds / resplit")
    ap.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--picklist-jsons", type=Path, default=DEFAULT_PICKLIST_JSONS)
    ap.add_argument("--annealing-labels-dir", type=Path, default=DEFAULT_PICKLIST_LABELS)
    args = ap.parse_args()

    feature_source: FeatureSource = args.feature_source
    if args.features_dir is None:
        features_dir = (
            DEFAULT_FEATURES_3D if feature_source == "processed_3d" else DEFAULT_CSV_OUTPUTS
        )
    else:
        features_dir = args.features_dir

    if args.k_fold is not None or args.cv_results_dir is not None:
        _run_kfold_train(args, feature_source, features_dir)
        return

    picklist_ids = discover_picklist_ids(feature_source, features_dir, args.labels_dir)
    train_classifier(
        picklist_ids=picklist_ids,
        feature_source=feature_source,
        features_dir=features_dir,
        labels_dir=args.labels_dir,
        ground_truth=args.ground_truth,
        output_dir=args.output_dir,
        model_type=args.model,
        hidden=args.hidden,
        tcn_channels=args.tcn_channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        class_weights=args.class_weights,
        select_on=args.select_on,
    )


if __name__ == "__main__":
    main()
