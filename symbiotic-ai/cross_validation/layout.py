"""On-disk layout for a combined-pipeline K-fold results directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


def fold_dirname(fold: int) -> str:
    return f"fold_{int(fold):02d}"


@dataclass(frozen=True)
class FoldLayout:
    """Paths for one fold under the global CV results directory."""

    root: Path
    fold: int

    @property
    def fold_dir(self) -> Path:
        return self.root / fold_dirname(self.fold)

    @property
    def fold_manifest(self) -> Path:
        return self.fold_dir / "fold_manifest.json"

    @property
    def status_file(self) -> Path:
        return self.fold_dir / "status.json"

    @property
    def nn_checkpoint_dir(self) -> Path:
        return self.fold_dir / "nn_model"

    @property
    def nn_checkpoint(self) -> Path:
        return self.nn_checkpoint_dir / "best_model.pt"

    @property
    def nn_config(self) -> Path:
        return self.nn_checkpoint_dir / "config.json"

    @property
    def nn_train_labels(self) -> Path:
        return self.nn_checkpoint_dir / "labels" / "train"

    @property
    def nn_test_labels(self) -> Path:
        return self.nn_checkpoint_dir / "labels" / "test"

    @property
    def nn_metrics(self) -> Path:
        return self.nn_checkpoint_dir / "metrics.json"

    @property
    def annealing_model(self) -> Path:
        return self.fold_dir / "annealing" / "model"

    @property
    def annealing_train(self) -> Path:
        return self.fold_dir / "annealing" / "train"

    @property
    def annealing_test(self) -> Path:
        return self.fold_dir / "annealing" / "test"


class CvLayout:
    """Top-level CV results tree: dataset manifest, folds, and summary."""

    def __init__(self, cv_root: Union[str, Path]):
        self.root = Path(cv_root)

    @property
    def dataset_manifest(self) -> Path:
        return self.root / "dataset_manifest.json"

    @property
    def summary_dir(self) -> Path:
        return self.root / "summary"

    @property
    def fold_metrics_csv(self) -> Path:
        return self.summary_dir / "fold_metrics.csv"

    @property
    def fold_metrics_json(self) -> Path:
        return self.summary_dir / "fold_metrics.json"

    @property
    def aggregate_metrics(self) -> Path:
        return self.summary_dir / "aggregate_metrics.json"

    def fold(self, n: int) -> FoldLayout:
        return FoldLayout(self.root, int(n))
