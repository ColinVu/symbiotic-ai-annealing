"""Unit tests for immutable K-fold manifests and leakage guards."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cross_validation.manifests import create_or_load_dataset, iter_fold_manifests
from cross_validation.splits import balanced_fold_assignments
from tests.cv_fixtures import build_synthetic_dataset


class TestCvManifests(unittest.TestCase):
    def test_deterministic_balanced_split(self):
        annealing = [f"picklist_{i:03d}" for i in range(1, 9)]
        segmentation = annealing[:3]
        a = balanced_fold_assignments(annealing, segmentation, k_fold=4, seed=7)
        b = balanced_fold_assignments(annealing, segmentation, k_fold=4, seed=7)
        self.assertEqual(a, b)
        test_ids = [t for _, t in a]
        flat = [s for fold in test_ids for s in fold]
        self.assertEqual(sorted(flat), sorted(annealing))
        self.assertEqual(len(flat), len(set(flat)))
        seg_per_fold = [sum(1 for s in t if s in set(segmentation)) for t in test_ids]
        self.assertTrue(max(seg_per_fold) - min(seg_per_fold) <= 1)

    def test_immutable_reuse_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_synthetic_dataset(Path(tmp), n=4, n_segmentation=2)
            first = create_or_load_dataset(
                cv_root=paths["cv_root"],
                k_fold=2,
                seed=3,
                segmentation_labels_dir=paths["csv_labels"],
                videos_dir=paths["videos"],
                json_dir=paths["jsons"],
                features_dir=paths["features"],
            )
            second = create_or_load_dataset(
                cv_root=paths["cv_root"],
                k_fold=2,
                seed=3,
                segmentation_labels_dir=paths["csv_labels"],
                videos_dir=paths["videos"],
                json_dir=paths["jsons"],
                features_dir=paths["features"],
            )
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            with self.assertRaises(ValueError):
                create_or_load_dataset(
                    cv_root=paths["cv_root"],
                    k_fold=3,
                    seed=3,
                    segmentation_labels_dir=paths["csv_labels"],
                    videos_dir=paths["videos"],
                    json_dir=paths["jsons"],
                    features_dir=paths["features"],
                )

    def test_no_train_test_overlap_and_segmentation_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_synthetic_dataset(Path(tmp), n=4, n_segmentation=2)
            create_or_load_dataset(
                cv_root=paths["cv_root"],
                k_fold=2,
                seed=3,
                segmentation_labels_dir=paths["csv_labels"],
                videos_dir=paths["videos"],
                json_dir=paths["jsons"],
                features_dir=paths["features"],
            )
            annealing = set(json.loads((paths["cv_root"] / "dataset_manifest.json").read_text())["annealing_ids"])
            segmentation = set(
                json.loads((paths["cv_root"] / "dataset_manifest.json").read_text())["segmentation_ids"]
            )
            self.assertTrue(segmentation.issubset(annealing))
            seen_test = []
            for fold in iter_fold_manifests(paths["cv_root"]):
                train = set(fold["annealing_train_ids"])
                test = set(fold["annealing_test_ids"])
                self.assertFalse(train & test)
                self.assertEqual(train | test, annealing)
                self.assertEqual(set(fold["segmentation_train_ids"]), train & segmentation)
                self.assertEqual(set(fold["segmentation_test_ids"]), test & segmentation)
                seen_test.extend(test)
            self.assertEqual(sorted(seen_test), sorted(annealing))


if __name__ == "__main__":
    unittest.main()
