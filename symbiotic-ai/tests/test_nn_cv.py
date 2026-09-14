"""NN K-fold leakage and checkpoint-matching tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cross_validation.layout import CvLayout
from cross_validation.manifests import create_or_load_dataset, iter_fold_manifests
from cross_validation.stems import id_from_stem
from nn_model.bulk_predict import generate_label_csvs
from nn_model.train import train_classifier
from tests.cv_fixtures import build_synthetic_dataset


class TestNnCv(unittest.TestCase):
    def test_outer_test_ids_never_enter_training(self):
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
            layout = CvLayout(paths["cv_root"])
            for fold in iter_fold_manifests(paths["cv_root"]):
                fl = layout.fold(int(fold["fold"]))
                train_ids = [id_from_stem(s) for s in fold["segmentation_train_ids"]]
                test_ids = [id_from_stem(s) for s in fold["segmentation_test_ids"]]
                self.assertTrue(train_ids)
                with self.assertRaises(ValueError):
                    train_classifier(
                        picklist_ids=train_ids + test_ids,
                        feature_source="processed_3d",
                        features_dir=paths["features"],
                        labels_dir=paths["csv_labels"],
                        ground_truth=paths["ground_truth"],
                        output_dir=fl.nn_checkpoint_dir / "leaked",
                        model_type="mlp",
                        epochs=1,
                        extra_config={"outer_test_ids": test_ids},
                        verbose=False,
                    )
                ckpt = train_classifier(
                    picklist_ids=train_ids,
                    feature_source="processed_3d",
                    features_dir=paths["features"],
                    labels_dir=paths["csv_labels"],
                    ground_truth=paths["ground_truth"],
                    output_dir=fl.nn_checkpoint_dir,
                    model_type="mlp",
                    epochs=2,
                    extra_config={"outer_test_ids": test_ids, "fold": fold["fold"]},
                    verbose=False,
                )
                config = json.loads(fl.nn_config.read_text(encoding="utf-8"))
                self.assertEqual(set(config["outer_train_ids"]), set(train_ids))
                self.assertEqual(set(config["outer_test_ids"]), set(test_ids))
                self.assertFalse(set(config["train_ids"]) & set(test_ids))
                self.assertFalse(set(config["val_ids"]) & set(test_ids))
                self.assertTrue(ckpt.is_file())

    def test_fold_labels_use_matching_checkpoint(self):
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
            layout = CvLayout(paths["cv_root"])
            fold = next(iter_fold_manifests(paths["cv_root"]))
            fl = layout.fold(int(fold["fold"]))
            train_ids = [id_from_stem(s) for s in fold["segmentation_train_ids"]]
            test_ids = [id_from_stem(s) for s in fold["segmentation_test_ids"]]
            train_classifier(
                picklist_ids=train_ids,
                feature_source="processed_3d",
                features_dir=paths["features"],
                labels_dir=paths["csv_labels"],
                ground_truth=paths["ground_truth"],
                output_dir=fl.nn_checkpoint_dir,
                model_type="mlp",
                epochs=2,
                extra_config={"outer_test_ids": test_ids, "fold": fold["fold"]},
                verbose=False,
            )
            all_ids = [id_from_stem(s) for s in fold["annealing_train_ids"] + fold["annealing_test_ids"]]
            dense = generate_label_csvs(
                checkpoint=fl.nn_checkpoint,
                features_dir=paths["features"],
                output_dir=fl.nn_train_labels,
                picklist_ids=all_ids,
                picklist_jsons=paths["jsons"],
                min_duration=8,
                verbose=False,
            )
            self.assertEqual(set(dense), set(all_ids))
            loaded = __import__("torch").load(fl.nn_checkpoint, map_location="cpu")
            self.assertEqual(int(loaded["config"]["fold"]), int(fold["fold"]))
            for pid in all_ids:
                self.assertTrue((fl.nn_train_labels / f"picklist_{pid}.csv").is_file())


if __name__ == "__main__":
    unittest.main()
