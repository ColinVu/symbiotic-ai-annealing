"""End-to-end synthetic smoke test for the four --k-fold stages."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from cross_validation.layout import CvLayout
from cross_validation.manifests import create_or_load_dataset, iter_fold_manifests
from cross_validation.status import mark_stage, stage_completed
from cross_validation.stems import id_from_stem, normalize_stem
from cross_validation.summary import write_summary
from nn_model.bulk_predict import generate_label_csvs
from nn_model.create_labels import _score_segmentation
from nn_model.train import train_classifier
from symbiote_weak_generalized.core.config import DEFAULT_CONFIG
from symbiote_weak_generalized.experiments.evaluator import evaluate_model
from symbiote_weak_generalized.pipelines.video_training import run_multi_video_training_from_cache
from tests.cv_fixtures import FRAME_SKIP, build_synthetic_dataset, resolve_video_path_safe

# local helper if videos.resolve isn't imported
from cross_validation.videos import resolve_video_path


class TestCvE2E(unittest.TestCase):
    def test_four_stage_kfold_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_synthetic_dataset(Path(tmp), n=4, n_segmentation=2)
            cv_root = paths["cv_root"]
            create_or_load_dataset(
                cv_root=cv_root,
                k_fold=2,
                seed=3,
                segmentation_labels_dir=paths["csv_labels"],
                videos_dir=paths["videos"],
                json_dir=paths["jsons"],
                features_dir=paths["features"],
            )
            layout_root = CvLayout(cv_root)

            # Stage 1: NN train
            for fold in iter_fold_manifests(cv_root):
                fl = layout_root.fold(int(fold["fold"]))
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
                    epochs=4,
                    extra_config={"outer_test_ids": test_ids, "fold": fold["fold"]},
                )
                mark_stage(cv_root, int(fold["fold"]), "nn_train")
                self.assertTrue(fl.nn_checkpoint.is_file())
                self.assertTrue(stage_completed(cv_root, int(fold["fold"]), "nn_train"))

            # Resume: completed folds are skipped by stage flags
            self.assertTrue(all(stage_completed(cv_root, int(f["fold"]), "nn_train") for f in iter_fold_manifests(cv_root)))

            # Stage 2: labels from matching checkpoint
            for fold in iter_fold_manifests(cv_root):
                fl = layout_root.fold(int(fold["fold"]))
                train_stems = [normalize_stem(s) for s in fold["annealing_train_ids"]]
                test_stems = [normalize_stem(s) for s in fold["annealing_test_ids"]]
                all_ids = [id_from_stem(s) for s in train_stems + test_stems]
                dense = generate_label_csvs(
                    checkpoint=fl.nn_checkpoint,
                    features_dir=paths["features"],
                    output_dir=fl.nn_train_labels,
                    picklist_ids=all_ids,
                    picklist_jsons=paths["jsons"],
                    min_duration=8,
                    verbose=False,
                )
                fl.nn_test_labels.mkdir(parents=True, exist_ok=True)
                for pid in list(dense):
                    src = fl.nn_train_labels / f"picklist_{pid}.csv"
                    if pid in {id_from_stem(s) for s in test_stems} and src.is_file():
                        dest = fl.nn_test_labels / f"picklist_{pid}.csv"
                        dest.write_bytes(src.read_bytes())
                        src.unlink()
                self.assertFalse(
                    any((fl.nn_train_labels / f"{s}.csv").is_file() for s in test_stems)
                )
                metrics = _score_segmentation(
                    picklist_ids=[id_from_stem(s) for s in fold["segmentation_test_ids"]],
                    dense_preds=dense,
                    feature_source="processed_3d",
                    features_dir=paths["features"],
                    labels_dir=paths["csv_labels"],
                    ground_truth=paths["ground_truth"],
                )
                fl.nn_metrics.write_text(
                    json.dumps({"fold": fold["fold"], "checkpoint": str(fl.nn_checkpoint), "segmentation": metrics}, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
                mark_stage(cv_root, int(fold["fold"]), "nn_labels")

            # Stage 3: annealing train-from-cache on train stems only
            config = DEFAULT_CONFIG.copy()
            config["apply_hand_pca"] = False
            config["skip_ilr"] = True
            config["use_clip_init"] = False
            config["ilr_epochs"] = 1
            config["min_frames_per_cluster"] = 2
            config["ground_truth_csv"] = str(paths["ground_truth"])
            for fold in iter_fold_manifests(cv_root):
                fl = layout_root.fold(int(fold["fold"]))
                train_stems = [normalize_stem(s) for s in fold["annealing_train_ids"]]
                test_stems = {normalize_stem(s) for s in fold["annealing_test_ids"]}
                self.assertFalse(set(train_stems) & test_stems)
                run_multi_video_training_from_cache(
                    videos_dir=str(paths["videos"]),
                    picklist_json_dir=str(paths["jsons"]),
                    manual_labels_dir=str(fl.nn_train_labels),
                    base_output_dir=str(fl.annealing_model),
                    config=config,
                    cache_dir=str(paths["cache"]),
                    frame_skip=FRAME_SKIP,
                    verbose=False,
                    include_stems=train_stems,
                )
                stems_trained = json.loads((fl.annealing_model / "model_metadata.json").read_text())[
                    "embedded_video_stems"
                ]
                self.assertEqual(set(stems_trained), set(train_stems))
                self.assertFalse(set(stems_trained) & test_stems)
                self.assertTrue((fl.annealing_model / "cache_coverage.json").is_file())
                mark_stage(cv_root, int(fold["fold"]), "annealing_train")

            # Stage 4: held-out test using first-SKU cache keys, no CLIP
            fold_records = []
            for fold in iter_fold_manifests(cv_root):
                fl = layout_root.fold(int(fold["fold"]))
                first_skus = fold.get("first_skus") or {}
                per_video = []
                for stem in fold["annealing_test_ids"]:
                    stem = normalize_stem(stem)
                    video = resolve_video_path(paths["videos"], stem)
                    ev = evaluate_model(
                        model_dir=str(fl.annealing_model),
                        video_path=str(video),
                        ground_truth_csv=str(paths["ground_truth"]),
                        state_label_csv_path=str(fl.nn_test_labels / f"{stem}.csv"),
                        frame_skip=FRAME_SKIP,
                        eval_cache_dir=str(paths["cache"] / stem),
                        cache_label=first_skus[stem],
                        embed_missing=False,
                    )
                    self.assertNotEqual(ev["cache_label"], "sweep_eval")
                    self.assertGreater(ev["cache_hits"], 0)
                    self.assertIn("frame_accuracy", ev["metrics"])
                    self.assertIn("segment_top1_accuracy", ev["metrics"])
                    self.assertIn(
                        "shelf_constrained_segment_top1_accuracy",
                        ev["metrics"],
                    )
                    self.assertIn("by_shelf", ev["metrics"])
                    per_video.append(ev)
                    (fl.annealing_test).mkdir(parents=True, exist_ok=True)
                    (fl.annealing_test / f"{stem}_eval.json").write_text(
                        json.dumps(ev, indent=2) + "\n", encoding="utf-8"
                    )
                top1 = sum(int(e["metrics"]["segment_top1_hits"]) for e in per_video)
                top3 = sum(int(e["metrics"]["segment_top3_hits"]) for e in per_video)
                segs = sum(int(e["metrics"]["carry_segments_used"]) for e in per_video)
                test_metrics = {
                    "segment_top1_hits": top1,
                    "segment_top3_hits": top3,
                    "carry_segments_used": segs,
                    "segment_top1_accuracy": top1 / segs if segs else None,
                    "segment_top3_hit_rate": top3 / segs if segs else None,
                }
                (fl.annealing_test / "metrics.json").write_text(
                    json.dumps({"metrics": test_metrics}, indent=2) + "\n", encoding="utf-8"
                )
                nn = json.loads(fl.nn_metrics.read_text())
                cov = json.loads((fl.annealing_model / "cache_coverage.json").read_text())
                fold_records.append(
                    {
                        "fold": fold["fold"],
                        "n_train": len(fold["annealing_train_ids"]),
                        "n_test": len(fold["annealing_test_ids"]),
                        "n_segmentation_train": len(fold["segmentation_train_ids"]),
                        "n_segmentation_test": len(fold["segmentation_test_ids"]),
                        "segmentation": nn.get("segmentation"),
                        "annealing_test": test_metrics,
                        "cache_coverage": cov,
                    }
                )
                mark_stage(cv_root, int(fold["fold"]), "annealing_test")

            aggregate = write_summary(cv_root, fold_records)
            self.assertEqual(aggregate["n_folds"], 2)
            self.assertIsNotNone(aggregate["annealing_test"]["micro_segment_top1_accuracy"])
            self.assertTrue((layout_root.summary_dir / "fold_metrics.csv").is_file())
            self.assertTrue((layout_root.aggregate_metrics).is_file())
            # Held-out scores exist and are not copied from train-fit
            for rec in fold_records:
                self.assertIn("annealing_test", rec)
                self.assertNotIn("assignment_hit_rate", rec["annealing_test"] or {})


if __name__ == "__main__":
    unittest.main()
