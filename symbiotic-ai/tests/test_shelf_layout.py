"""Focused tests for simulator-facing shelf-layout decisions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from shelf_layout.engine import DecisionEngine, ModelData, SnapshotError, _margin
from shelf_layout.cli import main


def fixture():
    labels = ("a", "b", "c", "d", "e", "f", "g", "h", "i")
    vectors = np.eye(9)
    vectors[1] = np.array([0.99, 0.1] + [0.0] * 7)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    model = ModelData(labels, vectors, lambda x: x, "synthetic")
    snapshot = {
        "step": 2,
        "capacities": {"s1": 3, "s2": 3, "s3": 3},
        "items": [{"sku": sku, "shelf": "s" + str(i // 3 + 1), "slot": str(i % 3 + 1)}
                  for i, sku in enumerate(labels)],
        "observations": [
            {"id": "a0", "revealed_at": 1, "inferred_sku": "a", "observed_shelf": "s1", "frames": [vectors[1].tolist()]},
            {"id": "a1", "revealed_at": 2, "inferred_sku": "a", "observed_shelf": "s1", "frames": [vectors[1].tolist()]},
            {"id": "b0", "revealed_at": 2, "inferred_sku": "b", "observed_shelf": "s1", "frames": [vectors[1].tolist()]},
        ],
    }
    return model, snapshot


class ShelfLayoutTests(unittest.TestCase):
    def test_recommendation_separates_confusing_pair_and_preserves_capacity(self):
        model, snapshot = fixture()
        engine = DecisionEngine(snapshot, model)
        self.assertEqual(engine.score(engine.baseline)["errors"], 2)
        result = engine.recommend(max_moved=2)
        layout = result["recommendation"]["layout"]
        self.assertNotEqual(layout["a"]["shelf"], layout["b"]["shelf"])
        self.assertEqual(result["recommendation"]["proxy_errors"], 0)
        self.assertEqual(result["recommendation"]["fixed"], 2)
        self.assertEqual(result["recommendation"]["new_errors"], 0)
        self.assertTrue(any(c["moved_skus"] == 0 for c in result["candidates"]))
        for shelf in snapshot["capacities"]:
            self.assertEqual(sum(x["shelf"] == shelf for x in layout.values()), 3)
            self.assertEqual(len({x["slot"] for x in layout.values() if x["shelf"] == shelf}), 3)
        self.assertEqual(snapshot["items"][0]["shelf"], "s1")

    def test_future_or_truth_cannot_enter_decision(self):
        model, snapshot = fixture()
        snapshot["observations"][0]["revealed_at"] = 3
        with self.assertRaises(SnapshotError):
            DecisionEngine(snapshot, model)
        snapshot["observations"][0]["revealed_at"] = 1
        snapshot["observations"][0]["true_label"] = "a"
        with self.assertRaises(SnapshotError):
            DecisionEngine(snapshot, model)

    def test_pickup_shelf_must_match_current_weak_assignment(self):
        model, snapshot = fixture()
        snapshot["observations"][0]["observed_shelf"] = "s2"
        with self.assertRaisesRegex(SnapshotError, "different shelf than its observed pickup"):
            DecisionEngine(snapshot, model)

    def test_no_move_when_no_observations(self):
        model, snapshot = fixture()
        snapshot["observations"] = []
        result = DecisionEngine(snapshot, model).recommend()
        self.assertEqual(result["recommendation"]["moved_skus"], 0)

    def test_confusion_trigger_uses_segment_counts_and_reports_gates(self):
        model, snapshot = fixture()
        snapshot["observations"].extend(
            {"id": f"a{i}", "revealed_at": 2, "inferred_sku": "a",
             "observed_shelf": "s1", "frames": [model.centroids[1].tolist()],
             "assignment_confidence": 0.7}
            for i in range(2, 5)
        )
        engine = DecisionEngine(snapshot, model)
        evidence = engine._confusion_evidence(engine.baseline, engine.score_current()["predictions"])
        pair = next(row for row in evidence if (row["a"], row["b"]) == ("a", "b"))
        self.assertTrue(pair["eligible"])
        self.assertEqual(pair["a_segments"], 5)
        self.assertEqual(pair["a_to_b"], 5)
        self.assertEqual(pair["b_to_a"], 0)
        self.assertAlmostEqual(pair["pair_rate"], 5 / 6)
        self.assertEqual(pair["confidence_segments"], 3)
        strict = engine._confusion_evidence(engine.baseline, engine.score_current()["predictions"],
                                            min_pair_segments=8)
        self.assertIn("insufficient_segments", strict[0]["exclusion_reasons"])

    def test_targeted_path_avoids_all_pairs_enumeration_and_respects_budget(self):
        model, snapshot = fixture()
        engine = DecisionEngine(snapshot, model)
        with patch("shelf_layout.engine.combinations", side_effect=AssertionError("all-pairs scan")):
            result = engine.recommend(max_moved=2, cycle_pool_size=0,
                                      min_pair_segments=1, min_pair_confusions=1,
                                      fallback_swap_budget=0, swap_replay_limit=1)
        self.assertTrue(result["search"]["passes"][0]["trigger_pairs"])
        self.assertEqual(result["search"]["passes"][0]["exact_swap_replays"], 1)
        self.assertTrue(all(p["exact_swap_replays"] <= 1 for p in result["search"]["passes"]))
        self.assertEqual(result["recommendation"]["proxy_errors"], 0)
        self.assertEqual(result["search"]["passes"][-1]["observed_pairs"], [])

    def test_sparse_evidence_uses_bounded_fallback(self):
        model, snapshot = fixture()
        engine = DecisionEngine(snapshot, model)
        result = engine.recommend(max_moved=2, cycle_pool_size=0, swap_replay_limit=2,
                                  fallback_swap_budget=1)
        self.assertFalse(result["search"]["passes"][0]["trigger_pairs"])
        self.assertTrue(result["search"]["passes"][0]["fallback_used"])
        self.assertLessEqual(result["search"]["passes"][0]["exact_swap_replays"], 1)
        self.assertEqual(result["recommendation"]["proxy_errors"], 0)

    def test_all_pairs_comparison_mode_and_determinism(self):
        model, snapshot = fixture()
        engine = DecisionEngine(snapshot, model)
        targeted = engine.recommend(max_moved=2)
        repeated = engine.recommend(max_moved=2)
        old = engine.recommend(max_moved=2, search_mode="all_pairs")
        self.assertEqual(targeted, repeated)
        self.assertEqual(targeted["recommendation"]["proxy_errors"],
                         old["recommendation"]["proxy_errors"])
        self.assertEqual(old["settings"]["search_mode"], "all_pairs")

    def test_destination_shelf_can_create_new_error(self):
        model, snapshot = fixture()
        e_view = (0.8 * model.centroids[0] + 0.6 * model.centroids[4]).tolist()
        snapshot["observations"].append({"id": "e0", "revealed_at": 2,
                                         "inferred_sku": "e", "observed_shelf": "s2", "frames": [e_view]})
        engine = DecisionEngine(snapshot, model)
        layout = dict(engine.baseline)
        layout["a"], layout["d"] = layout["d"], layout["a"]
        old = engine.score(engine.baseline)["predictions"]
        new = engine.score(layout)["predictions"]
        self.assertNotEqual(old["a0"], "a")
        self.assertEqual(new["a0"], "a")
        self.assertEqual(old["e0"], "e")
        self.assertNotEqual(new["e0"], "e")

    def test_margin_matches_evaluator_zero_gap_behavior(self):
        np.testing.assert_allclose(_margin(np.array([[0.0, 0.0, 1.0],
                                                      [0.0, 0.5, 1.0],
                                                      [0.0, 0.0, 0.0]])),
                                   np.array([1.0, 2.0 / 3.0, 0.0]))

    def test_true_labels_are_post_decision_only(self):
        model, snapshot = fixture()
        engine = DecisionEngine(snapshot, model)
        before = engine.recommend(max_moved=2)
        engine.retrospective({"a0": "a", "a1": "a", "b0": "b"}, engine.baseline)
        after = engine.recommend(max_moved=2)
        self.assertEqual(before["recommendation"], after["recommendation"])

    def test_missing_model_files_report_failure(self):
        _, snapshot = fixture()
        with tempfile.TemporaryDirectory() as td:
            snap = Path(td) / "snapshot.json"
            snap.write_text(json.dumps(snapshot), encoding="utf-8")
            out = Path(td) / "out"
            code = main(["preflight", "--snapshot", str(snap), "--model-dir", td,
                         "--output-dir", str(out)])
            self.assertEqual(code, 2)
            self.assertTrue((out / "missing_inputs.json").is_file())

    def test_saved_model_cli_reports(self):
        model, snapshot = fixture()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "model_metadata.json").write_text(
                json.dumps({"centroid_labels": list(model.labels), "embedding_dim": 9}), encoding="utf-8")
            np.save(model_dir / "centroids.npy", model.centroids)
            snap = root / "snapshot.json"
            snap.write_text(json.dumps(snapshot), encoding="utf-8")
            out = root / "out"
            self.assertEqual(main(["recommend", "--snapshot", str(snap),
                                   "--model-dir", str(model_dir), "--output-dir", str(out),
                                   "--max-moved", "2"]), 0)
            self.assertTrue((out / "summary.md").is_file())
            self.assertTrue((out / "candidate_layouts.csv").is_file())
            self.assertTrue((out / "confusion_evidence.csv").is_file())
            self.assertIn("search", json.loads((out / "summary.json").read_text()))
            self.assertEqual(json.loads((out / "summary.json").read_text())["recommendation"]["proxy_errors"], 0)
            truth = root / "truth.json"
            truth.write_text(json.dumps({"a0": "a", "a1": "a", "b0": "b"}), encoding="utf-8")
            other = root / "with_truth"
            self.assertEqual(main(["recommend", "--snapshot", str(snap),
                                   "--model-dir", str(model_dir), "--output-dir", str(other),
                                   "--max-moved", "2", "--truth-json", str(truth)]), 0)
            base_result = json.loads((out / "summary.json").read_text())
            truth_result = json.loads((other / "summary.json").read_text())
            self.assertEqual(base_result["recommendation"], truth_result["recommendation"])
            self.assertIn("retrospective", truth_result)

    def test_simulate_respects_supplied_next_layout(self):
        model, snapshot = fixture()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "model_metadata.json").write_text(
                json.dumps({"centroid_labels": list(model.labels), "embedding_dim": 9}), encoding="utf-8")
            np.save(model_dir / "centroids.npy", model.centroids)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            (snapshots / "one.json").write_text(json.dumps(snapshot), encoding="utf-8")
            first = DecisionEngine(snapshot, model).recommend(max_moved=2)
            later = json.loads(json.dumps(snapshot))
            later["step"] = 3
            later["previous_decision"] = {"step": 2, "accepted": True}
            later["items"] = [{"sku": sku, **place} for sku, place in first["recommendation"]["layout"].items()]
            for row in later["observations"]:
                row["observed_shelf"] = first["recommendation"]["layout"][row["inferred_sku"]]["shelf"]
            (snapshots / "two.json").write_text(json.dumps(later), encoding="utf-8")
            out = root / "out"
            self.assertEqual(main(["simulate", "--snapshots-dir", str(snapshots),
                                   "--model-dir", str(model_dir), "--output-dir", str(out),
                                   "--max-moved", "2"]), 0)
            timeline = json.loads((out / "summary.json").read_text())
            self.assertEqual(timeline["steps"], 2)
            self.assertEqual(timeline["decisions"][1]["baseline_proxy_errors"], 0)


if __name__ == "__main__":
    unittest.main()
